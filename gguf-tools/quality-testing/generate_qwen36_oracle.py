#!/usr/bin/env python3
"""Generate unreviewed Qwen3.6 oracle data into a marked staging directory.

This command deliberately has no promotion or acceptance operation.  It uses
the pinned upstream tokenizer/chat template to render messages for every
engine, then records each engine's own tokenization separately.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import math
import os
import sys
import sysconfig
from pathlib import Path
from urllib.parse import urlparse

from qwen36_fixtures import (
    FixtureError,
    ensure_staging,
    find_case,
    inventory_files,
    materialize_case,
    platform_provenance,
    sha256_file,
    validate_manifest,
    write_json,
)


_DLL_HANDLES = []


def activate_nvidia_dll_dirs() -> None:
    """Expose CUDA DLLs installed by NVIDIA pip wheels on Windows."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    site_packages = Path(sysconfig.get_paths()["purelib"])
    candidates = [site_packages / "llama_cpp" / "lib"]
    candidates.extend(sorted((site_packages / "nvidia").glob("*/bin")))
    for path in candidates:
        if path.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(path)))
    llama_lib = site_packages / "llama_cpp" / "lib"
    for name in ("ggml-base.dll", "ggml.dll", "ggml-cpu.dll", "ggml-cuda.dll"):
        path = llama_lib / name
        if path.is_file():
            _DLL_HANDLES.append(ctypes.CDLL(str(path)))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def top_k_python(values, k: int) -> list[dict[str, float | int]]:
    try:
        import numpy as np

        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size < k:
            raise FixtureError("invalid logit vector for top-k")
        indices = np.argpartition(array, -k)[-k:]
        indices = indices[np.argsort(array[indices])[::-1]]
        maximum = float(np.max(array))
        logsumexp = maximum + math.log(float(np.exp(array - maximum).sum()))
        return [
            {"token_id": int(token_id), "logit": float(array[token_id]), "logprob": float(array[token_id]) - logsumexp}
            for token_id in indices
        ]
    except ImportError:
        pass
    pairs = sorted(enumerate(values), key=lambda item: float(item[1]), reverse=True)[:k]
    maximum = float(pairs[0][1])
    total = sum(math.exp(float(value) - maximum) for value in values)
    logsumexp = maximum + math.log(total)
    return [
        {"token_id": int(token_id), "logit": float(value), "logprob": float(value) - logsumexp}
        for token_id, value in pairs
    ]


class BinaryLogits:
    def __init__(self, path: Path, vocab_size: int):
        self.path = path
        self.vocab_size = vocab_size
        self.rows = 0
        self.fp = path.open("wb")

    def append(self, values) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise FixtureError("numpy is required to write full logits") from exc
        array = np.asarray(values, dtype="<f4")
        if array.ndim != 1 or array.shape[0] != self.vocab_size:
            raise FixtureError("unexpected full-logit vector shape")
        self.fp.write(array.tobytes(order="C"))
        self.rows += 1

    def close(self) -> dict[str, object]:
        self.fp.close()
        try:
            output_path = self.path.relative_to(self.path.parent.parent).as_posix()
        except ValueError:
            output_path = self.path.name
        return {
            "path": output_path,
            "dtype": "float32-le",
            "shape": [self.rows, self.vocab_size],
            "row_stride_bytes": self.vocab_size * 4,
            "size_bytes": self.path.stat().st_size,
            "sha256": sha256_file(self.path),
        }


def load_renderer(manifest: dict):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise FixtureError("transformers is required for pinned Qwen chat rendering") from exc
    source = upstream_model_id(manifest)
    return AutoTokenizer.from_pretrained(
        source,
        revision=manifest["model"]["source_revision"],
        trust_remote_code=False,
    )


def upstream_model_id(manifest: dict) -> str:
    source = manifest["model"]["source"]
    parsed = urlparse(source)
    if parsed.netloc == "huggingface.co":
        source = parsed.path.strip("/")
    return source


def render_case(tokenizer, case: dict) -> tuple[str, list[int]]:
    case = materialize_case(case)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": case["thinking"],
        "preserve_thinking": bool(case.get("preserve_thinking", False)),
    }
    if case["tools"]:
        kwargs["tools"] = case["tools"]
    rendered = tokenizer.apply_chat_template(case["messages"], **kwargs)
    upstream_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if len(upstream_ids) < case["expected_min_prompt_tokens"]:
        raise FixtureError(
            f"{case['id']}: rendered prompt has {len(upstream_ids)} tokens, "
            f"expected at least {case['expected_min_prompt_tokens']}"
        )
    return rendered, [int(value) for value in upstream_ids]


def parse_artifact_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ROLE=PATH")
    role, raw_path = value.split("=", 1)
    if role not in {"target", "mtp"} or not raw_path:
        raise argparse.ArgumentTypeError("artifact role must be target or mtp")
    return role, Path(raw_path)


def resolve_artifact_paths(manifest: dict, overrides: list[tuple[str, Path]]) -> dict[str, Path]:
    supplied = dict(overrides)
    declared = {artifact["role"]: artifact for artifact in manifest["artifacts"]}
    if not set(supplied).issubset(declared):
        raise FixtureError("artifact override has a role not declared by the manifest")
    result: dict[str, Path] = {}
    for role, artifact in declared.items():
        raw_path = supplied.get(role)
        if raw_path is None and artifact["local_path"] is not None:
            raw_path = Path(artifact["local_path"])
        if raw_path is None:
            continue
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FixtureError(f"artifact file does not exist: {path}")
        if path.stat().st_size != artifact["size_bytes"]:
            raise FixtureError(f"artifact size mismatch: {path}")
        if sha256_file(path) != artifact["sha256"]:
            raise FixtureError(f"artifact SHA-256 mismatch: {path}")
        result[role] = path
    return result


def target_artifact_summary(manifest: dict) -> dict[str, object]:
    target = next(artifact for artifact in manifest["artifacts"] if artifact["role"] == "target")
    return {
        "filename": target["filename"],
        "sha256": target["sha256"],
        "quantization": target["quantization"],
    }


def create_llama(model_path: Path, context: int, n_gpu_layers: int):
    activate_nvidia_dll_dirs()
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise FixtureError("llama-cpp-python is required for the llama.cpp oracle") from exc
    return Llama(
        model_path=str(model_path), n_ctx=context, n_gpu_layers=n_gpu_layers,
        logits_all=True, verbose=False,
    )


def generate_llama(
    llm,
    manifest: dict,
    case: dict,
    rendered: str,
    upstream_ids: list[int],
    logits_dir: Path,
    steps: int,
    top_k: int,
    context: int,
    n_gpu_layers: int,
) -> dict:
    required_context = len(upstream_ids) + 2 * steps + 16
    if context < required_context:
        raise FixtureError(f"{case['id']}: --context {context} is below required {required_context}")
    context = min(manifest["model"]["context_length"], context)
    prompt_ids = [int(value) for value in llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)]
    if not prompt_ids:
        raise FixtureError(f"{case['id']}: llama.cpp produced an empty prompt")
    vocab_size = int(llm.n_vocab())
    greedy_logits = BinaryLogits(logits_dir / f"{case['id']}.greedy.f32", vocab_size)
    llm.reset()
    llm.eval(prompt_ids)
    greedy: list[int] = []
    top_rows: list[list[dict[str, float | int]]] = []
    for _ in range(steps):
        values = llm._scores[llm.n_tokens - 1]
        greedy_logits.append(values)
        row = top_k_python(values, top_k)
        token = int(row[0]["token_id"])
        greedy.append(token)
        top_rows.append(row)
        llm.eval([token])
    greedy_logits_info = greedy_logits.close()

    # A separate pass proves that the saved continuation can be teacher-forced.
    llm.reset()
    llm.eval(prompt_ids)
    teacher_rows = []
    teacher_logits = BinaryLogits(logits_dir / f"{case['id']}.teacher.f32", vocab_size)
    for token in greedy:
        values = llm._scores[llm.n_tokens - 1]
        teacher_logits.append(values)
        row = top_k_python(values, top_k)
        selected = next((item for item in row if item["token_id"] == token), None)
        if selected is None:
            maximum = max(float(value) for value in values)
            logsumexp = maximum + math.log(sum(math.exp(float(value) - maximum) for value in values))
            selected = {"token_id": token, "logit": float(values[token]), "logprob": float(values[token]) - logsumexp}
        teacher_rows.append(selected)
        llm.eval([token])
    teacher_logits_info = teacher_logits.close()
    return {
        "engine": "llama.cpp",
        "engine_version": package_version("llama-cpp-python"),
        "prompt_token_ids": prompt_ids,
        "upstream_render_token_ids": upstream_ids,
        "greedy_token_ids": greedy,
        "greedy_text": llm.detokenize(greedy).decode("utf-8", "replace"),
        "greedy_bytes_hex": llm.detokenize(greedy).hex(),
        "teacher_forced_source": "same-run greedy continuation",
        "teacher_forced": teacher_rows,
        "top_k": top_rows,
        "full_logits": {"greedy": greedy_logits_info, "teacher_forced": teacher_logits_info},
        "runtime": {"context": context, "n_gpu_layers": n_gpu_layers},
    }


def create_transformers(manifest: dict, dtype: str):
    """Load the large upstream model once and reuse it for the whole run."""
    try:
        import torch
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:
        raise FixtureError(
            "torch and a Transformers release with AutoModelForMultimodalLM are required "
            "for the Transformers oracle"
        ) from exc
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype not in dtype_map:
        raise FixtureError(f"unsupported Transformers dtype: {dtype}")
    model = AutoModelForMultimodalLM.from_pretrained(
        upstream_model_id(manifest),
        revision=manifest["model"]["source_revision"],
        torch_dtype=dtype_map[dtype],
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    text_config = getattr(model.config, "text_config", model.config)
    return model, device, int(text_config.vocab_size)


def generate_transformers(
    runtime,
    tokenizer,
    case: dict,
    rendered: str,
    prompt_ids: list[int],
    logits_dir: Path,
    steps: int,
    top_k: int,
    dtype: str,
) -> dict:
    import torch

    model, device, vocab_size = runtime
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    greedy_logits = BinaryLogits(logits_dir / f"{case['id']}.greedy.f32", vocab_size)
    greedy: list[int] = []
    top_rows = []
    with torch.inference_mode():
        outputs = model(input_ids=tokens, use_cache=True)
        for step in range(steps):
            values = outputs.logits[0, -1].float().cpu().numpy()
            greedy_logits.append(values)
            row = top_k_python(values, top_k)
            token = int(row[0]["token_id"])
            greedy.append(token)
            top_rows.append(row)
            if step + 1 < steps:
                outputs = model(
                    input_ids=torch.tensor([[token]], device=device),
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                )
    greedy_logits_info = greedy_logits.close()

    teacher_rows = []
    teacher_logits = BinaryLogits(logits_dir / f"{case['id']}.teacher.f32", vocab_size)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=tokens, use_cache=True)
        for step, token in enumerate(greedy):
            values_array = outputs.logits[0, -1].float().cpu().numpy()
            teacher_logits.append(values_array)
            values = values_array.tolist()
            row = top_k_python(values, top_k)
            selected = next((item for item in row if item["token_id"] == token), None)
            if selected is None:
                maximum = max(values)
                logsumexp = maximum + math.log(sum(math.exp(value - maximum) for value in values))
                selected = {"token_id": token, "logit": values[token], "logprob": values[token] - logsumexp}
            teacher_rows.append(selected)
            if step + 1 < len(greedy):
                outputs = model(
                    input_ids=torch.tensor([[token]], device=device),
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                )
    teacher_logits_info = teacher_logits.close()
    return {
        "engine": "transformers",
        "oracle_kind": "semantic_upstream",
        "engine_version": package_version("transformers"),
        "torch_version": package_version("torch"),
        "prompt_token_ids": prompt_ids,
        "greedy_token_ids": greedy,
        "greedy_text": tokenizer.decode(greedy, skip_special_tokens=False),
        "teacher_forced_source": "same-run greedy continuation",
        "teacher_forced": teacher_rows,
        "top_k": top_rows,
        "full_logits": {"greedy": greedy_logits_info, "teacher_forced": teacher_logits_info},
        "target_difference": {
            "model_identity": "same pinned upstream model family and revision",
            "weights": "upstream Transformers weights, not the target GGUF payload",
            "precision": f"{dtype}, not GGUF Q4_K_M",
            "numeric_equivalence_expected": False,
        },
    }


def _vllm_logprobs(raw) -> list[dict[str, float | int]]:
    if not raw:
        return []
    result = []
    for token_id, value in raw.items():
        result.append({
            "token_id": int(token_id),
            "logprob": float(value.logprob),
            "rank": int(value.rank) if value.rank is not None else None,
        })
    return sorted(result, key=lambda item: item["logprob"], reverse=True)


def generate_vllm(
    manifest: dict,
    tokenizer,
    case: dict,
    rendered: str,
    prompt_ids: list[int],
    case_dir: Path,
    steps: int,
    top_k: int,
    dtype: str,
) -> dict:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise FixtureError("vllm is required for the vLLM oracle") from exc
    llm = LLM(
        model=manifest["model"]["source"],
        revision=manifest["model"]["source_revision"],
        dtype=dtype,
        enforce_eager=True,
    )
    greedy_params = SamplingParams(temperature=0, max_tokens=steps, min_tokens=steps, logprobs=top_k, detokenize=False)
    greedy_output = llm.generate({"prompt_token_ids": prompt_ids}, greedy_params, use_tqdm=False)[0].outputs[0]
    greedy = [int(value) for value in greedy_output.token_ids]
    if len(greedy) != steps:
        raise FixtureError(f"{case['id']}: vLLM returned {len(greedy)} tokens, expected {steps}")
    top_rows = [_vllm_logprobs(value) for value in greedy_output.logprobs]

    combined = prompt_ids + greedy
    teacher_params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=top_k, logprobs=top_k)
    teacher_output = llm.generate({"prompt_token_ids": combined}, teacher_params, use_tqdm=False)[0]
    raw_prompt = teacher_output.prompt_logprobs or []
    teacher_rows = [_vllm_logprobs(value) for value in raw_prompt[-steps:]]
    return {
        "engine": "vllm",
        "oracle_kind": "semantic_upstream",
        "engine_version": package_version("vllm"),
        "prompt_token_ids": prompt_ids,
        "greedy_token_ids": greedy,
        "greedy_text": tokenizer.decode(greedy, skip_special_tokens=False),
        "teacher_forced_source": "same-run greedy continuation",
        "teacher_forced_top_k": teacher_rows,
        "top_k": top_rows,
        "full_logits": None,
        "full_logits_unavailable_reason": "vLLM public generation API exposes logprob slices, not full-vocabulary logits",
        "target_difference": {
            "model_identity": "same pinned upstream model family and revision",
            "weights": "upstream vLLM weights, not the target GGUF payload",
            "precision": f"{dtype}, not GGUF Q4_K_M",
            "numeric_equivalence_expected": False,
        },
    }


def write_case_inputs(run_dir: Path, case: dict, rendered: str, upstream_ids: list[int]) -> None:
    prompts = run_dir / "prompts"
    write_json(prompts / f"{case['id']}.case.json", materialize_case(case))
    (prompts / f"{case['id']}.txt").write_text(rendered, encoding="utf-8", newline="")
    (prompts / f"{case['id']}.bytes").write_bytes(rendered.encode("utf-8"))
    write_json(prompts / f"{case['id']}.tokens.json", upstream_ids)


def manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--oracle", required=True, choices=("llama.cpp", "transformers", "vllm"))
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True, help="stable, human-selected audit identifier")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--artifact-path", action="append", default=[], type=parse_artifact_path,
        metavar="ROLE=PATH", help="local artifact override; paths are never written to the tracked manifest",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--build-flags", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--dtype", required=True)
    args = parser.parse_args()
    try:
        if args.steps < 32:
            raise FixtureError("--steps must be at least 32")
        if args.top_k < 1:
            raise FixtureError("--top-k must be positive")
        manifest, corpus = validate_manifest(args.manifest)
        artifact_paths = resolve_artifact_paths(manifest, args.artifact_path)
        if args.oracle == "llama.cpp" and "target" not in artifact_paths:
            raise FixtureError("llama.cpp oracle requires --artifact-path target=PATH or artifacts[target].local_path")
        tokenizer = load_renderer(manifest)
        staging = ensure_staging(args.staging_dir)
        run_dir = staging / args.run_id
        if run_dir.exists():
            raise FixtureError(f"run already exists: {run_dir}")
        run_dir.mkdir()
        for name in ("prompts", "continuations", "responses", "logits"):
            (run_dir / name).mkdir()
        selected = [find_case(corpus, value) for value in args.cases] if args.cases else corpus["cases"]
        rendered_cases = [(case, *render_case(tokenizer, case)) for case in selected]
        llama_runtime = None
        transformers_runtime = None
        if args.oracle == "llama.cpp":
            required_context = max(len(ids) + 2 * args.steps + 16 for _, _, ids in rendered_cases)
            if args.context < required_context:
                raise FixtureError(f"--context {args.context} is below corpus requirement {required_context}")
            llama_runtime = create_llama(artifact_paths["target"], args.context, args.n_gpu_layers)
        elif args.oracle == "transformers":
            transformers_runtime = create_transformers(manifest, args.dtype)
        cases_index = []
        manifest_rows = ["# id\tprompt_file\tcontinuation_file\tresponse_file"]
        for case, rendered, upstream_ids in rendered_cases:
            write_case_inputs(run_dir, case, rendered, upstream_ids)
            if args.oracle == "llama.cpp":
                result = generate_llama(
                    llama_runtime, manifest, case, rendered, upstream_ids, run_dir / "logits",
                    args.steps, args.top_k, args.context,
                    args.n_gpu_layers,
                )
            elif args.oracle == "transformers":
                result = generate_transformers(
                    transformers_runtime, tokenizer, case, rendered, upstream_ids,
                    run_dir / "logits", args.steps, args.top_k, args.dtype,
                )
            else:
                result = generate_vllm(manifest, tokenizer, case, rendered, upstream_ids, run_dir, args.steps, args.top_k, args.dtype)
            continuation_path = run_dir / "continuations" / f"{case['id']}.txt"
            response_path = run_dir / "responses" / f"{case['id']}.json"
            continuation_path.write_text(result["greedy_text"], encoding="utf-8", newline="")
            write_json(response_path, result)
            prompt_path = run_dir / "prompts" / f"{case['id']}.txt"
            manifest_rows.append(
                "\t".join((case["id"], manifest_path(prompt_path), manifest_path(continuation_path), manifest_path(response_path)))
            )
            cases_index.append({
                "id": case["id"],
                "category": case["category"],
                "prompt_file": f"prompts/{case['id']}.txt",
                "continuation_file": f"continuations/{case['id']}.txt",
                "response_file": f"responses/{case['id']}.json",
            })

        (run_dir / "manifest.tsv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

        environment = {
            "oracle": args.oracle,
            "kind": manifest["oracle_environments"][args.oracle]["kind"],
            "engine_commit": args.engine_commit,
            "build_flags": args.build_flags,
            "backend": args.backend,
            "hardware": args.hardware,
            "dtype": args.dtype,
            "parameters": {
                "temperature": 0, "steps": args.steps, "top_k": args.top_k,
                "context": args.context, "n_gpu_layers": args.n_gpu_layers,
            },
            "artifacts": {
                role: {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for role, path in sorted(artifact_paths.items())
            },
            "host": platform_provenance(),
            "review_status": "generated_unreviewed",
        }
        if args.oracle in {"transformers", "vllm"}:
            environment["target_difference"] = {
                "target": target_artifact_summary(manifest),
                "upstream_model": upstream_model_id(manifest),
                "upstream_revision": manifest["model"]["source_revision"],
                "weights_differ": True,
                "precision_differ": args.dtype != "GGUF Q4_K_M",
                "numeric_equivalence_expected": False,
            }
        index = {
            "format": "ds4-qwen36-oracle-v1",
            "manifest": str(args.manifest.resolve()),
            "manifest_model": manifest["model"]["id"],
            "run_id": args.run_id,
            "environment": environment,
            "cases": cases_index,
            "files": inventory_files(run_dir),
            "promotion": "manual review and copy only; this tool has no acceptance command",
        }
        write_json(run_dir / "index.json", index)
        print(run_dir)
        return 0
    except FixtureError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
