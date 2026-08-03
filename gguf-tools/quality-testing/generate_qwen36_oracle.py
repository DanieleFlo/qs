#!/usr/bin/env python3
"""Generate unreviewed Qwen3.6 oracle data into a marked staging directory.

This command deliberately has no promotion or acceptance operation.  It uses
the pinned upstream tokenizer/chat template to render messages for every
engine, then records each engine's own tokenization separately.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
from pathlib import Path

from qwen36_fixtures import (
    FixtureError,
    ensure_staging,
    find_case,
    inventory_files,
    materialize_case,
    platform_provenance,
    validate_manifest,
    write_json,
)


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
        }


def load_renderer(manifest: dict):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise FixtureError("transformers is required for pinned Qwen chat rendering") from exc
    return AutoTokenizer.from_pretrained(
        manifest["model"]["source"],
        revision=manifest["model"]["source_revision"],
        trust_remote_code=False,
    )


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


def target_artifact(manifest: dict) -> dict:
    return next(item for item in manifest["artifacts"] if item["role"] == "target")


def generate_llama(
    manifest: dict,
    case: dict,
    rendered: str,
    upstream_ids: list[int],
    logits_path: Path,
    steps: int,
    top_k: int,
) -> dict:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise FixtureError("llama-cpp-python is required for the llama.cpp oracle") from exc
    artifact = target_artifact(manifest)
    model_path = artifact["local_path"]
    if not model_path:
        raise FixtureError("llama.cpp oracle requires artifacts[target].local_path")
    model_path = Path(model_path).expanduser().resolve()
    context = min(manifest["model"]["context_length"], max(8192, len(upstream_ids) + 2 * steps + 16))
    llm = Llama(model_path=str(model_path), n_ctx=context, logits_all=True, verbose=False)
    prompt_ids = [int(value) for value in llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)]
    if not prompt_ids:
        raise FixtureError(f"{case['id']}: llama.cpp produced an empty prompt")
    vocab_size = int(llm.n_vocab())
    logits_file = BinaryLogits(logits_path, vocab_size)
    llm.reset()
    llm.eval(prompt_ids)
    greedy: list[int] = []
    top_rows: list[list[dict[str, float | int]]] = []
    for _ in range(steps):
        values = llm._scores[llm.n_tokens - 1]
        logits_file.append(values)
        row = top_k_python(values, top_k)
        token = int(row[0]["token_id"])
        greedy.append(token)
        top_rows.append(row)
        llm.eval([token])
    logits_info = logits_file.close()

    # A separate pass proves that the saved continuation can be teacher-forced.
    llm.reset()
    llm.eval(prompt_ids)
    teacher_rows = []
    for token in greedy:
        values = llm._scores[llm.n_tokens - 1]
        row = top_k_python(values, top_k)
        selected = next((item for item in row if item["token_id"] == token), None)
        if selected is None:
            maximum = max(float(value) for value in values)
            logsumexp = maximum + math.log(sum(math.exp(float(value) - maximum) for value in values))
            selected = {"token_id": token, "logit": float(values[token]), "logprob": float(values[token]) - logsumexp}
        teacher_rows.append(selected)
        llm.eval([token])
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
        "full_logits": logits_info,
    }


def generate_transformers(
    manifest: dict,
    tokenizer,
    case: dict,
    rendered: str,
    prompt_ids: list[int],
    logits_path: Path,
    steps: int,
    top_k: int,
    dtype: str,
) -> dict:
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
        manifest["model"]["source"],
        revision=manifest["model"]["source_revision"],
        torch_dtype=dtype_map[dtype],
        device_map="auto",
        trust_remote_code=False,
    )
    device = model.get_input_embeddings().weight.device
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    text_config = getattr(model.config, "text_config", model.config)
    vocab_size = int(text_config.vocab_size)
    logits_file = BinaryLogits(logits_path, vocab_size)
    greedy: list[int] = []
    top_rows = []
    with torch.inference_mode():
        outputs = model(input_ids=tokens, use_cache=True)
        for step in range(steps):
            values = outputs.logits[0, -1].float().cpu().numpy()
            logits_file.append(values)
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
    logits_info = logits_file.close()

    teacher_rows = []
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=tokens, use_cache=True)
        for step, token in enumerate(greedy):
            values = outputs.logits[0, -1].float().cpu().tolist()
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
    return {
        "engine": "transformers",
        "engine_version": package_version("transformers"),
        "torch_version": package_version("torch"),
        "prompt_token_ids": prompt_ids,
        "greedy_token_ids": greedy,
        "greedy_text": tokenizer.decode(greedy, skip_special_tokens=False),
        "teacher_forced_source": "same-run greedy continuation",
        "teacher_forced": teacher_rows,
        "top_k": top_rows,
        "full_logits": logits_info,
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
        "engine_version": package_version("vllm"),
        "prompt_token_ids": prompt_ids,
        "greedy_token_ids": greedy,
        "greedy_text": tokenizer.decode(greedy, skip_special_tokens=False),
        "teacher_forced_source": "same-run greedy continuation",
        "teacher_forced_top_k": teacher_rows,
        "top_k": top_rows,
        "full_logits": None,
        "full_logits_unavailable_reason": "vLLM public generation API exposes logprob slices, not full-vocabulary logits",
    }


def write_case_inputs(run_dir: Path, case: dict, rendered: str, upstream_ids: list[int]) -> None:
    prompts = run_dir / "prompts"
    write_json(prompts / f"{case['id']}.case.json", materialize_case(case))
    (prompts / f"{case['id']}.txt").write_text(rendered, encoding="utf-8")
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
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
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
        manifest, corpus = validate_manifest(args.manifest, require_artifacts=args.oracle == "llama.cpp")
        tokenizer = load_renderer(manifest)
        staging = ensure_staging(args.staging_dir)
        run_dir = staging / args.run_id
        if run_dir.exists():
            raise FixtureError(f"run already exists: {run_dir}")
        run_dir.mkdir()
        for name in ("prompts", "continuations", "responses", "logits"):
            (run_dir / name).mkdir()
        selected = [find_case(corpus, value) for value in args.cases] if args.cases else corpus["cases"]
        cases_index = []
        manifest_rows = ["# id\tprompt_file\tcontinuation_file\tresponse_file"]
        for case in selected:
            rendered, upstream_ids = render_case(tokenizer, case)
            write_case_inputs(run_dir, case, rendered, upstream_ids)
            logits_path = run_dir / "logits" / f"{case['id']}.f32"
            if args.oracle == "llama.cpp":
                result = generate_llama(manifest, case, rendered, upstream_ids, logits_path, args.steps, args.top_k)
            elif args.oracle == "transformers":
                result = generate_transformers(manifest, tokenizer, case, rendered, upstream_ids, logits_path, args.steps, args.top_k, args.dtype)
            else:
                result = generate_vllm(manifest, tokenizer, case, rendered, upstream_ids, run_dir, args.steps, args.top_k, args.dtype)
            continuation_path = run_dir / "continuations" / f"{case['id']}.txt"
            response_path = run_dir / "responses" / f"{case['id']}.json"
            continuation_path.write_text(result["greedy_text"], encoding="utf-8")
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
            "parameters": {"temperature": 0, "steps": args.steps, "top_k": args.top_k},
            "host": platform_provenance(),
            "review_status": "generated_unreviewed",
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
