#!/usr/bin/env python3
"""Package a token-manifest scorer run in the audited task-2 layout."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from qwen36_fixtures import (
    FixtureError, ensure_staging, inventory_files, load_json, platform_provenance,
    sha256_file, validate_manifest, write_json,
)
from verify_qwen36_run import verify_run

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required to package full-vocabulary logits") from exc


def artifact_target(manifest: dict) -> dict:
    return next(item for item in manifest["artifacts"] if item["role"] == "target")


def logits_info(path: Path, rows: int, vocabulary: int, run_dir: Path) -> dict:
    expected = rows * vocabulary * 4
    if not path.is_file() or path.stat().st_size != expected:
        raise FixtureError(f"logits size mismatch: {path}")
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "dtype": "float32-le",
        "shape": [rows, vocabulary],
        "row_stride_bytes": vocabulary * 4,
        "size_bytes": expected,
        "sha256": sha256_file(path),
    }


def top_rows(path: Path, rows: int, vocabulary: int, top_k: int) -> list[list[dict]]:
    logits = np.memmap(path, dtype="<f4", mode="r", shape=(rows, vocabulary))
    result = []
    for raw in logits:
        values = raw.astype(np.float64)
        maximum = float(values.max())
        logsum = maximum + math.log(float(np.exp(values - maximum).sum()))
        count = min(top_k, vocabulary)
        ids = np.argpartition(values, vocabulary - count)[-count:]
        ids = ids[np.argsort(-values[ids], kind="stable")]
        result.append([
            {"token_id": int(token), "logit": float(values[token]), "logprob": float(values[token] - logsum)}
            for token in ids
        ])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--engine", required=True, choices=("ds4", "llama.cpp"))
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--build-flags", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--scorer-arg", action="append", default=[])
    args = parser.parse_args()
    try:
        if args.context < 1 or args.top_k < 1:
            raise FixtureError("context and top-k must be positive")
        args.source_run = args.source_run.resolve()
        args.scorer = args.scorer.resolve()
        args.model = args.model.resolve()
        args.staging_dir = args.staging_dir.resolve()
        if not args.scorer.is_file():
            raise FixtureError(f"scorer does not exist: {args.scorer}")
        manifest, _ = validate_manifest(args.manifest)
        source = verify_run(args.source_run, args.manifest, allow_partial=False)
        target = artifact_target(manifest)
        if not args.model.is_file():
            raise FixtureError(f"model does not exist: {args.model}")
        if args.model.stat().st_size != target["size_bytes"] or sha256_file(args.model) != target["sha256"]:
            raise FixtureError("model size or SHA-256 does not match the target artifact")
        staging = ensure_staging(args.staging_dir)
        run_dir = staging / args.run_id
        if run_dir.exists():
            raise FixtureError(f"run already exists: {run_dir}")
        for name in ("prompts", "continuations", "responses", "logits"):
            (run_dir / name).mkdir(parents=True, exist_ok=True)

        cases = []
        driver_rows = ["# id\trendered\tprompt_tokens\ttarget_tokens\tgreedy_logits\tteacher_logits\tresponse"]
        for entry in source["index"]["cases"]:
            case_id = entry["id"]
            source_response = source["responses"][case_id]
            source_prompt = args.source_run / "prompts" / case_id
            for suffix in (".txt", ".bytes", ".case.json"):
                shutil.copyfile(source_prompt.with_suffix(suffix), run_dir / "prompts" / f"{case_id}{suffix}")
            canonical_prompt = source_response.get("canonical_prompt_token_ids", source_response["prompt_token_ids"])
            teacher_tokens = [int(row["token_id"]) for row in source_response["teacher_forced"]]
            canonical_path = run_dir / "prompts" / f"{case_id}.canonical.tokens.json"
            target_path = run_dir / "continuations" / f"{case_id}.tokens.json"
            write_json(canonical_path, canonical_prompt)
            write_json(target_path, teacher_tokens)
            source_continuation = args.source_run / "continuations" / f"{case_id}.txt"
            if source_continuation.is_file():
                shutil.copyfile(source_continuation, run_dir / "continuations" / f"{case_id}.txt")
            greedy_path = run_dir / "logits" / f"{case_id}.greedy.f32"
            teacher_path = run_dir / "logits" / f"{case_id}.teacher.f32"
            response_path = run_dir / "responses" / f"{case_id}.json"
            driver_rows.append("\t".join(map(str, (
                case_id, run_dir / "prompts" / f"{case_id}.txt", canonical_path,
                target_path, greedy_path, teacher_path, response_path,
            ))))
            cases.append({
                "id": case_id, "category": entry.get("category"),
                "prompt_file": f"prompts/{case_id}.txt",
                "continuation_file": f"continuations/{case_id}.txt",
                "response_file": f"responses/{case_id}.json",
            })

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".tsv", dir=staging, delete=False) as driver:
            driver.write("\n".join(driver_rows) + "\n")
            driver_path = Path(driver.name)
        try:
            command = [str(args.scorer), "--token-manifest", str(args.model), str(driver_path), str(args.context), *args.scorer_arg]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise FixtureError(f"scorer exited with status {completed.returncode}")
        finally:
            driver_path.unlink(missing_ok=True)

        vocabulary = int(target["metadata"]["expected_model"]["vocab_size"])
        for entry in cases:
            case_id = entry["id"]
            response_path = run_dir / entry["response_file"]
            response = load_json(response_path)
            native_tokens = response["native_prompt_token_ids"]
            write_json(run_dir / "prompts" / f"{case_id}.tokens.json", native_tokens)
            rows = len(response["teacher_forced"])
            greedy_path = run_dir / "logits" / f"{case_id}.greedy.f32"
            teacher_path = run_dir / "logits" / f"{case_id}.teacher.f32"
            response["top_k"] = top_rows(greedy_path, rows, vocabulary, args.top_k)
            response["full_logits"] = {
                "greedy": logits_info(greedy_path, rows, vocabulary, run_dir),
                "teacher_forced": logits_info(teacher_path, rows, vocabulary, run_dir),
            }
            write_json(response_path, response)

        index = {
            "format": "ds4-qwen36-oracle-v1",
            "manifest": str(args.manifest.resolve()),
            "manifest_model": manifest["model"]["id"],
            "run_id": args.run_id,
            "environment": {
                "oracle": args.engine,
                "kind": "numeric_same_gguf",
                "engine_commit": args.engine_commit,
                "build_flags": args.build_flags,
                "backend": args.backend,
                "hardware": args.hardware,
                "dtype": args.dtype,
                "parameters": {
                    "temperature": 0, "steps": len(next(iter(source["responses"].values()))["teacher_forced"]),
                    "top_k": args.top_k, "context": args.context,
                    "prefill_chunk": args.prefill_chunk,
                },
                "artifacts": {"target": {"filename": args.model.name, "size_bytes": args.model.stat().st_size, "sha256": target["sha256"]}},
                "host": platform_provenance(),
                "review_status": "generated_unreviewed",
                "canonical_source_run": source["index"].get("run_id"),
            },
            "cases": cases,
            "files": inventory_files(run_dir),
            "promotion": "manual review and copy only; this tool has no acceptance command",
        }
        write_json(run_dir / "index.json", index)
        print(run_dir)
        return 0
    except (FixtureError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
