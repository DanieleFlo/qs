#!/usr/bin/env python3
"""Run the audited Qwen scorer sequentially across the 4K/16K profile."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from qwen36_fixtures import FixtureError, validate_context_profile, write_json
from qwen36_speed_gate import validate_speed_report


def source_run_arg(value: str) -> tuple[int, Path]:
    try:
        raw_context, raw_path = value.split("=", 1)
        context = int(raw_context)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected CONTEXT=PATH") from exc
    if context < 1 or not raw_path:
        raise argparse.ArgumentTypeError("expected positive CONTEXT=PATH")
    return context, Path(raw_path)


def free_vram_mib() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise FixtureError("nvidia-smi failed before a context-matrix run")
    values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise FixtureError("the 16K safety profile requires exactly one visible GPU")
    return values[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-run", action="append", required=True, type=source_run_arg)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--staging-dir", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--build-flags", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--performance-report", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        profile = validate_context_profile(args.profile)
        speed_report = None
        if args.execute:
            if args.performance_report is None:
                raise FixtureError(
                    "--performance-report is required with --execute; "
                    "a missing speed baseline is NOT VERIFIED"
                )
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            target = next(
                (item for item in manifest.get("artifacts", []) if item.get("role") == "target"),
                None,
            )
            if target is None:
                raise FixtureError("manifest has no target artifact")
            speed_report = validate_speed_report(
                args.performance_report, model_sha256=target["sha256"],
                backend=args.backend, hardware=args.hardware,
            )
        sources = dict(args.source_run)
        if sorted(sources) != profile["frontiers"]:
            raise FixtureError("--source-run must provide exactly the 4096 and 16384 frontiers")
        matrix = []
        chunks: list[int | None] = [None, *profile["prefill_chunks"]]
        for context in profile["frontiers"]:
            for chunk in chunks:
                for repetition in range(1, profile["repetitions"] + 1):
                    chunk_label = "monolithic" if chunk is None else str(chunk)
                    run_id = f"{args.run_prefix}-ctx{context}-chunk{chunk_label}-r{repetition}"
                    command = [
                        sys.executable,
                        str(Path(__file__).with_name("generate_qwen36_score.py")),
                        "--manifest", str(args.manifest),
                        "--source-run", str(sources[context]),
                        "--scorer", str(args.scorer),
                        "--model", str(args.model),
                        "--engine", "ds4",
                        "--staging-dir", str(args.staging_dir),
                        "--run-id", run_id,
                        "--context", str(context),
                        "--context-profile", str(args.profile),
                        "--top-k", "20",
                        "--case", profile["frontier_cases"][str(context)],
                        "--engine-commit", args.engine_commit,
                        "--build-flags", args.build_flags,
                        "--backend", args.backend,
                        "--hardware", args.hardware,
                        "--dtype", args.dtype,
                    ]
                    if chunk is not None:
                        command.extend(("--prefill-chunk", str(chunk)))
                    entry = {
                        "context": context, "prefill_chunk": chunk,
                        "repetition": repetition, "run_id": run_id,
                        "command": command, "status": "planned",
                    }
                    matrix.append(entry)
                    if not args.execute:
                        continue
                    free = free_vram_mib()
                    if free < profile["minimum_free_vram_mib"]:
                        raise FixtureError(
                            f"{run_id}: free VRAM {free} MiB is below "
                            f"required {profile['minimum_free_vram_mib']} MiB"
                        )
                    completed = subprocess.run(command, check=False)
                    entry["free_vram_before_mib"] = free
                    entry["exit_status"] = completed.returncode
                    entry["status"] = "pass" if completed.returncode == 0 else "not_verified"
                    if completed.returncode != 0:
                        raise FixtureError(f"{run_id}: scorer exited with status {completed.returncode}")
        result = {
            "format": "ds4-qwen36-context-matrix-v1",
            "profile": str(args.profile.resolve()),
            "executed": args.execute,
            "performance_gate": speed_report,
            "runs": matrix,
        }
        if args.execute:
            write_json(args.staging_dir / f"{args.run_prefix}-matrix.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (FixtureError, OSError, ValueError) as exc:
        print(json.dumps({"status": "NOT_VERIFIED", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
