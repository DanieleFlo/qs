#!/usr/bin/env python3
"""Materialize one deterministic long Qwen3.6 corpus case into staging."""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen36_fixtures import FixtureError, ensure_staging, find_case, materialize_case, validate_manifest, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--staging-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        _, corpus = validate_manifest(args.manifest)
        case = find_case(corpus, args.case)
        if case["category"] != "long-canary":
            raise FixtureError(f"{args.case} is not a long-canary case")
        staging = ensure_staging(args.staging_dir)
        materialized = materialize_case(case)
        case_dir = staging / case["id"]
        case_dir.mkdir(exist_ok=True)
        write_json(case_dir / "case.json", materialized)
        prompt = materialized["messages"][0]["content"]
        (case_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        print(case_dir / "prompt.txt")
        return 0
    except FixtureError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
