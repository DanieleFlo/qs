#!/usr/bin/env python3
"""Validate Qwen3.6 manifests and their corpus without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qwen36_fixtures import FixtureError, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    results = []
    failed = False
    for path in args.manifest:
        try:
            manifest, corpus = validate_manifest(path, require_artifacts=args.require_artifacts)
            result = {
                "manifest": str(path),
                "status": "valid",
                "model": manifest["model"]["id"],
                "cases": len(corpus["cases"]),
                "verification": manifest["verification"]["status"],
            }
        except FixtureError as exc:
            failed = True
            result = {"manifest": str(path), "status": "invalid", "error": str(exc)}
        results.append(result)

    if args.json_output:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            if result["status"] == "valid":
                print(f"valid: {result['manifest']} ({result['cases']} cases, {result['verification']})")
            else:
                print(f"invalid: {result['manifest']}: {result['error']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
