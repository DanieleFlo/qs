#!/usr/bin/env python3
"""Validate the pinned Qwen3.8 target/MTP manifest and optional local GGUFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen38_fixtures import FixtureError, validate_manifest


def parse_artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ROLE=PATH")
    role, raw_path = value.split("=", 1)
    if role not in {"target", "mtp"} or not raw_path:
        raise argparse.ArgumentTypeError("artifact role must be target or mtp")
    return role, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--artifact", action="append", default=[], type=parse_artifact)
    args = parser.parse_args()
    try:
        manifest, snapshots = validate_manifest(
            args.manifest, dict(args.artifact) if args.artifact else None)
    except (FixtureError, OSError) as exc:
        parser.error(str(exc))
    print(f"validated {manifest['model']['id']}: {', '.join(sorted(snapshots))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
