#!/usr/bin/env python3
"""Build a small, deterministic JSONSchemaBench coverage subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/guidance-ai/jsonschemabench"
SOURCE_DATASET = "https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench"
SOURCE_COMMIT = "ba103c73756198dd9b149ddc7db7867da7a077f6"
CATEGORIES = (
    "Glaiveai2K", "Github_trivial", "Github_easy", "Github_medium",
    "Github_hard", "Github_ultra", "JsonSchemaStore", "Kubernetes",
    "Snowplow", "WashingtonPost",
)
SEMANTIC_KEYS = {
    "type", "properties", "required", "additionalProperties", "items",
    "allOf", "anyOf", "oneOf", "enum", "const", "minLength", "maxLength",
    "minItems", "maxItems", "minProperties", "maxProperties", "minimum",
    "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "uniqueItems",
}
ANNOTATION_KEYS = {
    "$schema", "$id", "title", "description", "default", "examples",
    "deprecated", "readOnly", "writeOnly", "format",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def schema_features(schema: Any) -> set[str]:
    features: set[str] = set()
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in SEMANTIC_KEYS:
                features.add(key)
            if key == "properties" and isinstance(value, dict):
                for child in value.values():
                    features.update(schema_features(child))
            elif key in ("items", "additionalProperties"):
                features.update(schema_features(value))
            elif key in ("allOf", "anyOf", "oneOf") and isinstance(value, list):
                for child in value:
                    features.update(schema_features(child))
    return features


def json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def unsupported_reasons(schema: Any, path: str = "$") -> list[str]:
    if isinstance(schema, bool):
        return []
    if not isinstance(schema, dict):
        return [f"{path}:schema_is_not_object_or_boolean"]
    reasons: list[str] = []
    for key, value in schema.items():
        if key not in SEMANTIC_KEYS and key not in ANNOTATION_KEYS:
            reasons.append(f"{path}:{key}")
            continue
        if key == "type":
            known_types = {
                "null", "boolean", "number", "integer", "string", "array",
                "object",
            }
            if isinstance(value, str):
                if value not in known_types:
                    reasons.append(f"{path}.type:unknown_type")
            elif isinstance(value, list) and value:
                if any(not isinstance(item, str) or item not in known_types
                       for item in value):
                    reasons.append(f"{path}.type:invalid_type_array")
            else:
                reasons.append(f"{path}.type:not_type_or_nonempty_type_array")
        elif key == "properties":
            if not isinstance(value, dict):
                reasons.append(f"{path}.properties:not_object")
            else:
                for name, child in value.items():
                    reasons.extend(unsupported_reasons(child, f"{path}.{name}"))
        elif key == "required":
            if (not isinstance(value, list) or
                    any(not isinstance(item, str) for item in value)):
                reasons.append(f"{path}.required:not_string_array")
            elif len(value) != len(set(value)):
                reasons.append(f"{path}.required:duplicate")
        elif key in ("items", "additionalProperties"):
            if key == "additionalProperties" and isinstance(value, bool):
                continue
            reasons.extend(unsupported_reasons(value, f"{path}.{key}"))
        elif key in ("allOf", "anyOf", "oneOf"):
            if not isinstance(value, list) or not value:
                reasons.append(f"{path}.{key}:not_nonempty_array")
            else:
                for index, child in enumerate(value):
                    reasons.extend(
                        unsupported_reasons(child, f"{path}.{key}[{index}]")
                    )
        elif key in {
            "minLength", "maxLength", "minItems", "maxItems",
            "minProperties", "maxProperties",
        }:
            if (not json_number(value) or value < 0 or
                    int(value) != value):
                reasons.append(f"{path}.{key}:not_nonnegative_integer")
        elif key in {
            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        }:
            if not json_number(value):
                reasons.append(f"{path}.{key}:not_number")
        elif key == "multipleOf":
            if not json_number(value) or value <= 0:
                reasons.append(f"{path}.multipleOf:not_positive_number")
        elif key == "uniqueItems" and not isinstance(value, bool):
            reasons.append(f"{path}.uniqueItems:not_boolean")
        elif key == "enum":
            if not isinstance(value, list) or not value:
                reasons.append(f"{path}.enum:not_nonempty_array")
            elif len({canonical_bytes(item) for item in value}) != len(value):
                reasons.append(f"{path}.enum:duplicate")
    for minimum, maximum in (
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
        ("minProperties", "maxProperties"),
        ("minimum", "maximum"),
    ):
        low, high = schema.get(minimum), schema.get(maximum)
        if json_number(low) and json_number(high) and low > high:
            reasons.append(f"{path}:{minimum}_gt_{maximum}")
    return sorted(set(reasons))


def load_candidates(source: Path, max_bytes: int) -> list[dict[str, Any]]:
    candidates = []
    for category in CATEGORIES:
        directory = source / "data" / category
        if not directory.is_dir():
            raise RuntimeError(f"missing JSONSchemaBench category: {directory}")
        for path in sorted(directory.glob("*.json")):
            try:
                schema = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid source schema: {path}") from exc
            encoded = canonical_bytes(schema)
            candidates.append({
                "id": f"{category}/{path.name}",
                "category": category,
                "source_path": f"data/{category}/{path.name}",
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "schema": schema,
                "features": sorted(schema_features(schema)),
                "unsupported_reasons": unsupported_reasons(schema),
                "canonical_bytes": len(encoded),
                "selection_eligible": len(encoded) <= max_bytes,
            })
    return candidates


def select_supported(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    remaining = [
        item for item in candidates
        if item["selection_eligible"] and not item["unsupported_reasons"]
    ]
    selected: list[dict[str, Any]] = []
    covered_features: set[str] = set()
    covered_categories: set[str] = set()
    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda item: (
                len(set(item["features"]) - covered_features),
                item["category"] not in covered_categories,
                -item["canonical_bytes"],
                item["id"],
            ),
        )
        selected.append(best)
        covered_features.update(best["features"])
        covered_categories.add(best["category"])
        remaining.remove(best)
    return selected


def select_unsupported(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    remaining = [
        item for item in candidates
        if item["selection_eligible"] and item["unsupported_reasons"]
    ]
    selected: list[dict[str, Any]] = []
    covered_reasons: set[str] = set()
    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda item: (
                len({reason.rsplit(":", 1)[-1] for reason in item["unsupported_reasons"]}
                    - covered_reasons),
                -item["canonical_bytes"],
                item["id"],
            ),
        )
        selected.append(best)
        covered_reasons.update(
            reason.rsplit(":", 1)[-1] for reason in best["unsupported_reasons"]
        )
        remaining.remove(best)
    return selected


def public_entry(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item[key]
        for key in (
            "id", "category", "source_path", "sha256", "canonical_bytes",
            "features", "unsupported_reasons", "schema",
        )
    }


def checkout_pinned_source(destination: Path) -> None:
    """Sparse-fetch only the pinned dataset directories from GitHub."""
    commands = (
        ["git", "init", "--quiet", str(destination)],
        ["git", "-C", str(destination), "remote", "add", "origin",
         SOURCE_REPOSITORY],
        ["git", "-C", str(destination), "sparse-checkout", "init", "--cone"],
        ["git", "-C", str(destination), "sparse-checkout", "set",
         *[f"data/{category}" for category in CATEGORIES]],
        ["git", "-C", str(destination), "fetch", "--quiet", "--depth", "1",
         "--filter=blob:none", "origin", SOURCE_COMMIT],
        ["git", "-C", str(destination), "checkout", "--quiet", "--detach",
         "FETCH_HEAD"],
    )
    for command in commands:
        subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        help="existing checkout at SOURCE_COMMIT; omit for a sparse pinned fetch",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--supported-count", type=int, default=32)
    parser.add_argument("--unsupported-count", type=int, default=16)
    parser.add_argument("--smoke-count", type=int, default=12)
    parser.add_argument("--max-schema-bytes", type=int, default=16384)
    args = parser.parse_args()

    if not 0 < args.smoke_count <= args.supported_count:
        raise RuntimeError("smoke-count must be in 1..supported-count")
    temporary_source: tempfile.TemporaryDirectory[str] | None = None
    if args.source is None:
        temporary_source = tempfile.TemporaryDirectory(
            prefix="ds4-jsonschemabench-"
        )
        source = Path(temporary_source.name)
        checkout_pinned_source(source)
    else:
        source = args.source
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"JSONSchemaBench source is {commit}, expected {SOURCE_COMMIT}"
        )
    candidates = load_candidates(source, args.max_schema_bytes)
    examined_supported = sum(
        not item["unsupported_reasons"] for item in candidates
    )
    examined_unsupported = len(candidates) - examined_supported
    supported = select_supported(candidates, args.supported_count)
    unsupported = select_unsupported(candidates, args.unsupported_count)
    if len(supported) != args.supported_count or len(unsupported) != args.unsupported_count:
        raise RuntimeError("not enough schemas for the requested deterministic subset")
    record = {
        "schema_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "hugging_face": SOURCE_DATASET,
            "commit": SOURCE_COMMIT,
            "categories": list(CATEGORIES),
            "selection": (
                "greedy semantic-feature coverage, then category diversity, "
                "then smallest canonical schema; lexical id tie-break"
            ),
            "max_schema_bytes": args.max_schema_bytes,
        },
        "coverage": {
            "examined": len(candidates),
            "selection_eligible": sum(
                item["selection_eligible"] for item in candidates
            ),
            "skipped_oversize": sum(
                not item["selection_eligible"] for item in candidates
            ),
            "examined_supported": examined_supported,
            "examined_unsupported": examined_unsupported,
            "examined_support_rate": examined_supported / len(candidates),
            "selected_supported": len(supported),
            "selected_unsupported": len(unsupported),
            "selection_supported_fraction": (
                len(supported) / (len(supported) + len(unsupported))
            ),
        },
        "tiers": {
            "smoke": [item["id"] for item in supported[:args.smoke_count]],
            "safety": [item["id"] for item in supported],
        },
        "supported": [public_entry(item) for item in supported],
        "unsupported": [public_entry(item) for item in unsupported],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if temporary_source is not None:
        temporary_source.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
