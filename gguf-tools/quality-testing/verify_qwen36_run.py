#!/usr/bin/env python3
"""Validate generation-specific Qwen3.x oracle candidates without promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen36_fixtures import (
    FixtureError,
    inventory_files,
    load_json,
    sha256_file,
    validate_manifest as validate_qwen36_manifest,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def validate_generation_manifest(path: Path) -> tuple[dict, dict]:
    raw = load_json(path)
    if raw.get("schema_version") == "ds4-qwen38-fixture-manifest-v1":
        from qwen38_fixtures import validate_oracle_manifest

        return validate_oracle_manifest(path)
    return validate_qwen36_manifest(path)


def decode_hex_bytes(value: object, where: str) -> bytes:
    require(isinstance(value, str) and len(value) % 2 == 0, f"{where}: invalid byte hex")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise FixtureError(f"{where}: invalid byte hex") from exc


def verify_inventory(run: Path, index: dict) -> None:
    expected = index.get("files")
    require(isinstance(expected, list), f"{run}: index has no file inventory")
    actual = inventory_files(run)
    require(actual == expected, f"{run}: file inventory or checksum mismatch")


def verify_logits(run: Path, case_id: str, info: dict, steps: int) -> None:
    require(isinstance(info, dict), f"{case_id}: full logits are unavailable")
    require(set(info) == {"greedy", "teacher_forced"}, f"{case_id}: incomplete full-logit passes")
    vocabulary = None
    for pass_name, entry in info.items():
        require(entry.get("dtype") == "float32-le", f"{case_id}: invalid {pass_name} logits dtype")
        shape = entry.get("shape")
        require(isinstance(shape, list) and len(shape) == 2, f"{case_id}: invalid {pass_name} logits shape")
        require(shape[0] == steps and shape[1] > 0, f"{case_id}: incomplete {pass_name} logits")
        vocabulary = shape[1] if vocabulary is None else vocabulary
        require(shape[1] == vocabulary, f"{case_id}: vocabulary mismatch between logit passes")
        path = (run / entry["path"]).resolve()
        require(run.resolve() in path.parents, f"{case_id}: logits path escapes run directory")
        require(path.is_file(), f"{case_id}: missing logits file {path}")
        expected_size = steps * shape[1] * 4
        require(path.stat().st_size == expected_size == entry.get("size_bytes"), f"{case_id}: logits size mismatch")
        require(sha256_file(path) == entry.get("sha256"), f"{case_id}: logits SHA-256 mismatch")


def verify_run(run: Path, manifest_path: Path, allow_partial: bool) -> dict:
    run = run.resolve()
    index = load_json(run / "index.json")
    manifest, corpus = validate_generation_manifest(manifest_path)
    require(index.get("format") == manifest["output_format"]["version"],
            f"{run}: unsupported index format")
    require(index.get("manifest_model", manifest["model"]["id"]) ==
            manifest["model"]["id"], f"{run}: manifest model mismatch")
    environment = index.get("environment", {})
    review_status = environment.get("review_status")
    require(review_status in ("generated_unreviewed", "reviewed"), f"{run}: invalid review status")
    if review_status == "reviewed":
        review = environment.get("review")
        require(isinstance(review, dict), f"{run}: reviewed run is missing review metadata")
        require(isinstance(review.get("date"), str) and review["date"], f"{run}: reviewed run is missing review date")
        require(isinstance(review.get("basis"), str) and review["basis"], f"{run}: reviewed run is missing review basis")
    cases_by_id = {case["id"]: case for case in corpus["cases"]}
    listed = index.get("cases", [])
    ids = [case.get("id") for case in listed]
    require(len(ids) == len(set(ids)), f"{run}: duplicate case IDs")
    require(all(case_id in cases_by_id for case_id in ids), f"{run}: unknown case ID")
    if not allow_partial:
        require(set(ids) == set(cases_by_id), f"{run}: corpus coverage is incomplete")
        expected_categories = {case["category"] for case in corpus["cases"]}
        actual_categories = {cases_by_id[case_id]["category"] for case_id in ids}
        require(actual_categories == expected_categories, f"{run}: category coverage is incomplete")
    steps = index.get("environment", {}).get("parameters", {}).get("steps")
    require(isinstance(steps, int) and steps >= 32, f"{run}: oracle must contain at least 32 steps")

    responses = {}
    for case_id in ids:
        prompt_base = run / "prompts" / case_id
        rendered = prompt_base.with_suffix(".txt").read_bytes()
        require(rendered == prompt_base.with_suffix(".bytes").read_bytes(), f"{case_id}: rendered bytes mismatch")
        upstream_ids = json.loads(prompt_base.with_suffix(".tokens.json").read_text(encoding="utf-8"))
        require(isinstance(upstream_ids, list), f"{case_id}: rendered token file must contain an array")
        response = load_json(run / "responses" / f"{case_id}.json")
        recorded_native = response.get(
            "native_prompt_token_ids",
            response.get("upstream_render_token_ids", response.get("prompt_token_ids")),
        )
        require(recorded_native == upstream_ids, f"{case_id}: native rendered token IDs mismatch")
        native_bytes = decode_hex_bytes(
            response.get("native_rendered_bytes_hex", rendered.hex()),
            f"{case_id}: native rendered bytes",
        )
        if response.get("native_rendering_status", "verified") == "verified":
            require(native_bytes == rendered, f"{case_id}: native rendered bytes mismatch")
        canonical = response.get("canonical_prompt_token_ids", response.get("prompt_token_ids"))
        require(isinstance(canonical, list) and canonical, f"{case_id}: canonical prompt token IDs are missing")
        require(len(response.get("greedy_token_ids", [])) == steps, f"{case_id}: incomplete greedy continuation")
        decode_hex_bytes(response.get("greedy_bytes_hex"), f"{case_id}: greedy bytes")
        require(len(response.get("teacher_forced", [])) == steps, f"{case_id}: incomplete teacher-forced continuation")
        require(len(response.get("top_k", [])) == steps, f"{case_id}: incomplete top-k rows")
        verify_logits(run, case_id, response.get("full_logits"), steps)
        responses[case_id] = response
    verify_inventory(run, index)
    return {"index": index, "responses": responses}


def compare_runs(first: Path, first_data: dict, second: Path, second_data: dict) -> None:
    first_ids = set(first_data["responses"])
    require(first_ids == set(second_data["responses"]), "repeat run case sets differ")
    for case_id in sorted(first_ids):
        for suffix in (".txt", ".bytes", ".tokens.json"):
            left = first / "prompts" / f"{case_id}{suffix}"
            right = second / "prompts" / f"{case_id}{suffix}"
            require(left.read_bytes() == right.read_bytes(), f"{case_id}: repeat rendering differs for {suffix}")
        left = first_data["responses"][case_id]
        right = second_data["responses"][case_id]
        for field in (
            "prompt_token_ids", "canonical_prompt_token_ids", "native_prompt_token_ids",
            "upstream_render_token_ids", "native_rendered_bytes_hex",
            "greedy_token_ids", "greedy_bytes_hex",
        ):
            require(left.get(field) == right.get(field), f"{case_id}: repeat {field} differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        first = verify_run(args.run, args.manifest, args.allow_partial)
        if args.repeat:
            second = verify_run(args.repeat, args.manifest, args.allow_partial)
            compare_runs(args.run.resolve(), first, args.repeat.resolve(), second)
        print(json.dumps({
            "status": "PASS",
            "run": str(args.run.resolve()),
            "repeat": str(args.repeat.resolve()) if args.repeat else None,
            "cases": len(first["responses"]),
        }, sort_keys=True))
        return 0
    except (FixtureError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
