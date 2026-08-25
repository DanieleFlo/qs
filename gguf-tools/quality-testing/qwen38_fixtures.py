#!/usr/bin/env python3
"""Dependency-free validation for the pinned Qwen3.8 target and MTP fixtures."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from qwen36_fixtures import (
    FixtureError as QwenFixtureError,
    find_case,
    inventory_files,
    materialize_case,
    platform_provenance,
    validate_context_profile,
    validate_corpus,
    write_json,
)


SCHEMA_VERSION = "ds4-qwen38-fixture-manifest-v1"
SNAPSHOT_VERSION = "ds4-qwen38-gguf-snapshot-v1"
ORACLE_VERSION = "ds4-qwen38-oracle-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_CUDA_FORMATS = {
    "Q3_K", "IQ2_XS", "IQ3_XXS", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS",
}
# Compatibility alias for downstream fixture users written while the runtime
# intentionally rejected this model.  New validation uses REQUIRED_CUDA_FORMATS.
EXPECTED_MISSING_CUDA_FORMATS = REQUIRED_CUDA_FORMATS


class FixtureError(QwenFixtureError):
    pass


def ensure_staging(path: Path, *, create: bool = True) -> Path:
    """Create/validate a generation-specific unreviewed-oracle directory."""
    path = path.resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise FixtureError(f"staging directory does not exist: {path}")
    marker = path / ".ds4-qwen38-staging"
    if marker.exists():
        if marker.read_text(encoding="utf-8").strip() != SCHEMA_VERSION:
            raise FixtureError(f"invalid Qwen3.8 staging marker: {marker}")
    elif create:
        marker.write_text(SCHEMA_VERSION + "\n", encoding="utf-8")
    else:
        raise FixtureError(f"Qwen3.8 staging marker is missing: {marker}")
    return path


def _duplicate_guard(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fp:
            value = json.load(fp, object_pairs_hook=_duplicate_guard)
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FixtureError(f"{path}: top-level JSON value must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def _validate_snapshot(manifest_path: Path, artifact: dict) -> dict:
    metadata = artifact.get("metadata")
    _require(isinstance(metadata, dict), f"{artifact.get('role')}: metadata is missing")
    raw_snapshot = metadata.get("snapshot_file")
    _require(isinstance(raw_snapshot, str) and raw_snapshot,
             f"{artifact.get('role')}: snapshot_file is missing")
    snapshot_path = (manifest_path.parent / raw_snapshot).resolve()
    _require(snapshot_path.is_file(), f"snapshot is missing: {snapshot_path}")
    expected_snapshot_sha = metadata.get("snapshot_sha256")
    _require(isinstance(expected_snapshot_sha, str) and
             SHA256_RE.fullmatch(expected_snapshot_sha) is not None,
             f"{artifact.get('role')}: invalid snapshot SHA-256")
    _require(sha256_file(snapshot_path) == expected_snapshot_sha,
             f"{artifact.get('role')}: snapshot SHA-256 mismatch")

    snapshot = load_json(snapshot_path)
    _require(snapshot.get("format") == SNAPSHOT_VERSION,
             f"{artifact.get('role')}: unsupported snapshot format")
    actual_artifact = snapshot.get("artifact", {})
    for field in ("filename", "size_bytes", "sha256"):
        _require(actual_artifact.get(field) == artifact.get(field),
                 f"{artifact.get('role')}: snapshot {field} mismatch")
    gguf = snapshot.get("gguf", {})
    snap_metadata = snapshot.get("metadata", {})
    _require(gguf.get("version") == 3, f"{artifact.get('role')}: GGUF v3 required")
    _require(gguf.get("tensor_count") == metadata.get("tensor_count"),
             f"{artifact.get('role')}: tensor count mismatch")
    _require(snap_metadata.get("general.name") == "Qwen3.8-27B",
             f"{artifact.get('role')}: wrong model generation")
    _require(snap_metadata.get("general.architecture") == "qwen35",
             f"{artifact.get('role')}: wrong architecture")
    _require(snap_metadata.get("qwen35.context_length") == 262144,
             f"{artifact.get('role')}: wrong context length")
    _require(snapshot.get("tensor_type_counts") == metadata.get("tensor_type_counts"),
             f"{artifact.get('role')}: quant inventory mismatch")
    template = snap_metadata.get("tokenizer.chat_template", {})
    _require(template.get("utf8_size_bytes") == metadata.get("chat_template_size_bytes"),
             f"{artifact.get('role')}: chat template size mismatch")
    _require(template.get("utf8_sha256") == metadata.get("chat_template_sha256"),
             f"{artifact.get('role')}: chat template SHA-256 mismatch")
    return snapshot


def validate_manifest(manifest_path: Path,
                      artifact_paths: dict[str, Path] | None = None) -> tuple[dict, dict[str, dict]]:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    _require(manifest.get("schema_version") == SCHEMA_VERSION,
             "unsupported Qwen3.8 manifest schema")
    model = manifest.get("model", {})
    _require(model.get("id") == "Qwen3.8-27B-UD-Q4_K_S", "wrong model id")
    _require(model.get("architecture") == "qwen35", "wrong model architecture")
    for field in ("source_revision", "gguf_revision"):
        _require(isinstance(model.get(field), str) and
                 REVISION_RE.fullmatch(model[field]) is not None,
                 f"invalid {field}")

    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list), "artifacts must be an array")
    by_role = {item.get("role"): item for item in artifacts if isinstance(item, dict)}
    _require(set(by_role) == {"target", "mtp"}, "target and mtp artifacts are required")
    snapshots = {role: _validate_snapshot(manifest_path, artifact)
                 for role, artifact in by_role.items()}
    _require(snapshots["target"]["gguf"]["tokenizer_metadata_sha256"] ==
             snapshots["mtp"]["gguf"]["tokenizer_metadata_sha256"],
             "target and MTP tokenizer metadata differ")

    tokenizer = manifest.get("tokenizer", {})
    template = manifest.get("chat_template", {})
    for name, entry in (("tokenizer", tokenizer), ("chat_template", template)):
        _require(isinstance(entry.get("sha256"), str) and
                 SHA256_RE.fullmatch(entry["sha256"]) is not None,
                 f"invalid upstream {name} SHA-256")
        _require(entry.get("revision") == model.get("source_revision"),
                 f"{name} revision does not match upstream model")

    oracle = manifest.get("oracle", {})
    _require(oracle.get("output_format") == ORACLE_VERSION,
             "Qwen3.8 oracle namespace is required")
    reference = oracle.get("reference_model", "")
    _require("Qwen3.8" in reference and "Qwen3.6" not in reference,
             "Qwen3.8 must use a generation-specific numerical oracle")
    _require(oracle.get("must_not_compare_with") == "Qwen3.6",
             "cross-generation oracle guard is missing")
    output_format = manifest.get("output_format", {})
    _require(output_format.get("version") == ORACLE_VERSION,
             "Qwen3.8 output format namespace is required")
    environments = manifest.get("oracle_environments", {})
    _require(set(environments) == {"llama.cpp", "transformers", "vllm"},
             "Qwen3.8 oracle environments are incomplete")
    equivalence = manifest.get("equivalence_thresholds", {})
    _require(equivalence.get("version") ==
             "ds4-qwen38-equivalence-thresholds-v1",
             "Qwen3.8 equivalence threshold namespace is required")
    short_gates = manifest.get("short_message_gates", {})
    _require(short_gates.get("version") == "ds4-qwen38-short-gates-v1",
             "Qwen3.8 short-message gate namespace is required")

    gates = manifest.get("gates", {})
    _require(gates.get("minimum_greedy_tokens", 0) >= 32, "greedy gate is too short")
    _require(gates.get("minimum_teacher_forced_tokens", 0) >= 32,
             "teacher-forced gate is too short")
    _require(gates.get("prefill_tokens_per_second_min") == 500.0,
             "prefill performance gate must match the Qwen production contract")
    _require(gates.get("generation_tokens_per_second_min") == 15.0,
             "generation performance gate must match the Qwen production contract")
    _require(gates.get("mtp_short_generation_tokens_per_second_min") == 20.0,
             "short-context MTP regression gate must remain at 20 tok/s")

    runtime = manifest.get("runtime", {})
    _require(runtime.get("status") == "kernel_ready",
             "Qwen3.8 runtime kernel status is not promoted")
    _require(set(runtime.get("supported_cuda_formats", [])) == REQUIRED_CUDA_FORMATS,
             "supported CUDA quant-format inventory changed")
    _require(runtime.get("missing_cuda_formats") == [],
             "Qwen3.8 runtime still lists missing CUDA formats")

    for role, path in (artifact_paths or {}).items():
        _require(role in by_role, f"unknown artifact role: {role}")
        path = path.resolve()
        _require(path.is_file(), f"artifact is missing: {path}")
        _require(path.stat().st_size == by_role[role]["size_bytes"],
                 f"{role}: artifact size mismatch")
        _require(sha256_file(path) == by_role[role]["sha256"],
                 f"{role}: artifact SHA-256 mismatch")
    return manifest, snapshots


def validate_oracle_manifest(manifest_path: Path) -> tuple[dict, dict]:
    """Validate the pinned Qwen3.8 artifacts and its shared input corpus."""
    manifest, _snapshots = validate_manifest(manifest_path)
    corpus_path = (manifest_path.resolve().parent / manifest["corpus"]).resolve()
    return manifest, validate_corpus(corpus_path)
