#!/usr/bin/env python3
"""Dependency-free helpers for versioned Qwen3.6 quality fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


SCHEMA_VERSION = "ds4-qwen36-fixture-manifest-v1"
CORPUS_VERSION = "ds4-qwen36-corpus-v1"
OUTPUT_VERSION = "ds4-qwen36-oracle-v1"
STAGING_MARKER = ".ds4-qwen36-staging"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_CATEGORIES = {
    "single-token",
    "short-multilingual",
    "unicode-multilingual",
    "system-thinking-off",
    "system-thinking-on",
    "multi-turn",
    "code-completion",
    "tool-calling",
    "long-canary",
}


class FixtureError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fp:
            value = json.load(fp, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FixtureError(f"{path}: top-level JSON value must be an object")
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_object(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise FixtureError(f"{where} must be an object")
    return value


def _require_list(value: object, where: str) -> list:
    if not isinstance(value, list):
        raise FixtureError(f"{where} must be an array")
    return value


def _require_string(value: object, where: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise FixtureError(f"{where} must be a non-empty string")
    return value


def _require_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FixtureError(f"{where} must be a positive integer")
    return value


def _require_fields(value: dict, fields: set[str], where: str) -> None:
    missing = fields - value.keys()
    if missing:
        raise FixtureError(f"{where} missing required field(s): {', '.join(sorted(missing))}")


def _validate_url(value: object, where: str) -> None:
    text = _require_string(value, where)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        raise FixtureError(f"{where} must be an absolute HTTPS URL")


def _validate_sha(value: object, where: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise FixtureError(f"{where} must be a lowercase SHA-256")


def _validate_revision(value: object, where: str) -> None:
    if not isinstance(value, str) or not REVISION_RE.fullmatch(value):
        raise FixtureError(f"{where} must be a 40-character lowercase commit")


def _resolve_declared_file(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def validate_corpus(corpus_path: Path) -> dict:
    corpus = load_json(corpus_path)
    _require_fields(
        corpus,
        {"schema_version", "minimum_greedy_tokens", "minimum_teacher_forced_tokens", "cases"},
        str(corpus_path),
    )
    if corpus["schema_version"] != CORPUS_VERSION:
        raise FixtureError(f"{corpus_path}: unsupported corpus schema")
    if _require_int(corpus["minimum_greedy_tokens"], "minimum_greedy_tokens") < 32:
        raise FixtureError("minimum_greedy_tokens must be at least 32")
    if _require_int(corpus["minimum_teacher_forced_tokens"], "minimum_teacher_forced_tokens") < 32:
        raise FixtureError("minimum_teacher_forced_tokens must be at least 32")

    ids: set[str] = set()
    categories: set[str] = set()
    cases = _require_list(corpus["cases"], "cases")
    if not cases:
        raise FixtureError("corpus must contain at least one case")
    for index, raw_case in enumerate(cases):
        case = _require_object(raw_case, f"cases[{index}]")
        _require_fields(
            case,
            {"id", "category", "thinking", "messages", "tools", "expected_min_prompt_tokens", "canaries"},
            f"cases[{index}]",
        )
        case_id = _require_string(case["id"], f"cases[{index}].id")
        if case_id in ids:
            raise FixtureError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        category = _require_string(case["category"], f"{case_id}.category")
        categories.add(category)
        if not isinstance(case["thinking"], bool):
            raise FixtureError(f"{case_id}.thinking must be boolean")
        messages = _require_list(case["messages"], f"{case_id}.messages")
        if not messages:
            raise FixtureError(f"{case_id}.messages must not be empty")
        for message_index, raw_message in enumerate(messages):
            message = _require_object(raw_message, f"{case_id}.messages[{message_index}]")
            role = _require_string(message.get("role"), f"{case_id}.messages[{message_index}].role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise FixtureError(f"{case_id}: unsupported message role {role}")
            if "content" not in message and "content_generator" not in message:
                raise FixtureError(f"{case_id}: every message needs content or content_generator")
        _require_list(case["tools"], f"{case_id}.tools")
        _require_int(case["expected_min_prompt_tokens"], f"{case_id}.expected_min_prompt_tokens")
        canaries = _require_list(case["canaries"], f"{case_id}.canaries")
        if category == "long-canary":
            _require_fields(case, {"generator"}, case_id)
            generator = _require_object(case["generator"], f"{case_id}.generator")
            _require_fields(generator, {"seed", "paragraphs", "words_per_paragraph"}, f"{case_id}.generator")
            _require_int(generator["seed"], f"{case_id}.generator.seed")
            _require_int(generator["paragraphs"], f"{case_id}.generator.paragraphs")
            _require_int(generator["words_per_paragraph"], f"{case_id}.generator.words_per_paragraph")
            positions = {item.get("position") for item in canaries if isinstance(item, dict)}
            if positions != {"start", "middle", "end"}:
                raise FixtureError(f"{case_id}: canaries must cover start, middle and end exactly")

    missing_categories = REQUIRED_CATEGORIES - categories
    if missing_categories:
        raise FixtureError(f"corpus missing category/categories: {', '.join(sorted(missing_categories))}")
    return corpus


def validate_manifest(manifest_path: Path, *, require_artifacts: bool = False) -> tuple[dict, dict]:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    required = {
        "schema_version", "model", "artifacts", "tokenizer", "chat_template",
        "oracle_environments", "corpus", "output_format", "verification",
    }
    _require_fields(manifest, required, str(manifest_path))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise FixtureError(f"{manifest_path}: unsupported manifest schema")

    model = _require_object(manifest["model"], "model")
    _require_fields(model, {"id", "variant", "architecture", "context_length", "source", "source_revision"}, "model")
    if model["variant"] not in {"target", "target-with-mtp"}:
        raise FixtureError("model.variant must be target or target-with-mtp")
    if model["architecture"] != "qwen35":
        raise FixtureError("model.architecture must be qwen35")
    _require_int(model["context_length"], "model.context_length")
    _validate_url(model["source"], "model.source")
    _validate_revision(model["source_revision"], "model.source_revision")

    roles: list[str] = []
    artifacts = _require_list(manifest["artifacts"], "artifacts")
    for index, raw_artifact in enumerate(artifacts):
        artifact = _require_object(raw_artifact, f"artifacts[{index}]")
        _require_fields(artifact, {"role", "url", "filename", "size_bytes", "sha256", "quantization", "local_path", "metadata"}, f"artifacts[{index}]")
        role = artifact["role"]
        if role not in {"target", "mtp"}:
            raise FixtureError(f"artifacts[{index}].role is invalid")
        roles.append(role)
        _validate_url(artifact["url"], f"artifacts[{index}].url")
        _require_string(artifact["filename"], f"artifacts[{index}].filename")
        size = _require_int(artifact["size_bytes"], f"artifacts[{index}].size_bytes")
        _validate_sha(artifact["sha256"], f"artifacts[{index}].sha256")
        _require_string(artifact["quantization"], f"artifacts[{index}].quantization")
        metadata = _require_object(artifact["metadata"], f"artifacts[{index}].metadata")
        _require_fields(metadata, {"source", "repository_revision", "general.architecture", "context_length", "expected_model", "header_inspection"}, f"artifacts[{index}].metadata")
        _validate_revision(metadata["repository_revision"], f"artifacts[{index}].metadata.repository_revision")
        if metadata["general.architecture"] != "qwen35":
            raise FixtureError(f"artifacts[{index}]: architecture mismatch")
        if metadata["context_length"] != model["context_length"]:
            raise FixtureError(f"artifacts[{index}]: context length mismatch")
        expected_model = _require_object(metadata["expected_model"], f"artifacts[{index}].metadata.expected_model")
        expected_values = {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "mtp_num_hidden_layers": 1,
            "tie_word_embeddings": False,
        }
        if expected_model != expected_values:
            raise FixtureError(f"artifacts[{index}]: expected upstream model metadata mismatch")
        if metadata["header_inspection"] not in {"verified", "not_verified"}:
            raise FixtureError(f"artifacts[{index}]: invalid header_inspection")
        local_path = artifact["local_path"]
        if local_path is not None:
            local_file = _resolve_declared_file(manifest_path, _require_string(local_path, f"artifacts[{index}].local_path"))
            if not local_file.is_file():
                raise FixtureError(f"artifact file does not exist: {local_file}")
            if local_file.stat().st_size != size:
                raise FixtureError(f"artifact size mismatch: {local_file}")
            if sha256_file(local_file) != artifact["sha256"]:
                raise FixtureError(f"artifact SHA-256 mismatch: {local_file}")
        elif require_artifacts:
            raise FixtureError(f"artifacts[{index}] has no local_path")
    if roles.count("target") != 1:
        raise FixtureError("manifest must contain exactly one target artifact")
    if model["variant"] == "target-with-mtp" and roles.count("mtp") != 1:
        raise FixtureError("target-with-mtp manifest must contain exactly one MTP artifact")
    if model["variant"] == "target" and "mtp" in roles:
        raise FixtureError("target manifest must not contain an MTP artifact")

    for field in ("tokenizer", "chat_template"):
        source_file = _require_object(manifest[field], field)
        _require_fields(source_file, {"url", "revision", "size_bytes", "sha256", "embedded_gguf_sha256", "verification"}, field)
        _validate_url(source_file["url"], f"{field}.url")
        _validate_revision(source_file["revision"], f"{field}.revision")
        _require_int(source_file["size_bytes"], f"{field}.size_bytes")
        _validate_sha(source_file["sha256"], f"{field}.sha256")
        if source_file["embedded_gguf_sha256"] is not None:
            _validate_sha(source_file["embedded_gguf_sha256"], f"{field}.embedded_gguf_sha256")
        if source_file["verification"] not in {"source_verified", "gguf_not_verified"}:
            raise FixtureError(f"{field}.verification is invalid")

    environments = _require_object(manifest["oracle_environments"], "oracle_environments")
    if set(environments) != {"llama.cpp", "transformers", "vllm"}:
        raise FixtureError("oracle_environments must contain llama.cpp, transformers and vllm")
    for name, raw_environment in environments.items():
        environment = _require_object(raw_environment, f"oracle_environments.{name}")
        fields = {"kind", "status", "version", "commit", "build_flags", "backend", "hardware", "dtype", "parameters", "reason"}
        _require_fields(environment, fields, f"oracle_environments.{name}")
        expected_kind = "numeric_same_gguf" if name == "llama.cpp" else "semantic_upstream"
        if environment["kind"] != expected_kind:
            raise FixtureError(f"oracle_environments.{name}.kind must be {expected_kind}")
        if environment["status"] == "verified":
            for field in ("version", "commit", "build_flags", "backend", "hardware", "dtype"):
                _require_string(environment[field], f"oracle_environments.{name}.{field}")
            if environment["reason"] is not None:
                raise FixtureError(f"verified oracle {name} must not have a reason")
        elif environment["status"] == "not_verified":
            _require_string(environment["reason"], f"oracle_environments.{name}.reason")
        else:
            raise FixtureError(f"oracle_environments.{name}.status is invalid")
        _require_object(environment["parameters"], f"oracle_environments.{name}.parameters")

    output = _require_object(manifest["output_format"], "output_format")
    if output != {"version": OUTPUT_VERSION, "index": "JSON", "large_logits": "float32-le", "checksums": "SHA-256"}:
        raise FixtureError("output_format must match the documented v1 format")
    verification = _require_object(manifest["verification"], "verification")
    _require_fields(verification, {"status", "reasons"}, "verification")
    reasons = _require_list(verification["reasons"], "verification.reasons")
    if verification["status"] == "verified":
        if reasons:
            raise FixtureError("verified manifest must not contain verification reasons")
        if any(item["status"] != "verified" for item in environments.values()):
            raise FixtureError("manifest cannot be verified while an oracle is not verified")
        if any(item["metadata"]["header_inspection"] != "verified" for item in artifacts):
            raise FixtureError("manifest cannot be verified before every GGUF header is inspected")
    elif verification["status"] == "not_verified":
        if not reasons or not all(isinstance(reason, str) and reason for reason in reasons):
            raise FixtureError("not_verified manifest needs at least one reason")
    else:
        raise FixtureError("verification.status is invalid")

    corpus_file = _resolve_declared_file(manifest_path, _require_string(manifest["corpus"], "corpus"))
    if not corpus_file.is_file():
        raise FixtureError(f"corpus file does not exist: {corpus_file}")
    corpus = validate_corpus(corpus_file)
    return manifest, corpus


_WORDS = (
    "amber", "bridge", "cedar", "delta", "ember", "forest", "granite", "harbor",
    "island", "jasmine", "kernel", "lantern", "meadow", "nebula", "orchid", "pebble",
    "quartz", "river", "signal", "timber", "upland", "velvet", "willow", "xenon",
    "yellow", "zephyr",
)


def generate_long_content(case: dict) -> str:
    generator = case["generator"]
    state = int(generator["seed"]) & 0xFFFFFFFF
    paragraphs = int(generator["paragraphs"])
    words_per_paragraph = int(generator["words_per_paragraph"])
    canaries = {item["position"]: item["value"] for item in case["canaries"]}
    canary_at = {0: canaries["start"], paragraphs // 2: canaries["middle"], paragraphs - 1: canaries["end"]}
    rendered: list[str] = []
    for paragraph in range(paragraphs):
        words: list[str] = []
        for _ in range(words_per_paragraph):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            words.append(_WORDS[state % len(_WORDS)])
        prefix = f"Record {paragraph:04d}: "
        if paragraph in canary_at:
            prefix += canary_at[paragraph] + " "
        rendered.append(prefix + " ".join(words) + ".")
    rendered.append("Return the three DS4 canaries in start, middle, end order and nothing else.")
    return "\n".join(rendered)


def materialize_case(case: dict) -> dict:
    result = json.loads(json.dumps(case, ensure_ascii=False))
    for message in result["messages"]:
        if message.pop("content_generator", None) == "deterministic_canary_text_v1":
            message["content"] = generate_long_content(result)
    result.pop("generator", None)
    return result


def find_case(corpus: dict, case_id: str) -> dict:
    for case in corpus["cases"]:
        if case["id"] == case_id:
            return case
    raise FixtureError(f"unknown case id: {case_id}")


def ensure_staging(path: Path, *, create: bool = True) -> Path:
    resolved = path.expanduser().resolve()
    if any(part.lower() in {"golden", "goldens"} for part in resolved.parts):
        raise FixtureError(f"refusing to use a golden directory as staging: {resolved}")
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    marker = resolved / STAGING_MARKER
    if marker.exists():
        return resolved
    if any(resolved.iterdir()):
        raise FixtureError(f"staging directory is non-empty and unmarked: {resolved}")
    if not create:
        raise FixtureError(f"staging marker is missing: {marker}")
    marker.write_text("Generated Qwen3.6 oracle staging; never accept automatically.\n", encoding="utf-8")
    return resolved


def inventory_files(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"index.json", STAGING_MARKER}:
            entries.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return entries


def platform_provenance() -> dict[str, str]:
    import platform

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "pid": str(os.getpid()),
    }
