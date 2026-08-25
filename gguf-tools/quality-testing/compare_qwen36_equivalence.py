#!/usr/bin/env python3
"""Compare audited full-vocabulary Qwen3.x runs within one generation.

The large arrays stay in the task-2 float32-le files.  This tool validates the
run inventories before mapping them and writes only compact per-position and
aggregate metrics to the report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen36_fixtures import (
    FixtureError,
    inventory_files,
    load_json,
    validate_manifest as validate_qwen36_manifest,
    write_json,
)

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by deployment hosts
    raise SystemExit("numpy is required to compare full-vocabulary logits") from exc


REPORT_FORMAT = "ds4-qwen36-equivalence-report-v1"
RUN_FORMAT = "ds4-qwen36-oracle-v1"


def validate_generation_manifest(path: Path) -> tuple[dict, dict]:
    raw = load_json(path)
    if raw.get("schema_version") == "ds4-qwen38-fixture-manifest-v1":
        from qwen38_fixtures import validate_oracle_manifest

        return validate_oracle_manifest(path)
    return validate_qwen36_manifest(path)


def select_generation_formats(manifest: dict) -> None:
    global REPORT_FORMAT, RUN_FORMAT
    RUN_FORMAT = manifest["output_format"]["version"]
    if manifest.get("schema_version") == "ds4-qwen38-fixture-manifest-v1":
        REPORT_FORMAT = "ds4-qwen38-equivalence-report-v1"
    else:
        REPORT_FORMAT = "ds4-qwen36-equivalence-report-v1"


@dataclass
class RunData:
    root: Path
    index: dict[str, Any]
    responses: dict[str, dict[str, Any]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    require(path == root or root in path.parents, f"path escapes run directory: {relative}")
    return path


def _load_tokens(path: Path) -> list[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, list), f"{path}: token file must contain an array")
    require(all(isinstance(token, int) and token >= 0 for token in value), f"{path}: invalid token ID")
    return value


def _decode_hex_bytes(value: object, where: str) -> bytes:
    require(isinstance(value, str) and len(value) % 2 == 0, f"{where}: invalid byte hex")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise FixtureError(f"{where}: invalid byte hex") from exc


def _teacher_tokens(response: dict[str, Any]) -> list[int]:
    rows = response.get("teacher_forced")
    require(isinstance(rows, list), "response has no teacher_forced rows")
    tokens: list[int] = []
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("token_id"), int), "invalid teacher-forced row")
        tokens.append(int(row["token_id"]))
    return tokens


def load_run(path: Path) -> RunData:
    root = path.resolve()
    index = load_json(root / "index.json")
    require(index.get("format") == RUN_FORMAT, f"{root}: unsupported run format")
    require(index.get("files") == inventory_files(root), f"{root}: file inventory or checksum mismatch")
    listed = index.get("cases")
    require(isinstance(listed, list) and listed, f"{root}: no cases")
    responses: dict[str, dict[str, Any]] = {}
    for entry in listed:
        require(isinstance(entry, dict) and isinstance(entry.get("id"), str), f"{root}: invalid case entry")
        case_id = entry["id"]
        require(case_id not in responses, f"{root}: duplicate case ID {case_id}")
        response_path = _safe_child(root, entry.get("response_file", f"responses/{case_id}.json"))
        response = load_json(response_path)
        prompt_path = root / "prompts" / f"{case_id}.tokens.json"
        require(prompt_path.is_file(), f"{root}: missing prompt tokens for {case_id}")
        prompt_tokens = _load_tokens(prompt_path)
        native_tokens = response.get(
            "native_prompt_token_ids",
            response.get("upstream_render_token_ids", response.get("prompt_token_ids")),
        )
        require(native_tokens == prompt_tokens, f"{root}: native prompt token mismatch for {case_id}")
        canonical_tokens = response.get("canonical_prompt_token_ids", response.get("prompt_token_ids"))
        require(isinstance(canonical_tokens, list), f"{root}: missing canonical prompt tokens for {case_id}")
        require(isinstance(response.get("greedy_token_ids"), list), f"{root}: missing greedy tokens for {case_id}")
        _teacher_tokens(response)
        responses[case_id] = response
    return RunData(root=root, index=index, responses=responses)


def _logits_info(response: dict[str, Any], pass_name: str) -> dict[str, Any]:
    info = response.get("full_logits")
    require(isinstance(info, dict), "full-vocabulary logits are unavailable")
    entry = info.get(pass_name)
    require(isinstance(entry, dict), f"missing {pass_name} logits")
    require(entry.get("dtype") == "float32-le", f"unsupported {pass_name} dtype")
    shape = entry.get("shape")
    require(isinstance(shape, list) and len(shape) == 2, f"invalid {pass_name} shape")
    require(all(isinstance(value, int) and value > 0 for value in shape), f"invalid {pass_name} shape")
    require(entry.get("row_stride_bytes") == shape[1] * 4, f"invalid {pass_name} row stride")
    return entry


def _map_logits(run: RunData, response: dict[str, Any], pass_name: str) -> np.memmap:
    info = _logits_info(response, pass_name)
    shape = tuple(info["shape"])
    path = _safe_child(run.root, info["path"])
    require(path.is_file(), f"missing logits file: {path}")
    require(path.stat().st_size == shape[0] * shape[1] * 4 == info.get("size_bytes"), f"logits size mismatch: {path}")
    relative = path.relative_to(run.root).as_posix()
    inventory = {entry["path"]: entry for entry in run.index["files"]}
    require(relative in inventory, f"logits are absent from run inventory: {relative}")
    require(info.get("sha256") == inventory[relative]["sha256"], f"logits metadata SHA-256 mismatch: {path}")
    return np.memmap(path, dtype="<f4", mode="r", shape=shape)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 1.0 if np.array_equal(left, right) else 0.0
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denom)


def _top_ids(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(values))
    selected = np.argpartition(values, len(values) - k)[-k:]
    return selected[np.argsort(-values[selected], kind="stable")]


def _log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    shifted = values.astype(np.float64) - maximum
    return shifted - math.log(float(np.exp(shifted).sum()))


def position_metrics(left_raw: np.ndarray, right_raw: np.ndarray, oracle_token: int, top_k: int) -> dict[str, Any]:
    require(left_raw.shape == right_raw.shape and left_raw.ndim == 1, "logit row shape mismatch")
    left_nan = int(np.isnan(left_raw).sum())
    right_nan = int(np.isnan(right_raw).sum())
    left_inf = int(np.isinf(left_raw).sum())
    right_inf = int(np.isinf(right_raw).sum())
    finite = left_nan + right_nan + left_inf + right_inf == 0
    result: dict[str, Any] = {
        "left_nan": left_nan, "right_nan": right_nan,
        "left_inf": left_inf, "right_inf": right_inf,
    }
    if not finite:
        result["metrics_available"] = False
        return result

    left = left_raw.astype(np.float64)
    right = right_raw.astype(np.float64)
    left_top = _top_ids(left, 2)
    right_top = _top_ids(right, 2)
    left_argmax, right_argmax = int(left_top[0]), int(right_top[0])
    left_centered = left - left[left_argmax]
    right_centered = right - right[right_argmax]
    delta = right_centered - left_centered
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    cosine = float(np.dot(left_centered, right_centered) / denom) if denom else (1.0 if np.array_equal(left_centered, right_centered) else 0.0)

    left_logp = _log_softmax(left)
    right_logp = _log_softmax(right)
    left_prob = np.exp(left_logp)
    right_prob = np.exp(right_logp)
    log_middle = np.logaddexp(left_logp, right_logp) - math.log(2.0)
    kl = float(np.sum(left_prob * (left_logp - right_logp)))
    js = float(0.5 * np.sum(left_prob * (left_logp - log_middle)) + 0.5 * np.sum(right_prob * (right_logp - log_middle)))

    left_ids = _top_ids(left, top_k)
    right_ids = _top_ids(right, top_k)
    common = np.intersect1d(left_ids, right_ids, assume_unique=True)
    spearman = _correlation(_rankdata(left[common]), _rankdata(right[common])) if len(common) else 0.0
    pair_total = pair_agree = 0
    for first in range(len(left_ids)):
        for second in range(first + 1, len(left_ids)):
            a, b = int(left_ids[first]), int(left_ids[second])
            if right[a] == right[b]:
                continue
            pair_total += 1
            pair_agree += int(right[a] > right[b])
    require(0 <= oracle_token < len(left), f"oracle token {oracle_token} is outside vocabulary")
    bit_differences = int(np.count_nonzero(left_raw.view(np.uint32) != right_raw.view(np.uint32)))
    return {
        **result,
        "metrics_available": True,
        "left_argmax": left_argmax,
        "right_argmax": right_argmax,
        "greedy_agreement": left_argmax == right_argmax,
        "left_margin": float(left[left_top[0]] - left[left_top[1]]),
        "right_margin": float(right[right_top[0]] - right[right_top[1]]),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_error": float(np.max(np.abs(delta))),
        "cosine_similarity": cosine,
        "kl_left_right": max(0.0, kl),
        "js_divergence": max(0.0, js),
        "top_k": min(top_k, len(left)),
        "top_k_overlap": float(len(common) / min(top_k, len(left))),
        "top_k_spearman": spearman,
        "top_k_rank_agreement": float(pair_agree / pair_total) if pair_total else 1.0,
        "oracle_token": oracle_token,
        "left_oracle_logprob": float(left_logp[oracle_token]),
        "right_oracle_logprob": float(right_logp[oracle_token]),
        "oracle_logprob_abs_error": float(abs(right_logp[oracle_token] - left_logp[oracle_token])),
        "left_nll": float(-left_logp[oracle_token]),
        "right_nll": float(-right_logp[oracle_token]),
        "different_float_count": bit_differences,
    }


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def first_sequence_difference(
    left: list[int] | bytes, right: list[int] | bytes,
) -> int | None:
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            return prefix
        prefix += 1
    return None if len(left) == len(right) else prefix


def _rendering_metrics(left: RunData, right: RunData, case_id: str) -> dict[str, Any]:
    left_response, right_response = left.responses[case_id], right.responses[case_id]
    left_canonical = (left.root / "prompts" / f"{case_id}.bytes").read_bytes()
    right_canonical = (right.root / "prompts" / f"{case_id}.bytes").read_bytes()
    left_bytes = _decode_hex_bytes(
        left_response.get("native_rendered_bytes_hex", left_canonical.hex()),
        f"{case_id}: left native rendering",
    )
    right_bytes = _decode_hex_bytes(
        right_response.get("native_rendered_bytes_hex", right_canonical.hex()),
        f"{case_id}: right native rendering",
    )
    left_tokens = left_response.get(
        "native_prompt_token_ids",
        left_response.get("upstream_render_token_ids", left_response.get("prompt_token_ids")),
    )
    right_tokens = right_response.get(
        "native_prompt_token_ids",
        right_response.get("upstream_render_token_ids", right_response.get("prompt_token_ids")),
    )
    require(isinstance(left_tokens, list) and isinstance(right_tokens, list), f"{case_id}: native prompt tokens are missing")
    left_status = left_response.get("native_rendering_status", "verified")
    right_status = right_response.get("native_rendering_status", "verified")
    return {
        "status": "verified" if left_status == right_status == "verified" else "not_verified",
        "bytes_equal": left_bytes == right_bytes,
        "token_ids_equal": left_tokens == right_tokens,
        "first_byte_difference": first_sequence_difference(left_bytes, right_bytes),
        "first_token_difference": first_sequence_difference(left_tokens, right_tokens),
        "left_bytes": len(left_bytes),
        "right_bytes": len(right_bytes),
        "left_tokens": len(left_tokens),
        "right_tokens": len(right_tokens),
    }


def _perplexity(mean_nll: float | None) -> float | None:
    if mean_nll is None:
        return None
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return math.inf


def _teacher_diagnostics(positions: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [item for item in positions if item.get("metrics_available")]
    left_nll = [float(item["left_nll"]) for item in finite]
    right_nll = [float(item["right_nll"]) for item in finite]
    left_mean = sum(left_nll) / len(left_nll) if left_nll else None
    right_mean = sum(right_nll) / len(right_nll) if right_nll else None
    errors = [float(item["oracle_logprob_abs_error"]) for item in finite]
    return {
        "positions": len(positions),
        "finite_positions": len(finite),
        "left_mean_nll": left_mean,
        "right_mean_nll": right_mean,
        "left_perplexity": _perplexity(left_mean),
        "right_perplexity": _perplexity(right_mean),
        "oracle_logprob_mae": sum(errors) / len(errors) if errors else None,
    }


def _run_provenance(run: RunData) -> dict[str, Any]:
    environment = run.index.get("environment", {})
    first_response = next(iter(run.responses.values()))
    return {
        "run_id": run.index.get("run_id"),
        "model": run.index.get("manifest_model"),
        "engine": environment.get("oracle", environment.get("engine")),
        "engine_commit": environment.get("engine_commit"),
        "engine_version": environment.get("engine_version", first_response.get("engine_version")),
        "build_flags": environment.get("build_flags"),
        "backend": environment.get("backend"),
        "hardware": environment.get("hardware"),
        "dtype": environment.get("dtype"),
        "parameters": environment.get("parameters"),
        "artifacts": environment.get("artifacts"),
        "review_status": environment.get("review_status"),
    }


def _threshold_failure(metric: str, value: float, thresholds: dict[str, Any]) -> str | None:
    minima = {"cosine_similarity", "top_k_overlap", "top_k_spearman", "top_k_rank_agreement"}
    limit = thresholds.get(metric)
    if limit is None:
        return None
    failed = value < float(limit) if metric in minima else value > float(limit)
    return f"{metric}={value:.9g} threshold={limit}" if failed else None


def fixed_gate_failed(value: float, limit: float, *, minimum: bool) -> bool:
    return value < limit if minimum else value > limit


def compare_runs(
    left: RunData,
    right: RunData,
    manifest: dict[str, Any],
    mode: str,
    top_k: int,
    diagnostic: bool,
    *,
    suite: str = "all",
    corpus: dict[str, Any] | None = None,
    selected_cases: list[str] | None = None,
) -> tuple[dict[str, Any], int]:
    left_ids = list(left.responses)
    right_ids = list(right.responses)
    short_gates = manifest["short_message_gates"]
    if selected_cases:
        require(suite == "all", "--case cannot be combined with --suite short")
        require(len(selected_cases) == len(set(selected_cases)), "duplicate --case")
        missing = [
            case_id for case_id in selected_cases
            if case_id not in left.responses or case_id not in right.responses
        ]
        require(not missing, f"selected case(s) missing from a run: {', '.join(missing)}")
        selected_ids = selected_cases
    elif suite == "short":
        require(corpus is not None, "short suite requires the validated corpus")
        excluded = set(short_gates["excluded_categories"])
        selected_ids = [case["id"] for case in corpus["cases"] if case["category"] not in excluded]
        missing = [
            case_id for case_id in selected_ids
            if case_id not in left.responses or case_id not in right.responses
        ]
        require(not missing, f"short suite is missing case(s): {', '.join(missing)}")
        require(top_k == short_gates["top_k"], f"short suite requires top-k={short_gates['top_k']}")
    else:
        require(left_ids == right_ids, "run case sets or order differ")
        selected_ids = left_ids
    left_prov, right_prov = _run_provenance(left), _run_provenance(right)
    require(left_prov["model"] == right_prov["model"] == manifest["model"]["id"], "manifest model mismatch")
    require(left_prov["artifacts"] == right_prov["artifacts"], "model artifact provenance differs")
    target = next(item for item in manifest["artifacts"] if item["role"] == "target")
    expected_target = {
        "filename": target["filename"],
        "size_bytes": target["size_bytes"],
        "sha256": target["sha256"],
    }
    require(left_prov["artifacts"].get("target") == expected_target, "run target artifact differs from manifest")
    if not diagnostic:
        for label, provenance in (("left", left_prov), ("right", right_prov)):
            for field in ("engine", "engine_commit", "backend", "hardware", "dtype", "parameters", "artifacts", "review_status"):
                require(provenance.get(field) is not None, f"{label} run has incomplete provenance: {field}")
        if mode == "ds4-vs-llama":
            require(left_prov["engine"] == "llama.cpp" and right_prov["engine"] == "ds4", "ds4-vs-llama requires llama.cpp reference and DS4 candidate")
        else:
            require(left_prov["engine"] == right_prov["engine"] == "ds4", "ds4-vs-ds4 requires two DS4 runs")

    profile = manifest["equivalence_thresholds"]["cross_engine" if mode == "ds4-vs-llama" else "internal"]
    thresholds = profile.get("metrics") or {}
    # Diagnostic comparisons are how a new engine commit is measured before
    # its provenance can replace the reviewed calibration.  They can never
    # PASS (see the status selection below), so retain the metric thresholds
    # but do not require the candidate to claim the old calibrated commit.
    if mode == "ds4-vs-llama" and profile["status"] == "verified" and not diagnostic:
        calibration = profile["calibration"]
        target_artifact = right_prov["artifacts"]["target"]
        right_parameters = right_prov["parameters"]
        require(target_artifact["sha256"] == calibration["model_sha256"], "candidate model differs from threshold calibration")
        require(right_prov["engine_commit"] == calibration["ds4_commit"], "DS4 commit differs from threshold calibration")
        require(left_prov["engine_version"] == calibration["oracle_version"], "oracle version differs from threshold calibration")
        require(right_prov["hardware"] == calibration["hardware"], "hardware differs from threshold calibration")
        require(right_prov["backend"] == calibration["backend"], "backend differs from threshold calibration")
        require(right_prov["dtype"] == calibration["dtype"], "dtype differs from threshold calibration")
        require(right_parameters.get("context") == calibration["context"], "context differs from threshold calibration")
        require(right_parameters.get("prefill_chunk") == calibration["prefill_chunk"], "prefill chunk differs from threshold calibration")
    failures: list[str] = []
    cases: list[dict[str, Any]] = []
    all_positions: list[dict[str, Any]] = []
    teacher_positions: list[dict[str, Any]] = []
    low_margin_positions: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    for case_id in selected_ids:
        left_response, right_response = left.responses[case_id], right.responses[case_id]
        left_prompt = left_response.get("canonical_prompt_token_ids", left_response.get("prompt_token_ids"))
        right_prompt = right_response.get("canonical_prompt_token_ids", right_response.get("prompt_token_ids"))
        require(left_prompt == right_prompt, f"{case_id}: canonical prompt tokens differ")
        left_teacher, right_teacher = _teacher_tokens(left_response), _teacher_tokens(right_response)
        require(left_teacher == right_teacher, f"{case_id}: canonical teacher-forced tokens differ")
        rendering = _rendering_metrics(left, right, case_id)
        if rendering["status"] == "verified" and (not rendering["bytes_equal"] or not rendering["token_ids_equal"]):
            failures.append(f"{case_id}: independently rendered prompt differs")
            if first_divergence is None:
                first_divergence = {
                    "case_id": case_id, "pass": "rendering", "position": None,
                    "metric": "rendered_bytes" if not rendering["bytes_equal"] else "prompt_token_ids",
                    "left": left_response.get("native_rendered_bytes_hex") if not rendering["bytes_equal"] else left_response.get("native_prompt_token_ids"),
                    "right": right_response.get("native_rendered_bytes_hex") if not rendering["bytes_equal"] else right_response.get("native_prompt_token_ids"),
                }
                first_divergence["position"] = (
                    rendering["first_byte_difference"] if not rendering["bytes_equal"]
                    else rendering["first_token_difference"]
                )
        left_greedy = [int(value) for value in left_response["greedy_token_ids"]]
        right_greedy = [int(value) for value in right_response["greedy_token_ids"]]
        left_greedy_bytes = _decode_hex_bytes(left_response.get("greedy_bytes_hex"), f"{case_id}: left greedy bytes")
        right_greedy_bytes = _decode_hex_bytes(right_response.get("greedy_bytes_hex"), f"{case_id}: right greedy bytes")
        if suite == "short":
            require(len(left_greedy) >= short_gates["minimum_greedy_tokens"], f"{case_id}: left greedy continuation is incomplete")
            require(len(right_greedy) >= short_gates["minimum_greedy_tokens"], f"{case_id}: right greedy continuation is incomplete")
            require(len(left_teacher) >= short_gates["minimum_teacher_forced_tokens"], f"{case_id}: left teacher-forced continuation is incomplete")
            require(len(right_teacher) >= short_gates["minimum_teacher_forced_tokens"], f"{case_id}: right teacher-forced continuation is incomplete")
        if left_greedy != right_greedy:
            position = longest_common_prefix(left_greedy, right_greedy)
            failures.append(f"{case_id}: greedy token differs at position {position}")
            if first_divergence is None:
                first_divergence = {
                    "case_id": case_id, "pass": "greedy", "position": position,
                    "metric": "token_id",
                    "left": left_greedy[position] if position < len(left_greedy) else None,
                    "right": right_greedy[position] if position < len(right_greedy) else None,
                }
        if left_greedy_bytes != right_greedy_bytes:
            failures.append(f"{case_id}: greedy decoded bytes differ")
            if first_divergence is None:
                first_divergence = {
                    "case_id": case_id, "pass": "greedy",
                    "position": first_sequence_difference(left_greedy_bytes, right_greedy_bytes),
                    "metric": "decoded_bytes", "left": left_greedy_bytes.hex(), "right": right_greedy_bytes.hex(),
                }
        case_positions: list[dict[str, Any]] = []
        for pass_name in ("greedy", "teacher_forced"):
            left_logits = _map_logits(left, left_response, pass_name)
            right_logits = _map_logits(right, right_response, pass_name)
            require(left_logits.shape == right_logits.shape, f"{case_id}: {pass_name} shape mismatch")
            require(left_logits.shape[0] == len(left_teacher), f"{case_id}: {pass_name} position count mismatch")
            for position in range(left_logits.shape[0]):
                metric = position_metrics(left_logits[position], right_logits[position], left_teacher[position], top_k)
                metric.update({"pass": pass_name, "position": position})
                case_positions.append(metric)
                all_positions.append(metric)
                if pass_name == "teacher_forced":
                    teacher_positions.append(metric)
                if metric["left_nan"] or metric["right_nan"] or metric["left_inf"] or metric["right_inf"]:
                    failures.append(f"{case_id}:{pass_name}:{position}: non-finite logits")
                    if first_divergence is None:
                        first_divergence = {
                            "case_id": case_id, "pass": pass_name, "position": position,
                            "metric": "nonfinite_logits",
                            "left": {"nan": metric["left_nan"], "inf": metric["left_inf"]},
                            "right": {"nan": metric["right_nan"], "inf": metric["right_inf"]},
                        }
                elif mode == "ds4-vs-ds4" and metric["different_float_count"] != 0:
                    failures.append(f"{case_id}:{pass_name}:{position}: {metric['different_float_count']} float values differ")
                    if first_divergence is None:
                        first_divergence = {
                            "case_id": case_id, "pass": pass_name, "position": position,
                            "metric": "different_float_count", "left": 0,
                            "right": metric["different_float_count"],
                        }
                elif mode == "ds4-vs-llama" and profile["status"] == "verified":
                    for name in thresholds:
                        reason = _threshold_failure(name, float(metric[name]), thresholds)
                        if reason:
                            failures.append(f"{case_id}:{pass_name}:{position}: {reason}")
                if suite == "short" and metric.get("metrics_available"):
                    if min(float(metric["left_margin"]), float(metric["right_margin"])) <= float(short_gates["low_margin_max"]):
                        low_margin_positions.append({
                            "case_id": case_id, "pass": pass_name, "position": position,
                            "left_margin": metric["left_margin"], "right_margin": metric["right_margin"],
                        })
                if first_divergence is None and metric.get("metrics_available") and not metric["greedy_agreement"]:
                    first_divergence = {
                        "case_id": case_id, "pass": pass_name, "position": position,
                        "metric": "argmax", "left": metric["left_argmax"], "right": metric["right_argmax"],
                    }
        case_teacher = [item for item in case_positions if item["pass"] == "teacher_forced"]
        cases.append({
            "id": case_id,
            "rendering": rendering,
            "greedy_longest_common_prefix": longest_common_prefix(left_greedy, right_greedy),
            "greedy_lengths": [len(left_greedy), len(right_greedy)],
            "greedy_token_ids_equal": left_greedy == right_greedy,
            "greedy_decoded_bytes_equal": left_greedy_bytes == right_greedy_bytes,
            "teacher_forced_diagnostics": _teacher_diagnostics(case_teacher),
            "positions": case_positions,
        })

    finite_positions = [item for item in all_positions if item.get("metrics_available")]
    aggregate: dict[str, Any] = {
        "cases": len(cases), "positions": len(all_positions),
        "nonfinite_positions": len(all_positions) - len(finite_positions),
        "first_divergence": first_divergence,
    }
    mean_fields = ("mae", "rmse", "cosine_similarity", "kl_left_right", "js_divergence", "top_k_overlap", "top_k_spearman", "top_k_rank_agreement", "oracle_logprob_abs_error")
    for field in mean_fields:
        values = [float(item[field]) for item in finite_positions]
        aggregate[f"mean_{field}"] = float(sum(values) / len(values)) if values else None
    aggregate["max_error"] = max((float(item["max_error"]) for item in finite_positions), default=None)
    aggregate["different_float_count"] = sum(int(item["different_float_count"]) for item in finite_positions)
    aggregate["greedy_agreement_rate"] = (sum(bool(item["greedy_agreement"]) for item in finite_positions) / len(finite_positions)) if finite_positions else None
    aggregate["teacher_forced"] = _teacher_diagnostics(teacher_positions)
    aggregate["low_margin_positions"] = low_margin_positions
    if suite == "short":
        # Individual tail ranks are highly sensitive to harmless reduction
        # order even when every greedy token agrees.  The calibrated short
        # profile therefore gates aggregate top-k quality, while token IDs,
        # decoded bytes and non-finite values remain strict per-position gates.
        aggregate_checks = (
            ("mean_top_k_overlap", aggregate["mean_top_k_overlap"], short_gates["top_k_overlap_min"]),
            ("mean_top_k_rank_agreement", aggregate["mean_top_k_rank_agreement"], short_gates["top_k_rank_agreement_min"]),
        )
        for name, value, limit in aggregate_checks:
            if value is None or fixed_gate_failed(float(value), float(limit), minimum=True):
                failures.append(f"short suite: {name}={value} fixed_threshold={limit}")
                if first_divergence is None:
                    first_divergence = {
                        "case_id": None, "pass": None, "position": None,
                        "metric": name, "left": value, "right": limit,
                    }
        oracle_mae = aggregate["teacher_forced"]["oracle_logprob_mae"]
        if oracle_mae is None or fixed_gate_failed(
            oracle_mae, float(short_gates["oracle_logprob_mae_max"]), minimum=False,
        ):
            failures.append(
                f"short suite: oracle_logprob_mae={oracle_mae} "
                f"fixed_threshold={short_gates['oracle_logprob_mae_max']}"
            )
            if first_divergence is None:
                first_divergence = {
                    "case_id": None, "pass": "teacher_forced", "position": None,
                    "metric": "oracle_logprob_mae", "left": oracle_mae,
                    "right": short_gates["oracle_logprob_mae_max"],
                }
        aggregate["first_divergence"] = first_divergence

    reviewed = left_prov["review_status"] == right_prov["review_status"] == "reviewed"
    rendering_verified = all(case["rendering"]["status"] == "verified" for case in cases)
    calibrated = profile["status"] == "verified"
    if failures:
        status, exit_code = "FAIL", 1
    elif diagnostic or not reviewed or not calibrated or not rendering_verified:
        status, exit_code = "NOT_VERIFIED", 3
    else:
        status, exit_code = "PASS", 0
    report = {
        "format": REPORT_FORMAT,
        "status": status,
        "mode": mode,
        "suite": suite,
        "diagnostic": diagnostic,
        "left": left_prov,
        "right": right_prov,
        "threshold_profile": profile,
        "short_message_gates": short_gates if suite == "short" else None,
        "failures": failures,
        "aggregate": aggregate,
        "cases": cases,
    }
    return report, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("ds4-vs-llama", "ds4-vs-ds4"))
    parser.add_argument("--left-run", required=True, type=Path, help="reference run (llama.cpp for cross-engine)")
    parser.add_argument("--right-run", required=True, type=Path, help="candidate run")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--suite", choices=("all", "short"), default="all")
    parser.add_argument("--case", action="append", dest="selected_cases",
                        help="compare only this case; may be repeated")
    parser.add_argument("--diagnostic", action="store_true", help="allow unreviewed or nonstandard engine roles; result cannot PASS")
    args = parser.parse_args(argv)
    try:
        require(args.top_k > 0, "--top-k must be positive")
        manifest, corpus = validate_generation_manifest(args.manifest)
        select_generation_formats(manifest)
        report, exit_code = compare_runs(
            load_run(args.left_run), load_run(args.right_run), manifest,
            args.mode, args.top_k, args.diagnostic, suite=args.suite, corpus=corpus,
            selected_cases=args.selected_cases,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, report)
        print(json.dumps({"status": report["status"], "report": str(args.report.resolve())}, sort_keys=True))
        return exit_code
    except (FixtureError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        error = {"format": REPORT_FORMAT, "status": "ERROR", "reason": str(exc)}
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report, error)
        except OSError:
            pass
        print(json.dumps(error, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
