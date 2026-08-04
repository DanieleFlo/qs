#!/usr/bin/env python3
"""Compare audited full-vocabulary Qwen3.6 runs.

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

from qwen36_fixtures import FixtureError, inventory_files, load_json, validate_manifest, write_json

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by deployment hosts
    raise SystemExit("numpy is required to compare full-vocabulary logits") from exc


REPORT_FORMAT = "ds4-qwen36-equivalence-report-v1"
RUN_FORMAT = "ds4-qwen36-oracle-v1"


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


def _rendering_metrics(left: RunData, right: RunData, case_id: str) -> dict[str, Any]:
    left_bytes = (left.root / "prompts" / f"{case_id}.bytes").read_bytes()
    right_bytes = (right.root / "prompts" / f"{case_id}.bytes").read_bytes()
    left_tokens = _load_tokens(left.root / "prompts" / f"{case_id}.tokens.json")
    right_tokens = _load_tokens(right.root / "prompts" / f"{case_id}.tokens.json")
    left_status = left.responses[case_id].get("native_rendering_status", "verified")
    right_status = right.responses[case_id].get("native_rendering_status", "verified")
    return {
        "status": "verified" if left_status == right_status == "verified" else "not_verified",
        "bytes_equal": left_bytes == right_bytes,
        "token_ids_equal": left_tokens == right_tokens,
        "left_bytes": len(left_bytes),
        "right_bytes": len(right_bytes),
        "left_tokens": len(left_tokens),
        "right_tokens": len(right_tokens),
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


def compare_runs(left: RunData, right: RunData, manifest: dict[str, Any], mode: str, top_k: int, diagnostic: bool) -> tuple[dict[str, Any], int]:
    left_ids = list(left.responses)
    right_ids = list(right.responses)
    require(left_ids == right_ids, "run case sets or order differ")
    left_prov, right_prov = _run_provenance(left), _run_provenance(right)
    require(left_prov["model"] == right_prov["model"] == manifest["model"]["id"], "manifest model mismatch")
    require(left_prov["artifacts"] == right_prov["artifacts"], "model artifact provenance differs")
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
    if mode == "ds4-vs-llama" and profile["status"] == "verified":
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
    first_divergence: dict[str, Any] | None = None
    for case_id in left_ids:
        left_response, right_response = left.responses[case_id], right.responses[case_id]
        left_prompt = left_response.get("canonical_prompt_token_ids", left_response.get("prompt_token_ids"))
        right_prompt = right_response.get("canonical_prompt_token_ids", right_response.get("prompt_token_ids"))
        require(left_prompt == right_prompt, f"{case_id}: canonical prompt tokens differ")
        left_teacher, right_teacher = _teacher_tokens(left_response), _teacher_tokens(right_response)
        require(left_teacher == right_teacher, f"{case_id}: canonical teacher-forced tokens differ")
        rendering = _rendering_metrics(left, right, case_id)
        if rendering["status"] == "verified" and (not rendering["bytes_equal"] or not rendering["token_ids_equal"]):
            failures.append(f"{case_id}: independently rendered prompt differs")
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
                if metric["left_nan"] or metric["right_nan"] or metric["left_inf"] or metric["right_inf"]:
                    failures.append(f"{case_id}:{pass_name}:{position}: non-finite logits")
                elif mode == "ds4-vs-ds4" and metric["different_float_count"] != 0:
                    failures.append(f"{case_id}:{pass_name}:{position}: {metric['different_float_count']} float values differ")
                elif mode == "ds4-vs-llama" and profile["status"] == "verified":
                    for name in thresholds:
                        reason = _threshold_failure(name, float(metric[name]), thresholds)
                        if reason:
                            failures.append(f"{case_id}:{pass_name}:{position}: {reason}")
                if first_divergence is None and metric.get("metrics_available") and (not metric["greedy_agreement"] or metric["different_float_count"]):
                    first_divergence = {"case_id": case_id, "pass": pass_name, "position": position}
        left_greedy = [int(value) for value in left_response["greedy_token_ids"]]
        right_greedy = [int(value) for value in right_response["greedy_token_ids"]]
        cases.append({
            "id": case_id,
            "rendering": rendering,
            "greedy_longest_common_prefix": longest_common_prefix(left_greedy, right_greedy),
            "greedy_lengths": [len(left_greedy), len(right_greedy)],
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
        "diagnostic": diagnostic,
        "left": left_prov,
        "right": right_prov,
        "threshold_profile": profile,
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
    parser.add_argument("--diagnostic", action="store_true", help="allow unreviewed or nonstandard engine roles; result cannot PASS")
    args = parser.parse_args(argv)
    try:
        require(args.top_k > 0, "--top-k must be positive")
        manifest, _ = validate_manifest(args.manifest)
        report, exit_code = compare_runs(load_run(args.left_run), load_run(args.right_run), manifest, args.mode, args.top_k, args.diagnostic)
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
