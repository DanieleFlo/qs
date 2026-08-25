#!/usr/bin/env python3
"""Performance experiment harness for DS4.

The core remains dependency-free. JSONSchemaBench output validation lazily
requires the independent ``jsonschema`` package.
"""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOADS = ROOT / "performance" / "workloads.yaml"
DEFAULT_CONSTRAINED_WORKLOADS = ROOT / "performance" / "constrained-workloads.json"
DEFAULT_JSONSCHEMABENCH_SUBSET = (
    ROOT / "performance" / "jsonschemabench-subset.json"
)
DEFAULT_RESULTS = ROOT / "performance-results"
SCHEMA_VERSION = 1
CONTEXT_CURVE_SUITE = "context-curve-full"
MTP_CONTEXT_CURVE_SUITE = "mtp-context-curve"
MTP_SHORT_REGRESSION_SUITE = "mtp-short-regression"
MTP_DEPTH_CROSSOVER_SUITE = "mtp-depth-crossover"
MTP_DEPTH_2K_SUITE = "mtp-depth-2k"
MTP_DEPTH_BOUNDARY_SUITE = "mtp-depth-boundary"
MTP_WEAKEST_CONFIRM_SUITE = "mtp-weakest-confirm"
MTP_LONG_CONTEXT_SMOKE_SUITE = "mtp-long-context-smoke"
MTP_DEPTH_28K_SUITE = "mtp-depth-28k"
MTP_THRESHOLD_SEARCH_SUITE = "mtp-threshold-search"
MTP_THRESHOLD_MIDPOINT_SUITE = "mtp-threshold-midpoint"
AGENT_DSML_BASELINE_SUITE = "agent-dsml-unconstrained-baseline"
QWEN_SPLIT_K_MIN_CONTEXT = 96
QWEN_MTP_DEPTH1_MIN_CONTEXT = 2048
CONTEXT_CURVE_MIN_TPS = 20.0
CONTEXT_CURVE_TARGET_CEILING_TPS = 30.0
CONTEXT_CURVE_RECOVERY_TOLERANCE_TPS = 1.5
CONTEXT_CURVE_RECOVERY_TOLERANCE_FRACTION = 0.08
QUANT_BYTES = {
    "F32": (4, 1), "F16": (2, 1),
    "Q4_K": (144, 256), "Q5_K": (176, 256), "Q6_K": (210, 256),
    "Q8_0": (34, 32),
}
KNOWN_GPU_SPECS = {
    # Static architectural limits complement driver measurements. Keep the
    # source explicit in the emitted profile rather than presenting them as
    # values measured at runtime.
    "NVIDIA GeForce RTX 3090": {
        "sm_count": 82,
        "memory_bus_bits": 384,
        "theoretical_memory_bandwidth_gbps": 936.2,
        "nominal_fp32_compute_tflops": 35.58,
        "l2_cache_bytes": 6 * 1024 * 1024,
        "warp_size": 32,
        "max_threads_per_block": 1024,
        "max_threads_per_sm": 1536,
        "max_blocks_per_sm": 16,
        "registers_per_sm_32bit": 65536,
        "registers_per_block_32bit": 65536,
        "shared_memory_per_sm_bytes": 100 * 1024,
        "shared_memory_per_block_default_bytes": 48 * 1024,
        "shared_memory_per_block_optin_bytes": 99 * 1024,
        "spec_source": "NVIDIA Ampere tuning guide and RTX 3090 product specification",
    },
}
LAYER_PROFILE_RE = re.compile(
    r"QWEN_(?P<phase>PREFILL|DECODE)_LAYER_PROFILE "
    r"pos=(?P<position>\d+) (?:rows=(?P<rows>\d+) )?"
    r"layer=(?P<layer>\d+) kind=(?P<kind>\w+) "
    r"attn=(?P<attn>[0-9.]+)ms ffn=(?P<ffn>[0-9.]+)ms "
    r"total=(?P<total>[0-9.]+)ms"
)
DECODE_PROFILE_RE = re.compile(
    r"QWEN_DECODE_PROFILE pos=(?P<position>\d+) "
    r"embed=(?P<embed>[0-9.]+)ms "
    r"recurrent_attn=(?P<recurrent_attn>[0-9.]+)ms "
    r"full_attn=(?P<full_attn>[0-9.]+)ms "
    r"\[qkv=(?P<full_qkv>[0-9.]+) core=(?P<full_core>[0-9.]+) "
    r"out=(?P<full_out>[0-9.]+)\] "
    r"ffn=(?P<ffn>[0-9.]+)ms "
    r"output=(?P<output>[0-9.]+)ms "
    r"read=(?P<read>[0-9.]+)ms "
    r"total=(?P<total>[0-9.]+)ms"
)
SERVER_PROGRESS_RE = re.compile(
    r"ds4-server: (?P<kind>chat|completion) ctx=(?P<context>\S+) "
    r"gen=(?P<generation>\d+)[^\r\n]*?decoding chunk="
    r"(?P<chunk_tps>[0-9.]+) t/s avg=(?P<avg_tps>[0-9.]+) t/s"
)
SERVER_PHASE_PREFIX = "ds4-server: phase profile "
SERVER_PHASE_FIELD_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>\S+)")


class HarnessError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise HarnessError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"{label} must contain a JSON object: {path}")
    return value


def inspect_model_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HarnessError(f"model does not exist: {path}")
    quality_root = ROOT / "gguf-tools" / "quality-testing"
    module_path = quality_root / "inspect_qwen36_gguf.py"
    if str(quality_root) not in sys.path:
        sys.path.insert(0, str(quality_root))
    spec = importlib.util.spec_from_file_location("ds4_qwen_gguf_inspector", module_path)
    if not spec or not spec.loader:
        raise HarnessError(f"cannot load GGUF inspector: {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.inspect_gguf(path, include_artifact_sha256=False)
    except Exception as exc:
        raise HarnessError(f"cannot inspect GGUF model {path}: {exc}") from exc


def run_command(command: list[str], *, check: bool = True,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        tail = "\n".join(result.stderr.splitlines()[-30:])
        raise HarnessError(f"command failed ({result.returncode}): {' '.join(command)}\n{tail}")
    return result


def apply_env_overrides(base: dict[str, str],
                        overrides: list[str]) -> dict[str, str]:
    env = base.copy()
    for item in overrides:
        if "=" not in item:
            raise HarnessError(f"invalid --env {item!r}; expected NAME=VALUE")
        key, value = item.split("=", 1)
        if not key:
            raise HarnessError("--env variable name cannot be empty")
        env[key] = value
    return env


def executable(name: str) -> str | None:
    found = shutil.which(name) or shutil.which(name + ".exe")
    if found:
        return found
    candidates = []
    cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_root:
        candidates.append(Path(cuda_root) / "bin" / name)
    if platform.system() == "Linux":
        candidates.append(Path("/usr/local/cuda/bin") / name)
    return str(next((path for path in candidates if path.is_file()), "")) or None


def parse_number(value: str) -> float | int | str:
    raw = value.strip()
    try:
        number = float(raw)
        return int(number) if number.is_integer() else number
    except ValueError:
        return raw


def compute_capability_at_least(value: Any, minimum: tuple[int, int]) -> bool:
    match = re.fullmatch(r"(\d+)\.(\d+)", str(value).strip())
    return bool(match and (int(match.group(1)), int(match.group(2))) >= minimum)


def nvidia_query() -> dict[str, Any]:
    smi = executable("nvidia-smi")
    if not smi:
        return {"available": False, "reason": "nvidia-smi not found"}
    fields = [
        "name", "driver_version", "memory.total", "memory.free", "memory.used",
        "temperature.gpu", "power.limit", "power.draw", "clocks.current.sm",
        "clocks.current.memory", "clocks.max.sm", "clocks.max.memory",
        "pstate", "compute_cap", "clocks_event_reasons.active",
        "clocks_event_reasons.sw_power_cap",
        "clocks_event_reasons.hw_thermal_slowdown",
    ]
    result = run_command(
        [smi, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        check=False,
    )
    if result.returncode:
        return {"available": False, "reason": result.stderr.strip() or "nvidia-smi failed"}
    architectures: list[str | None] = []
    reported_cuda_version = None
    xml_result = run_command([smi, "-q", "-x"], check=False)
    if xml_result.returncode == 0:
        try:
            root = ET.fromstring(xml_result.stdout)
            reported_cuda_version = root.findtext("cuda_version")
            architectures = [gpu.findtext("product_architecture") for gpu in root.findall("gpu")]
        except ET.ParseError:
            architectures = []
    devices = []
    for index, row in enumerate(csv.reader(result.stdout.splitlines())):
        if len(row) == len(fields):
            device = {"index": index, **{
                key: parse_number(value) for key, value in zip(fields, row)
            }}
            device["architecture"] = (
                architectures[index] if index < len(architectures) else None
            )
            compute_cap = device.get("compute_cap")
            device["numeric_support"] = {
                "fp32": True,
                "fp16": compute_capability_at_least(compute_cap, (5, 3)),
                "bf16": compute_capability_at_least(compute_cap, (8, 0)),
                "tf32": compute_capability_at_least(compute_cap, (8, 0)),
                "fp8": compute_capability_at_least(compute_cap, (8, 9)),
                "int8": compute_capability_at_least(compute_cap, (6, 1)),
                "int4": compute_capability_at_least(compute_cap, (7, 5)),
                "tensor_cores": compute_capability_at_least(compute_cap, (7, 0)),
            }
            static_spec = KNOWN_GPU_SPECS.get(str(device.get("name")))
            device["static_spec"] = static_spec or {
                "status": "NOT_VERIFIED",
                "reason": "GPU model is not in the harness static specification table",
            }
            devices.append(device)
    return {"available": bool(devices), "devices": devices, "command": smi,
            "driver_reported_cuda_version": reported_cuda_version}


def package_version(name: str) -> dict[str, Any]:
    try:
        import importlib.metadata
        return {"available": True, "version": importlib.metadata.version(name)}
    except importlib.metadata.PackageNotFoundError:
        return {"available": False}


def tool_version(name: str, arguments: list[str]) -> dict[str, Any]:
    path = executable(name)
    if not path:
        return {"available": False}
    result = run_command([path, *arguments], check=False)
    output = (result.stdout or result.stderr).strip()
    return {
        "available": result.returncode == 0,
        "path": path,
        "version": output.splitlines()[0] if output else None,
        "version_output": output or None,
    }


def hardware_probe() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": utc_now(),
        "host": {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "processor": platform.processor(),
        },
        "gpu": nvidia_query(),
        "tools": {
            "nvcc": tool_version("nvcc", ["--version"]),
            "nsys": tool_version("nsys", ["--version"]),
            "ncu": tool_version("ncu", ["--version"]),
            "pytorch": package_version("torch"),
            "triton": package_version("triton"),
        },
    }


def hardware_profile() -> dict[str, Any]:
    """Backward-compatible internal name used by existing experiment records."""
    return hardware_probe()


def load_workloads(path: Path, suite: str) -> list[dict[str, Any]]:
    # Strict JSON is valid YAML and avoids a runtime PyYAML dependency.
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read workload file {path}: {exc}") from exc
    if document.get("schema_version") != SCHEMA_VERSION:
        raise HarnessError(f"unsupported workload schema in {path}")
    names = document.get("suites", {}).get(suite)
    if not names:
        raise HarnessError(f"unknown or empty suite {suite!r}")
    definitions = document.get("workloads", {})
    try:
        return [dict(definitions[name], id=name) for name in names]
    except KeyError as exc:
        raise HarnessError(f"suite {suite!r} references missing workload {exc}") from exc


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summary(values: list[float]) -> dict[str, float | int | bool]:
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": len(values), "min": min(values), "p10": percentile(values, .10),
        "median": statistics.median(values), "mean": mean,
        "p90": percentile(values, .90), "max": max(values), "stdev": stdev,
        "coefficient_of_variation": stdev / mean if mean else 0.0,
        "unstable": bool(mean and stdev / mean > 0.05),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_sha256(path: Path, cache_root: Path) -> str:
    """Avoid rereading a multi-GiB GGUF on every kernel iteration."""
    stat = path.stat()
    key = json.dumps({
        "path": str(path.resolve()), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }, sort_keys=True)
    cache_path = cache_root / ".hash-cache.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cache = {}
    if key not in cache:
        cache[key] = sha256(path)
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
        temporary.replace(cache_path)
    return cache[key]


def git_value(*args: str) -> str | None:
    result = run_command(["git", *args], check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def bench_once(binary: Path, model: Path, prompt: Path,
               workload: dict[str, Any], env: dict[str, str],
               output: Path,
               dump_logits_dir: Path | None = None,
               repetitions: int = 1) -> list[dict[str, Any]]:
    context_max = int(workload.get("context_max", workload["context"]))
    command = [
        str(binary), "--model", str(model), "--prompt-file", str(prompt),
        "--ctx-start", str(workload["context"]),
        "--ctx-max", str(context_max),
        "--ctx-alloc", str(context_max + workload["generation_tokens"] + 1),
        "--gen-tokens", str(workload["generation_tokens"]),
        "--repetitions", str(repetitions),
        "--csv", str(output), "--backend", workload.get("backend", "cuda"),
    ]
    if workload.get("step_incr"):
        command += ["--step-incr", str(workload["step_incr"])]
    if workload.get("prefill_chunk"):
        command += ["--prefill-chunk", str(workload["prefill_chunk"])]
    if dump_logits_dir:
        dump_logits_dir.mkdir(parents=True, exist_ok=True)
        command += ["--dump-frontier-logits-dir", str(dump_logits_dir)]
    run_command(command, env=env)
    with output.open(newline="", encoding="utf-8") as stream:
        return [
            {key: parse_number(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def aggregate_runs(runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for run in runs for row in run]
    metrics = {}
    for name in ("prefill_tps", "gen_tps", "gen_first_ms",
                 "gen_steady_tps", "kvcache_bytes"):
        values = [float(row[name]) for row in rows if name in row]
        if values:
            metrics[name] = summary(values)
    return {"metrics": metrics, "raw_rows": rows}


def analyze_context_curve(
        workloads: list[dict[str, Any]], *,
        minimum_tps: float = CONTEXT_CURVE_MIN_TPS,
        target_ceiling_tps: float = CONTEXT_CURVE_TARGET_CEILING_TPS,
        recovery_tolerance_tps: float = CONTEXT_CURVE_RECOVERY_TOLERANCE_TPS,
        recovery_tolerance_fraction: float = CONTEXT_CURVE_RECOVERY_TOLERANCE_FRACTION,
) -> dict[str, Any]:
    """Gate a decode curve without penalizing throughput above its target band."""
    points = []
    missing = []
    for workload in workloads:
        if workload.get("status") != "measured":
            missing.append(workload.get("id", "unknown"))
            continue
        context = workload.get("definition", {}).get("context")
        tps = metric_median(workload, "gen_steady_tps")
        if context is None or tps is None:
            missing.append(workload.get("id", "unknown"))
            continue
        points.append({
            "workload": workload["id"], "context": int(context), "tps": tps,
        })
    points.sort(key=lambda point: point["context"])
    below_floor = [point for point in points if point["tps"] < minimum_tps]
    above_target_ceiling = [
        point for point in points if point["tps"] > target_ceiling_tps
    ]
    material_recoveries = []
    for left, right in zip(points, points[1:]):
        increase = right["tps"] - left["tps"]
        tolerance = max(
            recovery_tolerance_tps,
            left["tps"] * recovery_tolerance_fraction,
        )
        # The upper band is intentionally advisory: small dispatch/clock
        # recoveries are irrelevant while both adjacent points already exceed
        # the target ceiling.  Valleys that touch the measured band still fail.
        both_above_target = (
            left["tps"] > target_ceiling_tps and
            right["tps"] > target_ceiling_tps
        )
        if increase > tolerance and not both_above_target:
            material_recoveries.append({
                "from_context": left["context"],
                "to_context": right["context"],
                "from_tps": left["tps"], "to_tps": right["tps"],
                "increase_tps": increase, "tolerance_tps": tolerance,
            })
    status = "PASS"
    if missing or below_floor or material_recoveries:
        status = "FAIL"
    return {
        "status": status,
        "minimum_tps": minimum_tps,
        "target_ceiling_tps": target_ceiling_tps,
        "ceiling_policy": "advisory_only; faster points are never penalized",
        "recovery_tolerance_tps": recovery_tolerance_tps,
        "recovery_tolerance_fraction": recovery_tolerance_fraction,
        "points": points,
        "observed_min_tps": min((point["tps"] for point in points), default=None),
        "observed_max_tps": max((point["tps"] for point in points), default=None),
        "missing_workloads": missing,
        "below_floor": below_floor,
        "above_target_ceiling": above_target_ceiling,
        "material_recoveries": material_recoveries,
    }


def parse_server_progress(text: str) -> list[dict[str, Any]]:
    return [
        {
            "kind": match.group("kind"),
            "context_span": match.group("context"),
            "generation_tokens": int(match.group("generation")),
            "chunk_tps": float(match.group("chunk_tps")),
            "avg_tps": float(match.group("avg_tps")),
        }
        for match in SERVER_PROGRESS_RE.finditer(text)
    ]


def _phase_float(value: str, suffix: str = "") -> float:
    if suffix and not value.endswith(suffix):
        raise HarnessError(f"phase value {value!r} lacks suffix {suffix!r}")
    return float(value[:-len(suffix)] if suffix else value)


def _phase_head(value: str, suffix: str = "ms") -> float:
    return _phase_float(value.split("/", 1)[0], suffix)


def _phase_int(value: str) -> int:
    return int(value.split("_", 1)[0])


def parse_server_phase_profiles(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    float_ms = (
        "wall", "forced_prefix_probe", "sampling_mask_build",
        "constraint_cpu", "constraint_cpu_exposed",
        "constraint_cpu_overlapped", "filter_setup", "filter", "residual",
        "grammar_compile_ms", "grammar_jit_ms",
        "trie_compile_ms", "static_mask_compile_ms",
    )
    integer_fields = (
        "vocab", "filter_calls", "accepted", "finite_allowed", "piece_bytes",
        "candidate_tokens_tested", "parser_transition_count",
        "parser_bytes_visited", "trie_nodes_visited", "subtrees_pruned",
        "trie_leaf_tokens_emitted", "trie_compiled_nodes",
        "trie_memory_bytes", "static_mask_memory_bytes",
        "dynamic_frontier_size", "mask_cache_hit",
        "mask_cache_miss", "constraint_state_checkpoint",
        "constraint_state_rollback", "exhaustive_fallback_steps",
    )
    for line in text.splitlines():
        marker = line.find(SERVER_PHASE_PREFIX)
        if marker < 0:
            continue
        fields = {
            match.group("key"): match.group("value")
            for match in SERVER_PHASE_FIELD_RE.finditer(
                line[marker + len(SERVER_PHASE_PREFIX):]
            )
        }
        required = {"ctx", "gen", "mode", "wall", "constraint_cpu"}
        missing = required.difference(fields)
        if missing:
            raise HarnessError(
                "server phase profile lacks fields: " + ", ".join(sorted(missing))
            )
        row: dict[str, Any] = {
            "context_span": fields["ctx"],
            "generation_tokens": int(fields["gen"]),
            "mode": fields["mode"],
        }
        for name in float_ms:
            if name in fields:
                row[f"{name}_ms" if not name.endswith("_ms") else name] = (
                    _phase_float(
                        fields[name], "" if name.endswith("_ms") else "ms"
                    )
                )
        for name in ("forced_build", "forced_sync", "oracle_compare",
                     "filtered_sample", "plain_sample", "eval"):
            if name in fields:
                row[f"{name}_ms"] = _phase_head(fields[name])
        if "oracle_compare" in fields:
            oracle_parts = fields["oracle_compare"].split("/")
            if len(oracle_parts) >= 3:
                row["oracle_compare_calls"] = _phase_int(oracle_parts[1])
                row["oracle_divergences"] = _phase_int(oracle_parts[2])
        for name in integer_fields:
            if name in fields:
                row[name] = _phase_int(fields[name])
        rows.append(row)
    return rows


def constrained_source_witnesses_valid(results: list[dict[str, Any]]) -> bool:
    """Require imported JSONSchemaBench witnesses, not internal fixtures."""
    return all(
        item["definition"].get("source") is None or
        item["definition"]["source"].get("witness_valid") is True
        for item in results
        if item["definition"]["kind"] == "json_schema"
    )


def prompt_filler_intercept(
        first_count: int, first_tokens: int,
        second_count: int, second_tokens: int) -> int:
    if second_count <= first_count:
        raise HarnessError("prompt calibration counts must be increasing")
    if second_tokens - first_tokens != second_count - first_count:
        raise HarnessError(
            "server prompt filler is not one token per repetition; "
            f"counts {first_count}->{second_count} produced "
            f"{first_tokens}->{second_tokens} prompt tokens"
        )
    return first_tokens - first_count


def server_curve_prompt(filler_count: int,
                        pattern: str = "numbered-list") -> str:
    if filler_count < 0:
        raise HarnessError("target context is smaller than the calibrated prompt wrapper")
    if pattern == "technical-explanation":
        return (
            "Ignore all occurrences of alpha. Read only the final "
            "instruction.\nFILLER:"
            + " alpha" * filler_count
            + "\nFINAL INSTRUCTION: Write a detailed technical explanation "
              "of reliable GPU benchmarking. Cover warmup, synchronization, "
              "memory, latency, throughput, determinism, and correctness. "
              "Use complete paragraphs and continue for at least 64 words."
              "\nANSWER:"
        )
    if pattern != "numbered-list":
        raise HarnessError(f"unknown server prompt pattern: {pattern}")
    return (
        "Read the following calibration filler silently.\nFILLER:"
        + " alpha" * filler_count
        + "\nContinue with a long numbered list about reliable GPU benchmarking.\n1."
    )


def parse_layer_profiles(text: str) -> list[dict[str, Any]]:
    rows = []
    for match in LAYER_PROFILE_RE.finditer(text):
        row = match.groupdict()
        rows.append({
            "phase": row["phase"].lower(),
            "position": int(row["position"]),
            "rows": int(row["rows"]) if row["rows"] else 1,
            "layer": int(row["layer"]), "kind": row["kind"],
            "attention_ms": float(row["attn"]),
            "ffn_ms": float(row["ffn"]), "total_ms": float(row["total"]),
        })
    return rows


def parse_decode_profiles(text: str) -> list[dict[str, Any]]:
    profiles = []
    for match in DECODE_PROFILE_RE.finditer(text):
        row = match.groupdict()
        profiles.append({
            "position": int(row["position"]),
            **{
                f"{name}_ms": float(row[name])
                for name in (
                    "embed", "recurrent_attn", "full_attn", "full_qkv",
                    "full_core", "full_out", "ffn", "output", "read", "total",
                )
            },
        })
    return profiles


def network_profile_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise HarnessError("DS4 emitted no Qwen layer profile rows")
    # The last position excludes CUDA lazy setup and first-use weight
    # resolution from the steady component ranking when two chunks exist.
    selected_positions = {
        phase: max(row["position"] for row in rows if row["phase"] == phase)
        for phase in {row["phase"] for row in rows}
    }
    all_rows = rows
    rows = [
        row for row in rows
        if row["position"] == selected_positions[row["phase"]]
    ]
    phase_totals: dict[str, float] = {}
    stage_totals: dict[str, float] = {"attention": 0.0, "ffn": 0.0}
    operation_totals: dict[str, float] = {}
    for row in rows:
        phase_totals[row["phase"]] = phase_totals.get(row["phase"], 0.0) + row["total_ms"]
        stage_totals["attention"] += row["attention_ms"]
        stage_totals["ffn"] += row["ffn_ms"]
        operation_totals[f"{row['kind']}_attention"] = (
            operation_totals.get(f"{row['kind']}_attention", 0.0) +
            row["attention_ms"]
        )
        operation_totals[f"{row['kind']}_ffn"] = (
            operation_totals.get(f"{row['kind']}_ffn", 0.0) +
            row["ffn_ms"]
        )
    total = sum(stage_totals.values())
    hotspots = []
    for row in rows:
        for stage in ("attention", "ffn"):
            duration = row[f"{stage}_ms"]
            hotspots.append({
                "phase": row["phase"], "position": row["position"],
                "rows": row["rows"], "layer": row["layer"], "kind": row["kind"],
                "stage": stage, "duration_ms": duration,
                "percent_profiled_time": 100.0 * duration / total if total else 0.0,
            })
    hotspots.sort(key=lambda item: item["duration_ms"], reverse=True)
    cumulative = 0.0
    covering_95_percent = 0
    for item in hotspots:
        cumulative += item["duration_ms"]
        item["cumulative_percent"] = 100.0 * cumulative / total if total else 0.0
        if covering_95_percent == 0 and item["cumulative_percent"] >= 95.0:
            covering_95_percent = hotspots.index(item) + 1
    return {
        "format": "ds4-qwen-network-profile-v1",
        "status": "PROFILED", "profiled_time_ms": total,
        "selected_positions": selected_positions,
        "discarded_cold_rows": len(all_rows) - len(rows),
        "phase_time_ms": phase_totals,
        "stage_time_ms": stage_totals,
        "operation_time_ms": operation_totals,
        "stage_percent": {
            name: 100.0 * value / total if total else 0.0
            for name, value in stage_totals.items()
        },
        "hotspots_covering_95_percent": covering_95_percent,
        "hotspots": hotspots, "layers": rows,
    }


def finite_logits(path: Path) -> tuple[dict[str, Any], list[float]]:
    document = read_json(path, "logits file")
    raw = document.get("logits")
    if not isinstance(raw, list) or not raw:
        raise HarnessError(f"no logits array in {path}")
    if any(value is None or not math.isfinite(float(value)) for value in raw):
        raise HarnessError(f"non-finite logits in {path}")
    return document, [float(value) for value in raw]


def logits_drift(baseline_path: Path, candidate_path: Path,
                 top_k: int = 20) -> dict[str, Any]:
    baseline_meta, baseline = finite_logits(baseline_path)
    candidate_meta, candidate = finite_logits(candidate_path)
    if len(baseline) != len(candidate):
        raise HarnessError("baseline and candidate vocabularies differ")
    baseline_max, candidate_max = max(baseline), max(candidate)
    left = [value - baseline_max for value in baseline]
    right = [value - candidate_max for value in candidate]
    delta = [a - b for a, b in zip(left, right)]
    mae = statistics.fmean(abs(value) for value in delta)
    rmse = math.sqrt(statistics.fmean(value * value for value in delta))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    cosine = (sum(a * b for a, b in zip(left, right)) / (norm_left * norm_right)
              if norm_left and norm_right else 0.0)
    count = min(top_k, len(left))
    top_left = sorted(range(len(left)), key=left.__getitem__, reverse=True)[:count]
    top_right = sorted(range(len(right)), key=right.__getitem__, reverse=True)[:count]
    argmax_left, argmax_right = top_left[0], top_right[0]
    argmax_equal = argmax_left == argmax_right
    overlap = len(set(top_left) & set(top_right)) / count
    passed = argmax_equal and overlap >= 0.95 and cosine >= 0.999
    return {
        "format": "ds4-logits-drift-v1",
        "status": "PASS" if passed else "FAIL",
        "baseline": str(baseline_path), "candidate": str(candidate_path),
        "prompt_tokens": {
            "baseline": baseline_meta.get("prompt_tokens"),
            "candidate": candidate_meta.get("prompt_tokens"),
        },
        "values": len(left), "nonfinite": 0,
        "argmax": {
            "baseline": argmax_left, "candidate": argmax_right,
            "equal": argmax_equal,
        },
        "centered_mae": mae, "centered_rmse": rmse,
        "centered_max_error": max(abs(value) for value in delta),
        "cosine_similarity": cosine,
        "top_k": count,
        "top_k_overlap": overlap,
        "gates": {
            "argmax_equal": True, "minimum_top_k_overlap": 0.95,
            "minimum_cosine_similarity": 0.999,
        },
    }


def decode_result_drift(baseline_path: Path, candidate_path: Path,
                        top_k: int = 20) -> dict[str, Any]:
    baseline_meta = read_json(baseline_path, "baseline decode result")
    candidate_meta = read_json(candidate_path, "candidate decode result")
    baseline_tokens = baseline_meta.get("generated_tokens")
    candidate_tokens = candidate_meta.get("generated_tokens")
    if not isinstance(baseline_tokens, list) or not isinstance(candidate_tokens, list):
        raise HarnessError("decode result is missing generated_tokens")
    tokens_equal = baseline_tokens == candidate_tokens
    report = logits_drift(baseline_path, candidate_path, top_k)
    logits_passed = report["status"] == "PASS"
    report.update({
        "format": "ds4-decode-drift-v1",
        "status": "PASS" if tokens_equal and logits_passed else "FAIL",
        "generated_tokens": {
            "baseline_count": len(baseline_tokens),
            "candidate_count": len(candidate_tokens),
            "equal": tokens_equal,
            "first_difference": next((
                index for index, (left, right) in enumerate(
                    zip(baseline_tokens, candidate_tokens)
                ) if left != right
            ), (min(len(baseline_tokens), len(candidate_tokens))
                if len(baseline_tokens) != len(candidate_tokens) else None)),
        },
        "gates": {
            **report["gates"],
            "generated_tokens_equal": True,
        },
    })
    return report


def q8_1_parity(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Compare packed Q8_1 data, separating MMVQ inputs from ds.y metadata."""
    try:
        left = left_path.read_bytes()
        right = right_path.read_bytes()
    except OSError as exc:
        raise HarnessError(f"cannot read Q8_1 input: {exc}") from exc
    if len(left) != len(right):
        raise HarnessError("Q8_1 inputs have different sizes")
    if not left or len(left) % 36:
        raise HarnessError("Q8_1 inputs must contain complete 36-byte blocks")

    scale_differences = 0
    metadata_differences = 0
    quant_differences = 0
    blocks_with_quant_differences = 0
    qsum_differences = 0
    first_quant_difference = None
    for block in range(len(left) // 36):
        offset = block * 36
        left_block = left[offset:offset + 36]
        right_block = right[offset:offset + 36]
        scale_differences += left_block[:2] != right_block[:2]
        metadata_differences += left_block[2:4] != right_block[2:4]
        left_qs = struct.unpack("<32b", left_block[4:])
        right_qs = struct.unpack("<32b", right_block[4:])
        block_differences = sum(a != b for a, b in zip(left_qs, right_qs))
        if block_differences:
            blocks_with_quant_differences += 1
            quant_differences += block_differences
            if first_quant_difference is None:
                index = next(i for i, (a, b) in enumerate(zip(left_qs, right_qs))
                             if a != b)
                first_quant_difference = {
                    "block": block, "index": index,
                    "left": left_qs[index], "right": right_qs[index],
                }
        qsum_differences += sum(left_qs) != sum(right_qs)

    consumed_equal = scale_differences == 0 and quant_differences == 0
    return {
        "format": "ds4-q8-1-parity-v1",
        "status": "PASS" if consumed_equal else "FAIL",
        "left": str(left_path), "right": str(right_path),
        "bytes": len(left), "blocks": len(left) // 36,
        "sha256": {
            "left": hashlib.sha256(left).hexdigest(),
            "right": hashlib.sha256(right).hexdigest(),
        },
        "mmvq_consumed_fields": {
            "equal": consumed_equal,
            "scale_ds_x_differing_blocks": scale_differences,
            "qs_differing_blocks": blocks_with_quant_differences,
            "qs_differing_bytes": quant_differences,
            "qsum_differing_blocks": qsum_differences,
            "first_qs_difference": first_quant_difference,
        },
        "metadata_ds_y": {
            "equal": metadata_differences == 0,
            "differing_blocks": metadata_differences,
            "used_by_decode_mmvq": False,
        },
    }


def read_f32_row(path: Path, row: int, width: int) -> list[float]:
    if row < 0 or width <= 0:
        raise HarnessError("row must be non-negative and width must be positive")
    expected_row_bytes = width * 4
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise HarnessError(f"cannot inspect logits file {path}: {exc}") from exc
    if size % expected_row_bytes:
        raise HarnessError(f"logits file is not a multiple of {width} F32 values: {path}")
    if row >= size // expected_row_bytes:
        raise HarnessError(f"row {row} is outside {path}")
    values = array.array("f")
    try:
        with path.open("rb") as file:
            file.seek(row * expected_row_bytes)
            values.fromfile(file, width)
    except (OSError, EOFError) as exc:
        raise HarnessError(f"cannot read logits row from {path}: {exc}") from exc
    if sys.byteorder != "little":
        values.byteswap()
    result = values.tolist()
    if any(not math.isfinite(value) for value in result):
        raise HarnessError(f"non-finite value in logits row {row}: {path}")
    return result


def centered_pair_metrics(left: list[float], right: list[float],
                          top_k: int) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise HarnessError("logits rows have different widths")
    left_max, right_max = max(left), max(right)
    errors = [(a - left_max) - (b - right_max) for a, b in zip(left, right)]
    left_centered = [value - left_max for value in left]
    right_centered = [value - right_max for value in right]
    norm_left = math.sqrt(sum(value * value for value in left_centered))
    norm_right = math.sqrt(sum(value * value for value in right_centered))
    count = min(top_k, len(left))
    top_left = sorted(range(len(left)), key=left.__getitem__, reverse=True)[:count]
    top_right = sorted(range(len(right)), key=right.__getitem__, reverse=True)[:count]
    return {
        "centered_mae": statistics.fmean(abs(value) for value in errors),
        "centered_rmse": math.sqrt(statistics.fmean(value * value for value in errors)),
        "centered_max_error": max(abs(value) for value in errors),
        "cosine_similarity": (
            sum(a * b for a, b in zip(left_centered, right_centered)) /
            (norm_left * norm_right) if norm_left and norm_right else 0.0
        ),
        "argmax_equal": top_left[0] == top_right[0],
        "top_k": count,
        "top_k_overlap": len(set(top_left) & set(top_right)) / count,
    }


def qwen_logits_row_report(runs: list[tuple[str, Path]], case: str,
                           stream: str, row: int, vocab: int,
                           top_k: int, focus_tokens: list[int]) -> dict[str, Any]:
    if len(runs) < 2:
        raise HarnessError("at least two labeled Qwen runs are required")
    if len({label for label, _ in runs}) != len(runs):
        raise HarnessError("Qwen run labels must be unique")
    if any(token < 0 or token >= vocab for token in focus_tokens):
        raise HarnessError("focus token is outside the vocabulary")
    rows: dict[str, list[float]] = {}
    summaries = []
    for label, run in runs:
        path = run / "logits" / f"{case}.{stream}.f32"
        values = read_f32_row(path, row, vocab)
        rows[label] = values
        order = sorted(range(vocab), key=values.__getitem__, reverse=True)[:top_k]
        focus = {str(token): values[token] for token in focus_tokens}
        summary = {
            "label": label, "run": str(run), "logits": str(path),
            "argmax": order[0],
            "top": [{"token": token, "logit": values[token]} for token in order],
            "focus_logits": focus,
        }
        if len(focus_tokens) == 2:
            summary["focus_margin"] = values[focus_tokens[0]] - values[focus_tokens[1]]
        summaries.append(summary)
    reference = runs[0][0]
    comparisons = [
        {"reference": reference, "candidate": label,
         **centered_pair_metrics(rows[reference], rows[label], top_k)}
        for label, _ in runs[1:]
    ]
    return {
        "format": "ds4-qwen-logits-row-v1",
        "case": case, "stream": stream, "row": row, "vocab": vocab,
        "focus_tokens": focus_tokens,
        "runs": summaries, "comparisons": comparisons,
    }


def qwen_oracle_cases(root: Path, label: str) -> list[tuple[str, dict[str, Any]]]:
    """Load response objects in the stable case order recorded by an oracle run."""
    index = read_json(root / "index.json", f"{label} Qwen oracle index")
    cases = index.get("cases")
    if not isinstance(cases, list) or not cases:
        raise HarnessError(f"{label} Qwen oracle index has no cases: {root}")
    result = []
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise HarnessError(f"{label} Qwen oracle index has a malformed case")
        case_id, response_file = case.get("id"), case.get("response_file")
        if not isinstance(case_id, str) or not isinstance(response_file, str):
            raise HarnessError(f"{label} Qwen oracle case lacks id/response_file")
        if case_id in seen:
            raise HarnessError(f"{label} Qwen oracle repeats case {case_id!r}")
        seen.add(case_id)
        response = read_json(root / response_file,
                             f"{label} Qwen response for {case_id}")
        result.append((case_id, response))
    return result


def qwen_token_stream(response: dict[str, Any], stream: str,
                      label: str) -> list[int]:
    values = response.get("greedy_token_ids" if stream == "greedy"
                          else "teacher_forced")
    if not isinstance(values, list):
        raise HarnessError(f"{label} response lacks {stream} token rows")
    if stream == "teacher":
        if any(not isinstance(row, dict) or not isinstance(row.get("token_id"), int)
               for row in values):
            raise HarnessError(f"{label} response has malformed teacher token rows")
        return [row["token_id"] for row in values]
    if any(not isinstance(token, int) for token in values):
        raise HarnessError(f"{label} response has malformed greedy token rows")
    return values


def qwen_argmax_gate(reference_root: Path, candidate_root: Path) -> dict[str, Any]:
    """Check full-suite greedy sequences and greedy/teacher argmax token IDs."""
    reference = qwen_oracle_cases(reference_root, "reference")
    candidate = qwen_oracle_cases(candidate_root, "candidate")
    if [case for case, _ in reference] != [case for case, _ in candidate]:
        raise HarnessError("reference and candidate Qwen oracle case orders differ")

    case_reports = []
    greedy_equal = teacher_equal = greedy_total = teacher_total = 0
    sequences_equal = 0
    for (case_id, left), (_, right) in zip(reference, candidate):
        if left.get("canonical_prompt_token_ids") != right.get("canonical_prompt_token_ids"):
            raise HarnessError(f"canonical prompt tokens differ for case {case_id}")
        if left.get("teacher_forced_source") != right.get("teacher_forced_source"):
            raise HarnessError(f"teacher-forced sources differ for case {case_id}")
        left_greedy = qwen_token_stream(left, "greedy", f"reference {case_id}")
        right_greedy = qwen_token_stream(right, "greedy", f"candidate {case_id}")
        left_teacher = qwen_token_stream(left, "teacher", f"reference {case_id}")
        right_teacher = qwen_token_stream(right, "teacher", f"candidate {case_id}")
        if len(left_greedy) != len(right_greedy):
            raise HarnessError(f"greedy row counts differ for case {case_id}")
        if len(left_teacher) != len(right_teacher):
            raise HarnessError(f"teacher row counts differ for case {case_id}")

        greedy_matches = [a == b for a, b in zip(left_greedy, right_greedy)]
        teacher_matches = [a == b for a, b in zip(left_teacher, right_teacher)]
        sequence_equal = all(greedy_matches)
        sequences_equal += sequence_equal
        greedy_equal += sum(greedy_matches)
        teacher_equal += sum(teacher_matches)
        greedy_total += len(greedy_matches)
        teacher_total += len(teacher_matches)
        case_reports.append({
            "id": case_id,
            "sequence_equal": sequence_equal,
            "greedy_argmax_equal": sum(greedy_matches),
            "greedy_argmax_total": len(greedy_matches),
            "teacher_argmax_equal": sum(teacher_matches),
            "teacher_argmax_total": len(teacher_matches),
            "first_greedy_difference": next(
                (index for index, equal in enumerate(greedy_matches) if not equal), None
            ),
            "first_teacher_difference": next(
                (index for index, equal in enumerate(teacher_matches) if not equal), None
            ),
        })

    argmax_equal = greedy_equal + teacher_equal
    argmax_total = greedy_total + teacher_total
    passed = sequences_equal == len(reference) and argmax_equal == argmax_total
    return {
        "format": "ds4-qwen-argmax-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "reference": str(reference_root), "candidate": str(candidate_root),
        "sequences": {"equal": sequences_equal, "total": len(reference)},
        "argmax": {
            "equal": argmax_equal, "total": argmax_total,
            "greedy_equal": greedy_equal, "greedy_total": greedy_total,
            "teacher_equal": teacher_equal, "teacher_total": teacher_total,
        },
        "cases": case_reports,
    }


def correctness_artifacts(context: int, generation_tokens: int) -> list[tuple[str, str]]:
    artifacts = [("frontier", f"frontier_{context:06d}.logits.json")]
    if generation_tokens > 0:
        artifacts.append(("decode", f"frontier_{context:06d}.decode.json"))
    return artifacts


def compare_correctness_artifact(kind: str, baseline: Path,
                                 candidate: Path) -> dict[str, Any]:
    return (decode_result_drift(baseline, candidate)
            if kind == "decode" else logits_drift(baseline, candidate))


def correctness_test(baseline_path: Path, candidate_path: Path,
                     top_k: int = 20) -> dict[str, Any]:
    return logits_drift(baseline_path, candidate_path, top_k)


def tensor_storage_bytes(tensor: dict[str, Any]) -> int:
    tensor_type = tensor.get("type")
    if tensor_type not in QUANT_BYTES:
        raise HarnessError(f"unsupported tensor type in cost model: {tensor_type}")
    values = math.prod(int(dim) for dim in tensor["shape"])
    block_bytes, block_values = QUANT_BYTES[tensor_type]
    return math.ceil(values / block_values) * block_bytes


def tensor_group(name: str, full_attention_interval: int) -> str:
    if name.startswith("token_embd."):
        return "embedding"
    if name.startswith("output"):
        return "output"
    match = re.match(r"blk\.(\d+)\.", name)
    if not match:
        return "other"
    layer = int(match.group(1))
    if ".ffn_" in name or "post_attention_norm" in name:
        return "ffn"
    return ("full_attention" if layer % full_attention_interval ==
            full_attention_interval - 1 else "recurrent_attention")


def model_cost(snapshot: dict[str, Any], *, phase: str, context: int,
               batch: int, activation_bytes: int = 4,
               memory_gbps: float = 936.2,
               compute_tflops: float = 35.58) -> dict[str, Any]:
    if phase not in {"prefill", "decode"}:
        raise HarnessError("phase must be prefill or decode")
    if context < 1 or batch < 1 or activation_bytes not in {2, 4}:
        raise HarnessError("context/batch must be positive and activation bytes must be 2 or 4")
    if memory_gbps <= 0 or compute_tflops <= 0:
        raise HarnessError("memory bandwidth and compute throughput must be positive")
    metadata = snapshot.get("metadata", {})
    required = {
        "layers": "qwen35.block_count",
        "hidden": "qwen35.embedding_length",
        "ffn": "qwen35.feed_forward_length",
        "heads": "qwen35.attention.head_count",
        "kv_heads": "qwen35.attention.head_count_kv",
        "key_length": "qwen35.attention.key_length",
        "value_length": "qwen35.attention.value_length",
        "gdn_state": "qwen35.ssm.state_size",
        "gdn_groups": "qwen35.ssm.group_count",
        "gdn_inner": "qwen35.ssm.inner_size",
        "gdn_time_rank": "qwen35.ssm.time_step_rank",
        "gdn_conv_kernel": "qwen35.ssm.conv_kernel",
        "full_interval": "qwen35.full_attention_interval",
    }
    missing = [key for key in required.values() if key not in metadata]
    if missing:
        raise HarnessError("snapshot lacks cost metadata: " + ", ".join(missing))
    shape = {name: int(metadata[key]) for name, key in required.items()}
    vocabulary = int(metadata["tokenizer.ggml.tokens"]["count"])
    full_layers = shape["layers"] // shape["full_interval"]
    recurrent_layers = shape["layers"] - full_layers
    weights = {"embedding": 0, "recurrent_attention": 0,
               "full_attention": 0, "ffn": 0, "output": 0, "other": 0}
    theoretical_f32_weight_bytes = 0
    for tensor in snapshot.get("tensors", []):
        group = tensor_group(tensor["name"], shape["full_interval"])
        weights[group] += tensor_storage_bytes(tensor)
        theoretical_f32_weight_bytes += (
            math.prod(int(dim) for dim in tensor["shape"]) * 4
        )
    tokens = context if phase == "prefill" else 1
    h, f = shape["hidden"], shape["ffn"]
    q_dim = shape["heads"] * shape["key_length"]
    ffn_flops = batch * tokens * shape["layers"] * 6 * h * f
    recurrent_projection_dims = (
        shape["gdn_inner"] + 2 * shape["gdn_groups"] * shape["gdn_state"] +
        shape["gdn_inner"] + 2 * shape["gdn_time_rank"]
    )
    recurrent_flops = (
        batch * tokens * recurrent_layers * 2 *
        (h * recurrent_projection_dims + shape["gdn_inner"] * h)
    )
    kv_dim = shape["kv_heads"] * shape["key_length"]
    full_projection_flops = (
        batch * tokens * full_layers * 2 *
        (h * (2 * q_dim + kv_dim + kv_dim) + q_dim * h)
    )
    if phase == "decode":
        full_core_flops = batch * full_layers * 4 * q_dim * context
        weight_passes = batch
    else:
        full_core_flops = batch * full_layers * 2 * q_dim * context * context
        weight_passes = batch
    output_flops = batch * 2 * h * vocabulary
    flops = {
        "recurrent_attention": recurrent_flops,
        "full_attention": full_projection_flops + full_core_flops,
        "ffn": ffn_flops, "output": output_flops,
    }
    minimum_bytes = {
        name: weights[name] * weight_passes
        for name in ("recurrent_attention", "full_attention", "ffn", "output")
    }
    # Full-attention KV is persistent; recurrent layers use a fixed state.
    kv_bytes = (
        batch * context * full_layers * shape["kv_heads"] *
        (shape["key_length"] + shape["value_length"]) * activation_bytes
    )
    value_heads = shape["gdn_inner"] // shape["gdn_state"]
    recurrent_state_bytes = (
        batch * recurrent_layers * value_heads *
        shape["gdn_state"] * shape["gdn_state"] * activation_bytes
    )
    conv_state_bytes = (
        batch * recurrent_layers *
        (shape["gdn_inner"] + 2 * shape["gdn_groups"] * shape["gdn_state"]) *
        shape["gdn_conv_kernel"] * activation_bytes
    )
    operations = {}
    for name in flops:
        bytes_value = minimum_bytes[name]
        flop_value = flops[name]
        memory_floor_ms = bytes_value / (memory_gbps * 1e9) * 1e3
        compute_floor_ms = flop_value / (compute_tflops * 1e12) * 1e3
        operations[name] = {
            "flops": flop_value, "minimum_bytes": bytes_value,
            "arithmetic_intensity_flop_per_byte":
                flop_value / bytes_value if bytes_value else None,
            "memory_floor_ms": memory_floor_ms,
            "compute_floor_ms": compute_floor_ms,
            "roofline_floor_ms": max(memory_floor_ms, compute_floor_ms),
            "predicted_bound": "memory" if memory_floor_ms >= compute_floor_ms else "compute",
        }
    return {
        "format": "ds4-qwen-model-cost-v1", "phase": phase,
        "workload": {"context": context, "batch": batch,
                     "activation_bytes": activation_bytes},
        "model": {**shape, "vocabulary": vocabulary,
                  "full_attention_layers": full_layers,
                  "recurrent_layers": recurrent_layers},
        "hardware_assumption": {
            "memory_bandwidth_gbps": memory_gbps,
            "fp32_compute_tflops": compute_tflops,
        },
        "weight_bytes_by_operation": weights,
        "theoretical_f32_weight_bytes": theoretical_f32_weight_bytes,
        "effective_quantized_weight_bytes": sum(weights.values()),
        "state": {"full_attention_kv_bytes": kv_bytes,
                  "gdn_recurrent_state_bytes": recurrent_state_bytes,
                  "gdn_conv_state_bytes": conv_state_bytes},
        "limitations": [
            "minimum_bytes counts quantized weights only; activation, cache, "
            "temporary and write traffic are excluded",
            "recurrent_attention FLOPs include projections but not the "
            "Gated DeltaNet recurrence core",
            "normalization and elementwise activation FLOPs are excluded",
            "roofline floors use nominal hardware peaks, not sustained measured peaks",
        ],
        "operations": operations,
    }


def attach_observed_profile(cost: dict[str, Any],
                            profile: dict[str, Any]) -> dict[str, Any]:
    phase = cost["phase"]
    rows = [row for row in profile.get("layers", []) if row["phase"] == phase]
    if not rows:
        raise HarnessError(f"network profile contains no {phase} layer rows")
    selected = profile.get("selected_positions", {}).get(phase)
    expected_context = int(cost["workload"]["context"])
    if phase == "decode" and selected != expected_context:
        raise HarnessError(
            f"decode profile position {selected!r} does not match cost context {expected_context}"
        )
    if phase == "prefill" and any(int(row.get("rows", 0)) != expected_context for row in rows):
        raise HarnessError(
            f"prefill profile chunk does not match cost context {expected_context}"
        )
    cost["observed_profile"] = {
        "format": profile.get("format"),
        "model": profile.get("model"),
        "profile_context": profile.get("context"),
        "selected_position": selected,
        "profiled_layer_rows": len(rows),
        "timer_warning": "per-layer CUDA synchronization adds profiling overhead",
    }
    observed = {
        "recurrent_attention": sum(row["attention_ms"] for row in rows
                                   if row["kind"] == "recurrent"),
        "full_attention": sum(row["attention_ms"] for row in rows
                              if row["kind"] == "full"),
        "ffn": sum(row["ffn_ms"] for row in rows),
    }
    for name, actual_ms in observed.items():
        operation = cost["operations"][name]
        floor = operation["roofline_floor_ms"]
        operation["observed_profile_ms"] = actual_ms
        operation["observed_over_roofline_floor"] = (
            actual_ms / floor if floor > 0 else None
        )
    cost["optimization_opportunities"] = sorted(
        (
            {"operation": name,
             "observed_over_roofline_floor":
                 operation.get("observed_over_roofline_floor")}
            for name, operation in cost["operations"].items()
            if operation.get("observed_over_roofline_floor") is not None
        ),
        key=lambda item: item["observed_over_roofline_floor"],
        reverse=True,
    )
    return cost


def cmd_probe(args: argparse.Namespace) -> int:
    profile = hardware_profile()
    target = Path(args.output) if args.output else DEFAULT_RESULTS / "hardware_profile.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2))
    print(f"\nwrote {target}", file=sys.stderr)
    return 0


def binary_freshness(binary: Path, inputs: list[Path]) -> dict[str, Any]:
    existing_inputs = [path for path in inputs if path.exists()]
    if not binary.exists():
        return {
            "exists": False, "fresh": False,
            "newer_inputs": [str(path) for path in existing_inputs],
        }
    binary_mtime_ns = binary.stat().st_mtime_ns
    newer_inputs = [
        str(path) for path in existing_inputs
        if path.stat().st_mtime_ns > binary_mtime_ns
    ]
    return {
        "exists": True, "fresh": not newer_inputs,
        "binary_mtime_ns": binary_mtime_ns,
        "newer_inputs": newer_inputs,
    }


def cmd_doctor(_args: argparse.Namespace) -> int:
    profile = hardware_profile()
    shared_inputs = [
        ROOT / "ds4_cuda.cu", ROOT / "ds4_cuda.o",
        ROOT / "ds4.c", ROOT / "ds4.o",
        ROOT / "ds4_distributed.c", ROOT / "ds4_distributed.o",
        ROOT / "ds4_tp.c", ROOT / "ds4_tp.o",
        ROOT / "ds4_ssd.c", ROOT / "ds4_ssd.o",
        ROOT / "ds4_layer_pack.c", ROOT / "ds4_layer_pack.o",
        ROOT / "ds4_help.c", ROOT / "ds4_help.o",
        ROOT / "ds4_gpu_args.c", ROOT / "ds4_gpu_args.o",
    ]
    client = binary_freshness(
        ROOT / "ds4",
        shared_inputs + [
            ROOT / "ds4_cli.c", ROOT / "ds4_cli.o",
            ROOT / "linenoise.c", ROOT / "linenoise.o",
        ],
    )
    benchmark = binary_freshness(
        ROOT / "ds4-bench",
        shared_inputs + [ROOT / "ds4_bench.c", ROOT / "ds4_bench.o"],
    )
    server = binary_freshness(
        ROOT / "ds4-server",
        shared_inputs + [
            ROOT / "ds4_server.c", ROOT / "ds4_server.o",
            ROOT / "ds4_kvstore.c", ROOT / "ds4_kvstore.o",
            ROOT / "rax.c", ROOT / "rax.o",
        ],
    )
    checks = {
        "ds4_client": client["exists"],
        "ds4_client_fresh": client["fresh"],
        "ds4_bench": benchmark["exists"],
        "ds4_bench_fresh": benchmark["fresh"],
        "ds4_server": server["exists"],
        "ds4_server_fresh": server["fresh"],
        "nvidia_smi": profile["gpu"]["available"],
        "nsight_systems": profile["tools"]["nsys"]["available"],
        "nsight_compute": profile["tools"]["ncu"]["available"],
    }
    ready_for_benchmark = (
        checks["ds4_bench"] and checks["ds4_bench_fresh"] and
        checks["nvidia_smi"]
    )
    print(json.dumps({
        "ready_for_benchmark": ready_for_benchmark,
        "ready_for_interactive": (
            checks["ds4_client"] and checks["ds4_client_fresh"] and
            checks["nvidia_smi"]
        ),
        "ready_for_server": (
            checks["ds4_server"] and checks["ds4_server_fresh"] and
            checks["nvidia_smi"]
        ),
        "checks": checks,
        "runtime_binaries": {
            "ds4": client, "ds4-bench": benchmark, "ds4-server": server,
        },
        "qwen_decode_defaults": {
            "q8_1_r8": True,
            "split_k_partitions": 32,
            "split_k_min_context": QWEN_SPLIT_K_MIN_CONTEXT,
            "mtp_depth1_min_context": QWEN_MTP_DEPTH1_MIN_CONTEXT,
            "gqa_query_heads_per_cta": 2,
            "rollback_environment": [
                "DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8=1",
                "DS4_CUDA_QWEN_NO_SPLIT_K_ATTN=1",
                "DS4_CUDA_QWEN_NO_GQA_GROUP_ATTN=1",
                "DS4_MTP_QWEN_FORCE_DRAFT2=1",
                "DS4_MTP_QWEN_V2_SAFE_SNAPSHOT=1",
            ],
        },
        "hardware": profile,
    }, indent=2))
    return 0 if checks["ds4_bench"] and checks["ds4_bench_fresh"] else 1


def benchmark_model(args: argparse.Namespace) -> int:
    binary, model, prompt = map(Path, (args.binary, args.model, args.prompt))
    for path, label in ((binary, "benchmark binary"), (model, "model"), (prompt, "prompt")):
        if not path.exists():
            raise HarnessError(f"{label} does not exist: {path}")
    if not args.hypothesis.strip() or not args.metric.strip():
        raise HarnessError("an experiment requires --hypothesis and --metric")
    workloads = load_workloads(Path(args.workloads), args.suite)
    experiment_id = args.id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.results) / experiment_id
    if out_dir.exists():
        raise HarnessError(f"experiment already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    env = apply_env_overrides(os.environ, args.env)
    results = []
    correctness_rows = []
    use_warmup = args.warmup == "always" or (
        args.warmup == "auto" and args.suite not in {"direction", "quick"}
    )
    baseline_logits_root = (
        None if args.baseline_run else Path(args.baseline).resolve().parent / "logits"
    )
    try:
        if args.suite in {
                "direction", "r8-slow", "long-context-slow",
                CONTEXT_CURVE_SUITE}:
            contexts = sorted(int(item["context"]) for item in workloads)
            common = ("generation_tokens", "prefill_chunk", "backend", "batch")
            if any(
                item.get(key, 1 if key == "batch" else None) !=
                workloads[0].get(key, 1 if key == "batch" else None)
                for item in workloads[1:] for key in common
            ):
                raise HarnessError(
                    f"{args.suite} workloads must share generation/prefill/backend/batch"
                )
            if (args.suite in {"direction", "r8-slow"} and
                    len(contexts) != 2):
                raise HarnessError(
                    f"{args.suite} suite must contain exactly two contexts"
                )
            if len(contexts) < 2:
                raise HarnessError(f"{args.suite} suite needs at least two contexts")
            step_incr = contexts[1] - contexts[0]
            if any(right - left != step_incr
                   for left, right in zip(contexts, contexts[1:])):
                raise HarnessError(
                    f"{args.suite} contexts must be evenly spaced for a resident sweep"
                )
            sweep = dict(
                workloads[0], context=contexts[0], context_max=contexts[1],
                step_incr=step_incr,
            )
            sweep["context_max"] = contexts[-1]
            sweep_logits = out_dir / f"{args.suite}-sweep-logits"
            if use_warmup:
                warm_csv = out_dir / f"{args.suite}-sweep.warmup.csv"
                bench_once(
                    binary, model, prompt, sweep, env, warm_csv,
                    out_dir / f"{args.suite}-sweep-warmup-logits",
                )
                warm_csv.unlink(missing_ok=True)
            rows = bench_once(
                binary, model, prompt, sweep, env,
                out_dir / f"{args.suite}-sweep.csv", sweep_logits,
                repetitions=args.repetitions,
            )
            for workload in workloads:
                context = int(workload["context"])
                samples = [row for row in rows if int(row["ctx_tokens"]) == context]
                if len(samples) != args.repetitions:
                    raise HarnessError(
                        f"{args.suite} sweep emitted {len(samples)} samples for context {context}; "
                        f"expected {args.repetitions}"
                    )
                candidate_logits_dir = out_dir / "logits" / workload["id"]
                candidate_logits_dir.mkdir(parents=True, exist_ok=True)
                for artifact_kind, artifact_name in correctness_artifacts(
                    context, int(workload.get("generation_tokens", 0))
                ):
                    candidate_artifact = candidate_logits_dir / artifact_name
                    shutil.copy2(sweep_logits / artifact_name, candidate_artifact)
                    if args.baseline_run:
                        correctness_rows.append({
                            "workload": workload["id"], "artifact": artifact_kind,
                            "status": "BASELINE",
                            "candidate_artifact": str(candidate_artifact),
                        })
                    else:
                        baseline_artifact = (
                            baseline_logits_root / workload["id"] / artifact_name
                        )
                        if baseline_artifact.is_file():
                            drift = compare_correctness_artifact(
                                artifact_kind, baseline_artifact, candidate_artifact
                            )
                            drift.update({
                                "workload": workload["id"],
                                "artifact": artifact_kind,
                            })
                            correctness_rows.append(drift)
                        else:
                            correctness_rows.append({
                                "workload": workload["id"],
                                "artifact": artifact_kind,
                                "status": "NOT_VERIFIED",
                                "reason": f"matching baseline {artifact_kind} artifact is missing",
                            })
                results.append({
                    "id": workload["id"], "status": "measured",
                    "definition": workload, **aggregate_runs([samples]),
                })
            workloads = []
        for workload in workloads:
            if workload.get("batch", 1) != 1:
                results.append({
                    "id": workload["id"], "status": "not_verified",
                    "reason": "ds4-bench currently supports one session per process",
                })
                continue
            warm_csv = out_dir / f"{workload['id']}.warmup.csv"
            candidate_logits_dir = out_dir / "logits" / workload["id"]
            if use_warmup:
                bench_once(binary, model, prompt, workload, env, warm_csv,
                           candidate_logits_dir)
                warm_csv.unlink(missing_ok=True)
            path = out_dir / f"{workload['id']}.csv"
            runs = [bench_once(
                binary, model, prompt, workload, env, path,
                candidate_logits_dir, repetitions=args.repetitions,
            )]
            for artifact_kind, artifact_name in correctness_artifacts(
                int(workload["context"]), int(workload.get("generation_tokens", 0))
            ):
                candidate_artifact = candidate_logits_dir / artifact_name
                if args.baseline_run:
                    correctness_rows.append({
                        "workload": workload["id"], "artifact": artifact_kind,
                        "status": "BASELINE",
                        "candidate_artifact": str(candidate_artifact),
                    })
                else:
                    baseline_artifact = (
                        baseline_logits_root / workload["id"] / artifact_name
                    )
                    if baseline_artifact.is_file() and candidate_artifact.is_file():
                        drift = compare_correctness_artifact(
                            artifact_kind, baseline_artifact, candidate_artifact
                        )
                        drift.update({
                            "workload": workload["id"],
                            "artifact": artifact_kind,
                        })
                        correctness_rows.append(drift)
                    else:
                        correctness_rows.append({
                            "workload": workload["id"],
                            "artifact": artifact_kind,
                            "status": "NOT_VERIFIED",
                            "reason": f"matching baseline {artifact_kind} artifact is missing",
                        })
            results.append({
                "id": workload["id"], "status": "measured",
                "definition": workload, **aggregate_runs(runs),
            })
    except Exception:
        (out_dir / "FAILED").write_text(utc_now() + "\n", encoding="utf-8")
        raise
    record = {
        "schema_version": SCHEMA_VERSION, "experiment_id": experiment_id,
        "created_at": utc_now(), "status": "measured",
        "hypothesis": args.hypothesis, "target_metric": args.metric,
        "suite": args.suite, "repetitions": args.repetitions,
        "warmup": use_warmup,
        "baseline": {
            "kind": "self" if args.baseline_run else "experiment",
            "path": None if args.baseline_run else str(Path(args.baseline).resolve()),
        },
        "provenance": {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
            "binary": str(binary.resolve()), "binary_sha256": sha256(binary),
            "model": str(model.resolve()),
            "model_sha256": cached_sha256(model, Path(args.results)),
            "prompt": str(prompt.resolve()), "prompt_sha256": sha256(prompt),
            "environment_overrides": dict(item.split("=", 1) for item in args.env),
        },
        "hardware": hardware_profile(), "workloads": results,
        "correctness": {
            "status": ("BASELINE" if args.baseline_run else
                       "FAIL" if any(row["status"] == "FAIL" for row in correctness_rows) else
                       "PASS" if correctness_rows and
                       all(row["status"] == "PASS" for row in correctness_rows) else
                       "NOT_VERIFIED"),
            "artifacts": correctness_rows,
            "frontier_logits": [
                row for row in correctness_rows if row.get("artifact") == "frontier"
            ],
            "decode_results": [
                row for row in correctness_rows if row.get("artifact") == "decode"
            ],
        },
    }
    if args.suite == CONTEXT_CURVE_SUITE:
        record["context_curve"] = analyze_context_curve(results)
    record_path = out_dir / "experiment.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print_compact(record)
    print(f"\nwrote {record_path}", file=sys.stderr)
    if args.baseline:
        baseline = read_json(Path(args.baseline), "baseline experiment")
        comparison = compare_records(baseline, record)
        comparison_path = out_dir / "comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"verdict: {comparison['verdict']} "
            f"({comparison['mean_target_improvement_percent']!r}% target change)"
        )
        for workload in comparison["workloads"]:
            target = workload["metrics"].get(comparison["target_metric"])
            if target:
                print(f"  {workload['id']}: {target['improvement_percent']:+.2f}%")
        print(f"wrote {comparison_path}", file=sys.stderr)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return benchmark_model(args)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HarnessError(f"server returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise HarnessError(f"server request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessError("server response is not a JSON object")
    return value


def post_json_expect_http_error(
    url: str, payload: dict[str, Any], expected_status: int, timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            detail = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code != expected_status:
            raise HarnessError(
                f"server returned HTTP {exc.code}, expected {expected_status}: {detail}"
            ) from exc
        return {"status": exc.code, "body": detail}
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HarnessError(f"server request failed: {exc}") from exc
    raise HarnessError(
        f"server accepted an unsupported schema with HTTP 2xx: {detail}"
    )


def wait_for_server(base_url: str, process: subprocess.Popen[Any],
                    log_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            )
            raise HarnessError(f"ds4-server exited during startup\n{tail}")
        try:
            with urllib.request.urlopen(base_url + "/v1/models", timeout=1.0):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    raise HarnessError(f"ds4-server was not ready after {timeout:.1f}s")


def server_progress_since(log_path: Path, offset: int,
                          timeout: float = 2.0) -> tuple[dict[str, Any], int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(offset)
            text = stream.read()
            end = stream.tell()
        rows = parse_server_progress(text)
        if rows:
            return rows[-1], end
        time.sleep(0.05)
    raise HarnessError("server response completed without a decode progress log")


def server_phase_since(log_path: Path, offset: int,
                       timeout: float = 2.0) -> tuple[dict[str, Any], int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(offset)
            text = stream.read()
            end = stream.tell()
        rows = parse_server_phase_profiles(text)
        if rows:
            return rows[-1], end
        time.sleep(0.05)
    raise HarnessError("server response completed without a phase profile log")


def load_constrained_workloads(path: Path, suite: str) -> list[dict[str, Any]]:
    record = read_json(path, "constrained workload file")
    if record.get("schema_version") != 1:
        raise HarnessError("unsupported constrained workload schema_version")
    suite_ids = record.get("suites", {}).get(suite)
    if not isinstance(suite_ids, list) or not suite_ids:
        raise HarnessError(f"unknown or empty constrained suite: {suite}")
    by_id = {
        item.get("id"): item
        for item in record.get("workloads", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    missing = [item_id for item_id in suite_ids if item_id not in by_id]
    if missing:
        raise HarnessError(
            "constrained suite references missing workloads: " + ", ".join(missing)
        )
    workloads = [by_id[item_id] for item_id in suite_ids]
    for item in workloads:
        if item.get("kind") not in ("dsml", "json_schema"):
            raise HarnessError(f"invalid constrained workload kind: {item.get('id')}")
        if not isinstance(item.get("endpoint"), str) or not item["endpoint"].startswith("/"):
            raise HarnessError(f"invalid constrained workload endpoint: {item.get('id')}")
        if not isinstance(item.get("payload"), dict):
            raise HarnessError(f"invalid constrained workload payload: {item.get('id')}")
    return workloads


def load_jsonschemabench_workloads(
    path: Path, selected_ids: list[str] | None = None, tier: str = "safety",
    prefix_steps: int = 0,
) -> list[dict[str, Any]]:
    record = read_json(path, "JSONSchemaBench subset")
    if record.get("schema_version") != 1:
        raise HarnessError("unsupported JSONSchemaBench subset schema_version")
    entries = record.get("supported")
    if not isinstance(entries, list) or not entries:
        raise HarnessError("JSONSchemaBench subset has no supported schemas")
    if selected_ids is None:
        tier_ids = record.get("tiers", {}).get(tier)
        if not isinstance(tier_ids, list) or not tier_ids or not all(
            isinstance(item, str) for item in tier_ids
        ):
            raise HarnessError(f"unknown or empty JSONSchemaBench tier: {tier}")
        selected_ids = tier_ids
    workloads: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise HarnessError("JSONSchemaBench supported entry lacks an id")
        schema = entry.get("schema")
        if not isinstance(schema, (dict, bool)):
            raise HarnessError(
                f"JSONSchemaBench schema is not object/boolean: {entry['id']}"
            )
        encoded = json.dumps(
            schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != entry.get("sha256"):
            raise HarnessError(
                f"JSONSchemaBench schema hash mismatch: {entry['id']}"
            )
        if entry.get("unsupported_reasons"):
            raise HarnessError(
                f"unsupported schema leaked into benchmark: {entry['id']}"
            )
        if selected_ids and entry["id"] not in selected_ids:
            continue
        witness = minimal_json_schema_witness(schema)
        witness_validation = validate_json_schema_instance(schema, witness)
        witness_text = json.dumps(
            witness, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        workloads.append({
            "id": "jsonschemabench/" + entry["id"],
            "kind": "json_schema",
            "endpoint": "/v1/chat/completions",
            "source": {
                "benchmark": "JSONSchemaBench",
                "id": entry["id"],
                "category": entry.get("category"),
                "schema_sha256": entry["sha256"],
                "witness_sha256": hashlib.sha256(
                    witness_text.encode("utf-8")
                ).hexdigest(),
                "witness_valid": witness_validation["valid"],
                "witness_validator": witness_validation["validator"],
                "prefix_steps": prefix_steps,
            },
            "payload": {
                "model": "model.gguf",
                "messages": [{
                    "role": "user",
                    "content": (
                        "Generate the shortest JSON value satisfying the supplied "
                        "schema. Output only that JSON value."
                    ),
                }],
                "temperature": 0,
                "seed": 424242,
                "max_tokens": prefix_steps if prefix_steps else 256,
                "stream": False,
                "thinking": {"type": "disabled"},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"jsonschemabench_{index:02d}",
                        "strict": True,
                        "schema": schema,
                    },
                },
            },
        })
    if selected_ids:
        loaded_ids = {item["source"]["id"] for item in workloads}
        missing = sorted(set(selected_ids) - loaded_ids)
        if missing:
            raise HarnessError(
                "JSONSchemaBench ids are not in the supported subset: " +
                ", ".join(missing)
            )
    return workloads


def load_jsonschemabench_unsupported_probes(path: Path) -> list[dict[str, Any]]:
    record = read_json(path, "JSONSchemaBench subset")
    entries = record.get("unsupported")
    if record.get("schema_version") != 1 or not isinstance(entries, list) or not entries:
        raise HarnessError("JSONSchemaBench subset has no unsupported probes")
    probes: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        schema = entry.get("schema") if isinstance(entry, dict) else None
        reasons = entry.get("unsupported_reasons") if isinstance(entry, dict) else None
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not isinstance(schema, (dict, bool)):
            raise HarnessError("malformed JSONSchemaBench unsupported entry")
        if not isinstance(reasons, list) or not reasons:
            raise HarnessError(f"unsupported probe lacks reasons: {entry['id']}")
        encoded = json.dumps(
            schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != entry.get("sha256"):
            raise HarnessError(
                f"JSONSchemaBench unsupported hash mismatch: {entry['id']}"
            )
        probes.append({
            "id": entry["id"], "reasons": reasons,
            "payload": {
                "model": "model.gguf",
                "messages": [{
                    "role": "user", "content": "Do not run inference."
                }],
                "temperature": 0, "max_tokens": 1, "stream": False,
                "thinking": {"type": "disabled"},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"jsonschemabench_unsupported_{index:02d}",
                        "strict": True, "schema": schema,
                    },
                },
            },
        })
    return probes


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HarnessError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def parse_json_strict(text: str, label: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_json_pairs)
    except HarnessError:
        raise
    except json.JSONDecodeError as exc:
        raise HarnessError(f"{label} is not JSON") from exc


def validate_json_schema_instance(
    schema: dict[str, Any] | bool, instance: Any,
) -> dict[str, Any]:
    """Validate with an implementation independent from DS4's C validator."""
    try:
        import jsonschema
    except ImportError as exc:
        raise HarnessError(
            "JSONSchemaBench safety validation requires jsonschema; install "
            "performance/jsonschemabench-requirements.txt"
        ) from exc

    try:
        validator_class = jsonschema.validators.validator_for(schema)
        validator_class.check_schema(schema)
        errors = sorted(
            validator_class(schema).iter_errors(instance),
            key=lambda item: (
                tuple(str(part) for part in item.absolute_path),
                tuple(str(part) for part in item.absolute_schema_path),
            ),
        )
    except jsonschema.exceptions.SchemaError as exc:
        raise HarnessError(
            f"JSONSchemaBench source schema is invalid: {exc.message}"
        ) from exc
    if errors:
        first = errors[0]
        instance_path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in first.absolute_path
        )
        raise HarnessError(
            "constrained output violates JSON Schema at "
            f"{instance_path}: {first.message}"
        )
    return {
        "valid": True,
        "validator": validator_class.__name__,
        "dialect": schema.get("$schema") if isinstance(schema, dict) else None,
    }


def _merge_schema_constraints(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge the supported intersection keywords needed to build a witness."""
    result: dict[str, Any] = {}
    properties: dict[str, list[dict[str, Any]]] = {}
    required: list[str] = []
    for part in parts:
        nested = part.get("allOf")
        base = {key: value for key, value in part.items() if key != "allOf"}
        expanded = [base]
        if isinstance(nested, list):
            expanded.extend(item for item in nested if isinstance(item, dict))
        if len(expanded) > 1:
            base = _merge_schema_constraints(expanded)
        for key, value in base.items():
            if key == "properties" and isinstance(value, dict):
                for name, child in value.items():
                    if isinstance(child, dict):
                        properties.setdefault(name, []).append(child)
                continue
            if key == "required" and isinstance(value, list):
                for name in value:
                    if isinstance(name, str) and name not in required:
                        required.append(name)
                continue
            if key in {"minLength", "minItems", "minProperties", "minimum"}:
                result[key] = max(value, result.get(key, value))
                continue
            if key in {"maxLength", "maxItems", "maxProperties", "maximum"}:
                result[key] = min(value, result.get(key, value))
                continue
            if key == "additionalProperties" and value is False:
                result[key] = False
                continue
            if key == "enum" and isinstance(value, list):
                previous = result.get("enum")
                result[key] = (
                    value if not isinstance(previous, list) else
                    [item for item in previous if item in value]
                )
                continue
            if key == "type" and "type" in result and result["type"] != value:
                left = result["type"] if isinstance(result["type"], list) else [result["type"]]
                right = value if isinstance(value, list) else [value]
                result["type"] = [item for item in left if item in right]
                continue
            result[key] = value
    if properties:
        result["properties"] = {
            name: (
                children[0] if len(children) == 1 else
                _merge_schema_constraints(children)
            )
            for name, children in properties.items()
        }
    if required:
        result["required"] = required
    return result


def _schema_candidate(schema: dict[str, Any] | bool, depth: int = 0) -> Any:
    if depth > 64:
        raise HarnessError("JSONSchemaBench witness exceeds nesting limit")
    if schema is True:
        return None
    if schema is False:
        raise HarnessError("false JSON Schema has no witness")
    if "const" in schema:
        return schema["const"]
    enumeration = schema.get("enum")
    if isinstance(enumeration, list) and enumeration:
        return min(
            enumeration,
            key=lambda item: len(json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")),
        )
    if isinstance(schema.get("allOf"), list):
        schema = _merge_schema_constraints([schema])
    for keyword in ("oneOf", "anyOf"):
        variants = schema.get(keyword)
        if isinstance(variants, list):
            base = {key: value for key, value in schema.items() if key != keyword}
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                candidate = _schema_candidate(
                    _merge_schema_constraints([base, variant]), depth + 1
                )
                try:
                    validate_json_schema_instance(schema, candidate)
                    return candidate
                except HarnessError:
                    continue
            raise HarnessError(f"cannot construct witness for {keyword}")

    declared = schema.get("type")
    types = declared if isinstance(declared, list) else [declared]
    if declared is None:
        if "properties" in schema or "required" in schema:
            types = ["object"]
        elif "items" in schema:
            types = ["array"]
        else:
            types = ["null", "boolean", "integer", "number", "string", "array", "object"]
    for kind in types:
        if kind == "null":
            return None
        if kind == "boolean":
            return False
        if kind in ("integer", "number"):
            multiple = float(schema.get("multipleOf", 1))
            lower = float(schema.get("minimum", 0))
            if "exclusiveMinimum" in schema:
                lower = max(lower, float(schema["exclusiveMinimum"]) + multiple)
            value = math.ceil(lower / multiple) * multiple
            if kind == "integer":
                value = math.ceil(value)
            maximum = schema.get("maximum")
            if maximum is not None and value > float(maximum):
                continue
            return int(value) if kind == "integer" else value
        if kind == "string":
            return "a" * int(schema.get("minLength", 0))
        if kind == "array":
            count = int(schema.get("minItems", 0))
            item_schema = schema.get("items", True)
            values = [_schema_candidate(item_schema, depth + 1) for _ in range(count)]
            if schema.get("uniqueItems") and len(values) > 1:
                for index in range(len(values)):
                    values[index] = index
            return values
        if kind == "object":
            props = schema.get("properties")
            props = props if isinstance(props, dict) else {}
            names = list(dict.fromkeys(schema.get("required", [])))
            minimum = int(schema.get("minProperties", 0))
            for name in sorted(props):
                if len(names) >= minimum:
                    break
                if name not in names:
                    names.append(name)
            result = {
                name: _schema_candidate(props.get(name, True), depth + 1)
                for name in names
            }
            while len(result) < minimum:
                name = f"p{len(result)}"
                if name not in result:
                    additional = schema.get("additionalProperties", True)
                    if additional is False:
                        raise HarnessError("minProperties exceeds declared properties")
                    result[name] = _schema_candidate(additional, depth + 1)
            return result
    raise HarnessError("cannot construct JSONSchemaBench witness")


def minimal_json_schema_witness(schema: dict[str, Any] | bool) -> Any:
    candidates = [None, False, 0, "", [], {}, True, 1]
    candidates.append(_schema_candidate(schema))
    valid: list[Any] = []
    for candidate in candidates:
        try:
            validate_json_schema_instance(schema, candidate)
            valid.append(candidate)
        except HarnessError:
            pass
    if not valid:
        raise HarnessError("cannot construct a valid JSONSchemaBench witness")
    return min(
        valid,
        key=lambda item: len(json.dumps(
            item, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")),
    )


def constrained_semantic_output(kind: str,
                                response: dict[str, Any]) -> dict[str, Any]:
    if kind == "dsml":
        calls = []
        for item in response.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                raise HarnessError("DSML response function_call lacks arguments")
            parsed = parse_json_strict(arguments, "DSML response arguments")
            calls.append({"name": item.get("name"), "arguments": parsed})
        if not calls:
            raise HarnessError("DSML response contains no function_call")
        return {"function_calls": calls}
    choices = response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else None
    if choice and choice.get("finish_reason") == "error":
        raise HarnessError("JSON Schema response ended with an error")
    content = choice.get("message", {}).get("content") if choice else None
    if not isinstance(content, str):
        raise HarnessError("JSON Schema response lacks assistant content")
    return {"json": parse_json_strict(
        content, "JSON Schema response content"
    )}


def constrained_json_prefix_output(
    response: dict[str, Any], completion_tokens: int, required_steps: int,
) -> dict[str, Any]:
    choices = response.get("choices") or []
    choice = choices[0] if len(choices) == 1 and isinstance(choices[0], dict) else None
    content = choice.get("message", {}).get("content") if choice else None
    finish_reason = choice.get("finish_reason") if choice else None
    if not isinstance(content, str) or finish_reason != "error":
        raise HarnessError(
            "JSONSchemaBench prefix probe must either complete valid JSON or "
            "end with the structured-output error"
        )
    if completion_tokens != required_steps:
        raise HarnessError(
            "JSONSchemaBench prefix probe ended before its requested step budget"
        )
    return {
        "json_prefix": content,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
    }


def constrained_usage(response: dict[str, Any]) -> tuple[int, int]:
    usage = response.get("usage") or {}
    prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
    completion = usage.get("output_tokens", usage.get("completion_tokens"))
    if not isinstance(prompt, int) or not isinstance(completion, int):
        raise HarnessError("constrained response lacks integer token usage")
    return prompt, completion


def completion_request(base_url: str, prompt: str, max_tokens: int,
                       timeout: float,
                       thinking: bool | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "deepseek-chat", "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0,
        "seed": 424242, "stream": False,
    }
    if thinking is not None:
        body["thinking"] = thinking
    return post_json(base_url + "/v1/completions", body, timeout)


def completion_usage(response: dict[str, Any]) -> tuple[int, int]:
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        raise HarnessError("completion response lacks integer token usage")
    return prompt_tokens, completion_tokens


def benchmark_server_curve(args: argparse.Namespace) -> int:
    binary, model = Path(args.binary), Path(args.model)
    for path, label in ((binary, "server binary"), (model, "model")):
        if not path.exists():
            raise HarnessError(f"{label} does not exist: {path}")
    workloads = load_workloads(Path(args.workloads), args.suite)
    contexts = [int(item["context"]) for item in workloads]
    generation_tokens = int(workloads[0]["generation_tokens"])
    if any(int(item["generation_tokens"]) != generation_tokens for item in workloads):
        raise HarnessError("server curve workloads must use one generation length")
    experiment_id = args.id or datetime.now().strftime("%Y%m%d-%H%M%S-server")
    out_dir = Path(args.results) / experiment_id
    if out_dir.exists():
        raise HarnessError(f"experiment already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    log_path = out_dir / "server.log"
    env = apply_env_overrides(os.environ, args.env)
    port = args.port
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
    base_url = f"http://127.0.0.1:{port}"
    minimum_context_alloc = max(contexts) + generation_tokens + 1
    context_alloc = (
        args.context_alloc
        if args.context_alloc is not None
        else minimum_context_alloc
    )
    if context_alloc < minimum_context_alloc:
        raise HarnessError(
            f"--context-alloc {context_alloc} is below the required "
            f"{minimum_context_alloc} tokens"
        )
    command = [
        str(binary), "--model", str(model), "--backend", "cuda",
        "--ctx", str(context_alloc), "--host", "127.0.0.1", "--port", str(port),
        "--prefill-chunk", str(workloads[0]["prefill_chunk"]),
    ]
    if args.mtp_model:
        mtp_model = Path(args.mtp_model)
        if not mtp_model.exists():
            raise HarnessError(f"MTP model does not exist: {mtp_model}")
        command += ["--mtp-model", str(mtp_model)]
    elif args.mtp:
        command += ["--mtp"]
    process: subprocess.Popen[Any] | None = None
    log_stream = None
    results = []
    try:
        log_stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log_stream,
            stderr=subprocess.STDOUT, text=True,
        )
        wait_for_server(base_url, process, log_path, args.startup_timeout)

        first_count, second_count = 32, 96
        first = completion_request(
            base_url, server_curve_prompt(first_count, args.prompt_pattern),
            1, args.timeout,
            args.thinking,
        )
        second = completion_request(
            base_url, server_curve_prompt(second_count, args.prompt_pattern),
            1, args.timeout,
            args.thinking,
        )
        first_tokens, _ = completion_usage(first)
        second_tokens, _ = completion_usage(second)
        intercept = prompt_filler_intercept(
            first_count, first_tokens, second_count, second_tokens
        )
        log_offset = log_path.stat().st_size

        for workload in workloads:
            context = int(workload["context"])
            # Zero is the requested empty-context bucket. Inference still needs
            # a real prompt, so use the smallest calibrated wrapper and retain
            # its actual token count in raw_rows.
            prompt = server_curve_prompt(
                0 if context == 0 else context - intercept,
                args.prompt_pattern,
            )
            samples = []
            output_hashes = []
            completion_counts = []
            for _ in range(args.repetitions):
                request_offset = log_path.stat().st_size
                response = completion_request(
                    base_url, prompt, generation_tokens, args.timeout,
                    args.thinking,
                )
                actual_context, completion_count = completion_usage(response)
                if context != 0 and actual_context != context:
                    raise HarnessError(
                        f"server prompt calibration missed context {context}: "
                        f"observed {actual_context}"
                    )
                if completion_count < args.minimum_completion_tokens:
                    raise HarnessError(
                        f"server generated only {completion_count} tokens at context "
                        f"{context}; need at least {args.minimum_completion_tokens}"
                    )
                progress, log_offset = server_progress_since(
                    log_path, request_offset
                )
                if progress["generation_tokens"] != completion_count:
                    raise HarnessError(
                        "server progress log and response usage disagree on completion tokens"
                    )
                samples.append(progress["avg_tps"])
                completion_counts.append(completion_count)
                choices = response.get("choices") or []
                output = choices[0].get("text", "") if choices else ""
                output_hashes.append(hashlib.sha256(output.encode("utf-8")).hexdigest())
            results.append({
                "id": workload["id"], "status": "measured",
                "definition": workload,
                "metrics": {"gen_steady_tps": summary(samples)},
                "raw_rows": [
                    {
                        "ctx_tokens": context,
                        "actual_ctx_tokens": (
                            intercept if context == 0 else context
                        ),
                        "gen_steady_tps": tps,
                        "completion_tokens": tokens,
                    }
                    for tps, tokens in zip(samples, completion_counts)
                ],
                "server_output": {
                    "deterministic": len(set(output_hashes)) == 1,
                    "sha256": output_hashes[0],
                    "completion_tokens": sorted(set(completion_counts)),
                },
            })
    except Exception:
        (out_dir / "FAILED").write_text(utc_now() + "\n", encoding="utf-8")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_stream is not None:
            log_stream.close()

    baseline_record = (
        None if args.baseline_run else read_json(Path(args.baseline), "baseline experiment")
    )
    deterministic = all(
        item["server_output"]["deterministic"] for item in results
    )
    baseline_outputs_match = True
    if baseline_record:
        baseline_by_id = {item["id"]: item for item in baseline_record["workloads"]}
        baseline_outputs_match = all(
            baseline_by_id.get(item["id"], {}).get("server_output", {}).get("sha256") ==
            item["server_output"]["sha256"]
            for item in results
        )
    record = {
        "schema_version": SCHEMA_VERSION, "experiment_id": experiment_id,
        "created_at": utc_now(), "status": "measured",
        "hypothesis": args.hypothesis, "target_metric": "gen_steady_tps",
        "suite": args.suite, "runtime": "ds4-server",
        "repetitions": args.repetitions,
        "baseline": {
            "kind": "self" if args.baseline_run else "experiment",
            "path": None if args.baseline_run else str(Path(args.baseline).resolve()),
        },
        "provenance": {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
            "binary": str(binary.resolve()), "binary_sha256": sha256(binary),
            "model": str(model.resolve()),
            "model_sha256": cached_sha256(model, Path(args.results)),
            "environment_overrides": dict(item.split("=", 1) for item in args.env),
            "command": command, "prompt_filler_token_intercept": intercept,
        },
        "hardware": hardware_profile(), "workloads": results,
        "context_curve": analyze_context_curve(results),
        "correctness": {
            "status": (
                "BASELINE" if args.baseline_run else
                "PASS" if deterministic and baseline_outputs_match else "FAIL"
            ),
            "deterministic_outputs": deterministic,
            "baseline_outputs_match": baseline_outputs_match,
        },
    }
    record_path = out_dir / "experiment.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print_compact(record)
    print(f"\nwrote {record_path}", file=sys.stderr)
    if baseline_record:
        comparison = compare_records(baseline_record, record)
        comparison_path = out_dir / "comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(f"verdict: {comparison['verdict']} ({comparison['reason']})")
    return 0


def cmd_server_curve(args: argparse.Namespace) -> int:
    return benchmark_server_curve(args)


def benchmark_constrained_server(args: argparse.Namespace) -> int:
    binary, model = Path(args.binary), Path(args.model)
    for path, label in ((binary, "server binary"), (model, "model")):
        if not path.exists():
            raise HarnessError(f"{label} does not exist: {path}")
    workload_path = Path(
        args.jsonschemabench_subset
        if args.jsonschemabench_subset else args.workloads
    )
    if args.jsonschemabench_id and not args.jsonschemabench_subset:
        raise HarnessError("--jsonschemabench-id requires --jsonschemabench-subset")
    if args.jsonschemabench_check_unsupported and not args.jsonschemabench_subset:
        raise HarnessError(
            "--jsonschemabench-check-unsupported requires --jsonschemabench-subset"
        )
    if args.jsonschemabench_prefix_steps:
        if not args.jsonschemabench_subset:
            raise HarnessError(
                "--jsonschemabench-prefix-steps requires --jsonschemabench-subset"
            )
        if args.constraint_mode != "compare_new_vs_oracle":
            raise HarnessError(
                "JSONSchemaBench prefix probes require compare_new_vs_oracle"
            )
        if not 1 <= args.jsonschemabench_prefix_steps <= 64:
            raise HarnessError("JSONSchemaBench prefix steps must be in 1..64")
    workloads = (
        load_jsonschemabench_workloads(
            workload_path, args.jsonschemabench_id, args.jsonschemabench_tier,
            args.jsonschemabench_prefix_steps,
        )
        if args.jsonschemabench_subset else
        load_constrained_workloads(workload_path, args.suite)
    )
    unsupported_probes = (
        load_jsonschemabench_unsupported_probes(workload_path)
        if args.jsonschemabench_check_unsupported else []
    )
    suite_name = (
        f"jsonschemabench-{args.jsonschemabench_tier}" + (
            f"-prefix-{args.jsonschemabench_prefix_steps}"
            if args.jsonschemabench_prefix_steps else ""
        )
        if args.jsonschemabench_subset else args.suite
    )
    experiment_id = args.id or datetime.now().strftime("%Y%m%d-%H%M%S-constrained")
    out_dir = Path(args.results) / experiment_id
    if out_dir.exists():
        raise HarnessError(f"experiment already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    log_path = out_dir / "server.log"
    env = apply_env_overrides(os.environ, args.env)
    env["DS4_SERVER_PHASE_PROFILE"] = "1"
    env["DS4_CONSTRAINT_MODE"] = args.constraint_mode
    port = args.port
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(binary), "--model", str(model), "--backend", "cuda",
        "--ctx", str(args.context), "--host", "127.0.0.1", "--port", str(port),
        "--prefill-chunk", str(args.prefill_chunk),
    ]
    process: subprocess.Popen[Any] | None = None
    log_stream = None
    results = []
    unsupported_results: list[dict[str, Any]] = []
    try:
        log_stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log_stream,
            stderr=subprocess.STDOUT, text=True,
        )
        wait_for_server(base_url, process, log_path, args.startup_timeout)
        model_name = model.name
        for probe in unsupported_probes:
            payload = json.loads(json.dumps(probe["payload"]))
            payload["model"] = model_name
            request_offset = log_path.stat().st_size
            try:
                rejection = post_json_expect_http_error(
                    base_url + "/v1/chat/completions", payload, 400, args.timeout
                )
            except HarnessError as exc:
                raise HarnessError(
                    f"unsupported schema probe failed for {probe['id']}: {exc}"
                ) from exc
            time.sleep(0.02)
            with log_path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(request_offset)
                request_log = stream.read()
            if " prompt start" in request_log or "phase profile" in request_log:
                raise HarnessError(
                    f"unsupported schema reached inference: {probe['id']}"
                )
            unsupported_results.append({
                "id": probe["id"], "status": rejection["status"],
                "reasons": probe["reasons"], "inference_started": False,
            })
        for workload in workloads:
            payload = json.loads(json.dumps(workload["payload"]))
            payload["model"] = model_name
            for _ in range(args.warmup):
                post_json(base_url + workload["endpoint"], payload, args.timeout)
            samples: list[dict[str, Any]] = []
            output_hashes: list[str] = []
            outputs: list[dict[str, Any]] = []
            for repetition in range(args.repetitions):
                request_offset = log_path.stat().st_size
                request_t0 = time.monotonic()
                response = post_json(
                    base_url + workload["endpoint"], payload, args.timeout
                )
                request_wall_ms = (time.monotonic() - request_t0) * 1000.0
                phase, _ = server_phase_since(log_path, request_offset)
                prompt_tokens, completion_tokens = constrained_usage(response)
                if phase["generation_tokens"] != completion_tokens:
                    raise HarnessError(
                        "phase profile and response usage disagree on completion tokens"
                    )
                try:
                    try:
                        semantic = constrained_semantic_output(
                            workload["kind"], response
                        )
                    except HarnessError:
                        if not (
                            args.jsonschemabench_prefix_steps and
                            workload["kind"] == "json_schema"
                        ):
                            raise
                        semantic = constrained_json_prefix_output(
                            response, completion_tokens,
                            args.jsonschemabench_prefix_steps,
                        )
                        schema_validation = {
                            "status": "PREFIX_ONLY", "valid": None,
                            "validator": workload["source"]["witness_validator"],
                        }
                    else:
                        schema_validation = None
                        if workload["kind"] == "json_schema":
                            schema = payload["response_format"]["json_schema"]["schema"]
                            schema_validation = {
                                "status": "PASS",
                                **validate_json_schema_instance(
                                    schema, semantic["json"]
                                ),
                            }
                except HarnessError:
                    (out_dir / "failure-response.json").write_text(
                        json.dumps({
                            "workload_id": workload["id"],
                            "repetition": repetition,
                            "payload": payload,
                            "response": response,
                            "phase": phase,
                        }, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    raise
                encoded = json.dumps(
                    semantic, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                output_hashes.append(hashlib.sha256(encoded).hexdigest())
                outputs.append(semantic)
                samples.append({
                    **phase,
                    "request_wall_ms": request_wall_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "output_tps": (
                        completion_tokens * 1000.0 / phase["wall_ms"]
                        if phase["wall_ms"] > 0 else 0.0
                    ),
                })
            metric_names = (
                "request_wall_ms", "wall_ms", "output_tps", "constraint_cpu_ms",
                "forced_build_ms", "forced_sync_ms", "filter_setup_ms",
                "filter_ms", "plain_sample_ms", "residual_ms",
                "forced_prefix_probe_ms", "sampling_mask_build_ms",
                "oracle_compare_ms", "filtered_sample_ms", "eval_ms",
                "candidate_tokens_tested", "parser_transition_count",
                "parser_bytes_visited", "trie_nodes_visited",
                "subtrees_pruned", "trie_leaf_tokens_emitted", "mask_cache_hit",
                "mask_cache_miss", "constraint_state_checkpoint",
                "constraint_state_rollback", "exhaustive_fallback_steps",
                "grammar_compile_ms",
                "grammar_jit_ms", "trie_compile_ms", "trie_compiled_nodes",
                "trie_memory_bytes", "static_mask_compile_ms",
                "static_mask_memory_bytes", "dynamic_frontier_size",
            )
            metrics = {
                name: summary([float(sample[name]) for sample in samples])
                for name in metric_names
                if all(name in sample for sample in samples)
            }
            results.append({
                "id": workload["id"], "status": "measured",
                "definition": {
                    "kind": workload["kind"],
                    "endpoint": workload["endpoint"],
                    "weight": workload.get("weight", 1.0),
                    "source": workload.get("source"),
                },
                "metrics": metrics,
                "raw_rows": samples,
                "server_output": {
                    "deterministic": len(set(output_hashes)) == 1,
                    "sha256": output_hashes[0],
                    "semantic": outputs[0],
                },
                "oracle": {
                    "compare_calls": sum(
                        int(sample.get("oracle_compare_calls", 0))
                        for sample in samples
                    ),
                    "divergences": sum(
                        int(sample.get("oracle_divergences", 0))
                        for sample in samples
                    ),
                },
                "schema_validation": schema_validation,
            })
    except Exception:
        (out_dir / "FAILED").write_text(utc_now() + "\n", encoding="utf-8")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_stream is not None:
            log_stream.close()

    baseline_record = (
        None if args.baseline_run else read_json(Path(args.baseline), "baseline experiment")
    )
    deterministic = all(item["server_output"]["deterministic"] for item in results)
    no_divergences = all(item["oracle"]["divergences"] == 0 for item in results)
    schema_outputs_valid = all(
        item["definition"]["kind"] != "json_schema" or
        item.get("schema_validation", {}).get("status") == "PASS"
        for item in results
    )
    prefix_probes_valid = all(
        item["definition"]["kind"] != "json_schema" or
        item.get("schema_validation", {}).get("status") in {"PASS", "PREFIX_ONLY"}
        for item in results
    )
    source_witnesses_valid = constrained_source_witnesses_valid(results)
    schema_contract_valid = (
        prefix_probes_valid
        if args.jsonschemabench_prefix_steps else schema_outputs_valid
    ) and source_witnesses_valid
    unsupported_rejected = (
        not unsupported_probes or
        len(unsupported_results) == len(unsupported_probes)
    )
    unsupported_gate_status = (
        unsupported_rejected if unsupported_probes else None
    )
    baseline_outputs_match = True
    if baseline_record:
        baseline_by_id = {item["id"]: item for item in baseline_record["workloads"]}
        baseline_outputs_match = all(
            baseline_by_id.get(item["id"], {}).get("server_output", {}).get("sha256") ==
            item["server_output"]["sha256"]
            for item in results
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "status": "measured",
        "hypothesis": args.hypothesis,
        "target_metric": "output_tps",
        "suite": suite_name,
        "runtime": "ds4-server-constrained",
        "constraint_mode": args.constraint_mode,
        "repetitions": args.repetitions,
        "warmup_requests": args.warmup,
        "baseline": {
            "kind": "self" if args.baseline_run else "experiment",
            "path": None if args.baseline_run else str(Path(args.baseline).resolve()),
        },
        "provenance": {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
            "binary": str(binary.resolve()),
            "binary_sha256": sha256(binary),
            "model": str(model.resolve()),
            "model_sha256": cached_sha256(model, Path(args.results)),
            "workloads": str(workload_path.resolve()),
            "workloads_sha256": sha256(workload_path),
            "environment_overrides": {
                **dict(item.split("=", 1) for item in args.env),
                "DS4_SERVER_PHASE_PROFILE": "1",
                "DS4_CONSTRAINT_MODE": args.constraint_mode,
            },
            "command": command,
        },
        "hardware": hardware_profile(),
        "workloads": results,
        "unsupported_schema_probes": {
            "requested": len(unsupported_probes),
            "rejected_before_inference": len(unsupported_results),
            "passed": unsupported_gate_status,
            "cases": unsupported_results,
        },
        "correctness": {
            "status": (
                "BASELINE" if args.baseline_run and deterministic and no_divergences and schema_contract_valid and unsupported_rejected else
                "PASS" if deterministic and no_divergences and schema_contract_valid and unsupported_rejected and baseline_outputs_match else
                "FAIL"
            ),
            "deterministic_outputs": deterministic,
            "schema_outputs_valid": schema_outputs_valid,
            "prefix_probes_valid": prefix_probes_valid,
            "source_witnesses_valid": source_witnesses_valid,
            "unsupported_rejected_before_inference": unsupported_gate_status,
            "oracle_divergences": sum(
                item["oracle"]["divergences"] for item in results
            ),
            "baseline_outputs_match": baseline_outputs_match,
        },
    }
    record_path = out_dir / "experiment.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print_compact(record)
    print(f"\nwrote {record_path}", file=sys.stderr)
    if baseline_record:
        comparison = compare_records(baseline_record, record)
        comparison_path = out_dir / "comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(f"verdict: {comparison['verdict']} ({comparison['reason']})")
    return 0


def cmd_constrained_server(args: argparse.Namespace) -> int:
    return benchmark_constrained_server(args)


def metric_median(workload: dict[str, Any], name: str) -> float | None:
    value = workload.get("metrics", {}).get(name, {}).get("median")
    return float(value) if value is not None else None


def compare_records(baseline: dict[str, Any],
                    candidate: dict[str, Any]) -> dict[str, Any]:
    base_by_id = {item["id"]: item for item in baseline["workloads"]}
    rows = []
    for current in candidate["workloads"]:
        previous = base_by_id.get(current["id"])
        if not previous or current.get("status") != "measured" or previous.get("status") != "measured":
            continue
        metrics = {}
        for name in (
            "prefill_tps", "gen_tps", "gen_first_ms", "gen_steady_tps",
            "output_tps", "request_wall_ms", "wall_ms", "constraint_cpu_ms",
            "forced_prefix_probe_ms", "sampling_mask_build_ms",
            "candidate_tokens_tested", "parser_bytes_visited",
            "grammar_compile_ms", "grammar_jit_ms",
        ):
            old, new = metric_median(previous, name), metric_median(current, name)
            if old is None or new is None or old == 0:
                continue
            higher_is_better = name in (
                "prefill_tps", "gen_tps", "gen_steady_tps", "output_tps"
            )
            raw = (new / old - 1.0) * 100.0
            metrics[name] = {
                "baseline": old, "candidate": new, "delta_percent": raw,
                "improvement_percent": raw if higher_is_better else -raw,
            }
        rows.append({"id": current["id"], "metrics": metrics})
    target = candidate.get("target_metric") or baseline.get("target_metric")
    improvements = []
    improvement_weights = []
    dominant_regressions = []
    unstable = False
    for row in rows:
        metric = row["metrics"].get(target) if target else None
        if metric:
            improvements.append(metric["improvement_percent"])
            base_item = base_by_id[row["id"]]
            candidate_item = next(item for item in candidate["workloads"]
                                  if item["id"] == row["id"])
            weight = float(candidate_item.get("definition", {}).get("weight", 1.0))
            improvement_weights.append(weight)
            if weight >= 1.0 and metric["improvement_percent"] <= -2.0:
                dominant_regressions.append({
                    "workload": row["id"],
                    "improvement_percent": metric["improvement_percent"],
                    "weight": weight,
                })
            unstable |= bool(base_item["metrics"][target].get("unstable"))
            unstable |= bool(candidate_item["metrics"][target].get("unstable"))
    weight_sum = sum(improvement_weights)
    weighted_improvement = (
        sum(value * weight for value, weight in zip(improvements, improvement_weights)) /
        weight_sum if improvements and weight_sum else None
    )
    correctness = candidate.get("correctness", {}).get("status", "NOT_VERIFIED")
    context_curve = (
        analyze_context_curve(candidate.get("workloads", []))
        if candidate.get("context_curve") is not None else None
    )
    if correctness == "FAIL":
        verdict = "REJECT_CANDIDATE"
        reason = "candidate failed a correctness gate"
    elif correctness != "PASS":
        verdict = "NEED_MORE_DATA"
        reason = "performance comparison has no passing correctness report"
    elif context_curve and context_curve.get("status") != "PASS":
        verdict = "REJECT_CANDIDATE"
        reason = "context curve misses the throughput floor or has a material recovery"
    elif candidate.get("suite") in {"direction", "long-context-direction", "quick"}:
        verdict = "NEED_MORE_DATA"
        reason = "direction suite is preliminary; confirm with the slow suite"
    elif dominant_regressions:
        verdict = "REJECT_CANDIDATE"
        reason = "one or more dominant workloads regress by at least 2%"
    elif not improvements or unstable:
        verdict = "NEED_MORE_DATA"
        reason = "missing target metric or unstable samples"
    elif weighted_improvement >= 2.0:
        verdict = "KEEP_CANDIDATE"
        reason = "target metric improves by at least 2%; correctness gates remain required"
    elif weighted_improvement <= -2.0:
        verdict = "REJECT_CANDIDATE"
        reason = "target metric regresses by at least 2%"
    else:
        verdict = "NEED_MORE_DATA"
        reason = "difference is inside the default 2% practical threshold"
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": baseline["experiment_id"],
        "candidate": candidate["experiment_id"], "target_metric": target,
        "mean_target_improvement_percent": weighted_improvement,
        "dominant_regressions": dominant_regressions,
        "verdict": verdict, "reason": reason, "workloads": rows,
    }


def compare_experiment(baseline: dict[str, Any],
                       candidate: dict[str, Any]) -> dict[str, Any]:
    return compare_records(baseline, candidate)


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = read_json(Path(args.baseline), "baseline experiment")
    candidate = read_json(Path(args.candidate), "candidate experiment")
    comparison = compare_records(baseline, candidate)
    print(
        f"{comparison['baseline']} -> {comparison['candidate']}: "
        f"{comparison['verdict']} ({comparison['reason']})",
        file=sys.stderr,
    )
    for workload in comparison["workloads"]:
        parts = [
            f"{name} {metric['improvement_percent']:+.2f}%"
            for name, metric in workload["metrics"].items()
        ]
        print(f"  {workload['id']}: " + ", ".join(parts), file=sys.stderr)
    print(json.dumps(comparison, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
    return 0


def cmd_profile_network(args: argparse.Namespace) -> int:
    binary, model, prompt = map(Path, (args.binary, args.model, args.prompt))
    for path, label in ((binary, "benchmark binary"), (model, "model"), (prompt, "prompt")):
        if not path.exists():
            raise HarnessError(f"{label} does not exist: {path}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    command = [
        str(binary), "--model", str(model), "--prompt-file", str(prompt),
        "--ctx-start", str(args.context), "--ctx-max", str(args.context),
        "--ctx-alloc", str(args.context + args.generation_tokens + 1),
        "--prefill-chunk", str(args.prefill_chunk),
        "--gen-tokens", str(args.generation_tokens),
        "--backend", "cuda", "--csv", str(csv_path),
    ]
    env = apply_env_overrides(os.environ, args.env)
    env["DS4_CUDA_QWEN_PROFILE"] = "1"
    if args.generation_tokens:
        env["DS4_CUDA_QWEN_DECODE_PROFILE_POS"] = str(args.context)
    if args.dump_logits:
        logits_dir = Path(args.dump_logits)
        logits_dir.mkdir(parents=True, exist_ok=True)
        command += ["--dump-frontier-logits-dir", str(logits_dir)]
    result = run_command(command, env=env)
    rows = parse_layer_profiles(result.stderr)
    decode_profiles = parse_decode_profiles(result.stderr)
    report = network_profile_report(rows)
    report.update({
        "created_at": utc_now(), "context": args.context,
        "generation_tokens": args.generation_tokens,
        "prefill_chunk": args.prefill_chunk,
        "model": str(model.resolve()), "binary": str(binary.resolve()),
        "environment_overrides": dict(item.split("=", 1) for item in args.env),
        "hardware": hardware_profile(),
        "decode_token_profiles": decode_profiles,
        "decode_token_summary": decode_profiles[-1] if decode_profiles else None,
    })
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"profiled {report['profiled_time_ms']:.3f} ms: "
        f"attention={report['stage_percent']['attention']:.1f}% "
        f"ffn={report['stage_percent']['ffn']:.1f}%"
    )
    for item in report["hotspots"][:args.top]:
        print(
            f"  {item['phase']} layer {item['layer']:02d} "
            f"{item['kind']} {item['stage']}: {item['duration_ms']:.3f} ms "
            f"({item['percent_profiled_time']:.1f}%)"
        )
    if report["decode_token_summary"]:
        detail = report["decode_token_summary"]
        print(
            "  decode token: "
            f"total={detail['total_ms']:.3f} ms "
            f"full-core={detail['full_core_ms']:.3f} ms "
            f"recurrent-attn={detail['recurrent_attn_ms']:.3f} ms "
            f"ffn={detail['ffn_ms']:.3f} ms "
            f"output={detail['output_ms']:.3f} ms "
            f"read={detail['read_ms']:.3f} ms"
        )
    print(f"wrote {output}", file=sys.stderr)
    if args.baseline:
        baseline = read_json(Path(args.baseline), "baseline network profile")
        comparison = compare_network_profiles(baseline, report)
        comparison_path = output.with_name(output.stem + ".comparison.json")
        comparison_path.write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        delta = comparison["total_delta_percent"]
        print(
            f"network delta vs baseline: {delta:+.2f}%"
            if delta is not None else "network delta vs baseline: NOT_VERIFIED"
        )
        for row in comparison["largest_changes"][:args.top]:
            print(
                f"  delta {row['phase']} layer {row['layer']:02d} "
                f"{row['stage']}: {row['delta_percent']:+.2f}% "
                f"({row['delta_ms']:+.3f} ms)"
            )
        print(f"wrote {comparison_path}", file=sys.stderr)
    return 0


def compare_network_profiles(baseline: dict[str, Any],
                             candidate: dict[str, Any]) -> dict[str, Any]:
    def keyed(report: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
        return {
            (row["phase"], row["position"], row["layer"], row["stage"]): row
            for row in report.get("hotspots", [])
        }
    left, right = keyed(baseline), keyed(candidate)
    changes = []
    for key in sorted(left.keys() & right.keys()):
        old, new = left[key]["duration_ms"], right[key]["duration_ms"]
        if old <= 0:
            continue
        changes.append({
            "phase": key[0], "position": key[1], "layer": key[2], "stage": key[3],
            "baseline_ms": old, "candidate_ms": new,
            "delta_ms": new - old, "delta_percent": (new / old - 1.0) * 100.0,
        })
    changes.sort(key=lambda row: abs(row["delta_ms"]), reverse=True)
    old_total = float(baseline.get("profiled_time_ms", 0.0))
    new_total = float(candidate.get("profiled_time_ms", 0.0))
    return {
        "format": "ds4-qwen-network-profile-comparison-v1",
        "status": "COMPARED" if changes else "NOT_VERIFIED",
        "baseline_profiled_ms": old_total, "candidate_profiled_ms": new_total,
        "total_delta_percent": ((new_total / old_total - 1.0) * 100.0
                                if old_total else None),
        "largest_changes": changes,
    }


def cmd_compare_network(args: argparse.Namespace) -> int:
    baseline = read_json(Path(args.baseline), "baseline network profile")
    candidate = read_json(Path(args.candidate), "candidate network profile")
    report = compare_network_profiles(baseline, candidate)
    print(
        f"network profiled time: {report['total_delta_percent']:+.2f}%"
        if report["total_delta_percent"] is not None else
        "network profiles have no comparable total"
    )
    for row in report["largest_changes"][:args.top]:
        print(
            f"  {row['phase']} layer {row['layer']:02d} {row['stage']}: "
            f"{row['delta_percent']:+.2f}% ({row['delta_ms']:+.3f} ms)"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "COMPARED" else 1


def cmd_drift(args: argparse.Namespace) -> int:
    report = logits_drift(Path(args.baseline), Path(args.candidate), args.top_k)
    print(
        f"{report['status']}: argmax_equal={report['argmax']['equal']} "
        f"top{report['top_k']}_overlap={report['top_k_overlap']:.3f} "
        f"mae={report['centered_mae']:.6g} cosine={report['cosine_similarity']:.9f}"
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


def cmd_q8_1_parity(args: argparse.Namespace) -> int:
    report = q8_1_parity(Path(args.left), Path(args.right))
    consumed = report["mmvq_consumed_fields"]
    metadata = report["metadata_ds_y"]
    print(
        f"{report['status']}: blocks={report['blocks']} "
        f"ds.x_diff={consumed['scale_ds_x_differing_blocks']} "
        f"qs_diff={consumed['qs_differing_bytes']} "
        f"qsum_diff={consumed['qsum_differing_blocks']} "
        f"ds.y_diff={metadata['differing_blocks']} (not consumed by decode MMVQ)"
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n",
                                     encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


def cmd_qwen_logits_row(args: argparse.Namespace) -> int:
    runs = []
    for item in args.run:
        if "=" not in item:
            raise HarnessError(f"Qwen run must be LABEL=PATH: {item}")
        label, path = item.split("=", 1)
        if not label or not path:
            raise HarnessError(f"Qwen run must be LABEL=PATH: {item}")
        runs.append((label, Path(path)))
    report = qwen_logits_row_report(
        runs, args.case, args.stream, args.row, args.vocab,
        args.top_k, args.focus_token,
    )
    for run in report["runs"]:
        suffix = (f" margin={run['focus_margin']:+.9g}"
                  if "focus_margin" in run else "")
        print(f"{run['label']}: argmax={run['argmax']}{suffix}")
    for pair in report["comparisons"]:
        print(
            f"  {pair['reference']} vs {pair['candidate']}: "
            f"mae={pair['centered_mae']:.6g} "
            f"cosine={pair['cosine_similarity']:.9f} "
            f"top{pair['top_k']}={pair['top_k_overlap']:.3f}"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n",
                                     encoding="utf-8")
    return 0


def cmd_qwen_argmax_gate(args: argparse.Namespace) -> int:
    report = qwen_argmax_gate(Path(args.reference), Path(args.candidate))
    sequences, argmax = report["sequences"], report["argmax"]
    print(
        f"{report['status']}: sequences={sequences['equal']}/{sequences['total']} "
        f"argmax={argmax['equal']}/{argmax['total']} "
        f"(greedy={argmax['greedy_equal']}/{argmax['greedy_total']}, "
        f"teacher={argmax['teacher_equal']}/{argmax['teacher_total']})"
    )
    for case in report["cases"]:
        if not case["sequence_equal"] or (
                case["teacher_argmax_equal"] != case["teacher_argmax_total"]):
            print(
                f"  {case['id']}: greedy_first={case['first_greedy_difference']} "
                f"teacher_first={case['first_teacher_difference']}"
            )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n",
                                     encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


def cmd_model_cost(args: argparse.Namespace) -> int:
    snapshot = (
        inspect_model_snapshot(Path(args.model))
        if args.model else
        read_json(Path(args.snapshot), "GGUF metadata snapshot")
    )
    hardware_source = "command line override"
    memory_gbps, compute_tflops = args.memory_gbps, args.compute_tflops
    if memory_gbps is None or compute_tflops is None:
        detected = hardware_probe()
        devices = detected.get("gpu", {}).get("devices", [])
        spec = devices[0].get("static_spec", {}) if devices else {}
        memory_gbps = memory_gbps or spec.get("theoretical_memory_bandwidth_gbps", 936.2)
        compute_tflops = compute_tflops or spec.get("nominal_fp32_compute_tflops", 35.58)
        hardware_source = (
            "detected GPU static specification" if spec.get("spec_source") else
            "RTX 3090 fallback; no known GPU specification detected"
        )
    report = model_cost(
        snapshot, phase=args.phase, context=args.context, batch=args.batch,
        activation_bytes=args.activation_bytes,
        memory_gbps=memory_gbps, compute_tflops=compute_tflops,
    )
    report["hardware_assumption"]["source"] = hardware_source
    if args.network_profile:
        profile = read_json(Path(args.network_profile), "network profile")
        report = attach_observed_profile(report, profile)
    print(
        f"{args.phase} context={args.context} batch={args.batch}: "
        f"KV={report['state']['full_attention_kv_bytes'] / 2**20:.1f} MiB, "
        f"GDN state={report['state']['gdn_recurrent_state_bytes'] / 2**20:.1f} MiB"
    )
    for name, operation in report["operations"].items():
        observed = operation.get("observed_profile_ms")
        suffix = (
            f", observed={observed:.3f} ms, "
            f"observed/floor={operation['observed_over_roofline_floor']:.1f}x"
            if observed is not None else ""
        )
        print(
            f"  {name}: {operation['predicted_bound']}-bound, "
            f"AI={operation['arithmetic_intensity_flop_per_byte']:.3f} flop/B, "
            f"floor={operation['roofline_floor_ms']:.3f} ms{suffix}"
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)
    return 0


def workflow_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env["CUDA_ARCH"] = args.cuda_arch
    if args.name == "validate":
        return [str(ROOT / "tools" / "perf-qwen-validate.sh")], env
    if args.name.startswith("r8-"):
        action = args.name.removeprefix("r8-")
        if action != "build" and not args.id:
            raise HarnessError("R8 benchmark workflows require --id")
        env["PERF_MODEL"] = str(Path(args.model))
        env["PERF_PROMPT"] = str(Path(args.prompt))
        env["PERF_RESULTS"] = str(Path(args.results))
        command = [str(ROOT / "tools" / "perf-qwen-r8.sh"), action]
        if args.id:
            command.append(args.id)
        for item in args.candidate_env:
            if "=" not in item:
                raise HarnessError(
                    f"candidate environment must be NAME=VALUE: {item}"
                )
            command.extend(("--env", item))
        return command, env
    if not args.id:
        raise HarnessError("long-context workflows require --id")
    env["PERF_MODEL"] = str(Path(args.model))
    env["PERF_PROMPT"] = str(Path(args.prompt))
    env["PERF_RESULTS"] = str(Path(args.results))
    action = args.name.removeprefix("long-context-")
    command = [
        str(ROOT / "tools" / "perf-qwen-long-context.sh"),
        action,
        args.id,
    ]
    for item in args.candidate_env:
        if "=" not in item:
            raise HarnessError(f"candidate environment must be NAME=VALUE: {item}")
        command.extend(("--env", item))
    return command, env


def cmd_workflow(args: argparse.Namespace) -> int:
    command, env = workflow_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def print_compact(record: dict[str, Any]) -> None:
    print(f"experiment {record['experiment_id']} ({record['suite']})")
    for workload in record["workloads"]:
        if workload["status"] != "measured":
            print(f"  {workload['id']}: {workload['status']} ({workload['reason']})")
            continue
        fields = []
        for name in (
            "prefill_tps", "gen_steady_tps", "output_tps",
            "gen_first_ms", "constraint_cpu_ms",
        ):
            metric = workload["metrics"].get(name)
            if metric:
                suffix = " ms" if name.endswith("_ms") else " tok/s"
                unstable = " unstable" if metric["unstable"] else ""
                fields.append(f"{name}={metric['median']:.2f}{suffix}{unstable}")
        print(f"  {workload['id']}: " + ", ".join(fields))
    curve = record.get("context_curve")
    if curve:
        print(
            f"  context curve: {curve['status']} "
            f"min={curve['observed_min_tps']:.2f} tok/s "
            f"floor={curve['minimum_tps']:.2f} tok/s "
            f"material_recoveries={len(curve['material_recoveries'])}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DS4 GPU performance experiment harness")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="capture a hardware profile")
    probe.add_argument("--output")
    probe.set_defaults(func=cmd_probe)
    doctor = commands.add_parser("doctor", help="show available capabilities")
    doctor.set_defaults(func=cmd_doctor)
    run = commands.add_parser("run", help="run a suite and record an experiment")
    run.add_argument("--id")
    run.add_argument("--model", required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--binary", default=str(ROOT / "ds4-bench"))
    run.add_argument("--workloads", default=str(DEFAULT_WORKLOADS))
    run.add_argument("--suite", default="direction",
                     choices=("direction", "r8-slow",
                              "long-context-direction", "quick",
                              "standard", "slow", "long-context-slow",
                              CONTEXT_CURVE_SUITE, "exhaustive"))
    run.add_argument("--repetitions", type=int, default=2)
    run.add_argument("--warmup", choices=("auto", "always", "never"), default="auto")
    run.add_argument("--hypothesis", required=True)
    run.add_argument("--metric", required=True)
    baseline = run.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-run", action="store_true",
                          help="declare this experiment as the frozen baseline")
    baseline.add_argument("--baseline",
                          help="baseline experiment.json; compare immediately")
    run.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    run.add_argument("--results", default=str(DEFAULT_RESULTS))
    run.set_defaults(func=cmd_run)
    server_curve = commands.add_parser(
        "server-curve",
        help="start ds4-server and measure a configured decode curve",
    )
    server_curve.add_argument("--id")
    server_curve.add_argument("--model", required=True)
    server_curve.add_argument("--binary", default=str(ROOT / "ds4-server"))
    server_curve.add_argument("--workloads", default=str(DEFAULT_WORKLOADS))
    server_curve.add_argument(
        "--suite", default=CONTEXT_CURVE_SUITE,
        choices=(CONTEXT_CURVE_SUITE, MTP_CONTEXT_CURVE_SUITE,
                 MTP_SHORT_REGRESSION_SUITE,
                 MTP_DEPTH_CROSSOVER_SUITE, MTP_LONG_CONTEXT_SMOKE_SUITE,
                 MTP_DEPTH_2K_SUITE, MTP_DEPTH_28K_SUITE,
                 MTP_DEPTH_BOUNDARY_SUITE,
                 MTP_WEAKEST_CONFIRM_SUITE,
                 MTP_THRESHOLD_SEARCH_SUITE, MTP_THRESHOLD_MIDPOINT_SUITE,
                 AGENT_DSML_BASELINE_SUITE),
    )
    server_curve.add_argument("--mtp", action="store_true")
    server_curve.add_argument("--mtp-model")
    server_curve.add_argument(
        "--prompt-pattern",
        choices=("numbered-list", "technical-explanation"),
        default="numbered-list",
        help="calibrated completion canary; technical-explanation resists early EOS",
    )
    server_thinking = server_curve.add_mutually_exclusive_group()
    server_thinking.add_argument(
        "--thinking", dest="thinking", action="store_true",
        help="explicitly enable thinking in completion requests",
    )
    server_thinking.add_argument(
        "--no-thinking", dest="thinking", action="store_false",
        help="explicitly disable thinking in completion requests",
    )
    server_curve.set_defaults(thinking=None)
    server_curve.add_argument("--repetitions", type=int, default=2)
    server_curve.add_argument("--minimum-completion-tokens", type=int, default=32)
    server_curve.add_argument(
        "--context-alloc", type=int,
        help="allocate this server context independently of the measured prompt length",
    )
    server_curve.add_argument("--port", type=int, default=0)
    server_curve.add_argument("--startup-timeout", type=float, default=180.0)
    server_curve.add_argument("--timeout", type=float, default=1800.0)
    server_curve.add_argument("--hypothesis", required=True)
    server_baseline = server_curve.add_mutually_exclusive_group(required=True)
    server_baseline.add_argument("--baseline-run", action="store_true")
    server_baseline.add_argument("--baseline")
    server_curve.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    server_curve.add_argument("--results", default=str(DEFAULT_RESULTS))
    server_curve.set_defaults(func=cmd_server_curve)
    constrained = commands.add_parser(
        "constrained-server",
        help="start ds4-server and benchmark tokenizer-exact DSML/JSON constraints",
    )
    constrained.add_argument("--id")
    constrained.add_argument("--model", required=True)
    constrained.add_argument("--binary", default=str(ROOT / "ds4-server"))
    constrained.add_argument(
        "--workloads", default=str(DEFAULT_CONSTRAINED_WORKLOADS)
    )
    constrained.add_argument(
        "--jsonschemabench-subset",
        nargs="?",
        const=str(DEFAULT_JSONSCHEMABENCH_SUBSET),
        help=(
            "benchmark the supported entries from the pinned external subset; "
            "optionally provide another subset path"
        ),
    )
    constrained.add_argument(
        "--jsonschemabench-id",
        action="append",
        help="run only this supported external schema id (repeatable)",
    )
    constrained.add_argument(
        "--jsonschemabench-tier", choices=("smoke", "safety", "regressions"),
        default="safety",
        help="pinned external tier used when no explicit schema id is supplied",
    )
    constrained.add_argument(
        "--jsonschemabench-prefix-steps", type=int, default=0,
        help=(
            "run a bounded incomplete-prefix differential gate; 1..64 and "
            "compare_new_vs_oracle are required"
        ),
    )
    constrained.add_argument(
        "--jsonschemabench-check-unsupported", action="store_true",
        help=(
            "require every pinned unsupported example to fail with HTTP 400 "
            "before inference"
        ),
    )
    constrained.add_argument("--suite", default="direction")
    constrained.add_argument(
        "--constraint-mode",
        choices=(
            "oracle_only", "incremental", "trie", "optimized",
            "compare_new_vs_oracle",
        ),
        default="trie",
    )
    constrained.add_argument("--context", type=int, default=4096)
    constrained.add_argument("--prefill-chunk", type=int, default=512)
    constrained.add_argument("--repetitions", type=int, default=2)
    constrained.add_argument("--warmup", type=int, default=0)
    constrained.add_argument("--port", type=int, default=0)
    constrained.add_argument("--startup-timeout", type=float, default=180.0)
    constrained.add_argument("--timeout", type=float, default=1800.0)
    constrained.add_argument("--hypothesis", required=True)
    constrained_baseline = constrained.add_mutually_exclusive_group(required=True)
    constrained_baseline.add_argument("--baseline-run", action="store_true")
    constrained_baseline.add_argument("--baseline")
    constrained.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    constrained.add_argument("--results", default=str(DEFAULT_RESULTS))
    constrained.set_defaults(func=cmd_constrained_server)
    compare = commands.add_parser("compare", help="compare experiment records")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--output")
    compare.set_defaults(func=cmd_compare)
    profile = commands.add_parser(
        "profile-network", help="measure Qwen attention/FFN cost per layer"
    )
    profile.add_argument("--model", required=True)
    profile.add_argument("--prompt", required=True)
    profile.add_argument("--binary", default=str(ROOT / "ds4-bench"))
    profile.add_argument("--context", type=int, default=1024)
    profile.add_argument("--generation-tokens", type=int, default=2)
    profile.add_argument("--prefill-chunk", type=int, default=512)
    profile.add_argument("--top", type=int, default=10)
    profile.add_argument("--dump-logits")
    profile.add_argument("--env", action="append", default=[],
                         metavar="NAME=VALUE")
    profile.add_argument("--baseline",
                         help="baseline network JSON; compare immediately")
    profile.add_argument("--output", required=True)
    profile.set_defaults(func=cmd_profile_network)
    network_compare = commands.add_parser(
        "compare-network", help="show which network stages changed"
    )
    network_compare.add_argument("baseline")
    network_compare.add_argument("candidate")
    network_compare.add_argument("--top", type=int, default=12)
    network_compare.add_argument("--output")
    network_compare.set_defaults(func=cmd_compare_network)
    drift = commands.add_parser(
        "drift", help="compare two ds4-bench frontier logits files"
    )
    drift.add_argument("baseline")
    drift.add_argument("candidate")
    drift.add_argument("--top-k", type=int, default=20)
    drift.add_argument("--output")
    drift.set_defaults(func=cmd_drift)
    q8_parity = commands.add_parser(
        "q8-1-parity",
        help="compare packed Q8_1 MMVQ fields while auditing ds.y separately",
    )
    q8_parity.add_argument("left")
    q8_parity.add_argument("right")
    q8_parity.add_argument("--output")
    q8_parity.set_defaults(func=cmd_q8_1_parity)
    qwen_row = commands.add_parser(
        "qwen-logits-row",
        help="compare one full-vocabulary row from labeled Qwen oracle runs",
    )
    qwen_row.add_argument("--run", action="append", required=True,
                          metavar="LABEL=PATH")
    qwen_row.add_argument("--case", required=True)
    qwen_row.add_argument("--stream", choices=("greedy", "teacher"),
                          default="greedy")
    qwen_row.add_argument("--row", type=int, required=True)
    qwen_row.add_argument("--vocab", type=int, default=248320)
    qwen_row.add_argument("--top-k", type=int, default=20)
    qwen_row.add_argument("--focus-token", action="append", type=int,
                          default=[])
    qwen_row.add_argument("--output")
    qwen_row.set_defaults(func=cmd_qwen_logits_row)
    qwen_gate = commands.add_parser(
        "qwen-argmax-gate",
        help="gate Qwen oracle suites on sequences and greedy/teacher argmax",
    )
    qwen_gate.add_argument("reference")
    qwen_gate.add_argument("candidate")
    qwen_gate.add_argument("--output")
    qwen_gate.set_defaults(func=cmd_qwen_argmax_gate)
    cost = commands.add_parser(
        "model-cost", help="estimate Qwen FLOP, bytes, state and roofline floors"
    )
    cost_source = cost.add_mutually_exclusive_group(required=True)
    cost_source.add_argument("--snapshot")
    cost_source.add_argument("--model",
                             help="inspect the GGUF header directly without hashing weights")
    cost.add_argument("--phase", choices=("prefill", "decode"), required=True)
    cost.add_argument("--context", type=int, required=True)
    cost.add_argument("--batch", type=int, default=1)
    cost.add_argument("--activation-bytes", type=int, default=4, choices=(2, 4))
    cost.add_argument("--memory-gbps", type=float)
    cost.add_argument("--compute-tflops", type=float)
    cost.add_argument("--network-profile")
    cost.add_argument("--output")
    cost.set_defaults(func=cmd_model_cost)
    workflow = commands.add_parser(
        "workflow", help="run a documented Qwen build/profile/benchmark command list"
    )
    workflow.add_argument(
        "--name", required=True,
        choices=("validate", "r8-build", "r8-direction", "r8-slow", "r8-long",
                 "long-context-profile",
                 "long-context-direction", "long-context-slow"),
    )
    workflow.add_argument("--id")
    workflow.add_argument("--model", default=str(ROOT / "gguf" / "Qwen3.6-27B-Q4_K_S.gguf"))
    workflow.add_argument("--prompt", default=str(ROOT / "tests" / "long_context_story_prompt.txt"))
    workflow.add_argument("--results", default=str(DEFAULT_RESULTS))
    workflow.add_argument("--cuda-arch", default="sm_86")
    workflow.add_argument("--candidate-env", action="append", default=[],
                          metavar="NAME=VALUE")
    workflow.add_argument("--dry-run", action="store_true")
    workflow.set_defaults(func=cmd_workflow)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if getattr(args, "repetitions", 2) < 2:
            raise HarnessError("at least two repetitions are required")
        return args.func(args)
    except (HarnessError, OSError) as exc:
        print(f"perf-harness: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
