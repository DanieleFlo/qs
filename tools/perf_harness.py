#!/usr/bin/env python3
"""Dependency-free performance experiment harness for DS4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOADS = ROOT / "performance" / "workloads.yaml"
DEFAULT_RESULTS = ROOT / "performance-results"
SCHEMA_VERSION = 1
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


def cmd_doctor(_args: argparse.Namespace) -> int:
    profile = hardware_profile()
    checks = {
        "ds4_bench": (ROOT / "ds4-bench").exists(),
        "nvidia_smi": profile["gpu"]["available"],
        "nsight_systems": profile["tools"]["nsys"]["available"],
        "nsight_compute": profile["tools"]["ncu"]["available"],
    }
    print(json.dumps({
        "ready_for_benchmark": checks["ds4_bench"] and checks["nvidia_smi"],
        "checks": checks, "hardware": profile,
    }, indent=2))
    return 0 if checks["ds4_bench"] else 1


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
        if args.suite == "direction":
            contexts = sorted(int(item["context"]) for item in workloads)
            common = ("generation_tokens", "prefill_chunk", "backend", "batch")
            if any(
                item.get(key, 1 if key == "batch" else None) !=
                workloads[0].get(key, 1 if key == "batch" else None)
                for item in workloads[1:] for key in common
            ):
                raise HarnessError(
                    "direction workloads must share generation/prefill/backend/batch"
                )
            if len(contexts) != 2:
                raise HarnessError("direction suite must contain exactly two contexts")
            sweep = dict(
                workloads[0], context=contexts[0], context_max=contexts[1],
                step_incr=contexts[1] - contexts[0],
            )
            sweep_logits = out_dir / "direction-sweep-logits"
            rows = bench_once(
                binary, model, prompt, sweep, env,
                out_dir / "direction-sweep.csv", sweep_logits,
                repetitions=args.repetitions,
            )
            for workload in workloads:
                context = int(workload["context"])
                samples = [row for row in rows if int(row["ctx_tokens"]) == context]
                if len(samples) != args.repetitions:
                    raise HarnessError(
                        f"direction sweep emitted {len(samples)} samples for context {context}; "
                        f"expected {args.repetitions}"
                    )
                frontier_name = f"frontier_{context:06d}.logits.json"
                candidate_logits_dir = out_dir / "logits" / workload["id"]
                candidate_logits_dir.mkdir(parents=True, exist_ok=True)
                candidate_logits = candidate_logits_dir / frontier_name
                shutil.copy2(sweep_logits / frontier_name, candidate_logits)
                if args.baseline_run:
                    correctness_rows.append({
                        "workload": workload["id"], "status": "BASELINE",
                        "candidate_logits": str(candidate_logits),
                    })
                else:
                    baseline_logits = baseline_logits_root / workload["id"] / frontier_name
                    if baseline_logits.is_file():
                        drift = logits_drift(baseline_logits, candidate_logits)
                        drift["workload"] = workload["id"]
                        correctness_rows.append(drift)
                    else:
                        correctness_rows.append({
                            "workload": workload["id"], "status": "NOT_VERIFIED",
                            "reason": "matching baseline frontier logits are missing",
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
            frontier_name = f"frontier_{workload['context']:06d}.logits.json"
            candidate_logits = candidate_logits_dir / frontier_name
            if args.baseline_run:
                correctness_rows.append({
                    "workload": workload["id"], "status": "BASELINE",
                    "candidate_logits": str(candidate_logits),
                })
            else:
                baseline_logits = baseline_logits_root / workload["id"] / frontier_name
                if baseline_logits.is_file() and candidate_logits.is_file():
                    drift = logits_drift(baseline_logits, candidate_logits)
                    drift["workload"] = workload["id"]
                    correctness_rows.append(drift)
                else:
                    correctness_rows.append({
                        "workload": workload["id"], "status": "NOT_VERIFIED",
                        "reason": "matching baseline frontier logits are missing",
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
            "frontier_logits": correctness_rows,
        },
    }
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
        for name in ("prefill_tps", "gen_tps", "gen_first_ms", "gen_steady_tps"):
            old, new = metric_median(previous, name), metric_median(current, name)
            if old is None or new is None or old == 0:
                continue
            higher_is_better = name != "gen_first_ms"
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
    if correctness == "FAIL":
        verdict = "REJECT_CANDIDATE"
        reason = "frontier logits drift exceeded a correctness gate"
    elif correctness != "PASS":
        verdict = "NEED_MORE_DATA"
        reason = "performance comparison has no passing correctness report"
    elif candidate.get("suite") in {"direction", "quick"}:
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
    report = network_profile_report(rows)
    report.update({
        "created_at": utc_now(), "context": args.context,
        "generation_tokens": args.generation_tokens,
        "prefill_chunk": args.prefill_chunk,
        "model": str(model.resolve()), "binary": str(binary.resolve()),
        "environment_overrides": dict(item.split("=", 1) for item in args.env),
        "hardware": hardware_profile(),
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


def print_compact(record: dict[str, Any]) -> None:
    print(f"experiment {record['experiment_id']} ({record['suite']})")
    for workload in record["workloads"]:
        if workload["status"] != "measured":
            print(f"  {workload['id']}: {workload['status']} ({workload['reason']})")
            continue
        fields = []
        for name in ("prefill_tps", "gen_steady_tps", "gen_first_ms"):
            metric = workload["metrics"].get(name)
            if metric:
                suffix = " ms" if name.endswith("_ms") else " tok/s"
                unstable = " unstable" if metric["unstable"] else ""
                fields.append(f"{name}={metric['median']:.2f}{suffix}{unstable}")
        print(f"  {workload['id']}: " + ", ".join(fields))


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
                     choices=("direction", "quick", "standard", "slow", "exhaustive"))
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
