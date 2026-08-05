#!/usr/bin/env python3
"""Triangulate DS4 traces against llama.cpp CPU and CUDA traces.

The CPU/CUDA disagreement is treated as an empirical floating-point envelope.
DS4 values outside that envelope are suspicious, but not automatically called an
implementation defect: an upstream arithmetic-policy difference (for example
Q4_K x F32 versus Q4_K x Q8_K) propagates to every later tensor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from compare_qwen36_trace import STAGE_MAP, metrics, trace_path
from qwen36_fixtures import FixtureError, write_json


def canonicalize(stage: str, engine: str, values: np.ndarray) -> np.ndarray:
    """Put engine-specific storage layouts into the DS4 logical layout."""
    if stage == "recurrent_state" and engine.startswith("llama"):
        head_values = 48 * 128 * 128
        if values.size == head_values:
            # llama.cpp stores [value_head, value_dim, key_dim]; DS4 stores
            # [value_head, key_dim, value_dim].
            return values.reshape(48, 128, 128).transpose(0, 2, 1).reshape(-1)
    return values


def envelope_metrics(ds4: np.ndarray, cpu: np.ndarray, cuda: np.ndarray,
                     factor: float, abs_floor: float, ulp_factor: float) -> dict:
    if ds4.shape != cpu.shape or ds4.shape != cuda.shape:
        raise FixtureError(
            f"trace shape mismatch: ds4={ds4.shape}, cpu={cpu.shape}, cuda={cuda.shape}"
        )
    finite = np.isfinite(ds4) & np.isfinite(cpu) & np.isfinite(cuda)
    if not np.all(finite):
        return {
            "classification": "invalid_nonfinite",
            "nonfinite": {
                "ds4": int(np.count_nonzero(~np.isfinite(ds4))),
                "llama_cpu": int(np.count_nonzero(~np.isfinite(cpu))),
                "llama_cuda": int(np.count_nonzero(~np.isfinite(cuda))),
            },
        }

    d64, c64, g64 = (x.astype(np.float64) for x in (ds4, cpu, cuda))
    backend_delta = np.abs(c64 - g64)
    nearest_delta = np.minimum(np.abs(d64 - c64), np.abs(d64 - g64))
    ulp = np.maximum(np.abs(np.spacing(cpu)), np.abs(np.spacing(cuda))).astype(np.float64)
    allowed = abs_floor + factor * backend_delta + ulp_factor * ulp
    outside = nearest_delta > allowed
    backend_rmse = float(np.sqrt(np.mean((c64 - g64) ** 2)))
    nearest_rmse = float(np.sqrt(np.mean(nearest_delta ** 2)))
    outside_count = int(np.count_nonzero(outside))
    outside_fraction = float(np.mean(outside))
    globally_inside = nearest_rmse <= abs_floor + factor * backend_rmse
    if not np.count_nonzero(nearest_delta):
        classification = "exact_reference_match"
    elif not outside_count or globally_inside:
        classification = "within_cpu_cuda_roundoff_envelope"
    else:
        classification = "suspicious_outside_backend_envelope"
    return {
        "classification": classification,
        "outside_values": outside_count,
        "outside_fraction": outside_fraction,
        "max_envelope_excess": float(np.max(np.maximum(nearest_delta - allowed, 0.0))),
        "nearest_reference_rmse": nearest_rmse,
        "cpu_cuda_rmse": backend_rmse,
        "rmse_envelope_ratio": (nearest_rmse / max(backend_rmse, abs_floor)),
        "ds4_vs_llama_cpu": metrics(ds4, cpu),
        "ds4_vs_llama_cuda": metrics(ds4, cuda),
        "llama_cpu_vs_cuda": metrics(cpu, cuda),
    }


def _llama_stage(root: Path, position: int, layer: int,
                 alternatives: str | tuple[str, ...]) -> tuple[str, Path]:
    names = (alternatives,) if isinstance(alternatives, str) else alternatives
    for name in names:
        candidate = trace_path(root, "llama", position, layer, name)
        if candidate.is_file():
            return name, candidate
    return names[0], trace_path(root, "llama", position, layer, names[0])


def diagnose(ds4_dir: Path, cpu_dir: Path, cuda_dir: Path, position: int,
             layers: list[int], factor: float, abs_floor: float,
             ulp_factor: float) -> tuple[dict, int]:
    rows: list[dict] = []
    for layer in layers:
        for ds4_stage, alternatives in STAGE_MAP.items():
            ds4_file = trace_path(ds4_dir, "ds4", position, layer, ds4_stage)
            cpu_stage, cpu_file = _llama_stage(cpu_dir, position, layer, alternatives)
            cuda_stage, cuda_file = _llama_stage(cuda_dir, position, layer, alternatives)
            if not (ds4_file.is_file() and cpu_file.is_file() and cuda_file.is_file()):
                continue
            if cpu_stage != cuda_stage:
                continue
            values = {
                "ds4": canonicalize(ds4_stage, "ds4", np.fromfile(ds4_file, dtype="<f4")),
                "cpu": canonicalize(ds4_stage, "llama_cpu", np.fromfile(cpu_file, dtype="<f4")),
                "cuda": canonicalize(ds4_stage, "llama_cuda", np.fromfile(cuda_file, dtype="<f4")),
            }
            rows.append({
                "layer": layer,
                "stage": ds4_stage,
                "llama_stage": cpu_stage,
                **envelope_metrics(values["ds4"], values["cpu"], values["cuda"],
                                   factor, abs_floor, ulp_factor),
            })
    if not rows:
        raise FixtureError("no stages shared by DS4, llama.cpp CPU, and llama.cpp CUDA")

    invalid = [r for r in rows if r["classification"] == "invalid_nonfinite"]
    suspicious = [r for r in rows
                  if r["classification"] == "suspicious_outside_backend_envelope"]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    report = {
        "format": "ds4-qwen36-numerics-envelope-v1",
        "status": "FAIL" if invalid else "DIAGNOSTIC",
        "interpretation": (
            "Outside the CPU/CUDA envelope identifies the first arithmetic divergence; "
            "it is not by itself proof of a kernel defect because upstream arithmetic "
            "policies can propagate. Pair this report with the isolated kernel probes."
        ),
        "position": position,
        "layers": layers,
        "envelope": {"factor": factor, "absolute_floor": abs_floor,
                     "ulp_factor": ulp_factor},
        "classification_counts": counts,
        "first_suspicious_divergence": suspicious[0] if suspicious else None,
        "stages": rows,
    }
    return report, 1 if invalid else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds4-dir", required=True, type=Path)
    parser.add_argument("--llama-cpu-dir", required=True, type=Path)
    parser.add_argument("--llama-cuda-dir", required=True, type=Path)
    parser.add_argument("--position", required=True, type=int)
    parser.add_argument("--layer", action="append", required=True, type=int, dest="layers")
    parser.add_argument("--envelope-factor", type=float, default=2.0)
    parser.add_argument("--absolute-floor", type=float, default=1e-7)
    parser.add_argument("--ulp-factor", type=float, default=8.0)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.position < 0 or any(layer < 0 for layer in args.layers):
            raise FixtureError("position and layers must be non-negative")
        if args.envelope_factor < 1 or args.absolute_floor < 0 or args.ulp_factor < 0:
            raise FixtureError("envelope factor must be >= 1; floor and ULP factor must be >= 0")
        report, code = diagnose(
            args.ds4_dir, args.llama_cpu_dir, args.llama_cuda_dir, args.position,
            sorted(set(args.layers)), args.envelope_factor, args.absolute_floor,
            args.ulp_factor,
        )
        write_json(args.report, report)
        print(json.dumps({
            "status": report["status"],
            "classification_counts": report["classification_counts"],
            "first_suspicious_divergence": report["first_suspicious_divergence"],
        }, sort_keys=True))
        return code
    except (FixtureError, OSError, ValueError) as exc:
        report = {"format": "ds4-qwen36-numerics-envelope-v1",
                  "status": "ERROR", "reason": str(exc)}
        write_json(args.report, report)
        print(json.dumps(report, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
