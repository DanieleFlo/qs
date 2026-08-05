#!/usr/bin/env python3
"""Compare matching DS4 and llama.cpp Qwen activation trace files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from qwen36_fixtures import FixtureError, write_json


STAGE_MAP = {
    "embed": "model.input_embed",
    "attn_norm": "attn_norm",
    "qkv": "linear_attn_qkv_mixed",
    "z": "z",
    "alpha": "alpha",
    "beta": "beta",
    "conv_silu": "conv_output_silu",
    "heads": "final_output",
    "recurrent_state": "new_state",
    "attn_out": ("linear_attn_out", "attn_output"),
    "after_attn": "attn_residual",
    "ffn_norm": "attn_post_norm",
    "ffn_out": "ffn_out",
    "layer_out": ("post_ffn", "l_out"),
    "output_norm": "result_norm",
    "logits": "result_output",
}


def metrics(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        raise FixtureError(f"trace shape mismatch: {left.shape} != {right.shape}")
    finite = np.isfinite(left) & np.isfinite(right)
    different = int(np.count_nonzero(left.view(np.uint32) != right.view(np.uint32)))
    result = {
        "values": int(left.size),
        "different_float_count": different,
        "left_nonfinite": int(np.count_nonzero(~np.isfinite(left))),
        "right_nonfinite": int(np.count_nonzero(~np.isfinite(right))),
    }
    if not np.all(finite):
        result.update({"mae": None, "rmse": None, "max_error": None,
                       "cosine_similarity": None})
        return result
    delta = left.astype(np.float64) - right.astype(np.float64)
    denom = float(np.linalg.norm(left.astype(np.float64)) *
                  np.linalg.norm(right.astype(np.float64)))
    result.update({
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_error": float(np.max(np.abs(delta))),
        "cosine_similarity": float(np.dot(left.astype(np.float64), right.astype(np.float64)) / denom)
        if denom else (1.0 if different == 0 else 0.0),
    })
    return result


def trace_path(root: Path, engine: str, position: int, layer: int, stage: str) -> Path:
    return root / f"{engine}-pos{position}-layer{layer}-{stage}.f32"


def compare_trace(ds4_dir: Path, llama_dir: Path, position: int,
                  layers: list[int]) -> tuple[dict, int]:
    rows = []
    first_divergence = None
    for layer in layers:
        for ds4_stage, llama_stages in STAGE_MAP.items():
            if isinstance(llama_stages, str):
                llama_stages = (llama_stages,)
            ds4_file = trace_path(ds4_dir, "ds4", position, layer, ds4_stage)
            llama_stage = next(
                (stage for stage in llama_stages
                 if trace_path(llama_dir, "llama", position, layer, stage).is_file()),
                llama_stages[0],
            )
            llama_file = trace_path(llama_dir, "llama", position, layer, llama_stage)
            if not ds4_file.is_file() or not llama_file.is_file():
                continue
            left = np.fromfile(ds4_file, dtype="<f4")
            right = np.fromfile(llama_file, dtype="<f4")
            row = {"layer": layer, "ds4_stage": ds4_stage,
                   "llama_stage": llama_stage, **metrics(left, right)}
            rows.append(row)
            if first_divergence is None and row["different_float_count"]:
                first_divergence = row
    if not rows:
        raise FixtureError("no matching DS4/llama.cpp trace stages were found")
    nonfinite = any(row["left_nonfinite"] or row["right_nonfinite"] for row in rows)
    report = {
        "format": "ds4-qwen36-trace-comparison-v1",
        "status": "FAIL" if nonfinite else "DIAGNOSTIC",
        "position": position,
        "layers": layers,
        "stages": rows,
        "first_float_divergence": first_divergence,
    }
    return report, 1 if nonfinite else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds4-dir", required=True, type=Path)
    parser.add_argument("--llama-dir", required=True, type=Path)
    parser.add_argument("--position", required=True, type=int)
    parser.add_argument("--layer", action="append", required=True, type=int, dest="layers")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.position < 0 or any(layer < 0 for layer in args.layers):
            raise FixtureError("position and layers must be non-negative")
        report, code = compare_trace(
            args.ds4_dir, args.llama_dir, args.position, sorted(set(args.layers)),
        )
        write_json(args.report, report)
        print(json.dumps({"status": report["status"],
                          "first_float_divergence": report["first_float_divergence"]},
                         sort_keys=True))
        return code
    except (FixtureError, OSError, ValueError) as exc:
        report = {"format": "ds4-qwen36-trace-comparison-v1",
                  "status": "ERROR", "reason": str(exc)}
        write_json(args.report, report)
        print(json.dumps(report, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
