#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = PROJECT_ROOT / "gguf-tools" / "quality-testing"
sys.path.insert(0, str(QUALITY_ROOT))

from compare_qwen36_trace import compare_trace  # noqa: E402
from qwen36_fixtures import FixtureError  # noqa: E402
from qwen36_speed_gate import validate_speed_report  # noqa: E402


class Qwen36SafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def speed_report(self, prefill: float = 500.0, decode: float = 15.0) -> Path:
        path = self.root / "speed.json"
        path.write_text(json.dumps({
            "format": "ds4-qwen36-speed-v2",
            "model_sha256": "a" * 64,
            "engine_commit": "test",
            "build_flags": "release CUDA",
            "backend": "CUDA",
            "hardware": "RTX 3090",
            "prefill_command": ["ds4-bench", "--ctx-start", "2048"],
            "decode_command": ["ds4-bench", "--ctx-start", "128"],
            "prefill_context": 2048,
            "decode_context": 128,
            "prefill_tokens": 2048,
            "decode_tokens": 64,
            "prefill_tokens_per_second": prefill,
            "decode_tokens_per_second": decode,
            "peak_memory_mib": 21000.0,
        }), encoding="utf-8")
        return path

    def test_speed_gate_accepts_exact_boundaries(self) -> None:
        report = validate_speed_report(
            self.speed_report(), model_sha256="a" * 64,
            backend="CUDA", hardware="RTX 3090",
        )
        self.assertEqual(report["prefill_tokens_per_second"], 500.0)
        self.assertEqual(report["decode_tokens_per_second"], 15.0)

    def test_speed_gate_rejects_missing_or_slow_measurements(self) -> None:
        with self.assertRaisesRegex(FixtureError, "prefill 499"):
            validate_speed_report(
                self.speed_report(prefill=499.9), model_sha256="a" * 64,
                backend="CUDA", hardware="RTX 3090",
            )
        with self.assertRaisesRegex(FixtureError, "backend/hardware"):
            validate_speed_report(
                self.speed_report(), model_sha256="a" * 64,
                backend="CPU", hardware="RTX 3090",
            )
        with self.assertRaisesRegex(FixtureError, "decode 14"):
            validate_speed_report(
                self.speed_report(decode=14.9), model_sha256="a" * 64,
                backend="CUDA", hardware="RTX 3090",
            )

    def test_trace_comparison_finds_first_stage_drift(self) -> None:
        ds4 = self.root / "ds4"
        llama = self.root / "llama"
        ds4.mkdir()
        llama.mkdir()
        exact = np.array([1.0, 2.0, 3.0], dtype="<f4")
        drift = exact.copy()
        drift[1] += np.float32(0.25)
        exact.tofile(ds4 / "ds4-pos17-layer0-attn_norm.f32")
        exact.tofile(llama / "llama-pos17-layer0-attn_norm.f32")
        drift.tofile(ds4 / "ds4-pos17-layer0-layer_out.f32")
        exact.tofile(llama / "llama-pos17-layer0-post_ffn.f32")
        report, code = compare_trace(ds4, llama, 17, [0])
        self.assertEqual((report["status"], code), ("DIAGNOSTIC", 0))
        self.assertEqual(report["first_float_divergence"]["ds4_stage"], "layer_out")
        self.assertAlmostEqual(report["first_float_divergence"]["mae"], 0.25 / 3)

    def test_trace_comparison_rejects_shape_and_nonfinite(self) -> None:
        ds4 = self.root / "ds4"
        llama = self.root / "llama"
        ds4.mkdir()
        llama.mkdir()
        np.array([1.0, 2.0], dtype="<f4").tofile(
            ds4 / "ds4-pos17-layer0-attn_norm.f32")
        np.array([1.0], dtype="<f4").tofile(
            llama / "llama-pos17-layer0-attn_norm.f32")
        with self.assertRaisesRegex(FixtureError, "shape mismatch"):
            compare_trace(ds4, llama, 17, [0])


if __name__ == "__main__":
    unittest.main()
