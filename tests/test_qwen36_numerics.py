#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = PROJECT_ROOT / "gguf-tools" / "quality-testing"
sys.path.insert(0, str(QUALITY_ROOT))

from diagnose_qwen36_numerics import canonicalize, envelope_metrics  # noqa: E402


class Qwen36NumericsTests(unittest.TestCase):
    def test_gguf_tiled_value_heads_require_modulo_key_mapping(self) -> None:
        key_heads, values_per_key = 16, 3
        # HF grouped order is [K0v0,K0v1,K0v2,K1v0,...].  The GGUF converter
        # reshapes [Hk,R], transposes to [R,Hk], then flattens.
        grouped_key = np.repeat(np.arange(key_heads), values_per_key)
        tiled_key = grouped_key.reshape(key_heads, values_per_key).T.reshape(-1)
        np.testing.assert_array_equal(
            tiled_key, np.arange(key_heads * values_per_key) % key_heads,
        )

    def test_llama_recurrent_state_layout_is_canonicalized(self) -> None:
        source = np.arange(48 * 128 * 128, dtype=np.float32)
        expected = source.reshape(48, 128, 128).transpose(0, 2, 1).reshape(-1)
        actual = canonicalize("recurrent_state", "llama_cpu", source)
        np.testing.assert_array_equal(actual, expected)
        self.assertIs(canonicalize("recurrent_state", "ds4", source), source)

    def test_roundoff_inside_cpu_cuda_envelope(self) -> None:
        cpu = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cuda = np.nextafter(cpu, np.float32(np.inf))
        result = envelope_metrics(cuda.copy(), cpu, cuda, 2.0, 0.0, 8.0)
        self.assertEqual(result["classification"], "exact_reference_match")

        ds4 = np.nextafter(cuda, np.float32(np.inf))
        result = envelope_metrics(ds4, cpu, cuda, 2.0, 0.0, 8.0)
        self.assertEqual(result["classification"], "within_cpu_cuda_roundoff_envelope")

    def test_material_error_is_outside_envelope(self) -> None:
        cpu = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cuda = np.nextafter(cpu, np.float32(np.inf))
        ds4 = cpu + np.float32(0.01)
        result = envelope_metrics(ds4, cpu, cuda, 2.0, 1e-7, 8.0)
        self.assertEqual(result["classification"], "suspicious_outside_backend_envelope")
        self.assertEqual(result["outside_values"], 3)

    def test_nonfinite_is_invalid(self) -> None:
        cpu = np.array([1.0, 2.0], dtype=np.float32)
        result = envelope_metrics(
            np.array([1.0, np.nan], dtype=np.float32), cpu, cpu, 2.0, 1e-7, 8.0,
        )
        self.assertEqual(result["classification"], "invalid_nonfinite")


if __name__ == "__main__":
    unittest.main()
