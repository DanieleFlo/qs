#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = PROJECT_ROOT / "gguf-tools" / "quality-testing"
MANIFEST = QUALITY_ROOT / "data" / "qwen36-27b" / "manifest.json"
sys.path.insert(0, str(QUALITY_ROOT))

from compare_qwen36_equivalence import (  # noqa: E402
    compare_runs, load_run, longest_common_prefix, main, position_metrics,
)
from qwen36_fixtures import FixtureError, inventory_files, load_json, sha256_file, write_json  # noqa: E402


class Qwen36EquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = load_json(MANIFEST)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_run(
        self,
        name: str,
        values: np.ndarray,
        *,
        engine: str = "ds4",
        review_status: str = "reviewed",
        native_status: str = "verified",
        corrupt: bool = False,
    ) -> Path:
        run = self.root / name
        for directory in ("prompts", "responses", "logits"):
            (run / directory).mkdir(parents=True, exist_ok=True)
        case_id = "case"
        rendered = b"rendered prompt"
        (run / "prompts" / f"{case_id}.txt").write_bytes(rendered)
        (run / "prompts" / f"{case_id}.bytes").write_bytes(rendered)
        write_json(run / "prompts" / f"{case_id}.tokens.json", [1, 2])
        logits = {}
        for pass_name in ("greedy", "teacher_forced"):
            path = run / "logits" / f"{case_id}.{pass_name}.f32"
            path.write_bytes(np.asarray(values, dtype="<f4").tobytes())
            logits[pass_name] = {
                "path": f"logits/{path.name}", "dtype": "float32-le",
                "shape": list(values.shape), "row_stride_bytes": values.shape[1] * 4,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        response = {
            "prompt_token_ids": [1, 2],
            "canonical_prompt_token_ids": [1, 2],
            "native_prompt_token_ids": [1, 2],
            "native_rendering_status": native_status,
            "greedy_token_ids": [int(np.argmax(row)) for row in values],
            "teacher_forced": [{"token_id": 0} for _ in values],
            "full_logits": logits,
        }
        write_json(run / "responses" / f"{case_id}.json", response)
        artifact = {"target": {"filename": "model.gguf", "size_bytes": 1, "sha256": "a" * 64}}
        index = {
            "format": "ds4-qwen36-oracle-v1", "manifest_model": self.manifest["model"]["id"],
            "run_id": name,
            "environment": {
                "oracle": engine, "engine_commit": "test", "build_flags": "test",
                "backend": "CPU", "hardware": "synthetic", "dtype": "float32",
                "parameters": {"context": 16, "prefill_chunk": 4},
                "artifacts": artifact, "review_status": review_status,
            },
            "cases": [{"id": case_id, "response_file": f"responses/{case_id}.json"}],
            "files": inventory_files(run),
        }
        write_json(run / "index.json", index)
        if corrupt:
            with (run / "logits" / f"{case_id}.greedy.f32").open("ab") as fp:
                fp.write(b"x")
        return run

    def refresh_inventory(self, run: Path) -> None:
        index = load_json(run / "index.json")
        index["files"] = inventory_files(run)
        write_json(run / "index.json", index)

    def test_uniform_shift_is_removed_from_error_metrics(self) -> None:
        left = np.array([1.0, 2.0, 4.0, -3.0], dtype=np.float32)
        right = left + np.float32(17.0)
        result = position_metrics(left, right, 2, 3)
        self.assertAlmostEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["cosine_similarity"], 1.0)
        self.assertAlmostEqual(result["js_divergence"], 0.0)

    def test_ranking_metrics_detect_inversion_and_partial_overlap(self) -> None:
        left = np.array([5.0, 4.0, 3.0, 0.0], dtype=np.float32)
        right = np.array([3.0, 4.0, 0.0, 5.0], dtype=np.float32)
        result = position_metrics(left, right, 0, 3)
        self.assertAlmostEqual(result["top_k_overlap"], 2 / 3)
        self.assertLess(result["top_k_rank_agreement"], 1.0)
        self.assertFalse(result["greedy_agreement"])

    def test_uniform_vectors_have_defined_cosine_and_spearman(self) -> None:
        values = np.ones(4, dtype=np.float32)
        result = position_metrics(values, values.copy(), 0, 4)
        self.assertEqual(result["cosine_similarity"], 1.0)
        self.assertEqual(result["top_k_spearman"], 1.0)

    def test_nonfinite_values_are_counted(self) -> None:
        result = position_metrics(
            np.array([0.0, np.nan, np.inf], dtype=np.float32),
            np.array([0.0, 1.0, -np.inf], dtype=np.float32), 0, 2,
        )
        self.assertEqual((result["left_nan"], result["left_inf"], result["right_inf"]), (1, 1, 1))
        self.assertFalse(result["metrics_available"])

    def test_longest_common_prefix(self) -> None:
        self.assertEqual(longest_common_prefix([1, 2, 3], [1, 2, 4, 5]), 2)

    def test_internal_identical_run_passes(self) -> None:
        values = np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]], dtype=np.float32)
        left = load_run(self.make_run("left", values))
        right = load_run(self.make_run("right", values))
        report, code = compare_runs(left, right, self.manifest, "ds4-vs-ds4", 2, False)
        self.assertEqual((report["status"], code), ("PASS", 0))
        self.assertEqual(report["aggregate"]["different_float_count"], 0)

    def test_internal_difference_fails_and_records_first_position(self) -> None:
        left_values = np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]], dtype=np.float32)
        right_values = left_values.copy()
        right_values[1, 0] += 0.25
        left = load_run(self.make_run("left", left_values))
        right = load_run(self.make_run("right", right_values))
        report, code = compare_runs(left, right, self.manifest, "ds4-vs-ds4", 2, False)
        self.assertEqual((report["status"], code), ("FAIL", 1))
        self.assertEqual(report["aggregate"]["first_divergence"]["position"], 1)

    def test_unreviewed_cross_engine_is_not_verified(self) -> None:
        values = np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]], dtype=np.float32)
        left = load_run(self.make_run("llama", values, engine="llama.cpp", review_status="generated_unreviewed"))
        right = load_run(self.make_run("ds4", values, review_status="generated_unreviewed"))
        report, code = compare_runs(left, right, self.manifest, "ds4-vs-llama", 2, False)
        self.assertEqual((report["status"], code), ("NOT_VERIFIED", 3))

    def test_nonfinite_run_fails_fixed_gate(self) -> None:
        values = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
        broken = values.copy()
        broken[0, 1] = np.nan
        left = load_run(self.make_run("left", values))
        right = load_run(self.make_run("right", broken))
        report, code = compare_runs(left, right, self.manifest, "ds4-vs-ds4", 2, False)
        self.assertEqual((report["status"], code), ("FAIL", 1))
        self.assertEqual(report["aggregate"]["nonfinite_positions"], 2)

    def test_vocabulary_shape_mismatch_is_rejected(self) -> None:
        left = load_run(self.make_run("left", np.array([[3.0, 2.0, 1.0]], dtype=np.float32)))
        right = load_run(self.make_run("right", np.array([[3.0, 2.0]], dtype=np.float32)))
        with self.assertRaisesRegex(FixtureError, "shape mismatch"):
            compare_runs(left, right, self.manifest, "ds4-vs-ds4", 2, False)

    def test_case_set_and_provenance_mismatches_are_rejected(self) -> None:
        values = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
        left_path = self.make_run("left", values)
        right_path = self.make_run("right", values)
        index = load_json(right_path / "index.json")
        index["environment"]["artifacts"]["target"]["sha256"] = "b" * 64
        write_json(right_path / "index.json", index)
        with self.assertRaisesRegex(FixtureError, "artifact provenance"):
            compare_runs(load_run(left_path), load_run(right_path), self.manifest, "ds4-vs-ds4", 2, False)

        index["environment"]["artifacts"]["target"]["sha256"] = "a" * 64
        index["cases"] = []
        write_json(right_path / "index.json", index)
        with self.assertRaisesRegex(FixtureError, "no cases"):
            load_run(right_path)

    def test_corrupt_inventory_is_input_error(self) -> None:
        values = np.array([[1.0, 0.0]], dtype=np.float32)
        run = self.make_run("corrupt", values, corrupt=True)
        report_path = self.root / "report.json"
        code = main([
            "--manifest", str(MANIFEST), "--mode", "ds4-vs-ds4",
            "--left-run", str(run), "--right-run", str(run),
            "--report", str(report_path),
        ])
        self.assertEqual(code, 2)
        self.assertEqual(load_json(report_path)["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
