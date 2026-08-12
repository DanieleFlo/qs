#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "perf_harness", ROOT / "tools" / "perf_harness.py"
)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


class PerfHarnessTests(unittest.TestCase):
    def test_canonical_quick_suite_names_phase_batch_and_context(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "quick"
        )
        self.assertEqual([item["id"] for item in workloads],
                         ["decode-b1-c128-direction", "prefill-b1-c512",
                          "decode-b1-c2k"])
        for item in workloads:
            self.assertIn(item["phase"], ("prefill", "decode"))
            self.assertGreater(item["batch"], 0)
            self.assertGreater(item["context"], 0)

    def test_direction_suite_can_share_one_resident_model_sweep(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "direction"
        )
        self.assertEqual([item["context"] for item in workloads], [128, 2048])
        for key in ("generation_tokens", "prefill_chunk", "backend", "batch"):
            self.assertEqual(workloads[0][key], workloads[1][key])

    def test_long_context_suites_cover_the_observed_cliff(self) -> None:
        direction = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "long-context-direction"
        )
        slow = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "long-context-slow"
        )
        self.assertEqual([item["context"] for item in direction], [10666])
        self.assertEqual([item["context"] for item in slow], [8192, 12288, 16384])
        for key in ("generation_tokens", "prefill_chunk", "backend", "batch"):
            self.assertEqual(len({item[key] for item in slow}), 1)

    def test_workflow_dry_run_builds_the_documented_script_command(self) -> None:
        parser = HARNESS.build_parser()
        args = parser.parse_args([
            "workflow", "--name", "long-context-profile", "--id", "probe",
            "--candidate-env", "DS4_CUDA_QWEN_SPLIT_K_ATTN=1", "--dry-run",
        ])
        command, env = HARNESS.workflow_command(args)
        self.assertTrue(command[0].endswith("perf-qwen-long-context.sh"))
        self.assertEqual(command[1:3], ["profile", "probe"])
        self.assertIn("DS4_CUDA_QWEN_SPLIT_K_ATTN=1", command)
        self.assertEqual(env["CUDA_ARCH"], "sm_86")

    def test_summary_marks_noisy_samples_unstable(self) -> None:
        stable = HARNESS.summary([100.0, 101.0, 99.0, 100.5, 99.5])
        noisy = HARNESS.summary([50.0, 150.0, 60.0, 140.0, 100.0])
        self.assertFalse(stable["unstable"])
        self.assertTrue(noisy["unstable"])
        self.assertEqual(stable["median"], 100.0)

    def test_compute_capability_comparison_handles_two_digit_major(self) -> None:
        self.assertTrue(HARNESS.compute_capability_at_least("10.0", (8, 9)))
        self.assertFalse(HARNESS.compute_capability_at_least("8.6", (8, 9)))

    def test_comparison_inverts_latency_improvement(self) -> None:
        def record(name: str, speed: float, latency: float) -> dict:
            return {
                "experiment_id": name,
                "target_metric": "gen_steady_tps",
                "correctness": {"status": "PASS"},
                "workloads": [{
                    "id": "decode-b1-c2k", "status": "measured",
                    "metrics": {
                        "gen_steady_tps": {"median": speed},
                        "gen_first_ms": {"median": latency},
                    },
                }],
            }
        result = HARNESS.compare_records(
            record("base", 10.0, 100.0), record("candidate", 11.0, 90.0)
        )
        metrics = result["workloads"][0]["metrics"]
        self.assertAlmostEqual(metrics["gen_steady_tps"]["improvement_percent"], 10.0)
        self.assertAlmostEqual(metrics["gen_first_ms"]["improvement_percent"], 10.0)
        self.assertEqual(result["verdict"], "KEEP_CANDIDATE")

    def test_json_subset_workload_file_needs_no_yaml_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workloads.yaml"
            path.write_text(json.dumps({
                "schema_version": 1,
                "workloads": {"one": {"context": 1}},
                "suites": {"quick": ["one"]},
            }), encoding="utf-8")
            self.assertEqual(HARNESS.load_workloads(path, "quick")[0]["id"], "one")

    def test_layer_profile_parser_and_hotspot_ranking(self) -> None:
        text = (
            "QWEN_PREFILL_LAYER_PROFILE pos=0 rows=128 layer=0 kind=recurrent "
            "attn=2.000ms ffn=5.000ms total=7.000ms\n"
            "QWEN_DECODE_LAYER_PROFILE pos=128 layer=1 kind=full "
            "attn=3.000ms ffn=1.000ms total=4.000ms\n"
            "QWEN_DECODE_PROFILE pos=128 embed=0.100ms "
            "recurrent_attn=5.000ms full_attn=3.000ms "
            "[qkv=0.500 core=2.000 out=0.500] ffn=6.000ms "
            "output=1.000ms read=0.200ms total=15.300ms\n"
        )
        report = HARNESS.network_profile_report(HARNESS.parse_layer_profiles(text))
        self.assertEqual(report["hotspots"][0]["stage"], "ffn")
        self.assertEqual(report["hotspots"][0]["layer"], 0)
        self.assertAlmostEqual(report["stage_percent"]["attention"], 100.0 * 5.0 / 11.0)
        decode = HARNESS.parse_decode_profiles(text)
        self.assertEqual(decode[0]["position"], 128)
        self.assertAlmostEqual(decode[0]["full_core_ms"], 2.0)

    def test_logits_drift_detects_direction_and_argmax(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps({
                "prompt_tokens": 3, "logits": [1.0, 3.0, 2.0, 0.0],
            }), encoding="utf-8")
            candidate.write_text(json.dumps({
                "prompt_tokens": 3, "logits": [1.0, 3.001, 2.0, 0.0],
            }), encoding="utf-8")
            report = HARNESS.logits_drift(baseline, candidate, top_k=4)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["argmax"]["equal"])

    def test_decode_drift_rejects_generated_token_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.decode.json"
            candidate = root / "candidate.decode.json"
            common = {"prompt_tokens": 3, "logits": [1.0, 3.0, 2.0, 0.0]}
            baseline.write_text(json.dumps({
                **common, "generated_tokens": [10, 11, 12],
            }), encoding="utf-8")
            candidate.write_text(json.dumps({
                **common, "generated_tokens": [10, 99, 12],
            }), encoding="utf-8")
            report = HARNESS.decode_result_drift(baseline, candidate, top_k=4)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["generated_tokens"]["equal"])
            self.assertEqual(report["generated_tokens"]["first_difference"], 1)

    def test_model_cost_separates_memory_decode_and_compute_prefill(self) -> None:
        metadata = {
            "qwen35.block_count": 4, "qwen35.embedding_length": 8,
            "qwen35.feed_forward_length": 16,
            "qwen35.attention.head_count": 2,
            "qwen35.attention.head_count_kv": 1,
            "qwen35.attention.key_length": 4,
            "qwen35.attention.value_length": 4,
            "qwen35.ssm.state_size": 2, "qwen35.ssm.group_count": 1,
            "qwen35.ssm.inner_size": 8,
            "qwen35.ssm.time_step_rank": 2,
            "qwen35.ssm.conv_kernel": 4,
            "qwen35.full_attention_interval": 4,
            "tokenizer.ggml.tokens": {"count": 32},
        }
        tensors = [
            {"name": "blk.0.ffn_gate.weight", "shape": [256, 256], "type": "Q4_K"},
            {"name": "blk.0.attn_qkv.weight", "shape": [256, 256], "type": "Q8_0"},
            {"name": "blk.3.attn_q.weight", "shape": [256, 256], "type": "Q8_0"},
            {"name": "output.weight", "shape": [256, 256], "type": "Q6_K"},
        ]
        snapshot = {"metadata": metadata, "tensors": tensors}
        decode = HARNESS.model_cost(snapshot, phase="decode", context=128, batch=1)
        prefill = HARNESS.model_cost(snapshot, phase="prefill", context=128, batch=1)
        self.assertEqual(decode["state"]["full_attention_kv_bytes"], 4096)
        self.assertGreater(prefill["operations"]["ffn"]["flops"],
                           decode["operations"]["ffn"]["flops"])
        self.assertGreater(decode["theoretical_f32_weight_bytes"],
                           decode["effective_quantized_weight_bytes"])

    def test_missing_json_is_reported_as_harness_error(self) -> None:
        with self.assertRaises(HARNESS.HarnessError):
            HARNESS.read_json(Path("definitely-missing.json"), "baseline")

    def test_observed_cost_requires_the_requested_phase(self) -> None:
        cost = {"phase": "prefill", "workload": {"context": 8}, "operations": {}}
        with self.assertRaisesRegex(HARNESS.HarnessError, "no prefill"):
            HARNESS.attach_observed_profile(cost, {
                "layers": [{"phase": "decode"}]
            })

    def test_observed_cost_rejects_a_mismatched_decode_context(self) -> None:
        cost = {"phase": "decode", "workload": {"context": 128}, "operations": {}}
        profile = {
            "selected_positions": {"decode": 2048},
            "layers": [{"phase": "decode", "rows": 1}],
        }
        with self.assertRaisesRegex(HARNESS.HarnessError, "does not match"):
            HARNESS.attach_observed_profile(cost, profile)

    def test_readme_commands_do_not_reference_bare_placeholder_inputs(self) -> None:
        readme = (ROOT / "performance" / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("baseline-network.json candidate-network.json", readme)
        self.assertNotIn("baseline.logits.json candidate.logits.json", readme)
        self.assertIn("performance-results/base-network.json", readme)
        self.assertIn("performance-results/base-direction/logits/", readme)


if __name__ == "__main__":
    unittest.main()
