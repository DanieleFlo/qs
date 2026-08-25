#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import array
import json
import os
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

    def test_r8_slow_reuses_the_short_context_frontiers(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "r8-slow"
        )
        self.assertEqual([item["context"] for item in workloads], [128, 2048])

    def test_long_context_suites_cover_the_observed_cliff(self) -> None:
        direction = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "long-context-direction"
        )
        slow = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "long-context-slow"
        )
        self.assertEqual([item["context"] for item in direction], [10666])
        self.assertEqual([item["context"] for item in slow], [8192, 12288, 16384])
        self.assertEqual([item["generation_tokens"] for item in slow], [64, 64, 64])
        for key in ("generation_tokens", "prefill_chunk", "backend", "batch"):
            self.assertEqual(len({item[key] for item in slow}), 1)

    def test_full_context_curve_covers_every_two_kib_tokens_through_30k(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "context-curve-full"
        )
        self.assertEqual(
            [item["context"] for item in workloads],
            list(range(2048, 30721, 2048)),
        )
        self.assertEqual(len(workloads), 15)
        for key in ("generation_tokens", "prefill_chunk", "backend", "batch"):
            self.assertEqual(len({item[key] for item in workloads}), 1)

    def test_mtp_context_curve_covers_zero_through_28k(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "mtp-context-curve"
        )
        self.assertEqual(
            [item["context"] for item in workloads],
            list(range(0, 28673, 2048)),
        )
        self.assertEqual(len(workloads), 15)
        self.assertLess(
            max(item["context"] for item in workloads) +
            workloads[0]["generation_tokens"] + 1,
            30000,
        )

    def test_mtp_short_regression_is_an_isolated_empty_frontier(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "mtp-short-regression"
        )
        self.assertEqual([item["context"] for item in workloads], [0])
        self.assertEqual([item["generation_tokens"] for item in workloads], [64])

    def test_mtp_threshold_search_follows_the_bisection_points(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "mtp-threshold-search"
        )
        self.assertEqual(
            [item["context"] for item in workloads],
            [64, 125, 250, 500, 1000],
        )
        midpoint = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "mtp-threshold-midpoint"
        )
        self.assertEqual([item["context"] for item in midpoint], [96])

    def test_mtp_long_context_smoke_covers_midpoint_and_28k(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "mtp-long-context-smoke"
        )
        self.assertEqual([item["context"] for item in workloads], [16384, 28672])
        self.assertLess(
            max(item["context"] for item in workloads) +
            workloads[0]["generation_tokens"] + 1,
            30000,
        )

    def test_mtp_depth_28k_is_an_isolated_tail_probe(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "mtp-depth-28k"
        )
        self.assertEqual([item["context"] for item in workloads], [28672])

    def test_mtp_depth_crossover_samples_4k_8k_12k(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "mtp-depth-crossover"
        )
        self.assertEqual(
            [item["context"] for item in workloads],
            [4096, 8192, 12288],
        )

    def test_mtp_depth_2k_is_a_single_boundary_probe(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "mtp-depth-2k"
        )
        self.assertEqual([item["context"] for item in workloads], [2048])

    def test_mtp_depth_boundary_covers_zero_2k_4k(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "mtp-depth-boundary"
        )
        self.assertEqual(
            [item["context"] for item in workloads], [0, 2048, 4096]
        )

    def test_mtp_weakest_confirm_is_24k(self) -> None:
        workloads = HARNESS.load_workloads(
            ROOT / "performance" / "workloads.yaml", "mtp-weakest-confirm"
        )
        self.assertEqual([item["context"] for item in workloads], [24576])

    @staticmethod
    def curve_workload(context: int, tps: float) -> dict:
        return {
            "id": f"c{context}", "status": "measured",
            "definition": {"context": context},
            "metrics": {"gen_steady_tps": {"median": tps}},
        }

    def test_context_curve_gate_accepts_flat_or_decreasing_throughput(self) -> None:
        report = HARNESS.analyze_context_curve([
            self.curve_workload(2048, 31.0),
            self.curve_workload(4096, 29.5),
            self.curve_workload(6144, 20.0),
        ])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["observed_min_tps"], 20.0)
        self.assertEqual(len(report["above_target_ceiling"]), 1)

    def test_context_curve_gate_rejects_a_valley_even_when_floor_passes(self) -> None:
        report = HARNESS.analyze_context_curve([
            self.curve_workload(2048, 28.0),
            self.curve_workload(4096, 21.0),
            self.curve_workload(6144, 27.0),
        ])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(len(report["material_recoveries"]), 1)

    def test_context_curve_ignores_recovery_entirely_above_target(self) -> None:
        report = HARNESS.analyze_context_curve([
            self.curve_workload(2048, 36.0),
            self.curve_workload(4096, 39.0),
            self.curve_workload(6144, 35.0),
        ])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["material_recoveries"], [])

    def test_context_curve_gate_rejects_points_below_twenty_tps(self) -> None:
        report = HARNESS.analyze_context_curve([
            self.curve_workload(2048, 19.99),
            self.curve_workload(4096, 19.0),
        ])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(len(report["below_floor"]), 2)

    def test_server_progress_parser_uses_final_average_not_chunk_speed(self) -> None:
        rows = HARNESS.parse_server_progress(
            "ds4-server: completion ctx=2048..2112 gen=64 "
            "decoding chunk=31.50 t/s avg=27.25 t/s 2.349s\n"
        )
        self.assertEqual(rows[0]["generation_tokens"], 64)
        self.assertEqual(rows[0]["chunk_tps"], 31.5)
        self.assertEqual(rows[0]["avg_tps"], 27.25)

    def test_server_phase_profile_parser_keeps_constraint_work_metrics(self) -> None:
        rows = HARNESS.parse_server_phase_profiles(
            "0814 14:00:00 ds4-server: phase profile ctx=4096..4149 gen=53 "
            "mode=compare_new_vs_oracle wall=2370.000ms "
            "forced_build=290.000ms/40 forced_sync=15.000ms/2/13_tok "
            "forced_prefix_probe=20.000ms sampling_mask_build=246.000ms "
            "oracle_compare=250.000ms/40/0_div constraint_cpu=520.000ms "
            "constraint_cpu_exposed=520.000ms constraint_cpu_overlapped=0.000ms "
            "filter_setup=0.100ms filter=245.900ms filtered_sample=4.000ms/40 "
            "plain_sample=0.000ms/0 eval=1360.000ms/40 residual=475.000ms "
            "vocab=9932800 filter_calls=9932800 accepted=994 "
            "finite_allowed=994 piece_bytes=73846280 "
            "candidate_tokens_tested=19865600 parser_transition_count=1234 "
            "parser_bytes_visited=5678 trie_nodes_visited=321 "
            "subtrees_pruned=77 trie_leaf_tokens_emitted=19 mask_cache_hit=0 "
            "mask_cache_miss=40 grammar_compile_ms=0.000 grammar_jit_ms=0.000 "
            "constraint_state_checkpoint=9932800 constraint_state_rollback=9932800 "
            "exhaustive_fallback_steps=3\n"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "compare_new_vs_oracle")
        self.assertEqual(rows[0]["generation_tokens"], 53)
        self.assertEqual(rows[0]["sampling_mask_build_ms"], 246.0)
        self.assertEqual(rows[0]["candidate_tokens_tested"], 19865600)
        self.assertEqual(rows[0]["trie_nodes_visited"], 321)
        self.assertEqual(rows[0]["subtrees_pruned"], 77)
        self.assertEqual(rows[0]["trie_leaf_tokens_emitted"], 19)
        self.assertEqual(rows[0]["constraint_state_checkpoint"], 9932800)
        self.assertEqual(rows[0]["exhaustive_fallback_steps"], 3)
        self.assertEqual(rows[0]["oracle_compare_calls"], 40)
        self.assertEqual(rows[0]["oracle_divergences"], 0)

    def test_constrained_direction_suite_covers_dsml_and_json(self) -> None:
        workloads = HARNESS.load_constrained_workloads(
            ROOT / "performance" / "constrained-workloads.json", "direction"
        )
        self.assertEqual(
            [item["kind"] for item in workloads], ["dsml", "json_schema"]
        )
        self.assertEqual(
            [item["id"] for item in workloads],
            ["dsml-required-enum-const", "json-nested-required-array"],
        )

    def test_constrained_dsml_json_fallback_suite_covers_both_param_kinds(self) -> None:
        workloads = HARNESS.load_constrained_workloads(
            ROOT / "performance" / "constrained-workloads.json", "dsml-json-fallback"
        )
        self.assertEqual(
            [item["id"] for item in workloads],
            ["dsml-required-enum-const", "dsml-json-const-enum"],
        )
        self.assertEqual(
            [item["kind"] for item in workloads], ["dsml", "dsml"]
        )

    def test_internal_json_fixture_does_not_require_source_witness(self) -> None:
        internal = [{"definition": {"kind": "json_schema", "source": None}}]
        external = [{
            "definition": {
                "kind": "json_schema",
                "source": {"witness_valid": False},
            }
        }]
        external_missing_witness = [{
            "definition": {"kind": "json_schema", "source": {}}
        }]
        self.assertTrue(HARNESS.constrained_source_witnesses_valid(internal))
        self.assertFalse(HARNESS.constrained_source_witnesses_valid(external))
        self.assertFalse(
            HARNESS.constrained_source_witnesses_valid(external_missing_witness)
        )
        free_string = HARNESS.load_constrained_workloads(
            ROOT / "performance" / "constrained-workloads.json", "free-string"
        )
        self.assertEqual([item["id"] for item in free_string], [
            "dsml-free-string-128"
        ])
        schema = free_string[0]["payload"]["tools"][0]["parameters"]
        self.assertEqual(schema["properties"]["content"], {"type": "string"})
        nullable = HARNESS.load_constrained_workloads(
            ROOT / "performance" / "constrained-workloads.json",
            "nullable-free-string",
        )
        self.assertEqual([item["id"] for item in nullable], [
            "dsml-nullable-free-string"
        ])
        variants = nullable[0]["payload"]["tools"][0]["parameters"]
        self.assertEqual(variants["properties"]["content"]["anyOf"], [
            {"type": "string"}, {"type": "null"}
        ])

    def test_jsonschemabench_subset_builds_pinned_server_workloads(self) -> None:
        workloads = HARNESS.load_jsonschemabench_workloads(
            ROOT / "performance" / "jsonschemabench-subset.json",
            tier="smoke",
        )
        self.assertEqual(len(workloads), 12)
        self.assertTrue(all(
            item["id"].startswith("jsonschemabench/") for item in workloads
        ))
        self.assertTrue(all(
            item["payload"]["seed"] == 424242 for item in workloads
        ))
        self.assertTrue(all(
            item["payload"]["response_format"]["json_schema"]["schema"]
            for item in workloads
        ))
        safety = HARNESS.load_jsonschemabench_workloads(
            ROOT / "performance" / "jsonschemabench-subset.json"
        )
        self.assertEqual(len(safety), 32)
        self.assertTrue({item["source"]["id"] for item in workloads}.issubset(
            {item["source"]["id"] for item in safety}
        ))
        unsupported = HARNESS.load_jsonschemabench_unsupported_probes(
            ROOT / "performance" / "jsonschemabench-subset.json"
        )
        self.assertEqual(len(unsupported), 16)
        self.assertTrue(all(item["reasons"] for item in unsupported))
        self.assertTrue(all(
            item["payload"]["max_tokens"] == 1 for item in unsupported
        ))
        selected = HARNESS.load_jsonschemabench_workloads(
            ROOT / "performance" / "jsonschemabench-subset.json",
            ["Github_trivial/o27825.json"],
        )
        self.assertEqual(
            [item["source"]["id"] for item in selected],
            ["Github_trivial/o27825.json"],
        )
        with self.assertRaisesRegex(HARNESS.HarnessError, "not in"):
            HARNESS.load_jsonschemabench_workloads(
                ROOT / "performance" / "jsonschemabench-subset.json",
                ["missing.json"],
            )

    def test_constrained_semantic_output_ignores_random_response_ids(self) -> None:
        output = HARNESS.constrained_semantic_output("dsml", {
            "id": "random",
            "output": [{
                "type": "function_call", "name": "record",
                "call_id": "also-random", "arguments": '{"value":1}',
            }],
        })
        self.assertEqual(output, {
            "function_calls": [{"name": "record", "arguments": {"value": 1}}]
        })
        structured = HARNESS.constrained_semantic_output("json_schema", {
            "choices": [{"message": {"content": '{"ok":true}'}}]
        })
        self.assertEqual(structured, {"json": {"ok": True}})

    def test_constrained_semantic_output_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(HARNESS.HarnessError, "duplicate JSON"):
            HARNESS.constrained_semantic_output("json_schema", {
                "choices": [{"message": {"content": '{"ok":true,"ok":false}'}}]
            })

    def test_jsonschemabench_prefix_probe_requires_exact_budget(self) -> None:
        response = {
            "choices": [{
                "finish_reason": "error",
                "message": {"content": '{"partial"'},
            }]
        }
        self.assertEqual(
            HARNESS.constrained_json_prefix_output(response, 8, 8),
            {
                "json_prefix": '{"partial"', "finish_reason": "error",
                "completion_tokens": 8,
            },
        )
        with self.assertRaisesRegex(HARNESS.HarnessError, "before"):
            HARNESS.constrained_json_prefix_output(response, 7, 8)
        with self.assertRaisesRegex(HARNESS.HarnessError, "duplicate JSON"):
            HARNESS.constrained_semantic_output("dsml", {
                "output": [{
                    "type": "function_call", "name": "record",
                    "arguments": '{"value":1,"value":2}',
                }],
            })

    @unittest.skipUnless(
        importlib.util.find_spec("jsonschema"),
        "install performance/jsonschemabench-requirements.txt",
    )
    def test_jsonschemabench_output_uses_independent_validator(self) -> None:
        schema = {
            "type": "object",
            "required": ["status", "values"],
            "additionalProperties": False,
            "properties": {
                "status": {"const": "ok"},
                "values": {
                    "type": "array", "minItems": 2,
                    "items": {"type": "integer"},
                },
            },
        }
        result = HARNESS.validate_json_schema_instance(
            schema, {"status": "ok", "values": [1, 2]}
        )
        self.assertTrue(result["valid"])
        self.assertIn("Validator", result["validator"])
        with self.assertRaisesRegex(
            HARNESS.HarnessError, "violates JSON Schema"
        ):
            HARNESS.validate_json_schema_instance(
                schema, {"status": "ok", "values": [1], "extra": True}
            )

    @unittest.skipUnless(
        importlib.util.find_spec("jsonschema"),
        "install performance/jsonschemabench-requirements.txt",
    )
    def test_minimal_witness_handles_intersections_and_unions(self) -> None:
        schema = {
            "type": "object", "required": ["payload"],
            "properties": {"payload": {"allOf": [
                {
                    "type": "object", "required": ["left"],
                    "properties": {"left": {"enum": ["L"]}},
                },
                {
                    "type": "object", "required": ["right"],
                    "properties": {"right": {"anyOf": [
                        {"type": "string"}, {"type": "null"},
                    ]}},
                },
            ]}},
        }
        witness = HARNESS.minimal_json_schema_witness(schema)
        HARNESS.validate_json_schema_instance(schema, witness)
        self.assertEqual(witness, {"payload": {"left": "L", "right": ""}})

    def test_constrained_server_parser_exposes_oracle_mode(self) -> None:
        args = HARNESS.build_parser().parse_args([
            "constrained-server", "--model", "model.gguf", "--baseline-run",
            "--hypothesis", "freeze exhaustive constrained baseline",
            "--constraint-mode", "oracle_only",
        ])
        self.assertEqual(args.constraint_mode, "oracle_only")
        self.assertEqual(args.repetitions, 2)
        self.assertEqual(args.context, 4096)

    def test_constrained_server_accepts_external_subset(self) -> None:
        args = HARNESS.build_parser().parse_args([
            "constrained-server", "--model", "model.gguf", "--baseline-run",
            "--hypothesis", "measure external schemas",
            "--jsonschemabench-subset",
        ])
        self.assertEqual(
            args.jsonschemabench_subset,
            str(HARNESS.DEFAULT_JSONSCHEMABENCH_SUBSET),
        )
        self.assertEqual(args.jsonschemabench_tier, "safety")
        self.assertEqual(args.jsonschemabench_prefix_steps, 0)
        self.assertFalse(args.jsonschemabench_check_unsupported)

    def test_server_prompt_calibration_requires_one_token_per_filler(self) -> None:
        self.assertEqual(HARNESS.prompt_filler_intercept(32, 47, 96, 111), 15)
        with self.assertRaisesRegex(HARNESS.HarnessError, "not one token"):
            HARNESS.prompt_filler_intercept(32, 47, 96, 112)

    def test_server_curve_parser_defaults_to_two_repetitions(self) -> None:
        args = HARNESS.build_parser().parse_args([
            "server-curve", "--model", "model.gguf", "--baseline-run",
            "--hypothesis", "server curve remains above the floor",
        ])
        self.assertEqual(args.repetitions, 2)
        self.assertEqual(args.port, 0)
        self.assertIsNone(args.context_alloc)

    def test_server_curve_accepts_mtp_curve_and_sidecar(self) -> None:
        args = HARNESS.build_parser().parse_args([
            "server-curve", "--model", "model.gguf", "--baseline-run",
            "--hypothesis", "MTP remains faster through 28K",
            "--suite", "mtp-context-curve", "--mtp", "--no-thinking",
            "--prompt-pattern", "technical-explanation",
        ])
        self.assertEqual(args.suite, "mtp-context-curve")
        self.assertTrue(args.mtp)
        self.assertFalse(args.thinking)
        self.assertEqual(args.prompt_pattern, "technical-explanation")

    def test_server_curve_accepts_explicit_large_context_allocation(self) -> None:
        args = HARNESS.build_parser().parse_args([
            "server-curve", "--model", "model.gguf", "--baseline-run",
            "--hypothesis", "short MTP remains fast with a large allocation",
            "--suite", "mtp-short-regression", "--mtp",
            "--context-alloc", "28768",
        ])
        self.assertEqual(args.suite, "mtp-short-regression")
        self.assertEqual(args.context_alloc, 28768)

    def test_technical_server_prompt_has_one_token_filler(self) -> None:
        short = HARNESS.server_curve_prompt(32, "technical-explanation")
        long = HARNESS.server_curve_prompt(96, "technical-explanation")
        self.assertEqual(long.count(" alpha") - short.count(" alpha"), 64)
        self.assertIn("at least 64 words", short)

    def test_doctor_reports_the_shared_split_k_crossover(self) -> None:
        self.assertEqual(HARNESS.QWEN_SPLIT_K_MIN_CONTEXT, 96)
        self.assertEqual(HARNESS.QWEN_MTP_DEPTH1_MIN_CONTEXT, 2048)

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

    def test_r8_workflow_uses_the_runtime_relink_script(self) -> None:
        parser = HARNESS.build_parser()
        args = parser.parse_args([
            "workflow", "--name", "r8-slow", "--id", "r8-confirm",
            "--dry-run",
        ])
        command, env = HARNESS.workflow_command(args)
        self.assertTrue(command[0].endswith("perf-qwen-r8.sh"))
        self.assertEqual(command[1:], ["slow", "r8-confirm"])
        self.assertEqual(env["CUDA_ARCH"], "sm_86")

    def test_r8_long_workflow_uses_the_same_automation(self) -> None:
        parser = HARNESS.build_parser()
        args = parser.parse_args([
            "workflow", "--name", "r8-long", "--id", "gqa-confirm",
            "--dry-run",
        ])
        command, _env = HARNESS.workflow_command(args)
        self.assertTrue(command[0].endswith("perf-qwen-r8.sh"))
        self.assertEqual(command[1:], ["long", "gqa-confirm"])

    def test_r8_workflow_builds_all_runtimes_and_uses_explicit_rollback(self) -> None:
        script = (ROOT / "tools" / "perf-qwen-r8.sh").read_text(encoding="utf-8")
        self.assertIn("ds4 ds4-bench ds4-server", script)
        self.assertIn("DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8=1", script)

    def test_binary_freshness_reports_a_newer_build_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, source = root / "ds4", root / "ds4_cuda.o"
            binary.write_bytes(b"binary")
            source.write_bytes(b"object")
            os.utime(binary, ns=(1_000_000_000, 1_000_000_000))
            os.utime(source, ns=(2_000_000_000, 2_000_000_000))
            report = HARNESS.binary_freshness(binary, [source])
            self.assertFalse(report["fresh"])
            self.assertEqual(report["newer_inputs"], [str(source)])

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

    def test_q8_1_parity_ignores_unconsumed_ds_y_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qs = bytes((index * 17) % 255 for index in range(32))
            left = b"\x00\x24\x01\x02" + qs
            right = b"\x00\x24\xfe\xfd" + qs
            left_path, right_path = root / "left.bin", root / "right.bin"
            left_path.write_bytes(left)
            right_path.write_bytes(right)
            report = HARNESS.q8_1_parity(left_path, right_path)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["mmvq_consumed_fields"]["equal"])
            self.assertEqual(report["metadata_ds_y"]["differing_blocks"], 1)

    def test_q8_1_parity_rejects_quant_byte_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = bytearray(36)
            right = bytearray(left)
            right[19] = 1
            left_path, right_path = root / "left.bin", root / "right.bin"
            left_path.write_bytes(left)
            right_path.write_bytes(right)
            report = HARNESS.q8_1_parity(left_path, right_path)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["mmvq_consumed_fields"]["qs_differing_bytes"], 1)

    def test_qwen_logits_row_compares_labeled_binary_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for label, last in (("A", 4.0), ("B", 5.0)):
                run = root / label / "logits"
                run.mkdir(parents=True)
                values = array.array("f", [0.0, 1.0, 2.0, 3.0,
                                            0.0, 1.0, 2.0, last])
                with (run / "case.greedy.f32").open("wb") as file:
                    values.tofile(file)
                runs.append((label, root / label))
            report = HARNESS.qwen_logits_row_report(
                runs, "case", "greedy", row=1, vocab=4,
                top_k=2, focus_tokens=[3, 2],
            )
            self.assertEqual(report["runs"][0]["argmax"], 3)
            self.assertAlmostEqual(report["runs"][1]["focus_margin"], 3.0)
            self.assertTrue(report["comparisons"][0]["argmax_equal"])

    def test_qwen_argmax_gate_counts_greedy_and_teacher_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, greedy, teacher in (
                    ("reference", [1, 2], [3, 4]),
                    ("candidate", [1, 2], [3, 4])):
                run = root / label
                (run / "responses").mkdir(parents=True)
                (run / "index.json").write_text(json.dumps({
                    "cases": [{"id": "case", "response_file": "responses/case.json"}],
                }), encoding="utf-8")
                (run / "responses" / "case.json").write_text(json.dumps({
                    "canonical_prompt_token_ids": [10, 11],
                    "teacher_forced_source": "continuation",
                    "greedy_token_ids": greedy,
                    "teacher_forced": [
                        {"token_id": token, "logprob": -0.1} for token in teacher
                    ],
                }), encoding="utf-8")
            report = HARNESS.qwen_argmax_gate(root / "reference", root / "candidate")
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["sequences"], {"equal": 1, "total": 1})
            self.assertEqual(report["argmax"]["equal"], 4)
            self.assertEqual(report["argmax"]["total"], 4)

    def test_qwen_argmax_gate_reports_teacher_only_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, teacher in (("reference", 3), ("candidate", 9)):
                run = root / label
                (run / "responses").mkdir(parents=True)
                (run / "index.json").write_text(json.dumps({
                    "cases": [{"id": "case", "response_file": "responses/case.json"}],
                }), encoding="utf-8")
                (run / "responses" / "case.json").write_text(json.dumps({
                    "canonical_prompt_token_ids": [10],
                    "teacher_forced_source": "continuation",
                    "greedy_token_ids": [1],
                    "teacher_forced": [{"token_id": teacher, "logprob": -0.1}],
                }), encoding="utf-8")
            report = HARNESS.qwen_argmax_gate(root / "reference", root / "candidate")
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["sequences"]["equal"], 1)
            self.assertEqual(report["argmax"]["equal"], 1)
            self.assertEqual(report["cases"][0]["first_teacher_difference"], 0)

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
