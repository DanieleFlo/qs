#!/usr/bin/env python3

from __future__ import annotations

import os
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = PROJECT_ROOT / "gguf-tools" / "quality-testing"
MANIFEST_PATH = QUALITY_ROOT / "data" / "qwen38-27b" / "manifest.json"
sys.path.insert(0, str(QUALITY_ROOT))

from qwen38_fixtures import (  # noqa: E402
    REQUIRED_CUDA_FORMATS,
    sha256_file,
    validate_manifest,
)
from qwen38_speed_gate import validate_speed_report  # noqa: E402
import generate_qwen36_oracle as oracle_generator  # noqa: E402


class FakeQwen38Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, rendered, add_special_tokens=False):
        return [1, 2, 3, 4]

    def decode(self, token_ids, skip_special_tokens=False):
        return "Qwen3.8 fixture continuation"


TARGET_EMBEDDED_MTP_TYPES = {
    "blk.64.attn_norm.weight": "F32",
    "blk.64.post_attention_norm.weight": "F32",
    "blk.64.attn_q.weight": "Q6_K",
    "blk.64.attn_k.weight": "Q8_0",
    "blk.64.attn_v.weight": "Q8_0",
    "blk.64.attn_q_norm.weight": "F32",
    "blk.64.attn_k_norm.weight": "F32",
    "blk.64.attn_output.weight": "Q6_K",
    "blk.64.ffn_gate.weight": "Q6_K",
    "blk.64.ffn_up.weight": "Q6_K",
    "blk.64.ffn_down.weight": "Q6_K",
    "blk.64.nextn.eh_proj.weight": "Q6_K",
    "blk.64.nextn.enorm.weight": "F32",
    "blk.64.nextn.hnorm.weight": "F32",
    "blk.64.nextn.shared_head_norm.weight": "F32",
}

SIDECAR_MTP_TYPES = {
    **{name: kind for name, kind in TARGET_EMBEDDED_MTP_TYPES.items()
       if kind == "F32"},
    "blk.64.attn_q.weight": "Q6_K",
    "blk.64.attn_k.weight": "Q6_K",
    "blk.64.attn_v.weight": "Q6_K",
    "blk.64.attn_output.weight": "Q6_K",
    "blk.64.ffn_gate.weight": "Q4_K",
    "blk.64.ffn_up.weight": "Q4_K",
    "blk.64.ffn_down.weight": "Q4_K",
    "blk.64.nextn.eh_proj.weight": "Q4_K",
    "token_embd.weight": "Q3_K",
    "output_norm.weight": "F32",
    "output.weight": "Q3_K",
}

EXPECTED_MTP_SHAPES = {
    "blk.64.attn_norm.weight": [5120],
    "blk.64.post_attention_norm.weight": [5120],
    "blk.64.attn_q.weight": [5120, 12288],
    "blk.64.attn_k.weight": [5120, 1024],
    "blk.64.attn_v.weight": [5120, 1024],
    "blk.64.attn_q_norm.weight": [256],
    "blk.64.attn_k_norm.weight": [256],
    "blk.64.attn_output.weight": [6144, 5120],
    "blk.64.ffn_gate.weight": [5120, 17408],
    "blk.64.ffn_up.weight": [5120, 17408],
    "blk.64.ffn_down.weight": [17408, 5120],
    "blk.64.nextn.eh_proj.weight": [10240, 5120],
    "blk.64.nextn.enorm.weight": [5120],
    "blk.64.nextn.hnorm.weight": [5120],
    "blk.64.nextn.shared_head_norm.weight": [5120],
}


class Qwen38CompatibilityTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.snapshots = validate_manifest(MANIFEST_PATH)

    def test_generation_uses_its_own_oracle_namespace(self) -> None:
        oracle = self.manifest["oracle"]
        self.assertEqual(oracle["output_format"], "ds4-qwen38-oracle-v1")
        self.assertEqual(oracle["reference_model"], "Qwen3.8-27B-UD-Q4_K_S.gguf")
        self.assertNotIn("Qwen3.6", oracle["reference_model"])

    def test_oracle_generator_writes_only_the_qwen38_namespace(self) -> None:
        fake_result = {
            "engine": "transformers",
            "engine_version": "test",
            "prompt_token_ids": [1, 2, 3, 4],
            "greedy_token_ids": [7] * 32,
            "greedy_text": "Qwen3.8 fixture continuation",
            "teacher_forced_source": "same-run greedy continuation",
            "teacher_forced": [{"token_id": 7, "logprob": 0.0}] * 32,
            "top_k": [[{"token_id": 7, "logprob": 0.0}]] * 32,
            "full_logits": None,
        }
        with tempfile.TemporaryDirectory() as raw_staging:
            staging = Path(raw_staging)
            argv = [
                "generate_qwen38_oracle.py",
                "--manifest", str(MANIFEST_PATH),
                "--oracle", "transformers",
                "--staging-dir", str(staging),
                "--run-id", "qwen38-namespace-test",
                "--engine-commit", "a" * 40,
                "--build-flags", "synthetic test build",
                "--backend", "CPU",
                "--hardware", "synthetic test host",
                "--dtype", "float32",
                "--case", "single_token_ascii",
            ]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(oracle_generator, "load_renderer",
                                   return_value=FakeQwen38Tokenizer()), \
                 mock.patch.object(oracle_generator, "create_transformers",
                                   return_value=object()), \
                 mock.patch.object(oracle_generator, "generate_transformers",
                                   return_value=fake_result), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(oracle_generator.main(), 0)
            run = staging / "qwen38-namespace-test"
            index = json.loads((run / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["format"], "ds4-qwen38-oracle-v1")
            self.assertEqual(index["manifest_model"], "Qwen3.8-27B-UD-Q4_K_S")
            self.assertNotIn("Qwen3.6", json.dumps(index))
            self.assertTrue((staging / ".ds4-qwen38-staging").is_file())
            self.assertFalse((staging / ".ds4-qwen36-staging").exists())

    def test_target_embedded_nextn_layout(self) -> None:
        tensors = {item["name"]: item for item in self.snapshots["target"]["tensors"]}
        block = {name: item for name, item in tensors.items() if name.startswith("blk.64.")}
        self.assertEqual(set(block), set(TARGET_EMBEDDED_MTP_TYPES))
        for name, expected_type in TARGET_EMBEDDED_MTP_TYPES.items():
            self.assertEqual(block[name]["type"], expected_type, name)
            self.assertEqual(block[name]["shape"], EXPECTED_MTP_SHAPES[name], name)

    def test_external_mtp_layout(self) -> None:
        tensors = {item["name"]: item for item in self.snapshots["mtp"]["tensors"]}
        self.assertEqual(set(tensors), set(SIDECAR_MTP_TYPES))
        for name, expected_type in SIDECAR_MTP_TYPES.items():
            self.assertEqual(tensors[name]["type"], expected_type, name)
            if name in EXPECTED_MTP_SHAPES:
                self.assertEqual(tensors[name]["shape"], EXPECTED_MTP_SHAPES[name], name)
        self.assertEqual(tensors["token_embd.weight"]["shape"], [5120, 248320])
        self.assertEqual(tensors["output.weight"]["shape"], [5120, 248320])
        self.assertEqual(tensors["output_norm.weight"]["shape"], [5120])

    def test_tokenizers_match_but_templates_are_audited_independently(self) -> None:
        target = self.snapshots["target"]
        mtp = self.snapshots["mtp"]
        self.assertEqual(target["gguf"]["tokenizer_metadata_sha256"],
                         mtp["gguf"]["tokenizer_metadata_sha256"])
        target_template = target["metadata"]["tokenizer.chat_template"]
        mtp_template = mtp["metadata"]["tokenizer.chat_template"]
        self.assertNotEqual(target_template["utf8_sha256"], mtp_template["utf8_sha256"])
        self.assertEqual(target_template["utf8_size_bytes"], 9993)
        self.assertEqual(mtp_template["utf8_size_bytes"], 8945)

    def test_runtime_kernel_inventory_is_promoted(self) -> None:
        runtime = self.manifest["runtime"]
        self.assertEqual(runtime["status"], "kernel_ready")
        self.assertEqual(runtime["missing_cuda_formats"], [])
        self.assertEqual(set(runtime["supported_cuda_formats"]),
                         REQUIRED_CUDA_FORMATS)

    def test_qwen38_performance_report_uses_the_production_gate(self) -> None:
        target = next(item for item in self.manifest["artifacts"]
                      if item["role"] == "target")
        report = {
            "format": "ds4-qwen38-speed-v2",
            "model_sha256": target["sha256"],
            "engine_commit": "a" * 40,
            "build_flags": "synthetic",
            "backend": "CUDA",
            "hardware": "RTX 3090",
            "prefill_command": ["./ds4-bench", "--prefill"],
            "decode_command": ["./ds4-bench", "--decode"],
            "prefill_context": 4096,
            "decode_context": 4096,
            "prefill_tokens": 4096,
            "decode_tokens": 128,
            "prefill_tokens_per_second": 500.0,
            "decode_tokens_per_second": 15.0,
            "peak_memory_mib": 24000.0,
        }
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "speed.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            validated = validate_speed_report(
                path, model_sha256=target["sha256"], backend="CUDA",
                hardware="RTX 3090",
            )
        self.assertEqual(validated["format"], "ds4-qwen38-speed-v2")

    def test_qwen38_quality_entrypoints_are_present(self) -> None:
        for name in (
            "generate_qwen38_oracle.py",
            "verify_qwen38_run.py",
            "generate_qwen38_score.py",
            "generate_qwen38_prompt.py",
            "compare_qwen38_equivalence.py",
            "qwen38_speed_gate.py",
        ):
            self.assertTrue((QUALITY_ROOT / name).is_file(), name)

    def test_downloader_pins_both_artifacts(self) -> None:
        source = (PROJECT_ROOT / "download_model.sh").read_text(encoding="utf-8")
        self.assertIn(self.manifest["model"]["gguf_revision"], source)
        for artifact in self.manifest["artifacts"]:
            self.assertIn(artifact["filename"], source)
            self.assertIn(artifact["sha256"], source)

    def _local_runtime(self) -> tuple[Path, Path, Path] | None:
        target = PROJECT_ROOT / "gguf" / "Qwen3.8-27B-UD-Q4_K_S.gguf"
        mtp = PROJECT_ROOT / "gguf" / "mtp-Qwen3.8-27B-Q4_0.gguf"
        binary = PROJECT_ROOT / "ds4"
        if not (target.is_file() and mtp.is_file() and binary.is_file()):
            return None
        return target, mtp, binary

    def test_live_inspection_when_artifacts_are_present(self) -> None:
        runtime = self._local_runtime()
        if runtime is None:
            self.skipTest("local Qwen3.8 GGUF pair or ds4 binary is not present")
        target, mtp, binary = runtime
        result = subprocess.run(
            [str(binary), "-m", str(target), "--inspect", "--mtp-model", str(mtp)],
            cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Qwen3.8-27B-UD-Q4_K_S.gguf", result.stderr)
        self.assertIn("support model (Qwen35 NextN MTP", result.stdout)

    def test_cli_auto_mtp_selects_qwen38_sidecar(self) -> None:
        runtime = self._local_runtime()
        if runtime is None:
            self.skipTest("local Qwen3.8 GGUF pair or ds4 binary is not present")
        target, _mtp, binary = runtime
        result = subprocess.run(
            [str(binary), "-m", str(target), "--inspect", "--mtp"],
            cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("model: Qwen3.8-27B", result.stdout)
        self.assertIn("support model (Qwen35 NextN MTP", result.stdout)

    @unittest.skipUnless(os.environ.get("DS4_TEST_QWEN38_LIVE") == "1",
                         "set DS4_TEST_QWEN38_LIVE=1 for model-backed CLI checks")
    def test_cli_target_and_mtp_generate_after_kernel_promotion(self) -> None:
        runtime = self._local_runtime()
        if runtime is None:
            self.skipTest("local Qwen3.8 artifacts or DS4 CLI are not present")
        target, _mtp, binary = runtime
        commands = (
            [str(binary), "-m", str(target), "--cuda", "--nothink",
             "--temp", "0", "-p", "x", "-n", "2"],
            [str(binary), "-m", str(target), "--cuda", "--mtp", "--nothink",
             "--temp", "0", "-p", "x", "-n", "2"],
        )
        for command in commands:
            with self.subTest(mtp="--mtp" in command):
                result = subprocess.run(
                    command, cwd=PROJECT_ROOT, text=True, capture_output=True,
                    timeout=180, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("CUDA kernels are still required", result.stderr)
                self.assertTrue(result.stdout.strip())

    @unittest.skipUnless(os.environ.get("DS4_TEST_QWEN38_PERF") == "1",
                         "set DS4_TEST_QWEN38_PERF=1 for the CLI/server MTP gate")
    def test_cli_and_server_mtp_performance_gate(self) -> None:
        runtime = self._local_runtime()
        server = PROJECT_ROOT / "ds4-server"
        if runtime is None or not server.is_file():
            self.skipTest("local Qwen3.8 artifacts, CLI, or server are not present")
        target, _mtp, binary = runtime
        minimum_tps = self.manifest["gates"][
            "mtp_short_generation_tokens_per_second_min"
        ]

        cli = subprocess.run(
            [str(binary), "-m", str(target), "--cuda", "--mtp",
             "--ctx", "22593", "--nothink", "--temp", "0",
             "-p", "ciao", "-n", "32"],
            cwd=PROJECT_ROOT, text=True, capture_output=True,
            timeout=600, check=False,
        )
        self.assertEqual(cli.returncode, 0, cli.stderr)
        timing = re.search(
            r"ds4: prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s",
            cli.stderr + cli.stdout,
        )
        self.assertIsNotNone(timing, cli.stderr[-4000:])
        self.assertGreaterEqual(float(timing.group(2)), minimum_tps)

        with tempfile.TemporaryDirectory() as raw_results:
            experiment_id = "qwen38-mtp-short-safe-allocation-regression"
            harness = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "tools" / "perf_harness.py"),
                 "server-curve", "--id", experiment_id,
                 "--model", str(target), "--binary", str(server),
                 "--suite", "mtp-short-regression", "--mtp",
                 "--context-alloc", "22593", "--no-thinking",
                 "--prompt-pattern", "technical-explanation",
                 "--minimum-completion-tokens", "32", "--repetitions", "2",
                 "--hypothesis",
                 "Qwen3.8 short MTP remains fast with the 22K-safe allocation",
                 "--baseline-run", "--results", raw_results],
                cwd=PROJECT_ROOT, text=True, capture_output=True,
                timeout=1200, check=False,
            )
            self.assertEqual(harness.returncode, 0,
                             harness.stderr + harness.stdout)
            record = json.loads(
                (Path(raw_results) / experiment_id / "experiment.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(record["context_curve"]["status"], "PASS")
        self.assertGreaterEqual(
            record["workloads"][0]["metrics"]["gen_steady_tps"]["min"],
            minimum_tps,
        )
        self.assertTrue(record["correctness"]["deterministic_outputs"])
        self.assertTrue(all(
            row["completion_tokens"] >= 32
            for row in record["workloads"][0]["raw_rows"]
        ))
        command = record["provenance"]["command"]
        self.assertEqual(command[command.index("--ctx") + 1], "22593")

    def test_qwen36_mtp_is_rejected_for_qwen38(self) -> None:
        runtime = self._local_runtime()
        old_mtp = PROJECT_ROOT / "gguf" / "mtp-Qwen3.6-27B-Q4_0.gguf"
        if runtime is None or not old_mtp.is_file():
            self.skipTest("local Qwen3.8 target or Qwen3.6 mismatch sidecar is absent")
        target, _mtp, binary = runtime
        result = subprocess.run(
            [str(binary), "-m", str(target), "--inspect", "--mtp-model", str(old_mtp)],
            cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=60, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MTP generation does not match", result.stderr)

    def test_server_model_id_cache_and_agentic_gates_cover_qwen38(self) -> None:
        server = (PROJECT_ROOT / "ds4_server.c").read_text(encoding="utf-8")
        engine = (PROJECT_ROOT / "ds4.c").read_text(encoding="utf-8")
        agent = (PROJECT_ROOT / "ds4_agent.c").read_text(encoding="utf-8")
        kvstore = (PROJECT_ROOT / "ds4_kvstore.c").read_text(encoding="utf-8")
        checkpoint = (PROJECT_ROOT / "tests" / "test_agentic_checkpoint.c").read_text(
            encoding="utf-8")
        checkpoint_runner = (PROJECT_ROOT / "tests" / "run_agentic_checkpoint.sh").read_text(
            encoding="utf-8")
        agent_ssd_runner = (PROJECT_ROOT / "tests" / "run_agent_ssd_live.ps1").read_text(
            encoding="utf-8")
        agent_server_launcher = (
            PROJECT_ROOT / "agent" / "start-ds4-server.bat"
        ).read_text(encoding="utf-8")
        agent_profiler = (
            PROJECT_ROOT / "tools" / "profile_agent_dsml_story.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"Qwen3.8-27B-UD-Q4_K_S.gguf"', server)
        self.assertIn('.canonical_id = "Qwen3.8-27B-UD-Q4_K_S"', engine)
        self.assertIn('.ds4/qwen38-ud-q4ks-kv', engine)
        self.assertIn("SERVER_MODEL_SYNTAX_QWEN", server)
        self.assertIn("render_qwen_chat_prompt_text", server)
        self.assertIn("append_qwen_agentic_scope_instruction", server)
        self.assertIn("ds4_engine_model_descriptor(engine)", server)
        self.assertIn("DS4_MODEL_FAMILY_QWEN35", server)
        self.assertIn("ds4_engine_model_descriptor(ds4_engine *e)", engine)
        self.assertIn("bool ds4_engine_is_qwen(ds4_engine *e)", engine)
        self.assertIn("ds4_engine_is_qwen(engine)", agent)
        self.assertIn("ds4_engine_is_qwen_q4_k_s(w->engine)", agent)
        self.assertIn("ds4_engine_is_qwen_q4_k_s(engine)", kvstore)
        self.assertIn("ds4_engine_is_qwen_q4_k_s(engine)", checkpoint)
        self.assertIn("mtp-Qwen3.8-27B-Q4_0.gguf", checkpoint_runner)
        self.assertIn("--model-id $ModelId", agent_ssd_runner)
        self.assertIn("--ctx 22593", agent_server_launcher)
        self.assertNotIn("--ctx 24768", agent_server_launcher)
        self.assertIn(
            'default="Qwen3.6-27B-Q4_K_S"', agent_profiler
        )

    @unittest.skipUnless(os.environ.get("DS4_TEST_QWEN38_SHA") == "1",
                         "set DS4_TEST_QWEN38_SHA=1 for the 16.7 GB checksum gate")
    def test_full_local_artifact_checksums(self) -> None:
        by_role = {item["role"]: item for item in self.manifest["artifacts"]}
        paths = {
            "target": PROJECT_ROOT / "gguf" / by_role["target"]["filename"],
            "mtp": PROJECT_ROOT / "gguf" / by_role["mtp"]["filename"],
        }
        for role, path in paths.items():
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, by_role[role]["size_bytes"])
            self.assertEqual(sha256_file(path), by_role[role]["sha256"])


if __name__ == "__main__":
    unittest.main()
