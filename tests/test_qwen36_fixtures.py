#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import io
import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "gguf-tools"
QUALITY_ROOT = TOOLS_ROOT / "quality-testing"
DATA_ROOT = QUALITY_ROOT / "data" / "qwen36-27b"
sys.path.insert(0, str(QUALITY_ROOT))

from qwen36_fixtures import (  # noqa: E402
    FixtureError, ensure_staging, generate_long_content, inventory_files,
    sha256_file, validate_manifest, write_json,
)
import generate_qwen36_oracle as oracle_generator  # noqa: E402
import inspect_qwen36_gguf as gguf_inspector  # noqa: E402
import verify_qwen36_run as run_verifier  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, rendered, add_special_tokens=False):
        return [1, 2, 3, 4]

    def decode(self, token_ids, skip_special_tokens=False):
        return "fixture continuation"


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_synthetic_gguf(path: Path, *, duplicate_tensor: bool = False, metadata_type: int = 8) -> None:
    metadata = bytearray()
    metadata += gguf_string("general.architecture") + struct.pack("<I", metadata_type)
    if metadata_type == 8:
        metadata += gguf_string("qwen35")
    metadata += gguf_string("qwen35.context_length") + struct.pack("<II", 4, 262144)
    names = ["weight", "weight" if duplicate_tensor else "bias"]
    tensors = bytearray()
    for index, name in enumerate(names):
        tensors += gguf_string(name)
        tensors += struct.pack("<IQIQ", 1, 1, 0, index * 4)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(names), 2) + metadata + tensors
    padding = b"\0" * ((32 - len(header) % 32) % 32)
    path.write_bytes(header + padding + b"\0" * 8)


class Qwen36FixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "prompts").mkdir()
        shutil.copy(DATA_ROOT / "prompts" / "cases.json", self.root / "prompts" / "cases.json")
        self.manifest_path = self.root / "manifest.json"
        shutil.copy(DATA_ROOT / "manifest.json", self.manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for artifact in manifest["artifacts"]:
            artifact["metadata"]["header_inspection"] = "not_verified"
            artifact["metadata"]["snapshot_file"] = None
            artifact["metadata"]["snapshot_sha256"] = None
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def load_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def test_valid_manifest(self) -> None:
        manifest, corpus = validate_manifest(self.manifest_path)
        self.assertEqual(manifest["verification"]["status"], "not_verified")
        self.assertGreaterEqual(len(corpus["cases"]), 9)

    def test_wrong_sha_is_rejected(self) -> None:
        artifact_file = self.root / "model.gguf"
        artifact_file.write_bytes(b"synthetic fixture, not a model")
        manifest = self.load_manifest()
        artifact = manifest["artifacts"][0]
        artifact["local_path"] = artifact_file.name
        artifact["size_bytes"] = artifact_file.stat().st_size
        artifact["sha256"] = "0" * 64
        self.save_manifest(manifest)
        with self.assertRaisesRegex(FixtureError, "SHA-256 mismatch"):
            validate_manifest(self.manifest_path)

    def test_matching_synthetic_sha_is_accepted(self) -> None:
        artifact_file = self.root / "model.gguf"
        artifact_file.write_bytes(b"synthetic fixture, not a model")
        manifest = self.load_manifest()
        artifact = manifest["artifacts"][0]
        artifact["local_path"] = artifact_file.name
        artifact["size_bytes"] = artifact_file.stat().st_size
        artifact["sha256"] = hashlib.sha256(artifact_file.read_bytes()).hexdigest()
        self.save_manifest(manifest)
        validate_manifest(self.manifest_path, require_artifacts=True)

    def test_missing_required_field_is_rejected(self) -> None:
        manifest = self.load_manifest()
        del manifest["model"]["context_length"]
        self.save_manifest(manifest)
        with self.assertRaisesRegex(FixtureError, "missing required field"):
            validate_manifest(self.manifest_path)

    def test_duplicate_case_id_is_rejected(self) -> None:
        corpus_path = self.root / "prompts" / "cases.json"
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        corpus["cases"][1]["id"] = corpus["cases"][0]["id"]
        corpus_path.write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(FixtureError, "duplicate case id"):
            validate_manifest(self.manifest_path)

    def test_missing_referenced_corpus_is_rejected(self) -> None:
        (self.root / "prompts" / "cases.json").unlink()
        with self.assertRaisesRegex(FixtureError, "corpus file does not exist"):
            validate_manifest(self.manifest_path)

    def test_long_prompt_is_deterministic_and_has_canaries(self) -> None:
        _, corpus = validate_manifest(self.manifest_path)
        case = next(item for item in corpus["cases"] if item["category"] == "long-canary")
        first = generate_long_content(case)
        second = generate_long_content(case)
        self.assertEqual(first, second)
        for canary in case["canaries"]:
            self.assertEqual(first.count(canary["value"]), 1)

    def test_staging_refuses_golden_directory(self) -> None:
        with self.assertRaisesRegex(FixtureError, "golden directory"):
            ensure_staging(self.root / "goldens" / "candidate")

    def test_staging_refuses_nonempty_unmarked_directory(self) -> None:
        path = self.root / "candidate"
        path.mkdir()
        (path / "user-file.txt").write_text("preserve me", encoding="utf-8")
        with self.assertRaisesRegex(FixtureError, "non-empty and unmarked"):
            ensure_staging(path)

    def test_oracle_run_uses_existing_fixture_layout(self) -> None:
        staging = self.root / "staging"
        fake_result = {
            "engine": "transformers",
            "engine_version": "test",
            "prompt_token_ids": [1, 2, 3, 4],
            "greedy_token_ids": [7] * 32,
            "greedy_text": "fixture continuation",
            "teacher_forced_source": "same-run greedy continuation",
            "teacher_forced": [{"token_id": 7, "logprob": 0.0}] * 32,
            "top_k": [[{"token_id": 7, "logprob": 0.0}]] * 32,
            "full_logits": None,
        }
        argv = [
            "generate_qwen36_oracle.py",
            "--manifest", str(self.manifest_path),
            "--oracle", "transformers",
            "--staging-dir", str(staging),
            "--run-id", "layout-test",
            "--engine-commit", "a" * 40,
            "--build-flags", "test build",
            "--backend", "CPU",
            "--hardware", "synthetic test host",
            "--dtype", "float32",
            "--case", "single_token_ascii",
        ]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(oracle_generator, "load_renderer", return_value=FakeTokenizer()), \
             mock.patch.object(oracle_generator, "create_transformers", return_value=object()), \
             mock.patch.object(oracle_generator, "generate_transformers", return_value=fake_result), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(oracle_generator.main(), 0)
        run = staging / "layout-test"
        self.assertTrue((run / "manifest.tsv").is_file())
        self.assertTrue((run / "prompts" / "single_token_ascii.txt").is_file())
        self.assertTrue((run / "continuations" / "single_token_ascii.txt").is_file())
        self.assertTrue((run / "responses" / "single_token_ascii.json").is_file())
        self.assertTrue((run / "logits").is_dir())
        index = json.loads((run / "index.json").read_text(encoding="utf-8"))
        difference = index["environment"]["target_difference"]
        self.assertTrue(difference["weights_differ"])
        self.assertTrue(difference["precision_differ"])
        self.assertFalse(difference["numeric_equivalence_expected"])

    def test_gguf_inspector_accepts_valid_synthetic_header(self) -> None:
        path = self.root / "model.gguf"
        write_synthetic_gguf(path)
        snapshot = gguf_inspector.inspect_gguf(path)
        self.assertEqual(snapshot["metadata"]["general.architecture"], "qwen35")
        self.assertEqual(snapshot["gguf"]["tensor_count"], 2)

    def test_gguf_inspector_rejects_truncated_header(self) -> None:
        path = self.root / "model.gguf"
        write_synthetic_gguf(path)
        path.write_bytes(path.read_bytes()[:20])
        with self.assertRaisesRegex(FixtureError, "truncated GGUF header"):
            gguf_inspector.inspect_gguf(path)

    def test_gguf_inspector_rejects_unknown_metadata_type(self) -> None:
        path = self.root / "model.gguf"
        write_synthetic_gguf(path, metadata_type=99)
        with self.assertRaisesRegex(FixtureError, "unsupported GGUF metadata type"):
            gguf_inspector.inspect_gguf(path)

    def test_gguf_inspector_rejects_duplicate_tensor(self) -> None:
        path = self.root / "model.gguf"
        write_synthetic_gguf(path, duplicate_tensor=True)
        with self.assertRaisesRegex(FixtureError, "duplicate GGUF tensor name"):
            gguf_inspector.inspect_gguf(path)

    def test_gguf_snapshot_detects_metadata_and_sha_mismatch(self) -> None:
        path = self.root / "model.gguf"
        write_synthetic_gguf(path)
        snapshot = gguf_inspector.inspect_gguf(path)
        artifact = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": snapshot["artifact"]["sha256"],
            "metadata": {"general.architecture": "wrong", "context_length": 262144},
        }
        with self.assertRaisesRegex(FixtureError, "general.architecture"):
            gguf_inspector.validate_against_manifest(snapshot, artifact)
        artifact["metadata"]["general.architecture"] = "qwen35"
        artifact["sha256"] = "0" * 64
        with self.assertRaisesRegex(FixtureError, "sha256 mismatch"):
            gguf_inspector.validate_against_manifest(snapshot, artifact)

    def test_run_verifier_checks_full_logits_and_inventory(self) -> None:
        run = self.root / "run"
        for name in ("prompts", "continuations", "responses", "logits"):
            (run / name).mkdir(parents=True, exist_ok=True)
        case_id = "single_token_ascii"
        rendered = b"<|im_start|>user\nA<|im_end|>\n"
        (run / "prompts" / f"{case_id}.txt").write_bytes(rendered)
        (run / "prompts" / f"{case_id}.bytes").write_bytes(rendered)
        write_json(run / "prompts" / f"{case_id}.tokens.json", [1, 2])
        write_json(run / "prompts" / f"{case_id}.case.json", {"id": case_id})
        (run / "continuations" / f"{case_id}.txt").write_text("x" * 32, encoding="utf-8")
        logits = {}
        for pass_name in ("greedy", "teacher"):
            path = run / "logits" / f"{case_id}.{pass_name}.f32"
            path.write_bytes(struct.pack("<64f", *([0.0] * 64)))
            logits["teacher_forced" if pass_name == "teacher" else "greedy"] = {
                "path": f"logits/{path.name}", "dtype": "float32-le",
                "shape": [32, 2], "row_stride_bytes": 8,
                "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            }
        response = {
            "prompt_token_ids": [1, 2], "upstream_render_token_ids": [1, 2],
            "greedy_token_ids": [7] * 32,
            "teacher_forced": [{"token_id": 7}] * 32,
            "top_k": [[{"token_id": 7}]] * 32,
            "full_logits": logits,
        }
        write_json(run / "responses" / f"{case_id}.json", response)
        index = {
            "format": "ds4-qwen36-oracle-v1",
            "environment": {"review_status": "generated_unreviewed", "parameters": {"steps": 32}},
            "cases": [{"id": case_id}],
            "files": inventory_files(run),
        }
        write_json(run / "index.json", index)
        data = run_verifier.verify_run(run, self.manifest_path, allow_partial=True)
        self.assertIn(case_id, data["responses"])
        (run / "logits" / f"{case_id}.greedy.f32").write_bytes(b"corrupt")
        with self.assertRaisesRegex(FixtureError, "logits size mismatch"):
            run_verifier.verify_run(run, self.manifest_path, allow_partial=True)


if __name__ == "__main__":
    unittest.main()
