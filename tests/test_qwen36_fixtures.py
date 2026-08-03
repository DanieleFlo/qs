#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import io
import json
import shutil
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

from qwen36_fixtures import FixtureError, ensure_staging, generate_long_content, validate_manifest  # noqa: E402
import generate_qwen36_oracle as oracle_generator  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, rendered, add_special_tokens=False):
        return [1, 2, 3, 4]

    def decode(self, token_ids, skip_special_tokens=False):
        return "fixture continuation"


class Qwen36FixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "prompts").mkdir()
        shutil.copy(DATA_ROOT / "prompts" / "cases.json", self.root / "prompts" / "cases.json")
        self.manifest_path = self.root / "manifest.json"
        shutil.copy(DATA_ROOT / "manifest.json", self.manifest_path)

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
             mock.patch.object(oracle_generator, "generate_transformers", return_value=fake_result), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(oracle_generator.main(), 0)
        run = staging / "layout-test"
        self.assertTrue((run / "manifest.tsv").is_file())
        self.assertTrue((run / "prompts" / "single_token_ascii.txt").is_file())
        self.assertTrue((run / "continuations" / "single_token_ascii.txt").is_file())
        self.assertTrue((run / "responses" / "single_token_ascii.json").is_file())
        self.assertTrue((run / "logits").is_dir())


if __name__ == "__main__":
    unittest.main()
