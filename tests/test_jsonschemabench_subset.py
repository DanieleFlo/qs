#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "import_jsonschemabench_subset",
    ROOT / "tools" / "import_jsonschemabench_subset.py",
)
assert SPEC and SPEC.loader
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


class JSONSchemaBenchSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.record = json.loads(
            (ROOT / "performance" / "jsonschemabench-subset.json").read_text(
                encoding="utf-8"
            )
        )

    def test_subset_is_pinned_and_separates_support(self) -> None:
        self.assertEqual(
            self.record["source"]["commit"], IMPORTER.SOURCE_COMMIT
        )
        self.assertEqual(
            self.record["source"]["repository"], IMPORTER.SOURCE_REPOSITORY
        )
        self.assertEqual(
            self.record["source"]["hugging_face"], IMPORTER.SOURCE_DATASET
        )
        self.assertEqual(self.record["source"]["categories"], list(IMPORTER.CATEGORIES))
        self.assertEqual(len(self.record["supported"]), 33)
        self.assertEqual(len(self.record["unsupported"]), 16)
        self.assertTrue(all(
            not item["unsupported_reasons"] for item in self.record["supported"]
        ))
        self.assertTrue(all(
            item["unsupported_reasons"] for item in self.record["unsupported"]
        ))
        coverage = self.record["coverage"]
        self.assertEqual(
            coverage["examined"],
            coverage["examined_supported"] + coverage["examined_unsupported"],
        )
        self.assertEqual(coverage["examined"], 9558)
        self.assertEqual(
            coverage["examined"],
            coverage["selection_eligible"] + coverage["skipped_oversize"],
        )
        self.assertAlmostEqual(
            coverage["examined_support_rate"],
            coverage["examined_supported"] / coverage["examined"],
        )
        self.assertEqual(coverage["selection_supported_fraction"], 33 / 49)
        self.assertNotIn("selected_support_rate", coverage)
        tiers = self.record["tiers"]
        self.assertEqual(len(tiers["smoke"]), 12)
        self.assertEqual(len(tiers["safety"]), 32)
        self.assertEqual(
            tiers["regressions"], list(IMPORTER.REGRESSION_IDS)
        )
        supported_ids = {item["id"] for item in self.record["supported"]}
        self.assertTrue(set(tiers["smoke"]).issubset(supported_ids))
        self.assertTrue(set(tiers["safety"]).issubset(supported_ids))
        self.assertTrue(set(tiers["regressions"]).issubset(supported_ids))
        self.assertGreaterEqual(
            len({item["category"] for item in self.record["supported"]}), 9
        )

    def test_vendored_schema_hashes_are_canonical(self) -> None:
        for item in self.record["supported"] + self.record["unsupported"]:
            digest = hashlib.sha256(
                IMPORTER.canonical_bytes(item["schema"])
            ).hexdigest()
            self.assertEqual(digest, item["sha256"], item["id"])

    def test_supported_subset_covers_roadmap_schema_features(self) -> None:
        features = {
            feature
            for item in self.record["supported"]
            for feature in item["features"]
        }
        self.assertTrue({
            "type", "properties", "required", "additionalProperties", "items",
            "enum", "const", "oneOf", "anyOf", "allOf", "minItems",
            "maxItems", "minProperties", "maxProperties", "minimum", "maximum",
            "exclusiveMinimum", "multipleOf", "uniqueItems", "maxLength",
        }.issubset(features))

    @unittest.skipUnless(
        importlib.util.find_spec("jsonschema"),
        "install performance/jsonschemabench-requirements.txt",
    )
    def test_selected_source_schemas_pass_independent_meta_validation(self) -> None:
        import jsonschema

        for item in self.record["supported"] + self.record["unsupported"]:
            with self.subTest(schema=item["id"]):
                validator = jsonschema.validators.validator_for(item["schema"])
                validator.check_schema(item["schema"])


if __name__ == "__main__":
    unittest.main()
