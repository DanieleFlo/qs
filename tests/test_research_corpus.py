#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "research"


class ResearchCorpusTests(unittest.TestCase):
    def test_manifest_lock_and_downloaded_hashes_are_complete(self) -> None:
        manifest = json.loads((RESEARCH / "papers" / "manifest.json").read_text(encoding="utf-8"))
        lock = json.loads((RESEARCH / "papers" / "corpus-lock.json").read_text(encoding="utf-8"))
        expected = {paper["id"] for paper in manifest["papers"]}
        actual = {artifact["id"] for artifact in lock["artifacts"]}
        self.assertEqual(actual, expected)
        self.assertEqual(lock["failures"], [])
        for artifact in lock["artifacts"]:
            path = ROOT / artifact["html"]
            payload = path.read_bytes()
            self.assertGreater(len(payload), 50_000, path)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), artifact["sha256"])
            self.assertIn(artifact["renderer"], {"arxiv-html", "ar5iv-fallback"})

    def test_catalog_ranges_point_to_the_declared_headings(self) -> None:
        catalog = json.loads((RESEARCH / "catalog.json").read_text(encoding="utf-8"))
        for group in catalog["groups"]:
            self.assertTrue((ROOT / group["index"]).is_file())
            for document in group["documents"]:
                path = ROOT / document["path"]
                lines = path.read_text(encoding="utf-8").splitlines()
                for section in document["sections"]:
                    start, end = int(section["start"]), int(section["end"])
                    self.assertGreaterEqual(start, 1)
                    self.assertLessEqual(start, end)
                    self.assertLessEqual(end, len(lines))
                    self.assertIn(str(section["title"]), lines[start - 1])

    def test_every_authored_repository_card_has_at_most_one_external_link(self) -> None:
        external = re.compile(r"\]\(https?://")
        for group in ("platforms", "forks"):
            for path in (RESEARCH / group).glob("*.md"):
                if path.name.startswith("_"):
                    continue
                self.assertLessEqual(len(external.findall(path.read_text(encoding="utf-8"))), 1, path)

    def test_repository_lock_uses_full_commits_and_evidence_classes(self) -> None:
        lock = json.loads((RESEARCH / "sources-lock.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(lock["repositories"]), 10)
        for source in lock["repositories"]:
            self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(source["evidence"])

    def test_master_and_index_of_indexes_reach_every_group(self) -> None:
        master = (RESEARCH / "INDEX.md").read_text(encoding="utf-8")
        meta = (RESEARCH / "indexes" / "INDEX.md").read_text(encoding="utf-8")
        for group in ("papers/text", "platforms", "forks", "guides", "cuda"):
            self.assertIn(f"{group}/_INDEX.md", master)
            self.assertIn(f"../{group}/_INDEX.md", meta)

    def test_all_local_markdown_links_resolve(self) -> None:
        link = re.compile(r"\[[^]]*\]\(([^)]+)\)")
        for path in RESEARCH.rglob("*.md"):
            for target in link.findall(path.read_text(encoding="utf-8")):
                target = target.strip().split("#", 1)[0]
                if not target or re.match(r"^[a-z]+://", target):
                    continue
                resolved = (path.parent / unquote(target)).resolve()
                self.assertTrue(resolved.exists(), f"{path}: broken link {target}")


if __name__ == "__main__":
    unittest.main()
