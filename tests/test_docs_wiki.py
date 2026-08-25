#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GROUPS = ("architecture", "agentic", "performance", "roadmaps", "hardware")


class DocsWikiTests(unittest.TestCase):
    def test_catalog_ranges_point_to_declared_headings(self) -> None:
        catalog = json.loads((DOCS / "catalog.json").read_text(encoding="utf-8"))
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

    def test_root_and_index_of_indexes_reach_every_group(self) -> None:
        master = (DOCS / "INDEX.md").read_text(encoding="utf-8")
        compact = (DOCS / "indexes" / "INDEX.md").read_text(encoding="utf-8")
        for group in GROUPS:
            self.assertIn(f"{group}/_INDEX.md", master)
            self.assertIn(f"../{group}/_INDEX.md", compact)
        self.assertIn("research/INDEX.md", master)
        self.assertIn("../research/INDEX.md", compact)

    def test_no_markdown_is_orphaned_or_flat_at_docs_root(self) -> None:
        self.assertEqual(
            sorted(path.name for path in DOCS.glob("*.md")),
            ["INDEX.md"],
        )
        catalog = json.loads((DOCS / "catalog.json").read_text(encoding="utf-8"))
        indexed = {
            document["path"]
            for group in catalog["groups"]
            for document in group["documents"]
        }
        expected = {
            path.relative_to(ROOT).as_posix()
            for group in GROUPS
            for path in (DOCS / group).glob("*.md")
            if path.name not in {"_INDEX.md", "INDEX.md"}
        }
        self.assertEqual(indexed, expected)

    def test_all_local_markdown_links_resolve(self) -> None:
        link = re.compile(r"\[[^]]*\]\(([^)]+)\)")
        for path in DOCS.rglob("*.md"):
            for target in link.findall(path.read_text(encoding="utf-8")):
                target = target.strip().split("#", 1)[0]
                if not target or re.match(r"^[a-z]+://", target):
                    continue
                resolved = (path.parent / unquote(target)).resolve()
                self.assertTrue(resolved.exists(), f"{path}: broken link {target}")


if __name__ == "__main__":
    unittest.main()
