#!/usr/bin/env python3
"""Fetch the curated arXiv HTML corpus and build readable semantic mirrors."""

from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "docs" / "research" / "papers"


class SemanticHTML(HTMLParser):
    BLOCKS = {"p", "li", "pre", "blockquote", "figcaption"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.capture: str | None = None
        self.level = 0
        self.buffer: list[str] = []
        self.lines: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "math", "svg", "nav"}:
            self.skip += 1
            return
        if self.skip:
            return
        heading = re.fullmatch(r"h([1-6])", tag)
        if heading:
            self.flush()
            self.capture = "heading"
            self.level = int(heading.group(1))
        elif tag in self.BLOCKS:
            self.flush()
            self.capture = tag
        elif tag == "br" and self.capture:
            self.buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "math", "svg", "nav"} and self.skip:
            self.skip -= 1
            return
        if not self.skip and (tag in self.BLOCKS or re.fullmatch(r"h[1-6]", tag)):
            self.flush()

    def handle_data(self, data: str) -> None:
        if not self.skip and self.capture:
            self.buffer.append(data)

    def flush(self) -> None:
        if not self.capture:
            return
        value = html.unescape(" ".join(self.buffer))
        value = re.sub(r"[ \t\r\f\v]+", " ", value)
        value = re.sub(r" *\n *", "\n", value).strip()
        if value:
            if self.capture == "heading":
                self.lines.extend(["#" * self.level + " " + value, ""])
            elif self.capture == "li":
                self.lines.extend(["- " + value.replace("\n", " "), ""])
            elif self.capture == "pre":
                self.lines.extend(["```text", value, "```", ""])
            else:
                self.lines.extend([value, ""])
        self.capture = None
        self.buffer = []


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DS4 research corpus; contact via repository"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_paper(arxiv_id: str) -> tuple[bytes, str, str]:
    sources = (
        (f"https://arxiv.org/html/{arxiv_id}", "arxiv-html"),
        (f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}", "ar5iv-fallback"),
    )
    errors = []
    for url, renderer in sources:
        try:
            return fetch(url), url, renderer
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    manifest_path = PAPERS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    html_dir, text_dir = PAPERS / "html", PAPERS / "text"
    html_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    lock_path = PAPERS / "corpus-lock.json"
    previous_artifacts: dict[str, dict[str, object]] = {}
    if lock_path.is_file():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        previous_artifacts = {
            str(item["id"]): item for item in previous.get("artifacts", [])
        }
    lock = {"schema_version": 1, "artifacts": []}
    failures = []
    for paper in manifest["papers"]:
        name = f"{paper['id']}-{paper['slug']}"
        html_path = html_dir / f"{name}.html"
        canonical_url = f"https://arxiv.org/abs/{paper['id']}"
        try:
            if args.refresh or not html_path.is_file():
                payload, rendered_url, renderer = fetch_paper(paper["id"])
                html_path.write_bytes(payload)
            else:
                payload = html_path.read_bytes()
                prior = previous_artifacts.get(paper["id"])
                if not prior:
                    raise RuntimeError(
                        f"local artifact {html_path} has no provenance; rerun with --refresh"
                    )
                rendered_url = str(prior["rendered_url"])
                renderer = str(prior["renderer"])
            parser_html = SemanticHTML()
            parser_html.feed(payload.decode("utf-8", "replace"))
            parser_html.flush()
            front = [
                f"# {paper['title']}", "",
                f"- arXiv: [{paper['id']}](https://arxiv.org/abs/{paper['id']})",
                f"- HTML locale: [../html/{html_path.name}](../html/{html_path.name})",
                f"- Uso DS4: {paper['why']}",
                "- Nota: testo estratto meccanicamente per ricerca e riferimenti di riga; formule e figure vanno verificate nell'HTML locale.",
                "", "## Testo estratto", "",
            ]
            text_path = text_dir / f"{name}.md"
            text_path.write_text("\n".join(front + parser_html.lines).rstrip() + "\n", encoding="utf-8")
            lock["artifacts"].append({
                "id": paper["id"], "canonical_url": canonical_url,
                "rendered_url": rendered_url, "renderer": renderer,
                "html": html_path.relative_to(ROOT).as_posix(),
                "semantic_text": text_path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            })
            print(f"fetched {paper['id']}: {len(payload)} bytes")
        except Exception as exc:  # keep the rest of the corpus useful
            failures.append({"id": paper["id"], "url": canonical_url, "error": str(exc)})
            print(f"FAILED {paper['id']}: {exc}")
    lock["failures"] = failures
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
