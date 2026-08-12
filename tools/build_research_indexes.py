#!/usr/bin/env python3
"""Generate hierarchical semantic indexes with stable, verified line ranges."""

from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "research"
GROUPS = ("papers/text", "platforms", "forks", "guides", "cuda")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def sentence(lines: list[str], start: int, end: int) -> str:
    for value in lines[start:end]:
        value = value.strip()
        if not value or value.startswith(("#", "- ", "```", "|")):
            continue
        value = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", value)
        value = re.sub(r"\s+", " ", value)
        first = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)[0]
        return first[:220] + ("…" if len(first) > 220 else "")
    return "Sezione strutturale; consultare il contenuto locale indicato."


def outline(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headings = []
    for number, value in enumerate(lines, 1):
        match = HEADING.match(value)
        if match:
            headings.append((number, len(match.group(1)), match.group(2)))
    rows = []
    for index, (start, level, title) in enumerate(headings):
        end = len(lines)
        for next_start, next_level, _ in headings[index + 1:]:
            if next_level <= level:
                end = next_start - 1
                break
        rows.append({
            "title": title, "level": level, "start": start, "end": end,
            "summary": sentence(lines, start, end),
        })
    return rows


def group_index(relative: str) -> dict[str, object]:
    directory = RESEARCH / relative
    files = sorted(
        path for path in directory.glob("*.md")
        if path.name not in {"_INDEX.md", "INDEX.md"}
    )
    lines = [f"# Indice: {relative}", "", "Indice generato; non modificare a mano.", ""]
    documents = []
    for path in files:
        rows = outline(path)
        title = rows[0]["title"] if rows else path.stem
        summary = "Documento senza heading indicizzabili."
        if rows:
            summary = next(
                (str(row["summary"]) for row in rows[1:]
                 if not str(row["summary"]).startswith("Sezione strutturale") and
                 str(row["summary"]) != "Repository ufficiale."),
                str(rows[0]["summary"]),
            )
        lines.extend([f"## [{title}]({path.name})", "", str(summary), ""])
        for row in rows[1:]:
            indent = "  " * max(0, int(row["level"]) - 2)
            lines.append(
                f"{indent}- righe {row['start']}–{row['end']}: "
                f"**{row['title']}** — {row['summary']}"
            )
        lines.append("")
        documents.append({"path": path.relative_to(ROOT).as_posix(), "sections": rows})
    target = directory / "_INDEX.md"
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"group": relative, "index": target.relative_to(ROOT).as_posix(), "documents": documents}


def main() -> int:
    catalog = [group_index(group) for group in GROUPS]
    master = [
        "# Knowledge base per kernel engineering", "",
        "Punto di ingresso per cercare soluzioni già implementate prima di progettare un kernel DS4.", "",
        "Accesso alternativo compatto: [indice degli indici](indexes/INDEX.md).", "",
        "## Percorso consigliato", "",
        "1. Cercare il problema nell'indice tematico e nelle schede piattaforma.",
        "2. Verificare i fork RTX 3090 e il loro livello di evidenza.",
        "3. Leggere il paper locale pertinente usando gli intervalli di righe.",
        "4. Tornare al codice upstream fissando commit, shape, formato numerico e hardware.", "",
        "## Indici di gruppo", "",
    ]
    descriptions = {
        "papers/text": "Testo semantico dei paper arXiv archiviati localmente.",
        "platforms": "Architettura e kernel delle maggiori piattaforme di inferenza locale.",
        "forks": "Fork e raccolte con ottimizzazioni consumer/RTX 3090, classificati per evidenza.",
        "guides": "Mappe trasversali problema → fonti → implementazioni da ispezionare.",
        "cuda": "Comandi, strumenti, primitive sm_86 e ipotesi CUDA collegate alle misure DS4.",
    }
    for item in catalog:
        rel = Path(item["index"]).relative_to("docs/research").as_posix()
        master.append(f"- [{item['group']}]({rel}) — {descriptions[item['group']]}")
    master.extend([
        "", "## Provenienza", "",
        "- [Manifesto paper](papers/manifest.json)",
        "- [Lock del corpus](papers/corpus-lock.json)",
        "- [Lock delle piattaforme e dei fork](sources-lock.json)",
        "- [Catalogo macchina](catalog.json)", "",
        "Rigenerare con `make research-index`; verificare con `make research-check`.", "",
    ])
    (RESEARCH / "INDEX.md").write_text("\n".join(master), encoding="utf-8")
    (RESEARCH / "catalog.json").write_text(
        json.dumps({"schema_version": 1, "groups": catalog}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    indexes = RESEARCH / "indexes"
    indexes.mkdir(exist_ok=True)
    lines = ["# Indice degli indici", "", "Accesso compatto a tutti gli indici della knowledge base.", "", "- [Indice principale](../INDEX.md)"]
    for item in catalog:
        rel = Path("..") / Path(item["index"]).relative_to("docs/research")
        lines.append(f"- [{item['group']}]({rel.as_posix()})")
    (indexes / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"indexed {sum(len(item['documents']) for item in catalog)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
