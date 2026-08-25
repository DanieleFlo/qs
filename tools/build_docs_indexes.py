#!/usr/bin/env python3
"""Generate the top-level DS4 documentation wiki and semantic group indexes."""

from __future__ import annotations

import json
from pathlib import Path

from build_research_indexes import outline


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GROUPS = ("architecture", "agentic", "performance", "roadmaps", "hardware")
DESCRIPTIONS = {
    "architecture": "Architettura dei modelli, MTP e contratti di compatibilità.",
    "agentic": "API agentica, skill gerarchiche e validazione dei checkpoint.",
    "performance": "Contratti, misure, drift numerico e progressione delle ottimizzazioni.",
    "roadmaps": "Piani operativi e TODO per decoding, server e kernel engineering.",
    "hardware": "Inventario RTX 3090, caratteristiche Ampere e numerica CUDA.",
}


def group_index(relative: str) -> dict[str, object]:
    directory = DOCS / relative
    files = sorted(
        path
        for path in directory.glob("*.md")
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
                (
                    str(row["summary"])
                    for row in rows[1:]
                    if not str(row["summary"]).startswith("Sezione strutturale")
                ),
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
        documents.append(
            {"path": path.relative_to(ROOT).as_posix(), "sections": rows}
        )
    target = directory / "_INDEX.md"
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "group": relative,
        "index": target.relative_to(ROOT).as_posix(),
        "documents": documents,
    }


def main() -> int:
    catalog = [group_index(group) for group in GROUPS]
    master = [
        "# Wiki tecnica DS4",
        "",
        "Punto di ingresso unico alla documentazione tecnica, organizzata per tema e navigabile fino alle sezioni dei singoli file.",
        "",
        "Accesso alternativo compatto: [indice degli indici](indexes/INDEX.md).",
        "",
        "## Percorso consigliato",
        "",
        "1. Scegliere il macro-argomento dall'elenco seguente.",
        "2. Usare il relativo `_INDEX.md` per cercare titolo, sintesi e intervallo di righe.",
        "3. Aprire il documento sorgente e verificare il contenuto nel contesto indicato.",
        "4. Per fonti esterne e tecniche trasferibili, partire dalla knowledge base `research` e controllarne provenienza ed evidenza.",
        "",
        "## Macro-argomenti",
        "",
    ]
    for item in catalog:
        relative = Path(item["index"]).relative_to("docs").as_posix()
        master.append(
            f"- [{item['group']}]({relative}) — {DESCRIPTIONS[item['group']]}"
        )
    master.extend(
        [
            "- [research](research/INDEX.md) — Paper, piattaforme, fork, guide e fonti CUDA con catalogo e lock di provenienza.",
            "",
            "## Regole di costruzione della wiki",
            "",
            "Queste regole generalizzano la struttura già adottata in `docs/research`:",
            "",
            "1. **Un tema per directory.** Ogni documento appartiene a un macro-argomento stabile; i file non restano dispersi nella radice di `docs`.",
            "2. **Un ingresso curato e un indice compatto.** `INDEX.md` spiega il percorso di lettura; `indexes/INDEX.md` collega rapidamente tutti gli indici.",
            "3. **Un `_INDEX.md` per gruppo.** L'indice locale elenca ogni documento, ne sintetizza lo scopo e rimanda al file originale.",
            "4. **Heading come indirizzi semantici.** Ogni sezione significativa usa heading Markdown gerarchici; l'indice registra titolo e intervallo `righe X–Y`.",
            "5. **Sintesi ricavata dal contenuto.** La descrizione dell'indice deriva dalla prima frase utile della sezione, non soltanto dal nome del file.",
            "6. **Indici generati, contenuti modificati a mano.** I file `_INDEX.md` e i cataloghi JSON non si editano direttamente: si rigenerano dopo aver cambiato heading o documenti.",
            "7. **Link relativi e verificabili.** I collegamenti interni devono sopravvivere agli spostamenti ed essere controllati automaticamente.",
            "8. **Provenienza esplicita.** Le ricerche esterne conservano repository, snapshot o commit, data di audit e grado dell'evidenza; manifesti e lock separano fonti da interpretazioni.",
            "9. **Catalogo machine-readable.** `catalog.json` replica gruppi, documenti, sezioni, righe e sintesi per controlli e strumenti futuri.",
            "10. **Nessun documento orfano.** Ogni Markdown tematico deve essere raggiungibile dall'indice del proprio gruppo e dall'indice madre.",
            "",
            "## Manutenzione",
            "",
            "- Rigenerare tutta la wiki con `make docs-index`.",
            "- Verificare intervalli, copertura e link con `make docs-check`.",
            "- Per il solo corpus di ricerca restano disponibili `make research-index` e `make research-check`.",
            "- Il catalogo generale è [catalog.json](catalog.json); quello specialistico è [research/catalog.json](research/catalog.json).",
            "",
        ]
    )
    (DOCS / "INDEX.md").write_text("\n".join(master), encoding="utf-8")
    (DOCS / "catalog.json").write_text(
        json.dumps(
            {"schema_version": 1, "groups": catalog},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    indexes = DOCS / "indexes"
    indexes.mkdir(exist_ok=True)
    lines = [
        "# Indice degli indici",
        "",
        "Accesso compatto a tutti gli indici della wiki DS4.",
        "",
        "- [Indice principale](../INDEX.md)",
    ]
    for item in catalog:
        relative = Path("..") / Path(item["index"]).relative_to("docs")
        lines.append(f"- [{item['group']}]({relative.as_posix()})")
    lines.extend(
        [
            "- [research](../research/INDEX.md)",
            "- [indici research](../research/indexes/INDEX.md)",
        ]
    )
    (indexes / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"indexed {sum(len(item['documents']) for item in catalog)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
