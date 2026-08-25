# Wiki tecnica DS4

Punto di ingresso unico alla documentazione tecnica, organizzata per tema e navigabile fino alle sezioni dei singoli file.

Accesso alternativo compatto: [indice degli indici](indexes/INDEX.md).

## Percorso consigliato

1. Scegliere il macro-argomento dall'elenco seguente.
2. Usare il relativo `_INDEX.md` per cercare titolo, sintesi e intervallo di righe.
3. Aprire il documento sorgente e verificare il contenuto nel contesto indicato.
4. Per fonti esterne e tecniche trasferibili, partire dalla knowledge base `research` e controllarne provenienza ed evidenza.

## Macro-argomenti

- [architecture](architecture/_INDEX.md) — Architettura dei modelli, MTP e contratti di compatibilità.
- [agentic](agentic/_INDEX.md) — API agentica, skill gerarchiche e validazione dei checkpoint.
- [performance](performance/_INDEX.md) — Contratti, misure, drift numerico e progressione delle ottimizzazioni.
- [roadmaps](roadmaps/_INDEX.md) — Piani operativi e TODO per decoding, server e kernel engineering.
- [hardware](hardware/_INDEX.md) — Inventario RTX 3090, caratteristiche Ampere e numerica CUDA.
- [research](research/INDEX.md) — Paper, piattaforme, fork, guide e fonti CUDA con catalogo e lock di provenienza.

## Regole di costruzione della wiki

Queste regole generalizzano la struttura già adottata in `docs/research`:

1. **Un tema per directory.** Ogni documento appartiene a un macro-argomento stabile; i file non restano dispersi nella radice di `docs`.
2. **Un ingresso curato e un indice compatto.** `INDEX.md` spiega il percorso di lettura; `indexes/INDEX.md` collega rapidamente tutti gli indici.
3. **Un `_INDEX.md` per gruppo.** L'indice locale elenca ogni documento, ne sintetizza lo scopo e rimanda al file originale.
4. **Heading come indirizzi semantici.** Ogni sezione significativa usa heading Markdown gerarchici; l'indice registra titolo e intervallo `righe X–Y`.
5. **Sintesi ricavata dal contenuto.** La descrizione dell'indice deriva dalla prima frase utile della sezione, non soltanto dal nome del file.
6. **Indici generati, contenuti modificati a mano.** I file `_INDEX.md` e i cataloghi JSON non si editano direttamente: si rigenerano dopo aver cambiato heading o documenti.
7. **Link relativi e verificabili.** I collegamenti interni devono sopravvivere agli spostamenti ed essere controllati automaticamente.
8. **Provenienza esplicita.** Le ricerche esterne conservano repository, snapshot o commit, data di audit e grado dell'evidenza; manifesti e lock separano fonti da interpretazioni.
9. **Catalogo machine-readable.** `catalog.json` replica gruppi, documenti, sezioni, righe e sintesi per controlli e strumenti futuri.
10. **Nessun documento orfano.** Ogni Markdown tematico deve essere raggiungibile dall'indice del proprio gruppo e dall'indice madre.

## Manutenzione

- Rigenerare tutta la wiki con `make docs-index`.
- Verificare intervalli, copertura e link con `make docs-check`.
- Per il solo corpus di ricerca restano disponibili `make research-index` e `make research-check`.
- Il catalogo generale è [catalog.json](catalog.json); quello specialistico è [research/catalog.json](research/catalog.json).
