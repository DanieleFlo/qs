# Knowledge base per kernel engineering

Punto di ingresso per cercare soluzioni già implementate prima di progettare un kernel DS4.

Accesso alternativo compatto: [indice degli indici](indexes/INDEX.md).

## Percorso consigliato

1. Cercare il problema nell'indice tematico e nelle schede piattaforma.
2. Verificare i fork RTX 3090 e il loro livello di evidenza.
3. Leggere il paper locale pertinente usando gli intervalli di righe.
4. Tornare al codice upstream fissando commit, shape, formato numerico e hardware.

## Indici di gruppo

- [papers/text](papers/text/_INDEX.md) — Testo semantico dei paper arXiv archiviati localmente.
- [platforms](platforms/_INDEX.md) — Architettura e kernel delle maggiori piattaforme di inferenza locale.
- [forks](forks/_INDEX.md) — Fork e raccolte con ottimizzazioni consumer/RTX 3090, classificati per evidenza.
- [guides](guides/_INDEX.md) — Mappe trasversali problema → fonti → implementazioni da ispezionare.

## Provenienza

- [Manifesto paper](papers/manifest.json)
- [Lock del corpus](papers/corpus-lock.json)
- [Lock delle piattaforme e dei fork](sources-lock.json)
- [Catalogo macchina](catalog.json)

Rigenerare con `make research-index`; verificare con `make research-check`.
