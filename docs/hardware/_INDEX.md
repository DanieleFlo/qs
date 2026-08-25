# Indice: hardware

Indice generato; non modificare a mano.

## [Numerica CUDA e determinismo](cuda-numerics.md)

Ogni somma/prodotto in precisione finita arrotonda.

- righe 6–22: **Floating point non associativo** — Ogni somma/prodotto in precisione finita arrotonda.
- righe 23–38: **TF32, FP32 e kernel custom** — TF32 usa Tensor Core con input a precisione ridotta rispetto a FP32 e
- righe 39–53: **Effetti di `--use_fast_math`** — Il flag aggrega opzioni di compilazione che privilegiano throughput e usa
- righe 54–74: **Riproducibilita' cuBLAS** — NVIDIA garantisce in generale risultati bitwise ripetibili per la stessa
- righe 75–92: **Errori attesi contro errori sospetti** — Differenze di pochi ULP distribuite e stabili possono essere compatibili con
- righe 93–103: **Checklist prima di modificare un kernel** — logits finali.

## [Inventario della macchina di validazione](host-inventory.md)

Al momento della fotografia la scheda era anche display-attached, in P8, con

- righe 6–28: **Componenti osservate** — Al momento della fotografia la scheda era anche display-attached, in P8, con
- righe 29–41: **Implicazioni per Qwen3.6 27B Q4_K_M** — 18,50 GiB.
- righe 42–56: **Snapshot consigliato per ogni run** — Conservare in un file di testo accanto al report:
- righe 57–62: **Limiti dell'inventario** — Modello esatto della scheda partner, RAM e motherboard non sono ancora stati

## [Hardware e runtime di riferimento](reference-environment.md)

Questa directory raccoglie la conoscenza hardware necessaria per diagnosticare


## [RTX 3090: architettura rilevante per DS4](rtx-3090.md)

La RTX 3090 Founders Edition usa il die GA102 Ampere e, secondo il whitepaper

- righe 3–17: **Specifiche di riferimento** — La RTX 3090 Founders Edition usa il die GA102 Ampere e, secondo il whitepaper
- righe 18–38: **Compute capability 8.6** — La compute capability 8.6 identifica le funzionalita' ISA/runtime della classe
- righe 39–53: **Percorsi aritmetici disponibili** — Ampere introduce Tensor Core TF32: range simile a FP32 ma precisione degli
- righe 54–73: **Memoria e trasferimenti** — I 24 GB consentono la residency del GGUF Q4_K_M osservato, ma non equivalgono a
- righe 74–83: **Cosa l'hardware puo' e non puo' spiegare** — Puo' spiegare differenze riproducibili quando cambia il percorso matematico:
