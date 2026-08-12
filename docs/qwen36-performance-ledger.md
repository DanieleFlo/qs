# Ledger prestazionale Qwen3.6 CUDA

Questo è il registro canonico delle ottimizzazioni Qwen3.6 27B Q4_K_S/M su
RTX 3090. I documenti datati conservano il racconto completo delle singole
sessioni; qui ogni tecnica compare una sola volta con stato, evidenza e motivo
della decisione. Aggiornare questo file anche quando un esperimento fallisce.

## Contratto

Ogni riga segue:

```text
OSSERVA → MISURA → CLASSIFICA → RECUPERA CONOSCENZA → FORMULA IPOTESI
→ PREVEDI GLI EFFETTI → IMPLEMENTA → VERIFICA CORRETTEZZA
→ MICROBENCHMARK → BENCHMARK END-TO-END → KEEP / REJECT
→ MEMORIZZA → ITERA
```

`KEEP` richiede correttezza della primitiva, logits/argmax pertinenti, benchmark
fuori dal profiler e conferma slow con almeno cinque ripetizioni. `DIRECTION`
e `NEED_MORE_DATA` non rendono predefinito un percorso. Le fonti principali
sono `docs/qwen36-performance-experiments-2026-08-05.md`,
`docs/qwen36-lmstudio-decoding-analysis-2026-08-06.md`,
`docs/qwen36-cuda-optimization-progression.md`,
`docs/qwen36-numerics-lab.md` e `docs/qwen36-drift-hypotheses.md`.

## Stato corrente

- Hardware: RTX 3090 GA102, `sm_86`, 24 GiB, 936,2 GB/s nominali.
- Artefatto principale: Q4_K_S SHA-256
  `ff857ba9f2184d8be408e8cabda12c89ba5adb202fddc1a88b3774d7bb232aca`.
- Pesi residenti: circa 14,76 GiB; MTP disattivato per questi confronti.
- Target intermedio: almeno 20 tok/s anche oltre 10K; target finale 40 tok/s
  senza MTP e senza regressione qualitativa.
- Baseline harness 2026-08-11: 15,89 tok/s a 128, 9,93 a 2K; nell'A/B
  model-backed a posizione 10.666 il legacy misura 3,51 tok/s steady.
- Diagnosi 10.666: 231,21 dei 282,56 ms profilati sono i 16 core
  full-attention. Il costo è seriale sulla sequenza, non un calo della
  residency dei pesi.
- Dopo split-K32 il token profilato misura 86,35 ms: FFN 46,40 ms,
  attenzione ricorrente 18,92 ms, ramo full-attention 18,00 ms (core 11,78),
  output 1,84 ms e readback 0,42 ms. Il collo residuo non è più il KV lungo.
- A 8K il decode stabile misura 15,49 tok/s nella suite slow e 15,62--15,88
  tok/s nei run diretti. Il profilo diagnostic-sync attribuisce 35,60 ms al
  FFN, 15,48 ms all'attenzione ricorrente, 13,50 ms alla full-attention,
  2,35 ms all'output e 0,42 ms al readback, per 68,01 ms osservati.
- I tensori di peso effettivamente attraversati valgono circa 15,13 GB
  (14,09 GiB) per token: circa 234 GB/s utili a 15,49 tok/s. Il target di
  30 tok/s richiede circa 454 GB/s utili, ancora sotto i 936,2 GB/s nominali.
  Un campionamento `nvidia-smi` durante il decode rileva GPU al 99%, memory
  controller mediamente al 38,8% (34--40%) e circa 376 W: il percorso è
  limitato soprattutto da unpack/istruzioni/dipendenze, non dalla banda HBM.
- Lo smoke del dispatch di produzione, senza flag candidato, conferma a 10.666
  token 3,62 → 14,27 tok/s (+293,66%); 16/16 token greedy, argmax e top-20
  coincidono e il coseno finale è 0,9999999999996.
- Il candidato opt-in Q8_1-R8 conserva 8/8 sequenze e 512/512 argmax. Nel
  direction sweep residente porta il decode 128 da 17,41 a 32,69 tok/s e il
  decode 2K da 10,71 a 14,93 tok/s; la variante finale riusa il weight decode
  fra i due residui ed è stata confermata con cinque campioni non instabili.
  Non è ancora default: manca la conferma slow a cinque ripetizioni e resta
  sotto il riferimento storico di circa 40 tok/s.

## Soluzioni mantenute

| Area | Soluzione | Evidenza | Decisione |
|---|---|---|---|
| Correttezza Qwen | Mapping value head tiled `h % 16` per i pesi GGUF già riordinati | 512/512 argmax e sequenze corte coincidenti con llama.cpp; tre run bit-exact | KEEP |
| Residency | Copia predefinita del Q4_K_S su GPU singola | 14,76 GiB residenti; elimina fallback/PCIe per layer | KEEP |
| Prefill | Scheduling layer-major e output head solo sull’ultima riga | circa 9,75 → 205,74 tok/s a 2K prima delle ottimizzazioni successive | KEEP |
| Prefill | Dequant Q4_K + cuBLAS da 128 righe | 14,52 → 247,01 tok/s su 512/chunk128; 32/32 greedy a 4K | KEEP, drift chunk-order documentato |
| Prefill | Attention GEMM da 512 righe | rimuove il collo scalare a 2K; parte del percorso sopra 500 tok/s | KEEP |
| Prefill | Chunk Qwen predefinito 2048 | più rapido verificato a 16K; 128 e 512 più lenti, 4096 in timeout | KEEP |
| Decode pesi | Matvec Q4_K/Q5_K F32 warp8 | circa 9,8 → 15–18 tok/s corto; probe bit-exact rispetto al block256 | KEEP |
| Output head | Q6_K F32 warp8 | 248.320 → 31.040 blocchi; bit-exact contro block256 | KEEP |
| GDN | Stato ricorrente e convoluzionale mantenuti in registri nel percorso multi-riga | prefill 549,50 → 740,15 tok/s su Q4_K_M; probe entro roundoff | KEEP |
| Long-context | Split-K32 automatico da 8K, legacy sotto soglia | slow residente, 5 run/punto: 4,56 → 15,49 tok/s a 8K, 3,31 → 14,70 a 12K, 2,60 → 13,94 a 16K; 16/16 token identici su tutti i punti | KEEP |
| Decode opt-in | Q8_1-R8 a due stadi per Q4_K/Q5_K/Q6_K, weight decode fuso | probe entro roundoff, 8/8 e 512/512; 17,41→32,69 a 128 e 10,71→14,93 a 2K; cinque run finali stabili | DIRECTION; flag `DS4_CUDA_QWEN_DECODE_Q8_1_R8=1`, non default |

## Soluzioni scartate o non promosse

| Tecnica | Misura osservata | Motivo | Decisione |
|---|---:|---|---|
| `CUBLAS_GEMM_DEFAULT_TENSOR_OP` forzato | 136,39 tok/s prefill | regressione prestazionale | REJECT |
| Q4_K×Q8_K MMA per prefill | 150,53 tok/s a chunk128 | peggiore del GEMM dequantizzato | REJECT |
| Prefill Q8/TF32 aggressivo | 447,15 tok/s | gate top-20 fallito | REJECT |
| Prefill FP16 completo | 636,55 tok/s | drift numerico | REJECT |
| FP16 primi 52 layer / ultimi 12 F32 | 551,02 tok/s | Spearman sentinella 0,849 | REJECT |
| FP16 solo gate/up | 522,61 tok/s | Spearman sentinella 0,854 | REJECT |
| Decode Q4/Q5×Q8_K | fino a 26,45 tok/s | 6/8 sequenze, 458/512 argmax nella suite completa | REJECT |
| Q8_K solo gate/up | 20,34 tok/s | 7/8 sequenze, 508/512 argmax | REJECT |
| Q8_K gate/up+down | 21,96 tok/s | una continuazione divergente | REJECT |
| Q8_K solo down | 17,15 tok/s | nessun vantaggio utile | REJECT |
| Q8 a gruppi 32/128 | 14,59–17,08 tok/s | più accurato ma più lento/sotto target | REJECT |
| Scala Q8 least-squares e split 52/12 o 12/52 | circa 20–21 tok/s | stessa biforcazione autoregressiva | REJECT |
| Q8_1 + MMVQ Q4/Q5 | 34,50 tok/s, 40 registri, zero spill | 7/8 sequenze e 484/512 argmax; errore di politica, non shuffle/FMA | REJECT |
| Q6_K×Q8_1 output head | +5%, 18,06 tok/s | eredita 7/8 e 484/512 dal percorso Q8_1 | REJECT |
| Q6_K×Q8_K artigianale | 25,82 contro 26,45 tok/s | estrazione bit e pressione registri | REJECT, rimosso |
| CUDA Graph F32 | 17,74 contro 17,92 tok/s | grafo riusato ma launch overhead non dominante | REJECT, rimosso |
| Preallocazione scratch Q8_1 | 5.760–19.584 byte stimati | serviva solo candidati Q8_1 già respinti | NOT IMPLEMENTED |
| GDN warp-per-column stile llama.cpp | circa 0,049 contro 0,013 ms/layer | 1.536 CTA e norma ripetuta superano il traffico risparmiato | REJECT |
| Grouped GQA, sei query head per KV head | 3,69 tok/s a 10,6K | solo 4 CTA e sei accumulatori: meno parallelismo, più registri | REJECT, rimosso |
| Prefill chunk 4096 a 16K | timeout oltre 904 s | pressione scratch/VRAM con modello residente | REJECT come default |
| Build senza fast-math | residuo lievemente migliore, logits 0,452 → 0,469 MAE | non cambia top-1 e peggiora l’uscita finale | REJECT |
| Score4 decode | −0,88% a 128, +2,72% a 2K | suite direction e differenza dentro/accanto al rumore | NEED_MORE_DATA |
| Hoist esplicito di `d`, `dmin`, scale/min nel matvec Q4_K F32 | 14,25 contro 14,29 tok/s a 10.666 con split-K32 | bit-exact nel probe, ma il compilatore eliminava già il lavoro; nessun guadagno end-to-end | REJECT, rimosso |
| Geometria matvec F32 a 4 o 16 warp | 15,62 e 15,46 tok/s a 8K | 8 warp è già vicino al punto migliore; la residency non è il limite | REJECT, rimosso |
| Unpack Q4/Q5 a coppie di nibble | 14,93 contro 15,62--15,88 tok/s a 8K | più istruzioni/registri senza ridurre abbastanza il lavoro | REJECT, rimosso |
| Q8 residuo doppio, un solo load dei pesi | 32,00 tok/s a 128 ma 14,65 a 2K | non recupera il costo del ramo attention legacy e non supera il candidato Q8_1 | REJECT, rimosso |
| Q8_1 con un outlier F32 esatto ogni 32 attivazioni | 28,51 tok/s a 8K; 39--40 registri, zero spill | 255/256 argmax teacher-forced ma 7/8 sequenze; una biforcazione a margine basso viola il gate completo | REJECT, rimosso |

## Parity Q8_1/MMVQ sul riferimento storico, 2026-08-12

### FATTO DIMOSTRATO — dispatch reale

Il checkout LM Studio esatto `1a064ab0921238c1daa397d6f4a900ef33884de2`
è stato ricostruito in Release con CUDA 12.4, `sm_86`, fast attention e CUDA
Graphs abilitati, `FORCE_MMQ=OFF` e `FORCE_CUBLAS=OFF`. La strumentazione
temporanea sul caso `multi_turn_preserve_thinking`, poi rimossa, mostra per
decode `n_tok=1`:

| Operazione | Shape osservata | Tipo | Quantizer | Kernel | Geometria |
|---|---:|---|---|---|---|
| proiezioni Q4 | esempi 5.120→6.144 | Q4_K | `quantize_q8_1` | `mul_mat_vec_q` MMVQ | block `(32,4,1)`, una CTA/riga |
| proiezioni Q5 | esempi 5.120→10.240 | Q5_K | `quantize_q8_1` | `mul_mat_vec_q` MMVQ | block `(32,4,1)`, una CTA/riga |
| output head | 5.120→248.320 | Q6_K | `quantize_q8_1` | `mul_mat_vec_q` MMVQ | block `(32,4,1)`, una CTA/riga |

Non sono stati osservati MMQ, cuBLAS o Tensor Core in queste operazioni. Il
dispatcher sceglie MMVQ prima di valutare MMQ; perciò `FORCE_MMQ` non avrebbe
riscritto il dispatch del batch singolo.

### FATTO DIMOSTRATO — A/B/C al greedy step 3

Configurazione congelata: GGUF Q4_K_S SHA-256 `ff857ba9…232aca`, RTX 3090
driver 610.62, MTP off, full residency, prompt da 55 token, teacher forcing e
greedy da 32 passi. Alla riga 0-based 3:

| Percorso | Argmax | logit 310 | logit 728 | margine 310−728 |
|---|---:|---:|---:|---:|
| A — DS4 F32 | 310 | 23,672911 | 23,660215 | +0,012695 |
| B — DS4 Q8_1 Q4/Q5 | 728 | 23,660852 | 23,759184 | −0,098331 |
| C — llama.cpp storico MMVQ Q8_1 | 310 | 23,784367 | 23,623466 | +0,160900 |

Verdetto: `C≈A!=B` sul top-1. La presenza di Q8_1 nel riferimento non basta a
spiegare il flip DS4; il risultato precedente non autorizza a classificare
MMVQ o la quantizzazione dell'attivazione come bug generico.

### FATTO DIMOSTRATO — primitive e quantizer parity

Un entry point diagnostico temporaneo ha alimentato i kernel DS4 e llama.cpp
reali con gli stessi pesi e gli stessi blocchi Q8_1 prequantizzati: 257 righe,
20 superblocchi, `in_dim=5.120`. Gli output Q4_K, Q5_K e Q6_K hanno SHA-256
identico per ogni famiglia, quindi la parity è bit-exact. Questo elimina come
causa mapping `iqs`, unpack, signedness, selezione dei blocchi Q8, `dp4a` e
topologia di riduzione del MMVQ osservato.

Il vero `output_norm` F32 alla posizione 57 (greedy step 3) ha SHA-256
`0b80605b38400eaec1b902a3a2ea030aef401779c9c76b475fb4377e55512b0b`.
Quantizzandolo nei due backend reali:

- 160/160 `ds.x` identici;
- 5.120/5.120 `qs` identici;
- 160/160 somme intere `qsum` identiche;
- `ds.y` differente in 153/160 blocchi, come previsto dalle due semantiche:
  DS4 conserva una ricostruzione quantizzata, il commit storico la somma F32
  originale arrotondata a half. Il MMVQ decode auditato non legge `ds.y`.

Il nuovo comando riproducibile è:

```sh
python3 tools/perf_harness.py q8-1-parity \
  gguf-tools/quality-testing/staging/qwen-phase2-trace/ds4-q8_1.bin \
  gguf-tools/quality-testing/staging/qwen-phase2-trace/llama-q8_1.bin \
  --output gguf-tools/quality-testing/staging/qwen-phase2-trace/q8-1-parity.json
```

### FATTO DIMOSTRATO — frozen output head e scala FP16

Con lo stesso hidden pre-output-norm, il solo output head F32 riproduce
bit-exact i logits A. Sostituendo soltanto la proiezione Q6_K con Q8_1 il
margine passa da `+0,012695` a `−0,009998`: la singola proiezione può quindi
produrre il flip su un margine piccolo, pur essendo primitive e quantizer
cross-backend esatti.

L'ablazione con stessi `qs`, stesso MMVQ e scala pre-cast F32 porta il margine
a `−0,009874`. Il cast FP16 della scala spiega soltanto `−0,000124` del
cambiamento totale `−0,022694` (circa 0,55%); non corregge l'argmax. Il clone
diagnostico è stato rimosso e i due build originali sono stati ricompilati.

### INFERENZA E DECISIONE

La precisione Q8_1 può cambiare una decisione a margine basso anche quando DS4
replica bit-per-bit quantizer e primitive del riferimento. La differenza A/B/C
restante è quindi dominata dallo stato F32 fornito alla proiezione nei due
motori, non da un'incompatibilità locale del MMVQ. `GO` alla precision
staircase/frozen-input prevista dal protocollo; `NO-GO` a redesign MMVQ,
modifica di `ds.y`, rounding o sola scala FP32.

### FATTO DIMOSTRATO — sensibilità Q4/Q5/Q6

Due selettori temporanei, poi rimossi, hanno separato Q4-only e Q5-only. Il
Q6-only usa il run già disponibile `ds4-q4ks-q6-q8-1-point2-001`. Alla stessa
riga teacher-forced e greedy 3 i due stream sono identici:

| Variante Q8_1 | Argmax | margine 310−728 | Δmargin vs F32 | MAE centrata vs F32 |
|---|---:|---:|---:|---:|
| F32 | 310 | +0,012695 | — | 0 |
| Q4-only | 728 | −0,001743 | −0,014439 | 0,331254 |
| Q5-only | 310 | +0,005444 | −0,007252 | 0,082262 |
| Q4+Q5 | 728 | −0,098331 | −0,111027 | 0,095412 |
| Q6-only output head | 728 | −0,009998 | −0,022694 | 0,009979 |

Q4 e Q6 sono ciascuno sufficienti a far cambiare questo argmax dal margine
molto basso; Q5 da solo non lo è. Poiché tutte e tre le primitive sono verdi,
la classificazione è sensibilità alla quantizzazione dell'attivazione, non
mapping discrepancy. Q4+Q5 mostra inoltre che l'errore finale non è la somma
lineare delle MAE delle famiglie: il hidden viene propagato attraverso il
corpo del modello.

Il confronto ora è un singolo comando dell'harness:

```sh
python3 tools/perf_harness.py qwen-logits-row \
  --run A=gguf-tools/quality-testing/staging/oracles/ds4-current-f32-phase0-mtp-001 \
  --run Q4=gguf-tools/quality-testing/staging/oracles/ds4-phase4-q4-only-mtp-001 \
  --run Q5=gguf-tools/quality-testing/staging/oracles/ds4-phase4-q5-only-mtp-001 \
  --run Q45=gguf-tools/quality-testing/staging/oracles/ds4-current-q8-1-phase0-mtp-001 \
  --case multi_turn_preserve_thinking --stream teacher --row 3 \
  --focus-token 310 --focus-token 728
```

### FATTO DIMOSTRATO — precision staircase

La ricostruzione offline dello stesso `output_norm` F32 e il risultato
teacher-forced end-to-end Q4-only sono:

| Rappresentazione | blocco/scala | activation MAE | activation RMSE | logit MAE | margine | δcommon | δdiff = Δmargin |
|---|---|---:|---:|---:|---:|---:|---:|
| F32 | — | 0 | 0 | 0 | +0,012695 | 0 | 0 |
| Q8_32 | 32/F32 | 0,008362 | 0,011895 | 0,250489 | −0,014473 | +0,233946 | −0,027168 |
| Q8_1 | 32/FP16 | 0,008393 | 0,011901 | 0,331254 | −0,001743 | +0,301071 | −0,014439 |
| Q8_128 | 128/F32 | 0,013780 | 0,019595 | 0,472234 | +0,122486 | −0,436222 | +0,109791 |
| Q8_K | 256/F32 | 0,018579 | 0,025772 | 0,446967 | −0,088627 | −0,451477 | −0,101322 |

Q8_128 conserva il top-1 del singolo passo ma ha il maggiore logit MAE della
tabella e un grande errore differenziale positivo: è una compensazione
specifica, non un candidato corretto. Q8_32 ricostruisce meglio di Q8_1 ma
peggiora `Δmargin`. Il risultato falsifica una relazione monotona fra MAE di
attivazione, granularità e decisione greedy; i gate sul margine restano
necessari.

### FATTO DIMOSTRATO — Q8_1 LSQ offline e proiezione

Sul vero vettore critico, un passo LSQ per blocco da 32 riduce activation MAE
da `0,00836246` a `0,00831473` (−0,57%) e RMSE da `0,01189454` a
`0,01178218` (−0,94%), cambiando 71/5.120 `qs`. Il clone CUDA temporaneo ha
poi mantenuto rounding, layout, weight decode e MMVQ, cambiando soltanto la
scala e la seconda quantizzazione. Sul frozen output head:

| Quantizer Q8_1 | logit MAE vs F32 | RMSE | margine 310−728 |
|---|---:|---:|---:|
| maxabs baseline | 0,009979 | 0,012513 | −0,009998 |
| LSQ one-step | 0,010206 | 0,012763 | −0,016533 |

LSQ migliora lievemente la ricostruzione dell'attivazione ma peggiora sia la
proiezione sia il margine critico. Verdetto `NO-GO`: nessun kernel LSQ resta
nel runtime, nessun benchmark prestazionale è giustificato e il build
originale è stato ripristinato.

### FATTO DIMOSTRATO — residuo diffuso e Q8_1-R8

Il simulatore offline ha decodificato direttamente le righe 310 e 728 del vero
output Q6_K e ha riprodotto i logits F32 entro circa `2e-6`. Sul vero
`output_norm` del greedy step 3:

| Politica | activation MAE | RMSE | max | δdiff previsto | margine previsto |
|---|---:|---:|---:|---:|---:|
| Q8_1 | 0,008390 | 0,011897 | 0,114351 | −0,025563 | −0,012867 |
| G16 | 0,006578 | 0,009490 | 0,114351 | −0,015719 | −0,003024 |
| A1 | 0,007856 | 0,011299 | 0,100096 | −0,026490 | −0,013794 |
| A2 | 0,007342 | 0,010731 | 0,093791 | −0,021851 | −0,009155 |
| Q8_1-R8 | 0,0000324 | 0,0000467 | 0,000414 | +0,0000332 | +0,012729 |

G16 e gli anchor A1/A2 non recuperano il margine; il residuo è diffuso. R8
conserva due blocchi Q8_1 per `x ≈ d0*q0 + d1*q1`, riusa lo stesso decode dei
pesi e somma due accumulatori DP4A. Sul frozen output head misura logit MAE
`7,71e-5`, RMSE `8,81e-5` e margine `+0,012766`. End-to-end sul caso critico
misura MAE centrata `4,39e-4`, top-20 `1,0` e margine `+0,012877`.

Il probe `make qwen-numerics CUDA_ARCH=sm_86` confronta ora il kernel R8 reale
con una politica CPU indipendente per Q4_K, Q5_K e Q6_K. Tutte le famiglie
sono entro roundoff (massimo rispettivamente `2,29e-5`, `4,58e-5`, `2,29e-5`)
e il riepilogo è `PASS`. La suite rigenerata con il weight decode fuso è
`ds4-phase7-q8-1-r8-fused-full-001`; il comando ripetibile del gate è:

```sh
python3 tools/perf_harness.py qwen-argmax-gate \
  gguf-tools/quality-testing/staging/oracles/ds4-q4ks-f32-point1-002 \
  gguf-tools/quality-testing/staging/oracles/ds4-phase7-q8-1-r8-fused-full-001 \
  --output performance-results/phase7-r8-fused-argmax-gate.json
```

Risultato: `8/8` sequenze, `256/256` greedy e `256/256` teacher-forced,
quindi `512/512` argmax.

Il direction benchmark con modello residente, warm-up e tre ripetizioni non
instabili è:

| Contesto | F32 mediana | R8 mediana | Δ | CV F32 / R8 |
|---:|---:|---:|---:|---:|
| 128 | 17,41 tok/s | 32,69 tok/s | +87,77% | 1,63% / 1,33% |
| 2.048 | 10,71 tok/s | 14,93 tok/s | +39,40% | 4,56% / 1,61% |

Gli artefatti sono `performance-results/phase7-r8-{base,candidate}-20260812`
e `performance-results/phase7-r8-fused-dual-confirm5-20260812`.
Il profiler opt-in, sul primo matvec reale di ciascun tipo a contesto 128,
separa quantizzatore e MMVQ:

| Peso / shape | quantizer | MMVQ | totale |
|---|---:|---:|---:|
| Q5_K 5.120→10.240 | 0,009216 ms | 0,096256 ms | 0,105472 ms |
| Q4_K 5.120→6.144 | 0,006144 ms | 0,054272 ms | 0,060416 ms |
| Q6_K 5.120→248.320 | 0,005120 ms | 2,253824 ms | 2,258944 ms |

`ptxas` su `sm_86` riporta zero stack e zero spill per tutti i kernel. Q4/Q5
R8 usano 40 registri e 768 B shared contro 40/384 B del Q8_1 singolo; Q6 usa
40/768 B contro 36/384 B; il quantizzatore R8 usa 24 registri contro 16. Con
128 thread i limiti di thread e registri consentono teoricamente 12 blocchi e
48 warp per SM (100% occupancy); il quantizzatore a 256 thread consente sei
blocchi e la stessa occupancy teorica. Ogni proiezione R8 usa due launch,
quantizer più MMVQ. I contatori hardware DRAM/achieved bandwidth restano
`NOT_VERIFIED` perché Nsight Compute/CUPTI non sono disponibili nel WSL
osservato; il proxy weight-only a 128 è circa 483 GiB/s (`14,76×32,69`).

L'ablazione successiva ha eliminato nel solo quantizzatore R8 le riduzioni
`qsum` e la scrittura utile di `ds.y`, metadata non letto dal MMVQ. Il probe e
il gate direction sono rimasti verdi, ma le mediane sono scese 32,26→31,18
tok/s a 128 e 14,93→14,48 a 2K (−3,18% medio; prefill 2K instabile). Verdetto
`NO-GO`: la patch è stata annullata e il lavoro va concentrato sul MMVQ.

Il candidato seguente fonde invece il decode di pesi, scale e minimi tra i due
residui R8, conservando due accumulatori e lo stesso ordine aritmetico per
ciascuno. `ptxas` resta a 40 registri, 768 B shared e zero spill. Nel
microprofilo Q6 MMVQ scende 3,295232→2,253824 ms (−31,60%) e Q4 scende
0,058368→0,054272 ms; Q5 è invariato nel primo campione. Il primo direction
sweep indicava +7,03% a 2K ma aveva un outlier a 128; la conferma a cinque run
produce 32,69 tok/s a 128 (CV 1,33%, +1,33% sul R8 non fuso) e 14,93 tok/s a
2K (CV 1,61%, invariato). Il gate completo sul run
`ds4-phase7-q8-1-r8-fused-full-001` è nuovamente 8/8 e 512/512. Verdetto:
`KEEP` dentro il candidato R8 opt-in; non cambia la decisione di non renderlo
ancora default.

La variante mista successiva manteneva R8 su Q4/Q6 e usava il solo primo
blocco Q8_1 su Q5, motivata dal frozen step dove Q5-only conservava il top-1.
Il ramo reale Q5 ha superato il probe contro la policy Q8_1 (MAE `1,75e-5`,
cosine 1), ma la suite completa ha prodotto 7/8 sequenze e 485/512 argmax:
teacher-forced 256/256, greedy 229/256, prima divergenza al passo 3 di
`multi_turn_preserve_thinking`. Verdetto `NO-GO` senza benchmark; flag e probe
diagnostici rimossi. La sicurezza locale di Q5-only non si trasferisce alla
dinamica autoregressiva quando gli stati precedenti sono R8.

Verdetto: `DIRECTION`, patch mantenuta dietro flag e non promossa a default.
La qualità e il vantaggio su F32 sono dimostrati, ma il target 40 tok/s e la
regola ledger dei cinque run slow non sono ancora soddisfatti. Il prossimo
esperimento deve ottimizzare il MMVQ R8, in particolare Q6, non il quantizzatore
che vale meno dello 0,2% del tempo dell'output head.

| Test | Kernel reale | Quantizer | Primitive parity | Quantizer parity | δcommon | δdiff | Δmargin | 8/8 | 512/512 | tok/s | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F32 | warp8 F32 | — | PASS | — | 0 | 0 | 0 | 8/8 | 512/512 | 17,41 | baseline |
| Q8_1 | MMVQ | maxabs/FP16 | bit-exact vs storico | 5.120/5.120 `qs` | +0,301071 | −0,014439 Q4-only | −0,111027 Q4+Q5 | 7/8 | 484/512 | 34,50 storico | REJECT |
| Q8_1-R8 | doppio MMVQ con weight decode fuso | due stadi maxabs/FP16 | PASS Q4/Q5/Q6 | PASS CPU-policy | circa +0,0000035 output head | +0,0000706 output head | +0,0000706 | 8/8 | 512/512 | 32,69 | DIRECTION, opt-in |

## Analisi del residuo a 8K, 2026-08-11

Il conto dei soli byte di peso per operazione è: ricorrente 3,426 GB, full
attention 0,945 GB, FFN 9,716 GB e output 1,043 GB. Ai 936,2 GB/s nominali i
floor ideali sono rispettivamente 3,66, 1,01, 10,38 e 1,11 ms; le misure sono
15,48, 13,50, 35,60 e 2,35 ms. La full-attention include il core KV, quindi il
confronto col solo peso non è diretto; ricorrente e FFN impiegano invece
4,23x e 3,43x il floor di banda. Questo, insieme all'utilizzo memory-controller,
localizza il residuo nel matvec quantizzato F32.

Il confronto con llama.cpp mostra il compromesso: il suo MMVQ quantizza
l'attivazione in Q8_1 e usa prodotti interi vettoriali; DS4 ricostruisce il
Q4_K/Q5_K e accumula in F32 per rispettare il riferimento qualitativo corrente.
Il candidato Q8_1 con outlier conferma che qui esistono circa 13 tok/s
recuperabili, ma conferma anche che un'approssimazione delle attivazioni non è
ancora promuovibile. Marlin e QUICK indicano la direzione trasferibile senza
cambiare politica numerica: repack offline, load vettoriali/coalescenti e
layout che riduce shuffle e bank conflict. Il formato GPTQ/FP16 di quei kernel
non è però sostituibile direttamente al Q4_K_S/F32 di DS4.

Artefatti: `performance-results/qwen-8k-network-20260811.json`,
`performance-results/qwen-8k-model-cost-20260811.json`,
`performance-results/qwen-8k-decode-smi-20260811.csv` e
`performance-results/ds4-q4ks-q8-1o-vs-f32-point1-007.json`.

## Esperimenti long-context 2026-08-11

Fonti recuperate: guida locale `problem-to-source.md`, FlashDecoding++,
FlashInfer (split-K solo per KV lunghi e workspace stabile), FlashAttention
(composizione online-softmax) e implementazioni llama.cpp/ik_llama come
riferimento di dispatch, non come codice copiato.

| ID | Ipotesi e previsione | Correttezza primitiva | Microprofilo 10.666 | End-to-end | Stato |
|---|---|---|---|---|---|
| `lc-warp` | shuffle warp riduce barriere; +15–30% | max 3,73e-8, roundoff | core 186,18 ms; 4,32 tok/s | storico 200/200 token | DIRECTION |
| `lc-split-warp` | 8 warp elaborano partizioni temporali indipendenti; core 3–6× | MAE 5,24e-9, max 4,84e-8, cosine 1 | core 38,67 ms; 11,04 tok/s profilati | inferiore allo split-K | REJECT, da rimuovere |
| `lc-split-k8` | 8 CTA/head aumentano il parallelismo senza duplicare K/V | MAE 5,14e-9, max 5,59e-8, cosine 1 | core 25,27 ms; 13,19 tok/s profilati | inferiore a K32 | REJECT come configurazione |
| `lc-split-k16` | 16 CTA/head avvicinano la copertura completa degli SM | stessa primitiva split-K verde | core 17,31 ms; 13,18 tok/s profilati | inferiore a K32 | REJECT come configurazione |
| `lc-split-k32` | più wave riducono la coda dei blocchi; core <17 ms | MAE 5,14e-9, max 5,59e-8; slow: 16/16 token uguali a 8K/12K/16K, top-20 1,0, cosine >0,9999999999991 | core 11,78 ms nel profiler v2; 14,29 tok/s fuori profiler a 10.666 | mediane steady 4,56 → 15,49 a 8K; 3,31 → 14,70 a 12K; 2,60 → 13,94 a 16K; +339,99% medio, CV <0,9% | KEEP; default automatico da 8K |

Artefatti locali: `performance-results/qwen-longctx-10666-*-profile.json` e
`performance-results/splitk32-longslow-20260811-01-{baseline,candidate}`.
Questi file sono ignorati da Git e non sono golden. Il fallback riproducibile
per gli A/B è `DS4_CUDA_QWEN_NO_SPLIT_K_ATTN=1`; il force flag
`DS4_CUDA_QWEN_SPLIT_K_ATTN=1` rimane solo per probe e soglie sperimentali.

## Coda ordinata

1. **MMVQ R8, soprattutto output Q6_K**: confermare prima con cinque run slow,
   poi ridurre il doppio dot/load mantenendo i due residui; il profilo attribuisce
   3,295 ms al solo MMVQ Q6 e appena 0,005 ms al quantizzatore.
2. **Repack/layout Q4_K_S compatibile F32**: guadagno potenziale alto
   (il candidato MMVQ espone circa 13 tok/s), difficoltà medio-alta. Preparare
   offline scale/nibble in ordine di consumo, misurando prima load transaction,
   istruzioni e byte reali; mantenere la stessa ricostruzione e accumulazione.
3. **Load vettoriali e mapping lane del matvec F32**: guadagno potenziale
   medio-alto e modifica locale. È il miglior prodotto guadagno/facilità dopo
   il repack; non ripetere il solo hoist o il cambio di geometria già falsificati.
4. **Fusione esatta di gate/up/SwiGLU o riuso dell'attivazione**: guadagno
   potenziale medio, difficoltà media; conservare F32 e verificare che il
   minor traffico/launch superi la pressione registri.
5. **Output/argmax interamente GPU**: guadagno massimo circa 0,4--2,4 ms,
   facilità alta ma impatto piccolo; utile solo dopo il matvec.
6. Non aumentare ancora split-K: a 8K tutta la full-attention vale 13,50 ms e
   non può da sola portare 15,49 a 30 tok/s. Non promuovere altri Q8 senza
   gate completo: 255/256 argmax non soddisfa il contratto.
7. Target falsificabile per il prossimo candidato F32: FFN <25 ms e
   ricorrente <11 ms a 8K, quindi almeno 20 tok/s prima del gate slow finale.

## Template per il prossimo record

```text
ID / data / commit / dirty state:
Osservazione:
Misura baseline:
Classificazione:
Fonti recuperate e trasferibilità sm_86/Q4_K_S:
Ipotesi falsificabile:
Effetto previsto:
Modifica isolata:
Correctness primitiva:
Correctness modello:
Microbenchmark/profilo:
Benchmark end-to-end:
KEEP / REJECT / NEED_MORE_DATA:
Lezione e prossimo esperimento:
```
