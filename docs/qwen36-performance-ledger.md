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

1. **Repack/layout Q4_K_S compatibile F32**: guadagno potenziale alto
   (il candidato MMVQ espone circa 13 tok/s), difficoltà medio-alta. Preparare
   offline scale/nibble in ordine di consumo, misurando prima load transaction,
   istruzioni e byte reali; mantenere la stessa ricostruzione e accumulazione.
2. **Load vettoriali e mapping lane del matvec F32**: guadagno potenziale
   medio-alto e modifica locale. È il miglior prodotto guadagno/facilità dopo
   il repack; non ripetere il solo hoist o il cambio di geometria già falsificati.
3. **Fusione esatta di gate/up/SwiGLU o riuso dell'attivazione**: guadagno
   potenziale medio, difficoltà media; conservare F32 e verificare che il
   minor traffico/launch superi la pressione registri.
4. **Output/argmax interamente GPU**: guadagno massimo circa 0,4--2,4 ms,
   facilità alta ma impatto piccolo; utile solo dopo il matvec.
5. Non aumentare ancora split-K: a 8K tutta la full-attention vale 13,50 ms e
   non può da sola portare 15,49 a 30 tok/s. Non promuovere altri Q8 senza
   gate completo: 255/256 argmax non soddisfa il contratto.
6. Target falsificabile per il prossimo candidato F32: FFN <25 ms e
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
