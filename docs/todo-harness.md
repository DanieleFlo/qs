# TODO — GPU Kernel Engineering Harness per DS4 / Qwen 3.6 27B

Obiettivo finale: costruire su DS4 un sistema completo che permetta a un agente locale di **misurare, diagnosticare, progettare, implementare, verificare e iterare ottimizzazioni GPU**, usando dati hardware reali, profiling strutturato, una knowledge base tecnica, un sistema di esperimenti e pipeline decisionali riproducibili.

## Stato verificato dell'incremento iniziale (2026-08-11)

I checkbox completati sotto indicano funzionalità presenti e testate; i
macro-task restano aperti quando manca anche un solo requisito della relativa
condizione di completamento.

- CLI unica in tools/perf_harness.py: probe, benchmark, confronto esperimenti,
  profilo per layer/stadio, confronto profili e drift logits.
- Suite direction con due prove e due campioni per feedback rapido; non può
  emettere KEEP. Suite slow per conferma con warm-up e almeno cinque
  ripetizioni richieste dal contratto operativo.
- Baseline e candidate conservano automaticamente logits alle frontiere,
  sequenza greedy e logits finali post-decode. Uguaglianza dei token, argmax,
  top-20 overlap, cosine, MAE/RMSE e valori non finiti partecipano al verdetto
  prima della promozione.
- Il runtime Qwen CUDA emette tempi strutturati per layer separando attention e
  FFN e distinguendo Gated DeltaNet ricorrente e full attention.
- Smoke reale RTX 3090: build sm_86 riuscita. Il profilo definitivo 1.024/512
  ha scartato le 64 righe cold del primo chunk e classificato 64 layer prefill
  più 64 layer decode: 1.027,636 ms profilati, 46,2% attention e 53,8% FFN.
- Cost model verificato sul GGUF Q4_K_S reale. Decode 1K: floor memoria
  recurrent attention 3,659 ms, full attention 1,009 ms, FFN 10,379 ms;
  i tempi profilati sono rispettivamente 4,2x, 28,7x e 3,8x il floor. Prefill
  chunk 512 risulta compute-bound e FFN osservato è 1,0x il floor nominale.
- Nsight Systems/Compute, NVTX, traffic amplification,
  microbenchmark generico e knowledge base restano aperti.
- I comandi documentati producono baseline e candidate prima del confronto; gli
  input mancanti terminano con errore conciso senza traceback. Il target
  `perf-direction-ab` esegue baseline, candidate, misure e drift con un comando.
- Il probe reale RTX 3090 rileva Ampere/CC 8.6, memoria libera e totale, clock,
  power/temperature/P-state e motivi di throttling; completa i limiti non esposti
  dal driver con una tabella statica marcata e individua `nvcc` anche fuori PATH.
- Le ripetizioni di `ds4-bench` riusano modello e sessione residenti. La suite
  direction raggruppa 128 e 2K nello stesso sweep: smoke A/B completo in 257,5 s
  incluse build, due caricamenti e quattro campioni per lato. Drift PASS a
  entrambe le frontiere; Score4 -0,88% a 128 e +2,72% a 2K, quindi
  NEED_MORE_DATA per rumore e natura preliminare della suite.
- Corpus di ricerca locale creato: 24 paper HTML con SHA-256 e testo semantico,
  8 schede piattaforma, 6 schede fork/RTX 3090 e 3 guide problema→soluzione.
  Quattro indici di gruppo, indice principale e indice degli indici contengono
  analisi e intervalli di riga verificati automaticamente da `research-check`.
- Workflow ripetuti consolidati in script eseguibili richiamati dal comando
  `perf_harness.py workflow`; aggiunte direction a 10.666 e slow residente
  8K/12K/16K. Le
  decisioni Qwen mantenute e respinte confluiscono nel ledger canonico
  `docs/qwen36-performance-ledger.md`.
- Split-K32 è KEEP e default automatico da 96 token dopo la bisezione comune
  target-only/MTP del 2026-08-13. La matrice completa residente
  2K–30K usa due run/punto, richiede almeno 20 tok/s a ogni frontiera e
  controlla che non esistano recuperi materiali dopo una valle. Nel confronto
  isolato la curva passa da 14,25/9,05/6,51 tok/s a 2K/4K/6K a
  33,61/32,33/31,20, poi scende fino a 20,91 a 30K, con 30/30 artefatti
  frontier/decode PASS. Nello slow residente storico (cinque
  run/punto) porta 4,56 → 15,49 tok/s a 8K, 3,31 → 14,70 a 12K e
  2,60 → 13,94 a 16K; tutti i 16 token greedy coincidono in ogni frontiera,
  top-20 è 1,0 e il coseno resta sopra 0,9999999999991. Uno smoke successivo
  senza flag conferma il dispatch di produzione a 10.666: 3,62 → 14,27
  tok/s. Il profilo del residuo classifica FFN (46,40 ms) e attenzione
  ricorrente (18,92 ms) prima del core full-attention lungo (11,78 ms).
- MTP long-context è KEEP: V(3) sotto 2K, V(2) da 2K, stesso ingresso split-K
  a 96 token. La curva 0–28K migliora ogni punto (+11,73% medio, +10,04% a
  28K); il margine minimo 24K è confermato 5× a +2,79% con output identico.
- Aggiunto il componente `docs/research/cuda/`: fonti primarie NVIDIA,
  comandi di profiling e ispezione binaria, primitive Ampere sm_86 e mappa
  falsificabile verso i prossimi esperimenti DS4. `research-check` ne verifica
  indicizzazione e catalogo.

Principio generale:

```text
OSSERVA
→ MISURA
→ CLASSIFICA
→ RECUPERA CONOSCENZA
→ FORMULA IPOTESI
→ PREVEDI GLI EFFETTI
→ IMPLEMENTA
→ VERIFICA CORRETTEZZA
→ MICROBENCHMARK
→ BENCHMARK END-TO-END
→ KEEP / REJECT
→ MEMORIZZA
→ ITERA
```

---

# [x] 1. Creare il manifesto operativo dell'agente

## Sottotask

* [x] Creare `docs/agent_performance_contract.md`.
* [x] Definire formalmente che l'obiettivo finale non è:

  * occupancy massima;
  * SM utilization massima;
  * TFLOP massimi;
  * bandwidth massima;
  * kernel isolato più veloce.
* [x] Definire come obiettivi principali:

  * TTFT;
  * prefill tokens/s;
  * decode tokens/s;
  * decode ms/token;
  * peak VRAM;
  * stabilità numerica;
  * correttezza del modello.
* [x] Stabilire che nessuna ottimizzazione può essere accettata soltanto perché migliora una metrica hardware.
* [x] Imporre la distinzione obbligatoria fra:

  * prefill;
  * decode;
  * batch differenti;
  * context length differenti.
* [x] Definire il ciclo obbligatorio:

  * baseline;
  * profiling;
  * diagnosi;
  * ipotesi;
  * previsione;
  * modifica;
  * correctness test;
  * microbenchmark;
  * benchmark end-to-end;
  * decisione;
  * logging.
* [x] Vietare esplicitamente l'ottimizzazione “a intuito” senza baseline.
* [x] Vietare la modifica contemporanea di molte variabili quando impedisce di attribuire il risultato a una causa.
* [x] Imporre che ogni ipotesi sia falsificabile.
* [x] Imporre che ogni esperimento abbia un identificatore univoco.
* [x] Imporre rollback automatico o semplice recupero della baseline precedente.
* [x] Definire i criteri minimi di significatività statistica dei benchmark.

## Condizione di completamento

Il macro-task è completo quando esiste un documento letto automaticamente dall'agente prima di ogni sessione di kernel engineering e l'harness **rifiuta un esperimento** privo di baseline, ipotesi, workload target, correctness test e metrica finale da migliorare.

---

# [ ] 2. Costruire il rilevatore hardware della GPU

## Sottotask

* [x] Implementare `hardware_probe()`.
* [x] Rilevare automaticamente:

  * nome GPU;
  * compute capability;
  * architettura NVIDIA;
  * numero SM;
  * VRAM totale;
  * VRAM disponibile;
  * memory bus;
  * bandwidth teorica;
  * clock memoria;
  * L2 cache;
  * dimensione warp;
  * massimo thread/block;
  * massimo thread/SM;
  * massimo blocchi/SM;
  * registri/SM;
  * registri/block;
  * shared memory/SM;
  * shared memory/block;
  * limiti configurable shared memory;
  * supporto FP32;
  * FP16;
  * BF16;
  * TF32;
  * FP8 se disponibile;
  * INT8;
  * INT4;
  * Tensor Core disponibili;
  * caratteristiche specifiche dell'architettura.
* [ ] Rilevare:

  * versione driver;
  * versione CUDA runtime;
  * versione CUDA toolkit;
  * compilatore CUDA;
  * PyTorch;
  * Triton;
  * librerie utilizzate da DS4.
* [x] Rilevare eventuali limiti:

  * power limit;
  * temperature;
  * clock throttling.
* [x] Salvare il risultato in formato macchina:

```text
hardware_profile.json
```

* [x] Generare anche una sintesi leggibile dall'agente.
* [ ] Associare automaticamente la GPU alla relativa sezione della wiki:

  * Ampere;
  * Ada;
  * Hopper;
  * Blackwell;
  * altra architettura.

## Condizione di completamento

Il macro-task è completo quando una sola chiamata restituisce un `hardware_profile.json` sufficientemente completo da impedire all'agente di proporre una tecnica non supportata dall'hardware senza segnalarlo esplicitamente.

---

# [ ] 3. Creare il modello teorico dei costi del modello

## Sottotask

* [x] Implementare `model_cost()`.
* [ ] Estrarre automaticamente dal modello:

  * numero layer;
  * hidden size;
  * intermediate size;
  * numero attention heads;
  * numero KV heads;
  * head dimension;
  * vocabulary size;
  * dtype attivazioni;
  * dtype pesi;
  * quantizzazione;
  * group size;
  * eventuali scale/zero-point.
* [x] Calcolare dimensione teorica dei pesi.
* [x] Calcolare dimensione effettiva con metadata di quantizzazione.
* [x] Calcolare KV cache per:

```text
batch
context length
dtype
numero layer
KV heads
head_dim
```

* [x] Implementare formula generale per K + V.
* [x] Stimare FLOP teorici:

  * attention;
  * QKV projections;
  * output projection;
  * MLP;
  * normalization;
  * logits.
* [x] Separare il cost model per:

  * prefill;
  * decode.
* [x] Calcolare minimum required bytes per ogni operazione principale.
* [x] Calcolare arithmetic intensity teorica:

```text
FLOP / byte
```

* [x] Calcolare il roofline teorico.
* [x] Stimare:

```text
memory_time_floor
compute_time_floor
roofline_time_floor
```

* [x] Calcolare il rapporto:

```text
actual_time / theoretical_floor
```

* [x] Evidenziare operazioni con grande margine teorico.

## Condizione di completamento

Il macro-task è completo quando, dato un workload reale, il sistema può produrre per ogni operazione principale una stima verificabile di **FLOP, minimum bytes, arithmetic intensity e limite teorico memoria/compute**.

---

# [ ] 4. Definire la suite canonica dei workload

## Sottotask

* [x] Creare `workloads.yaml`.
* [x] Definire workload separati per prefill e decode.
* [x] Includere almeno:

  * decode batch 1 context corto;
  * decode batch 1 context medio;
  * decode batch 1 context lungo;
  * decode batch 1 context molto lungo;
  * prefill corto;
  * prefill medio;
  * prefill lungo;
  * workload throughput con batch > 1.
* [x] Rendere context length configurabile.
* [x] Rendere batch size configurabile.
* [x] Rendere generation length configurabile.
* [x] Aggiungere:

  * quick suite;
  * standard suite;
  * exhaustive suite.
* [x] Definire input deterministici.
* [ ] Definire seed riproducibili.
* [ ] Salvare versione del modello e quantizzazione associata.
* [ ] Definire workload specifici per singole shape dei kernel.
* [ ] Registrare distribuzione delle shape incontrate realmente durante una sessione.
* [ ] Permettere all'harness di pesare una shape in base alla frequenza reale durante l'inferenza.

## Condizione di completamento

Il macro-task è completo quando ogni esperimento può essere riprodotto sulla stessa matrice di workload e nessuna dichiarazione “questo kernel è più veloce” può essere emessa senza specificare **batch, context, fase e shape**.

---

# [ ] 5. Implementare il benchmark end-to-end affidabile

## Sottotask

* [x] Implementare `benchmark_model()`.
* [ ] Separare:

  * initialization;
  * model load;
  * warmup;
  * prefill;
  * decode.
* [ ] Misurare:

  * TTFT;
  * prefill ms;
  * prefill tokens/s;
  * decode ms/token;
  * decode tokens/s;
  * tempo totale;
  * peak VRAM.
* [ ] Usare timing GPU corretto dove applicabile.
* [x] Evitare timing finale sotto profiler.
* [ ] Implementare warm-up obbligatorio.
* [x] Eseguire più ripetizioni.
* [x] Calcolare:

  * min;
  * median;
  * mean;
  * p10;
  * p90;
  * standard deviation.
* [x] Segnalare run instabili.
* [ ] Controllare clock e thermal throttling.
* [x] Salvare configurazione hardware insieme al benchmark.
* [x] Normalizzare i risultati rispetto alla baseline.

## Condizione di completamento

Il macro-task è completo quando due esecuzioni consecutive della suite producono risultati statisticamente sufficientemente stabili da permettere di distinguere in modo affidabile una regressione o un miglioramento reale.

---

# [ ] 6. Inserire una gerarchia NVTX nel runtime

## Sottotask

* [ ] Annotare con NVTX:

  * prefill;
  * decode;
  * layer;
  * attention;
  * QKV;
  * RoPE;
  * KV cache;
  * attention kernel;
  * output projection;
  * RMSNorm;
  * MLP;
  * activation;
  * logits;
  * sampling dove rilevante.
* [ ] Associare ID stabili ai layer.
* [ ] Associare shape ai range quando possibile.
* [ ] Annotare dtype e quantizzazione rilevanti.
* [ ] Annotare kernel candidati personalizzati.
* [ ] Verificare che Nsight Systems mostri chiaramente la gerarchia.
* [ ] Permettere profiling selettivo di un range NVTX.

## Condizione di completamento

Il macro-task è completo quando è possibile isolare automaticamente, per esempio:

```text
decode → layer 17 → attention
```

oppure:

```text
prefill → layer 8 → MLP
```

senza analizzare manualmente l'intera timeline GPU.

---

# [ ] 7. Costruire il sistema di tracing con Nsight Systems

## Sottotask

* [ ] Implementare `trace_model()`.
* [ ] Avviare Nsight Systems tramite CLI.
* [ ] Limitare la cattura ai range interessanti.
* [ ] Raccogliere:

  * kernel launch;
  * durata;
  * numero invocazioni;
  * CUDA runtime calls;
  * memcpy;
  * synchronization;
  * CPU/GPU gaps.
* [ ] Implementare parser del report.
* [ ] Aggregare kernel per:

  * nome;
  * shape;
  * call site;
  * range NVTX.
* [ ] Calcolare:

  * tempo totale;
  * tempo medio;
  * mediana;
  * numero call;
  * percentuale del runtime.
* [ ] Identificare:

  * launch gaps;
  * sincronizzazioni;
  * allocazioni;
  * copie CPU↔GPU;
  * sequenze di micro-kernel.
* [ ] Implementare `rank_hotspots()`.
* [ ] Classificare separatamente:

  * kernel costoso singolarmente;
  * kernel economico ma chiamato moltissime volte.
* [ ] Identificare sequenze candidate alla fusion.

## Condizione di completamento

Il macro-task è completo quando l'agente riceve automaticamente una classifica dei kernel/range responsabili di almeno il **90–95% del tempo GPU osservato** e può distinguere kernel bottleneck da launch/runtime bottleneck.

---

# [ ] 8. Costruire il profiler dettagliato con Nsight Compute

## Sottotask

* [ ] Implementare `profile_kernel()`.
* [ ] Profilare solo kernel/range selezionati.
* [ ] Estrarre almeno:

  * kernel duration;
  * SM throughput;
  * memory throughput;
  * DRAM bytes read;
  * DRAM bytes write;
  * L2 traffic;
  * L1 traffic;
  * L2 hit rate;
  * L1 hit rate;
  * registers/thread;
  * shared memory/block;
  * local memory;
  * register spills;
  * theoretical occupancy;
  * achieved occupancy;
  * active warps;
  * eligible warps;
  * issued warps;
  * instruction throughput;
  * Tensor Core utilization;
  * warp stall reasons.
* [ ] Estrarre almeno stall correlati a:

  * long scoreboard;
  * short scoreboard;
  * barrier;
  * memory throttle;
  * math pipeline;
  * not selected;
  * branch/divergence quando disponibile.
* [ ] Estrarre launch configuration:

  * grid;
  * block;
  * dynamic shared memory.
* [ ] Estrarre settori/transazioni memoria quando disponibili.
* [ ] Generare un report JSON normalizzato.
* [ ] Evitare profiling indiscriminato di tutto il modello.
* [ ] Creare livelli:

  * light profile;
  * memory profile;
  * compute profile;
  * full investigation.
* [ ] Documentare l'overhead dei replay di Nsight Compute.

## Condizione di completamento

Il macro-task è completo quando un kernel selezionato produce automaticamente una scheda strutturata sufficiente a diagnosticare **bandwidth, latency, compute, occupancy, register pressure, shared-memory pressure, insufficient parallelism e principali warp stalls**.

---

# [ ] 9. Implementare l'analisi del traffico memoria

## Sottotask

* [ ] Calcolare per ogni kernel:

```text
minimum_required_bytes
actual_DRAM_bytes
actual_L2_bytes
actual_L1_bytes
```

* [ ] Calcolare:

```text
traffic_amplification =
actual_DRAM_bytes / minimum_required_bytes
```

* [ ] Calcolare:

  * bytes/token;
  * bytes/layer;
  * bytes/output element;
  * bytes/call.
* [ ] Separare:

  * weight traffic;
  * activation traffic;
  * KV-cache traffic;
  * intermediate traffic.
* [ ] Stimare riuso teorico dei tile.
* [ ] Analizzare coalescing.
* [ ] Analizzare transazioni/request.
* [ ] Segnalare accessi ridondanti.
* [ ] Segnalare materializzazione di intermedi evitabili.
* [ ] Segnalare conversioni dtype che scrivono buffer temporanei.
* [ ] Segnalare eventuale dequantizzazione separata dalla computazione successiva.
* [ ] Calcolare bandwidth utile:

```text
minimum_required_bytes / kernel_time
```

* [ ] Calcolare bandwidth fisica:

```text
actual_DRAM_bytes / kernel_time
```

* [ ] Confrontarle con la bandwidth teorica.

## Condizione di completamento

Il macro-task è completo quando l'agente può distinguere chiaramente fra:

```text
"serve più bandwidth"
```

e:

```text
"stiamo trasferendo troppi byte per ottenere lo stesso risultato"
```

e può quantificare l'amplificazione del traffico.

---

# [ ] 10. Implementare il classificatore automatico dei bottleneck

## Sottotask

* [ ] Creare `diagnosis_engine.py`.
* [ ] Supportare almeno le classi:

  * memory bandwidth bound;
  * memory latency bound;
  * compute bound;
  * Tensor Core underutilization;
  * register-pressure bound;
  * shared-memory bound;
  * occupancy/latency-hiding problem;
  * insufficient parallelism;
  * launch bound;
  * CPU/runtime bound;
  * synchronization bound;
  * memory access inefficiency;
  * branch/divergence;
  * mixed bottleneck.
* [ ] Implementare regole basate su combinazioni di metriche.
* [ ] Non usare una singola metrica come prova conclusiva.
* [ ] Assegnare:

  * classe;
  * probability/confidence;
  * evidence;
  * counter-evidence.
* [ ] Esempio:

```yaml
classification:
  memory_bandwidth_bound: 0.84

evidence:
  dram_pct_peak: 92
  sm_pct_peak: 26
  arithmetic_intensity: low

counter_evidence:
  traffic_amplification: 1.55
```

* [ ] Implementare distinzione specifica fra:

  * bandwidth saturation;
  * bandwidth waste.
* [ ] Implementare regola:

  * occupancy alta ≠ automaticamente buona;
  * occupancy bassa ≠ automaticamente problema.
* [ ] Utilizzare scheduler/eligible warp per interpretare occupancy.
* [ ] Collegare ogni classe ai possibili test successivi.

## Condizione di completamento

Il macro-task è completo quando almeno una suite di kernel noti può essere classificata correttamente nei principali tipi di bottleneck e ogni diagnosi contiene **evidenze, contro-evidenze e prossimo test consigliato**.

---

# [ ] 11. Creare il sistema di ispezione della compilazione

## Sottotask

* [ ] Implementare `inspect_kernel()`.
* [ ] Estrarre automaticamente:

  * registers/thread;
  * shared memory;
  * local memory;
  * spill load/store;
  * occupancy teorica.
* [ ] Salvare PTX.
* [ ] Salvare SASS quando disponibile.
* [ ] Individuare istruzioni:

  * FP32;
  * FP16;
  * BF16;
  * INT;
  * Tensor Core/MMA;
  * load/store;
  * conversion/dequant.
* [ ] Distinguere:

  * storage dtype;
  * compute dtype;
  * accumulator dtype.
* [ ] Rilevare se un'intenzione “Tensor Core” viene realmente compilata in istruzioni appropriate.
* [ ] Rilevare conversioni inattese.
* [ ] Confrontare automaticamente compilazione baseline/candidate.
* [ ] Segnalare variazioni importanti:

  * registri;
  * shared memory;
  * spills;
  * instruction count.

## Condizione di completamento

Il macro-task è completo quando ogni modifica kernel genera un diff automatico delle risorse hardware e delle principali classi di istruzione prodotte dal compilatore.

---

# [ ] 12. Costruire il framework di microbenchmark dei kernel

## Sottotask

* [ ] Implementare `benchmark_kernel()`.
* [ ] Eseguire warm-up.
* [ ] Usare CUDA events o sistema equivalente.
* [ ] Sincronizzare correttamente.
* [ ] Eseguire abbastanza iterazioni.
* [ ] Calcolare distribuzione dei tempi.
* [ ] Eseguire benchmark per più shape.
* [ ] Eseguire benchmark per:

  * context;
  * batch;
  * dtype;
  * quantization group.
* [ ] Calcolare throughput utile:

  * GB/s;
  * output/s;
  * FLOP/s dove sensato.
* [ ] Confrontare:

  * reference;
  * baseline;
  * candidate.
* [ ] Calcolare speedup.
* [ ] Non utilizzare valori di timing di Nsight Compute come benchmark finale.

## Condizione di completamento

Il macro-task è completo quando qualsiasi kernel candidato può essere confrontato in modo riproducibile con reference e baseline su tutte le shape rilevanti.

---

# [ ] 13. Creare il framework di correctness e numerical validation

## Sottotask

* [x] Implementare `correctness_test()`.
* [x] Mantenere una reference affidabile.
* [ ] Confrontare:

  * max absolute error;
  * mean absolute error;
  * relative error;
  * cosine similarity quando appropriato.
* [ ] Controllare:

  * NaN;
  * Inf;
  * overflow;
  * underflow critico.
* [ ] Definire tolerance per dtype.
* [ ] Definire tolerance per quantizzazione.
* [ ] Testare edge case:

  * dimensioni non multiple del tile;
  * sequence corte;
  * sequence lunghe;
  * batch diversi;
  * valori estremi.
* [ ] Eseguire correctness prima del benchmark prestazionale completo.
* [x] Aggiungere test end-to-end del modello.
* [x] Se possibile confrontare logits/output.
* [x] Impedire promozione di kernel non corretti.

## Condizione di completamento

Il macro-task è completo quando nessun kernel può entrare nel ramo `accepted` senza aver superato automaticamente correctness unitario e almeno un controllo end-to-end.

---

# [ ] 14. Creare la Knowledge Base — fondamenti GPU

## Sottotask

* [ ] Creare struttura:

```text
knowledge/
    fundamentals/
    optimization_patterns/
    anti_patterns/
    llm_inference/
    architectures/
    case_studies/
    profiler_interpretation/
```

* [ ] Creare card atomiche su:

  * memoria globale;
  * L2;
  * L1;
  * shared memory;
  * registri;
  * coalescing;
  * cache locality;
  * arithmetic intensity;
  * roofline;
  * occupancy;
  * latency hiding;
  * warp scheduling;
  * register pressure;
  * spills;
  * bank conflicts;
  * branch divergence;
  * Tensor Core;
  * tiling;
  * register blocking;
  * shared-memory tiling;
  * instruction-level parallelism;
  * vectorized loads;
  * async copies;
  * software pipeline;
  * double buffering.
* [ ] Ogni card deve contenere:

  * principio;
  * perché funziona;
  * quando è rilevante;
  * metriche da guardare;
  * come verificarlo;
  * possibili azioni;
  * trade-off;
  * quando NON applicarlo;
  * fonte;
  * architettura;
  * confidence.
* [ ] Collegare ogni card alle metriche Nsight correlate.
* [ ] Aggiungere tag per retrieval.

## Condizione di completamento

Il macro-task è completo quando esistono almeno **50 card GPU atomiche e operative**, ognuna recuperabile tramite sintomo, metrica, tecnica o architettura.

---

# [ ] 15. Creare la Knowledge Base — pattern di ottimizzazione

## Sottotask

* [ ] Creare card su:

  * kernel fusion;
  * operator fusion;
  * epilogue fusion;
  * dequant fusion;
  * quantized GEMV;
  * quantized GEMM;
  * weight packing;
  * tiling;
  * split-K;
  * persistent kernels;
  * warp specialization;
  * software pipelining;
  * double buffering;
  * async global→shared;
  * swizzling;
  * L2-aware ordering;
  * vectorized memory access;
  * reductions;
  * online softmax;
  * CUDA Graphs;
  * launch reduction;
  * autotuning.
* [ ] Per ogni tecnica descrivere:

  * prerequisiti;
  * expected metric changes;
  * failure modes;
  * costi;
  * shape in cui è utile.
* [ ] Collegare tecnica → diagnosi.
* [ ] Collegare tecnica → controindicazioni.
* [ ] Collegare tecnica → architetture compatibili.

## Condizione di completamento

Il macro-task è completo quando a ogni principale classe di bottleneck corrispondono almeno **3 strategie concrete**, corredate da metriche che permettono di verificare se hanno funzionato.

---

# [ ] 16. Creare la Knowledge Base — anti-pattern e failure modes

## Sottotask

* [ ] Creare card specifiche per:

  * “massimizzare occupancy a prescindere”;
  * “più shared memory è sempre meglio”;
  * “fusion è sempre meglio”;
  * “tile più grande è sempre meglio”;
  * “bandwidth al 100% significa kernel perfetto”;
  * “SM utilization bassa significa compute inefficiente”;
  * “INT4 significa automaticamente 4× speedup”;
  * “meno kernel significa automaticamente più veloce”;
  * “microbenchmark più veloce significa modello più veloce”.
* [ ] Creare failure card su:

  * register explosion;
  * shared-memory explosion;
  * occupancy collapse;
  * spills;
  * bank conflicts;
  * non-coalesced access;
  * traffic amplification;
  * tail inefficiency;
  * insufficient CTA;
  * synchronization overhead;
  * launch gaps;
  * redundant dequantization;
  * materialized intermediates;
  * excessive dtype conversion.
* [ ] Per ogni failure mode aggiungere:

  * sintomo;
  * errore di ragionamento tipico;
  * metriche discriminanti;
  * test;
  * possibile soluzione.

## Condizione di completamento

Il macro-task è completo quando il retrieval può fornire all'agente **non soltanto idee da provare ma anche ragioni tecniche per cui l'idea proposta potrebbe peggiorare il kernel**.

---

# [ ] 17. Creare la Knowledge Base — LLM inference

## Sottotask

* [ ] Creare sezione specifica:

  * prefill;
  * decode;
  * GEMM;
  * GEMV;
  * quantized GEMM/GEMV;
  * QKV;
  * attention;
  * KV cache;
  * GQA;
  * RoPE;
  * RMSNorm;
  * MLP;
  * activation;
  * logits.
* [ ] Spiegare differenza arithmetic intensity fra prefill e decode.
* [ ] Spiegare perché decode batch 1 tende frequentemente a essere memory dominated.
* [ ] Inserire formule KV cache.
* [ ] Collegare context length → costo attention/KV.
* [ ] Creare card su:

  * FlashAttention;
  * Flash-Decoding;
  * PagedAttention;
  * MARLIN;
  * AWQ/TinyChat;
  * CUTLASS GEMM;
  * Triton matmul;
  * Triton fused softmax.
* [ ] Estrarre da ogni caso studio:

  * problema originale;
  * cambio di prospettiva;
  * soluzione;
  * metriche;
  * limiti;
  * cosa generalizzare.

## Condizione di completamento

Il macro-task è completo quando l'agente può ricevere una diagnosi come:

```text
decode / batch1 / quantized linear / bandwidth bound
```

e recuperare conoscenza specifica per quel tipo di workload, senza essere sommerso da materiale generico CUDA.

---

# [ ] 18. Creare la Knowledge Base per architettura GPU

## Sottotask

* [ ] Creare:

```text
architectures/
    ampere/
    ada/
    hopper/
    blackwell/
```

* [ ] Mappare capacità supportate.
* [ ] Aggiungere:

  * Tensor Core capabilities;
  * shared memory limits;
  * async copy capabilities;
  * istruzioni specializzate;
  * TMA dove applicabile;
  * warp-group capabilities dove applicabile.
* [ ] Associare ogni tecnica a:

  * minimum compute capability;
  * recommended architecture;
  * unsupported architectures.
* [ ] Fare in modo che il retrieval filtri automaticamente per hardware corrente.

## Condizione di completamento

Il macro-task è completo quando l'agente non propone come azione primaria un meccanismo hardware assente dalla GPU installata senza marcarlo come incompatibile.

---

# [ ] 19. Implementare il sistema di retrieval della Knowledge Base

## Sottotask

* [ ] Creare metadata strutturati per ogni card.
* [ ] Indicizzare:

  * operation;
  * phase;
  * bottleneck;
  * architecture;
  * dtype;
  * quantization;
  * symptom;
  * profiler metric;
  * optimization pattern.
* [ ] Implementare query costruite automaticamente dalla diagnosi.
* [ ] Evitare query generiche tipo:

```text
make kernel faster
```

* [ ] Generare query del tipo:

```text
quantized GEMV
decode
batch 1
Ada
memory bandwidth bound
traffic amplification
INT4
```

* [ ] Limitare il numero di card recuperate.
* [ ] Ordinare per:

  * compatibilità hardware;
  * compatibilità workload;
  * rilevanza diagnosi;
  * confidence;
  * qualità fonte.
* [ ] Implementare retrieval secondario “contrarian”:

  * failure modes della tecnica proposta;
  * controindicazioni;
  * metriche che potrebbero peggiorare.
* [ ] Registrare quali card hanno influenzato l'esperimento.

## Condizione di completamento

Il macro-task è completo quando, dato un report strutturato di un kernel, il sistema restituisce automaticamente un piccolo set di conoscenze **specifiche, compatibili con l'hardware e pertinenti al bottleneck** più un set di contro-argomentazioni.

---

# [ ] 20. Implementare il motore di generazione delle ipotesi

## Sottotask

* [ ] Creare schema obbligatorio:

```yaml
problem:
evidence:
hypothesis:
mechanism:
expected_changes:
risks:
counter_hypothesis:
experiment:
success_condition:
failure_condition:
```

* [ ] Vietare ipotesi senza meccanismo causale.
* [ ] Richiedere previsioni quantitative quando possibile.
* [ ] Esempio:

```text
DRAM bytes ↓ 15–25%
registers/thread ↑ 10–20
occupancy 75% → ~60%
kernel time ↓ almeno 8%
```

* [ ] Richiedere che l'agente dichiari quale metrica **non** dovrebbe cambiare.
* [ ] Richiedere almeno una contro-ipotesi.
* [ ] Collegare ipotesi alle knowledge card.
* [ ] Dare priorità a modifiche a basso rischio e alto impatto.
* [ ] Penalizzare esperimenti già provati senza nuove evidenze.

## Condizione di completamento

Il macro-task è completo quando l'agente non può modificare un kernel senza aver prodotto prima una previsione falsificabile degli effetti hardware e prestazionali.

---

# [ ] 21. Creare il sistema di esperimenti e memoria tecnica

## Sottotask

* [ ] Implementare `experiment_log()`.
* [ ] Salvare per ogni esperimento:

  * ID;
  * timestamp;
  * commit;
  * kernel;
  * workload;
  * hardware;
  * baseline;
  * modifica;
  * ipotesi;
  * knowledge card consultate;
  * codice candidate;
  * compile stats;
  * correctness;
  * microbenchmark;
  * profiler before;
  * profiler after;
  * end-to-end before;
  * end-to-end after;
  * decisione.
* [ ] Salvare esperimenti falliti.
* [ ] Salvare motivazione del fallimento.
* [ ] Creare fingerprint delle modifiche.
* [ ] Evitare ripetizione involontaria di prove equivalenti.
* [ ] Permettere query:

  * “abbiamo già provato BLOCK_K=256?”;
  * “quali fusion hanno aumentato i registri?”;
  * “cosa ha migliorato decode su context lungo?”.
* [ ] Implementare stato:

  * proposed;
  * compiling;
  * invalid;
  * benchmarked;
  * rejected;
  * accepted;
  * superseded.

## Condizione di completamento

Il macro-task è completo quando l'agente può recuperare automaticamente la storia tecnica delle precedenti ottimizzazioni e non deve “riscoprire” continuamente gli stessi risultati.

---

# [ ] 22. Implementare il comparatore automatico degli esperimenti

## Sottotask

* [x] Implementare `compare_experiment()`.
* [ ] Confrontare:

  * correctness;
  * kernel latency;
  * DRAM traffic;
  * cache;
  * register usage;
  * occupancy;
  * stalls;
  * TTFT;
  * prefill;
  * decode.
* [x] Calcolare regressioni per workload.
* [x] Definire soglie di accettazione.
* [x] Non accettare un kernel che migliora una sola shape sacrificando quelle dominanti senza decisione esplicita.
* [x] Pesare i workload per frequenza reale.
* [x] Evidenziare trade-off.
* [x] Implementare:

  * KEEP;
  * REJECT;
  * NEED_MORE_DATA.
* [x] Rendere motivazione obbligatoria.

## Condizione di completamento

Il macro-task è completo quando il sistema può decidere automaticamente se una modifica è globalmente utile rispetto agli obiettivi dichiarati, senza basarsi su una singola misura.

---

# [ ] 23. Implementare una pipeline specifica per kernel memory-bound

## Sottotask

* [ ] Quando classificato memory-bound, calcolare:

  * bandwidth % peak;
  * minimum bytes;
  * actual bytes;
  * traffic amplification.
* [ ] Controllare:

  * coalescing;
  * vectorization;
  * data layout;
  * repeated loads;
  * cache reuse;
  * unnecessary intermediates.
* [ ] Recuperare strategie:

  * fusion;
  * dequant fusion;
  * packing;
  * tiling;
  * shared-memory staging;
  * cache-aware ordering;
  * reduced precision.
* [ ] Se bandwidth è già quasi al limite, spostare l'obiettivo da:

```text
più GB/s
```

a:

```text
meno byte/output
```

* [ ] Verificare che la riduzione traffico non causi register/shared-memory explosion.

## Condizione di completamento

Il macro-task è completo quando ogni kernel memory-bound viene analizzato automaticamente distinguendo **saturazione della banda** da **traffico evitabile**.

---

# [ ] 24. Implementare una pipeline specifica per kernel latency-bound

## Sottotask

* [ ] Analizzare:

  * long scoreboard;
  * eligible warps;
  * active warps;
  * cache hit rate.
* [ ] Verificare locality.
* [ ] Verificare accessi scattered.
* [ ] Verificare numero di CTA.
* [ ] Verificare capacità di latency hiding.
* [ ] Considerare:

  * più parallelismo;
  * più ILP;
  * prefetch;
  * async copy;
  * migliore layout;
  * staging.
* [ ] Non confondere latency bound con bandwidth saturation.

## Condizione di completamento

Il macro-task è completo quando il sistema sa proporre esperimenti diversi per `memory latency` e `memory bandwidth`, giustificando la distinzione con metriche osservate.

---

# [ ] 25. Implementare una pipeline specifica per compute-bound

## Sottotask

* [ ] Analizzare:

  * SM throughput;
  * Tensor Core throughput;
  * instruction mix;
  * dtype.
* [ ] Verificare se il kernel utilizza le istruzioni hardware attese.
* [ ] Verificare shape Tensor Core friendly.
* [ ] Analizzare:

  * MMA tiling;
  * register blocking;
  * pipeline;
  * instruction dependencies.
* [ ] Valutare precisione inferiore quando consentito.
* [ ] Verificare che eventuale quantizzazione non aggiunga overhead maggiore del beneficio.
* [ ] Confrontare actual FLOP/s con roofline compute.

## Condizione di completamento

Il macro-task è completo quando un kernel compute-bound viene trattato con strategie di utilizzo delle pipeline di calcolo e non erroneamente con ottimizzazioni orientate alla DRAM.

---

# [ ] 26. Implementare pipeline register/shared-memory/occupancy

## Sottotask

* [ ] Monitorare automaticamente registers/thread.
* [ ] Monitorare shared memory/CTA.
* [ ] Calcolare CTA residenti/SM.
* [ ] Calcolare warp residenti.
* [ ] Evidenziare risorsa limitante.
* [ ] Controllare spills.
* [ ] Analizzare eligible warps.
* [ ] Impedire conclusione:

```text
occupancy più alta = migliore
```

* [ ] Valutare trade-off:

  * registri vs ILP;
  * tile size vs occupancy;
  * shared memory vs reuse.
* [ ] Creare esperimenti isolati sulle dimensioni del tile.

## Condizione di completamento

Il macro-task è completo quando il sistema sa spiegare **perché** l'occupancy è limitata e se quella limitazione è realmente correlata alle prestazioni osservate.

---

# [ ] 27. Implementare pipeline per quantizzazione

## Sottotask

* [ ] Distinguere sempre:

  * storage dtype;
  * load representation;
  * compute dtype;
  * accumulator dtype.
* [ ] Calcolare byte risparmiati dalla quantizzazione.
* [ ] Misurare costo:

  * unpack;
  * dequant;
  * scaling;
  * zero-point.
* [ ] Verificare se dequant genera buffer temporaneo.
* [ ] Considerare dequant + compute fusion.
* [ ] Analizzare weight packing.
* [ ] Analizzare allineamento.
* [ ] Analizzare access pattern.
* [ ] Integrare conoscenze MARLIN/AWQ/TinyChat.
* [ ] Verificare vantaggio end-to-end, non solo riduzione dimensione pesi.

## Condizione di completamento

Il macro-task è completo quando l'agente può determinare se una quantizzazione accelera realmente il workload oppure trasferisce il costo dalla memoria a unpack/dequant/compute inefficiente.

---

# [ ] 28. Implementare pipeline specifica attention/KV-cache

## Sottotask

* [ ] Calcolare dimensione KV cache per workload.
* [ ] Calcolare bytes letti/token.
* [ ] Separare:

  * prefill attention;
  * decode attention.
* [ ] Analizzare parallelismo sull'asse sequence.
* [ ] Analizzare GQA/MQA quando presenti.
* [ ] Collegare context length alle performance.
* [ ] Inserire pattern:

  * FlashAttention;
  * Flash-Decoding;
  * online softmax;
  * PagedAttention quando rilevante.
* [ ] Analizzare riuso K/V.
* [ ] Analizzare layout KV.
* [ ] Identificare context threshold in cui cambia il bottleneck.

## Condizione di completamento

Il macro-task è completo quando l'harness può spiegare perché l'attention cambia comportamento al crescere del context e proporre strategie diverse per prefill e decode.

---

# [ ] 29. Implementare pipeline per launch overhead e runtime

## Sottotask

* [ ] Analizzare timeline Nsight Systems.
* [ ] Misurare gap fra kernel.
* [ ] Identificare kernel molto brevi e numerosi.
* [ ] Identificare sincronizzazioni CPU/GPU.
* [ ] Identificare memory allocation nel loop.
* [ ] Considerare:

  * fusion;
  * CUDA Graphs;
  * pre-allocation;
  * launch batching.
* [ ] Distinguere kernel lento da sequenza inefficiente di kernel.
* [ ] Misurare end-to-end dopo ogni modifica.

## Condizione di completamento

Il macro-task è completo quando il sistema può riconoscere un workload launch-bound e impedire all'agente di perdere tempo ottimizzando l'aritmetica di kernel già molto brevi.

---

# [ ] 30. Creare il kernel workspace controllato

## Sottotask

* [ ] Creare:

```text
kernel_workspace/
    reference/
    baseline/
    candidate/
    accepted/
    rejected/
```

* [ ] Ogni candidate deve avere:

  * source;
  * build command;
  * target architecture;
  * metadata;
  * experiment ID.
* [ ] Automatizzare compilazione.
* [ ] Automatizzare failure capture.
* [ ] Salvare log compilatore.
* [ ] Salvare PTX/SASS.
* [ ] Consentire rollback.
* [ ] Integrare CUDA C++/Triton/altro backend usato da DS4.
* [ ] Rendere l'interfaccia verso DS4 stabile e intercambiabile.

## Condizione di completamento

Il macro-task è completo quando un agente può creare, compilare, testare, confrontare e scartare un kernel senza modificare manualmente lo stato stabile di DS4.

---

# [ ] 31. Creare l'interfaccia tool-facing per l'agente

## Sottotask

* [ ] Esporre tool ad alto livello:

```text
hardware_probe()
model_cost()
benchmark_model()
trace_model()
rank_hotspots()
profile_kernel()
inspect_kernel()
benchmark_kernel()
correctness_test()
search_knowledge()
generate_diagnosis()
compare_experiment()
experiment_history()
```

* [x] Evitare di esporre output enormi non filtrati.
* [x] Restituire JSON sintetici.
* [x] Aggiungere unità sempre esplicite.
* [ ] Aggiungere confidence.
* [x] Aggiungere provenance/source.
* [x] Permettere drill-down quando l'agente chiede più dettaglio.
* [x] Limitare token sprecati su profiler output irrilevante.

## Condizione di completamento

Il macro-task è completo quando Qwen può svolgere un ciclo completo di profiling e ottimizzazione attraverso tool strutturati senza dover interpretare direttamente migliaia di righe di output Nsight.

---

# [ ] 32. Creare il report sintetico standard di un kernel

## Sottotask

* [ ] Definire uno schema simile:

```yaml
kernel:
operation:
phase:
shape:
calls:
total_time_ms:
median_us:

roofline:
  arithmetic_intensity:
  memory_floor_us:
  compute_floor_us:

memory:
  minimum_bytes:
  dram_bytes:
  traffic_amplification:
  dram_gbps:
  dram_pct_peak:
  l2_hit_pct:

compute:
  sm_pct_peak:
  tensor_pct_peak:

resources:
  registers_per_thread:
  shared_memory:
  spills:
  theoretical_occupancy:
  achieved_occupancy:

scheduler:
  eligible_warps:
  issued_warps:

stalls:
  long_scoreboard:
  short_scoreboard:
  barrier:

diagnosis:
  primary:
  confidence:
  evidence:
```

* [ ] Produrre sempre baseline e candidate affiancati.

## Condizione di completamento

Il macro-task è completo quando l'agente può capire la situazione principale di un kernel leggendo un singolo oggetto strutturato di dimensioni limitate.

---

# [ ] 33. Creare la pipeline automatica “hotspot → proposta”

## Sottotask

* [ ] Benchmark iniziale.
* [ ] Trace.
* [ ] Ranking hotspot.
* [ ] Selezione top candidate.
* [ ] Detailed profile.
* [ ] Cost model.
* [ ] Diagnosi.
* [ ] Retrieval KB.
* [ ] Retrieval failure modes.
* [ ] Generazione ipotesi.
* [ ] Stima expected improvement.
* [ ] Selezione esperimento.
* [ ] Passaggio al coding agent.

## Condizione di completamento

Il macro-task è completo quando il sistema può partire da un'inferenza DS4 non ottimizzata e arrivare autonomamente a una **specifica tecnica di modifica di un kernel**, accompagnata da evidenze e previsione quantitativa.

---

# [ ] 34. Creare la pipeline automatica “proposta → verdict”

## Sottotask

* [ ] Compilare candidate.
* [ ] Ispezionare risorse.
* [ ] Eseguire correctness.
* [ ] Se correctness fallisce → REJECT.
* [ ] Microbenchmark.
* [ ] Se microbenchmark peggiora significativamente → REJECT o investigazione.
* [ ] Profilare candidate.
* [ ] Confrontare metriche previste/reali.
* [ ] Eseguire benchmark end-to-end.
* [ ] Valutare tutte le workload rilevanti.
* [ ] Emettere:

  * KEEP;
  * REJECT;
  * NEED_MORE_DATA.
* [ ] Aggiornare experiment DB.
* [ ] Aggiornare eventuali lesson learned.

## Condizione di completamento

Il macro-task è completo quando una modifica può attraversare autonomamente l'intero percorso dal codice candidato a un verdetto basato su correttezza, microbenchmark, profiler ed end-to-end.

---

# [ ] 35. Creare un sistema di apprendimento dagli esperimenti

## Sottotask

* [ ] Dopo ogni esperimento generare:

```text
what_we_expected
what_happened
why_the_prediction_was_right_or_wrong
lesson
```

* [ ] Distinguere:

  * lesson specifica della shape;
  * lesson specifica della GPU;
  * lesson generale.
* [ ] Promuovere lesson ricorrenti nella knowledge base.
* [ ] Non promuovere automaticamente conclusioni da un solo esperimento.
* [ ] Aggiungere confidence.
* [ ] Collegare lesson agli experiment ID.
* [ ] Creare card locali specifiche di DS4/Qwen/GPU.

## Condizione di completamento

Il macro-task è completo quando il sistema migliora progressivamente la propria conoscenza locale e può usare esperimenti precedenti come evidenza nelle decisioni future.

---

# [ ] 36. Costruire benchmark di validazione dell'intero harness

## Sottotask

Creare piccoli kernel artificiali con bottleneck intenzionali:

* [ ] bandwidth-bound;
* [ ] latency-bound;
* [ ] compute-bound;
* [ ] register-pressure;
* [ ] shared-memory pressure;
* [ ] bank conflicts;
* [ ] uncoalesced access;
* [ ] insufficient parallelism;
* [ ] launch-bound;
* [ ] excessive fusion;
* [ ] spill-heavy.
* [ ] Verificare che il profiler osservi il sintomo corretto.
* [ ] Verificare che il classifier assegni la classe corretta.
* [ ] Verificare che la KB recuperi materiale corretto.
* [ ] Verificare che l'agente proponga una trasformazione plausibile.
* [ ] Verificare che il comparatore riconosca miglioramento/regressione.

## Condizione di completamento

Il macro-task è completo quando l'harness supera una suite di casi con **bottleneck conosciuto a priori** e dimostra che la catena misurazione → diagnosi → retrieval → esperimento funziona correttamente.

---

# [ ] 37. Eseguire la prima caratterizzazione completa di Qwen su DS4

## Sottotask

* [ ] Congelare una baseline.
* [ ] Salvare hardware profile.
* [ ] Salvare modello e quantizzazione.
* [ ] Eseguire workload canonici.
* [ ] Misurare:

  * TTFT;
  * prefill;
  * decode;
  * VRAM.
* [ ] Generare trace completo.
* [ ] Identificare top hotspot.
* [ ] Classificare per:

  * attention;
  * quantized linear;
  * MLP;
  * RMSNorm;
  * RoPE;
  * launch/runtime;
  * altro.
* [ ] Profilare kernel dominanti.
* [ ] Calcolare roofline.
* [ ] Calcolare traffic amplification.
* [ ] Creare lista ordinata del potenziale di miglioramento:

```text
impatto totale
×
margine teorico
×
confidence diagnosi
÷
costo/risco implementazione
```

## Condizione di completamento

Il macro-task è completo quando esiste un report quantitativo che identifica **dove DS4/Qwen passa il tempo, perché e quali 3–5 interventi hanno il miglior rapporto potenziale/costo** sulla GPU installata.

---

# [ ] 38. Ottimizzare il primo kernel reale end-to-end

## Sottotask

* [ ] Selezionare il primo hotspot sulla base dei dati.
* [ ] Produrre diagnosi.
* [ ] Recuperare KB.
* [ ] Formulare ipotesi.
* [ ] Predire metriche.
* [ ] Implementare una modifica.
* [ ] Compilare.
* [ ] Correctness.
* [ ] Microbenchmark.
* [ ] Profiling.
* [ ] End-to-end.
* [ ] Registrare esperimento.
* [ ] Emettere KEEP/REJECT.
* [ ] Ripetere almeno una seconda iterazione basata sui risultati della prima.

## Condizione di completamento

Il macro-task è completo quando almeno una modifica reale di un kernel DS4 produce un miglioramento end-to-end statisticamente verificabile **senza regressione di correttezza**, e l'intero processo è stato eseguito attraverso l'harness.

---

# [ ] 39. Ottimizzare separatamente prefill e decode

## Sottotask

* [ ] Costruire una classifica hotspot prefill.
* [ ] Costruire una classifica hotspot decode.
* [ ] Confrontare arithmetic intensity.
* [ ] Confrontare bandwidth.
* [ ] Confrontare Tensor Core utilization.
* [ ] Confrontare KV-cache cost.
* [ ] Non riutilizzare automaticamente la stessa configurazione kernel.
* [ ] Consentire dispatch differente per:

  * phase;
  * batch;
  * context;
  * shape.

## Condizione di completamento

Il macro-task è completo quando DS4 può selezionare strategie/kernel differenti per workload significativamente diversi invece di utilizzare un'unica implementazione “media”.

---

# [ ] 40. Implementare dispatch e autotuning controllato

## Sottotask

* [ ] Identificare le dimensioni rilevanti:

  * M;
  * N;
  * K;
  * batch;
  * sequence;
  * dtype;
  * quantization.
* [ ] Definire candidate configurations.
* [ ] Applicare limiti hardware prima della compilazione.
* [ ] Escludere configurazioni palesemente impossibili.
* [ ] Benchmarkare candidate.
* [ ] Salvare migliore configurazione per classe di shape.
* [ ] Evitare overfitting su una singola shape.
* [ ] Generare tabella di dispatch.
* [ ] Prevedere fallback robusto.

## Condizione di completamento

Il macro-task è completo quando DS4 può scegliere automaticamente la configurazione kernel migliore per classi di workload rappresentative usando risultati misurati e persistenti.

---

# [ ] 41. Integrare tutto nel ciclo operativo dell'agente

## Sottotask

* [ ] Creare prompt/system contract per il kernel agent.
* [ ] Esporre tool.
* [ ] Esporre knowledge retrieval.
* [ ] Esporre experiment history.
* [ ] Dare all'agente solo il contesto necessario.
* [ ] Definire il ciclo:

```text
1. benchmark
2. hotspot
3. diagnosis
4. knowledge retrieval
5. counter-retrieval
6. hypothesis
7. prediction
8. code
9. correctness
10. microbenchmark
11. profile
12. end-to-end
13. verdict
14. lesson
```

* [ ] Impedire salti di fase senza motivazione.
* [ ] Impedire l'accettazione basata esclusivamente sul giudizio dell'LLM.
* [ ] Rendere profiler e benchmark fonte finale del verdetto.

## Condizione di completamento

Il macro-task è completo quando un agente può condurre autonomamente più cicli consecutivi di kernel engineering e ogni decisione importante è ancorata a dati, conoscenza recuperata e risultati sperimentali.

---

# [ ] 42. Definition of Done dell'intero progetto

Il progetto è completo soltanto quando **tutte** le condizioni seguenti sono vere:

* [ ] la GPU viene caratterizzata automaticamente;
* [ ] il modello viene caratterizzato automaticamente;
* [ ] prefill e decode sono benchmarkati separatamente;
* [ ] esiste una suite riproducibile di workload;
* [ ] Nsight Systems individua automaticamente gli hotspot;
* [ ] Nsight Compute produce report strutturati sui kernel selezionati;
* [ ] vengono misurati bandwidth, cache, occupancy, scheduler, stalls, registers e shared memory;
* [ ] vengono calcolati minimum bytes e traffic amplification;
* [ ] viene calcolato il roofline teorico;
* [ ] storage dtype e compute dtype sono distinti;
* [ ] la quantizzazione viene valutata anche come costo di unpack/dequant;
* [ ] KV-cache e attention hanno analisi dedicate;
* [ ] launch overhead e CUDA Graphs sono considerati come possibile livello di ottimizzazione;
* [ ] esiste una knowledge base tecnica atomica;
* [ ] esistono anti-pattern e controindicazioni;
* [ ] esistono case study LLM/GPU;
* [ ] il retrieval filtra per architettura e workload;
* [ ] ogni ipotesi contiene una previsione falsificabile;
* [ ] ogni candidate passa correctness test;
* [ ] ogni candidate viene microbenchmarkato;
* [ ] il benchmark finale viene eseguito fuori dal profiler;
* [ ] ogni modifica viene confrontata end-to-end;
* [ ] tutti gli esperimenti vengono memorizzati;
* [ ] i fallimenti vengono memorizzati tanto quanto i successi;
* [ ] l'agente recupera esperimenti precedenti prima di ripetere una strategia;
* [ ] il sistema ha superato test sintetici con bottleneck noti;
* [ ] almeno un kernel reale di DS4 è stato migliorato;
* [ ] almeno un miglioramento produce un beneficio end-to-end misurabile su Qwen;
* [ ] il sistema sa produrre autonomamente un nuovo ciclo di ottimizzazione senza intervento umano sui passaggi di profiling, diagnosi e valutazione.

## Condizione finale di successo

```text
Un agente che non possiede preventivamente una conoscenza profonda
della specifica GPU deve essere in grado di:

1. capire dove Qwen/DS4 sta perdendo tempo;
2. identificare la classe del collo di bottiglia;
3. recuperare le conoscenze tecniche pertinenti;
4. riconoscere i rischi della strategia scelta;
5. formulare un'ipotesi misurabile;
6. scrivere o modificare un kernel;
7. verificarne la correttezza;
8. misurarne l'effetto hardware;
9. misurarne l'effetto reale sull'inferenza;
10. accettare o scartare autonomamente la modifica;
11. conservare ciò che ha imparato;
12. utilizzare quell'apprendimento negli esperimenti successivi.
```

Quando questa condizione è soddisfatta, il sistema non è più soltanto un “LLM che prova a scrivere kernel”, ma un **ambiente autonomo di GPU performance engineering**.
