# Indice tecnico di DwarfStar

Questo file è la mappa di ingresso al progetto. Deve essere letto insieme ad
`AGENT.md` e `todo.md` prima di iniziare un task e aggiornato quando un file
rilevante viene aggiunto, rimosso, rinominato o cambia responsabilità.

DwarfStar non è un runner GGUF generico. È un motore specializzato che riconosce
un insieme ristretto di famiglie e forme di modello, ne valida esplicitamente i
metadata e costruisce percorsi di inferenza dedicati. La semantica del modello e
lo scheduling principale vivono in C; Objective-C è usato solo per il runtime
Metal, mentre CUDA e HIP contengono le implementazioni GPU specifiche.

## Ordine della spiegazione dettagliata

Questi sono i capitoli da seguire per comprendere il motore dall'esterno verso
l'interno. Ogni capitolo presuppone quelli precedenti.

1. **Confini del sistema e build**
   - Quali eseguibili vengono prodotti.
   - Come vengono scelti backend, modello e modalità di esecuzione.
   - Separazione fra policy applicativa, semantica del modello e kernel.

2. **API pubblica: engine, token e sessione**
   - Ruolo di `ds4_engine`, `ds4_session` e `ds4_tokens`.
   - Vita di un engine e di una sessione.
   - Contratto di `sync`, `eval`, logits, sampling e snapshot.

3. **Apertura e lettura del GGUF**
   - Apertura del file, lock, `mmap`, header, metadata e directory dei tensor.
   - Perché i pesi restano mmap-backed.
   - Ricerca, shape, tipo, offset e allineamento dei tensor.

4. **Riconoscimento e validazione del modello**
   - Selezione della famiglia tramite `general.architecture`.
   - Shape supportate e parametri globali.
   - Validazione semantica di attention, RoPE, MoE, quantizzazione e MTP.
   - Rifiuto anticipato di modelli incompatibili.

5. **Binding e disposizione dei pesi**
   - Collegamento fra nomi GGUF e strutture interne dei layer.
   - Pesi residenti, no-copy e pesi streamed da SSD.
   - Placement su una o più GPU.

6. **Tokenizer e protocollo di chat**
   - Caricamento del vocabolario e dei merge.
   - Token speciali, encoding/decoding e template per famiglia.
   - Costruzione del prompt completo e modalità thinking.

7. **Creazione della sessione**
   - Allocazione di tensori persistenti, scratch, logits e KV.
   - Context capacity e stato iniziale.
   - Differenze fra CPU reference, Metal, CUDA, ROCm e distribuito.

8. **Sincronizzazione del prompt e riuso del prefisso**
   - Confronto fra token richiesti e timeline già viva.
   - Riuso, estensione, rollback o ricostruzione della KV.
   - Scelta fra prefill e decode.

9. **Prefill, passo per passo**
   - Embedding e stato residuo.
   - Elaborazione layer per layer.
   - Norm, proiezioni attention, RoPE, aggiornamento KV e attention.
   - FFN/MoE, shared expert, routed expert e combinazione del residuo.
   - Output norm, output head e logits del token successivo.
   - Chunking dei prompt lunghi.

10. **Decode autoregressivo e sampling**
    - Lettura dei logits correnti.
    - Argmax o filtri temperature/top-k/top-p/min-p.
    - Inserimento del token scelto, aggiornamento KV e nuovo forward pass.
    - EOS, stop sequence e streaming del testo.

11. **Attention e KV cache**
    - Layout delle cache raw e compresse per DeepSeek/GLM.
    - Posizioni, RoPE, finestre, indexer e selezione delle righe.
    - Capacità, lifetime, memoria per token e vincoli di correttezza.

12. **Mixture of Experts**
    - Router, scelta top-k, pesi di routing e normalizzazione.
    - Shared expert e routed expert.
    - Cache degli esperti, hot set e sovrapposizione I/O/calcolo.

13. **Backend GPU**
    - API tensoriale condivisa.
    - Runtime e command lifetime Metal.
    - CUDA, cuBLAS, kernel custom e multi-GPU.
    - ROCm/HIP, hipBLASLt e differenze di piattaforma.

14. **Batching e più sessioni**
    - Proprietà della KV per sessione.
    - Batching di decode e batch misti prefill/decode.
    - Fallback ordinati e oracle full-vocabulary.

15. **SSD streaming dei pesi**
    - Quali pesi restano residenti.
    - Lettura degli esperti e delle layer mancanti.
    - Cache, preload, cold mode e sincronizzazione con i comandi GPU.

16. **Snapshot e KV cache su disco**
    - Serializzazione dello stato di una sessione.
    - Envelope, payload ABI, fingerprint e compatibilità.
    - Politiche differenti di CLI, server e Agent Q.

17. **Inferenza distribuita e tensor parallel**
    - Suddivisione del modello fra worker.
    - Trasporto di attivazioni e logits.
    - Ownership distribuita della KV.
    - TCP/RDMA e ricomposizione degli snapshot.

18. **MTP, DSpark e speculative decoding**
    - Support model, proposta e verifica.
    - Accettazione completa o parziale.
    - Rollback della KV e identità con il decode ordinario.

19. **Server**
    - Parsing OpenAI/Responses/Anthropic.
    - Coda dei job, worker residenti e coordinatore del batching.
    - SSE, tool call, prompt rendering e disk KV policy.

20. **Agent Q**
    - Transcript, sessioni, strumenti e ciclo agentico.
    - Stato della directory di lavoro e tool replay.
    - Session switching, persistenza, future diramazioni KV e path-dependent
      tool calls.

21. **Correttezza e osservabilità**
    - Dump di token, logits e top-logprob.
    - Oracle ufficiali, llama.cpp e golden locali.
    - NLL/perplexity, test brevi e long-context.
    - Bit-exact interno e tolleranze cross-engine.

22. **Prestazioni e diagnosi**
    - TTFT, prefill e decode token/s.
    - Profiler dei singoli stadi.
    - Memoria, cache hit, throughput aggregato e regressioni.

## Mappa dei file principali

### Istruzioni, obiettivi e documentazione

- `AGENT.md` — vincoli architetturali, qualità, sicurezza e test obbligatori per
  chi modifica il progetto.
- `todo.md` — roadmap ordinata, obiettivi Qwen3.6 e gate di correttezza.
- `README.md` — manuale utente e panoramica delle funzionalità, inclusi backend,
  streaming, distribuito, server, agent, KV e strumenti diagnostici.
- `MODEL_CARD.md` — descrizione dei meccanismi specifici dei modelli DeepSeek,
  in particolare attention compressa, indexer e KV.
- `QA_BEFORE_RELEASES.md` — matrice manuale e automatica che blocca una release
  quando un percorso non è stato verificato.
- `CONTRIBUTING.md` — regole di sviluppo, build, test e contributi.
- `STRIXHALO.md` — note operative per build ed esecuzione ROCm su Strix Halo.
- `PROJECT_INDEX.md` — questo indice e ordine della spiegazione tecnica.
- `docs/qwen36-drift-hypotheses.md` — registro vivo delle ipotesi sul drift
  Qwen3.6, con evidenze, esperimenti ordinati e gate dei 32 token.
- `docs/qwen36-numerics-lab.md` — banco differenziale per distinguere errori
  semantici, roundoff CPU/CUDA e policy aritmetiche Q4_K/Q8_K; include l'audit
  del percorso Ollama/llama.cpp senza eseguire Ollama.
- `docs/qwen36-performance-experiments-2026-08-05.md` — registro riproducibile
  dei test prestazionali Qwen CUDA, inclusi candidati scartati, confronto locale
  con LM Studio, gate 500/15 e verifiche numeriche associate.
- `docs/qwen36-performance-ledger.md` — registro canonico e compatto delle
  ottimizzazioni Qwen CUDA mantenute, scartate o ancora da confermare; unifica
  le decisioni disperse nei documenti datati e conserva il ciclo sperimentale.
- `docs/research/cuda/` — componente della knowledge base dedicato a toolchain,
  profiling, SASS/resource usage, primitive Ampere `sm_86` e mappa delle
  prossime ipotesi CUDA falsificabili per DS4.
- `docs/qwen36-lmstudio-decoding-analysis-2026-08-06.md` — guida didattica al
  decoding LM Studio/llama.cpp su RTX 3090: MMVQ Q8_1, caratteristiche GPU,
  implementazioni DS4 riuscite o scartate, risultati dei gate e piano per
  replicare il percorso veloce senza quantizzare la KV cache.
- `docs/qwen36-cuda-optimization-progression.md` — checklist ordinata delle
  ottimizzazioni CUDA Qwen3.6 Q4_K_S ancora da sviluppare: ricerca upstream e
  letteratura, gate numerici/prestazionali, budget RTX 3090 da 24 GB, policy di
  rollback e confronto CPU obbligatoriamente finale.
- `docs/qwen36-mtp-design.md` — progetto di supporto Qwen3.6 MTP: audit dei
  repository LM Studio/Unsloth/llama.cpp e del GGUF NextN, semantica del kernel,
  allineamento token-hidden, modello prestazionale, verifier target microbatch,
  rollback Gated DeltaNet, scheduling adattivo, fallback e gate di test per il
  target CUDA Q4_K_S/Q4_K_M.
- `docs/agent-api.md` — contratto dell'estensione Responses `agentic`: registry
  statico, capability tool/skill disgiunte, checkpoint SSD gerarchici e
  semantica di `return` senza full prefill del parent.
- `docs/agentic-checkpoint-validation-2026-08-07.md` — gate CUDA Q4_K_S dei
  task 10–11: equivalenza bit-exact, casi lunghi 10k, rollback MTP, metriche
  SSD/GPU e copertura della suite HTTP agentica.
- `docs/constrained-json-decoding-plan.md` — checklist e criteri di completamento
  per simulatore DSML, JSON Schema, masking dei logits e output strutturati.
- `docs/agent_performance_contract.md` — gate operativo per gli esperimenti CUDA:
  baseline, ipotesi falsificabile, correttezza, statistica e verdetto.
- `docs/todo-harness.md` — roadmap completa del GPU kernel engineering harness.
- `docs/hardware/` — inventario della macchina RTX 3090 e note di riferimento
  su GA102, compute capability 8.6, memoria, numerica CUDA e determinismo.

### Build ed eseguibili

- `Makefile` — selezione piattaforma, compilazione C/Objective-C/CUDA/ROCm,
  gate rapidi e confronto model-backed `perf-direction-ab` in un comando,
  linking, target dei binari e suite di test.
- `download_model.sh` — download dei modelli e support model conosciuti dal
  progetto.
- `ds4_cli.c` — eseguibile `ds4`: parsing opzioni, modalità one-shot e REPL,
  prompt, generazione, sampling, perplexity e dump diagnostici.
- `ds4_server.c` — eseguibile `ds4-server`: API HTTP compatibili, code, worker,
  session batching, streaming, tool-call mapping e policy della KV su disco;
  per Responses gestisce inoltre il namespace `agentic`, il masking DSML dei
  nomi e le frame gerarchiche delle skill. Per Qwen3.6 Q4_K_S abilita
  automaticamente una cache durevole con budget condiviso fra sessioni,
  prefissi di sistema e checkpoint delle skill; le history Responses vengono
  canonicalizzate senza reasoning storico prima della persistenza.
- `ds4_agent.c` — eseguibile `ds4-agent`: TUI, transcript, sessioni persistenti,
  ciclo tool, file/shell/web tools e orchestrazione della generazione.
- `ds4_bench.c` — benchmark di prefill/decode a diverse frontiere di contesto;
  ripete gli sweep mantenendo il modello residente e usa snapshot o replay fuori
  dalle finestre cronometrate.
- `tools/perf_harness.py` — CLI senza dipendenze che orchestra probe hardware,
  suite ripetute di `ds4-bench` e confronti baseline/candidate in JSON.
- `tools/perf-qwen-validate.sh`, `tools/perf-qwen-long-context.sh` — workflow
  eseguibili richiamati dall'harness per build/correctness e per il ciclo
  profile, direction A/B e slow sui contesti 8K–16K.
- `performance/workloads.yaml` — workload canonici direction/quick/standard/slow/exhaustive,
  separati per fase, batch, context e prefill chunk.
- `performance/README.md` — comandi brevi e confini verificati del harness.
- `performance/references.md` — fonti primarie, implementazioni di confronto e
  letteratura HTML, con criteri espliciti per separare evidenza e fuffa.
- `docs/research/INDEX.md` — ingresso alla knowledge base locale: paper HTML
  archiviati e hashati, schede delle piattaforme, fork RTX 3090, guide tematiche
  e indici gerarchici con intervalli di riga.
- `docs/research/papers/manifest.json`, `corpus-lock.json` — selezione motivata
  dei paper e provenance esatta degli artefatti scaricati.
- `docs/research/sources-lock.json` — commit fissati e grado di evidenza dei
  runtime, fork e raccolte operative analizzati.
- `tools/fetch_research_corpus.py` — scarica arXiv HTML, usa ar5iv come fallback
  dichiarato e genera il testo semantico locale.
- `tools/build_research_indexes.py` — rigenera catalogo e indici con range di
  riga calcolati dal contenuto effettivo.
- `tests/test_research_corpus.py` — verifica completezza, SHA, range, indici e
  limite dei link esterni nelle schede repository.
- `ds4_eval.c` — harness di valutazione end-to-end su piccoli set di domande,
  con rendering, inferenza reale ed estrazione/valutazione delle risposte.
- `ds4_help.c`, `ds4_help.h` — testo e rendering condiviso dell'help dei binari.
- `linenoise.c`, `linenoise.h` — editor di linea usato da CLI e agent.

### API, modello e orchestrazione centrale

- `ds4.h` — API pubblica stretta: opzioni engine, token, sessioni, sync/eval,
  logits, sampling filtrato, batching, snapshot, checkpoint ricorrenti Qwen
  per skill e callback.
- `ds4.c` — nucleo verticale del motore: parser GGUF mmap-backed, metadata,
  shape dei modelli, tensor binding, tokenizer, CPU reference, costruzione e
  scheduling del grafo, sessioni, prefill/decode, KV, logits, MTP e payload
  degli snapshot. Include il percorso specializzato Qwen `qwen35` text-only:
  validazione del 27B Q4_K_M e Q4_K_S, tokenizer/template Qwen e sessione ibrida
  full-attention/Gated DeltaNet CUDA e prefill layer-major a chunk. Espone il
  salvataggio/ripristino session-scoped su SSD dello stato ricorrente Qwen per
  le frontiere delle skill; per Qwen3.6 Q4_K_S il payload generale portabile
  comprende inoltre timeline, logits, stato GDN, KV full-attention e stato MTP.
  Il fork Qwen in memoria resta escluso. Il profiler Qwen opzionale emette
  tempi strutturati attention/FFN per ogni layer di prefill e decode, consumati
  dal comando profile-network del performance harness.
- `ds4_gpu.h` — interfaccia tensoriale condivisa fra il grafo C e i backend
  accelerati; descrive tensori opachi, operazioni, attention e batching,
  incluse le due primitive interne Qwen per full-attention gated e Gated
  DeltaNet.
- `ds4_gpu_args.c`, `ds4_gpu_args.h` — parsing condiviso di dispositivi, budget
  VRAM e configurazione multi-GPU.
- `ds4_layer_pack.c`, `ds4_layer_pack.h` — algoritmo puro C per assegnare
  intervalli contigui di layer ai tier/GPU rispettando i budget.

### Backend Metal

- `ds4_metal.m` — creazione del device, queue e library Metal; viste no-copy
  dei pesi mmap, command buffer, tensori persistenti, scratch e wrapper dei
  kernel.
- `metal/norm.metal` — RMS normalization e operazioni di normalizzazione.
- `metal/dense.metal` — moltiplicazioni dense e proiezioni.
- `metal/flash_attn.metal` — nuclei di attention accelerata.
- `metal/dsv4_kv.metal` — aggiornamento e lettura delle strutture KV specifiche.
- `metal/dsv4_hc.metal` — operazioni hyper-connection DeepSeek.
- `metal/moe.metal` — routing e calcolo Mixture-of-Experts.
- `metal/dsv4_rope.metal` — trasformazioni RoPE.
- `metal/dsv4_misc.metal` — operazioni specializzate non collocate negli altri
  moduli.
- `metal/softmax.metal`, `metal/argsort.metal`, `metal/sum_rows.metal` —
  riduzioni, ranking e normalizzazione usati da attention/router.
- `metal/get_rows.metal`, `metal/set_rows.metal`, `metal/cpy.metal`,
  `metal/concat.metal`, `metal/repeat.metal`, `metal/unary.metal`,
  `metal/glu.metal`, `metal/bin.metal` — primitive tensoriali di supporto.

### Backend CUDA

- `ds4_cuda.cu` — runtime CUDA principale: allocazione/copie, cuBLAS, kernel
  custom, quantizzazione, attention, MoE, KV, batching e supporto multi-GPU;
  contiene inoltre embedding/matvec Q4_K/Q5_K/Q6_K, RoPE parziale, attention gated
  e aggiornamento ricorrente Qwen task 6, più i kernel multi-riga causali e i
  GEMM di prefill Qwen. Il percorso MTP Qwen aggiunge embedding/matvec Q4_0,
  range device distinti per target e sidecar, kernel warp-8 Q4_0 del drafter e
  Q4_K/Q5_K/Q6_K microbatch del verifier. I Qwen single-GPU che entrano in VRAM sono copiati per
  default sul device; `DS4_CUDA_NO_MODEL_COPY=1` conserva il percorso host-map
  diagnostico.
- `ds4_iq2_tables_cuda.inc` — tabelle costanti per dequantizzazione/calcolo IQ2
  nel backend CUDA.

### Backend ROCm

- `ds4_rocm.cu` — unità di build HIP/ROCm e integrazione del backend.
- `ds4_rocm.h` — API e tipi specifici ROCm esposti al resto del runtime.
- `ds4_rocm_compat.cu` — livello di compatibilità fra API comuni e backend
  ROCm.
- `ds4_rocm_unavailable.cu` — stub diagnostico per build senza ROCm.
- `rocm/ds4_rocm_runtime.cuh` — device, stream, memoria e lifecycle runtime.
- `rocm/ds4_rocm_common.cuh` — tipi, macro e utility condivise.
- `rocm/ds4_rocm_hipblaslt.cuh`, `rocm/ds4_rocm_matmul.cuh` — integrazione
  hipBLASLt e moltiplicazioni.
- `rocm/ds4_rocm_attention.cuh`,
  `rocm/ds4_rocm_attention_launch.cuh` — kernel e launcher attention.
- `rocm/ds4_rocm_norm_rope.cuh` — norm e RoPE.
- `rocm/ds4_rocm_moe.cuh`, `rocm/ds4_rocm_moe_launch.cuh`,
  `rocm/ds4_rocm_router.cuh`, `rocm/ds4_rocm_shared_expert.cuh` — router,
  shared/routed experts e launcher MoE.
- `rocm/ds4_rocm_compressor.cuh`, `rocm/ds4_rocm_indexer.cuh`,
  `rocm/ds4_rocm_hc.cuh` — attention compressa, selezione e hyper-connection.
- `rocm/ds4_rocm_fp8_kv.cuh`, `rocm/ds4_rocm_fp8_kv_launch.cuh` — cache KV FP8.
- `rocm/ds4_rocm_q8.cuh` — primitive per tensori Q8.
- `rocm/ds4_rocm_embedding_launch.cuh`,
  `rocm/ds4_rocm_output.cuh`,
  `rocm/ds4_rocm_hc_output_launch.cuh`,
  `rocm/ds4_rocm_misc_launch.cuh` — embedding, output head e launcher
  specializzati.
- `rocm/ds4_rocm_glm.cuh` — percorsi specifici GLM.
- `rocm/ds4_rocm_current_api_compat.cuh` — adattamento alla versione corrente
  dell'API GPU condivisa.

### SSD, KV persistente e stato

- `ds4_ssd.c`, `ds4_ssd.h` — parsing dei budget SSD, mapping/allocazioni e
  primitive comuni usate dallo streaming.
- `ds4_kvstore.c`, `ds4_kvstore.h` — formato durevole dei checkpoint KV,
  validazione dell'envelope e del payload ABI, lookup, budget/eviction e
  gestione dei file; per Qwen rende durevoli soltanto gli anchor
  `agent-system`, limitati a 10 con LRU stretto. `ds4_server.c` conserva invece
  una sola history HDS temporanea per slot (`.dsh`, cambio system prompt con
  chiamata child pendente) e checkpoint deep-skill temporanei per `call_id`
  (`.dsk`); entrambi vengono consumati al ritorno e non sopravvivono al server.
- `ds4_streaming_hotlist.inc` — hotlist degli esperti DeepSeek da precaricare.
- `ds4_streaming_hotlist_glm52.inc` — hotlist equivalente per GLM 5.2.

### Multi-GPU e distribuito

- `ds4_gpu_mgpu.h` — strutture e API del runtime multi-tier/multi-GPU.
- `ds4_distributed.c`, `ds4_distributed.h` — coordinatore e worker di pipeline
  distribuita, protocollo, routing dei layer, KV distribuita e snapshot.
- `ds4_tp.c`, `ds4_tp.h` — tensor parallel fra Mac, protocollo lockstep e
  trasporto delle attivazioni via TCP o RDMA.
- `ds4_layer_pack.c`, `ds4_layer_pack.h` — placement contiguo dei layer,
  condiviso anche con i percorsi locali multi-GPU.

### Supporto applicativo

- `ds4_web.c`, `ds4_web.h` — servizio/browser helper controllato da Agent Q per
  fetch e interazione web.
- `rax.c`, `rax.h`, `rax_malloc.h` — radix tree usato per lookup e strutture
  indicizzate del server/cache.
- `dir-steering/` — strumenti e fixture per costruire e applicare direction
  vectors di steering.

### Quantizzazione e confronto con altri motori

- `gguf-tools/deepseek4-quantize.c` — conversione/quantizzazione dei tensor nei
  formati supportati dal progetto.
- `gguf-tools/quants.c`, `gguf-tools/quants.h` — implementazioni CPU e
  definizioni dei formati quantizzati.
- `gguf-tools/quality-testing/data/qwen36-27b/` and
  `data/qwen36-27b-mtp/` — Qwen3.6 manifests and fixtures following the other
  sets under `data/`; shared validators and generators stay at the
  `quality-testing/` root.
- `gguf-tools/mixed/` — strumenti per comporre GGUF con precisioni differenti
  fra gruppi di layer/tensor.
- `gguf-tools/imatrix/` — costruzione dataset e strumenti per importance
  matrix.
- `gguf-tools/quality-testing/score_official.c` — scorer DS4 teacher-forced:
  logits locali, NLL, token e bytes decodificati e confronto con
  logprob/top-k ufficiali.
- `gguf-tools/quality-testing/score_llama.cpp` — scorer oracle equivalente
  basato su llama.cpp.
- `gguf-tools/quality-testing/compare_scores.py` — confronto e aggregazione dei
  report di qualità.
- `gguf-tools/quality-testing/collect_official.py` — acquisizione controllata
  delle continuazioni e top-logprob da API esterne.
- `gguf-tools/quality-testing/inspect_qwen36_gguf.py` — ispettore offline degli
  header Qwen3.6: verifica artefatti, metadata, tokenizer/template e directory
  tensor, producendo gli snapshot condivisi in `data/qwen36-metadata/`.
- `gguf-tools/quality-testing/generate_qwen36_oracle.py` — generatore in
  staging per llama.cpp, Transformers e vLLM; conserva path locali fuori dai
  manifest e separa i logits greedy e teacher-forced con checksum.
- `gguf-tools/quality-testing/verify_qwen36_run.py` — valida inventario,
  copertura, lunghezza e determinismo fra due candidati oracle senza
  promuoverli a golden.
- `gguf-tools/quality-testing/generate_qwen36_score.py` — prepara un run DS4 o
  llama.cpp sulla sequenza canonica di token di un oracle, lasciando agli
  scorer nativi inferenza e dump float32 e registrando checksum e provenienza
  nello stesso layout del task 2.
- `gguf-tools/quality-testing/run_qwen36_context_matrix.py` — espande il profilo
  16K in run sequenziali per frontiera, chunk e ripetizione, controllando la
  VRAM libera e il gate prestazionale prima di avviare ogni processo modello.
- `gguf-tools/quality-testing/qwen36_speed_gate.py` — valida il report breve di
  prefill/decode e blocca i run long-context sotto 500/15 token/s o con
  provenienza non confrontabile.
- `gguf-tools/quality-testing/compare_qwen36_trace.py` — confronta i dump
  float32 per layer/stadio prodotti da DS4 e llama.cpp sullo stesso prefisso e
  localizza la prima divergenza senza definire nuove soglie semantiche.
- `gguf-tools/quality-testing/diagnose_qwen36_numerics.py` — triangola le trace
  DS4, llama.cpp CPU e CUDA, canonicalizza lo stato GDN e classifica le
  deviazioni rispetto all'inviluppo numerico dei backend.
- `gguf-tools/quality-testing/compare_qwen36_equivalence.py` — harness unico
  DS4-vs-llama.cpp e DS4-vs-DS4: valida inventari, calcola metriche
  full-vocabulary per posizione, applica gate versionati inclusa la suite
  corta del task 4 e produce report auditabili `PASS`, `FAIL`, `NOT_VERIFIED`
  o `ERROR`.

### Test rilevanti per comprendere la correttezza

- `tests/ds4_test.c` — runner principale di unit test e test model-backed;
  include equivalenza Qwen MTP/target-only e fixture deterministiche per
  accettazione completa, rifiuto e rollback parziale.
- `tests/test_constrained_json_api.py` — suite HTTP live e model-backed per
  masking DSML/JSON Schema: matrice senza/con history sotto 20k token, prompt
  di injection ostili, duplicati, Unicode/escape e validazione degli output.
- `tests/test-vectors/` — prompt corti/lunghi, vettori ufficiali e golden locali
  per token e distribuzione.
- `tests/test_engine_correctness.c` — confronto della continuazione greedy fra
  configurazioni CUDA.
- `tests/test_metal_session_batch.c` — oracle full-vocabulary del batching
  Metal contro sessioni indipendenti.
- `tests/test_cuda_session_batch.c` — oracle del batching CUDA.
- `tests/test_cuda_mixed_batch.c` — oracle bit-exact per batch misti
  prefill/decode CUDA.
- `tests/cuda_long_context_smoke.c` — controlli CUDA mirati alle strutture
  long-context e regressione sintetica del matmul Q6_K contro la
  dequantizzazione CPU.
- `tests/glm_long_context_smoke.sh` — integrazione GLM su prompt lunghi.
- `tests/test_server_batching.py` — concorrenza e batching osservati tramite
  API server.
- `tests/test_agentic_api.py` — suite HTTP sottile e concatenabile per API
  Responses agentiche: compatibilità standard, capability dinamiche, masking
  avversariale, distinzione tool/skill, nesting/ricorsione, restore e rifiuto
  atomico dei checkpoint mancanti, corrotti o troncati.
- `tests/test_agentic_checkpoint.c` — gate model-backed CUDA Q4_K_S per
  checkpoint/return: casi lunghi 10k, confronto full-vocabulary bit-exact,
  isolamento sessioni, cancellazione, nesting, context boundary, rollback MTP
  e ripristino del payload generale fra oggetti sessione distinti.
- `tests/run_agentic_checkpoint.sh` — runner riproducibile del gate precedente
  nelle varianti target-only e target con sidecar MTP, con report JSON e log.
- `tests/test_qwen36_fixtures.py`, `tests/test_qwen36_equivalence.py`,
  `tests/test_qwen36_numerics.py` — schema, staging, checksum, metriche/gate e
  classificazione sintetica dell'inviluppo numerico Qwen3.6.
- `tests/qwen_numerics_probe.c` — oracle CPU indipendente per GDN, full
  attention Qwen e matvec Q4_K/Q5_K/Q6_K contro i kernel CUDA DS4, eseguibile
  con `make qwen-numerics CUDA_ARCH=sm_86`.
- `tests/test_sampling.c` — comportamento deterministico e filtri del sampling.
- `tests/test_gpu_args.c`, `tests/test_gpu_args_cli.sh` — parsing e propagazione
  della configurazione GPU.
- `tests/test_layer_pack.c` — invarianti del placement contiguo dei layer.
- `tests/test_gpu_model_cache.c`, `tests/test_gpu_lookup_cache_strict.c` —
  lifetime e correttezza delle cache di pesi/tensor GPU.
- `tests/test_engine_mgpu_placement.c`,
  `tests/test_engine_mgpu_runtime.c`,
  `tests/test_engine_mgpu_refusal.c` — placement, equivalenza e rifiuto sicuro
  delle configurazioni multi-GPU.
- `tests/ds4_agent_test.c` — parser, stato e operazioni isolate di Agent Q.
- `tests/dspark_acceptance_fixture.sh` — correttezza e acceptance del percorso
  speculativo DSpark.

### Verifica del percorso Qwen task 6

I gate locali minimi sono `make cpu`, `make cuda CUDA_ARCH=sm_86`,
`make ds4_test` e `./ds4_test --qwen35-layer-pattern`. Il confronto numerico
sul GGUF reale parte da `generate_qwen36_score.py` con lo scorer
`gguf-tools/quality-testing/score_official`, quindi usa
`compare_qwen36_equivalence.py` prima con `--suite short` e poi senza filtro.
`verify_qwen36_run.py` controlla separatamente inventario, 32 passi greedy,
32 posizioni teacher-forced e determinismo dei nove oracle llama.cpp. Un
comando non eseguibile per toolchain, backend, memoria o modello mancante va
registrato `NOT VERIFIED`; i test distribuiti richiedono autorizzazione
separata.

## File intenzionalmente non elencati uno per uno

Non fanno parte della mappa principale gli output di build, i dati voluminosi
delle fixture, i singoli prompt/continuation, i CSV e grafici di benchmark e le
risorse puramente grafiche. Vanno consultati attraverso il modulo o README che
li possiede quando il capitolo corrispondente lo richiede.
