# Indice: guides

Indice generato; non modificare a mano.

## [Politica dell'evidenza](evidence-policy.md)

Paper, documentazione NVIDIA, repository che implementa direttamente il kernel, commit e pull request con test.

- righe 3–6: **Tier A — fonte primaria** — Paper, documentazione NVIDIA, repository che implementa direttamente il kernel, commit e pull request con test.
- righe 7–10: **Tier B — implementazione o benchmark riproducibile** — Fork attivo, bench card con commit/modello/quantizzazione/hardware/warm-up/ripetizioni/CV e correctness.
- righe 11–14: **Tier C — pista debole** — Issue, discussione, README promozionale, tabella senza script, screenshot, post o benchmark senza commit.
- righe 15–18: **Fuffa** — Numeri privi di modello esatto, context, batch, token misurati, warm-up, quantizzazione e controllo numerico; claim “X volte più veloce” contro una baseline non identificata; repository che rinomina upstream senza diff t…
- righe 19–22: **Regola di triangolazione** — Prima di implementare: una fonte semantica primaria, una implementazione concreta e una misura locale.
- righe 23–25: **Regola di citazione** — Ogni scheda conserva un solo link di ingresso al repository.

## [Mappa problema → soluzione esistente](problem-to-source.md)

Leggere Marlin, QUICK e Atom; poi confrontare llama.cpp MMVQ/vec-dot, ik_llama MMVQ e ExLlamaV3 GEMM.

- righe 7–10: **Decode Q4_K memory-bound** — Leggere Marlin, QUICK e Atom; poi confrontare llama.cpp MMVQ/vec-dot, ik_llama MMVQ e ExLlamaV3 GEMM.
- righe 11–14: **Prefill Q4_K compute-bound** — Confrontare llama.cpp MMQ, kernel Marlin/QServe e schedule MLC.
- righe 15–18: **Full-attention decode a context lungo** — Leggere FlashDecoding++, FlashInfer e SparQ.
- righe 19–22: **Gated Delta Net** — Partire dal paper GDN e dall'implementazione autori; confrontare llama.cpp e ik_llama `gated_delta_net.cu`.
- righe 23–26: **KV cache e 24 GiB** — Leggere PagedAttention, KIVI e KVQuant.
- righe 27–30: **Launch overhead e CUDA Graph** — Confrontare vLLM, TensorRT-LLM e SGLang.
- righe 31–34: **Speculative e MTP** — Leggere i due paper speculative, Medusa e multi-token prediction.
- righe 35–37: **Offload e memoria insufficiente** — Leggere FlexGen e le ricette consumer, ma non confondere throughput batch con latenza batch 1.

## [Qwen3.6 MTP: workflow llama.cpp, confronto DS4 e audit CUDA](qwen36-mtp-llamacpp-ds4-control-flow.md)

Questa scheda descrive il percorso MTP realmente eseguito per Qwen3.6, dal

- righe 3–14: **Ruolo e perimetro** — Questa scheda descrive il percorso MTP realmente eseguito per Qwen3.6, dal
- righe 15–36: **Fonti primarie** — integrazione MTP, context target/draft separati, hidden NextN e rollback
- righe 37–54: **Semantica NextN** — Il trunk target espone `h_nextn` dopo `output_norm`.
- righe 55–112: **Workflow llama.cpp** — Il context target conserva logits, KV e memoria ricorrente.
  - righe 57–63: **Prefill e frontiera** — Il context target conserva logits, KV e memoria ricorrente.
  - righe 64–69: **Draft** — Il token appena campionato non e' ancora nel target state.
  - righe 70–82: **Verifica target** — Il server costruisce un unico batch:
  - righe 83–91: **Accept e rollback** — `process()` conserva tutti gli hidden target in `verify_h`.
  - righe 92–112: **Pseudocodice** — sampled = target_sampler(target_logits)
- righe 113–131: **DS4 prima dell'audit** — Il vecchio ciclo eseguiva `target_seed(1)` e poi `verify(drafts)`.
- righe 132–173: **DS4 finale** — Con `--mtp`, il percorso predefinito e':
  - righe 156–173: **Sampling, masking e doppio controllo** — La soluzione segue `common_sampler_sample_and_accept_n` del llama.cpp locale di
- righe 174–192: **Problemi trovati e ritorno a llama.cpp** — Il bug lane 1 e' il caso piu' importante.
- righe 193–228: **Audit dei kernel MTP** — Il verifier usa attivazioni Q8_1 piu' un secondo residuo Q8_1.
  - righe 195–214: **Target Q4_K/Q5_K/Q6_K** — Il verifier usa attivazioni Q8_1 piu' un secondo residuo Q8_1.
  - righe 215–221: **Sidecar Q4_0** — Il shared head usa Q4_0 x Q8_1+R8 con DP4A e otto output rows per CTA.
  - righe 222–228: **Rollback GDN** — Le 48 convolution state e recurrent state vengono snapshotate direttamente
- righe 229–239: **Esperimenti scartati** — qualita' fallita; non presente nel percorso finale.
- righe 240–380: **Validazione finale** — Comando:
  - righe 242–268: **Suite model-backed** — Comando:
  - righe 269–291: **CLI Q4_K_S** — Context 768, prefill chunk 16, 128 token, greedy:
  - righe 292–316: **Confronto locale llama.cpp** — Il binario CUDA upstream al commit auditato e' stato eseguito sulla stessa RTX
  - righe 317–359: **Server reale** — Quattro richieste `/v1/chat/completions`, una warm-up e tre misurate, 180 token
  - righe 360–380: **Correzione VRAM 24 GiB** — Il vecchio report `15.02 GiB planned` era incompleto: trattava il KV Qwen come
- righe 381–388: **Nota vLLM** — L'assunzione che vLLM non implementi Qwen MTP non e' piu' valida.
- righe 389–396: **Limiti dell'evidenza** — Il workflow e l'audit CUDA llama.cpp sono fissati al commit indicato.

## [Checklist di trasferimento RTX 3090](rtx3090-transfer-checklist.md)

Confermare percorso sm_86, niente FP8 nativo, shared memory opt-in entro 99 KiB, registri e massimo 1.536 thread/SM.

- righe 3–6: **Compatibilità hardware** — Confermare percorso sm_86, niente FP8 nativo, shared memory opt-in entro 99 KiB, registri e massimo 1.536 thread/SM.
- righe 7–10: **Compatibilità dei dati** — Verificare shape Qwen3.6, Q4_K_S/Q5_K/Q6_K, group size, scale, layout GGUF e dtype delle attivazioni.
- righe 11–14: **Compatibilità numerica** — Definire ordine di accumulo, fast math, conversione Q8, tolleranza e oracle.
- righe 15–18: **Modello prestazionale** — Calcolare FLOP, minimum bytes, arithmetic intensity e limite roofline; prevedere anche registri, shared memory e numero di CTA.
- righe 19–22: **Esperimento isolato** — Una sola tecnica, 2–3 shape reali, warm-up residente, campioni grezzi e baseline nello stesso processo quando possibile.
- righe 23–25: **Promozione** — Prima PASS numerico, poi microbenchmark, suite direction e infine slow.
