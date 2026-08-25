# Roadmap di integrazione Qwen3.8 27B UD-Q4_K_S

## Obiettivo e criterio di completamento

Portare `Qwen3.8-27B-UD-Q4_K_S.gguf` allo stesso livello funzionale del target
`Qwen3.6-27B-Q4_K_S.gguf` su CUDA: CLI, server, MTP, sessioni, checkpoint SSD,
cache durevole, diagnostica e test analoghi. Il lavoro e' completo soltanto
quando i gate numerici e model-backed del 3.8 passano senza regressioni sul
3.6. Un test non eseguito resta `NOT_VERIFIED`, mai `PASS`.

Artefatti auditati locali:

- target: `gguf/Qwen3.8-27B-UD-Q4_K_S.gguf`, 15.358.213.024 byte,
  SHA-256 `75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017`;
- sidecar: `gguf/mtp-Qwen3.8-27B-Q4_0.gguf`, 1.369.590.656 byte,
  SHA-256 `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e`;
- hardware di accettazione: RTX 3090 `sm_86`, un solo processo modello
  voluminoso alla volta.

## Stato sintetico

| Area | Stato | Prossima prova decisiva |
| --- | --- | --- |
| Audit GGUF, downloader e model ID | `DONE` | 15/15 test live e checksum completi |
| Loader e validazione Qwen3.8 | `DONE` | target/sidecar corretti e mismatch 3.6/3.8 rifiutato |
| Kernel CUDA dei formati UD | `DONE` | build pulita e oracle CPU/GPU/model-backed verdi |
| Decode/prefill target | `DONE` | 548,98/24,13 tok/s a 512; 627,06/23,34 a 4.096 |
| MTP target + sidecar | `DONE`; regressione corta corretta, limite 28K pieno aperto | CLI 47,71 e server 29,59 tok/s a `--ctx 28768`; 28K pieno 18,43 |
| CLI e server live | `DONE` | one-shot, REPL, perplexity e matrice API breve |
| ChatML, tool calling e masking DSML | `DONE` | Qwen3.8 e Qwen3.6: 20/20 live; confronto oracle senza divergenze osservate |
| Curve contesto server | `PARTIAL / PERFORMANCE_GAP` | 2K/16K/28K misurati target/MTP; MTP 28K sotto floor 20 tok/s |
| Checkpoint/cache SSD | `DONE` | target/MTP, corruzione e cambi direzionali verdi |
| Regressione Qwen3.6 | `DONE` | suite aggregata verde; cinque casi logits bit-exact |
| Ottimizzazioni comuni | `DEFERRED` | nessuna variante comune promossa senza doppio gate |

## Fase 0 — Baseline, fonti e invarianti

- [x] Leggere `AGENT.md`, `PROJECT_INDEX.md`, il contratto prestazionale, la
  knowledge base e la scheda RTX 3090.
- [x] Confermare che il worktree contiene modifiche dell'utente da preservare;
  evitare riscritture o rollback di file non attribuibili a questo task.
- [x] Confermare la presenza locale della coppia target/MTP Qwen3.8.
- [x] Ispezionare `docs/research/forks/syv-ai-qwen38-27b-rtx3090.md`.
- [x] Fissare l'upstream `syv-ai/qwen38-27b-rtx3090` ispezionato a
  `00210159df4366704b98b178258b3f618005611a`.
- [x] Decisione: il fork `syv-ai` e' una fonte per speculative decoding,
  split-KV, sampler e gestione dello stato, ma non per il packing UD-GGUF. Il
  fork serve un checkpoint W4A16 AutoRound tramite Marlin/vLLM; i kernel dei
  sette formati GGML devono seguire le strutture e gli oracle GGML/llama.cpp.
- [x] Salvare la regressione Qwen3.6 prima di promuovere ottimizzazioni comuni:
  build, numerica, cinque casi tensor-equivalence bit-exact, CLI e MTP. Nessuna
  ottimizzazione comune e' stata promossa.

## Fase 1 — Kernel indispensabili per il target UD

Formati mancanti, rilevati dall'inventario pinned:

- `Q3_K`;
- `IQ2_XS` e `IQ2_S`;
- `IQ3_XXS` e `IQ3_S`;
- `IQ4_NL` e `IQ4_XS`.

Checklist:

- [x] Mappare per ogni formato block size, byte layout, scale, griglia e tabella
  dei segni usando una revisione fissata di llama.cpp/ggml.
- [x] Inventariare le shape reali del target per distinguere embedding/output,
  decode matvec e prefill GEMM.
- [x] Aggiungere strutture CUDA con `static_assert` sulla dimensione GGUF.
- [x] Implementare prima un percorso corretto e leggibile per embedding,
  matvec decode e dequantizzazione prefill.
- [x] Usare primitive dedicate/ottimizzate gia' verificate quando il loro
  layout coincide; evitare conversioni o requantizzazioni lossy del target.
- [x] Aggiungere oracle CPU per blocco e matrice a `qwen-numerics`, inclusi
  valori limite, segni, scale e piu' blocchi.
- [x] Confrontare l'output CUDA con dequantizzazione CPU indipendente prima del
  primo run del modello.
- [x] Rimuovere il rifiuto anticipato Qwen3.8 solo quando tutti i formati
  incontrati dal target hanno un percorso CUDA verificato.

Gate della fase:

- nessun NaN/Inf;
- errore della primitiva entro l'inviluppo definito dagli oracle Qwen;
- nessun formato o shape accettato senza implementazione;
- build CPU e CUDA verdi;
- Qwen3.6 invariato sui test di primitiva esistenti.

## Fase 2 — Target Qwen3.8 end-to-end

- [x] Caricamento CUDA completo senza fallback semantico o copia fuori budget.
- [x] Inspect e primo token per CLI.
- [x] Prefill corto, decode corto e prompt multi-token.
- [x] 32 token greedy e 32 posizioni teacher-forced contro oracle Qwen3.8
  indipendente alla revisione llama.cpp fissata; 32/32 greedy identici.
- [x] Confronto full-vocabulary, NLL/logprob, top-1, top-20 e cosine senza usare
  golden Qwen3.6. Diagnostica cross-engine su un caso: cosine minima
  0,999963797, overlap greedy minimo 0,95 e logprob-MAE teacher 0,0250284.
  La promozione dell'intero corpus oracle resta separatamente non verificata.
- [x] Prompt lunghi e frontiere canoniche, inclusi full-attention e stato GDN:
  prefill+decode a 4.096 token e allocazioni live a contesto 30.000.
- [x] Gate prestazionale minimo: almeno 500 tok/s prefill e 15 tok/s decode
  nella configurazione dichiarata, con VRAM registrata.

## Fase 3 — Parita' funzionale con Qwen3.6

### CLI

- [x] Auto-selezione del sidecar Qwen3.8 con `--mtp`.
- [x] Override `--mtp-model`, inspect, one-shot greedy, REPL, sampling, thinking,
  perplexity e dump diagnostici. Sessioni/checkpoint e cache su disco sono
  verificati dai gate engine/agentic dedicati, non da un comando REPL inesistente.

### Server

- [x] Avvio target-only e MTP, model ID stabile e directory KV separata.
- [x] API OpenAI Chat/Responses, Anthropic e streaming SSE; tool calling,
  constrained output e batching/multi-sessione nei percorsi applicabili.
- [x] Renderer Qwen ChatML auditato per system/user/assistant, thinking,
  cronologia tool raggruppata e continuation live; DSML resta il wire format
  constrained comune e non viene sostituito con il formato nativo Qwen.
- [x] Allowlist agentic dinamica nel turno user piu' recente e reminder mirato
  per il singolo tool zero-argomenti richiesto, senza contaminare il system
  prefix durevole o il percorso multi-call.
- [x] Matrice constrained live 20/20 su Qwen3.8 e 20/20 su Qwen3.6; modalita'
  `compare_new_vs_oracle` su Qwen3.8 senza divergenze osservate.
- [x] Riavvio cold/warm, riuso prefisso, corruzione/troncamento cache e
  rigenerazione sicura.

### MTP

- [x] Proposta/verifica, accettazione completa, rifiuto e rollback parziale.
- [x] Allineamento hidden/token, KV e GDN dopo commit/rollback.
- [x] Equivalenza target-only/MTP secondo il contratto numerico e acceptance
  registrata a contesti brevi e lunghi.
- [x] Regressione prompt corto con allocazione `--ctx 28768`: kernel verifier
  UD Q8_1+R8 dedicato a 2/3 righe, CLI 47,71 tok/s e server 29,59 tok/s medio
  (campioni 29,40/29,78), due output deterministici da 64 token.
- [ ] Portare la frontiera server realmente riempita a 28K sopra il floor di
  20 tok/s. La misura corrente e' 18,43 MTP contro 16,21 target-only; MTP
  accelera il target ma non soddisfa ancora il contratto assoluto.
- [x] Qwen3.8 verificato con allocazione contesto 30.000 nella matrice Agent SSD
  e con prefill reale a 4.096/allocazione 8.192 (16,15 GiB pianificati). Questo
  e' il profilo accettato, non un'affermazione sul massimo teorico 262K.

### SSD e agentic

- [x] Snapshot portabile fra sessioni target-only e fra sessioni MTP.
- [x] Rifiuto atomico del cambio modalita' target-only/MTP incompatibile.
- [x] Checkpoint GDN di skill, nesting, return/restore, cleanup e isolamento.
- [x] Matrice live `run_agent_ssd_live.ps1` cold/warm nelle due direzioni e
  con MTP, usando il model ID Qwen3.8.

## Fase 4 — Regressione Qwen3.6

- [x] Build CPU/CUDA e suite unitarie pertinenti. La suite aggregata e' verde.
  Le fixture predefinite dichiaratamente DeepSeek e Metal Flash sono saltate
  soltanto quando l'engine caricato e' Qwen; gli override espliciti continuano
  a eseguirle contro il modello corretto.
- [x] `qwen-numerics` per Q4_K/Q5_K/Q6_K e GDN/attention.
- [x] Full-vocabulary e determinismo: cinque casi Qwen3.6 reference/candidate
  bit-exact, top-1 e top-20 identici, RMS/max-abs zero.
- [x] CLI target-only/MTP e test MTP depth/reject/partial/sampling/disabled.
- [x] Checkpoint SSD, cache durevole e agentic coperti dai test condivisi; il
  nuovo modello passa inoltre l'intera matrice live target/MTP.
- [x] Baseline prestazionale sopra 500/15 e long-context 31.181 verde.
- [x] Nessuna modifica distribuita eseguita.

## Fase 5 — Ottimizzazioni opzionali comuni

Valutare soltanto dopo la parita' funzionale del 3.8:

- split-KV verify attention e sampler sort-free dal fork `syv-ai`, adattati
  alla semantica DS4;
- compressione del solo drafter o vocab draft ridotto, se compatibili con il
  sidecar GGUF e con rejection sampling target-preserving;
- prefix caching di KV + stato GDN;
- miglioramenti ai matvec comuni Q4_K/Q5_K/Q6_K o allo scheduling GDN.

Per ogni candidato:

1. una sola variabile e ipotesi falsificabile;
2. test della primitiva, poi gate numerico Qwen3.6 e Qwen3.8;
3. benchmark direction solo per scartare; `KEEP` esclusivamente dopo suite
   slow, cinque ripetizioni e correttezza completa;
4. `REJECT` immediato per argmax differente, top-20 overlap < 0,95, cosine
   < 0,999, valori non finiti, drift non spiegato o errore MTP/rollback;
5. registrare anche gli esperimenti scartati; non lasciare varianti semantiche
   permanenti dietro flag.

## Registro breve di decisioni e risultati

| Data | ID | Decisione/risultato | Stato |
| --- | --- | --- | --- |
| 2026-08-24 | Q38-000 | Preservare il worktree dirty e costruire sopra l'audit/test Qwen3.8 gia' presente. | `KEEP` |
| 2026-08-24 | Q38-001 | I sette formati UD sono il blocker reale; il rifiuto anticipato corrente evita output corrotti. | `KEEP` |
| 2026-08-24 | Q38-002 | `syv-ai` non fornisce kernel GGML trasferibili: usa W4A16 Marlin. Usarlo soltanto per idee runtime dopo la correttezza base. | `KEEP` |
| 2026-08-24 | Q38-003 | Vietato confrontare logits/golden 3.8 con il 3.6; input condivisibili, oracle separati. | `KEEP` |
| 2026-08-24 | Q38-004 | Layout, griglie e segni dei sette formati sono fissati alla revisione llama.cpp `2468576f241235452013308597e6de1b78866996`; strutture CUDA protette da `static_assert`. | `KEEP` |
| 2026-08-24 | Q38-005 | Dequantizzazione indipendente CPU, decode F32, prefill dequant+GEMM e MMVQ Q8_1/R8 passano `qwen_numerics_probe`; il primo forward reale Qwen3.8 e' coerente. | `KEEP` |
| 2026-08-24 | Q38-006 | Variante decode F32 a 8 warp: corretta ma senza vantaggio misurabile (8,29 contro 8,54 tok/s); rimossa. | `REJECT` |
| 2026-08-24 | Q38-007 | MMVQ con una sola quantizzazione Q8_1 raggiunge 27,83 tok/s ma diverge nel greedy reale dopo pochi token; non e' il percorso di produzione. | `REJECT` |
| 2026-08-24 | Q38-008 | MMVQ residuale Q8_1+R8 e' il default decode: 29,72 tok/s e 16/32 token visibili identici al riferimento F32. Sul confronto full-logit interno 32+32: top-1 64/64; cosine minima greedy 0,9999990314 e teacher 0,9999999916; top-20 overlap minimo 0,95/1,00. | `KEEP` |
| 2026-08-24 | Q38-009 | Prefill F16 per i sette formati UD promosso solo da 128 righe: su 512 righe il profilo caldo passa da circa 451 a 560 tok/s. Contro F32 su prompt 512 + 32 greedy + 32 teacher: top-1 64/64, top-20 minimo 1,00, cosine minima 0,9999963063; sotto 128 righe resta F32 e `DS4_CUDA_QWEN38_NO_PREFILL_F16` effettua rollback. | `KEEP` |
| 2026-08-24 | Q38-010 | Baseline iniziale CLI MTP target-preserving: output greedy 32 token uguale al target-only; acceptance 17/28 con casi full, partial e reject/rollback, zero fallback. Il verifier scalare iniziale misurava 5,96 tok/s contro 28,42 target-only; risultato storico, superato da Q38-021. | `SUPERSEDED` |
| 2026-08-24 | Q38-011 | Server live target/MTP passa Chat Completions, Responses, Anthropic Messages e SSE; model ID Qwen3.8 stabile. Il passaggio target-only/MTP rifiuta e scarta il payload persistente incompatibile prima di rigenerarlo. | `KEEP` |
| 2026-08-24 | Q38-012 | `test_agentic_checkpoint` fast passa target-only e MTP: logits 248.320 bit-exact, tre nesting, corruzione/troncamento/cancel, isolamento, sessione portabile e frontiera 511 bit-exact. | `KEEP` |
| 2026-08-24 | Q38-013 | Primo `warm/hds` live ha restituito il risultato intermedio del child invece del canary root: checkpoint integro, istruzione ambigua con call ID casuali. Rafforzato il prompt senza allentare l'assert; ripetizione cold/warm target-only e MTP verde. | `FIX` |
| 2026-08-24 | Q38-014 | Matrice Agent SSD live a ctx 30.000 verde per system/HDS/skill cold+warm target-only e MTP, incluse entrambe le transizioni target-only<->MTP con refresh obbligatorio. | `KEEP` |
| 2026-08-24 | Q38-015 | Costruito `llama.cpp` CUDA `sm_86` alla revisione fissata e adattato lo scorer alla rimozione upstream del campo `use_mmap` (il default resta mmap). Sul medesimo token manifest, DS4 R8 e llama hanno 32/32 greedy identici; full-logit cosine minima 0,999963797, overlap greedy minimo 0,95 e logprob-MAE teacher 0,0250284. Un argmax teacher a basso margine differisce; la prova resta diagnostica e non promuove da sola il corpus completo. | `KEEP` |
| 2026-08-24 | Q38-016 | Rebuild finale CPU e CUDA completa verde; `qwen-numerics` passa tutti i formati vecchi e nuovi. `make test-qwen38` con live opt-in passa 15/15; il checksum opt-in dei 16,7 GB e' stato eseguito separatamente ed e' verde. | `KEEP` |
| 2026-08-24 | Q38-017 | Il primo run aggregato ha mostrato 20 assert in due gruppi di fixture storiche (`logprob-vectors`, `local-golden-vectors`). L'ispezione ha confermato che dichiarano rispettivamente `deepseek-v4-flash` e Metal Flash, mentre il modello predefinito locale e' Qwen; il gate tensor-equivalence subito dopo era comunque bit-exact su cinque casi. | `DISCOVERED` |
| 2026-08-24 | Q38-018 | `mtp-verify-depth` dedicato passa sia 3.8 sia 3.6: 231 token, chunk massimo 3, gap argmax 0, reject/partial/raw-copy/sampling e percorso MTP-disabled bit-exact su 248.320 logits. | `KEEP` |
| 2026-08-24 | Q38-019 | `ds4-bench` finale: warm 512 = 548,98/24,13 tok/s; 4.096 = 627,06/23,34 tok/s. REPL produce `OK.` e termina pulito; perplexity CLI completa 3.461 token senza non-finiti. | `KEEP` |
| 2026-08-24 | Q38-020 | Il runner ora salta le sole fixture predefinite DeepSeek/Metal Flash quando l'engine e' Qwen, dopo averne aperto e identificato il modello; un path esplicito non viene mai saltato. Il rerun selettivo e `make test CUDA_ARCH=sm_86` completo terminano verdi. | `FIX` |
| 2026-08-24 | Q38-021 | Causa del caso segnalato: il verifier MTP usa 3 righe sotto 2K e 2 righe da 2K; il dispatch UD ottimizzava soltanto una riga e lasciava 2/3 al matvec scalare. Aggiunto MMVQ Q8_1+R8 multi-riga per tutti i sette formati UD, con riuso di due righe peso per CTA e rollback `DS4_CUDA_QWEN38_NO_VERIFY_Q8_1_R8`. Il tempo target-verify scende da circa 368-392 ms a 55 ms. | `FIX` |
| 2026-08-24 | Q38-022 | Caso esatto `ciao`, MTP, `--ctx 28768`: 47,71 tok/s CLI contro circa 7 prima. Server con la stessa capacita' KV: 29,59 tok/s medio (29,40/29,78), CV 0,91%, due risposte deterministiche da 64 token; gate 20 tok/s `PASS`. | `KEEP` |
| 2026-08-24 | Q38-023 | Curve server con prompt realmente riempito: a 2K MTP 27,74 tok/s; a 16K target 19,49 e MTP 22,01; a 28K target 16,21 e MTP 18,43. MTP resta vantaggioso, ma 28K non raggiunge il floor assoluto 20: completamento prestazionale long-context non dichiarato. | `PERFORMANCE_GAP` |
| 2026-08-24 | Q38-024 | Il canary storico a lista numerata terminava dopo 1-3 token su Qwen3.8. Aggiunti prompt tecnico deterministico, `--no-thinking`, suite `mtp-short-regression` e `--context-alloc` indipendente dal prompt; il nuovo test model-backed copre CLI e server a 28.768 allocati. | `FIX` |
| 2026-08-24 | Q38-025 | Scartate per misure peggiori o non significative: prefill corto tile4/tile8 e F16, verifier a una riga per CTA (43,72 contro 48,91 tok/s), GQA3 a 28K (84,58 contro 65,18 ms/token) e depth2 forzata a 28K (16,37 contro 18,30-18,55 tok/s). Nessuna variante resta nel default. | `REJECT` |
| 2026-08-24 | Q38-026 | Dopo il kernel multi-riga, `qwen-numerics` verifica 2/3 righe contro oracle indipendente per tutti i sette formati. `mtp-verify-depth` passa sia 3.8 sia 3.6: 231 token, reject/partial/raw-copy, sampling 128 target-match e percorso disabled bit-exact su 248.320 logits. | `KEEP` |
| 2026-08-25 | Q38-027 | Causa del tool calling 3.8: server e agent usavano marker ruolo DeepSeek o testo DSML non racchiuso in un ruolo Qwen. Introdotto un percorso Qwen ChatML completo per prompt, history, tool response, recovery e checkpoint, mantenendo DSML come superficie constrained condivisa. | `FIX` |
| 2026-08-25 | Q38-028 | `ds4-agent` Qwen3.8 model-backed ha emesso ed eseguito esattamente una chiamata DSML `read` su `PROJECT_INDEX.md`; la continuation usa `<tool_response>` ChatML e la risposta finale e' corretta. | `PASS` |
| 2026-08-25 | Q38-029 | Suite constrained Qwen3.8 20/20 in `compare_new_vs_oracle`, senza divergenze osservate; suite finale Qwen3.6 20/20. La regressione 3.6 zero-argomenti e' stata corretta con un reminder ristretto al solo schema vuoto, preservando il multi-call. | `PASS` |
| 2026-08-25 | Q38-030 | Harness trie, 1 warm-up + 5 run: DSML mask 3.8 355,826 ms contro 357,851 ms 3.6 (-0,57%); constrained CPU 1.004,302 contro 999,680 ms (+0,46%, parita' pratica). L'end-to-end 3.8 resta piu' lento: 19,08 contro 22,10 tok/s, spiegato soprattutto da eval UD 2.144,873 contro 1.757,448 ms (+22,0%), non dal masking. | `MEASURED` |
| 2026-08-25 | Q38-031 | Il cache system-prompt dell'agent interrogava il bit-width degli expert routed; Qwen3.8 e' dense e restituiva 0, pur avendo payload persistente Qwen supportato. Allineata l'identita' a Q4 come nel KV store server; save e warm hit model-backed verificati. | `FIX` |

## Comandi e risultati da compilare durante il lavoro

| Gate | Comando/artefatto | Risultato |
| --- | --- | --- |
| Metadata/fixture 3.8 | `DS4_TEST_QWEN38_LIVE=1 make test-qwen38` + checksum opt-in | `PASS`: 15/15; checksum target+MTP `PASS` |
| Build CPU | `make cpu` | `PASS` |
| Build CUDA | `make -B cuda CUDA_ARCH=sm_86` | `PASS`: CLI/server/bench/eval/agent |
| Primitive Qwen | `make qwen-numerics CUDA_ARCH=sm_86` | `PASS`: sette formati, F32/Q8_1/R8, verifier 2/3 righe, F16 prefill e percorsi 3.6 |
| Unit/regression | `make test CUDA_ARCH=sm_86` | `PASS`: suite aggregata verde; fixture cross-model predefinite saltate per identita', override espliciti invariati |
| Quality 3.8 breve | 32 greedy + 32 teacher, DS4 F32/R8/F16 e llama.cpp esterno | `PASS` interno; cross-engine diagnostico entro inviluppo, corpus completo `NOT_VERIFIED` |
| Performance 3.8 | `ds4-bench`, stessa sessione, ripetizione calda | `PASS`: 548,98/24,13 a 512; 627,06/23,34 a 4.096 |
| Regressione MTP corta 28.768 allocati | `make test-qwen38-perf CUDA_ARCH=sm_86` | `PASS`: CLI 47,71; server 29,59 medio, minimo 29,40, output deterministico |
| Server contesto riempito | `mtp-depth-2k` + `mtp-long-context-smoke`, target/MTP | `PARTIAL`: 2K 27,74 MTP; 16K 22,01 MTP; 28K 18,43 MTP sotto floor 20 |
| Target/MTP checkpoint | `DS4_AGENTIC_FAST=1 tests/test_agentic_checkpoint` con target 3.8 | `PASS` target-only e MTP |
| Agent SSD live | `tests/run_agent_ssd_live.ps1`, ctx 30.000 | `PASS` system/HDS/skill target-only+MTP e cambi bidirezionali |
| MTP depth 3.8/3.6 | `./ds4_test --mtp-verify-depth` con target/sidecar espliciti | `PASS` entrambi; reject/partial/sampling/disabled inclusi |
| CLI/server live | matrice breve target-only/MTP | `PASS` one-shot/REPL/perplexity; Chat/Responses/Anthropic/SSE server |
| Tool calling Qwen3.8 | `tests.test_constrained_json_api` + `ds4-agent` one-shot `read` | `PASS`: matrice 20/20, DSML reale eseguito e continuation ChatML corretta |
| Regressione tool calling Qwen3.6 | `tests.test_constrained_json_api` | `PASS`: 20/20 in 344,148 s |
| Masking DSML Qwen3.8 | `compare_new_vs_oracle` + `qwen38-chatml-dsml-trie-20260825-002` | `PASS`: nessuna divergenza osservata; mask mediana 355,826 ms, output deterministico |
| Parita' prestazionale Qwen3.6 | `qwen36-chatml-dsml-trie-20260825` | `PASS` masking: 357,851 ms; end-to-end 3.8 `PERFORMANCE_GAP` per eval UD, non per constraint |
| Wiki/docs | `python3 -m unittest tests.test_docs_wiki -v` | `PASS` 4/4 dopo rigenerazione indici |
