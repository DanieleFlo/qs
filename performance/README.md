# DS4 performance harness

Il punto d'ingresso unico è tools/perf_harness.py. Riusa ds4-bench e gli scorer
Qwen esistenti: non introduce un secondo percorso di inferenza.

## Constrained decoding DSML e JSON Schema

Le suite riproducibili sono in `performance/constrained-workloads.json` e si
eseguono con il sottocomando `constrained-server`. `oracle_only` congela la
baseline esaustiva; `compare_new_vs_oracle` costruisce la mask candidata e
quella oracle nella stessa richiesta e rende il run non promuovibile se trova
divergenze.
L'importer `tools/import_jsonschemabench_subset.py` genera il subset esterno
pinned descritto da `performance/jsonschemabench-subset.json`, senza riscrivere
gli schemi non supportati. Senza `--source` esegue uno sparse-fetch temporaneo
del solo commit fissato; il corpus risultante esamina tutte le 9.558 schema e
conserva tier `smoke` (12) e `safety` (32).
Il tier `regressions` conserva inoltre i due casi che hanno bloccato il gate
prestazionale di MACRO 5, anche quando la selezione greedy `safety` cambia.

    python3 tools/import_jsonschemabench_subset.py \
      --output performance/jsonschemabench-subset.json

Il gate esterno usa `jsonschema` come validatore indipendente, rifiuta chiavi
JSON duplicate e verifica che ogni schema selezionato abbia un witness valido.
La dipendenza è isolata dal core dell'harness:

    python3 -m pip install -r performance/jsonschemabench-requirements.txt

Il percorso rapido confronta candidate e oracle per otto token su tutte le 32
schema, due volte. Un output terminato viene validato integralmente; un output
che raggiunge il budget è registrato esplicitamente come `PREFIX_ONLY` e passa
solo se ha consumato esattamente il budget e tutte le mask coincidono. Non va
presentato come prova di completamento dell'intero output. Nello stesso gate i
16 esempi non supportati devono ricevere HTTP 400 senza che il log mostri
l'avvio dell'inferenza: una risposta 2xx o un'inferenza iniziata falliscono il
run.

    make test-jsonschemabench-safety \
      JSONSCHEMABENCH_MODEL=gguf/Qwen3.6-27B-Q4_K_S.gguf

Il profilo end-to-end del free-text DSML reale usa
`tools/profile_agent_dsml_story.py`: costruisce il grafo `bootstrap-wiki`
installato in `/agent`, richiede la storia nel tool `exit-with-info-tool`,
verifica almeno 10k token di contesto e salva anche i fallimenti di schema.
È un profilo funzionale costoso, non una fixture da includere nei test ordinari.
Le quote di fase provengono dal log server con
`DS4_SERVER_PHASE_PROFILE=1`; non vanno sostituite col solo tempo HTTP.
La baseline non vincolata equivalente usa un prompt calibrato a 10.770 token e
147 token di decode, la lunghezza del gate agente valido:

    python3 tools/perf_harness.py server-curve \
      --id constraint-m56-unconstrained-c10770-g147-001 \
      --suite agent-dsml-unconstrained-baseline \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf --repetitions 1 \
      --minimum-completion-tokens 147 --baseline-run \
      --hypothesis "same-context unconstrained ceiling for the DSML agent gate"

Il prototipo DSML state-per-edge è stato ritirato dopo un gate slow neutro entro
il 2%; il percorso predefinito conserva il backend precedente. La phase metric
`exhaustive_fallback_steps` misura invece il fallback adattivo promosso. La suite
constrained `dsml-json-fallback` include un tool con parametri JSON tipizzati:
in `DSML_TRACK_JSON_PARAM` il backend sceglie l'analisi esaustiva perché token
BPE che uniscono fine JSON e tag di chiusura rendono il linguaggio dei prefix
del trie non chiuso.

## Release gate segmentato

La suite GPU completa supera normalmente 10 minuti anche sul commit baseline.
Per ottenere un esito conclusivo senza un timeout ambiguo, eseguire sullo stesso
modello e sulla stessa build i tre segmenti seguenti e sommare i tempi:

    ./ds4_test --long-context
    ./ds4_test --tool-call-quality --think-tool-recovery
    ./ds4_test --logprob-vectors --metal-ssd-streaming-cache-pressure \
      --local-golden-vectors --metal-short-prefill --metal-kernels \
      --metal-tensor-equivalence --streaming-decode-prefill-correctness \
      --mtp-verify-depth --dspark-verify-depth --qwen35-layer-pattern \
      --constraint-trie --server

Impostare `DS4_TEST_MODEL` allo stesso path in tutti i segmenti. Se quel modello
non corrisponde ai fixture dei golden vector, confrontare esplicitamente anche
numero e posizione dei fallimenti col commit baseline; non interpretare il solo
exit code come regressione del constrained decoder.

## Workflow eseguibili

Le sequenze ripetute non vanno ricostruite dalla shell history. Sono mantenute
in script eseguibili e richiamate dal sottocomando `workflow`:

    python3 tools/perf_harness.py workflow --name validate

    python3 tools/perf_harness.py workflow \
      --name long-context-profile --id splitk-default-01

    python3 tools/perf_harness.py workflow \
      --name long-context-direction --id splitk-default-01

    python3 tools/perf_harness.py workflow \
      --name long-context-slow --id splitk-default-confirm-01

    python3 tools/perf_harness.py workflow --name r8-build

    python3 tools/perf_harness.py workflow \
      --name r8-direction --id r8-default-check-01

    python3 tools/perf_harness.py workflow \
      --name r8-slow --id r8-confirm-01

    python3 tools/perf_harness.py workflow \
      --name r8-long --id r8-gqa-confirm-01

`tools/perf-qwen-validate.sh` contiene build CUDA e gate numerici/unitari.
`tools/perf-qwen-long-context.sh` contiene profile, A/B direction e conferma
slow. `tools/perf-qwen-r8.sh` rilinka sempre `ds4`, `ds4-bench` e `ds4-server`,
esegue il probe numerico nei percorsi build/slow/long e confronta il rollback
F32 con R8 predefinito
oppure split-K scalare con il riuso GQA senza dover ricostruire a mano i
comandi. Questo evita che un benchmark aggiornato conviva con un client o un
server obsoleto. `--dry-run` stampa il comando senza
eseguirlo. Le suite canoniche
`long-context-direction` e `long-context-slow` sono definite in
`performance/workloads.yaml`; la prima non può emettere KEEP.
La conferma slow usa un solo sweep residente 8K/12K/16K per lato, con warm-up
separato, 64 token per frontiera e cinque ripetizioni, evitando un nuovo
caricamento del modello per ogni frontiera.
Il baseline forza il fallback riproducibile con
`DS4_CUDA_QWEN_NO_SPLIT_K_ATTN=1`; il candidato verifica il dispatch di
produzione, che usa split-K32 automaticamente da 96 token. `--candidate-env` resta
disponibile per esperimenti su partizioni e soglie, ma non serve nel run
canonico.

Con R8 predefinito, `r8-long` isola il secondo livello del dispatch: il baseline
usa `DS4_CUDA_QWEN_NO_GQA_GROUP_ATTN=1`, mentre il candidato di produzione
riusa ogni riga K/V per due query head del gruppo GQA. L'override diagnostico
`DS4_CUDA_QWEN_GQA_GROUP_ATTN=1|2|3|6` resta disponibile, ma il default
confermato è 2.

Il registro unico di soluzioni mantenute e scartate è
`docs/qwen36-performance-ledger.md`. I documenti datati restano fonti storiche,
ma una decisione nuova va sempre memorizzata anche nel ledger.

## Curva completa di contesto 2K-30K

La suite non rapida `context-curve-full` misura 15 frontiere, da 2048 a 30720
token con passo 2048, 64 token di decode e due ripetizioni. Un unico sweep
mantiene modello e sessione residenti. Il record include un gate autonomo che
richiede almeno 20 tok/s a ogni frontiera e rifiuta una risalita adiacente
maggiore di `max(1.5 tok/s, 8%)`; i punti sopra 30 tok/s sono solo annotati e
non vengono rallentati o penalizzati.

    python3 tools/perf_harness.py run \
      --id context-curve-default-01 \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --prompt tests/long_context_story_prompt.txt \
      --suite context-curve-full --repetitions 2 --warmup always \
      --hypothesis "default decode stays above 20 tok/s from 2K to 30K" \
      --metric gen_steady_tps \
      --baseline performance-results/context-curve-baseline/experiment.json

`server-curve` avvia un `ds4-server` isolato su una porta locale libera e usa
lo stesso elenco di frontiere. Calibra il filler contro `usage.prompt_tokens`,
verifica che ogni richiesta abbia esattamente il contesto richiesto, richiede
almeno 32 token generati e usa l'`avg t/s` di decode emesso dal server, non il
tempo HTTP comprensivo del prefill. Le due risposte per frontiera devono inoltre
avere lo stesso hash; un candidato confrontato a un baseline deve conservare
anche gli hash del baseline.

    python3 tools/perf_harness.py server-curve \
      --id server-context-curve-default-01 \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --repetitions 2 \
      --hypothesis "server decode stays above 20 tok/s from 2K to 30K" \
      --baseline-run

## Ciclo di sviluppo rapido

`make perf-fast-test` esegue soltanto i test unitari del harness e termina in
pochi secondi; non viene presentato come benchmark. Per un confronto model-backed
in un solo comando, la suite `direction` esegue baseline e candidate con due
workload e due campioni:

    make perf-direction-ab PERF_ID=score4-01 \
      PERF_CANDIDATE_ENV=DS4_CUDA_DECODE_SCORE4=1

Il target crea `score4-01-baseline` e `score4-01-candidate`, salva i logits e
stampa differenze prestazionali e drift. `PERF_ID` deve essere nuovo: gli
esperimenti esistenti non vengono sovrascritti.

`ds4-bench --repetitions N` mantiene modello e sessione residenti fra i campioni.
Per `direction`, context 128 e 2K sono inoltre due frontiere dello stesso sweep:
baseline e candidate richiedono quindi un caricamento del modello ciascuna, non
uno per campione e workload. Lo smoke RTX 3090 ha prodotto quattro righe per
lato e completato l'A/B in 257,5 s inclusa la ricompilazione; il costo residuo è
principalmente costituito dai due caricamenti necessari a cambiare le opzioni
CUDA lette durante l'inizializzazione.

La suite direction contiene soltanto due prove model-backed: context 128 e
context 2K. Entrambe misurano prefill e decode. Esegue due campioni senza un
processo di warm-up separato e serve a capire subito la direzione; non può
emettere KEEP, soltanto NEED_MORE_DATA o un rifiuto per drift numerico.

    python3 tools/perf_harness.py run --id base-direction \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --prompt tests/long_context_story_prompt.txt \
      --hypothesis "freeze direction baseline" \
      --metric gen_steady_tps --baseline-run

La candidate usa automaticamente gli artefatti della baseline. Per ogni
workload con generazione l'harness confronta sia i logits alla frontiera
post-prefill, sia la sequenza greedy completa e i logits finali post-decode.
Il verdetto mostra quindi velocità e drift nello stesso comando e il gate
attraversa i kernel di decode oggetto dell'esperimento:

    python3 tools/perf_harness.py run --id candidate-direction \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --prompt tests/long_context_story_prompt.txt \
      --hypothesis "the candidate reduces decode traffic" \
      --metric gen_steady_tps \
      --env DS4_CUDA_DECODE_SCORE4=1 \
      --baseline performance-results/base-direction/experiment.json

Quando la direzione è promettente, confermarla con suite slow, almeno cinque
ripetizioni e warm-up:

    python3 tools/perf_harness.py run --id base-slow --suite slow \
      --repetitions 5 --warmup always \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --prompt tests/long_context_story_prompt.txt \
      --hypothesis "freeze slow confirmation baseline" \
      --metric gen_steady_tps --baseline-run

    python3 tools/perf_harness.py run --id candidate-slow --suite slow \
      --repetitions 5 --warmup always \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --prompt tests/long_context_story_prompt.txt \
      --hypothesis "confirm score4 on representative contexts" \
      --metric gen_steady_tps \
      --env DS4_CUDA_DECODE_SCORE4=1 \
      --baseline performance-results/base-slow/experiment.json

## Profilo delle parti della rete

Il runtime emette tempi per layer, separando attention e FFN e distinguendo
layer Gated DeltaNet ricorrenti e full-attention. Un solo comando produce quote,
hotspot ordinati e il numero di elementi che copre il 95% del tempo profilato:

    python3 tools/perf_harness.py profile-network \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --prompt tests/long_context_story_prompt.txt \
      --context 1024 --generation-tokens 2 --prefill-chunk 512 \
      --output performance-results/base-network.json

Ripetere sulla candidate e localizzare subito le differenze nello stesso
comando:

    python3 tools/perf_harness.py profile-network \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --prompt tests/long_context_story_prompt.txt \
      --context 1024 --generation-tokens 2 --prefill-chunk 512 \
      --env DS4_CUDA_DECODE_SCORE4=1 \
      --baseline performance-results/base-network.json \
      --output performance-results/candidate-network.json

Questo crea sia candidate-network.json sia
candidate-network.comparison.json. Il comando compare-network rimane utile
per confrontare due report già esistenti:

    python3 tools/perf_harness.py compare-network \
      performance-results/base-network.json \
      performance-results/candidate-network.json

I timer per layer inseriscono sincronizzazioni intenzionali: servono ad
attribuire il costo alle parti della rete, non sostituiscono i token/s misurati
fuori dal profiler. Qwen usa intenzionalmente il percorso token-by-token sotto
512 token; per osservare anche il prefill layer-major il profilo usa quindi una
frontiera di 1.024 token divisa in due chunk. Il report usa il secondo chunk:
il primo riscalda lazy setup e risoluzione dei pesi e viene registrato fra le
righe cold scartate, non come costo stabile del layer 0.
Con `--generation-tokens` il report include inoltre `decode_token_summary`, che
scompone il token profilato in embedding, attenzione ricorrente, full-attention
(`qkv`, core e proiezione di uscita), FFN, output head e readback.

È anche possibile confrontare separatamente due dump realmente prodotti dai
run direction precedenti:

    python3 tools/perf_harness.py drift \
      performance-results/base-direction/logits/decode-b1-c128-direction/frontier_000128.logits.json \
      performance-results/candidate-direction/logits/decode-b1-c128-direction/frontier_000128.logits.json

Il gate controlla valori non finiti, argmax, overlap top-20, MAE/RMSE/errore
massimo dei logits centrati e cosine similarity. Gli artefatti
`frontier_N.decode.json` aggiungono uguaglianza esatta degli ID generati e il
confronto dei logits dopo l'ultimo token; una divergenza della sequenza è FAIL
anche quando il frontier prefill è identico.

Per la parity del quantizzatore Q8_1, il confronto dei dump packed distingue
`ds.x` e i 32 `qs` realmente consumati dal decode MMVQ dal metadata `ds.y`:

    python3 tools/perf_harness.py q8-1-parity \
      performance-results/ds4-q8_1.bin \
      performance-results/llama-q8_1.bin \
      --output performance-results/q8-1-parity.json

Il comando fallisce se scala, `qs` o `qsum` differiscono. Una differenza nel
solo `ds.y` resta visibile nel report, ma non viene promossa a causa del dot
MMVQ finché quel campo non entra nel dataflow del kernel osservato.

Per evitare script ad hoc quando si ripete la triangolazione A/B/C, una riga
full-vocabulary di più run Qwen si confronta con etichette e token sentinella:

    python3 tools/perf_harness.py qwen-logits-row \
      --run A=performance-results/qwen-a \
      --run B=performance-results/qwen-b \
      --run C=performance-results/qwen-c \
      --case multi_turn_preserve_thinking --stream greedy --row 3 \
      --focus-token 310 --focus-token 728 --output performance-results/abc.json

Il gate semantico di una suite oracle completa non richiede piu script ad hoc:

    python3 tools/perf_harness.py qwen-argmax-gate \
      gguf-tools/quality-testing/staging/oracles/ds4-q4ks-f32-point1-002 \
      gguf-tools/quality-testing/staging/oracles/candidate \
      --output performance-results/qwen-argmax-gate.json

Il report conta separatamente sequenze greedy complete, argmax greedy e argmax
teacher-forced; con otto casi da 32 step il gate atteso e `8/8` e `512/512`.

Il decode Q8_1-R8 a due stadi è la baseline di produzione per Qwen3.6 CUDA:

    ./ds4-bench \
      --cuda -m gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --prompt-file tests/long_context_story_prompt.txt \
      --ctx-start 128 --ctx-max 2048 --gen-tokens 8 --repetitions 3

Il kernel finale decodifica ogni blocco peso una volta sola e alimenta i due
accumulatori residui, evitando di duplicare unpack, scale e minimi. Non serve
più anteporre variabili a `./ds4` o `./ds4-server`; per un A/B diagnostico il
solo rollback è `DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8=1`.

Per un audit isolato del primo matvec Q4_K, Q5_K e Q6_K, aggiungere
`DS4_CUDA_QWEN_DECODE_Q8_1_R8_PROFILE=1`: il runtime stampa separatamente
durata del quantizzatore e del MMVQ, shape e tipo del peso. Il profiling crea
eventi e sincronizza soltanto con questo flag; non va usato per il benchmark.

## Cost model e margine teorico

Il comando legge soltanto header, metadata e directory tensor del GGUF; non
carica né calcola lo SHA-256 dei pesi. Per il decode collega byte minimi, FLOP,
arithmetic intensity e roofline al profilo appena prodotto:

    python3 tools/perf_harness.py model-cost \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --phase decode --context 1024 \
      --network-profile performance-results/base-network.json \
      --output performance-results/base-decode-cost.json

Per il secondo chunk prefill da 512 token:

    python3 tools/perf_harness.py model-cost \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --phase prefill --context 512 \
      --network-profile performance-results/base-network.json \
      --output performance-results/base-prefill-cost.json

I picchi hardware vengono ricavati dalla specifica statica associata alla GPU
rilevata (sulla RTX 3090: 936,2 GB/s e 35,58 TFLOP/s FP32) e sono sempre
registrati nel JSON. Se la GPU non è riconosciuta il report dichiara il fallback
RTX 3090. I valori possono essere sostituiti con
le opzioni `--memory-gbps` e `--compute-tflops`. Il rapporto observed/floor è
diagnostico: i timer per layer sincronizzano il device e il minimum traffic non
include attivazioni, cache, temporanei o scritture. Il JSON dichiara inoltre che
i FLOP del core ricorrente Gated DeltaNet, normalization e operazioni elementwise
non sono ancora inclusi: il floor è un limite inferiore, non una previsione del
tempo atteso.

## Comandi di servizio

    make perf-doctor
    make perf-harness-test
    python3 tools/perf_harness.py doctor
    python3 tools/perf_harness.py probe
    python3 tools/perf_harness.py run --id baseline --suite quick \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --prompt tests/long_context_story_prompt.txt \
      --hypothesis "freeze current release baseline" \
      --metric gen_steady_tps --baseline-run
    python3 tools/perf_harness.py run --id candidate --suite quick \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --prompt tests/long_context_story_prompt.txt \
      --hypothesis "score4 reduces decode weight traffic" \
      --metric gen_steady_tps --env DS4_CUDA_DECODE_SCORE4=1 \
      --baseline performance-results/baseline/experiment.json
    python3 tools/perf_harness.py compare \
      performance-results/baseline/experiment.json \
      performance-results/candidate/experiment.json

`doctor` controlla anche la freschezza di `ds4`, `ds4-bench` e `ds4-server`: se uno dei
binari è più vecchio dei sorgenti o degli oggetti da cui dipende, espone gli
input più recenti in `runtime_binaries.*.newer_inputs` e non dichiara pronto il
relativo percorso. La correzione canonica è `workflow --name r8-build` oppure
il workflow `validate`, non la sola build di `ds4-bench`.
Lo stesso report espone `qwen_decode_defaults`: R8, split-K32 da 96 token e GQA2,
insieme ai tre flag di rollback. È quindi possibile verificare la baseline
attiva senza dedurla dalla shell history.

Nel client interattivo `--ctx 32768` riserva la capacità massima ma non significa
che ogni token stia già elaborando 32K posizioni. Per confrontare un risultato
con il harness occorre registrare posizione corrente, token di prompt e fase:
`prefill` non è `generation`, il primo token non è la mediana steady e i turni
successivi accumulano contesto. Un saluto breve con client rilinkato deve essere
confrontato con la frontiera 128, non con il workload 8K/16K.

Le suite slow/standard eseguono un processo di warm-up non cronometrato per
workload; direction/quick lo omettono in modalità auto per ridurre il tempo di
feedback. I CSV grezzi e experiment.json sono
salvati sotto performance-results/, che Git ignora. Una directory esistente non
viene mai sovrascritta. Lo SHA-256 del GGUF viene conservato in una cache
invalidata da path, dimensione e mtime, così le iterazioni successive non
rileggono decine di GiB.

Il file workloads.yaml usa sintassi JSON, sottoinsieme valido di YAML, per
evitare dipendenze Python. Il workload batch 4 è deliberatamente
NOT_VERIFIED: l'attuale ds4-bench misura una sessione per processo e il harness
non finge throughput aggregato.

## Confine del primo incremento

Questo incremento copre manifesto, workload canonici, hardware/thermal snapshot,
statistiche, provenance, confronto immediato, drift dei logits, profiling
attention/FFN per layer e cost model da GGUF. Nsight non è installato nel PATH
Windows osservato: doctor lo registra esplicitamente. Parser Nsight, NVTX e
microbenchmark isolati restano incrementi successivi; i risultati mancanti
sono non verificati.

Su Windows il binario ds4-bench del repository è ELF: i comandi model-backed
vanno eseguiti nella stessa distribuzione WSL/Linux usata per compilare DS4.
Probe e test puramente Python funzionano anche dall'host.

## Gate MTP Qwen3.6 depth 2

I prompt canonici sono `mtp-copy-simple-prompt.txt` (copia ripetitiva ad alta
acceptance) e `mtp-copy-prompt.txt` (testo naturale a acceptance piu' bassa).
Il confronto greedy va eseguito sullo stesso binario e senza flag di tuning:

    ./ds4 --cuda -m gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --ctx 768 --prefill-chunk 16 --raw-prompt \
      --prompt-file performance/mtp-copy-simple-prompt.txt -n 128 --temp 0

    DS4_MTP_STATS=1 ./ds4 --cuda \
      -m gguf/Qwen3.6-27B-Q4_K_S.gguf --mtp \
      --ctx 768 --prefill-chunk 16 --raw-prompt \
      --prompt-file performance/mtp-copy-simple-prompt.txt -n 128 --temp 0

La gate sampled usa lo stesso seed e gli stessi filtri nei due processi:

    ./ds4 --cuda -m gguf/Qwen3.6-27B-Q4_K_S.gguf \
      --ctx 768 --prefill-chunk 16 --raw-prompt --seed 123 \
      --prompt-file performance/mtp-copy-simple-prompt.txt -n 128 \
      --temp 1 --top-p 1 --min-p 0.05

    DS4_MTP_STATS=1 ./ds4 --cuda \
      -m gguf/Qwen3.6-27B-Q4_K_S.gguf --mtp \
      --ctx 768 --prefill-chunk 16 --raw-prompt --seed 123 \
      --prompt-file performance/mtp-copy-simple-prompt.txt -n 128 \
      --temp 1 --top-p 1 --min-p 0.05

La gate model-backed di correttezza e':

    DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q4_K_S.gguf \
    DS4_TEST_MTP=gguf/mtp-Qwen3.6-27B-Q4_0.gguf \
    DS4_TEST_MTP_CTX=768 DS4_TEST_MTP_PREFILL_CHUNK=16 \
    DS4_TEST_QWEN_MTP_PATHS=1 DS4_MTP_STATS=1 \
    ./ds4_test --mtp-verify-depth

La baseline RTX 3090 del 2026-08-13 misura 30.48 -> 46.39 token/s nel CLI
(1.52x) e mediana server 31.47 -> 51.07 token/s (1.62x) sul prompt semplice.
Il prompt naturale misura soltanto 1.12x perche' accetta 70/111 draft; non deve
essere nascosto o sostituito dalla sola gate favorevole. Ogni regressione deve
conservare gap argmax 0.000, replay 0 nei percorsi partial/reject coperti e zero
fallback. L'audit completo e' in
`docs/research/guides/qwen36-mtp-llamacpp-ds4-control-flow.md`.

La gate sampled RTX 3090 misura 31.64 -> 48.74 token/s nel CLI (1.54x),
output byte-identico e 84/87 draft accettati. Nel server lo stesso profilo di
sampling misura 34.55 -> 50.15 token/s di decode (1.45x) e 4.526 -> 3.391 s
server-internal end-to-end (1.34x). La suite model-backed confronta inoltre 128
token sampled target-only/MTP con seed fisso e richiede uguaglianza esatta e un
chunk speculativo maggiore di uno.

Temperature, top-k, top-p e min-p filtrano una sola volta i logits target di
ogni riga verifier. Filtrare anche i logits MTP non sostituisce il sampler
target e duplicherebbe lavoro. I mask stateful per tool e response schema
richiedono invece avanzamento token-per-token; il percorso MTP stocastico li
esclude finche' lo stato del filtro non puo' essere checkpointato nel batch.

Come riferimento esterno locale, llama.cpp commit `1a064ab09` sullo stesso
hardware, modelli, file sorgente, 128 token greedy e depth 2 misura 40.9 ->
72.8 token/s (1.78x); gli output on/off differiscono solo nella riga dei tempi.
Il valore assoluto e' direzionale, perche' il frontend llama.cpp applica la
propria conversazione mentre il gate DS4 sopra usa `--raw-prompt`.

La curva MTP server usa capacità 28.737 e non supera 30K. Copre il bucket di
prompt minimo e poi 2K–28K a passo 2K:

    python3 tools/perf_harness.py server-curve \
      --id mtp-curve --suite mtp-context-curve \
      --model gguf/Qwen3.6-27B-Q4_K_S.gguf --mtp \
      --repetitions 2 --hypothesis "MTP stays faster through 28K" \
      --baseline performance-results/target-curve/experiment.json

La ricerca `mtp-threshold-search` confronta seriale e split-K a
64/125/250/500/1000 token; `mtp-threshold-midpoint` misura 96. La bisezione
RTX 3090 del 2026-08-13 ha trovato pareggio MTP a 64 e vantaggio split-K a 96
(+3,54% MTP, +7,53% target-only). Il cutoff automatico comune è quindi 96;
`DS4_CUDA_QWEN_NO_SPLIT_K_ATTN=1` resta il rollback seriale.

La profondità MTP è adattiva: V(3) sotto 2K, V(2) da 2K. Il confronto
V(3)/V(2) usa `mtp-depth-boundary`, `mtp-depth-crossover` e
`mtp-long-context-smoke`; `mtp-weakest-confirm` ripete il punto col margine
minore. La curva finale 0–28K migliora tutte le mediane target-only (+11,73%
medio, +10,04% a 28K). La conferma 24K a cinque run misura 23,26 → 23,91
tok/s (+2,79%, CV 0,71%/1,32%, `KEEP_CANDIDATE`). In V(2) i snapshot GDN
per riga rendono superfluo il pre-snapshot completo; il gate model-backed
passa reject/rollback, sampling 128/128 identico, gap argmax 0 e fallback 0.
