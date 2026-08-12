# DS4 performance harness

Il punto d'ingresso unico è tools/perf_harness.py. Riusa ds4-bench e gli scorer
Qwen esistenti: non introduce un secondo percorso di inferenza.

## Workflow eseguibili

Le sequenze ripetute non vanno ricostruite dalla shell history. Sono mantenute
in due script eseguibili e richiamate dal sottocomando `workflow`:

    python3 tools/perf_harness.py workflow --name validate

    python3 tools/perf_harness.py workflow \
      --name long-context-profile --id splitk-default-01

    python3 tools/perf_harness.py workflow \
      --name long-context-direction --id splitk-default-01

    python3 tools/perf_harness.py workflow \
      --name long-context-slow --id splitk-default-confirm-01

`tools/perf-qwen-validate.sh` contiene build CUDA e gate numerici/unitari.
`tools/perf-qwen-long-context.sh` contiene profile, A/B direction e conferma
slow. `--dry-run` stampa il comando senza eseguirlo. Le suite canoniche
`long-context-direction` e `long-context-slow` sono definite in
`performance/workloads.yaml`; la prima non può emettere KEEP.
La conferma slow usa un solo sweep residente 8K/12K/16K per lato, con warm-up
separato, 16 token per frontiera e cinque ripetizioni, evitando un nuovo
caricamento del modello per ogni frontiera.
Il baseline forza il fallback riproducibile con
`DS4_CUDA_QWEN_NO_SPLIT_K_ATTN=1`; il candidato verifica il dispatch di
produzione, che usa split-K32 automaticamente da 8K. `--candidate-env` resta
disponibile per esperimenti su partizioni e soglie, ma non serve nel run
canonico.

Il registro unico di soluzioni mantenute e scartate è
`docs/qwen36-performance-ledger.md`. I documenti datati restano fonti storiche,
ma una decisione nuova va sempre memorizzata anche nel ledger.

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
