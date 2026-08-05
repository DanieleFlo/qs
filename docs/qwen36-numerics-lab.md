# Laboratorio numerico Qwen3.6

Questo laboratorio serve a separare tre fenomeni che non devono essere
confusi: un errore semantico del kernel, il normale errore floating point del
backend e una diversa policy aritmetica (per esempio attivazioni F32 oppure
riquantizzate Q8_K). Non richiede di eseguire Ollama e i probe sintetici non
caricano il GGUF.

## Componenti

- `tests/qwen_numerics_probe.c` confronta direttamente i kernel CUDA DS4 con
  oracle CPU indipendenti. Copre quattro passi consecutivi della Gated
  DeltaNet, stato convolutivo e ricorrente, il mapping fisico GGUF tiled di 16
  teste Q/K verso 48 teste V, normalizzazione/gating e un matvec Q4_K.
- `diagnose_qwen36_numerics.py` triangola le trace reali DS4, llama.cpp CPU e
  llama.cpp CUDA. Canonicalizza lo stato GDN da
  `[value_head,value_dim,key_dim]` a
  `[value_head,key_dim,value_dim]`, misura RMSE/coseno/errori per elemento e
  usa la distanza CPU-CUDA come scala empirica del roundoff.
- `tests/test_qwen36_numerics.py` verifica canonicalizzazione e classificatore
  con fixture sintetiche.

Il report usa `exact_reference_match`,
`within_cpu_cuda_roundoff_envelope`,
`suspicious_outside_backend_envelope` e `invalid_nonfinite`. “Suspicious” non
significa automaticamente bug: bisogna controllare il primo stadio e poi
isolare la sua primitiva, perché una differenza a monte si propaga.

## Comandi

Su RTX 3090 (`sm_86`):

```sh
make qwen-numerics CUDA_ARCH=sm_86
```

Per una trace reale alla posizione 23:

```sh
python gguf-tools/quality-testing/diagnose_qwen36_numerics.py \
  --ds4-dir gguf-tools/quality-testing/staging/qwen-full-prompt-trace/ds4 \
  --llama-cpu-dir gguf-tools/quality-testing/staging/qwen-full-prompt-trace/llama-cpu \
  --llama-cuda-dir gguf-tools/quality-testing/staging/qwen-full-prompt-trace/llama-cuda \
  --position 23 --layer 0 --report numerics-envelope.json
```

## Audit del sorgente Ollama

Audit eseguito il 2026-08-05 sul commit Ollama
`43983edf1826be640b76e88ae68a5f721d2fb274`, che fissa llama.cpp `b10242`
(commit risolto `96278e39fc83e1d97c881e34bcec39ac7ea98820`).

Su CUDA Ollama costruisce il runner tramite il llama.cpp fissato in
`LLAMA_CPP_VERSION`; non esiste quindi un terzo oracle NVIDIA indipendente.
Nel sorgente fissato:

- il GDN fuso e' abilitato di default;
- il converter llama.cpp riordina tutte le componenti value-side da ordine HF
  grouped a ordine GGUF tiled, incluse V, Z, alpha, beta, conv e colonne della
  proiezione di uscita;
- per questo il kernel GGUF CPU/CUDA sceglie correttamente la testa Q/K con
  `value_head % key_head_count`;
- il backend MLX di Ollama sceglie
  `value_head / (value_head_count / key_head_count)` perche' opera sui pesi
  upstream non riordinati;
- DS4 usava erroneamente `/3` sul GGUF riordinato. Il mapping e' stato corretto
  in `%16`.

La differenza MLX/GGUF non e' quindi un conflitto semantico: e' compensata dal
riordino del converter. Un oracle che forza `repeat_interleave` direttamente
sul GGUF verifica il layout sbagliato. Unsloth o Transformers restano utili
come oracle sui pesi upstream, a condizione di confrontarli prima del riordino
oppure di applicare esplicitamente la stessa permutazione.

## Risultati ottenuti

Il probe CUDA sintetico passa:

- `conv_state` bit-exact per tutti e quattro i passi;
- `conv_silu` errore massimo `5.59e-9`;
- stato ricorrente errore massimo `2.33e-9`;
- output heads errore massimo `2.24e-8`;
- matvec Q4_K DS4 contro dequantizzazione F32: errore massimo `1.53e-5`,
  classificato roundoff;
- cambiando soltanto la policy dell'attivazione da F32 a Q8_K, lo stesso test
  sintetico produce MAE `0.2016`: e' una differenza di policy misurabile, non
  un errore del kernel Q4_K.

Le precedenti 723 coppie di trace erano state prodotte dopo aver patchato
llama.cpp verso `repeat_interleave` e con DS4 ancora in `/3`: confrontavano due
implementazioni coerenti fra loro ma non col layout fisico del GGUF. Restano
utili per misurare roundoff locale, ma non possono validare la semantica
end-to-end. Devono essere rigenerate dopo la correzione `%16`.

Il test manuale che ha localizzato il problema ha usato il prompt canonico
senza system implicito:

```text
Rispondi con una sola parola: qual è il colore del cielo?
```

Con `/3` DS4 ripeteva i token iniziali del prompt; con `%16` risponde `Blu`.
Il probe GDN aggiornato al layout fisico continua a passare su quattro passi.

### Suite corta rigenerata dopo la correzione

Il run `ds4-q4-cuda-tiled-short-001` usa llama.cpp non patchato
`7d442abf` come riferimento GGUF e contiene tutti gli otto casi corti, con 32
passi greedy e 32 teacher-forced per caso. La verifica strutturale e dei
checksum passa. Su 256 posizioni greedy DS4 e llama.cpp scelgono sempre lo
stesso argmax e le otto sequenze di token sono identiche; non compaiono NaN o
Inf. Le metriche aggregate sono: overlap top-20 medio `0.97207`, rank agreement
medio `0.96994`, coseno medio `0.9999900` e MAE teacher-forced delle logprob
oracle `0.005927`.

Il generatore llama.cpp tokenizzava con `special=True` ma detokenizzava col
default `special=False`. Il run rigenerato `llama-q4-cuda-special-001` conserva
esplicitamente i token speciali: contro
`ds4-q4-cuda-tiled-special-short-001`, tutti gli otto casi hanno token e byte
decodificati identici. Non era un errore del forward pass né del tokenizer DS4.

Il profilo cross-engine v2 è calibrato sugli estremi osservati, con margine
esplicito e provenienza hardware/modello nel manifest. Token, byte, argmax e
non-finite restano gate rigidi per posizione; overlap e rank top-20 del profilo
corto sono gate aggregati `>= 0.95`, mentre gli inviluppi numerici per posizione
restano separati e versionati. Il report calibrato ha zero failure.

La ripetibilità è stata controllata su tre run (`001`, `002`, `003`). I
confronti full-vocabulary `001-vs-002` e `001-vs-003` coprono 512 righe
ciascuno e riportano `different_float_count = 0`, MAE ed errore massimo zero,
argmax agreement 1 e nessun non-finite. I due run nuovi sono stati revisionati
manualmente dopo verifica di inventario, checksum e provenienza. Il report
finale resta `NOT_VERIFIED` soltanto perché DS4 usa ancora
`native_rendering_status=tokenizer_only`: il token-manifest scorer verifica il
tokenizer ma non ricostruisce autonomamente tutti i chat template
multi-turn/tool. Questo stato non va promosso artificialmente.

### Regressioni e long context

Le regressioni CUDA granulari condivise sono verdi, inclusi
`glm_selected_attention`, `glm_indexer_scores`, `glm_decode_attention_staged`,
MoE, Q8, sampling, placement GLM e lo smoke long-context sintetico. Non sono
presenti GGUF DeepSeek/GLM locali per un test model-backed. `make test` ha
compilato i target, ma `ds4_test` monolitico è stato interrotto dopo oltre 15
minuti di CPU senza un risultato finale; i restanti test granulari passano.

La matrice 4K/16K valida correttamente 30 run pianificati. L'esecuzione è
bloccata prima del caricamento modello perché manca un report prestazionale
verde. La baseline disponibile è circa `9.75 tok/s` di prefill a 1K e
`11 tok/s` di decode, contro i gate `500/20`; ripetere ora il 16K ricreerebbe
il precedente prefill da oltre 56 minuti senza risolvere il collo di bottiglia.

### System prompt

Il system prompt non era stato rimosso: la CLI usa per default
`You are a helpful assistant`; `--system ''` lo disabilita. Dopo la correzione
del layout è stato verificato esplicitamente anche con il default presente:
il prompt italiano contiene 36 token e produce `Blu`. L'help della CLI indica
ora sia il valore predefinito sia il modo per disabilitarlo.

### Prefill layer-major e tentativi di decode veloce

Il collo di bottiglia originale era strutturale: `ds4_session_sync` inoltrava
un token alla volta attraverso 64 layer, sincronizzava il device e calcolava
l'output head da 248.320 vocaboli a ogni posizione. Il nuovo grafo Qwen elabora
un chunk per layer, conserva l'ordine causale dentro convolution/Gated
DeltaNet e full attention, e legge i logits soltanto per l'ultima riga.

Misure release su RTX 3090, driver 610.62, Qwen3.6 27B Q4_K_M:

- prefill predefinito affidabile, chunk/context 2.048: `205.74 tok/s`;
- candidato con espansione Q8 FP16: `447.15 tok/s`, non promosso;
- candidato FP16 completo: `636.55 tok/s`, non promosso;
- candidato FP16 solo gate/up, down FP32: `522.61 tok/s`, non promosso;
- decode F32 warp a context 128: `15.11 tok/s` (`9.28 tok/s` a context 2.048);
- decode Q4_K x Q8_K a context 128: `21.03 tok/s`, non promosso.

Il run corto FP16 + decode F32 conserva tutti i 512 argmax, non contiene
NaN/Inf e ottiene overlap top-20 medio `0.9717`, ma fallisce l'inviluppo
per-posizione nel caso tool (Spearman `0.863 < 0.88`). Limitare FP16 alle sole
gate/up porta quel punto a `0.854`. Il decode Q8, incluso il fit least-squares
della scala, cambia invece il token greedy di `short_fact_english` al passo 20
(margine oracle `0.114`) e genera una continuazione diversa. Questi percorsi
restano diagnostici: il default usa attention GEMM soltanto da 512 righe,
kernel Q8 nativo e matvec decode F32. Il gate long-context non va sbloccato
finché una singola configurazione affidabile non supera sia 500 sia 20 tok/s.

Il percorso consegnato applica inoltre il crossover token-exact sotto 512
righe. Il run finale `ds4-q4-cuda-optimized-default-short-007` supera tutti i
gate numerici cross-engine: 512/512 argmax, zero non-finiti, overlap top-20
medio `0.972070`, Spearman `0.984507` e rank agreement `0.969942`. Il confronto
contro `ds4-q4-cuda-tiled-special-short-001` è bit-identico su 127.139.840
float (`different_float_count = 0`). Il report resta `NOT_VERIFIED`, senza
failure, perché il nuovo commit non è ancora la calibrazione revisionata e il
rendering DS4 è ancora `tokenizer_only`.
