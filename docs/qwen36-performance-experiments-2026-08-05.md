# Esperimenti prestazionali Qwen3.6 27B CUDA

> Registro storico. Le decisioni `KEEP/REJECT/NEED_MORE_DATA`, incluse quelle
> provenienti dagli altri documenti Qwen, sono consolidate in
> `docs/qwen36-performance-ledger.md`.

Data: 2026-08-05

## Obiettivo

Velocizzare il percorso CUDA di Qwen3.6 27B Q4_K_M senza degradare la
correttezza già raggiunta. I target richiesti sono:

- almeno 500 token/s di prefill;
- almeno 15 token/s di decode/inferenza;
- nessuna regressione nei token greedy, nei logits e nei gate numerici.

Il gate è considerato raggiunto soltanto se una singola configurazione di
release supera entrambe le soglie e passa i controlli di affidabilità.

## Ambiente di prova

- GPU: NVIDIA GeForce RTX 3090 24 GB;
- driver Windows: 610.62;
- architettura CUDA compilata: `sm_86`;
- modello: `gguf/Qwen3.6-27B-Q4_K_M.gguf`;
- SHA-256 modello:
  `65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921`;
- commit di partenza: `44ceb3e4271a155b974bd4d5b39a440438eb3eb7` con modifiche nel working tree;
- compilazione principale: `make -j2 cuda CUDA_ARCH=sm_86`.

Il caricamento del modello richiede normalmente 40-48 secondi e non è incluso
nei token/s riportati.

## Baseline iniziale

Prima delle modifiche, il prefill inoltrava un token alla volta attraverso i
64 layer. Ogni token causava inoltre sincronizzazione GPU, lettura dei logits e
calcolo dell'output head da 248.320 vocaboli.

Misure iniziali:

| Percorso | Risultato |
|---|---:|
| Prefill a circa 1K token | 9,75 token/s |
| Decode | circa 9,8-11 token/s |
| Prefill 16K tentato in precedenza | interrotto dopo oltre 56 minuti |

## Modifica strutturale: prefill layer-major

È stato introdotto un grafo Qwen multi-riga che:

- incorpora un intero chunk di token;
- attraversa ogni layer una sola volta per chunk;
- mantiene l'ordine causale nella convolution e nella Gated DeltaNet;
- aggiorna KV e full attention per tutte le righe;
- calcola l'output head soltanto sull'ultima riga del chunk;
- esegue una sola sincronizzazione/lettura logits per chunk.

Per preservare la stabilità dei prompt brevi, il percorso consegnato usa:

- percorso token-exact per chunk inferiori a 512 token;
- percorso layer-major da 512 token in su;
- attention GEMM da 512 righe in su;
- kernel Q8 nativo e decode Q4_K con attivazioni F32.

## Esperimenti sul prefill

Le misure principali usano 2.048 token, chunk 2.048 e RTX 3090.

| Esperimento | Prefill token/s | Esito numerico | Decisione |
|---|---:|---|---|
| Vecchio percorso token-per-token | 9,75 | baseline calibrata | sostituito sui chunk lunghi |
| Layer-major F32, chunk 128 | 154,81 | nessun errore CUDA | utile, ma sotto target |
| Layer-major F32, chunk 512 | 175,01 | nessun errore CUDA | utile, ma sotto target |
| Layer-major F32 + Q8 GEMM, chunk 128 | 180,07 | candidato | sotto target |
| Layer-major F32/TF32 + Q8 GEMM + attention GEMM, 2.048 | 447,15 | fallisce un gate top-20 | non promosso |
| FP16 completo + Q8 GEMM + attention GEMM, 2.048 | 636,55 | drift numerico | non promosso |
| FP16 primi 52 layer, ultimi 12 FP32, 2.048 | 551,02 | Spearman sentinella 0,849 | non promosso |
| FP16 solo gate/up, down FP32, 2.048 | 522,61 | Spearman sentinella 0,854 | non promosso |
| Default affidabile: Q8 nativo, Q4 FP32/TF32, attention GEMM | 205,74 | percorso corto bit-identico | promosso |

### Profilazione del prefill

Su un chunk da 128 token, prima dell'attention GEMM, la profilazione aggregata
ha mostrato:

| Fase | Tempo |
|---|---:|
| Attention dei 48 layer ricorrenti | 382,5 ms |
| FFN dei 48 layer ricorrenti | 295,0 ms |
| Attention dei 16 layer full-attention | 101,3 ms |
| FFN dei 16 layer full-attention | 75,2 ms |

Con Q8 GEMM, la parte attention ricorrente è scesa a circa 283,1 ms e la full
attention a circa 48,5 ms. Su 2.048 token la full attention scalare diventava
invece il collo di bottiglia principale; il nuovo percorso GEMM ha portato il
candidato FP16 da 366,87 a 636,55 token/s.

### Tentativi di prefill scartati

- Forzare `CUBLAS_GEMM_DEFAULT_TENSOR_OP` ha peggiorato il risultato a
  136,39 token/s.
- Il kernel dense Q4_K x Q8_K basato su MMA ha misurato circa 150,53 token/s
  su chunk 128, senza vantaggio rispetto al GEMM dequantizzato.
- Usare FP16 su tutte le proiezioni ha superato ampiamente il target, ma ha
  riordinato il top-20 oltre gli inviluppi congelati.
- Mantenere FP32 soltanto negli ultimi 12 layer non ha corretto il punto
  sentinella.
- Limitare FP16 a gate/up ha superato 500 token/s, ma non ha superato il gate
  numerico.

## Esperimenti sul decode

Le misure corte usano context 128 e 16 token generati, salvo indicazione
diversa.

| Esperimento | Decode token/s | Esito numerico | Decisione |
|---|---:|---|---|
| Matvec Q4_K F32 originale | circa 9,8 | affidabile | baseline |
| Q4_K F32 warp8 | 15,11-16,40 | bit-identico al DS4 calibrato | promosso |
| Q4_K F32 warp8 + output head Q6_K warp8 | 16,26; steady 16,29 su 64 token | Q6_K bit-exact contro block256 | promosso |
| Q4_K x Q8_K warp8 | 21,03; steady 21,11 | diverge al passo 20 | non promosso |
| Q8 soltanto su gate/up, down F32 | 18,02 | stessa divergenza | non promosso |
| Q8 con scala per gruppi di 32 | 14,59-14,65 | più accurato ma più lento | rimosso/scartato |
| Q8 con scala per gruppi di 128 | 17,05-17,08 | sotto target | rimosso/scartato |
| Q8_K con scala least-squares | 21,02; steady 21,11 | stessa divergenza | non promosso |
| 52 layer Q8 + 12 finali F32 | 20,11; steady 20,07 | stessa divergenza | non promosso |
| 12 layer iniziali F32 + 52 Q8 | circa 20 attesi | stessa divergenza | non promosso |
| Default F32 warp8, context 2.048 | 9,28 | affidabile | promosso |

### Causa del rifiuto del decode Q8

Nel caso `short_fact_english`, il candidato Q8 coincide con l'oracolo per i
primi 20 token, poi sceglie il token 11 invece del token 321. Il margine
dell'oracolo in quella posizione è 0,1137 logit.

Prima della biforcazione le metriche restano molto vicine:

- MAE centrata: 0,0922;
- coseno: 0,9999962;
- overlap top-20: 0,90.

La differenza è comunque sufficiente a cambiare l'argmax e quindi tutta la
continuazione successiva. Anche il fit least-squares della scala Q8 e le due
politiche ibride per layer producono la stessa biforcazione. Per questo il
decode Q8 non è stato reso predefinito.

## Test di correttezza eseguiti

### Probe numerici CUDA

Comando:

```sh
make qwen-numerics CUDA_ARCH=sm_86
```

Risultato: `PASS`.

Valori massimi osservati:

| Probe | Errore massimo |
|---|---:|
| GDN convolution + SiLU | 5,59e-9 |
| Stato convolution | 0, bit-exact |
| Stato ricorrente | 2,33e-9 |
| Output heads | 2,33e-8 |
| Q4_K matvec vs dequantizzazione F32 | 1,53e-5 |

Il probe separato F32 contro attivazione Q8_K misura MAE 0,2016: conferma che
la differenza è una policy aritmetica visibile, non un errore casuale.

### Pattern dei layer Qwen

Comando:

```sh
./ds4_test --qwen35-layer-pattern
```

Risultato: `OK`.

### Test del comparatore

Comando:

```sh
python3 -m unittest tests.test_qwen36_equivalence
```

Risultato: 19 test superati.

È stato inoltre corretto il comportamento di `--diagnostic`: un nuovo commit
può ora essere misurato contro le soglie calibrate senza dichiarare falsamente
di essere il vecchio commit. Il risultato diagnostico non può comunque essere
`PASS`; resta `NOT_VERIFIED` fino alla revisione della provenienza.

### Suite corta full-vocabulary finale

Run DS4 finale:

```text
ds4-q4-cuda-optimized-default-short-007
```

Riferimento llama.cpp:

```text
llama-q4-cuda-special-001
```

La suite contiene otto categorie, 32 passi greedy e 32 teacher-forced per
caso, per un totale di 512 posizioni e logits completi da 248.320 valori.

Risultato cross-engine:

| Metrica | Risultato |
|---|---:|
| Argmax agreement | 512/512 |
| Posizioni non-finite | 0 |
| MAE centrata media | 0,259644 |
| Errore massimo | 10,8924 |
| Overlap top-20 medio | 0,972070 |
| Spearman top-20 medio | 0,984507 |
| Rank agreement medio | 0,969942 |
| Greedy agreement | 1,0 |

Il confronto interno con `ds4-q4-cuda-tiled-special-short-001` copre
127.139.840 float e riporta:

- `different_float_count = 0`;
- MAE 0;
- errore massimo 0;
- greedy agreement 1,0;
- nessun valore non finito.

I report sono `NOT_VERIFIED` senza failure numeriche perché il nuovo commit non
è ancora la calibrazione revisionata e DS4 dichiara ancora
`native_rendering_status=tokenizer_only`.

## Configurazione finale consegnata

Il percorso predefinito conserva soltanto le modifiche considerate affidabili:

- prefill layer-major per chunk di almeno 512 token;
- percorso token-exact sotto 512 token;
- embedding Q4_K multi-riga;
- Gated DeltaNet e full attention multi-riga causali;
- attention basata su GEMM da 512 righe;
- Q4_K batch dequantizzato in FP32 con compute TF32;
- kernel Q8 nativo nel percorso predefinito;
- decode Q4_K F32 warp8;
- output head Q6_K F32 warp8, con lo stesso ordine di riduzione del kernel
  block256;
- output head calcolato soltanto sull'ultima riga del prefill.

I candidati FP16 e Q8 decode restano esclusi dal percorso di release perché
non superano i gate numerici.

## Confronto locale con LM Studio

Su richiesta e' stata analizzata l'installazione locale di LM Studio senza
modificarne configurazione o modelli. I log del 26 luglio mostrano sessioni con
prefill fra 987 e 1014 token/s e decode fra 52 e 60 token/s. La configurazione
persistita per Qwen usa:

- runtime CUDA llama.cpp `2.27.1`, release b10099, commit `1a064ab09`;
- full GPU offload (`offloadRatio=1`);
- eval batch 2048 e physical/ubatch 512;
- KV cache K e V in Q8_0, residente su GPU;
- mmap disabilitato;
- Gated Delta Net chunked fuso;
- speculative decoding MTP, con acceptance osservata nei log.

Il confronto non e' artifact-identical. LM Studio usa
`Qwen3.6-27B-Q4_K_S.gguf`, 16.121.357.440 byte, SHA-256
`a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`:
ha 65 blocchi, `nextn_predict_layers=1`, 866 tensori e il blocco MTP integrato.
DS4 usa il target non-MTP Q4_K_M da 19.095.766.304 byte, 64 blocchi e 851
tensori. Quindi il decode LM Studio include token MTP accettati e non puo'
essere attribuito soltanto ai kernel autoregressivi.

La prova A/B piu' istruttiva nei log LM Studio passa da circa 180/7 tok/s a
circa 1000/55 tok/s quando scompare l'avviso che il layer 0 e' rimasto sulla
CPU e che il GDN fuso e' stato disabilitato. Le parti trasferibili senza
cambiare semantica sono state applicate in DS4: full offload, batch lungo 2048,
GEMM multi-riga, attention GEMM e raggruppamento warp delle proiezioni decode.
Il GDN chunked parallelo e MTP restano ottimizzazioni successive, perche'
richiedono gate di equivalenza dedicati.

### Baseline di release dopo il confronto

Con la stessa build CUDA e lo stesso Q4_K_M DS4:

| Misura | Prima | Dopo | Esito |
|---|---:|---:|---|
| Prefill, 2048 token/chunk 2048 | 205,74 | 549,50 tok/s | supera 500 |
| Decode, contesto 128, 64 token | 14,74 | 16,94 tok/s | supera 15 |
| Decode steady, stesso run | 14,77 | 16,98 tok/s | supera 15 |

Il nuovo output head Q6_K warp8 riduce il numero di blocchi schedulati per
token da 248.320 a 31.040. Il probe A/B fra il vecchio block256 e warp8 copre
16 righe sintetiche e riporta MAE 0, errore massimo 0 e classificazione
`exact`. La suite numerica completa e il pattern dei layer restano PASS.

Il gate macchina-verificabile e' salvato in
`gguf-tools/quality-testing/staging/qwen36-speed-rtx3090-2026-08-05.json`.
Il formato v2 registra separatamente comando e contesto di prefill e decode,
come nei benchmark standard pp/tg, evitando di attribuire due misure a una
sola frontiera.

### Crossover Q4_K per il physical batch da 128

Il primo tentativo di imitare il physical batch 512 di LM Studio aveva lasciato
il GEMM Q4_K attivo soltanto da 512 righe. La profilazione
`DS4_CUDA_QWEN_PROFILE=1` ha mostrato che questa scelta rendeva i chunk 128
quasi interamente FFN-bound:

| Configurazione 512 token, chunk 128 | Prefill | Attention/chunk | FFN/chunk |
|---|---:|---:|---:|
| Q4_K riga-per-riga sotto 512 | 14,52 tok/s | 0,28-0,30 s | 8,11-8,76 s |
| Q4_K dequant + cuBLAS da 128 | 247,01 tok/s | 0,23-0,31 s | 0,22-0,28 s |

Il guadagno e' circa 17x. Il percorso conserva i pesi residenti quantizzati e
dequantizza soltanto la matrice corrente in scratch, la stessa separazione fra
modello quantizzato, eval batch e physical batch osservata nella configurazione
LM Studio.

Il run reale
`ds4-q4-cuda-gemm128-check-ctx4096-chunk128-r1` usa 4.009 token di prompt e 32
passi greedy piu' 32 teacher-forced. Contro il run monolitico conserva 32/32
token greedy, token teacher-forced e bytes, senza valori non finiti. Il diverso
ordine di calcolo chunked non e' bit-exact: MAE media 0,09117, cosine media
0,99999887, overlap top-20 0,990625 e Spearman top-20 0,994495. Il report
`staging/gemm128-ctx4096-chunk-invariance.json` resta quindi `FAIL` rispetto al
gate DS4-vs-DS4 bit-exact e la matrice long-context non viene considerata
completa; questi numeri sono registrati come evidenza, non trasformati in una
nuova soglia permissiva.

### Verifiche reali 4K e 16K dopo il crossover

Sul caso `long_canary_ctx4096` sono stati eseguiti i chunk 128, 512, 2048 e
4096, ciascuno con 4.009 token di prompt, 32 passi greedy e 32 teacher-forced.
Tutti conservano 32/32 token greedy rispetto a llama.cpp, zero non-finiti e
nessun fallimento metrico nel profilo diagnostico. L'overlap top-20 medio e'
fra 0,978125 e 0,98125 e la cosine media e' almeno 0,9999966. Il chunk 128 e'
stato inoltre ripetuto in tre processi: r1-r2 e r1-r3 hanno zero float diversi,
MAE 0 e massimo errore 0.

Il caso `long_canary_ctx16384` ha poi completato realmente con 16.302 token di
prompt e chunk 2048 in 177-180 secondi complessivi per processo, inclusi circa
40 secondi di caricamento e i due passaggi dello scorer. Contro llama.cpp:

| Metrica 16K/chunk2048 | Risultato |
|---|---:|
| Greedy token agreement | 32/32 |
| Valori non finiti | 0 |
| MAE media centrata | 0,192708 |
| Cosine media | 0,999997285 |
| Overlap top-20 medio | 0,98125 |
| Spearman top-20 medio | 0,990774 |
| NLL teacher-forced DS4 | 0,0204315 |

Tre processi 16K indipendenti sono bit-identici fra loro: zero float diversi,
MAE 0 e massimo errore 0 per r1-r2 e r1-r3. I confronti cross-engine restano
marcati `NOT_VERIFIED`, non `PASS`, perche' il manifest long-context dichiara
esplicitamente che le soglie non sono ancora calibrate. I dati sono positivi ma
non autorizzano a chiudere la matrice completa o i test 32K/massimo contesto.

Una ripetizione 16K aggiuntiva per chunk mostra il costo end-to-end dello
scorer, model load incluso:

| Chunk 16K | Tempo | Correttezza osservata |
|---:|---:|---|
| 128 | 629,8 s | 32/32 greedy, zero non-finiti, overlap 0,97656 |
| 512 | 210,0 s | 32/32 greedy, zero non-finiti, overlap 0,97656 |
| 2048 | 177-180 s | 32/32 greedy, zero non-finiti, overlap 0,98125 |
| 4096 | > 904 s | timeout; processo figlio terminato, nessun report promosso |

Il chunk 4096 esercita una pressione di scratch e memoria molto maggiore con
17,77 GiB di modello gia' residenti; il timeout non viene interpretato come un
risultato numerico. Poiche' 2048 e' il punto piu' rapido verificato ed e' anche
l'eval batch del runtime LM Studio locale, il default Qwen CUDA passa da 512 a
2048. La modifica e' limitata alla famiglia Qwen; gli override espliciti e i
default DeepSeek/GLM non cambiano.

Lo smoke finale senza `--prefill-chunk` conferma il cap reale 2048 nel log del
benchmark e misura 604,36 tok/s su 4.096 token. Il benchmark ora calcola il
riepilogo memoria dopo la creazione della sessione usando
`ds4_session_prefill_cap()`: in precedenza stampava una stima generica 4096 e
poteva attribuire la misura Qwen al chunk sbagliato.

### Regressioni finali

- build CUDA di `ds4`, server, bench, eval, agent e scorer: PASS;
- `tests/qwen_numerics_probe`: PASS, incluso Q6 warp8 bit-exact;
- `tests/cuda_long_context_smoke`: PASS;
- suite Python safety/equivalence: 24/24 PASS;
- `ds4_test --qwen35-layer-pattern --server`: PASS;
- agent ed estrattori eval: PASS;
- layer pack 97/97, placement multi-GPU 98/98, GPU args e CLI 44/44: PASS;
- sampling: PASS.

Il `make test` senza selezione e' stato interrotto dopo oltre 16 minuti mentre
eseguiva inferenza CPU model-backed sul `ds4flash.gguf` da 19 GB. Il processo
era attivo, non in deadlock, ma il run non ha prodotto un exit status finale e
non viene contato come PASS. I gruppi unitari e CUDA applicabili sono stati
rilanciati separatamente con esito verificabile come elencato sopra.

## Conclusione

La configurazione di release misurata con la stessa build raggiunge 549,50
tok/s di prefill e 16,94 tok/s di decode (16,98 steady): il gate 500/15 e'
verde. I candidati piu' rapidi che modificavano la continuazione greedy o
fallivano i gate top-20 restano diagnostici e non sono stati promossi. Il run
4K chunk-128 conserva i token ma non i logits bit-exact rispetto al monolitico;
per questo la matrice long-context rimane aperta e viene riportata separatamente
dal risultato prestazionale.

## Aggiornamento 2026-08-06: Ottimizzazione Register-State Caching per GDN CUDA

In data 2026-08-06 e' stata implementata l'ottimizzazione del caching dello stato nei registri GPU nei kernel CUDA `qwen35_gdn_rows_kernel` e `qwen35_conv_rows_kernel` in `ds4_cuda.cu`.

### Modifiche Apportate
- Caching locale della colonna dello stato GDN (`s_col[128]`) nei registri / L1 cache per ciascun thread CUDA, eliminando la scrittura/lettura continua in VRAM globale ad ogni token del chunk di prefill.
- Caching dello stato convoluzionale a 4 tap (`s0, s1, s2, s3`) e dei pesi nei registri locali per l'intera durata del chunk.

### Risultati delle Misure
- **Prefill Batched (4.714 token su RTX 3090):** **740,15 tok/s** (in aumento da 549,50 tok/s).
- **Decode Steady:** **18,54 - 19,51 tok/s** (in aumento da 16,94 tok/s).
- **Mantenimento Bit-Exactness:** Tutti i probe numerici `make qwen-numerics` risultano `PASS` con scostamento zero o di mero arrotondamento `float32` (MAE $< 10^{-10}$).
- **Suite di Test:** 29/29 test automatizzati (`test_qwen36_equivalence`, `test_qwen36_numerics`, `test_qwen36_safety`, `ds4_test --qwen35-layer-pattern`) superati con esito `PASS` / `OK`.

### Verifica Q4_K_S e residency del 2026-08-06

Il GGUF Unsloth Q4_K_S verificato ha SHA-256
`ff857ba9f2184d8be408e8cabda12c89ba5adb202fddc1a88b3774d7bb232aca`,
851 tensor e il mix F32=449, Q4_K=341, Q5_K=60, Q6_K=1. Il layout Q5_K è
ora validato per nome e ruolo, non accettato genericamente come qualunque peso
numerico. Il percorso Qwen single-GPU copia per default il modello in VRAM;
`DS4_CUDA_NO_MODEL_COPY=1` resta disponibile soltanto per diagnosi.

Sulla RTX 3090 il Q4_K_S occupa 14,76 GiB di pesi residenti. Il gate breve
misura 498,87 tok/s prefill e 15,96 tok/s decode steady a 512 token; il gate
rappresentativo a 2.048 token misura 691,43 tok/s prefill e 10,99 tok/s decode
steady, con 14,90 GiB pianificati. Il probe GDN copre anche quattro token in
una singola chiamata multi-riga: stato convoluzionale bit-exact e stato
ricorrente/output entro il solo roundoff float32. Il matvec Q5_K warp8 è
bit-exact rispetto al kernel block-256.

### Decode long-context: warp attention e tentativo grouped GQA del 2026-08-10

Sul Q4_K_S, con 10.666 token di prompt e 200 token greedy a context 32.768,
il kernel attention con riduzione warp passa da 3,46 a 4,31 token/s medi nel
decode (+24,6%). Il tempo HTTP comprensivo del prefill passa da 80,30 a 69,27
secondi. Il probe sintetico a posizione 255 confronta 6.144 valori: legacy e
warp restano nell'inviluppo di solo roundoff rispetto al riferimento CPU
(`max_abs` 3,35e-8 e 3,73e-8), e la continuazione reale di 200 token resta
identica. Il percorso rimane opt-in con `DS4_CUDA_QWEN_WARP_ATTN=1`.

È stato inoltre implementato e misurato un kernel grouped GQA che elaborava
insieme le sei query head associate alla stessa KV head, preservando bit-exact
l'output del kernel warp. Ha raggiunto soltanto 3,69 token/s: +6,6% rispetto al
legacy, ma -14,4% rispetto al warp. Ridurre da 24 a 4 CTA e mantenere sei
accumulatori per thread ha sacrificato parallelismo e aumentato la pressione
sui registri più di quanto il riuso K/V abbia ridotto il traffico. Il candidato
è stato rimosso senza ulteriore tuning.
