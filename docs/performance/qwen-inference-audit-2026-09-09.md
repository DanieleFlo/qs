# Primo intervento utile dall'audit dell'inferenza

## Risultato e perimetro

La modifica mantenuta corregge il carry MTP nel prefill token per token:
`qwen_graph_forward_token_mode` produce `output_norm` anche quando non servono
logits, se MTP è attivo. Il passo successivo riceve così la coppia corretta
`(token[p], target_h[p-1])`. La proiezione sul vocabolario e il readback dei
logits restano saltati sui token intermedi. Nessun quantizzatore, kernel
numerico, formato KV o dispatch delle proiezioni è cambiato.

Misure GPU del 9 settembre 2026; chiusura e correzioni degli strumenti il
16 settembre 2026.

È il prerequisito funzionale segnalato nell'audit del 5 settembre, non un
guadagno da attribuire a un nuovo kernel. I tentativi prestazionali 1–5 sotto
sono stati chiusi senza promozione. Il 16 settembre il lavoro è ripreso sui
punti rimanenti; gli esiti sono registrati sotto, con un commit per punto.
Da questa ripresa Qwen3.6 è escluso dagli ulteriori approfondimenti, su
indicazione dell'utente; il target prestazionale è Qwen3.8.

Ambiente: RTX 3090, `sm_86`, WSL Ubuntu 24.04, driver Windows 610.62,
CUDA 12.4.131, `-O3 -g -lineinfo --use_fast_math -arch=sm_86`.
Base Git: `de9fcc5b0f514b93d408510c1324658b257e7369`, con modifiche locali
preesistenti al caricamento CUDA e ai launcher, conservate in tutti i confronti.
I modelli sono stati letti dalla copia nativa WSL in `/home/daniele/ds4/gguf`.

| Artefatto | SHA-256 verificato |
|---|---|
| Qwen3.8 UD-Q4_K_S | `75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017` |
| MTP Qwen3.8 Q4_0 | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |

## Riproduzione e correttezza

`tests/qwen_mtp_catchup_probe.c` controlla senza tracing:

- hidden normalizzato bit-exact contro RMSNorm dello stato target appena
  calcolato, avvelenando prima il vecchio buffer per intercettare letture stale;
- logits lasciati al valore sentinella, quindi output head effettivamente saltato;
- KV MTP rispetto al passo con il carry precedente, e successivo avanzamento
  del carry e della lunghezza cache;
- primo e secondo token, poi token 512/513 dopo un prefill di 512 righe.

Il probe compilato con `ds4.c` originale fallisce al confronto dell'hidden.
Con la patch passa su **Qwen3.8 e Qwen3.6 Q4_K_S**, usando i rispettivi sidecar.
La suite Python finale esegue **108 test: 105 PASS, 3 SKIP**. I tre test saltati
richiedono l'abilitazione esplicita delle suite live. Un primo lancio parallelo
al benchmark aveva incontrato il lock del modello; la riesecuzione isolata
passa. Build CUDA di CLI, server e benchmark completate; compilazione del
runtime CPU completata. Metal e ROCm non sono stati eseguiti.

Il gate GPU `ds4_test --mtp-verify-depth`, con
`DS4_TEST_QWEN_MTP_PATHS=1`, passa anche rifiuto totale, accettazione parziale,
raw-copy e 128 token con sampling a temperatura 1 e seed fisso, confrontati
con il target. Con MTP disabilitato i 248.320 logits del prompt sono bit-exact.

Comandi di regressione:

```sh
make -j2 tests/qwen_mtp_catchup_probe ds4 ds4-server ds4-bench CUDA_ARCH=sm_86
./tests/qwen_mtp_catchup_probe TARGET.gguf MTP.gguf
python3 -m unittest tests.test_perf_harness tests.test_qwen36_equivalence \
  tests.test_qwen36_numerics tests.test_qwen38_compatibility
make ds4_cpu.o
DS4_TEST_BACKEND=cuda DS4_TEST_MODEL=TARGET.gguf DS4_TEST_MTP=MTP.gguf \
  DS4_TEST_MTP_CTX=2048 DS4_TEST_MTP_PREFILL_CHUNK=512 \
  DS4_TEST_QWEN_MTP_PATHS=1 ./ds4_test --mtp-verify-depth
```

L'oracolo completo Qwen3.8 resta nello stato precedente; questa correzione
non ne certifica il completamento. La proprietà bit-exact della regressione
si riferisce a hidden e cache costruiti dalla medesima coppia token/hidden.
Una diversa accettazione MTP può cambiare la sequenza di batch del verifier:
non si presume automaticamente identità dei logits finali tra scheduling diversi.

## Misure del carry

Screening CLI, greedy, `--nothink`, 128 token, contesto allocato 2048,
chunk 512, stesso system prompt CLI e stesso GGUF, un processo alla volta:

| Caso | Prompt effettivo | Prima, token/s | Dopo, token/s | Accettazione prima → dopo |
|---|---:|---:|---:|---:|
| Guida di viaggio | 48 | 37,47 | 38,60 | 57,63% → 63,39% |
| LRU cache Python | 52 | 44,38 | 48,96 | 79,59% → 91,11% |
| Ristoranti JSON | 53 | 47,09 | 47,77 | 86,02% → 89,01% |

Il testo dei tre output è identico prima/dopo. Sono misure esplorative singole,
non una stima universale del beneficio. Nessun tempo di caricamento è incluso
nei token/s. Il caso JSON è un prompt testuale con output limitato a 128 token,
non una prova di conformità a uno schema JSON completo.

Una prova residente preliminare ometteva il system prompt CLI: è conservata
come workload distinto (`resident-session1.jsonl`), ma non conferma questi
numeri. Evidenzia inoltre che il beneficio dipende dal prompt: nel caso codice
senza system prompt i conteggi aggregati di accettazione coincidono.

La conferma usa `carry-cli-resident.c`, con system prompt
`You are a helpful assistant`, 48/52/53 token effettivi, capacità 1024, chunk
512 e 128 token di decode. Il modello resta residente per una sessione;
ogni caso ha una coppia di warm-up esclusa e cinque coppie A/B a ordine alternato.
La seconda sessione riapre il modello e ripete le cinque coppie. Il riferimento
del vecchio difetto congela lo scratch stale a zero e omette la sola norm
intermedia; il codice di produzione non espone questa variante funzionale.
I logits del prompt, i token generati e i logits finali vengono confrontati.

Risultati su **10 coppie per caso, due sessioni**, con intervallo bootstrap
appaiato al 95% (20.000 ricampionamenti, seed 20260909). La velocizzazione è
la mediana di `decode_ms_prima / decode_ms_dopo - 1`, distinta dal rapporto
fra le due mediane dei token/s.

| Caso | Prima → dopo, token/s mediani | Velocizzazione appaiata | IC 95% | CV decode prima → dopo |
|---|---:|---:|---:|---:|
| Guida di viaggio | 33,15 → 34,92 | **+5,38%** | +3,68% / +5,71% | 0,40% → 1,98% |
| LRU cache Python | 39,63 → 43,29 | **+9,05%** | +8,56% / +9,31% | 0,44% → 2,57% |
| Ristoranti JSON | 41,68 → 42,57 | **+2,01%** | +1,80% / +2,51% | 1,89% → 0,44% |

I conteggi di accettazione coincidono con lo screening CLI in tutte le
ripetizioni. Prompt logits e 128 token generati sono identici; anche i logits
finali sono bit-exact per codice e JSON. Nel caso testuale i logits finali
presentano 248.287 differenze su 248.320 valori, max abs `0.00155746937`:
la diversa accettazione modifica i batch del verifier. Il controllo separato
`carry-boundary.c` misura cosine `0.999999990456`, top-20 identico e stesso
argmax. Ripetendo il medesimo transcript un token target alla volta, tutti i
248.320 logits sono bit-exact a ciascuno dei 128 passi. Questo isola la
differenza dovuta al diverso scheduling del verifier, senza allentare gate.

Il prefill non mostra un beneficio sistematico: le mediane sono circa
1,72/1,71 s (testo), 1,86/1,86 s (codice), 1,90/1,90 s (JSON). Il beneficio
misurato riguarda il **decode con MTP dopo prompt brevi**, non caricamento,
prefill lungo o decode senza MTP. Il caso JSON resta sotto il 3% e il suo
guadagno è comparabile al CV della baseline: non è la base della decisione.
Il caso codice fornisce l'evidenza più netta, con output e logits bit-exact.

Questi sono confronti del percorso di generazione residente mediante API di
sessione, sincronizzati dalle letture dei logits. Non sono misure `ds4-bench`
né una certificazione di miglioramento universale dei kernel. Gli artefatti
`resident-cli-session{1,2}.jsonl` e `resident-cli-summary.json` conservano tutti
i campioni e, per prefill/decode, minimo, p10, mediana, media, p90, massimo,
deviazione standard e CV. Nessun campione misurato è stato scartato.

Tutti i frontend CUDA sono stati rilinkati dopo la ricompilazione del runtime
definitivo. Il CLI finale riproduce esattamente l'output codice precedente,
con 82 draft accettati su 90 proposti e zero fallback. Non contiene più il
flag diagnostico della variante accodata scartata.

## Bug degli strumenti corretti

- `compare_records`: non promuove più due campioni senza warm-up, modelli o
  input differenti, hardware/toolchain incompatibili, contesti effettivi o
  configurazioni MTP differenti. Verifica campioni grezzi, CV, definizioni e
  insieme completo dei workload. I record incompleti producono
  `NEED_MORE_DATA` e motivi espliciti, pur mostrando i delta descrittivi.
  Per binari diversi richiede metadati di build compatibili; non presume
  uguali i flag perché nvcc ha la stessa versione. I nuovi record includono
  configurazione MTP effettiva e hash del sidecar esplicito. MTP con sidecar
  automatico non identificato e server senza warm-up restano non promuovibili.
- `perf-qwen-r8.sh`: la baseline F32 disabilita entrambi i flag R8, incluso
  quello dei tipi UD Qwen3.8. Questa baseline diagnostica resta distinta dal
  default di produzione usato per nuove ottimizzazioni.
- `model_cost`: supporta tutti i tipi presenti nel target UD censito,
  considera 64 layer target (48 GDN, 16 full attention) ed esclude i pesi del
  NextN incorporato. Espone byte NextN esclusi e pesi di decode per formato
  senza embedding. I test confrontano il modello con e senza NextN ed
  esercitano le incompatibilità che prima potevano produrre un falso KEEP.

La discrepanza storica del default FP16 gate/up è una **riqualificazione
ancora aperta**, non un bug numerico dimostrato da questa sessione. Il default
esistente è stato mantenuto in entrambi i lati dei confronti; non è stato
riclassificato come validato. Rimane aperto anche l'oracolo completo Qwen3.8.

## Tentativi chiusi

| Punto dell'audit | Prova | Esito |
|---|---|---|
| 1. Unpack IQ4_XS condiviso fra q0/q1 | SASS di decode e verifier 2/3 righe | **Già eliminato dal compilatore.** 32 PRMT nel decode, 64 nel verifier; gli stessi valori decodificati alimentano entrambi gli stadi e le righe. Nessuna patch kernel. |
| 2. Accodare catch-up MTP | Un unico completamento per chunk, baseline dopo la correzione funzionale | **REJECT prestazionale.** Su 2048 righe mediana circa 292,54 → 246,90 ms nel catch-up; riduzione ~15,6%, sotto il 25%. Nelle due coppie end-to-end TTFT ~0,1% e ~1,8%, sotto il 10%. |
| 3. GEMM attention sui suffissi | 128 righe dopo 2048 posizioni KV | **REJECT numerico.** 786.353/786.432 valori diversi, max abs `0.000817775726`, cosine `0.999999971451`. Il gate interno non è stato allentato. |
| 4. Due righe di output per CTA R8 | IQ4_XS, Q4_K, Q5_K, Q6_K, forme 5120→17408 e 17408→5120; output Q6_K 5120→248320 | **Screening negativo.** Parità bit-exact, ma IQ4_XS migliora solo ~0,4–1,2%; diverse forme regrediscono. Output Q6_K ~5,2% sul solo kernel, con rumore e quota end-to-end insufficiente a motivare la promozione. |
| 5. Tile temporali GQA2 | Tile 2 e 4, 32 partizioni, batch 1/2/3, fino a posizione 28672 | **REJECT prestazionale.** Dopo aver fissato la contrazione FMA originaria, tutte le 114 combinazioni numeriche passano bit-exact sulle partial. Nessuna tile raggiunge il 15% richiesto; miglior caso misurato ~6,8%, con regressioni in altre forme. |

Per il punto 2 sono state confrontate anche KV complete e carry su 1, 2,
127, 128, 511, 512 e 2048 righe, prefisso non nullo, token invalidi e limiti
di capacità. Prompt logits, 128 token e logits finali restavano bit-exact
nello screening sincrono/accodato. La variante è stata rimossa comunque.

Nel punto 5, l'unroll aveva fatto contrarre al compilatore il prodotto opposto
nell'aggiornamento dell'accumulatore. Il SASS originale esegue prima
`acc * alpha`, poi FMA `value * beta + prodotto`. Imporre la medesima
sequenza con `__fmul_rn` e `__fmaf_rn` risolve la divergenza; non migliora
abbastanza i tempi. La variante è stata rimossa.

Fonti e trasferibilità: l'audit fornito indica il
[GEMV con residuo ExLlamaV3](https://github.com/turboderp-org/exllamav3/blob/499890c75d20d8e7c9d061f37189ae611a5c9f0b/exllamav3/exllamav3_ext/quant/exl3_gemv_int8_kernel.cuh),
il [catch-up MTP di llama.cpp](https://github.com/ggml-org/llama.cpp/blob/4d9176092d00586775af140581bb0b558ddc4389/common/speculative.cpp),
MMVQ e attention tiled dello stesso engine. Sono state lette le schede locali
`docs/research/platforms/{exllamav3,llama-cpp}.md`, la mappa problemi e la
scheda del fork RTX 3090. Si trasferisce il principio di riuso e scheduling;
formati EXL3, KV half e diverso ordine delle riduzioni non sono stati importati.

## Artefatti locali

Directory sotto `performance-results/`:

- `iq4-xs-r8-reuse-20260909`: baseline sorgente/binari, SASS, risorse,
  primitive CUDA PASS e nvdisasm 12.4.127 estratto localmente;
- `mtp-catchup-20260909`: patch sperimentale accodata, probe, rigetto e
  correzione mantenuta, output CLI, benchmark residenti, regressioni e checksum;
- `suffix-attention-20260909`: variante GEMM e differenza numerica;
- `r8-rows2-20260909`: microbenchmark, parità e risorse;
- `attention-tile-20260909`: entrambe le tile, SASS, divergenza iniziale,
  correzione FMA, parità e tempi finali.

I tempi dei microbenchmark sono misurati fuori dal profiler. Nsight non era
disponibile: i conteggi statici SASS non sono misure di traffico DRAM.
Non sono stati modificati golden, tolleranze o manifest per promuovere un esperimento.

## Punto 6 — input R8 condiviso e fusione selettiva (16 settembre)

**REJECT per Qwen3.8: guadagno end-to-end sotto il 3%.** Provati separatamente
riuso del packing, poi riuso più gate/up/SwiGLU fusi solo per un token. I batch
2/3 conservano i kernel separati: la fusione li rallenta. Il prototipo usa
34.560 byte di storage per sessione, con descrittore locale alla proiezione;
Q8_0 alpha/beta e i percorsi diagnostici incompatibili mantengono il fallback.

Le 72 combinazioni di primitiva passano bit-exact per gate/up, SwiGLU e byte
impacchettati: IQ4_XS/IQ4_XS, IQ4_XS/Q4_K, Q4_K/Q4_K, Q5_K/Q5_K, 1–3 righe,
forme 256→7, 4352→7 e 5120→17408. Questo comprende lo scratch interposto;
non costituisce una certificazione completa del verifier MTP.

Sul target Qwen3.8, dopo warm-up e cinque coppie residenti, la variante
selettiva produce prompt e tutti i logits di 128 token consecutivi bit-exact.
Riduzione appaiata mediana del decode: **1,49% a 128**, CI95 bootstrap della
mediana `[-2,15; 1,93]%`; **1,70% a 2048**, CI95 `[1,16; 6,10]%`.
CV baseline/candidate rispettivamente `0,26/1,91%` e `2,06/0,85%`.
Il riuso isolato nello screening non dà una direzione migliore e stabile.
Nessuna variante entra nel runtime, nemmeno come flag permanente.

Fonte letta: fusione MMVQ in `cuda/mmq/mmvq.cu`, pin del vendor
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`, e scheda ExLlamaV3 locale.
Il criterio selettivo richiesto dall'utente è stato applicato, ma il vantaggio
di primitiva non basta per superare la soglia del punto sul target mantenuto.
I controlli estesi MTP e l'oracolo Qwen3.8 rimangono NOT_VERIFIED per questa
candidate scartata; non è stato necessario promuoverla per eseguire le prove.

Artefatti: `performance-results/r8-shared-input-20260916/`, con baseline,
generatori, probe, `primitive.jsonl`, `direction.jsonl`,
`fusion-session1.jsonl` e relativo riepilogo. I risultati iniziali Qwen3.6
restano solo come dati storici e non motivano alcuna modifica.

## Punto 7 — dequantizzazione cooperativa IQ4_XS

**KEEP e default su richiesta dell'utente**, che accetta per questo punto
la soglia del 3% invece del 5% iniziale. Due varianti, entrambe esatte: 32 lane per blocco di
256 pesi, caricamenti scalari cooperativi oppure word allineate da 32 bit.
Il test iniziale ha intercettato un mapping errato dei sottoblocchi nel
prototipo; corretto prima dei benchmark, mai inserito nel runtime.
Il mapping DS4 usa gruppi da 32 valori con nibble basso/alto separati da 16.

Parità F32 e F16 su 256, 4352, 52.428.800 e 89.128.960 valori. Sulla matrice
5120×17408 il tempo mediano F32 scende circa da 812 a 617 µs; in F16 da
786 a 569 µs con la variante vettoriale. Per le forme F16 più piccole la
variante scalare è migliore. La candidate seleziona per dtype e dimensione,
senza cambiare GEMM, soglie, precisioni o chunk e senza cache di pesi F16.

Con chunk 512, modello residente, warm-up e dieci coppie bilanciate in due
sessioni (tutti i campioni inclusi, nessuna esclusione a posteriori):

| Contesto | Prefill baseline/candidate, mediana | Riduzione appaiata | CI95 | CV baseline/candidate |
|---|---|---|---|---|
| 2048 | 3455,31 / 3342,75 ms | 3,46% | [3,26; 3,81]% | 1,96 / 1,27% |
| 8192 | 14954,43 / 14349,17 ms | 3,32% | [3,11; 4,18]% | 1,69 / 2,69% |

Prompt e tutti i logits di 32 token successivi passano bit-exact in ogni
coppia. Decode non modificato: riduzioni mediane −0,31% / +0,16%, CI95
[-1,17; -0,01]% / [-0,42; 1,67]%; CV sotto 3,77%. Nessun guadagno decode
dichiarato. Nella seconda sessione isolata il CI a 8K include zero;
la decisione usa le dieci coppie previste, senza ulteriori campioni.

Il binario di produzione passa anche il confronto scalare/cooperativo con
chunk predefinito 2048 ai confini 95/96/97, 127/128/129, 511/512/513,
2047/2048/2049 e a 8K/16K/28K/30K. Prompt e tutti i logits dei passi
successivi sono bit-exact (128 passi nei contesti lunghi, 8 nei brevi).
Con MTP passa a 513/2049/28K/30K: token greedy e logits finali identici.
Nessun OOM a 30K; nessuna nuova allocazione nel decoder. Queste prove lunghe
sono gate di correttezza, non conferme statistiche di velocità: con chunk
2048 il beneficio varia e il 3,3–3,5% non va generalizzato al default chunk.

La regressione permanente `tests/qwen_iq4_dequant_probe.cu` passa 12 casi
F32/F16, entrambe le varianti, ogni scala half finita e code di CTA.
Il flag diagnostico `DS4_CUDA_QWEN_NO_COOPERATIVE_DEQUANT=1` permette il
confronto con lo stesso binario. Build CUDA completa e oggetto CPU riusciti;
test Python 105 PASS e 3 SKIP. L'oracolo ufficiale completo Qwen3.8 e la
riqualificazione del ramo FP16 preesistente restano NOT_VERIFIED: la nuova
primitiva conserva esattamente gli input GEMM della baseline, non ne certifica
la qualità cross-engine. Prestazioni verificate sulla sola RTX 3090 sm_86.

Fonte: decoder locale `dev_iq4_xs_value`, layout/tabelle del vendor llama.cpp
e scheda della piattaforma; l'upstream fissato dall'audit non è stato scaricato.
Artefatti in `performance-results/audit-remaining-20260916/`: `dequant.cu`,
`prepare-dequant.py`, `dequant*.jsonl`, `dequant-confirmation.summary.json`,
`default-primitive.jsonl`, `default-validation.*` e `default-mtp-validation.*`.

## Punto 8 — attention prefill a tile senza score globali

**REJECT numerico.** Prototipo di fattibilità F32, una query per CTA e tile
di 64/128 chiavi; KV, causalità, scala 1/16 e gate sono conservati. QK usa
la riduzione CUDA locale, softmax online aggiorna massimo, denominatore e
accumulatore fra tile. Nessuna conversione a half. Il riferimento è la
primitiva GEMM di produzione dopo la normalizzazione/RoPE, con gli stessi
input congelati e prefisso zero: è il percorso prefill, distinto dal punto 5.

| Query | Tile chiavi | Valori diversi / totali | Max abs |
|---|---|---|---|
| 128 | 64 | 786406 / 786432 | 0,0000516125 |
| 128 | 128 | 786409 / 786432 | 0,0000516125 |
| 512 | 64 | 3145691 / 3145728 | 0,0000516125 |
| 512 | 128 | 3145697 / 3145728 | 0,0000516125 |

Il diverso ordine di QK, softmax e PV non conserva il contratto interno
bit-exact. Il piano impone lo stop a questo gate: nessun tuning o benchmark
TTFT per promuovere queste varianti. Non si conclude che ogni futura attention
a tile sia impossibile; queste due implementazioni non sono ammissibili.
Fonti consultate: scheda llama.cpp/fattn e kernel online già presente in DS4;
nessun port di un'implementazione con KV half. Artefatti locali:
`audit-remaining-20260916/attention.cu`, `attention.jsonl`, `attention.log`.

## Punto 9 — residuo e RMSNorm

**REJECT al controllo del beneficio recuperabile**, soglia 2% invariata.
Prototipo a 256 thread con somma F32 esplicita e lo stesso albero di riduzione
della RMSNorm originale. Residuo e norm passano bit-exact per 1/2/3/128/512
righe da 5120 elementi, incluse cancellazioni di valori grandi e pesi negativi.
La fonte concreta è il kernel GLM locale, adattato alla riduzione Qwen senza
importare la configurazione GLM a 1024 thread.

Cinque coppie di microbenchmark dopo warm-up: su una riga la mediana passa
da 16,23 µs (somma + norm) a 10,69 µs, circa 5,53 µs recuperati. Ripetuto
nei 64 layer, il primo sito può risparmiare circa 0,35 ms/token; aggiungere
il secondo sito del layer successivo porta la stima a circa 0,70 ms/token.
Rispetto ai 38–41 ms/token misurati, entrambe le stime restano sotto il 2%.
Il profilo a 2K attribuisce a tutte le somme residue circa 10,27 ms su
660,51 ms per 16 token (1,55%); a 28K sono 14,96 su 971,78 ms (1,54%).

Queste sono stime diagnostiche e microbenchmark, non speedup end-to-end.
Non è stato introdotto un nuovo dispatch né completato un gate modello per
una candidate sotto soglia. Artefatti: `audit-remaining-20260916/residual.cu`,
`residual.jsonl` e `profile-{2048,28672}.*`.

## Punto 10 — convoluzione causale parallela nel tempo

**REJECT prima dell'implementazione**, applicando il limite superiore
richiesto dall'audit. Con chunk 512 il profilo CUDA event della baseline
attribuisce alla convoluzione dei 48 layer:

| Contesto reale | Convoluzione | Prefill totale profilato | Quota |
|---|---|---|---|
| 2048 | 19,48 ms, 192 chiamate | 3844,69 ms | 0,507% |
| 28672 | 300,20 ms, 2688 chiamate | 62228,14 ms | 0,482% |

Anche eliminarla interamente non raggiunge il 3% sul prefill. La riduzione
del totale introdotta dal punto 7 non cambia questa conclusione. I tempi
sono diagnostici, non benchmark ufficiali di una candidate; nessuna variante
32/64 viene aggiunta e nessun buffer da 80 MiB viene allocato. Convoluzione
1–3 righe, snapshot, stato finale e ricorrenza GDN restano invariati.
Il confronto di riferimento è la convoluzione causale FLA indicata nell'audit;
la verifica del kernel DS4 conferma che il tempo è seriale ma non dominante.
Artefatti: `audit-remaining-20260916/profile-{2048,28672}.{jsonl,log}` e
`prepare-profile.py`; ogni campione usa il modello residente dopo warm-up.

## Punto B — workspace e durata dei buffer

**Chiuso senza nuova prenotazione.** Il profilo della baseline registra sei
crescite dello scratch a freddo: 105, 120, 175, 187, 200 e 340 MiB. Dopo il
warm-up, zero crescite sia durante prefill sia durante decode a 2K e 28K.
Non è quindi dimostrato un costo di riallocazione nella fase misurata.
Il decoder cooperativo del punto 7 usa lo stesso workspace e zero byte extra;
i gate a 28K/30K, chunk 2048 e MTP acceso/spento terminano senza OOM.
La capacità allocata è 30976, il prompt massimo 30720 e l'output 128 token.

Il graph distingue già scratch delle proiezioni (viste alternative ricorrente
e full-attention), KV/stato persistente e snapshot MTP. Prenotare nuovamente
il massimo duplicando questi buffer aumenterebbe solo il consumo. Il packing
R8 condiviso del punto 6 e il secondo stream del punto D non sono stati
promossi: non richiedono nuove regioni permanenti. Nessun guadagno percentuale
o picco VRAM hardware viene attribuito a questo punto; l'assenza di OOM non
equivale alla misura del picco comprensivo di driver/desktop.

Fonte: allocatori `cuda_tmp_alloc` / `cuda_tmp_alloc_on` e proprietà delle
viste in `ds4_qwen_gpu_graph`. Artefatti: `profile-{2048,28672}.log`
(`AUDIT_GROW measured=0` soltanto), `default*-validation.*`. La decisione
segue il criterio del piano di conservare solo separazioni indispensabili.

## Punto C — accodamento e attesa GPU

**REJECT; coincide con il pilota del punto 2.** Il prototipo separava enqueue
e completamento del catch-up MTP, conservando l'ordine sullo stesso stream e
l'attesa prima del consumo CPU. Mediana del catch-up su 2048 righe:
292,54 → 246,90 ms (−15,6%); le due coppie direction davano soltanto
circa 0,1% e 1,8% sul TTFT. Entrambe sono sotto anche la soglia del 3%
del punto C, oltre che sotto quelle più alte del punto 2.

Non si ripete lo stesso esperimento come nuovo beneficio. Il prototipo è
archiviato in `performance-results/mtp-catchup-20260909/`; le verifiche
interne dello screening non autorizzano una sostituzione globale delle
sincronizzazioni. Il controllo degli errori asincroni e i confini di attesa
di produzione restano quelli già validati. La correzione del carry è
mantenuta separatamente e non viene contata come vantaggio dell'accodamento.
Fonte: confini di comando DS4 e riferimento MTP llama.cpp del punto 2.

## Punto D — due stream per le proiezioni

**REJECT nello screening.** Prima misurato il costo eliminabile delle due
configurazioni, con il profiler delle singole proiezioni a 2K:

| Regione da sovrapporre | Prefill | Decode |
|---|---|---|
| alpha + beta | 30,91 / 3798,29 ms (0,814%) | 23,13 / 653,60 ms (3,54%) |
| K + V | 55,91 / 3798,29 ms (1,472%) | 7,55 / 653,60 ms (1,155%) |

Q contro K/V e alpha/beta del prefill non possono raggiungere il 3% anche
eliminando interamente quei tempi. Implementato quindi solo il pilota
alpha/beta del decode: secondo stream non bloccante, 5760 byte dedicati per
packing Q8 e scale, output distinti già presenti nel graph, evento dopo la norm
produttrice e attesa prima di GDN.
QKV e Z restano sullo stream principale; layer/stato ricorrente sono seriali.
Il secondo stream usa soltanto kernel con stream esplicito, senza handle
cuBLAS condivisi né riuso dello scratch temporaneo globale.

Con dequantizzazione cooperativa già attiva, warm-up e due coppie bilanciate
direction, i logits del prompt e tutti i logits di 128 token sono bit-exact.
Riduzione mediana decode: 2,05% a contesto 128 e −0,045% a 2048; la seconda
forma non guadagna e nessuna mediana supera il 3%. Il prefill, non oggetto
del prototipo, varia di −1,30% / −2,38% in questo screening.
Due coppie non promuovono una candidate: stop, senza suite slow né nuovi
stream nel runtime. Overlap nella timeline, MTP/restore/cancellazione del
prototipo restano NOT_VERIFIED e non sono presentati come PASS.

Fonti: dipendenze di `qwen_graph_recurrent_attention_rows`, proiezioni CUDA
Q8_0 esistenti e contratto degli eventi producer/consumer. Artefatti:
`audit-remaining-20260916/projections*`, `prepare-streams.py`, `streams*`.
