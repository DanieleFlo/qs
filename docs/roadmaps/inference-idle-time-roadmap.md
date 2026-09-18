# Roadmap: tempi morti CPU/GPU nell'inferenza Qwen

## Obiettivo e invarianti

Migliorare il tempo end-to-end a contesti diversi tramite scheduling, senza
cambiare kernel matematici, quantizzazione, logits, ordine RNG o semantica HTTP.
Baseline sorgenti: e21968e3e94878ef9de55ec9e7a74bb77bedea66 (main pulito).
Hardware locale: RTX 3090 24 GiB, WSL2, CUDA 12.4, driver 610.43.02.
Target esclusivo richiesto: Qwen3.8 UD-Q4_K_S con sidecar MTP Qwen3.8.
Qwen3.6 escluso dai benchmark e dalla matrice di validazione.
Il messaggio di partenza usa il prefisso qs/: in questo checkout i file sono
nella radice. `todo.md` citato dall'indice non esiste: usare le roadmap in docs.

## Metodo e criteri

Ogni esperimento registra ipotesi, modifica, comandi, artefatti, correttezza,
prestazioni e verdetto KEEP / REJECT / NEED_MORE_DATA / NOT_VERIFIED.
Prima una direzione breve, poi warm-up e cinque ripetizioni per promuovere.
Confrontare stesso modello, prompt, token generati, contesto, build e GPU;
riportare dispersione oltre alla mediana. Un beneficio inferiore al rumore non
basta. Target: almeno 2-3% stabile E2E oppure beneficio sostanziale constrained
/ streaming. Un solo processo modello alla volta. Misurare contesti corti,
medi e lunghi entro il budget VRAM. Nessun test saltato viene dichiarato PASS.
Non collegarsi agli host remoti: Metal/ROCm/distribuito non disponibili localmente.
Commit dopo ogni punto completato e verificato, push finale su origin.

## Sequenza

- [ ] 0. Congelare baseline, leggere harness e profilare testo / JSON / tool,
  streaming e sampled MTP. Verificare build e gate preesistenti.
- [ ] 1. Separare begin/finish nel server batched e sovrapporre postprocessing
  CPU; drain obbligatorio prima di invalidazione, riuso o fine richiesta.
  Conservare commit dell'output solo dopo eval riuscita.
- [ ] 2. Verificare preparazione anticipata constraint: distinguere stato
  grammaticale dai dati dipendenti dai logits; non leggere sessione in flight.
  Misurare CPU esposta e realmente sovrapposta, senza contare due volte le fasi.
- [ ] 3. A/B del flag CUDA stream-sync esistente; poi variante di readback
  pinned asincrono se la diagnosi ne giustifica il costo. Audit SSD/multi-GPU
  e lifetime. Niente sostituzione globale non dimostrata delle barriere.
- [ ] 4. Valutare SSE producer/consumer bounded: client rapido/lento,
  cancellazione, backpressure, errori e ordine di invio. Verificare se il
  socket costituisce davvero un collo di bottiglia nel workload misurato.
- [ ] 5. Sampled MTP: bulk verifier-row readback e, se utile, pipeline per riga;
  confrontare acceptance alta/bassa, identita RNG/output, contesti diversi.
- [ ] 6. Provare altre riduzioni di attese emerse dai profili; confermare
  candidati mantenuti, aggiornare ledger/indici, commit e push finale.

## Registro

### Preparazione

- Letto integralmente il messaggio dell'agente e ispezionati i percorsi server.
- Shell e Node nel sandbox non avviabili (`setup refresh had errors`).
  Letture via PowerShell/WSL fuori dal sandbox riuscite con auto-review.
- `rg` assente in WSL: fallback a grep/find. CUDA presente in /usr/local/cuda.
- Nessuna modifica al codice e nessun benchmark ancora eseguito.

### 0. Baseline e gate iniziali (completato)

Build esplicita: `make -j2 ds4 ds4-server ds4-bench ds4_test ds4_server_test CUDA_ARCH=sm_86`.
Su Linux `make` senza target stampa l'help: non costituisce build.
Binario baseline congelato: `performance-results/idle-time/ds4-server-baseline`,
SHA256 `14b0671747ea6022b2a40617429445a794caf0352a313eee4c991f83a5ad6d4d`.
Modello SHA256 `75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017`.

Artefatti: `performance-results/idle-baseline-curve/experiment.json`,
`idle-baseline-constrained/experiment.json`, `idle-time/*baseline*.log`.
Curva direzionale: 64 token, due ripetizioni, contesti effettivi 2048/8192/16384,
allocazione 16449, prefill chunk 2048, no thinking, prompt technical-explanation.
A 16K: mediana 21.30 tok/s; seconda misura 3090.905 ms di decode,
3084.687 ms eval, 5.545 ms sampling, 0.673 ms residuo.
Nel tool richiesto: 23.48 tok/s e 879.54 ms CPU constraint; JSON: 31.29 tok/s,
9.76 ms CPU constraint. Sono misure direzionali, non una promozione.

Nota provenienza: l'harness curva calcola l'hash del binario alla fine del run;
il processo baseline aveva gia caricato l'eseguibile originale quando il link
successivo ha sostituito ds4-server. Il campo binary_sha256 della prima curva
indica erroneamente il prototipo begin (2cb894c...). L'identita effettivamente
eseguita e quella del baseline congelato sopra. Non usare quel report per una
promozione automatica; i confronti successivi usano solo binari congelati.

Gate: ds4_server_test PASS; ds4_test --server --constraint-trie
--qwen35-layer-pattern PASS (quest'ultimo nome storico verifica il layout).
92 test Python: 89 PASS, 3 SKIP, dopo ripetizione a macchina libera.
Il primo tentativo aveva tre FAIL di compatibilita dovuti al lock del server
ancora attivo, non a un difetto funzionale: log conservato e rerun passato.
Non avviati benchmark Qwen3.6. Le fixture di rifiuto sidecar errato verificano
Qwen3.8 senza eseguire inferenza Qwen3.6.

### 1. Begin/finish + postprocessing (completato, REJECT standalone)

Implementato prototipo batched con enqueue separato da wait. Detokenizzazione,
append, thinking e tracker JSON/DSML eseguiti durante eval; output/trace dopo
successo, drain prima di qualunque uscita; testo ripristinato su errore.
Build senza warning e ds4_server_test PASS.
Confronto direzionale identico --batched-session 1, context 4096, warm-up 1,
ripetizioni 2, suite constrained direction:

| Workload | Baseline tok/s | Begin/finish tok/s |
| --- | ---: | ---: |
| DSML tool | 21.39 | 20.57 |
| JSON nested | 27.52 | 27.63 |

Output semantici conservati. Nessun beneficio convincente: non promuovere e
non spendere una suite slow su questa variante. La curva baseline dimostra
che il postprocessing puro disponibile e troppo piccolo. Artefatti:
`idle-batched-baseline`, `idle-batched-begin`; binari in `idle-time`.
Il supporto begin/finish viene riutilizzato soltanto per il successivo prototipo.

### 2a. Lookahead dei constraint engine-only (completato, REJECT)

L'analisi corrente non e CPU-only rispetto alla sessione: legge logits e
scrive sample_masked, sample_allowed e seriale dell'analisi. Non e sicuro
spostarla semplicemente fra begin e finish.
Provata una variante conservativa: anticipare build_constrained_forced_tokens
soltanto quando tutte le analisi restano nel vocabolario immutabile dell'engine;
se serve un'analisi session-scoped, annullare il lookahead e usare il percorso
normale. Niente logits letti durante eval, niente RNG anticipato.
Build senza warning, ds4_server_test PASS; output dei due workload invariati.
DSML 21.28 tok/s, JSON 27.61 tok/s contro 21.39/27.52 baseline.
Solo ~31.47 ms di CPU riutilizzati su ~893 ms constraint nel tool, 0 nel JSON:
molto meno del 70-90% atteso. `lookahead_cpu_reused` NON misura overlap GPU
reale: puo includere CPU dopo completamento GPU e non viene spacciato per tale.
I contatori di fase del prototipo includono doppio conteggio eval/lookahead;
non usare il residuo come prova prestazionale. Usare il wall time.
Patch e riproduttori conservati in `performance-results/idle-time`;
nessuna di queste modifiche sperimentali e nel codice di produzione.

### 6a. Provenienza dei benchmark server (completato, KEEP correctness)

Durante il primo run e emerso che server-curve/constrained-server calcolavano
l'hash dell'eseguibile dopo la fine della misura. Un link concorrente poteva
attribuire i risultati del processo originale al nuovo binario. Ora entrambi
congelano l'hash prima del run e rifiutano il report se il file e cambiato.
Test di regressione: sostituzione atomica del binario fra lettura iniziale e
validazione finale; il report deve essere rifiutato. Suite test_perf_harness PASS.
Questa e una correzione dell'affidabilita delle misure, non uno speedup.
Tutti i nuovi esperimenti locali usano copie immutabili degli eseguibili.

### 3. Readback pinned: gate numerico intermedio

Prototipo CUDA con un buffer pinned persistente, stream di copia nonblocking,
evento sullo stream produttore e attesa dell'evento di fine copia. Per il decode
ordinario basta un buffer: il sampling precedente e terminato prima del riuso.
Non introdotto double buffering senza un consumatore simultaneo reale.
Matematica dei kernel invariata; solo il confine finale Qwen usa la variante.
32 righe full-vocabulary da 248320 float a ciascuno dei contesti 128/2048/8192/
16384: zero differenze bit-exact e zero valori non finiti (128 righe totali).
Probe locale: idle-time/readback_probe.c; log readback-correctness.log.
Il primo tentativo di compilare il probe usava ids anziche v per ds4_tokens:
corretto prima di eseguire il test. Build del candidato senza warning.
Conferma end-to-end ancora in corso: nessuna promozione.

### 4. SSE bounded: gate trasporto intermedio

Prototipo con writer request-scoped e limite 256 KiB; send/poll nel consumer,
ordine FIFO e drain prima della risposta finale. Test diretto di oltre 768 KiB
su socketpair con buffer piccolo: ordine e completezza PASS, backpressure PASS.
Disconnessione e distruzione del writer PASS. Il primo test di disconnessione
terminava per SIGPIPE: il runner di test non installava l'handler del server;
allineato il test all'handler reale e ripetuto con successo. Nessun cambiamento
alla gestione SIGPIPE di produzione. Beneficio HTTP ancora da misurare.

### 3. Conferma stream-sync/pinned (completato, REJECT)

`idle-readback-confirm/results.json`: stesso binario congelato
`ds4-server-readback`, contesti 2048/8192/16384, 64 token, warm-up per workload,
cinque campioni misurati, seed 424242. Output identico in tutte le varianti.
Tempi di decode misurati dal server, non TTFT/prefill inclusi nel tempo HTTP.

| Variante | 2K tok/s | 8K tok/s | 16K tok/s | CV 2K/8K/16K |
| --- | ---: | ---: | ---: | --- |
| Baseline | 30.337 | 27.267 | 22.953 | 0.25% / 0.24% / 5.31% |
| Stream-sync | 28.669 | 23.979 | 21.591 | 2.71% / 7.09% / 2.21% |
| Pinned/event | 26.739 | 24.714 | 21.091 | 0.92% / 0.19% / 2.77% |

I piccoli vantaggi direzionali non sono confermati. Alcune misure superano il
5% CV e nel corso del run sono state compilate altre prove CPU: queste misure
NON dimostrano causalmente un rallentamento della nuova API. Sono sufficienti
per rifiutare la promozione: nessun miglioramento stabile provato. Non affermare
che pinned e intrinsecamente piu lento. Il gate bit-exact rimane PASS.
Il processo gia attivo e stato lasciato completare come richiesto dall'utente.
Nessuna barriera globale viene sostituita nel codice di produzione.
La primitiva sperimentale resta nei binari congelati per le prove sampled MTP;
il codice definitivo verra ripulito dopo l'ultimo verdetto.

### 4. SSE asincrono (completato, REJECT per questi workload)

Risultati `idle-sse-fast/results.json` e `idle-sse-slow/results.json`:
Qwen3.8 target-only, 64 token, seed fisso, warm-up 1 e due ripetizioni;
stesso binario con writer abilitato/disabilitato. Output byte-identico PASS,
marker finale SSE presente. Test coda/ordine/disconnessione sopra: PASS.

| Client e contesto | Baseline decode tok/s | Writer decode tok/s |
| --- | ---: | ---: |
| Rapido, 128 | 28.220 | 26.855 |
| Rapido, 8192 | 23.436 | 23.581 |
| Lento, 128 | 27.066 | 26.996 |

Client lento: SO_RCVBUF 1024, server SO_SNDBUF 4096, letture da 128 byte con
30 ms di ritardo. Tempo HTTP mediano completo: 6834.70 ms baseline,
7073.76 ms writer. La metrica server include il drain della coda, evitando di
presentare l'accumulo in memoria come miglioramento end-to-end.
Nessun vantaggio significativo in questi casi: non mantenere thread, TLS,
coda e nuove condizioni di errore senza un beneficio misurato. Il risultato
non esclude utilita con client bloccati o WAN; quella non e una dimostrazione
di speedup per questo target locale. Il socket possiede gia buffering e un
timeout di stallo di 2 secondi. Non introdotta una modalita di flush a due
token, dato il residuo trascurabile misurato e l'assenza di un collo di bottiglia.

Fonti usate nell'audit delle dipendenze CUDA: documentazione NVIDIA
[stream synchronization](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)
e [event management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html).
Lo stream nonblocking non sincronizza implicitamente con quello legacy:
il prototipo usa quindi eventi produttore/consumatore espliciti.

### 2b. Revisione constraint: callback dopo submission (gate intermedi)

Dopo il rifiuto del lookahead engine-only, provata una sola revisione piu ampia:
callback CPU dopo submission Qwen e prima del readback. Prepara il prossimo
stato grammaticale usando i vecchi logits host, quindi riapplica la maschera
ai logits nuovi prima del sampling. Nessun thread aggiunto e nessun cambio RNG.
Variante diagnostica `DS4_SERVER_CONSTRAINT_LOOKAHEAD=1`, solo server unbatched.
Binario congelato `ds4-server-callback`, SHA256
`ea8c9c3d569163efee690b02ac40cd68b2b4166c47b0d012c16df1cb6b4f8c4a`.

Direzione (`idle-callback-baseline` / `idle-callback-candidate`): tool
21.49 -> 21.78 tok/s, JSON 27.20 -> 28.61 tok/s (baseline JSON instabile).
`idle-callback-oracle`: confronto esaustivo dell'analisi, zero divergenze,
schema valido e output identico ai due casi baseline. Build e test server PASS.
Circa 590 ms di preparazione tool sono eseguiti dentro eval, ma NON sono una
misura di CPU realmente nascosta: le fasi diagnostiche si sovrappongono.
Usare esclusivamente decode wall e tempo HTTP per il confronto prestazionale.

Conferma 1 (`idle-callback-confirm-*`), warm-up 1 e cinque campioni:

| Workload | Baseline tok/s | Callback tok/s | CV baseline / callback |
| --- | ---: | ---: | --- |
| Tool required enum/const | 21.785 | 22.494 | 2.36% / 3.83% |
| JSON nested required array | 28.596 | 30.403 | 2.06% / 5.57% |

Output SHA256 identici e schema valido. Il +3.25% tool e confrontabile con la
dispersione e il JSON e instabile: nessuna promozione. In corso una seconda
serie a ordine invertito, senza compilazioni concomitanti. Il prototipo richiede
anche un cleanup del rollback dello stato testuale su errore eval e metriche non
additive esplicite prima di un'eventuale promozione; non e codice release.

### 5. MTP sampled: bulk e segmentazione (gate intermedi)

Variante CUDA circoscritta al verifier sampled: staging pinned persistente,
evento produttore sullo stream compute, copie su stream nonblocking e attesa
per riga. Nessuna modifica dei kernel o dell'ordine del sampling. Bulk e
segmentato sono confrontati con lo stesso binario e flag disabilitati.

Prima direzione `idle-mtp-readback-direction`: bulk 32.549 / 32.718 / 29.659
tok/s contro baseline 33.254 / 34.484 / 29.861 a 128 / 2048 / 8192 token.
Il primo prototipo segmentato conservava una sincronizzazione globale dopo
l'accodamento delle copie: corretto questo limite prima di giudicare l'idea.
Le letture dei logits/top-index drenano lo stream compute; gli eventi per riga
proteggono separatamente le copie. Il cleanup drena prima di liberare tensor e
staging. Un'eventuale release deve mantenere esplicita anche la dipendenza prima
del riuso del buffer verifier, senza affidarsi alle sincronizzazioni del catchup.

Seconda direzione `idle-mtp-real-segment`, temperature 0.7, seed 424242,
64 token, warm-up 1 + due campioni, output identico in tutte le varianti:

| Contesto | Baseline tok/s | Bulk tok/s | Segmentato tok/s |
| --- | ---: | ---: | ---: |
| 128 | 31.952 | 32.688 | 35.070 |
| 2048 | 36.152 | 36.217 | 37.472 |

Gate `DS4_TEST_QWEN_MTP_PATHS=1 DS4_QWEN_BULK_VERIFIER=1
DS4_READBACK_SEGMENTED=1 ./ds4_test --mtp-verify-depth` con modello e sidecar
Qwen3.8 espliciti: PASS. Verifica forced reject, partial, raw-copy/full,
128 token sampled con seed fisso identici al target; logits prompt MTP
abilitato/disabilitato bit-exact su 248320 float. Log `mtp-depth-segmented.log`.
Il vantaggio direzionale richiede conferma: accodata serie a ordine invertito
su contesti 128/2048/8192/16384, warm-up 1 + cinque campioni. Nessun KEEP ancora.
### 2b. Conferma a ordine invertito e candidato ripulito

`idle-callback-reverse-candidate` eseguito prima di
`idle-callback-reverse-baseline`, senza compilazioni o altri modelli concorrenti,
warm-up 1 e cinque campioni. Output identico ai precedenti run.

| Workload | Baseline tok/s | Callback tok/s | CV baseline / callback | HTTP baseline / callback ms |
| --- | ---: | ---: | --- | --- |
| Tool | 20.560 | 21.372 | 1.78% / 1.14% | 7700.671 / 7452.137 |
| JSON | 27.764 | 27.929 | 0.59% / 0.76% | 2175.532 / 2170.402 |

Il +3.95% tool (+3.25% nella prima conferma) giustifica la pulizia e i gate
release; il +0.59% JSON non dimostra un miglioramento pratico. Questi sono
prompt brevi con capacita sessione 4096: NON chiamarli prompt lunghi 4K.

Il candidato ripulito prepara su copie private di testo/thinking/tracker/lexer;
il postprocessing originale commette solo dopo eval riuscita. Riutilizza il
controllo di cancellazione e il mutex gia presenti. Nessun callback GPU su
Metal/ROCm/CPU/distribuito: li il callback viene saltato e la preparazione
resta nel punto originale della successiva iterazione. Il batching
mantiene il percorso esistente. `DS4_SERVER_NO_CONSTRAINT_LOOKAHEAD=1` e un
interruttore diagnostico per confrontare lo stesso eseguibile, non una variante
semantica. La somma delle fasi sottrae il lavoro preparato gia incluso in eval;
`prepared_cpu` non rappresenta overlap GPU misurato. I campi legacy exposed /
overlapped restano una stima conservativa (nessun overlap attribuito).

Aggiunti gate per maschera riapplicata ai nuovi logits, token disabilitato con
argmax alto, cambi della disponibilita di valori finiti, analisi scaduta,
consumo RNG e cancellazione prima del callback. Preparato probe full-vocabulary
su 128/2048/8192/16384: verifica callback eseguita una volta, host logits vecchi
intatti durante submission, logits finali bit-exact e sampling mascherato
identico preparando prima/dopo eval. Build, probe e suite slow accodati solo
dopo la fine dei benchmark MTP, per non contaminarli.
### 5. Conferma MTP sampled (completato, REJECT)

`idle-mtp-segment-confirm/results.json`, stesso binario congelato
`ds4-server-real-segment`, ordine segmentato -> baseline, temperature 0.7,
64 token, warm-up 1 + cinque campioni per contesto, nessuna compilazione
concomitante. Identita degli output: PASS.

| Contesto | Baseline tok/s | Segmentato tok/s | CV baseline / segmentato |
| --- | ---: | ---: | --- |
| 128 | 32.090 | 32.010 | 0.58% / 1.06% |
| 2048 | 33.352 | 33.150 | 3.61% / 5.44% |
| 8192 | 29.641 | 30.106 | 3.31% / 0.32% |
| 16384 | 23.295 | 23.177 | 1.00% / 0.87% |

Il +1.57% a 8K non supera la soglia pratica e la dispersione della baseline;
negli altri contesti il vantaggio direzionale scompare. Nessuna promozione,
nonostante i gate numerici verdi. Eliminati readback pinned, eventi e stream
aggiunti nel prototipo: ds4_cuda.cu e ds4_gpu.h tornano identici alla baseline.
La copertura forced reject/partial/full dimostra correttezza nei percorsi testati;
non e una misura prestazionale distinta per ogni regime di acceptance.
Patch e binari degli esperimenti restano in performance-results/idle-time/.

### 2b. Gate del candidato ripulito (completato, REJECT)

Build, test server e nuovi test maschera/RNG/cancellazione: PASS.
Probe callback full-vocabulary: 32 passi a ciascuno dei contesti
128/2048/8192/16384, 248320 logits per passo, zero differenze bit-exact,
zero valori non finiti, sampling mascherato identico. Callback una volta per
passo e vecchi logits host invariati durante la preparazione: PASS.
Log `callback-correctness.log`; patch locale `clean-callback.patch`.
La build segnalava un wrapper statico divenuto inutilizzato: rimosso durante
la review. Il binario congelato misurato precede questa rimozione senza effetti
sul percorso eseguito. Non promuovere quel warning a nuova baseline release.

Suite **slow**, warm-up 1 + cinque campioni, stesso eseguibile congelato e
interruttore diagnostico, senza compilazioni concorrenti:
`idle-callback-clean-baseline` / `idle-callback-clean-candidate`.

| Workload | Baseline tok/s | Callback tok/s | CV baseline / callback | HTTP baseline / callback ms |
| --- | ---: | ---: | --- | --- |
| Tool | 21.006 | 21.404 | 0.51% / 0.45% | 7583.919 / 7500.138 |
| JSON | 27.092 | 27.612 | 1.61% / 0.86% | 2225.318 / 2182.735 |

Output identico, schema valido, zero divergenze: PASS. Il comparatore non trova
incompatibilita di misurazione, ma emette NEED_MORE_DATA: miglioramento medio
1.907%, sotto la soglia pratica 2%. Il tool guadagna 1.90% nel decode e 1.10%
nel tempo HTTP completo. Non si nega il piccolo effetto misurato: si rifiuta
la promozione delle nuove API e dello stato aggiuntivo dopo l'unica revisione
seria prevista, perche il beneficio finale non raggiunge la soglia 2-3%.
I precedenti +3-4% non sono il risultato del codice finale e non vanno pubblicizzati.
Ripristinati ds4.c, ds4.h, ds4_server.c e tests/ds4_test.c dalla baseline.
La matrice live aggiuntiva era stata predisposta ma non e stata eseguita dopo
il rifiuto prestazionale: NOT_VERIFIED per quel prototipo, non PASS implicito.
Nessuna nuova variante diagnostica o API di inferenza rimane in produzione.
