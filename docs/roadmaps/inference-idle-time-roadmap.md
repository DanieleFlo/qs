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
