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
