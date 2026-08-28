# TODO — Prestazioni del server agentico e constrained decoding

Ultimo aggiornamento: 2026-08-28

## Scopo e vincoli

Questo diario separa il lavoro sulle prestazioni del server agentico dal lavoro
sui kernel CUDA. In questo ciclo **non si modificano i kernel CUDA**: si misurano
e si ottimizzano orchestrazione del server, sampling vincolato DSML/JSON e
sincronizzazione dei piccoli suffissi con la sessione live.

Vincoli non negoziabili:

- preservare esattamente validità DSML/JSON Schema, semantica del sampling,
  `tool_choice`, protezione dei marker riservati e contabilità dei token;
- misurare prima di modificare e registrare anche i tentativi scartati;
- cambiare una sola variabile per esperimento;
- non attribuire a inferenza GPU il tempo CPU di filtro, sampling, streaming o
  sincronizzazione;
- nessuna modifica al percorso dei piccoli suffissi prima dell'audit storico e
  della matrice di crossover descritti nel punto 3;
- un risultato prestazionale è `KEEP` solo dopo i gate di correttezza e una
  misura ripetuta stabile; altrimenti è `REJECT` o `NEED_MORE_DATA`.

## Stato sintetico

- [x] Separato questo TODO dal piano di correttezza del constrained decoding.
- [x] Ricostruita la prima baseline dai log forniti il 2026-08-12.
- [x] Individuato il costo algoritmico dominante candidato nel sampling DSML.
- [x] Completata la prima ricerca su fonti primarie per il punto 2.
- [x] Aggiunta telemetria di fase opt-in e prodotta una baseline ripetibile.
- [x] Completato il primo intervento 2A: cache dei piece e workspace persistenti.
- [x] Completato l'audit storico e misurato il crossover rilevante del punto 3.
- [x] Ottimizzato il prefill live di 2--127 token senza toccare i kernel CUDA.
- [x] Completati i gate finali di regressione e la misura end-to-end del server.
- [x] Reso Qwen3.8 il riferimento operativo e congelate curve fino a 22K.
- [x] Ottimizzato lo stato DSML opzionale `SEARCH` con partizione statica sicura.
- [x] Aggiunto il test live della storia lunga, target-only e MTP.
- [x] Aggiunta la matrice live temperatura zero/nonzero, thinking on/off e
  target-only/MTP su Chat e Responses agentiche.
- [x] Integrato MTP grammar-aware per JSON e tool opzionali, con gate adattivo
  target-only sul tratto DSML required in cui la speculazione regredisce.
- [x] Verificato Agent SSD in entrambe le direzioni target-only/MTP e corretto
  il buffer `fmemopen` del resident frontier host.

## Baseline iniziale ricavata dai log

Questi numeri sono osservazioni preliminari, non ancora un benchmark controllato.
Servono a formulare ipotesi e a dimensionare gli strumenti di misura.

| Scenario osservato | Misura | Interpretazione iniziale |
|---|---:|---|
| Server agentico, contesto circa 10.7k–15.3k | circa 8.9–14.0 tok/s, tipicamente 10–12 | Il timer del server include più della sola eval GPU. |
| `ds4-bench`, circa 12k | mediana 28.91 tok/s | Riferimento della sola generazione steady-state, CV 1.92%. |
| `ds4-bench`, 16k | mediana 27.08 tok/s | Riferimento corretto a 16k, CV 0.80%; non 29 tok/s. |
| Tempo server per token, campioni tipici | circa 75–113 ms | Superiore ai circa 35–37 ms attesi dalla eval alla stessa lunghezza. |
| Residuo preliminare non spiegato dalla eval | circa 39–77 ms/token | Compatibile con filtro/sampling CPU e lavoro server; da separare con timer interni. |
| Prefill di suffissi 5/42/123 token | circa 6.6/9.1/9.3 tok/s | Forte anomalia per suffissi piccoli; non va corretta senza audit del dispatch. |
| Prefill di suffissi 152–265 token | circa 45–52 tok/s | Indica un crossover o un percorso diverso. |
| Prefill di suffissi circa 988–1308 token | circa 470–563 tok/s | Il percorso layer-major lungo è molto più efficiente. |

Fatti di implementazione già verificati:

- `ds4_session_sample_filtered()` visita tutti i 248,320 token del vocabolario a
  ogni token campionato, decodifica il piece, chiama il filtro e crea una copia
  mascherata dei logits;
- `agentic_filter_token()` può allocare e costruire `history + piece` per ogni
  candidato del vocabolario;
- `ds4_engine_forced_prefix_token()` ripete una scansione completa del
  vocabolario mentre cerca un prefisso deterministico comune;
- il throughput server è cronometrato attorno a filtro, sampling, fast-forward,
  eval e post-processing; non è direttamente confrontabile con il solo tempo
  steady-state di `ds4-bench`;
- il dispatch Qwen storico usava `prompt->len < 512`, cioè la lunghezza totale
  del prompt, non la lunghezza del suffisso da sincronizzare. Le misure live
  hanno confermato che questo penalizzava i piccoli suffissi di una sessione
  lunga; il criterio è ora `prompt->len < 512 || suffix_len < 128`.

### Baseline controllata del server

Payload fisso: Qwen3.6-27B Q4_K_S, contesto 32768, tool
`record_decision`, `tool_choice=required`, temperatura 0, thinking disabilitato.
Tutte le run hanno prodotto esattamente `{"decision":"approve"}`, 53 token di
completamento e 13 token sincronizzati tramite fast-forward.

| Fase, 5 run | Baseline | Dopo cache/workspace 2A | Variazione |
|---|---:|---:|---:|
| Decode wall | 4.242--4.419 s | 2.362--2.379 s | circa -44% |
| Costruzione forced prefix | 0.756--0.775 s | 0.290--0.292 s | circa -62% |
| Filtro candidati | 1.667--1.690 s | 0.264--0.266 s | circa -84% |
| Eval | 1.354--1.442 s | 1.358--1.371 s | invariata |
| Throughput decode, mediana | circa 12.47 tok/s | circa 22.32 tok/s | circa +79% |

La baseline ha contato 9,932,800 candidati esaminati per risposta, solo 994
accettati e 73,846,280 byte di piece decodificati. La GPU/eval non era il collo
di bottiglia dominante: filtro e forced-prefix assorbivano circa 2.45 s.

### Confronto Qwen3.8/Qwen3.6 dopo la correzione ChatML

Il confronto del 2026-08-25 usa lo stesso binario, RTX 3090, contesto 32768,
modalita' trie, un warm-up e cinque ripetizioni misurate. Entrambi gli artifact
producono output semantico corretto e deterministico. Il workload DSML richiesto
genera 76 token in entrambi i modelli.

| Mediana, DSML required | Qwen3.8 UD-Q4_K_S | Qwen3.6 Q4_K_S | Lettura |
|---|---:|---:|---|
| Sampling mask build | 355,826 ms | 357,851 ms | 3.8 circa 0,57% piu' rapido |
| Constraint CPU totale | 1.004,302 ms | 999,680 ms | parita' pratica; 3.8 +0,46% |
| Eval modello | 2.144,873 ms | 1.757,448 ms | 3.8 +22,0% |
| Decode wall | 3.982,537 ms | 3.439,630 ms | 3.8 +15,8% |
| Output throughput | 19,083 tok/s | 22,095 tok/s | 3.8 -13,6% |

Sul workload JSON Schema annidato il constraint CPU e' ancora in parita'
(11,338 contro 11,318 ms), mentre il throughput e' 25,286 contro 30,864 tok/s.
Quindi il requisito di parita' del masking e' soddisfatto e la costruzione della
maschera DSML e' marginalmente piu' rapida sul 3.8; non e' invece corretto
dichiarare il modello 3.8 piu' veloce end-to-end. Il gap misurato e' dominato
dall'eval dei pesi UD.

Artifact:

- `performance-results/qwen38-chatml-dsml-trie-20260825-002/experiment.json`;
- `performance-results/qwen36-chatml-dsml-trie-20260825/experiment.json`.

### Ciclo Qwen3.8 del 2026-08-27: storia agentica senza tool call

Il nuovo fixture usa il vero `DeepAgent` `bootstrap-wiki`, tool disponibili con
scelta automatica e `reasoning.effort=none`. Il prompt occupa 10.814 token e la
risposta deterministica contiene 432 token e 298 parole, termina con `stop`, usa
una sola richiesta modello e non emette tool call.

| Mediana, 1 warm-up + 2 run | Baseline | Partizione `SEARCH` | Variazione |
|---|---:|---:|---:|
| Target-only decode | 11,739 tok/s | 23,457 tok/s | +99,82% |
| Target-only masking | 17.293,25 ms | 136,34 ms | -99,21% |
| MTP caricato, decode constrained target-only | 11,524 tok/s | 23,417 tok/s | +103,21% |

Gli hash delle risposte candidate coincidono con le rispettive baseline. Ogni
run usa la partizione in tutti i 433 passi `SEARCH`, con zero fallback; stop,
thinking control, token vuoti e piece contenenti `<` restano dinamici. In
modalità `compare_new_vs_oracle` la storia completa ha prodotto 433 confronti
full-vocabulary e zero divergenze. DSML obbligatorio e JSON Schema annidato
hanno superato lo stesso gate strutturale.

Con MTP caricato, il nuovo contatore riporta 433 passi constrained target-only:
lo speculative decoding non è stato modificato né usato per rivendicare il
guadagno. I checkpoint post-risposta sono ricostruiti da `memory-request` in
VRAM senza rilettura SSD nella finestra di pubblicazione.

Questa misura descrive il primo candidato `SEARCH`. Il successivo commit
grammar-aware `4cfcb76` abilita invece il verifier constrained quando porta un
vantaggio: sulla stessa storia lunga passa da 23,756 a 26,882 tok/s (+13,16%);
su JSON Schema annidato la mediana passa da 29,285 a 37,317 tok/s (+27,43%, CV
6,50%). Il tratto required senza thinking resta target-only e a parità, mentre
required con thinking usa MTP soltanto nel reasoning e guadagna 9,97--10,60%.
Output/hash, schema e contratto pubblico restano invariati e i fallback sono
zero.

Dettagli e artifact sono in
`docs/performance/qwen38-agent-search-static-2026-08-27.md`.

## Punto 1 — Strumentazione e benchmark ripetibile

### Telemetria da aggiungere

- [x] Cronometrare separatamente, per token e per risposta:
  - costruzione del fast-forward (`forced_prefix`);
  - filtro DSML/JSON e costruzione della maschera;
  - sampling sulla maschera;
  - `server_eval_token()`;
  - residuo aggregato di commit/stato agentico, streaming e post-processing
    (0.3--0.5 ms nel payload controllato, quindi non ulteriormente suddiviso);
  - sincronizzazione di token deterministici nella sessione live.
- [x] Contare candidati visitati, candidati ammessi, byte dei token esaminati,
  chiamate al simulatore e token fast-forward. Le allocazioni sono state
  eliminate dal loop caldo e verificate per audit del codice/A-B, non tramite
  un contatore runtime permanente.
- [ ] Distinguere nelle metriche `wall`, CPU e GPU quando il backend lo consente.
- [x] Rendere la telemetria opt-in (`DS4_SERVER_PHASE_PROFILE`) e fuori dal
  percorso caldo quando è disattiva.

### Matrice minima di misura

- [ ] Stesso modello, stesso prompt e stesso contesto con:
  - generazione non agentica;
  - agentic senza tool call in corso;
  - tool call DSML con schema piccolo;
  - tool call DSML con schema grande/dinamico;
  - `tool_choice=required` e scelta facoltativa;
  - fast-forward attivo e caso senza continuazione deterministica.
- [ ] Warm-up esplicito, almeno 5 run lente per variante e ordine alternato A/B.
- [ ] Registrare mediana, p10/p90 o deviazione, CV, contesto, token generati,
  build/commit, flags, modello, hardware e correttezza.
- [ ] Separare `prefill tok/s`, `eval tok/s` e `end-to-end tok/s`.

## Punto 2 — Rendere efficiente il constrained decoding DSML/JSON

### Risultato della ricerca in letteratura

Il percorso corrente ha costo almeno lineare nella dimensione del vocabolario a
ogni passo, aggravato dalla simulazione e dalle allocazioni per candidato. È lo
stesso limite descritto dalla letteratura recente. Le fonti indicano quattro
famiglie di interventi trasferibili a DS4: rappresentazione indicizzata dei
token (trie/FSM), cache dipendenti dallo stato del parser, maschere adattive per
stati accept-heavy/reject-heavy e sovrapposizione del lavoro CPU con la eval GPU.

| Fonte primaria | Idea misurata/proposta | Trasferimento prudente a DS4 | Rischio o costo |
|---|---|---|---|
| [XGrammar, MLSys 2025](https://arxiv.org/abs/2411.15100) | Adaptive token mask cache, stack persistente, context expansion, mask CPU sovrapposta alla GPU; jump-forward complementare | Cache per stato DSML, storage accept/reject adattivo e, solo in una fase successiva, preparazione asincrona della maschera seguente | Memoria, invalidazione con schemi dinamici, sincronizzazione corretta |
| [XGrammar 2](https://arxiv.org/abs/2601.04426) | TagDispatch per alternare testo libero e sottogrammatiche; cache riusabile fra grammatiche dinamiche | DFA dei marker DSML e cache delle parti statiche del wrapper/schema, senza imporre il parser di XGrammar | Conservare esattamente protezione dei marker e `tool_choice` |
| [DOMINO](https://arxiv.org/abs/2403.06988) | Allineamento corretto con token subword, strutture precomputate e speculative decoding della struttura | Trie byte-level dei piece e fast-forward solo quando la continuazione tokenizzata è davvero univoca | Espansione degli stati e rischio di cambiare la distribuzione se si forza troppo |
| [Efficient Guided Generation / Outlines](https://arxiv.org/abs/2307.09702) | Indicizzazione del vocabolario per stato FSM per spostare lavoro fuori dal loop per token | Precomputare piece/offset e indici per stati DSML stabili | Tempo di compilazione e memoria per schemi dinamici |
| [LLGuidance](https://github.com/guidance-ai/llguidance) | Lexer a derivative, parser Earley, attraversamento del trie dei token e fast-forward | Usare il disegno del trie e della workspace persistente; nessuna nuova dipendenza nella prima fase | I risultati del progetto non sono una garanzia per DS4; integrare Rust sarebbe un cambio architetturale |
| [Parser State Classification](https://arxiv.org/abs/2608.03065) | Preprocessare l'accettazione dei token come classificatore dello stack, evitando la scansione online del vocabolario | Valutare una versione limitata agli stati statici DSML solo dopo le ottimizzazioni semplici | Preprocessing e memoria probabilmente eccessivi per registri tool dinamici |

Nota metodologica: i numeri pubblicati da questi lavori valgono per i rispettivi
parser, hardware, vocabolari e grammatiche; qui giustificano gli esperimenti, non
sono una previsione del guadagno DS4.

### Sequenza sperimentale

#### 2A — Eliminare lavoro ripetuto senza cambiare l'algoritmo

- [x] Precalcolare una vista stabile dei piece del vocabolario e la lunghezza
  massima invece di richiamare e ricopiare il testo per ogni sampling.
- [x] Riutilizzare per sessione le workspace dei logits mascherati e del
  fast-forward; nessuna `malloc/free` nel loop caldo a regime.
- [x] Rendere heap-free il buffer temporaneo `history + piece` tramite scratch
  persistente del filtro. Il simulatore resta semanticamente invariato.
- [x] Eseguiti A/B separati: lo scratch da solo ha dato un vantaggio modesto;
  cache dei piece e logits workspace hanno prodotto il guadagno dominante.

Gate 2A: output e candidati ammessi identici sui test del piano constrained;
allocazioni nel filtro a regime pari a zero; riduzione misurabile di
`filter_mask_ms/token` senza regressione dell'eval.

#### 2B — Ridurre i candidati visitati

- [x] Costruire un trie byte-level dei piece tokenizzati una sola volta per
  engine/vocabolario.
- [x] Esporre dal simulatore DSML transizioni incrementali sufficienti a potare
  interi rami del trie appena il prefisso diventa invalido.
- [x] Confrontare scansione completa e trie su stati: wrapper, nome tool, chiavi,
  numero, stringa libera/escaped e marker di chiusura.
- [x] Misurare nodi visitati e token finali verificati, non solo wall time.

Gate 2B: insieme esatto dei token ammessi uguale alla scansione esaustiva su un
corpus di stati e fuzz differenziale; nessuna falsa accettazione o esclusione.
Il gate e' coperto da `ds4_test --constraint-trie`, dalla suite live Qwen3.8 in
`compare_new_vs_oracle` (nessuna divergenza osservata) e dagli artifact a cinque
ripetizioni sopra; il fallback esaustivo resta fail-closed.

#### 2C — Cache adattiva per stato e schemi dinamici

- [x] Riutilizzare la partizione statica DSML nello stato opzionale `SEARCH`
  quando tracker e frontiera non contengono marker parziali.
- [ ] Identificare una chiave canonica minima dello stato del simulatore.
- [ ] Separare componenti statiche (marker/wrapper DSML, lessico JSON) da tool e
  schema dinamici, seguendo l'idea di cache cross-grammar.
- [ ] Memorizzare allow-list negli stati reject-heavy e deny-list negli stati
  accept-heavy; misurare hit rate, byte e costo di invalidazione.
- [ ] Applicare un limite/LRU e verificare registri tool che cambiano fra turni.

#### 2D — Parallelismo e tecniche più invasive, solo se ancora necessarie

- [ ] Valutare la maschera per il token successivo in parallelo alla eval GPU del
  token appena committato: le dipendenze lo permettono, ma serve un protocollo
  thread-safe e una prova che il tempo di mask resti significativo.
- [ ] Valutare una classificazione precomputata degli stati statici in stile PSC
  solo se trie+cache non bastano.
- [ ] Non usare scorciatoie di rejection sampling che alterino top-k/top-p/min-p.
- [ ] Non spostare il masking in un nuovo kernel CUDA in questo ciclo.

## Punto 3 — Sincronizzazione efficiente dei piccoli suffissi

Questo punto resta deliberatamente dopo la telemetria e il primo intervento
sicuro sul filtro. La documentazione esistente segnala problemi di correttezza e
di ordine di riduzione nel percorso layer-major; il criterio storico a 512 token
non va reinterpretato senza riprodurre quei casi.

### Requisito esplicito per sessioni live e MTP

Per preparare il percorso MTP, un prefill live di **2 token o più** deve essere
efficiente. L'intervento corrente è volutamente limitato a suffissi di 2--127
token: a 128 token il percorso layer-major è già nettamente più veloce. Un
prefill di un solo token produce già i logits del token successivo nel normale
forward; non richiede un caso speciale aggiuntivo.

Nel percorso token-exact vengono sempre aggiornati stato ricorrente, posizioni e
KV per ogni token del suffisso. L'output head viene invece eseguito solo
sull'ultimo token: i suoi logits sono precisamente quelli necessari per il primo
sampling successivo. I logits intermedi non vengono usati e non influenzano lo
stato. I percorsi decode e MTP esistenti conservano il wrapper che emette sempre
i logits.

### Audit storico completato prima del codice

- [x] Ricostruire le modifiche che hanno introdotto il crossover a 512 token e
  il fast-forward DSML (`git log`, `git blame`, diff e test associati).
- [x] Rileggere e collegare i tentativi già annotati in
  `docs/performance/qwen36-performance-experiments-2026-08-05.md`, nel piano constrained e
  nei ledger performance.
- [x] Elencare i casi di non equivalenza: ordine delle riduzioni, posizioni/KV,
  chunk boundary, resume di una sessione lunga e output head sull'ultima riga.
- [x] Non assumere che sostituire `prompt->len` con `suffix_len` sia corretto solo
  perché appare semanticamente più naturale.

### Matrice di crossover da misurare

- [ ] Contesti live: 0, 2k, 8k, 12k, 16k e il massimo pratico del modello.
- [ ] Suffissi: 1, 2, 5, 8, 16, 32, 42, 64, 123, 128, 152, 256, 512 e 1024 token.
- [ ] Confrontare percorso token-by-token e layer-major con identico KV iniziale.
- [ ] Misurare wall, GPU, tok/s, memoria temporanea e prima divergenza numerica.
- [ ] Ripetere sui suffissi deterministici DSML reali oltre ai token sintetici.

### Misure mirate del crossover live a contesto 8k

| Suffisso | Percorso | Prima | Dopo | Nota |
|---:|---|---:|---:|---|
| 2 | layer-major, poi token-exact | 9.05--9.29 tok/s | 31.88--33.86 tok/s | circa 216 ms -> 60 ms per blocco; include skip output head intermedio |
| 4 | token-exact | non misurato | 32.54--33.41 tok/s | circa 121 ms per blocco |
| 64 | token-exact | non misurato | 30.12 tok/s | ancora più rapido del vecchio layer-major corto |
| 127 | token-exact | non misurato | 32.31--32.65 tok/s | bordo superiore incluso verificato a contesto 8k |
| 128 | layer-major | non misurato | 63.69 tok/s | conferma il crossover e il limite superiore esclusivo a 128 |

La prima variante, che selezionava il percorso in base al solo `suffix_len`, ha
portato i blocchi da 2 token a 30.63--32.04 tok/s. Saltare l'output head sul
primo token ha aggiunto circa il 6--7%. La matrice ampia 4--512 avviata in più
processi è scaduta senza un dataset valido; è stata scartata e sostituita dalle
misure seriali 2/4/64/127/128, sufficienti a delimitare questo intervento.

Scelta risultante: preservare il comportamento storico dei prompt freddi sotto
512 token; per una sessione live usare token-exact quando `suffix_len < 128` e
layer-major da 128 in su. Il gate numerico obbligatorio è stato superato.

### Gate finali eseguiti

- `make qwen-numerics`: `PASS`; stato ricorrente e conv-state esatti o entro il
  normale roundoff del probe.
- Build CUDA e CPU dei binari interessati senza warning nuovi.
- `ds4_server_test`, `ds4_test --server`, extractor, agent, layer-pack,
  placement multi-GPU, parser CLI e sampling: tutti `PASS`.
- Cinque tool call live identiche: sempre 53 token, 13 token fast-forward,
  finish `tool_calls` e argomenti `{"decision":"approve"}`.
- Profilo live dopo il primo 2A: decode 2.365--2.389 s, mediana 2.372 s =
  22.34 tok/s; filtro 0.264--0.269 s.
- Nel successivo A/B il riuso del prefisso già copiato nella scratch ha ridotto
  la mediana del solo filtro a 0.246 s (circa -7%). Il decode mediano è rimasto
  2.377 s = 22.30 tok/s perché la variabilità dell'eval ha assorbito circa lo
  stesso margine: `KEEP` per il lavoro CPU eliminato, non come guadagno
  end-to-end rivendicato.
- La suite aggregata `make test` è stata infine lasciata completare sul modello
  Qwen3.8 locale: exit 0, incluso il prefill long-context a 31.181 token, tool
  call, constraint-trie, server, layer-pack, placement, CLI e sampling. Sono
  inoltre passati build CUDA completa, smoke CLI target/MTP e
  `mtp-verify-depth` esplicito (231 token, chunk massimo 3, gap argmax zero).

## Ledger degli esperimenti e dei tentativi

| ID | Ipotesi / attività | Variabile | Metriche e gate | Stato | Esito / note |
|---|---|---|---|---|---|
| OBS-20260812-01 | Il gap server/bench non è spiegato dalla sola eval GPU | Nessuna, audit dei log e timer | t/token server meno t/token bench alla stessa lunghezza | `KEEP` | Confermato: baseline decode 4.24 s, di cui circa 2.45 s in filtro/forced-prefix e circa 1.36 s in eval. |
| OBS-20260812-02 | Il filtro DSML è candidato dominante | Nessuna, audit e contatori | complessità, vocab visitato, allocazioni | `KEEP` | 9,932,800 candidati e 73,846,280 byte decodificati per la risposta controllata. |
| RES-20260812-01 | La letteratura offre alternative all'`O(|V|)` online | Ricerca fonti primarie | tecniche applicabili senza cambiare semantica | `KEEP` | Trie, cache di stato, storage adattivo e overlap sono i filoni prioritari. |
| AUD-20260812-01 | Il dispatch dei piccoli suffissi richiede audit storico | Nessuna modifica | storia, test e crossover riprodotti | `KEEP` | `f789883` introdusse token-exact sotto 512 per stabilità; i test storici non coprivano un piccolo suffisso dopo una sessione lunga. |
| INST-20260812-01 | Timer di fase spiegano il residuo | Telemetria soltanto | overhead off≈0; somma fasi≈wall | `KEEP` | Profilo opt-in; nessuna lettura clock nel sampler quando disattivato. |
| OPT-20260812-01 | Scratch persistente elimina heap churn per candidato | Solo scratch del filtro | output identico e tempi fase | `KEEP` | Vantaggio modesto ma ripetibile; nessun cambio di semantica. |
| OPT-20260812-02 | Cache dei piece e workspace logits persistente riducono il filtro | Solo cache/workspace | output identico, 5 run | `KEEP` | Filtro circa 1.67 s -> 0.265 s; decode mediano circa 12.47 -> 22.32 tok/s. |
| OPT-20260812-03 | Riutilizzare la maschera parziale del forced-prefix evita la seconda scansione | Maschera forced-prefix | token ammessi e output identici | `REJECT` | La scansione si arresta appena prova il ramo comune: la maschera è incompleta e cambiava 53 token in 32. Completarla ripristina l'output ma annulla il vantaggio. Codice rimosso. |
| OPT-20260812-04 | Token-exact sul suffisso live 2--127 evita il layer-major inefficiente | Dispatch sulla lunghezza del suffisso | 2/4/64/127/128, regressioni numeriche | `KEEP` | 2 token circa 216 -> 60 ms; 127 = 32.31--32.65 tok/s; 128 resta layer-major a 63.69 tok/s. |
| OPT-20260812-05 | Omettere i logits intermedi nel micro-prefill | Output head solo sull'ultima riga | stato/KV invariato, regressioni, benchmark | `KEEP` | 2 token 30.63--32.04 -> 31.88--33.86 tok/s; output head finale alimenta direttamente sampling/MTP. |
| OPT-20260812-06 | Copiare history/raw una sola volta per scansione | Riuso del prefisso nella scratch del filtro | output identico, 5 run | `KEEP` | Filtro mediano circa 0.265 -> 0.246 s; decode mediano statisticamente invariato (2.372 -> 2.377 s). |
| BENCH-20260812-01 | Matrice ampia parallela 4--512 | Più processi benchmark | CSV completo e isolamento | `REJECT` | Timeout senza dataset valido; sostituita da run seriali mirate per non contaminare la GPU. |
| BASE-20260827-01 | La storia agentica libera espone il costo residuo di `SEARCH` | Nessuna modifica server | warm-up + 2 run target/MTP, phase profile | `KEEP` | 11,739 tok/s target; masking mediano 17.293 ms su 432 token. |
| TEST-20260827-01 | Un fixture plain-story deve distinguere risposta testuale e tool call | Solo profiler/runner | stop, una richiesta, >=400 token, >=250 parole, SSD | `KEEP` | 432 token, 298 parole, zero tool call; reasoning none esplicito evita il recupero Agent Wiki. |
| OPT-20260827-01 | La partizione statica DSML è sicura in `SEARCH` sincronizzato | Eligibility `SEARCH` soltanto | A/B, hash, compare-oracle, SSD | `KEEP` | 23,457 tok/s (+99,82%), masking 136 ms (-99,21%), 433 confronti e zero divergenze. |
| TEST-20260828-01 | Temperatura, thinking e MTP devono essere assi espliciti del server | Harness Chat/Responses e runner target/MTP | 12 casi per variante, contatori, semantica/contratto | `KEEP` | 12/12 target e 12/12 MTP; confronto 6 semantic-exact + 6 contract, inclusa transizione fallback→SEARCH statico dopo `</think>`. |
| FIX-20260828-01 | Il frontier host deve sopravvivere al cambio MTP → target | Capacità fisica `payload+1` per `fmemopen`, lunghezza logica invariata | Ultimo byte, checksum, system/HDS/skill cold-warm nelle due direzioni | `KEEP` | Rimossa la corruzione dell'ultimo byte; nessun reload SSD post-risposta e tutti i gruppi Agent SSD passano. |
| OPT-20260828-02 | Avanzare la grammatica su ogni verifier row rende MTP utile anche nei vincoli | Stato temporaneo DSML/thinking/JSON e sampling target autorevole | storia, JSON, matrice 24 richieste, fallback e A/B required | `KEEP` | +13,16% sulla storia; +27,43% mediano su JSON; required strutturale resta target-only perché la variante esaustiva regrediva; zero fallback e output invariati. |

Stati ammessi:

- `PLANNED`: non ancora eseguito;
- `IN_PROGRESS`: raccolta dati o implementazione in corso;
- `KEEP`: correttezza superata e vantaggio ripetibile;
- `REJECT`: ipotesi falsificata, regressione o complessità non giustificata;
- `NEED_MORE_DATA`: misura instabile, incompleta o non isolata.

## Registro cronologico

### 2026-08-12 — Apertura

- Creato un piano separato; esclusi esplicitamente i kernel CUDA.
- Corretta l'interpretazione del riferimento lungo: circa 29 tok/s a 12k e
  27.08 tok/s a 16k.
- Dai log, isolato un residuo end-to-end molto maggiore del solo tempo di eval.
- Dall'audit statico, identificati scansione completa del vocabolario,
  ricostruzione dei piece e allocazioni nel filtro come prime ipotesi misurabili.
- Completata la ricerca primaria del punto 2 e trasformata in una sequenza di
  esperimenti conservativi.
- Segnalato il punto 3 come area con precedenti problemi: nessuna modifica prima
  di audit storico, test di equivalenza e matrice di crossover.

### 2026-08-12 — Implementazione e chiusura del ciclo

- Aggiunto il profilo di fase opt-in e misurato il costo reale su cinque run.
- Conservati cache dei piece, workspace logits e scratch persistenti; il decode
  DSML controllato è passato da circa 12.47 a circa 22.3 tok/s.
- Scartato e rimosso il riuso della maschera forced-prefix perché la maschera
  ottenuta dall'early-exit era incompleta; la versione completa non accelerava.
- Auditato il commit storico `f789883` e mantenuto il comportamento dei prompt
  freddi corti.
- Applicato il percorso live 2--127 e misurati 2, 4, 64, 127 e 128 token. Il
  blocco da 2 token è passato da circa 216 a circa 60 ms; 127 token raggiungono
  32.31--32.65 tok/s; 128 resta layer-major a 63.69 tok/s.
- L'output head viene saltato solo sui token intermedi non osservabili; l'ultimo
  forward produce già i logits del token successivo e MTP usa quei logits per
  verificare il primo draft senza un altro forward target.
- Completati gate numerici, unitari e cinque tool call live identiche. Nessuna
  modifica ai kernel CUDA.
- Eliminata anche la ricopia della history per ciascuno dei 248,320 candidati:
  miglioramento locale del filtro di circa il 7%, neutro end-to-end entro la
  variabilità dell'eval e registrato senza sovrastimarne l'effetto.

### 2026-08-27 — Qwen3.8, `SEARCH` statico e storia live

- Congelate curve server target-only/MTP a 2K, 8K, 10,8K, 16K e 22K.
- Aggiunto `plain-story` al profiler reale Agent Wiki e un runner PowerShell per
  target-only/MTP con phase profiling e KV SSD.
- Isolato il masking `SEARCH` come collo dominante: circa 17,3 s su 36,8 s di
  decode per 432 token.
- Riutilizzata la partizione statica DSML solo con tracker sincronizzato e
  nessun `<` nella finestra trattenuta; ogni caso non dimostrato sicuro resta
  dinamico e fail-closed.
- Misurato un raddoppio stabile del throughput, con hash identici, checkpoint
  residenti e zero divergenze full-vocabulary.
- Resi Qwen3.8 e 22.593 token i default operativi di launcher, harness,
  JSONSchemaBench e runner agentici; Qwen3.6 resta uno smoke di compatibilità.

### 2026-08-28 — Matrice sampling, thinking e MTP

- Auditata la copertura esistente: constrained API, Agent SSD e storia usavano
  temperatura zero e non coprivano il prodotto cartesiano richiesto.
- Aggiunto un gate model-free a `make test` e un runner live che esegue 12 casi
  target-only e 12 con sidecar MTP.
- Usata Chat con seed per il testo libero e Responses per l'estensione
  `agentic`, rispettando i campi realmente supportati da ciascun endpoint.
- Verificati sampling Qwen raccomandato, greedy, thinking on/off, SEARCH
  opzionale, tool obbligatorio, cicli MTP e fallback constrained target-only.
- Aggiunto il verifier grammar-aware per ogni draft/bonus row, con stato parser
  temporaneo e rollback congiunto a RNG e stato ricorrente. La policy usa MTP
  su JSON e tool opzionali, nel reasoning dei required, e passa al target al
  confine DSML required dove le misure mostrano una regressione.
- Conservata la distinzione fra identità semantica e contratto pubblico: i
  near-argmax del verifier batch possono cambiare il reasoning nascosto, mai la
  validità o la scelta/struttura delle call.
- Eseguito Agent SSD su system, HDS e deep-skill sia target-only → MTP sia MTP
  → target-only. Il secondo verso ha scoperto che `fmemopen` richiedeva un byte
  fisico extra per il terminatore; dopo il fix, checksum residenti,
  ricostruzioni `memory-request` e tutti i canary sono verdi senza reload SSD
  post-risposta.

## Prossima azione

Il ciclo corrente è chiuso. Nella storia libera l'eval GPU è di nuovo il costo
dominante; il masking `SEARCH` è sceso a circa 0,14 s per 432 token. La prossima
ottimizzazione server deve partire da una nuova attribuzione su tool payload,
JSON Schema e prefissi parziali, non da un'altra scorciatoia `SEARCH`. Una cache
adattiva per stati strutturali o overlap CPU/GPU resta subordinata agli stessi
gate full-vocabulary e live SSD.
