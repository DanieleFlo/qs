# Analisi tecnica di `syv-ai/qwen38-27b-rtx3090`: da dove viene la velocità e quanto costa in qualità

## Provenienza e perimetro

**Fonte primaria:** [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090).

**Snapshot analizzato:** stato di `main` dichiarato nel testo al 21 agosto
2026, quando la repository contava 52 commit; il testo ricevuto non includeva
lo SHA.

**Controllo di provenienza:** il 24 agosto 2026 l'HEAD pubblico verificato era
`00210159df4366704b98b178258b3f618005611a` e GitHub mostrava 105 commit.

L'upstream è quindi evoluto dopo lo snapshot: benchmark e configurazioni qui
discussi vanno letti come analisi storica, non come descrizione dell'HEAD corrente.

Analisi riferita allo stato attuale di `main` del **21 agosto 2026**. La repository è in rapido sviluppo e al momento mostra 52 commit.

Interpreto “DS4 di San Filippo” come **`antirez/ds4` di Salvatore Sanfilippo**. Userò DS4 soprattutto come **standard metodologico di affidabilità**, non confrontando direttamente i suoi valori di NLL con Qwen: un NLL di DeepSeek/GLM e uno di Qwen non sono comparabili numericamente. Quello che invece è molto utile è il modo in cui DS4 distingue una vera ottimizzazione da un cambiamento dei logits.

## Conclusione principale

La cosa più importante da capire è questa:

> **“Speculative decoding lossless” e “modello identico all'originale” sono due affermazioni completamente diverse.**

MTP, DFlash2 o il context lookup possono essere statisticamente esatti **rispetto al target che stanno verificando**. Ma se quel target ha `lm_head` quantizzata, recurrent state FP16, KV FP8/INT8/K4V2 o attivazioni INT8, il verificatore sta riproducendo esattamente la distribuzione di **un target già numericamente modificato**, non quella del Qwen BF16 originale.

Questa distinzione è centrale nella repo, perché il README tende talvolta a usare “lossless” riferendosi alla speculazione, mentre l'intero serving stack è esplicitamente quantizzato.

### La mia classificazione

| Classe                                                       | Significato                                                                                                                      |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **A — ottimizzazione pura**                                  | Non modifica intenzionalmente il target; può aumentare molto la velocità senza degradare il modello.                             |
| **A− — semanticamente pura, ma numericamente da verificare** | Stessa matematica/algoritmo, ma cambia kernel, ordine delle operazioni o batching; possibili differenze floating-point near-tie. |
| **B — drift piccolo/trascurabile nei test disponibili**      | Modifica realmente il target, ma i test mostrano degradazione molto piccola.                                                     |
| **C — trade-off di qualità materiale**                       | Aumento prestazionale reale acquistato con una degradazione misurabile.                                                          |
| **D — correctness failure**                                  | Non è un normale trade-off: produce logits sbagliati, output corrotti, crash o verifica contro la distribuzione sbagliata.       |

---

# 1. Prima di tutto: cosa significano davvero 1.000, 381 e 132 tok/s

I numeri più appariscenti della repo non devono essere messi sullo stesso piano.

**~1.094 tok/s** è throughput aggregato di decode con **64 richieste concorrenti**. La variante che quantizza tutte le attivazioni lineari arriva a ~1.222 tok/s steady-state / 1.042 e2e. Non significa che una singola chat generi a 1.000 token al secondo.

Per una singola richiesta realistica, la configurazione MTP attuale fa circa **113,6 tok/s con sampling normale e 118,3 greedy**. DFlash2 arriva a circa **122 tok/s default / 132 greedy** nei risultati principali.

I **381-382 tok/s** sono invece un caso specializzato: il modello sta **riproducendo materiale già presente nel contesto**, e il context-lookup riesce a fornire fino a 15 token verificabili per step. Su testo libero il vantaggio del verify block lungo è nell'ordine di pochi punti percentuali, non 3×.

Quindi, per giudicare la repo, userei come headline:

**~114-132 tok/s = velocità single-stream general-purpose.**

**~1.100 tok/s = throughput server con batch/concorrenza.**

**~381 tok/s = workload specializzato di context reproduction.**

---

# 2. Le innovazioni, ordinate per qualità

## 1. Quantizzazione del solo drafter MTP — Classe A

Questa è probabilmente una delle idee migliori dell'intera repo.

Il modulo MTP è quantizzato INT8, e nella variante veloce anche INT4. Ma il drafter **non decide il token finale**: propone soltanto candidati, successivamente verificati dal target. Una quantizzazione pessima del drafter abbassa l'acceptance rate e quindi la velocità, ma, assumendo un rejection sampler corretto, **non modifica la distribuzione finale del target**.

Il codice lo dice esplicitamente e riporta nessun cambiamento misurabile dell'acceptance passando a INT8.

**Giudizio: eccellente.** È esattamente il genere di posto in cui conviene spendere aggressivamente precisione: la perdita viene convertita in minore performance, non in minore qualità del modello.

---

## 2. Quantizzazione INT4 del drafter DFlash2 — Classe A

Stesso principio.

DFlash2 BF16 pesa circa 3,85 GB e sulla 3090 aggiunge circa 5 ms per step, tanto da annullare il vantaggio. La repo GPTQ-quantizza 36 matrici portandolo a circa 1,19 GB. Il costo riportato è circa **−5% di acceptance con sampling normale e praticamente zero a greedy**.

Poiché è ancora il **drafter**, non il target, la perdita di precisione non dovrebbe degradare la distribuzione finale.

**Giudizio: eccellente.** Molto più “pulito” che quantizzare KV, hidden state o `lm_head` del target.

---

## 3. Draft vocabulary costruito sugli output del modello — Classe A

Il drafter non proietta più sul vocabolario completo da ~248k token, ma su circa 40k token scelti osservando ciò che Qwen stesso tende a generare.

La vecchia vocabulary derivata da web text copriva circa il 92% generale e 83% sul codice; quella costruita su 5,4 milioni di token prodotti dal modello arriva al **97,5% generale e 96% sul codice**, e la repo misura circa **+10% di throughput single-stream**.

Un token assente non può essere proposto dal drafter, quindi produce semplicemente una rejection anticipata. Non cambia ciò che il target può emettere.

**Giudizio: una delle ottimizzazioni migliori della repo.** Guadagno rilevante, praticamente nessun costo di qualità del target.

---

## 4. Speculative decoding MTP — Classe A come algoritmo

Qwen3.8-27B ha MTP nativo; l'architettura ufficiale ha 64 layer, 48 Gated DeltaNet + 16 attention layer e MTP addestrato nativamente.

Nel benchmark della repo si passa indicativamente da:

* ~46 tok/s senza speculation;
* 66/79 con MTP-2 originale;
* 78/99 con MTP-4 economico;
* fino a ~114/~124 nella variante ottimizzata.

Come algoritmo, speculative decoding con rejection sampling corretto è target-preserving.

**Giudizio: eccellente.**

La riserva importante è sull'**implementazione**, non sulla teoria: custom patch e floating-point possono ancora introdurre bug, come vedremo.

---

## 5. DFlash2 — Classe A per il principio, A− per questa integrazione custom

DFlash2 propone un intero blocco di 7 token con un solo passaggio anziché concatenare quattro predizioni MTP. Nella repo porta il C1 general-purpose all'area dei 118-132 tok/s.

Ma qui DS4 diventa importante.

La repo ha dovuto correggere un **vero bug semantico** nel backport: vLLM 0.27.1 conservava draft logits dopo l'applicazione della temperatura, mentre la logica upstream si aspettava logits raw. Senza la correzione, per `0 < T != 1` la verifica sarebbe avvenuta contro la **q sbagliata**.

Quindi:

**DFlash2 teorico: A.**

**Questa integrazione: A−**, perché contiene abbastanza codice custom da richiedere test formali sui logits dopo ogni upgrade/rebase.

---

## 6. Context lookup drafting — Classe A

Questa è l'idea dietro i 260-381 tok/s quando il modello copia o modifica materiale presente nel prompt.

La patch cerca nel token history una sequenza già vista e propone i token successivi. Per i token provenienti dal lookup usa una proposta point-mass, che può essere verificata dal normale rejection sampler.

L'aspetto elegante è che **non si fida del contesto**: il target continua a verificare.

Per testo libero non regala 381 tok/s; per copie quasi verbatim può invece arrivare a circa 15 token accettati per step.

**Giudizio: eccellente come ottimizzazione specializzata.** Non attribuirei però i 381 tok/s alla generazione general-purpose.

---

## 7. Prefix caching dello stato ibrido KV + GDN — Classe A/A−

Su una conversazione con documento da 24k token, il follow-up passa da circa 23 secondi a circa **0,85-1,15 s TTFT**, riusando sia KV sia stato recurrente dal confine del blocco cached. La repo riporta risposte token-per-token identiche.

È un enorme vantaggio in un'assistente documentale, senza dover comprimere ulteriormente il modello.

**Giudizio: tra le migliori ottimizzazioni della repo.**

Gli darei **A− anziché A assoluto** solo per il criterio DS4: DS4 chiederebbe non soltanto “stesso testo”, ma anche confronto dei logprob/local golden vectors.

---

## 8. Correzione dei gruppi KV ibridi e contabilizzazione CUDA graph — Classe A

DFlash2 introduceva spreco di KV pool perché vLLM paddingava i gruppi di layer in modo sfavorevole. La patch riduce il costo da circa **105 KB/token a 78 KB/token**, riportando la capacità da ~45k a ~70k token senza quantizzare ulteriormente il target. Inoltre contabilizza esplicitamente circa 1,2 GiB di CUDA graph memory.

**Giudizio: A pieno.** È esattamente il tipo di miglioramento che preferirei sempre a una quantizzazione: recupero di memoria tramite layout/accounting, non tramite perdita di precisione.

---

# 3. Ottimizzazioni “quasi pure”, ma che cambiano l'aritmetica

## 9. Split-KV attention per il verify step — Classe A−

È un kernel Triton progettato specificamente per il multi-query verify dello speculative decoding. La repo riporta, per layer:

* ~57 µs → 23 µs a 1,5k context;
* ~1,3 ms → 120 µs a 16k.

Insieme al sampler ottimizzato vale circa +4% nella configurazione misurata.

Concettualmente dovrebbe calcolare la stessa attenzione.

Ma un nuovo kernel CUDA/Triton può cambiare l'ordine delle riduzioni floating-point. Quindi **semanticamente è un'ottimizzazione pura, numericamente non la dichiarerei bit-exact senza golden-logit vectors**.

DS4 è molto utile proprio qui: i suoi test `local-golden.vec` sono progettati per intercettare backend drift che può lasciare invariato perfino il token greedy ma alterare significativamente i logits.

**Giudizio: ottimo, ma manca ancora il livello di dimostrazione DS4 per chiamarlo “zero drift”.**

---

## 10. Sampler small-top-k / sort-free — Classe A−

Invece di ordinare l'intero vocab da ~248k elementi, usa un percorso specializzato quando `top-k <= 64`, più una softmax parallelizzata.

Matematicamente dovrebbe campionare dallo stesso supporto troncato.

Anche qui però cambiare softmax/riduzione/order può produrre minime differenze floating-point.

**Giudizio: molto buono.** Nessun trade-off intenzionale di qualità, ma richiede un test di distribuzione per essere certificato “identico”.

---

# 4. FP16 recurrent state: probabilmente il miglior trade-off numericamente lossy — Classe B

Questo punto è particolarmente interessante per Qwen3.8.

48 dei 64 layer sono Gated DeltaNet. Lo stato recurrente FP32 occupa circa **150 MB per richiesta**; con 64 slot soltanto 37 richieste riuscivano effettivamente a essere residenti. Portandolo a FP16 la memoria e il traffico si dimezzano e tutte e 64 le richieste entrano.

La waterfall di qualità è:

FP32 state: **PPL 8.045, 516 tok/s**

FP16 state: **PPL 8.044, 707 tok/s**

Quindi il guadagno prestazionale è enorme e nel test disponibile il PPL non peggiora neppure alla terza cifra decimale.

Ma non lo chiamerei matematicamente lossless: **si sta cambiando il formato dello stato interno del target da FP32 a FP16**. Arrotondamento e accumulo esistono.

**Giudizio: B molto alto, quasi A.** Fra tutte le approssimazioni che modificano realmente il target è quella che mi convince di più.

Con uno suite DS4-like su continuazioni lunghe probabilmente sarebbe il primo candidato a essere promosso a “praticamente gratuito”.

---

# 5. Quantizzare `embed_tokens` e `lm_head` INT8 — Classe B

Qui invece il target cambia senza ambiguità.

`embed_tokens` viene quantizzata symmetric INT8 group-128. La repo misura **0,56% di errore round-trip**.

`lm_head` viene anch'essa portata a INT8 group-128, liberando circa 1,3 GB e riducendo della metà la lettura di una matrice BF16 da circa 2,5 GB; il codice riporta **0,64% Frobenius round-trip error** e circa +12% aggregate throughput.

Questa non è un'ottimizzazione “lossless”:

* l'embedding modifica lo stato fin dal primo layer;
* `lm_head` modifica direttamente i logits finali.

Il fatto che l'errore Frobenius sia 0,56/0,64% non ci dice quanto cambia la distribuzione dei token in casi near-tie.

La qualità complessiva della base requantizzata è comunque buona: PPL 8.045 e GSM8K 95,5% nel test della repo.

**Giudizio: B. Probabilmente un ottimo trade-off, ma definitivamente non lossless.**

---

# 6. `lm_head` GPTQ INT4 — Classe B/C

Nella variante single-user più veloce anche il `lm_head` del **target** diventa INT4 GPTQ.

Qui il benchmark è abbastanza chiaro:

* INT8 `lm_head`: PPL **8.045**
* INT4 round-to-nearest non calibrato: **8.17, +1,5%**
* INT4 GPTQ calibrato + MTP INT4: **8.095, +0,6%**.

La parte MTP INT4 non conta per la qualità, quindi gran parte del +0,6% residuo è attribuibile al percorso target/`lm_head`.

**Giudizio: B.** Un +0,6% PPL è piccolo e mi sembra ragionevole per passare da ~107/109 a ~114/~124 tok/s, ma non deve essere descritto come “stesso modello”.

---

# 7. INT8 delle attivazioni: qui inizia il vero quality-for-speed

La tabella della repo è molto istruttiva.

| Target                   |             PPL | Throughput 64 conc. | Valutazione |
| ------------------------ | --------------: | ------------------: | ----------- |
| W4A16 + FP16 state       |           8.044 |           707 tok/s | B           |
| INT8 solo gate/up        | 8.12, **+0,9%** |                 787 | B           |
| INT8 intero MLP, default | 8.22, **+2,2%** |                 942 | C           |
| INT8 tutte le lineari    | 8.34, **+3,7%** |               1.042 | C/C−        |

### Gate/up INT8: B

+0,9% PPL per un buon aumento di throughput. GSM8K resta 95,5% nel campione da 200.

**Buon compromesso.**

### Intero MLP INT8: C

È il default batch della repo: circa **942 tok/s e2e**, ma PPL sale del **2,2%**. GSM8K passa 95,5 → 95,0%, che su 200 domande è appena una domanda, quindi quel dato specifico ha poca potenza statistica.

**Non lo chiamerei più “trascurabile”.**

È un trade-off consapevole: molto throughput in cambio di una distribuzione target chiaramente più approssimata.

### Tutte le lineari INT8: C−

Arriva a ~1.042 tok/s e2e / ~1.222 steady decode, ma PPL è **+3,7%**.

Il guadagno marginale rispetto al default MLP è molto meno impressionante del costo ulteriore di qualità.

**Per workload fidelity-sensitive non la userei.**

Questa configurazione è importante perché significa che il numero massimo “oltre 1.000 tok/s e2e” non rappresenta la configurazione qualitativamente più conservativa.

---

# 8. KV cache quantizzata: il punto su cui concordo maggiormente con la tua osservazione

Qui bisogna correggere una possibile lettura del README.

**Quantizzare la KV modifica il target.**

Lo stesso codice della repo offre una dimostrazione molto pulita. Per il kernel INT8 per-token/head misura:

* errore max ~`1e-4` rispetto alla **KV già dequantizzata**;
* ma **1,02% mean relative error rispetto alla reference non quantizzata**.

Quindi il kernel può essere quasi esatto rispetto alla rappresentazione quantizzata, ma è la rappresentazione quantizzata ad aver già introdotto il drift.

Questo è precisamente l'esempio della distinzione:

> “kernel esatto rispetto alla cache INT8” ≠ “inferenza identica alla cache BF16”.

E c'è un'altra sorpresa importante: **la KV quantizzata qui non serve principalmente ad andare più veloce.**

Gli autori scrivono esplicitamente che una cache INT8 contiguous è persino leggermente più lenta della BF16 per queste shape; la cache quantizzata **compra context, non speed**.

Infatti il massimo single-stream `CTX=fast` usa **BF16 KV / FlashAttention e ~64k context**; la modalità long da 150k usa FP8/FlashInfer ed è più lenta.

### Quindi la mia classificazione è:

**BF16 KV fast:** preferibile se 64k bastano.

**FP8 KV:** B, se serve 150k; non la considererei un'innovazione di speed.

**INT8 per-token-head KV:** B/C; modifica logits e non è neppure vantaggiosa per velocità pura.

---

# 9. KVarN K4V2 e cache 4/2 bit — Classe C per fidelity, ottima solo se serve context estremo

KVarN consente di spingere la capacità molto più in alto: la repo riporta 262k che entrano con margine e needle test corretti fino a 240k. Ma:

* PPL FP8: **8.223**
* PPL KVarN K4V2: **8.236, +0,16%**
* 100k decode: 27 ms/token FP8 contro 33 ms KVarN;
* throughput 64 richieste corte: 876 contro 692 tok/s.

Ancora più significativo: a 112k il modello con KVarN passa da 2,56 a 2,38 accepted token/step, circa **−7% acceptance**, perché la cache quantizzata sposta abbastanza i logits del target da farlo concordare meno con il drafter.

Questa è una prova indiretta molto convincente che il drift non è solamente teorico.

La PPL media si muove poco, ma il comportamento del verifier cambia abbastanza da essere visibile nell'acceptance.

**Giudizio: C come modifica numerica, ma ottima tecnologia di capacity.**

La userei solo quando l'alternativa è “il prompt non entra”. Non per andare più veloce.

---

# 10. Il prerequisito nascosto: il body è già W4A16

Prima ancora delle innovazioni della repo, il modello caricato è il checkpoint **Qwen3.8-27B-W4A16-AutoRound**, non il BF16 ufficiale.

Quindi esiste già una quantizzazione a 4 bit dei pesi principali.

La repo riporta IFBench:

* stack W4A16: **78,3**
* modello Qwen ufficiale unquantized: **79,5**.

È circa 1,2 punti assoluti.

Ma qui farei una correzione metodologica importante: **non considero questo un A/B formale**, perché il 79,5 viene dalla model card ufficiale di Qwen, mentre il 78,3 è ottenuto dal serving stack della repo. Non è una stessa binary/config/harness con un'unica variabile modificata.

Il risultato suggerisce che la degradazione totale sia moderata, ma non permette di attribuire con rigore “1,2 punti ad AutoRound”.

---

# 11. I casi realmente problematici: Classe D

Questi non sono “quantizzazione con lieve drift”. Sono bug.

### W4A8 Marlin con negative scales non corretto

Il percorso originale di vLLM interpretava come unsigned scale requantizzate che potevano essere negative; il risultato era **output garbage pur mostrando benchmark eccellenti**. La repo ha introdotto una patch specifica per correggerlo.

**Classificazione: D se usato senza fix.**

Questo è anche un ottimo esempio del perché “tok/s alto + qualche prompt che sembra funzionare” non può essere il criterio di validazione.

### DFlash2 backport senza semantic fix della temperatura

Come detto prima, senza la patch si verificavano draft contro una `q` sbagliata per temperature intermedie.

**Classificazione: D.**

Non è un drift accettabile: rompe la correttezza del rejection sampling.

### FlashInfer/FP8 con k=4

La repo segnala un **illegal memory access** quando una richiesta termina mentre un'altra è ancora in generazione; per questo `CTX=long` usa tre draft anziché quattro.

**D per affidabilità**, anche se non è una degradazione qualitativa silenziosa.

### Sticky long-verify in batch

Il soak test della repo ha trovato che il contatore globale del long-block poteva far dipendere la lunghezza del verify dalle altre richieste del batch; cambiando block size cambia anche l'ordine floating-point e in un caso è uscito **greedy text differente**. Per questo l'hold viene applicato soltanto con una richiesta in flight.

Questo lo metterei tra **C/D operativo**: non produce “garbage”, ma è una fragilità che un sistema di QA severo deve catturare.

---

# 12. Dove DS4 è più rigoroso della suite di questa repo

Questo è probabilmente l'aspetto più importante per valutare quanto fidarsi delle affermazioni di qualità.

La repo `syv-ai` fa già molto più di un semplice eyeballing:

* IFBench 299 prompt;
* ~33k token per la perplexity su inglese, danese e codice;
* GSM8K 200 esempi;
* test long-context;
* soak test che hanno realmente scoperto divergenze.

Quindi non definirei i suoi test superficiali.

Però **DS4 imposta l'asticella più in alto**.

DS4 acquisisce una continuazione ufficiale deterministica e poi misura **token per token la NLL assegnata esattamente a quella continuazione**, evitando che una singola risposta campionata nasconda cambiamenti della distribuzione. Conserva inoltre top-logprobs ufficiali e `local-golden.vec`.

E soprattutto dichiara questi test **release-blocking dopo modifiche a tokenizer, KV cache, attention, quantizzazione, logits o model graph**.

Questo è esattamente lo standard che applicherei alla repo Qwen.

Un esempio DS4 particolarmente utile: un'ottimizzazione della expert cache viene validata cambiando l'hit-rate dal 33% al 74%, ottenendo testo greedy byte-identical e soprattutto **avg NLL identica fino a sei decimali con cache on/off**.

Quello è il tipo di evidenza che permette di chiamare una cosa “ottimizzazione pura” con molta più fiducia.

Al contrario, DS4 mantiene `local-golden.vec` proprio perché può verificarsi un backend drift che **conserva lo stesso greedy token ma rovina la distribuzione dei logits**.

Questa osservazione rende insufficienti, da soli:

* “il needle test passa”;
* “GSM8K cambia zero/una domanda”;
* “la risposta sembra uguale”;
* “Frobenius error è <1%”.

---

# 13. Come rivaluterei i test della repo usando il metro DS4

### FP16 recurrent state

**Evidenza syv:** ottima PPL, differenza nulla a tre decimali.

**Verdetto DS4-like:** molto promettente, ma servirebbero local golden logits e continuazioni lunghe per verificare accumulo del recurrent state.

**Fiducia:** alta.

### INT8 `embed_tokens` e `lm_head`

**Evidenza syv:** Frobenius error + PPL/task benchmark.

Il Frobenius è però un test sul parametro, non sulla distribuzione finale.

**Verdetto DS4-like:** manca il test decisivo top-k/logprob baseline-vs-quant.

**Fiducia:** media-alta che il danno sia piccolo, bassa che sia “zero”.

### INT8 activations

Qui la PPL mostra già il danno.

**Verdetto:** non serve cercare di dimostrare losslessness: non lo è.

Il vero problema è quantificare meglio la coda degli errori: +2,2% medio può significare piccoli cambiamenti diffusi oppure alcuni prompt gravemente peggiorati. La methodology DS4 per-case sarebbe molto utile.

### KV FP8/INT8/KVarN

Qui un test DS4 è particolarmente necessario.

Il needle test dimostra che l'informazione rimane recuperabile, non che i logits rimangano vicini.

Il fatto che KVarN faccia perdere ~7% di speculative acceptance dimostra già che i logits si stanno spostando.

**Fiducia nella claim “quality impact piccolo in media”: discreta.**

**Fiducia nella claim “lossless”: nessuna; non è lossless rispetto al target BF16.**

### Speculative decoding

Qui la teoria è molto più forte.

Ma DS4 è utile nel separare:

1. correttezza matematica dell'algoritmo;
2. correttezza del codice che mantiene KV/state durante verify/rollback/commit.

DS4 ha dedicato test specifici proprio a speculative decode e considera errori del verifier, invalid text o regressioni materiali release blockers, anche quando differenze floating-point possono impedire di richiedere sempre byte identity.

**Per syv chiederei gli stessi test.**

---

# 14. Ranking finale delle innovazioni

Se le ordinassi per rapporto **speed / fidelity**, la mia lista sarebbe:

1. **Quantizzare MTP/DFlash2 drafter** — quasi ideale: degrada eventualmente l'acceptance, non il target.
2. **Draft vocabulary basata sugli output del modello** — ottima; +~10% single-stream senza restringere il target.
3. **Speculative decoding MTP/DFlash2** — enorme guadagno, target-preserving se verifier corretto.
4. **Context lookup drafting** — target-preserving e straordinario per copy/edit/RAG, ma workload-specific.
5. **Prefix caching KV + recurrent state** — enorme vantaggio sulle conversazioni lunghe senza dover approssimare il modello.
6. **Fix del KV grouping / CUDA graph memory accounting** — recupero di memoria puro.
7. **Split-KV + sampler ottimizzato** — molto buono, ma vorrei golden-logit regression tests.
8. **FP16 recurrent state** — modifica numerica reale ma finora praticamente gratuita; forse il miglior compromesso lossy.
9. **INT8 embedding + lm_head** — compromesso probabilmente ottimo, ma modifica direttamente hidden states e logits.
10. **GPTQ INT4 lm_head** — +0,6% PPL nella variante calibrata: accettabile se si vuole il massimo single-user.
11. **INT8 gate/up activations** — +0,9% PPL: ragionevole per throughput.
12. **INT8 intero MLP** — +2,2% PPL: trade-off ormai chiaramente visibile.
13. **INT8 tutte le lineari** — +3,7% PPL per l'ultimo incremento verso 1.042/1.222 tok/s: rapporto qualità/velocità molto meno convincente.
14. **FP8/INT8 KV** — utile per context capacity, non per performance pura; drift reale.
15. **KVarN 4-bit K / 2-bit V** — tecnologia interessante per far entrare 200-262k+, ma non la sceglierei se il contesto entra già in BF16/FP8.
16. **W4A8 Marlin senza negative-scale fix / DFlash2 senza semantic fix** — non compromessi: errori di correttezza, da evitare completamente.

---

# 15. Qual è, quindi, la vera innovazione della repo?

Non è principalmente “quantizzare tutto”.

La parte tecnicamente più interessante è che gli autori hanno individuato **dove si può essere aggressivi senza toccare il target**:

* comprimere il drafter;
* ridurne il vocab;
* aumentare acceptance;
* verificare più token per forward;
* riusare il contesto come fonte gratuita di draft;
* correggere layout e memory accounting;
* riusare prefix KV + recurrent state;
* specializzare attention e sampling per la shape reale dello speculative decoding.

Queste sono le idee che considero più trasferibili ad altri modelli.

Le quantizzazioni di target — W4A16, INT8 embeddings/head, INT4 head, FP16 recurrent state, INT8 activations, FP8/INT8/KVarN KV — sono invece **trade-off separati**. Alcuni sono eccellenti, altri meno, ma non dovrebbero essere confusi con il guadagno “gratis” dello speculative/runtime engineering.

In particolare, un punto controintuitivo è che **la configurazione single-user più veloce general-purpose non deve il suo risultato alla KV quantizzata**. `CTX=fast` usa BF16 KV e FlashAttention e arriva alle prestazioni migliori; FP8/INT8/KVarN servono soprattutto ad aumentare il context window e spesso rallentano il decode.

---

# 16. Le configurazioni che sceglierei

**Massima fedeltà possibile sulla 3090:** terrei BF16 KV (`CTX=fast`), speculative decoding, drafter quantizzato, draft vocabulary, split-KV, sampler ottimizzato e prefix cache. Eviterei l'INT8 delle attivazioni. Se la VRAM lo permette, preferirei inoltre INT8 `lm_head` alla variante INT4.

**Miglior equilibrio single-user:** la variante fast attuale con DFlash2 è molto ragionevole. Il costo misurato del `lm_head` INT4 GPTQ è circa +0,6% PPL, mentre il drafter INT4 è quality-neutral rispetto al target. ~122-132 tok/s è quindi un risultato molto più interessante dei 381 tok/s pubblicitari per context-copy.

**Server batch con qualità ancora alta:** FP16 recurrent state + eventualmente INT8 gate/up. Mi fermerei prima dell'intero MLP se la qualità conta molto.

**Massimo throughput:** whole-MLP INT8 porta a ~942 e2e/~1.094 decode, ma bisogna accettare esplicitamente il +2,2% PPL. La modalità all-linear a ~1.042/~1.222 con +3,7% PPL la considererei una modalità “throughput-first”, non equivalente al modello base.

**Context >64k:** FP8 prima; KVarN/INT4/INT8 KV soltanto quando la capacità addizionale serve davvero. Non le userei pensando che siano un'accelerazione.

---

# 17. Il test che manca per poter certificare queste conclusioni alla maniera di DS4

Per portare questa repo allo standard di evidenza DS4, farei un'unica matrice Qwen “golden” con **lo stesso checkpoint, tokenizer, template, prompt e hardware**, registrando per ogni configurazione:

`BF16/W4 baseline → INT8 embed/head → FP16 GDN state → INT8 gate/up → INT8 MLP → INT8 all → FP8 KV → INT8 KV → KVarN → MTP → DFlash2 → lookup → prefix cache`.

Per ciascuna variante misurerei almeno:

**teacher-forced target-token NLL; top-1 agreement; top-k overlap; MAE dei top logits; pairwise logit ordering; greedy longest-common-prefix; per-case NLL delta, non soltanto media; long-context vectors a varie profondità; e gli stessi test sotto C1/C4/C64 e cache pressure.**

È il passaggio che permetterebbe di dire, con una base molto più solida:

* “questa patch è realmente neutra”;
* “questa introduce soltanto floating drift”;
* “questa modifica il modello dello 0,2% medio”;
* oppure “questa ha una coda di errori che IFBench/GSM8K/PPL media non mostrano”.

È precisamente il vantaggio del modello di QA di DS4: le official continuations vengono misurate token per token e i vector test vengono richiesti dopo modifiche a KV, kernel, quantizzazione e model graph, invece di affidarsi soltanto al testo generato.

**Sintesi finale:** la repo contiene vera ingegneria di inference molto interessante, e i ~114-132 tok/s single-stream non sono ottenuti semplicemente “rovinando” il modello. Una parte importante del guadagno deriva da interventi target-preserving. Però i numeri massimi di batch incorporano approssimazioni del target via INT8 activations, e le modalità KV quantizzate introducono effettivamente drift. Con il criterio rigoroso di DS4, diverse cose che la documentazione chiama “lossless” andrebbero riformulate come **“lossless speculative sampling rispetto al target quantizzato”**, che è una proprietà molto più precisa e difendibile.
