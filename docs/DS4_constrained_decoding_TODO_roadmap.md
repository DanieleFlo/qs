# TODO Roadmap — Constrained Decoding DSML/JSON ad alte prestazioni

> Obiettivo: costruire per DS4 un constraint engine che mantenga la correttezza fail-closed dell'implementazione attuale, ma elimini progressivamente la dipendenza runtime dalla scansione completa del vocabolario e dalla ri-parsing del prefisso.
>
> Target architetturale: **stato grammaticale incrementale + indice/trie dei token + mask/cache precompilate + fast-forward deterministico + classificazione PSC per gli stati riusabili + JIT/cross-grammar cache per schemi dinamici + overlap con l'inferenza**.

---

# REGOLE DI ESECUZIONE

Queste regole sono vincolanti per tutti i macro-task.

- [ ] **R1 — L'implementazione esaustiva corrente resta l'oracle di correttezza.** Non eliminarla finché la nuova pipeline non ha superato tutti i gate finali.
- [ ] **R2 — Nessuna ottimizzazione può cambiare l'insieme dei token validi.** Per ogni stato testato deve valere `new_allowed_tokens == exhaustive_allowed_tokens`.
- [ ] **R3 — Fail closed.** In caso di stato non riconosciuto, cache incoerente, compilazione fallita o ambiguità non gestita, usare il percorso esaustivo; non allargare mai silenziosamente la grammatica.
- [ ] **R4 — Tokenizer-exact.** Tutte le decisioni devono essere verificate sui veri piece/byte del tokenizer. Non assumere che caratteri, byte e token coincidano.
- [ ] **R5 — Separare correttezza e performance.** Ogni nuova struttura deve avere prima test differenziali, poi benchmark.
- [ ] **R6 — Misurare cold e hot path separatamente.** Compilation/JIT/cache-miss non devono essere nascosti nelle metriche di decode.
- [ ] **R7 — Misurare sempre il lavoro effettuato, non solo tok/s.** Registrare almeno candidate/token visitati, byte analizzati, transizioni parser, nodi trie, cache hit/miss e tempo per fase.
- [ ] **R8 — Una macro-ottimizzazione alla volta.** Non fondere modifiche indipendenti prima di aver misurato il contributo di ciascuna.
- [ ] **R9 — Conservare rollback/checkpoint del constraint state.** È necessario per candidate testing, speculative decoding e fallback.
- [ ] **R10 — Nessun fast-forward su una biforcazione reale.** Il fast-forward è consentito solo quando la continuazione è semanticamente deterministica rispetto alla grammatica.
- [ ] **R11 — Non dedurre determinismo dalla sola segmentazione BPE.** Più token possono condividere byte iniziali senza rappresentare una singola scelta grammaticale.
- [ ] **R12 — DSML e JSON Schema devono condividere l'infrastruttura, non necessariamente lo stesso parser.** Il core può essere comune; gli stati semantici possono restare specializzati.
- [ ] **R13 — Ogni cache deve avere una chiave semanticamente completa.** La chiave deve includere tutto ciò che può cambiare il set dei token validi.
- [ ] **R14 — Ogni macro-task termina con una retrospettiva scritta.** Compilare obbligatoriamente la sezione `Stato e retrospettiva`.
- [ ] **R15 — Prima di passare al macro-task successivo, soddisfare i criteri di uscita del task corrente oppure documentare esplicitamente perché viene saltato.**
- [ ] **R16 — Mantenere benchmark riproducibili.** Stesso modello, tokenizer, seed, schema, prompt, hardware e impostazioni di sampling nelle comparazioni A/B.
- [ ] **R17 — Non ottimizzare esclusivamente il caso JSON semplice.** La suite deve includere tool calling DSML, enum, const, optional/required, array, object annidati, stringhe libere e combinatori supportati.
- [ ] **R18 — Non introdurre una compilazione più costosa del risparmio ottenuto.** Per ogni tecnica precompilata misurare il break-even tra costo di compilazione e numero di riusi.
- [ ] **R19 — I risultati negativi vanno conservati.** Un'ottimizzazione scartata deve restare documentata con benchmark e motivazione.
- [ ] **R20 — Il criterio finale non è solo la mask latency.** Misurare TTFT, inter-token latency, decode wall time, throughput end-to-end e regressioni qualitative.

---

# DEFINIZIONE DI DONE GLOBALE

Il progetto può essere considerato completato quando:

- [ ] La nuova pipeline produce **mask bit-identiche** all'oracle esaustivo per tutta la suite supportata.
- [ ] Non sono presenti regressioni nei test DSML, JSON Schema, thinking boundary, tool-choice e continuation.
- [ ] La scansione completa di tutto il vocabolario non è più il normale percorso runtime.
- [ ] Il parser non ri-analizza l'intero output ad ogni candidate.
- [ ] I tratti deterministici vengono fast-forwardati senza trasformare una scelta grammaticale in una scelta forzata.
- [ ] Gli schemi/tool ripetuti riusano strutture compilate.
- [ ] Gli schemi dinamici possono essere compilati/integrati senza bloccare inutilmente il decode.
- [ ] Il costo medio del constraint engine è una piccola frazione del decode wall time.
- [ ] I casi non ottimizzabili ricadono automaticamente nell'oracle senza perdita di correttezza.
- [ ] Esiste un report finale con baseline, risultati di ogni macro-task e benchmark end-to-end.

---

# ARCHITETTURA TARGET

```text
                     request: tools / JSON Schema
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ normalize + grammar fragments│
                 └──────────────┬───────────────┘
                                │
                    cache hit?  │  cache miss
                     ┌──────────┴──────────┐
                     ▼                     ▼
             compiled fragments      JIT compilation
                     │                     │
                     └──────────┬──────────┘
                                ▼
                   incremental constraint state
                                │
                     ┌──────────┴──────────┐
                     │                     │
              deterministic          real choice
                     │                     │
                     ▼                     ▼
               fast-forward       state → allowed-set
                     │                     │
                     │         ┌───────────┼────────────┐
                     │         ▼           ▼            ▼
                     │   cached mask   PSC class   trie/dynamic
                     │         │           │        frontier
                     │         └───────────┴────────────┘
                     │                     │
                     │                     ▼
                     │                  sampling
                     │                     │
                     └──────────┬──────────┘
                                ▼
                        evaluate chosen tokens
                                │
                                ▼
                    update parser + KV/model state
```

Il percorso esaustivo esistente resta disponibile come:

```text
unknown / unsupported / debug / differential test
                    │
                    ▼
             exhaustive oracle
```

---

# MACRO 0 — Costruire baseline, oracle e benchmark differenziale

## Obiettivo

Prima di cambiare l'algoritmo, rendere misurabile in modo riproducibile:

1. dove viene speso il tempo;
2. quanti token/candidati vengono realmente esaminati;
3. quanto lavoro parser viene eseguito;
4. quale mask produce l'implementazione corrente.

La baseline attuale del server dispone già di metriche come `forced_build_ms`, `filter_ms`, `eval_ms`, `vocab_tokens`, `accepted_tokens`, `finite_allowed_tokens` e `decoded_piece_bytes`. Queste metriche devono diventare la base dell'oracle prestazionale.

## Letteratura di riferimento

### JSONSchemaBench — Geng et al., 2025

Estratto chiave dall'abstract:

> “10K real-world JSON schemas”

Il paper propone di valutare constrained decoding lungo tre dimensioni: **efficienza, coverage e qualità**, non soltanto validità sintattica.

Fonte:
- Paper: https://arxiv.org/abs/2501.10868
- Benchmark indicato dagli autori: https://github.com/guidance-ai/jsonschemabench

### Perché è rilevante per DS4

Una micro-suite costruita solo sui casi che DS4 gestisce bene rischia di ottimizzare il benchmark invece dell'algoritmo. Serve una suite sia DS4-specifica sia esterna.

## Sottotask

### 0.1 — Congelare l'oracle

- [x] Identificare il percorso esaustivo che decide la validità di ogni token.
- [x] Aggiungere una modalità `oracle_only`.
- [x] Aggiungere una modalità `compare_new_vs_oracle`.
- [x] Fare in modo che il confronto possa essere eseguito senza cambiare il token realmente campionato.
- [x] Serializzare, solo in debug, lo stato grammaticale che ha prodotto una divergenza.
- [x] Salvare il primo token differente e i relativi piece/byte.

### 0.2 — Estendere il profiling

- [x] Contare `parser_transition_count`.
- [x] Contare `parser_bytes_visited`.
- [x] Contare `candidate_tokens_tested`.
- [x] Contare `trie_nodes_visited` anche se inizialmente sarà zero.
- [x] Contare `mask_cache_hit` / `mask_cache_miss`.
- [x] Contare `grammar_compile_ms`.
- [x] Contare `grammar_jit_ms`.
- [x] Contare `constraint_state_checkpoint` / `rollback`.
- [x] Separare `forced_prefix_probe` da `sampling_mask_build`.
- [x] Misurare `constraint_cpu_ms` sovrapposto e non sovrapposto alla GPU.

### 0.3 — Costruire la suite DSML

- [ ] 1 tool, 1 parametro required.
- [ ] 1 tool, più parametri required.
- [ ] Parametri required + optional.
- [ ] Più tool con nomi che condividono prefissi.
- [ ] Tool name molto lunghi.
- [ ] Property name che condividono prefissi.
- [x] `string="true"` con testo libero.
- [ ] Parametro JSON.
- [x] `enum`.
- [x] `const`.
- [ ] Integer bounded.
- [ ] Più invoke.
- [x] Tool-choice required.
- [ ] Thinking + apertura/chiusura DSML.
- [ ] Marker DSML spezzati su più token BPE.
- [ ] Piece che contiene più di una transizione grammaticale.

### 0.4 — Costruire la suite JSON Schema

- [x] Object piatto.
- [x] Object annidato.
- [x] Array.
- [x] Array di object.
- [x] String libera lunga.
- [ ] String con escape.
- [x] Boolean/null/number/integer.
- [x] Enum corto.
- [ ] Enum grande.
- [x] Const.
- [x] Required/optional.
- [x] `additionalProperties=false`.
- [ ] `oneOf`.
- [ ] `anyOf`.
- [ ] `allOf`.
- [x] Min/max items/properties.
- [x] Min/max numeric.
- [ ] Output da 32, 128, 512 e 2K+ token quando supportato.

### 0.5 — Aggiungere benchmark esterno

- [x] Importare una selezione riproducibile di JSONSchemaBench.
- [x] Separare schema supportati e non supportati da DS4.
- [x] Non trasformare automaticamente schema non supportati per farli passare.
- [x] Registrare compilation time.
- [x] Registrare mask time.
- [x] Registrare end-to-end decode time.
- [x] Registrare percentuale di schema supportati.

## Criteri di uscita

- [x] Baseline salvata in forma machine-readable.
- [x] Ogni run può essere ripetuto.
- [x] Il nuovo codice può essere confrontato mask-per-mask con l'oracle.
- [x] Esistono test che coprono tokenizzazione problematica e non solo JSON semplice.

## Stato e retrospettiva — MACRO 0

- **Stato:** 🟨 In corso (oracle, profiling e benchmark esterno completati; matrice DSML/JSON lunga ancora parziale)
- **Data/commit di riferimento:** 2026-08-14, working tree dopo `4c3103357095ddee5dbc905a8b8ace8cc340bb1d`
- **Baseline raccolta:** `constraint-m0-oracle-slow-001` e `constraint-m0-jsonschemabench-oracle-003` in `performance-results/`.
- **Metriche principali:** corpus pinned esaminato integralmente: 9.558 schema in 10 categorie, 3.703 classificate supportate (38,74%), 5.855 classificate unsupported; 8.859 eleggibili sotto 16 KiB. Tier versionati: 12 `smoke`, 32 `safety`, 16 esempi unsupported. Il run post-fix `constraint-m5-jsonschemabench-safety-prefix8-compare-002` copre 32 schema × 2, 304 token e 380 confronti oracle: zero divergenze, 22 output completi validi e 10 prefissi bounded dichiarati `PREFIX_ONLY`. Nello stesso run 16/16 schema unsupported ricevono HTTP 400 senza avviare inferenza.
- **Cosa ha funzionato:** harness riproducibile, output semantic hash, modalità oracle/compare, tempi per fase e importer sparse-fetch pinned al commit JSONSchemaBench `ba103c73756198dd9b149ddc7db7867da7a077f6`. Gli output completi vengono ora verificati con `jsonschema`, indipendente dal validatore C; chiavi duplicate e schema sorgente meta-invalidi falliscono il gate. Tutti i 32 witness safety sono validi esternamente; i probe live verificano anche il rifiuto pre-inferenza degli schema unsupported.
- **Cosa non ha funzionato:** i primi due run esterni hanno esposto assunzioni errate, correttamente conservate come risultati negativi. Un gate completion model-driven sul caso `Github_hard/o12281.json` può saturare il budget pur avendo `{}` come witness valido; i tentativi `constraint-m5-jsonschemabench-safety-compare-001`, `...-002` e `constraint-m5-jsonschemabench-hard-exact-replay-001` sono conservati come negativi. Il gate rapido usa quindi prefissi bounded e non pretende di dimostrare il completamento dei dieci casi `PREFIX_ONLY`.
- **Bug/regressioni trovati:** default JSON Schema di `additionalProperties`; stop/control token con spelling non vuoto che potevano terminare JSON incompleto; precedente assenza di validazione indipendente dell'istanza e accettazione implicita delle chiavi JSON duplicate nell'harness. Il primo probe live unsupported (`constraint-m5-jsonschemabench-unsupported-reject-001`) ha inoltre scoperto che keyword sconosciute venivano ignorate: il server ora usa una whitelist ricorsiva e fail-closed; `...-reject-002` e il gate safety `...-compare-002` confermano 16/16 rifiuti prima dell'inferenza.
- **Decisioni prese:** nessuna riscrittura degli schemi unsupported; fallback fail-closed; output fallito salvato nell'artefatto; separazione esplicita fra completion validation e prefix differential safety.
- **Debito tecnico rimasto:** completare la matrice DSML/JSON 32–2K+ e combinatori supportati.
- **Gate:** ☑ GO
- **Note:** R15: si procede perché i criteri di uscita sono soddisfatti; l'espansione della matrice resta un task di coverage, non un prerequisito dell'oracle differenziale.

---

# MACRO 1 — Eliminare il doppio lavoro tra forced-prefix e costruzione della mask

## Obiettivo

Fare una sola analisi dell'insieme dei token validi quando il decoder deve decidere se:

1. la continuazione è deterministica;
2. esiste una scelta reale da campionare.

Nell'architettura server corrente, `build_constrained_forced_tokens(...)` tenta prima di costruire token forzati; se non ne trova, il codice passa al constrained sampler. Il primo obiettivo è evitare di interrogare due volte lo stesso spazio di candidati quando una singola analisi può produrre entrambe le informazioni.

## Letteratura di riferimento

### Efficient Guided Generation — Willard & Louf, 2023 / Outlines

Estratto chiave:

> “construction of an index over a language model's vocabulary”

Fonte:
- https://arxiv.org/abs/2307.09702

### Interpretazione per DS4

La decisione “token valido?”, la mask e l'eventuale determinismo devono derivare dalla stessa rappresentazione dell'insieme ammesso, non da passate vocabulary-wide indipendenti.

## Sottotask

### 1.1 — Definire un risultato unico dell'analisi

- [x] Definire una struttura `constraint_analysis`.
- [x] Includere `allowed_count`.
- [x] Includere `allowed_mask` o rappresentazione equivalente.
- [x] Includere `single_allowed_token` quando esiste.
- [ ] Includere eventuale `common_forced_bytes`.
- [x] Includere metriche del lavoro effettuato.
- [x] Includere flag `analysis_complete`.
- [ ] Includere fallback reason.

### 1.2 — Condividere il risultato

- [x] Fare usare `constraint_analysis` al forced path.
- [x] Fare usare lo stesso risultato al sampler.
- [x] Evitare una seconda callback validation per gli stessi token.
- [x] Non cambiare ancora l'algoritmo di validazione.
- [x] Mantenere il vecchio comportamento dietro flag.

### 1.3 — Correttezza

- [x] Confrontare la mask prima/dopo.
- [x] Confrontare forced token prima/dopo.
- [x] Verificare i casi di vera biforcazione grammaticale.
- [x] Verificare tag DSML parzialmente tokenizzati.
- [x] Verificare enum con segmentazioni BPE multiple.

### 1.4 — Performance

- [x] Misurare numero di candidate test prima/dopo.
- [x] Misurare `forced_build_ms`.
- [x] Misurare `filter_ms`.
- [x] Misurare wall time.
- [x] Verificare che la modifica non aumenti il costo del caso senza forced-prefix.

## Criteri di uscita

- [x] Nessuna divergenza dall'oracle.
- [x] Nessuna doppia passata evitabile sullo stesso set di token.
- [ ] Miglioramento misurabile nei casi constrained con scelta reale.

## Stato e retrospettiva — MACRO 1

- **Stato:** ✅ Completato come esperimento, non promosso
- **Data/commit di riferimento:** 2026-08-14, run `constraint-m1-shared-slow-001`
- **Risultato prestazionale:** media pesata −0,687% tok/s contro l'oracle slow; DSML +0,235%, JSON −1,610%.
- **Riduzione candidate visitati:** trascurabile (circa 0,003–0,007%): il probe legacy termina presto e non era una seconda scansione completa.
- **Cosa ha funzionato:** una singola mask completa può essere condivisa in modo corretto da forced decision e sampler.
- **Cosa non ha funzionato:** sostituire sempre il probe early-exit col risultato completo sposta costo nel mask build senza ridurre abbastanza il lavoro.
- **Divergenze dall'oracle:** zero.
- **Decisioni prese:** conservare `optimized` per esperimenti e riusare `constraint_analysis` nei backend trie/cache; non promuovere la sola Macro 1.
- **Gate:** ☑ ROLLBACK
- **Note:** risultato negativo conservato (R19). R15: si passa alla Macro 2 perché la premessa “doppia passata completa” è stata falsificata dalle misure; il risultato unificato rimane infrastruttura utile.

---

# MACRO 2 — Rendere DSML e JSON Schema realmente incrementali

## Obiettivo

Smettere di ricostruire o ri-parsare il prefisso per ogni candidate.

Il constraint engine deve mantenere uno **stato persistente** aggiornato una sola volta quando un token viene accettato.

Schema concettuale:

```text
state_t + accepted_piece → state_t+1
```

Per testare un candidate:

```text
checkpoint(state_t)
feed(candidate)
accept/reject
rollback()
```

Non:

```text
parse(full_output + candidate)
```

## Letteratura di riferimento

### XGrammar — Dong et al., 2024

Estratto chiave:

> “an efficient persistent stack to accelerate the context-dependent token checks”

Fonte:
- https://arxiv.org/abs/2411.15100

### Interpretazione per DS4

Il parser stack e lo stato semantico devono essere strutture runtime persistenti. Il testo già accettato non deve essere reinterpretato ad ogni token candidato.

## Sottotask

### 2.1 — Disegnare `constraint_state`

- [x] Separare stato DSML da stato JSON.
- [ ] Definire uno stato comune per checkpoint/rollback.
- [x] Conservare dialect/syntax DSML.
- [x] Conservare `mode`.
- [x] Conservare tool attivo.
- [x] Conservare proprietà attiva.
- [ ] Conservare proprietà già viste.
- [ ] Conservare required ancora mancanti.
- [x] Conservare stato string/escape.
- [x] Conservare stack JSON container.
- [ ] Conservare schema node corrente.
- [ ] Conservare array position.
- [ ] Conservare stato enum/const quando utile.

### 2.2 — API incrementale

- [x] `constraint_state_init(...)`.
- [x] `constraint_state_feed_bytes(...)`.
- [ ] `constraint_state_feed_token(...)`.
- [x] `constraint_state_checkpoint(...)`.
- [x] `constraint_state_rollback(...)`.
- [x] `constraint_state_clone_light(...)`.
- [x] `constraint_state_is_complete(...)`.
- [x] `constraint_state_can_stop(...)`.

### 2.3 — Migrare DSML

- [x] Usare lo stato incrementale per marker DSML.
- [x] Usarlo per `invoke`.
- [x] Usarlo per parametri.
- [ ] Usarlo per required/optional.
- [x] Usarlo per string body.
- [ ] Usarlo per JSON param.
- [ ] Eliminare gradualmente dipendenze dal raw completo nelle decisioni candidate-specifiche.

### 2.4 — Migrare structured JSON output

- [ ] Rimuovere la necessità normale di `raw + piece`.
- [ ] Non richiamare un parser del documento completo per ogni candidate.
- [x] Mantenere validation finale completa come barriera di sicurezza.
- [x] Verificare number frontier.
- [x] Verificare escape e Unicode.
- [x] Verificare nested containers.
- [ ] Verificare combinatori supportati.

### 2.5 — Validazione differenziale

- [x] Per ogni token della suite, confrontare incremental state vs parser esistente.
- [ ] Fuzzare sequenze di byte/token.
- [x] Testare checkpoint/rollback profondi.
- [x] Testare piece che aprono e chiudono più strutture nello stesso token.

## Criteri di uscita

- [x] Il costo candidate-specifico non cresce linearmente con tutta la lunghezza dell'output già prodotto nei casi normali.
- [x] Validation finale completa rimane presente.
- [x] Stato incrementale e oracle concordano.

## Stato e retrospettiva — MACRO 2

- **Stato:** 🟨 In corso (lexer JSON e frontier stringa DSML/JSON incrementali; semantica generale ancora ibrida)
- **Data/commit di riferimento:** 2026-08-14, run `constraint-m2-jsonschemabench-001`
- **Costo parser prima/dopo:** stringa JSON libera: constraint 8,33→2,80 s (−66%); DSML testo libero: constraint −31,13%, parser bytes −41,44%.
- **Memoria per stato/checkpoint:** stato lessicale JSON a dimensione fissa (stack limitato da `JSON_MAX_NESTING`) copiato per candidate; tracker DSML persistente già request-local.
- **Cosa ha funzionato:** DFA JSON byte-level; fast path dentro la stessa stringa semanticamente illimitata; decisione DSML sulla proprietà attiva precomputata fuori dal callback vocabulary-wide.
- **Cosa non ha funzionato:** il solo lexer, senza frontier semantica, riduceva byte ma non migliorava tok/s; la prima versione DSML rifaceva lookup schema per token e perdeva 3,19%.
- **Casi difficili trovati:** chiusura stringa, escape/Unicode, numeri incompleti, stop token con piece non vuoto e `minLength` verificato solo alla chiusura.
- **Divergenze dall'oracle:** zero nei run compare; subset esterno candidato +32,699% tok/s pesati.
- **Gate:** ☑ GO
- **Note:** R15: i criteri di uscita sono soddisfatti per i normali hot path misurati; la migrazione semantica completa di required/optional/combinatori resta esplicitamente ibrida e continua a ricadere nel parser completo.

---

# MACRO 3 — Precomputare i piece e costruire un trie byte-level del vocabolario

## Obiettivo

Smettere di eseguire:

```text
for token in entire_vocabulary:
    simulate(piece(token))
```

e passare a una visita gerarchica:

```text
walk(token_trie, parser_state)
```

Se un prefisso di byte è impossibile nello stato grammaticale corrente, l'intero sottoalbero viene eliminato.

## Letteratura di riferimento

### DOMINO — Beurer-Kellner, Fischer, Vechev, 2024

Estratto chiave:

> “fully subword-aligned fashion, while leveraging pre-computation and speculative decoding”

Fonte:
- https://arxiv.org/abs/2403.06988

### Outlines — Willard & Louf, 2023

Riferimento concettuale:
- indicizzazione del vocabolario rispetto agli stati di una macchina formale.
- https://arxiv.org/abs/2307.09702

### Interpretazione per DS4

Il trie deve essere costruito sui **piece reali del tokenizer**, non su stringhe teoriche della grammatica. La potatura deve avvenire sui byte effettivamente generabili.

## Sottotask

### 3.1 — Token piece table

- [x] Precomputare una sola volta `token_id → piece bytes`.
- [x] Conservare lunghezza.
- [x] Gestire piece vuoti/speciali.
- [x] Identificare stop/eos separatamente.
- [x] Verificare UTF-8 vs raw byte behavior del tokenizer usato.
- [x] Evitare allocazioni nel loop caldo.

### 3.2 — Trie

- [x] Costruire trie sui byte dei piece.
- [x] Ogni foglia deve poter contenere uno o più token ID.
- [x] Gestire token con piece identici se esistono.
- [x] Ottimizzare layout per locality.
- [x] Evitare pointer-heavy layout se il profiling mostra cache miss elevati.
- [x] Considerare array compatto / edge ranges / radix compression.

### 3.3 — Traversal con parser incrementale

- [ ] Applicare una transizione parser per edge.
- [x] Potare sottoalbero su transizione impossibile.
- [ ] Fare checkpoint solo quando necessario.
- [ ] Riutilizzare stato comune fra fratelli.
- [x] Raccogliere token validi alle foglie.
- [x] Gestire candidate che completano una struttura e continuano nello stesso piece.

### 3.4 — Differential mode

- [x] Costruire mask con trie.
- [x] Costruire mask con oracle esaustivo.
- [x] Confrontare bit per bit.
- [x] Loggare primo sottoalbero responsabile di divergenza.
- [ ] Fuzzare trie traversal su stati casuali validi.

### 3.5 — Benchmark

- [x] `trie_nodes_visited`.
- [x] `leaf_tokens_emitted`.
- [x] `subtrees_pruned`.
- [ ] CPU cycles.
- [ ] cache misses se disponibile.
- [x] confronto con `vocab_tokens` dell'oracle.
- [x] testare stati molto permissivi, dove il trie può potare poco.
- [x] testare stati molto restrittivi, dove il trie deve vincere molto.

## Criteri di uscita

- [x] Mask identiche all'oracle.
- [x] Nei tratti strutturali il numero di nodi visitati è molto inferiore al numero di token del vocabolario.
- [x] Nessuna allocazione per candidate nel loop caldo.

## Stato e retrospettiva — MACRO 3

- **Stato:** 🟨 In corso (backend funzionante e promosso in policy adattiva; transizione parser realmente edge-local ancora da completare)
- **Data/commit di riferimento:** 2026-08-14, run `constraint-m3-trie-slow-001` e `constraint-m3-jsonschemabench-compare-001`
- **Nodi trie:** esposti da `trie_compiled_nodes`; costruzione lazy una volta per tokenizer.
- **Memoria trie:** esposta da `trie_memory_bytes`; layout flat a indici, più una chain token per foglia duplicata.
- **Nodi visitati medi per decode step:** canonico: 4.307/7 = 615 DSML e 4.507/14 = 322 JSON nei passi confrontati.
- **Speedup mask:** JSON annidato 1.958,03→2,63 ms (−99,87%); throughput 4,11→7,07 tok/s (+72,13%). DSML throughput +2,03%, constraint CPU −8,43%.
- **Cosa ha funzionato:** pruning byte-exact, terminali duplicati, piece vuoti, mask session-owned riusata dal sampler, threshold adattivo su massimo 256 root edge.
- **Cosa non ha funzionato:** negli stati permissivi la visita di tutti i nodi interni sarebbe più costosa della scansione token; tali stati vengono riconosciuti e inviati al percorso incrementale/esaustivo.
- **Stati in cui il trie non aiuta:** private reasoning, body di stringa libera e frontier con più di 64 famiglie di primo byte ammesse.
- **Gate:** ☑ GO
- **Note:** 678 mask esterne + 42 canoniche confrontate, zero divergenze. R15: si procede perché tutti i criteri di uscita sono soddisfatti; edge-state sharing e fuzz casuale restano miglioramenti espliciti, coperti nel frattempo da test isolato e subset esterno.

---

# MACRO 4 — Separare token context-independent e context-dependent; introdurre mask precompilate

## Obiettivo

Evitare di interpretare dinamicamente token la cui validità è già determinabile dalla parte statica dello stato grammaticale.

Target:

```text
allowed_mask =
    precomputed_context_independent_mask
    + small_runtime_context_dependent_delta
```

## Letteratura di riferimento

### XGrammar — Dong et al., 2024

Estratto chiave:

> “context-independent tokens that can be prechecked and context-dependent tokens that need to be interpreted during runtime”

Fonte:
- https://arxiv.org/abs/2411.15100

### SynCode — Ugare et al., 2024

Estratto chiave:

> “offline-constructed, efficient lookup table, the DFA mask store”

Fonte:
- https://arxiv.org/abs/2403.01632

### Interpretazione per DS4

La struttura DSML statica, la sintassi JSON e molte transizioni locali possono essere trasformate in mask riusabili. Solo le parti dipendenti da stack, tool attivo, proprietà già usate o schema dinamico devono essere decise runtime.

## Sottotask

### 4.1 — Classificare lo stato

- [x] Identificare campi che influenzano la mask.
- [x] Separare stato lessicale da stato semantico.
- [ ] Definire `constraint_state_signature`.
- [x] Dimostrare quali campi possono essere esclusi dalla signature.
- [ ] Inserire assertion/debug hash per intercettare collisioni semantiche.

### 4.2 — Static mask store

- [x] Precomputare mask per stati puramente statici DSML.
- [x] Precomputare mask per stati lessicali JSON comuni.
- [ ] Precomputare delimitatori.
- [ ] Precomputare literal `true/false/null`.
- [ ] Precomputare transizioni strutturali ricorrenti.
- [x] Misurare dimensione del mask store.

### 4.3 — Dynamic frontier

- [x] Identificare token che dipendono realmente dal runtime stack/schema.
- [x] Verificare solo questi token dinamicamente.
- [ ] Usare trie per il dynamic subset se conveniente.
- [x] Misurare la cardinalità del dynamic subset per stato.

### 4.4 — Mask representation

- [ ] Confrontare bitset denso.
- [x] Confrontare sparse token list.
- [x] Confrontare base-mask + patch.
- [ ] Confrontare adaptive representation in base alla densità.
- [ ] Evitare conversioni costose prima del sampler.

### 4.5 — Cache

- [ ] Cache `state_signature → mask`.
- [ ] LRU/clock o politica adatta.
- [x] Contatori hit/miss.
- [x] Budget memoria esplicito.
- [x] Invalidazione legata a tokenizer/schema/tool registry.
- [x] No cache per stato non canonico finché non è dimostrata la sicurezza.

## Criteri di uscita

- [x] La maggioranza degli stati strutturali non richiede più una scansione dinamica completa.
- [x] Dynamic subset misurato e significativamente più piccolo di `|V|`.
- [x] Cache key dimostrata semanticamente corretta dai test.

## Stato e retrospettiva — MACRO 4

- **Stato:** 🟨 In corso (base-mask specializzata alle stringhe illimitate; store strutturale generale ancora da estendere)
- **Data/commit di riferimento:** 2026-08-14, run `constraint-m4-dsml-static-mask-slow-001` e `constraint-m4-json-static-mask-slow-001`
- **Static states compilati:** interno stringa DSML senza `<`; interno stringa JSON lessicalmente valido che non chiude la stringa.
- **Dynamic tokens medi:** DSML 598; JSON 2.651 nel workload libero (contro `|V|=248.320`).
- **Mask cache hit-rate:** 24 hit per richiesta DSML e 21 per richiesta JSON nei workload misurati; miss separati per gli stati non canonici.
- **Memoria cache:** 509.636 byte complessivi; compile cold 3,6–3,7 ms, lifetime legato al server/tokenizer.
- **Cosa ha funzionato:** base-mask + sparse dynamic patch; stop/thinking control sempre dinamici; chiusure ed escape di frontiera rivalidati; DSML +2,06% tok/s vs Macro 2 e CPU vincolo −11,39%.
- **Cosa non ha funzionato:** JSON libero è neutro (−0,19% tok/s, −0,48% CPU); scrivere comunque la mask float densa e campionare su `|V|` limita il beneficio.
- **Stati troppo dinamici:** stringhe non semanticamente illimitate, escape/Unicode pendenti, tag di chiusura parziali e qualunque stato con signature incompleta.
- **Gate:** ☑ GO
- **Note:** il run `constraint-m4-dsml-static-mask-compare-001` ha trovato 2 divergenze e viene conservato (R19). Aggiungere `tracker.pos == raw_len` alla signature ha portato `compare-002` a 76/76 mask identiche; JSON 68/68.

---

# MACRO 5 — Trasformare il forced-prefix in vero jump-forward deterministico

## Obiettivo

Quando la grammatica determina una sequenza di byte/token senza lasciare scelte semantiche al modello, evitarne il normale ciclo mask → sample ripetuto.

DS4 possiede già `build_constrained_forced_tokens(...)` e il suffix sync dei token forzati. Questo macro-task deve trasformare il concetto in una proprietà diretta dell'automa compilato, invece di scoprirlo principalmente tramite scansioni ripetute del vocabolario.

## Letteratura di riferimento

### SGLang — Zheng et al., 2023

Estratto chiave:

> “compressed finite state machines for faster structured output decoding”

Fonte:
- https://arxiv.org/abs/2312.07104

Approfondimento tecnico degli autori/progetto:
- https://www.lmsys.org/blog/2024-02-05-compressed-fsm/

### Interpretazione per DS4

Le catene di transizioni singole possono essere compresse in archi/path deterministici. Il modello torna a scegliere soltanto quando esiste una vera biforcazione.

## Sottotask

### 5.1 — Definire determinismo grammaticale

- [x] Distinguere `single token` da `single byte/string continuation`.
- [x] Distinguere più segmentazioni BPE della stessa stringa da più scelte grammaticali.
- [x] Non forzare una scelta tra optional parameter e close.
- [x] Non forzare una scelta fra tool diversi.
- [x] Non forzare enum diversi che condividono prefisso.

### 5.2 — Estrarre deterministic path

- [ ] API `constraint_next_deterministic_bytes`.
- [ ] API `constraint_next_deterministic_tokens`.
- [x] Fermarsi al primo branch.
- [x] Fermarsi su stop boundary.
- [x] Fermarsi su limite output.
- [x] Fermarsi quando la tokenizzazione non può essere preservata con certezza.

### 5.3 — Tokenizzazione corretta

- [ ] Retokenizzare in modo compatibile con il prefisso reale.
- [x] Confrontare token sequence con un tokenization oracle.
- [x] Testare path che attraversano confini BPE difficili.
- [x] Testare marker DSML con Unicode/full-width markers.
- [x] Testare enum con token multipli.

### 5.4 — Sync modello

- [x] Riutilizzare `append_exact_tokens_to_live_session(...)` o equivalente.
- [x] Misurare forced sync separatamente da discovery.
- [ ] Non calcolare logits intermedi se il backend consente di evitarli.
- [x] Mantenere KV/state esattamente allineati.
- [ ] Verificare MTP sidecar/state quando attivo.

### 5.5 — Dispatch micro-suffix

- [ ] Conservare benchmark token-exact vs layer-major.
- [ ] Non fissare per sempre `128` come costante universale.
- [ ] Rendere la soglia configurabile o derivabile dal benchmark hardware/model.
- [ ] Testare 1, 2, 4, 8, 16, 32, 64, 127, 128, 256 token.
- [ ] Valutare una piccola lookup/crossover table per backend.

## Criteri di uscita

- [ ] Discovery del determinismo non richiede normalmente una scansione completa del vocabolario.
- [x] Nessuna scelta grammaticale reale viene collassata.
- [x] Forced path produce gli stessi token ammessi dell'oracle.

## Stato e retrospettiva — MACRO 5

- **Stato:** 🟨 In corso (shadow trie JSON promosso; gate esterno JSON chiuso con esito rosso; residuo DSML free-text identificato e corretto senza ancora ripetere il caso agente)
- **Data/commit di riferimento:** 2026-08-14, run `constraint-m5-json-shadow-trie-slow-003`, `constraint-final-jsonschemabench-slow-001`, `constraint-m5-jsonschemabench-safety-prefix8-compare-002` e `constraint-m5-agent-dsml-story-300-current-001`
- **Token fast-forwardati medi:** workload canonico JSON: 39 token generati, 14 mask/decisioni iniziali; le catene shadow eliminano le scansioni intermedie prima dei 14 branch reali.
- **Sampling step eliminati:** invariati rispetto al fast-forward preesistente; cambia il costo di discovery delle catene.
- **Costo discovery:** JSON `forced_prefix_probe` 2.282,13→circa 6–7 ms; constraint CPU 2.286,20→10,61 ms.
- **Costo sync:** resta misurato separatamente come `forced_sync_ms`; nessuna modifica a KV/model sync.
- **Cosa ha funzionato:** JSON annidato 7,07→35,72 tok/s (+405% circa), vicino al decode non vincolato; 52 shadow mask canoniche confrontate nel prototipo, zero divergenze. Il gate correctness finale sul subset JSONSchemaBench originario totalizza 718 confronti oracle senza divergenze; il nuovo tier safety post-fix aggiunge 380 confronti su 32 schema, ancora zero divergenze, con validazione indipendente di 22 output completi e dei witness per tutti i 32 casi. I 16 esempi unsupported sono respinti con HTTP 400 prima dell'inferenza; questo gate ha trovato e fatto correggere l'accettazione precedente delle keyword sconosciute. Il free-text nullable `string | null` ora riusa conservativamente la mask statica solo quando esiste un'unica variante stringa semanticamente illimitata: la fixture mirata passa 80 confronti oracle senza divergenze e l'A/B migliora 26,76→29,02 tok/s (+8,46%), con constraint CPU 767,07→451,07 ms (−41,20%).
- **Cosa non ha funzionato:** applicare lo stesso shadow trie a DSML regredisceva 7,9% perché il simulatore semantico non è ancora edge-local; run `constraint-m5-shadow-trie-slow-001` conservato e policy ritirata per DSML. Il gate prestazionale finale JSONSchemaBench è `REJECT_CANDIDATE`: media pesata +73,99%, ma `Github_easy/o8466.json` regredisce del 16,94% e `Github_trivial/o25195.json` del 2,44%, oltre la soglia per workload dominante.
- **Profilo reale DSML free-text:** un solo tentativo con l'agente `/agent`, grafo `bootstrap-wiki`, contesto occupato 10.770 token e richiesta di una storia italiana di 300 parole nel campo `message` di `exit-with-info-tool`. La fase lunga genera 1.024 token a 16,25 tok/s: `eval_ms` pesa 58,68% del wall, `sampling_mask_build_ms` 40,59%, sampling filtrato 0,30%, forced probe 0,38% e residuo non attribuito 0,05%. L'output raggiunge il limite e fallisce la validazione del tool; il run è quindi conservato come `measured_failed_output`, non come successo funzionale.
- **Nuovo residuo identificato:** lo schema reale di `message` è il nullable Pydantic `anyOf: [string, null]`. Il riconoscitore della stringa libera rifiutava ogni `anyOf`, producendo 1.022 cache miss e zero hit nella fase lunga: il 40,59% non era overhead inevitabile ma copertura mancante della mask statica. Dopo la correzione, il costo dominante atteso sul caso reale è l'eval del modello; per disposizione del profiling a tentativo unico il caso agente non è stato ripetuto e questo dato resta da confermare in un futuro gate autorizzato.
- **Problemi di tokenizzazione:** nessuna divergenza osservata; common-prefix forcing resta tokenizer-exact e si arresta sulle biforcazioni.
- **Gate:** ☑ ITERATE / ☐ GO
- **Note:** il gate JSON è concluso ma rosso, quindi non si lascia MACRO 5 e non si avvia MACRO 6. R15 non è ancora applicabile: occorre risolvere o spiegare con una conferma stabile le due regressioni esterne e chiudere un output reale DSML valido. DSML resta esplicitamente ibrido finché il parser semanticamente edge-local della Macro 2 non rende conveniente il trie shadow.

---

# MACRO 6 — Decomporre DSML in grammatica statica + dispatch dinamico per tool/schema

## Obiettivo

Evitare di ricompilare un enorme automa monolitico per ogni richiesta.

Separare:

```text
STATICO
- sintassi DSML
- JSON lexical grammar
- delimitatori
- struttura invoke/parameter
- primitive schema comuni

DINAMICO
- tool names
- property names
- required set
- enum/const
- schema fragments
```

## Letteratura di riferimento

### XGrammar 2 — Li et al., 2026

Estratto chiave:

> “cross-grammar caching mechanism to leverage the common sub-structures across different grammars”

Il paper introduce anche **TagDispatch** e **JIT compilation** per workload strutturati dinamici e agentici.

Fonte:
- https://arxiv.org/abs/2601.04426

### Interpretazione per DS4

Tool calling è precisamente un caso di grammatica dinamica: il protocollo resta quasi identico, mentre cambiano nomi tool e schemi. La compilazione deve riusare i frammenti invarianti.

## Sottotask

### 6.1 — Normalizzazione dei frammenti

- [ ] Canonicalizzare schema supportati.
- [ ] Hash stabile dei frammenti.
- [ ] Separare ordine logico da dettagli di rendering quando semanticamente possibile.
- [ ] Identificare schema fragment duplicati fra tool.

### 6.2 — Static core

- [ ] Compilare una volta DSML core.
- [ ] Compilare una volta JSON lexical core.
- [ ] Compilare primitive schema comuni.
- [ ] Rendere immutabili/thread-safe le parti statiche.

### 6.3 — Dynamic dispatch

- [ ] Dispatch su tool name.
- [ ] Dispatch su property name.
- [ ] Dispatch su schema node.
- [ ] Dispatch su enum/const.
- [ ] Non duplicare tutto il core per ogni tool.

### 6.4 — Cross-grammar cache

- [ ] Cache per tool schema fragment.
- [ ] Cache per property schema fragment.
- [ ] Cache per enum.
- [ ] Cache per common object pattern.
- [ ] Reference counting o ownership chiaro.
- [ ] Budget memoria.
- [ ] Statistiche di riuso.

### 6.5 — JIT

- [ ] Percorso cold: compila solo ciò che serve al primo attraversamento.
- [ ] Percorso hot: nessuna compilazione nel loop token.
- [ ] Misurare JIT latency.
- [ ] Misurare numero di frammenti compilati ma mai usati.
- [ ] Valutare background compilation soltanto se l'architettura runtime lo consente senza bloccare la request critica.

## Criteri di uscita

- [ ] Cambiare toolset non richiede ricompilare l'intero core.
- [ ] Tool/schema ripetuti mostrano cache hit.
- [ ] Compilation time è contabilizzato separatamente dal decode.

## Stato e retrospettiva — MACRO 6

- **Stato:** ⬜ Non iniziato / 🟨 In corso / ✅ Completato / ⛔ Bloccato
- **Data/commit di riferimento:**
- **Cold compile:**
- **Hot compile:**
- **Cross-grammar cache hit-rate:**
- **Memoria compilata:**
- **Cosa ha funzionato:**
- **Cosa non ha funzionato:**
- **Frammenti con scarso riuso:**
- **Gate:** ⬜ GO / ⬜ ITERATE / ⬜ ROLLBACK
- **Note:**

---

# MACRO 7 — Implementare Parser Stack Classification (PSC) per gli stati ad alto riuso

## Obiettivo

Per stati/sottogrammatiche sufficientemente stabili, eliminare anche il costo runtime residuo proporzionale alla dimensione del vocabolario o al numero di candidate dinamici.

Target concettuale:

```text
parser stack/state
       │
       ▼
state classifier
       │
       ▼
complete vocabulary mask
```

## Letteratura di riferimento

### Efficient Grammar-Constrained Decoding via Parser Stack Classification — Li et al., 2026

Estratto chiave:

> “time complexity independent of the vocabulary size”

Il paper combina in preprocessing le condizioni di accettazione dei token in un classificatore dello stack del parser. Riporta mask computation fino a 30× più veloce per JSON schema rispetto ai baseline studiati.

Fonte:
- https://arxiv.org/abs/2608.03065

### Nota critica

PSC è molto recente. Non adottarlo come sostituzione globale prima di misurare il costo di preprocessing sul workload DS4, dove tool e schema possono cambiare ad ogni request.

## Sottotask

### 7.1 — Identificare candidati PSC

- [ ] JSON core.
- [ ] Schema ripetuti.
- [ ] Response format ricorrenti.
- [ ] Tool registry ricorrenti.
- [ ] Frammenti DSML ad alto riuso.
- [ ] Escludere inizialmente schemi one-shot molto grandi.

### 7.2 — Costruire classifier prototype

- [ ] Definire stack/state feature necessarie.
- [ ] Generare classi equivalenti di stato.
- [ ] Associare classe → mask.
- [ ] Verificare equivalenza con oracle.
- [ ] Misurare dimensione classifier.

### 7.3 — Break-even

- [ ] Misurare preprocessing time.
- [ ] Misurare memoria.
- [ ] Misurare runtime saved per token.
- [ ] Calcolare numero di riusi necessario a ripagare la compilazione.
- [ ] Definire una policy automatica `compile PSC / use trie / exhaustive fallback`.

### 7.4 — Hybrid selection

- [ ] Se classifier presente → PSC.
- [ ] Altrimenti se mask cache hit → cache.
- [ ] Altrimenti → trie/dynamic frontier.
- [ ] Se unsupported → exhaustive oracle.
- [ ] Registrare quale backend di constraint viene usato per step/request.

### 7.5 — Validazione

- [ ] Mask bit-identiche.
- [ ] Fuzz stack states.
- [ ] Stress test cache invalidation.
- [ ] Testare schema version/hash mismatch.
- [ ] Testare combinatori.

## Criteri di uscita

- [ ] PSC viene usato solo dove il break-even è positivo.
- [ ] Runtime mask non dipende da `|V|` negli stati PSC.
- [ ] Fallback 100% corretto disponibile.

## Stato e retrospettiva — MACRO 7

- **Stato:** ⬜ Non iniziato / 🟨 In corso / ✅ Completato / ⛔ Bloccato
- **Data/commit di riferimento:**
- **Classifier costruiti:**
- **Compile cost:**
- **Break-even osservato:**
- **Mask latency PSC:**
- **Memoria PSC:**
- **Cosa ha funzionato:**
- **Cosa non ha funzionato:**
- **Casi in cui PSC non conviene:**
- **Gate:** ⬜ GO / ⬜ ITERATE / ⬜ ROLLBACK
- **Note:**

---

# MACRO 8 — Sovrapporre constraint computation e inferenza; integrare speculative decoding solo dopo

## Obiettivo

Una volta ridotto il costo algoritmico del constraint engine, togliere dal critical path la quota CPU residua sovrapponendola all'eval GPU quando possibile.

Solo successivamente valutare constrained speculative/MTP.

## Letteratura di riferimento

### XGrammar — Dong et al., 2024

Il paper descrive il co-design fra grammar engine e inference engine per sovrapporre il grammar processing all'esecuzione GPU.

Fonte:
- https://arxiv.org/abs/2411.15100

### DOMINO — Beurer-Kellner et al., 2024

DOMINO combina pre-computation, allineamento subword e speculative decoding.

Fonte:
- https://arxiv.org/abs/2403.06988

### Interpretazione per DS4

Non usare MTP per nascondere un constraint engine ancora inefficiente: prima ridurre il lavoro; poi sovrapporlo e infine speculare.

## Sottotask

### 8.1 — Overlap CPU/GPU

- [ ] Disegnare timeline `sample/mask` vs `eval`.
- [ ] Identificare computation che può partire mentre la GPU valuta il token precedente.
- [ ] Precomputare lo stato successivo candidato quando sicuro.
- [ ] Evitare lock globali sul constraint engine.
- [ ] Misurare tempo CPU nascosto vs esposto.

### 8.2 — Double buffering

- [ ] Buffer mask corrente.
- [ ] Buffer mask successiva.
- [ ] Ownership chiaro.
- [ ] Nessuna race con schema/cache invalidation.
- [ ] Fallback sincrono.

### 8.3 — Speculative prerequisites

- [ ] Constraint state con checkpoint economico.
- [ ] Feed di più token in sequenza.
- [ ] Rollback di draft non accettato.
- [ ] Validazione batch di draft token.
- [ ] Nessuna divergenza di tokenizer state.

### 8.4 — Constrained MTP

- [ ] Fare draft MTP.
- [ ] Intersecare/validare draft con constraint state.
- [ ] Accettare prefisso valido.
- [ ] Rollback al primo token non valido.
- [ ] Misurare acceptance rate.
- [ ] Misurare costo constraint per draft token.
- [ ] Disabilitare automaticamente se il vincolo rende la speculation non conveniente.

### 8.5 — Benchmark

- [ ] Unconstrained baseline.
- [ ] Constrained senza overlap.
- [ ] Constrained con overlap.
- [ ] Constrained + MTP.
- [ ] Separare workload strutturale da string payload libero.

## Criteri di uscita

- [ ] L'overlap riduce wall time, non solo CPU-accounted time.
- [ ] Speculation è attivata solo quando migliora end-to-end throughput/latency.
- [ ] Rollback constraint e rollback modello restano coerenti.

## Stato e retrospettiva — MACRO 8

- **Stato:** ⬜ Non iniziato / 🟨 In corso / ✅ Completato / ⛔ Bloccato
- **Data/commit di riferimento:**
- **CPU constraint nascosta:**
- **GPU idle prima/dopo:**
- **MTP acceptance rate:**
- **Speedup end-to-end:**
- **Cosa ha funzionato:**
- **Cosa non ha funzionato:**
- **Casi in cui MTP peggiora:**
- **Gate:** ⬜ GO / ⬜ ITERATE / ⬜ ROLLBACK
- **Note:**

---

# MACRO 9 — Hardening, adaptive policy e rimozione del full-scan dal percorso normale

## Obiettivo

Integrare i backend sviluppati in una policy adattiva e dimostrare che la scansione completa del vocabolario è diventata un fallback eccezionale.

Policy target:

```text
deterministic path?
    yes → jump-forward
    no  ↓

PSC classifier available and profitable?
    yes → PSC mask
    no  ↓

state mask cache hit?
    yes → cached mask
    no  ↓

compiled/trie dynamic path supported?
    yes → trie + context-dependent checks
    no  ↓

exhaustive oracle fallback
```

## Letteratura di riferimento

Questa macro-task sintetizza i risultati di:

- Outlines / vocabulary index: https://arxiv.org/abs/2307.09702
- SynCode / DFA mask store: https://arxiv.org/abs/2403.01632
- DOMINO / subword alignment + precomputation: https://arxiv.org/abs/2403.06988
- SGLang / compressed FSM: https://arxiv.org/abs/2312.07104
- XGrammar / context-independent vs dependent + persistent stack + overlap: https://arxiv.org/abs/2411.15100
- XGrammar 2 / TagDispatch + JIT + cross-grammar cache: https://arxiv.org/abs/2601.04426
- PSC / parser stack classification: https://arxiv.org/abs/2608.03065
- JSONSchemaBench / evaluation: https://arxiv.org/abs/2501.10868

## Sottotask

### 9.1 — Adaptive backend selection

- [ ] Definire cost model.
- [ ] Considerare compilation amortization.
- [ ] Considerare cache hit-rate.
- [ ] Considerare dynamic frontier size.
- [ ] Considerare permissività dello stato.
- [ ] Considerare lunghezza deterministic path.
- [ ] Loggare backend scelto e motivazione.

### 9.2 — Safety

- [ ] Oracle fallback sempre disponibile.
- [ ] Circuit breaker su divergence in debug/canary.
- [ ] Cache checksum/version.
- [ ] Tokenizer fingerprint.
- [ ] Grammar compiler version.
- [ ] Schema hash.
- [ ] Tool registry hash.
- [ ] Limiti di memoria.
- [ ] Limiti di nesting.
- [ ] Limiti di compilation.

### 9.3 — End-to-end regression suite

- [ ] Tutta la suite DSML.
- [ ] Tutta la suite JSON.
- [ ] JSONSchemaBench subset.
- [ ] Multi-turn tool continuation.
- [ ] Exact DSML replay/cache path.
- [ ] Thinking.
- [ ] Streaming.
- [ ] OpenAI-compatible tool calls.
- [ ] Responses.
- [ ] Anthropic path se supportato.
- [ ] Cold/hot cache.
- [ ] Long context + short suffix.
- [ ] Batch/multi-session.

### 9.4 — Performance gate

- [ ] Confronto con baseline iniziale.
- [ ] Constraint CPU per output token.
- [ ] p50/p95/p99 inter-token latency.
- [ ] TTFT.
- [ ] Decode wall time.
- [ ] Throughput.
- [ ] Percentuale step PSC.
- [ ] Percentuale cache.
- [ ] Percentuale trie.
- [ ] Percentuale jump-forward.
- [ ] Percentuale exhaustive fallback.
- [ ] Break-even compilation.

### 9.5 — Cleanup

- [ ] Rimuovere duplicazioni non più necessarie.
- [ ] Conservare oracle in build debug/test o fallback controllato.
- [ ] Documentare invarianti.
- [ ] Documentare cache keys.
- [ ] Documentare ownership/thread safety.
- [ ] Documentare formato metriche.
- [ ] Scrivere troubleshooting guide.
- [ ] Congelare benchmark finale.

## Criteri di uscita

- [ ] Exhaustive full-vocabulary scan non è più il path normale.
- [ ] Nessuna regressione di correttezza.
- [ ] Il sistema seleziona automaticamente il backend più economico disponibile.
- [ ] Tutte le metriche finali sono confrontabili con MACRO 0.
- [ ] È disponibile un report conclusivo.

## Stato e retrospettiva — MACRO 9

- **Stato:** ⬜ Non iniziato / 🟨 In corso / ✅ Completato / ⛔ Bloccato
- **Data/commit di riferimento:**
- **Speedup end-to-end finale:**
- **Overhead vs unconstrained:**
- **Percentuale exhaustive fallback:**
- **p50/p95/p99:**
- **Memoria aggiuntiva:**
- **Compilation overhead:**
- **Cosa ha funzionato:**
- **Cosa non ha funzionato:**
- **Rischi residui:**
- **Debito tecnico residuo:**
- **Gate finale:** ⬜ SHIP / ⬜ ITERATE / ⬜ ROLLBACK
- **Note:**

---

# ORDINE VINCOLANTE CONSIGLIATO

```text
MACRO 0  Baseline + oracle
   ↓
MACRO 1  Una sola analisi forced/mask
   ↓
MACRO 2  Parser incrementale
   ↓
MACRO 3  Token trie
   ↓
MACRO 4  Static/dynamic masks + cache
   ↓
MACRO 5  Jump-forward deterministico
   ↓
MACRO 6  Dynamic grammar decomposition + cross-grammar JIT/cache
   ↓
MACRO 7  PSC sugli stati/schema ad alto riuso
   ↓
MACRO 8  Overlap + constrained MTP/speculation
   ↓
MACRO 9  Adaptive policy + hardening + production gates
```

Non anticipare MACRO 7/8 se MACRO 2–4 non sono corretti e misurati: PSC e speculative decoding possono nascondere o moltiplicare un costo parser che dovrebbe essere eliminato alla radice.

---

# METRICHE DA MANTENERE FINO ALLA FINE

## Correttezza

- [ ] Mask equality rate.
- [ ] False allowed token count = 0.
- [ ] False rejected token count = 0.
- [ ] Structured output validation rate.
- [ ] Tool-call parse success.
- [ ] Differential fuzz failures.

## Constraint engine

- [ ] `constraint_cpu_ms`.
- [ ] `parser_transition_count`.
- [ ] `parser_bytes_visited`.
- [ ] `candidate_tokens_tested`.
- [ ] `trie_nodes_visited`.
- [ ] `subtrees_pruned`.
- [ ] `dynamic_frontier_size`.
- [ ] `mask_cache_hit_rate`.
- [ ] `psc_hit_rate`.
- [ ] `jump_forward_tokens`.
- [ ] `exhaustive_fallback_steps`.

## Compilation

- [ ] `compile_ms`.
- [ ] `jit_ms`.
- [ ] `compiled_bytes`.
- [ ] `cross_grammar_cache_hit_rate`.
- [ ] `break_even_reuse_count`.

## Inference end-to-end

- [ ] TTFT.
- [ ] Inter-token latency p50/p95/p99.
- [ ] Decode wall time.
- [ ] Output tok/s.
- [ ] Requests/s.
- [ ] GPU utilization.
- [ ] CPU utilization.
- [ ] GPU idle attributable to constraint processing.

---

# FONTI PRINCIPALI

1. **Willard, Louf — Efficient Guided Generation for Large Language Models (Outlines)**  
   https://arxiv.org/abs/2307.09702

2. **Zheng et al. — SGLang: Efficient Execution of Structured Language Model Programs**  
   https://arxiv.org/abs/2312.07104

3. **Ugare et al. — SynCode: LLM Generation with Grammar Augmentation**  
   https://arxiv.org/abs/2403.01632

4. **Beurer-Kellner, Fischer, Vechev — Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation (DOMINO)**  
   https://arxiv.org/abs/2403.06988

5. **Dong et al. — XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models**  
   https://arxiv.org/abs/2411.15100

6. **Geng et al. — Generating Structured Outputs from Language Models: Benchmark and Studies (JSONSchemaBench)**  
   https://arxiv.org/abs/2501.10868  
   https://github.com/guidance-ai/jsonschemabench

7. **Li et al. — XGrammar 2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs**  
   https://arxiv.org/abs/2601.04426

8. **Li et al. — Efficient Grammar-Constrained Decoding via Parser Stack Classification (PSC)**  
   https://arxiv.org/abs/2608.03065

9. **SGLang/LMSYS — Fast JSON Decoding with Compressed FSM / Jump-Forward Decoding**  
   https://www.lmsys.org/blog/2024-02-05-compressed-fsm/

---

# TEMPLATE REPORT FINALE

Da compilare al termine di MACRO 9.

## Risultato complessivo

- **Commit/versione:**
- **Modello/tokenizer:**
- **Hardware:**
- **Baseline decode wall:**
- **Nuovo decode wall:**
- **Speedup:**
- **Overhead constrained vs unconstrained:**
- **Mask latency prima/dopo:**
- **TTFT prima/dopo:**
- **p50/p95/p99 prima/dopo:**
- **Memoria aggiuntiva:**
- **Compile/JIT overhead:**
- **Exhaustive fallback rate:**

## Contributo di ogni macro-task

| Macro | Modifica | Speedup incrementale | Memoria | Correttezza | Decisione |
|---|---|---:|---:|---|---|
| 0 | Baseline/oracle | — | — | — | |
| 1 | Single analysis forced/mask | | | | |
| 2 | Incremental parser | | | | |
| 3 | Token trie | | | | |
| 4 | Static/dynamic mask cache | | | | |
| 5 | Jump-forward | | | | |
| 6 | Dynamic JIT/cross-grammar cache | | | | |
| 7 | PSC | | | | |
| 8 | Overlap/MTP | | | | |
| 9 | Adaptive production path | | | | |

## Conclusioni

- **Ottimizzazione con maggiore impatto:**
- **Ottimizzazione con peggiore rapporto costo/beneficio:**
- **Collo di bottiglia rimasto:**
- **Regressioni note:**
- **Casi ancora in exhaustive fallback:**
- **Lavoro futuro:**
