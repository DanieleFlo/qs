# PROTOCOLLO DEFINITIVO — CUDA Q8_1 / MMVQ, PARITY LLAMA.CPP E OTTIMIZZAZIONE QWEN3.6 27B

## RUOLO

Agisci come un senior CUDA/LLM inference engineer con competenza specifica in:

* `llama.cpp` / `ggml-cuda`;
* quantizzazione GGUF K-quants;
* Q4_K / Q5_K / Q6_K;
* Q8_1 / Q8_K;
* MMVQ / MMQ;
* `dp4a`;
* warp-level programming;
* FP16 / FP32 numerical behavior;
* CUDA compiler semantics;
* FMA contraction;
* reduction topology;
* profiling Ampere;
* reverse engineering di kernel ad alte prestazioni.

Il tuo compito non è trovare una spiegazione plausibile.

Il tuo compito è stabilire **la verità causale** tramite esperimenti falsificabili che cambino una variabile alla volta.

Non partire dall'assunzione che DS4 sia corretto.

Non partire dall'assunzione che DS4 sia sbagliato.

Non partire dall'assunzione che Q8_1 sia necessariamente l'unica causa.

---

# TARGET

Modello:

**Qwen3.6 27B Q4_K_S**

Nota:

i simboli `qwen35_*` presenti nel codice sono nomi storici e NON significano che il modello analizzato sia Qwen3.5.

Hardware target:

**NVIDIA RTX 3090 / GA102 / sm_86**

Configurazione:

* MTP disabilitato;
* modello interamente residente su GPU;
* KV non quantizzata;
* greedy / argmax;
* stesso GGUF;
* stesso prompt/template;
* stessa tokenizzazione;
* stesso contesto;
* stessi token precedenti;
* stessa logica teacher-forced quando richiesta.

Gate di correttezza inderogabile:

* **8/8 sequenze**
* **512/512 argmax**

Non accettare:

* 7/8;
* 511/512;
* "quasi identico";
* workaround prompt-specifici.

---

# RIFERIMENTO LLAMA.CPP

Il riferimento prestazionale e comportamentale è il commit esatto:

`1a064ab09`

Performance osservata:

**~40.00 tok/s**

Non utilizzare il `master` corrente di llama.cpp come sostituto del commit storico per stabilire:

* rounding;
* layout;
* quantizer;
* dispatcher;
* kernel selection;
* compiler behavior;
* semantica MMVQ/MMQ.

Il `master` può essere usato esclusivamente come riferimento secondario o per comprendere l'evoluzione del codice.

Ogni conclusione di parity deve riferirsi al checkout esatto:

`1a064ab09`

---

# STATO PRESTAZIONALE DS4 GIÀ OSSERVATO

Baseline F32:

**~17.20 tok/s**

Q8_K diagnostico:

**~24.65 tok/s**

Q8_1 + Q4_K/Q5_K MMVQ:

**~34.50 tok/s**

Caratteristiche osservate del percorso ~34.5 tok/s:

* ~40 registri/thread;
* ~384 B shared;
* zero spill;
* probe sintetici Q4/Q5 numericamente buoni.

Ma il gate diventa:

* **7/8**
* **484/512**

La prima divergenza nota è:

`multi_turn_preserve_thinking`

greedy step:

**3**

Il margine baseline tra i due logits principali in quel punto è circa:

**0.0127**

---

# EVIDENZA Q6_K GIÀ DISPONIBILE

È stato eseguito un test nel quale:

* Q4_K resta F32;
* Q5_K resta F32;
* soltanto l'output head Q6_K usa Q8_1.

Risultato:

* ~18.06 tok/s;
* probe Q6_K contro oracolo CPU scalare: PASS;
* cosine = 1;
* max_abs ~3.81e-5;
* 36 registri;
* zero spill;
* gate ancora 7/8;
* ancora 484/512;
* stesso greedy step 3 ribaltato.

Interpretazione corretta:

> Una singola introduzione di Q8_1 sull'output head è SUFFICIENTE a causare il flip.

Interpretazione NON consentita:

> Questo dimostra che Q8_1 è necessariamente l'unica discrepanza dell'intero port.

Il probe CPU rende inoltre poco probabile un errore grossolano nell'unpacking Q6_K, ma NON sostituisce una parity cross-backend contro il commit llama.cpp.

---

# FATTI GIÀ DIMOSTRATI DAL CODICE DS4

## Q8_1 / MMVQ esiste già

Sono già presenti:

* `qwen35_q8_1_quantize_kernel`;
* `qwen35_dot_q4_k_q8_1_mmvq`;
* `qwen35_dot_q5_k_q8_1_mmvq`;
* `qwen35_dot_q6_k_q8_1_mmvq`;
* `qwen35_qk_q8_1_mmvq_kernel`;
* `qwen35_q6_k_q8_1_mmvq_kernel`.

Sono collegati al dispatch decode e il percorso Q4/Q5 è quello che ha già raggiunto ~34.5 tok/s.

NON proporre di "implementarli" come nuovo lavoro.

Il MMVQ esistente è un asset da preservare finché una parity non lo incrimina.

Sono ammessi clone diagnostici minimi che cambino una sola variabile.

---

# `ds.y`: COSA SAPPIAMO E COSA NON SAPPIAMO

Nel MMVQ decode DS4 Q4_K/Q5_K:

* viene letto `ds.x` tramite `__low2half`;
* `qsum` viene ricostruito dagli `int8` con `dp4a`;
* `ds.y` NON entra nel dataflow del dot product.

Pertanto:

> `ds.y` NON è una spiegazione causale valida del flip nel MMVQ decode locale.

Non ritornare su questa pista salvo dimostrare che durante il caso analizzato venga effettivamente eseguito un altro kernel che lo consuma.

Tuttavia, durante l'audit del formato:

* documentare comunque la semantica di `ds.y`;
* verificare separatamente `qsum_gpu == Σ qs_cpu`;
* verificare cosa contiene `ds.y` nel commit storico.

NON richiedere byte-parity di `ds.y` se DS4 e llama.cpp gli assegnano intenzionalmente semantiche differenti.

Inoltre:

`qsum` è una somma intera.

Con 32 elementi in circa `[-127,+127]`:

`|qsum|max = 4064`

quindi:

* non c'è rischio di overflow `int32`;
* l'ordine della riduzione intera non introduce roundoff;
* alberi differenti devono dare lo stesso risultato se sommano gli stessi `qs`.

Una differenza di `qsum` indica quindi:

* lane coverage errata;
* indexing errato;
* dati differenti;

non una diversa associatività numerica.

---

# PRINCIPIO EPISTEMOLOGICO

Distinguere sempre:

## Causa sufficiente

Una modifica da sola è capace di produrre il problema.

## Causa necessaria

Senza quella modifica il problema non può verificarsi.

## Causa esclusiva

Non esiste nessun'altra discrepanza rilevante.

Il test Q6-only dimostra sufficienza.

NON dimostra necessità o esclusività.

Ogni conclusione deve essere classificata come:

* dimostrata;
* fortemente supportata;
* plausibile;
* esclusa;
* ancora aperta.

---

# PHASE 0D — ACTUAL LLAMA.CPP DISPATCH VERIFICATION

Questa fase deve precedere l'interpretazione delle parity cross-backend.

Prima di dire:

> "DS4 MMVQ differisce dal MMVQ llama.cpp"

stabilisci cosa llama.cpp ha REALMENTE eseguito nel benchmark di riferimento.

Sul checkout esatto `1a064ab09`, determina per batch/decode `n_tok = 1`:

* funzione C++ di dispatch;
* quantizer effettivamente invocato;
* kernel CUDA effettivamente lanciato;
* MMVQ vs MMQ vs altra implementazione;
* eventuale cuBLAS fallback;
* eventuali Tensor Core;
* geometria grid/block;
* activation layout;
* compile-time switches;
* runtime switches;
* compute capability dependent branches.

Non assumere che:

`MMQ == Tensor Core`

in ogni percorso.

Non assumere che batch=1 implichi automaticamente MMVQ.

Non trasferire il comportamento del `master` corrente al commit storico.

## Metodo consigliato

Usare una o più delle seguenti tecniche:

* instrumentazione temporanea del dispatcher;
* logging delle branch selezionate;
* Nsight Systems per acquisire i nomi reali dei kernel;
* Nsight Compute dove necessario;
* simboli CUDA / profiler timeline;
* build diagnostica del commit esatto.

Non assumere l'esistenza di una particolare macro debug senza verificarla nel commit.

## Output Phase 0D

Produrre una tabella:

| Tensor/operation | Shape | n_tok | Weight type | Quantizer | Kernel reale | MMVQ/MMQ/altro | Compile flags |
| ---------------- | ----: | ----: | ----------- | --------- | ------------ | -------------- | ------------- |

Questa tabella diventa ground truth per tutte le parity successive.

---

# PHASE 0 — ESPERIMENTO A/B/C SUL VERO FALLIMENTO

Riprodurre il greedy step 3 problematico tramite teacher forcing.

Confrontare:

## A — DS4 F32

Percorso baseline corretto.

## B — DS4 Q8_1

Percorso ~34.5 tok/s.

## C — llama.cpp `1a064ab09`

Con il kernel/dispatcher reale identificato nella Phase 0D.

Usare rigorosamente:

* stesso GGUF;
* stessi token;
* stessa sequenza precedente;
* stesso prompt/template;
* stesso context position;
* stessa logica di masking;
* MTP off;
* configurazione GPU equivalente.

---

# DUE LIVELLI DI A/B/C

Per evitare di confondere propagazione dello stato e singola proiezione, eseguire quando possibile due confronti.

## A/B/C end-to-end teacher-forced

Ogni engine calcola normalmente il proprio stato fino al passo critico, forzando gli stessi token precedenti.

Questo misura la divergenza complessiva realmente presente.

## A/B/C frozen-input

Catturare lo STESSO hidden vector F32 immediatamente prima della proiezione critica e fornirlo ai primitive candidati.

Questo isola la singola proiezione dall'eventuale drift precedente.

I due risultati non vanno confusi.

---

# LOGGING OBBLIGATORIO DEI LOGITS

Per i due token critici registrare:

`z1_F32`
`z2_F32`

e per ogni candidato:

`z1_candidate`
`z2_candidate`

Definire:

`δ1 = z1_candidate - z1_F32`

`δ2 = z2_candidate - z2_F32`

`δ_common = (δ1 + δ2) / 2`

`δ_diff = δ1 - δ2`

e:

`margin = z1 - z2`

`Δmargin = margin_candidate - margin_F32`

La metrica direttamente responsabile del flip è soprattutto:

**`δ_diff` / `Δmargin`**

Un puro common-mode shift dei due logits non cambia l'argmax.

Se economico, registrare inoltre sull'intero top-k o vocabolario:

* media `Δlogit`;
* deviazione standard;
* bias;
* max_abs;
* percentile error.

Questo permette di distinguere:

* shift globale;
* errore differenziale;
* outlier specifici;
* compressione/espansione dello spettro dei logits.

---

# INTERPRETAZIONE A/B/C

## Caso 1

`B ≈ C != A`

Conclusione supportata:

DS4 Q8_1 replica sostanzialmente la semantica Q8 del riferimento llama.cpp.

Il problema diventa quindi:

> progettare una rappresentazione più accurata del Q8_1 originale senza perdere il vantaggio prestazionale MMVQ.

In questo caso LSQ/G16/residui sono miglioramenti numerici rispetto al riferimento, non correzioni del port.

---

## Caso 2

`C ≈ A != B`

STOP.

DS4 non replica ancora il riferimento.

NON progettare nuove rappresentazioni.

Localizzare prima la discrepanza.

---

## Caso 3

A/B/C tutti significativamente differenti

Non dedurre la causa.

Procedere con primitive parity e quantizer parity.

---

# PHASE 1 — PRIMITIVE PARITY CON Q8 PREQUANTIZZATO IDENTICO

Eliminare il quantizzatore come variabile.

Creare una sola rappresentazione Q8 e alimentarla a:

* primitive DS4;
* primitive/reference del percorso realmente usato da llama.cpp.

Confrontare almeno separatamente:

* Q4_K × Q8;
* Q5_K × Q8;
* Q6_K × Q8.

Usare gli stessi:

* `qs`;
* scale;
* pesi;
* block indices;
* input layout.

---

# AUDIT Q4_K/Q5_K

Verificare:

* `iqs`;
* `bq8_offset`;
* scale unpack;
* min unpack;
* signedness;
* nibble extraction;
* high bits Q5;
* `dp4a`;
* weight block indexing;
* Q8 block indexing.

---

# AUDIT Q6_K DEDICATO

Q6_K merita un controllo dedicato perché:

* è il primitive più complesso;
* il Q6-only riproduce il flip.

NON partire però dall'assunzione che sia il colpevole.

Il probe CPU scalare esistente rende poco probabile un errore grossolano di unpacking.

Verificare comunque cross-backend:

* `iqs`;
* `bq8_offset`;
* `scale_offset`;
* indice `qh`;
* `vh_shift`;
* low bits;
* high bits;
* ricostruzione signed 6-bit;
* Q8 block selection;
* scale selection.

Nel DS4 attuale distinguere correttamente:

`bq8_offset`

da:

`8*(iqs/16) + iqs%8`

che è invece parte dell'indicizzazione `qh`.

Confrontare se possibile anche il valore Q6 ricostruito prima del `dp4a`, bit per bit.

---

# FMA E REDUCTION TOPOLOGY

Non imporre `--fmad=false` come ground truth.

Prima recuperare dal build del commit `1a064ab09`:

* NVCC flags;
* CMake options;
* fast-math;
* FMA contraction;
* eventuali intrinsic espliciti;
* FTZ;
* precision flags.

Poi replicare il comportamento del riferimento.

La parity primaria deve confrontare:

> build semantics reali del riferimento

contro:

> build DS4 equivalente.

Solo DOPO eseguire come ablazione:

* `--fmad=false`;
* oppure `__fmul_rn` + `__fadd_rn`;
* eventualmente `__fmaf_rn` esplicito.

Obiettivo:

quantificare il contributo della contraction.

NON assumere che "no-FMA" sia matematicamente più corretto per il progetto.

Se il riferimento usa FMA, la FMA fa parte della sua ground truth.

---

# RIDUZIONI FP32

Confrontare:

* `__shfl_down_sync`;
* `__shfl_xor_sync`;
* ordine intra-warp;
* ordine inter-warp;
* shared-memory merge;
* numero di warp;
* row scheduling.

L'addizione FP32 non è associativa.

Una topologia differente può quindi produrre roundoff differente senza implicare un errore logico.

Distinguere:

## Mapping error

Gli stessi termini non vengono sommati.

da:

## Reduction-order difference

Gli stessi termini vengono sommati in ordine differente.

Solo il primo è necessariamente un bug.

---

# OUTPUT PHASE 1

Per primitive:

* max_abs;
* MAE;
* RMSE;
* cosine;
* bias;
* ULP distance dove utile;
* intermedi;
* classificazione della differenza.

Classificare ogni mismatch come:

* exact;
* expected FP32 roundoff;
* quantizer policy;
* reduction topology;
* mapping discrepancy;
* unexplained.

Se la primitive parity è verde:

> NON ridisegnare MMVQ.

---

# PHASE 2 — QUANTIZER PARITY BYTE-PER-BYTE

Confrontare il quantizzatore DS4 contro quello REALMENTE utilizzato dal percorso llama.cpp del commit `1a064ab09`.

Usare il vero hidden tensor del passo critico.

Non limitarsi a synthetic probes.

Per ogni blocco registrare:

* input F32;
* `amax`;
* scala F32 pre-storage;
* scala memorizzata;
* `qs[32]`;
* metadata;
* eventuale `qsum`.

---

# ROUNDING

Verificare esattamente quale operazione usa il commit storico:

* `roundf`;
* `__float2int_rn`;
* cast;
* intrinsics;
* PTX risultante se necessario.

Non assumere che i tie `.5` siano frequenti.

Non assumere che siano rari.

CONTARLI.

Per ogni mismatch:

* indice;
* valore originale;
* `x/d`;
* distanza dal nearest half-integer;
* q DS4;
* q llama;
* `Δq`.

Riportare:

* quant totali;
* mismatch totali;
* mismatch %;
* histogram `Δq`.

---

# RANGE E CAST

Non assumere che un cast verso `int8_t` sia saturating.

Verificare gli interi prima del cast.

Confrontare:

* clamp esplicito DS4;
* comportamento effettivo del commit;
* range realmente raggiunto.

---

# `qsum` E `ds.y`

Verificare direttamente:

`qsum_gpu == Σqs_cpu`

Questo è un sanity check per:

* lane coverage;
* indexing;
* write correctness.

NON usare `ds.y` come prova della topologia della riduzione intera.

Documentare separatamente la semantica del metadata nel commit storico.

`ds.y` rimane non causale nel MMVQ decode DS4 finché il dataflow non dimostra il contrario.

---

# PHASE 3 — ABLAZIONE PURA DELLA SCALA FP16

Non usare semplicemente:

`Q8_32 kernel`

contro:

`Q8_1 MMVQ`

come prova causale.

I due kernel hanno differenti:

* geometrie;
* reduction topology;
* scheduling;
* accumulation order.

Serve un esperimento che cambi UNA SOLA variabile.

---

# ESPERIMENTO `d_float` VS `d_half`

Quantizzare una sola volta.

Congelare:

* stessi `qs`;
* stessi pesi;
* stessi indici;
* stesso primitive;
* stesso ordine di accumulo.

Per ogni blocco conservare:

`d_float = scala F32 prima del cast`

e:

`d_half = float(half(d_float))`

Eseguire lo stesso dot due volte:

## Variante A

usa `d_float`

## Variante B

usa `d_half`

Tutto il resto identico.

Misurare la propagazione:

`scale error`
→ `block dot error`
→ `projection error`
→ `logit error`
→ `δdiff`
→ `Δmargin`

Questo isola precisamente il contributo della quantizzazione FP16 della scala.

---

# DECISIONE DOPO PHASE 3

Se il solo `d_float -> d_half` spiega una quota dominante dell'errore differenziale critico:

GO:

studiare una variante Q8 con scala FP32 o metadata più precisi.

Se invece l'effetto è piccolo:

NO-GO:

non spendere tempo a trasformare Q8_1 in Q8_1-FP32 come soluzione principale.

---

# PHASE 4 — LOCALIZZAZIONE Q4 / Q5 / Q6

Aggiungere, se necessario, flag diagnostici temporanei per eseguire:

1. solo Q4_K con Q8_1;
2. solo Q5_K con Q8_1;
3. Q4_K + Q5_K con Q8_1;
4. solo Q6_K output head con Q8_1.

Q6-only è già stato eseguito e va usato come dato.

Non creare whitelist permanenti.

Interpretazione:

se primitive parity è già verde, un:

`Q4-only FAIL`

NON significa automaticamente:

`Q4 unpack bug`.

Può significare:

> maggiore sensibilità delle proiezioni Q4 al rumore di activation quantization.

Separare sempre:

* correctness del primitive;
* numerical sensitivity della proiezione.

---

# PHASE 5 — PRECISION STAIRCASE

Usare le rappresentazioni diagnostiche già disponibili:

* F32;
* Q8_K / 256;
* Q8_128 / 128;
* Q8_32 / 32 + scala FP32;
* Q8_1 / 32 + scala FP16;
* Q6-only Q8_1.

Sul medesimo stato critico misurare:

* activation reconstruction error;
* projection error;
* hidden error;
* logit error;
* `δcommon`;
* `δdiff`;
* `Δmargin`.

Lo scopo è distinguere:

* granularità;
* metadata precision;
* intrinsic INT8 error;
* projection sensitivity.

Q8_32 rimane un ottimo diagnostico, ma non una ablazione causale pura della scala.

---

# NOTA SU UN PRECEDENTE TEST FMA/KAHAN

Esistono già test diagnostici in cui:

* Kahan;
* `-fmad=false`;

hanno ridotto solo modestamente un errore Q8 osservato in un trace differente.

Questo è un indizio contro FMA/reduction come causa dominante.

NON è però una conclusione causale sul flip reale perché quel test:

* non utilizzava il vero prompt critico;
* non utilizzava il vero step 3.

Ripetere le ablazioni soltanto sullo stato corretto se necessario.

---

# PHASE 6 — Q8_1-32-LSQ OFFLINE

Solo dopo parity.

Nel repository esiste già un quantizzatore Q8_K LSQ.

Riutilizzare la stessa idea per blocchi Q8_1 da 32.

Baseline:

`d0 = maxabs / 127`

`q0 = R(x / d0)`

Raffinamento:

`d1 = Σ(x*q0) / Σ(q0²)`

poi:

`q1 = R(x / d1)`

IMPORTANTE:

`R()` deve essere la stessa policy di rounding stabilita dalla parity.

Non cambiare contemporaneamente:

* scale selection;
* rounding.

Altrimenti l'esperimento non è attribuibile.

---

# METRICHE LSQ

Non giudicare LSQ soltanto da MAE/L2.

Misurare:

* activation MAE;
* activation RMSE;
* max_abs;
* projection error;
* logit top1;
* logit top2;
* `δcommon`;
* `δdiff`;
* `Δmargin`.

Una quantizzazione con MAE migliore ma `Δmargin` peggiore deve essere respinta.

---

# DECISIONE LSQ

Se LSQ offline NON migliora significativamente il margine critico:

NO-GO.

Non scrivere il kernel CUDA.

Se LSQ migliora chiaramente:

GO.

Implementare Q8_1-32-LSQ CUDA mantenendo il più possibile invariati:

* layout;
* MMVQ;
* weight decode;
* hot loop.

Cambiare preferibilmente soltanto il quantizzatore.

---

# COSTO LSQ

Misurare separatamente:

* tempo quantizzazione baseline;
* tempo quantizzazione LSQ;
* tempo MMVQ;
* end-to-end decode.

Non assumere che il costo extra sia trascurabile.

Non assumere neppure che raddoppi.

Misurarlo.

---

# PHASE 7 — RAPPRESENTAZIONI SUCCESSIVE SOLO DATA-DRIVEN

Se LSQ non basta, studiare la distribuzione degli errori.

NON selezionare la prossima architettura per intuizione.

---

## CASO A — ERRORE DOMINATO DALLA GRANULARITÀ

Provare:

**G16**

Due scale per 32 valori.

---

## CASO B — POCHI ELEMENTI DOMINANO IL RESIDUO

Provare:

**A1 / A2 residual anchors**

Conservare uno o due valori/residui ad alta precisione per blocco e applicare una correzione fused usando il peso già decodificato.

Preferire conservazione dell'informazione rispetto a clipping arbitrario.

Non usare soglie come `6σ` senza evidenza statistica specifica.

---

## CASO C — RESIDUO DIFFUSO

Provare:

**Q8_1-R8**

Due stadi:

`x ≈ d0*q0 + d1*q1`

Riutilizzare quando possibile:

* stesso decode Q4/Q5;
* due accumulatori `dp4a`;
* stessa scheduling structure.

---

# REGOLA PER OGNI NUOVO FORMATO

Prima:

1. simulatore offline;
2. vero tensore critico;
3. `δdiff`;
4. `Δmargin`.

Solo se promettente:

5. CUDA prototype;
6. primitive probe;
7. full semantic gate;
8. benchmark.

---

# PHASE 8 — W4A16 / MARLIN-LIKE

Se nessuna rappresentazione Q8 permette contemporaneamente:

* 8/8;
* 512/512;
* velocità competitiva;

valutare W4A16.

Obiettivo:

* pesi Q4_K/Q5_K preservati losslessly;
* eventuale repack lossless;
* FP16 activations;
* accumulo sufficientemente preciso;
* scheduling Marlin-like / Tensor Core se utile.

Non affrontare questa architettura prima di aver esaurito gli esperimenti Q8 causali.

---

# PROFILING OBBLIGATORIO

Per ogni candidato CUDA che supera la simulazione numerica:

* tok/s per almeno tre run;
* mediana;
* dispersione;
* quantizer time/token;
* MMVQ time/token;
* register count;
* shared memory;
* local memory;
* spill load/store;
* occupancy;
* DRAM throughput;
* achieved bandwidth;
* kernel duration;
* launch count.

Hardware:

RTX 3090 nominalmente ~936 GB/s.

Lower bound grossolano peso/token:

~14.76 GiB.

Ordine di grandezza:

* 34.5 tok/s → ~509 GiB/s weight-only;
* 40 tok/s → ~590 GiB/s weight-only.

Il percorso Q8_1 attuale è quindi già un asset prestazionale importante.

Non sacrificarlo senza dati.

---

# OTTIMIZZAZIONE DEL QUANTIZZATORE — SOLO DOPO LA PARITY

Il quantizzatore DS4 Q8_1 contiene lavoro apparentemente inutile per il decode MMVQ:

* riduzione `sumf` calcolata e scartata;
* riduzione `qsum`;
* scrittura `ds.y`, non consumata nel MMVQ locale.

NON eliminare questo lavoro durante la parity.

Prima stabilire:

* comportamento del riferimento;
* call path;
* layout;
* eventuali consumer alternativi.

Dopo la correttezza, creare eventualmente una versione decode-specialized.

Valutare anche:

* vectorized load/store;
* `float4`;
* `char4`;
* riduzione del launch overhead.

Queste sono ottimizzazioni finali, non strumenti diagnostici.

---

# DIVIETI METODOLOGICI

NON:

* reimplementare kernel Q8_1/MMVQ già esistenti perché si presume che manchino;
* dichiarare Q6_K colpevole prima della parity;
* dichiarare Q8_1 unica causa dal solo Q6-only;
* inseguire `ds.y` come causa del MMVQ decode;
* imporre `--fmad=false` come ground truth senza verificare il build di llama.cpp;
* assumere che MMQ significhi necessariamente Tensor Core;
* assumere che batch=1 significhi necessariamente MMVQ;
* confrontare un kernel DS4 con un kernel llama teorico invece che con quello realmente lanciato;
* usare il `master` corrente al posto del commit `1a064ab09`;
* trattare una differenza nell'ordine FP32 come automaticamente un bug;
* trattare `qsum` come una riduzione numericamente non associativa;
* usare Q8_32 vs Q8_1 end-to-end come prova pura della precisione della scala;
* introdurre whitelist permanenti;
* introdurre fallback dopo aver già contaminato hidden/KV;
* ottimizzare prima della causalità;
* giudicare un candidato soltanto dalla MAE;
* accettare 7/8 o 511/512.

---

# DECISION TREE RIASSUNTIVO

## STEP 1

Verifica actual llama dispatch.

↓

## STEP 2

A/B/C sul vero step 3.

↓

### Se `B≈C!=A`

Il port Q8 è probabilmente corretto.

Studiare maggiore precisione.

### Se `C≈A!=B`

Correggere DS4.

### Se ambiguo

Primitive parity.

↓

## STEP 3

Frozen Q8 primitive parity.

↓

### Se fallisce

Correggere kernel/mapping/reduction.

### Se passa

Non toccare MMVQ.

↓

## STEP 4

Quantizer byte parity.

↓

### Se produce qs differenti

Localizzare rounding/arithmetic/compiler semantics.

### Se produce qs equivalenti

Passare alle ablazioni numeriche.

↓

## STEP 5

`d_float` vs `d_half`.

↓

## STEP 6

Precision staircase + Q4/Q5/Q6 sensitivity.

↓

## STEP 7

LSQ offline.

↓

## STEP 8

LSQ CUDA se GO.

↓

## STEP 9

G16 → A1/A2 → R8.

↓

## STEP 10

W4A16 solo se necessario.

---

# OUTPUT RICHIESTO

Per ogni fase produrre:

1. cosa è stato verificato;
2. file e funzioni coinvolte;
3. comando esatto;
4. configurazione;
5. risultato numerico;
6. interpretazione;
7. ipotesi eliminate;
8. ipotesi ancora aperte;
9. GO / NO-GO;
10. patch o rollback.

Mantenere una tabella cumulativa:

| Test | Kernel reale | Quantizer | Primitive parity | Quantizer parity | δcommon | δdiff | Δmargin | 8/8 | 512/512 | tok/s | Verdict |
| ---- | ------------ | --------- | ---------------: | ---------------: | ------: | ----: | ------: | --: | ------: | ----: | ------- |

---

# AGGIORNAMENTO DELLA DOCUMENTAZIONE

Aggiornare il markdown di avanzamento dopo ogni risultato significativo con:

* ipotesi;
* esperimento;
* comando riproducibile;
* risultato;
* profiler data;
* semantic gate;
* decisione;
* patch;
* rollback;
* nuova conoscenza acquisita.

Separare chiaramente:

## FATTO DIMOSTRATO

supportato direttamente da codice/test.

## INFERENZA

fortemente suggerita ma non provata.

## IPOTESI

ancora da testare.

Non riscrivere retroattivamente le ipotesi come fatti.

---

# PRIMO COMPITO DELLA SESSIONE

Prima di modificare il codice:

1. leggere integralmente il CUDA DS4 fornito;
2. leggere il markdown di avanzamento;
3. leggere la cronologia dei test;
4. aprire il checkout llama.cpp `1a064ab09`;
5. ricostruire i compile flags del riferimento;
6. eseguire la **Phase 0D — Actual Dispatch Verification**;
7. identificare il kernel e il quantizzatore realmente usati durante il decode del riferimento;
8. eseguire subito dopo l'A/B/C sul vero greedy step 3;
9. procedere quindi alla frozen-input primitive parity.

NON proporre una nuova architettura prima che questi test abbiano prodotto un verdetto.

---

# PRINCIPIO GUIDA

**Una variabile alla volta.**

**Prima identificare ciò che il riferimento esegue realmente.**

**Poi separare quantizzatore, primitive, reduction e rappresentazione numerica.**

**Solo dopo ottimizzare.**

Il risultato desiderato non è una spiegazione elegante.

Il risultato desiderato è sapere, con evidenza riproducibile:

1. perché DS4 Q8_1 ribalta il token;
2. se llama.cpp `1a064ab09` fa lo stesso;
3. quale componente numerico produce il `δdiff` decisivo;
4. quale modifica minima preserva 8/8 e 512/512;
5. quanto vicino possiamo arrivare ai ~40 tok/s sulla RTX 3090 senza violare la semantica.

---

# NOTA DI AVANZAMENTO — 2026-08-12

Questa nota registra l'esecuzione effettiva del piano sopra. I dettagli estesi,
i comandi e gli artefatti restano nel registro canonico
`docs/qwen36-performance-ledger.md`; le ipotesi H15–H21 sono aggiornate in
`docs/qwen36-drift-hypotheses.md`.

## FIN DOVE SI È ARRIVATI

Completati STEP 1–9, inclusi Phase 0D, triangolazione A/B/C, primitive parity,
quantizer parity, scala F32, staircase, sensibilità Q4/Q5/Q6, LSQ, G16,
anchor A1/A2 e Q8_1-R8. STEP 10 W4A16 non è stato avviato: R8 soddisfa già i
gate qualitativi rigidi ed è competitivo con F32, quindi W4A16 non è ancora
necessario secondo la regola della Phase 8.

In questa fase il risultato era ancora **opt-in**; la promozione a default è
registrata nell'aggiornamento finale 2026-08-12 più sotto:

```sh
DS4_CUDA_QWEN_DECODE_Q8_1_R8=1 ./ds4-bench ...
```

Il formato usa `x ≈ d0*q0 + d1*q1`, due accumulatori DP4A e un solo decode di
pesi/scale/minimi condiviso dai due residui. Q4_K, Q5_K e Q6_K sono coperti.

## FATTI DIMOSTRATI

1. Il checkout storico esatto `1a064ab0921238c1daa397d6f4a900ef33884de2`
   usa nel decode batch 1 `quantize_q8_1` + MMVQ `mul_mat_vec_q` per Q4_K,
   Q5_K e Q6_K, con block `(32,4,1)`. Non usa MMQ, cuBLAS o Tensor Core in
   queste proiezioni.
2. Sul vero greedy step 3 il triangolo è `C≈A!=B`: DS4 F32 e llama.cpp storico
   scelgono token 310; DS4 Q8_1 sceglie 728. I margini 310−728 sono
   rispettivamente `+0,012695`, `+0,160900` e `−0,098331`.
3. Con input Q8_1 packed e pesi identici, i kernel reali DS4 e llama.cpp sono
   bit-exact per Q4_K, Q5_K e Q6_K su 257 righe × 5.120 valori. Non c'è un bug
   di unpack, mapping lane, signedness, DP4A o riduzione nel MMVQ DS4.
4. Sul vero `output_norm`, DS4 e llama.cpp producono 160/160 scale, 5.120/5.120
   `qs` e 160/160 `qsum` identici. Differisce soltanto `ds.y`, non letto dal
   MMVQ osservato.
5. Il cast FP16 della scala spiega circa lo 0,55% del cambiamento di margine.
   Mantenerla F32 non recupera il token.
6. Q4-only e Q6-only a Q8_1 sono ciascuno sufficienti a ribaltare il margine
   critico; Q5-only lo riduce ma conserva token 310 sul frozen step.
7. Granularità e activation MAE non predicono monotonamente `δdiff`: Q8_32
   ricostruisce meglio di Q8_1 ma peggiora il margine; Q8_128 conserva il token
   soltanto per compensazione con logit MAE elevata.
8. LSQ one-step riduce activation RMSE dello 0,94%, ma peggiora logit MAE
   `0,009979→0,010206` e margine `−0,009998→−0,016533`; rollback eseguito.
9. Il residuo è diffuso: G16 e A1/A2 non recuperano il margine. R8 porta
   activation MAE a `3,24e-5` e predice correttamente un margine positivo.
10. Il kernel R8 reale passa il probe CPU indipendente per Q4/Q5/Q6 entro
    roundoff. Massimi errori osservati: `2,29e-5`, `4,58e-5`, `2,29e-5`.
11. La suite completa finale `ds4-phase7-q8-1-r8-fused-full-001` passa
    **8/8 sequenze e 512/512 argmax**: 256/256 greedy e 256/256 teacher-forced.
12. `ptxas` su `sm_86` riporta zero stack e zero spill. Q4/Q5 R8 usano 40
    registri e 768 B shared; Q6 40/768 B; il quantizzatore 24 registri.
13. La fusione del weight decode riduce il Q6 MMVQ isolato
    `3,295232→2,253824 ms` (−31,60%) e Q4 `0,058368→0,054272 ms`, senza
    aumentare registri o spill.

## RISULTATO PRESTAZIONALE MANTENUTO

Direction sweep residente, warm-up esplicito, baseline F32 e candidato finale:

| Contesto | F32 | R8 fuso | Miglioramento | Dispersione finale |
|---:|---:|---:|---:|---:|
| 128 | 17,41 tok/s | 32,69 tok/s | +87,77% | CV 1,33%, 5 run |
| 2.048 | 10,71 tok/s | 14,93 tok/s | +39,40% | CV 1,61%, 5 run |

Il percorso resta sotto il riferimento storico di circa 40 tok/s. Il proxy
weight-only a 128 è circa 483 GiB/s. DRAM throughput e achieved bandwidth
hardware sono `NOT_VERIFIED`: Nsight Compute/CUPTI non sono disponibili nel
WSL osservato.

## CANDIDATI RESPINTI DOPO R8

- Eliminazione `qsum`/`ds.y` nel quantizzatore R8: correttezza verde ma
  `32,26→31,18 tok/s` a 128 e `14,93→14,48` a 2K, −3,18% medio. Rollback.
- Q4/Q6 R8 + Q5 Q8_1 singolo: probe Q5 verde, ma gate completo **7/8** e
  **485/512**; teacher 256/256, greedy 229/256, prima divergenza al passo 3 di
  `multi_turn_preserve_thinking`. Nessun benchmark; flag diagnostico rimosso.
- G16, A1/A2, LSQ, scala F32 e le rappresentazioni staircase non hanno
  soddisfatto insieme margine e gate; i prototipi temporanei sono stati rimossi.

## HARNESS AGGIUNTO

`tools/perf_harness.py` ora evita gli script ad hoc ripetuti:

```sh
# Parity dei blocchi Q8_1: ds.x/qs/qsum consumati, ds.y auditato a parte
python3 tools/perf_harness.py q8-1-parity LEFT.bin RIGHT.bin

# Confronto di una riga full-vocabulary tra run A/B/C
python3 tools/perf_harness.py qwen-logits-row --run A=... --run B=... \
  --run C=... --case multi_turn_preserve_thinking --row 3

# Gate completo di sequenze e argmax greedy/teacher
python3 tools/perf_harness.py qwen-argmax-gate \
  gguf-tools/quality-testing/staging/oracles/ds4-q4ks-f32-point1-002 \
  gguf-tools/quality-testing/staging/oracles/ds4-phase7-q8-1-r8-fused-full-001
```

Il test del harness è `python3 -m unittest tests.test_perf_harness` e passa
26/26. Il probe CUDA completo è
`make qwen-numerics CUDA_ARCH=sm_86` e termina con `summary: PASS`.

## PATCH MANTENUTE E STATO GIT

- `ds4_cuda.cu`: Q8_1-R8 predefinito, MMVQ Q4/Q5/Q6 con weight decode fuso e
  profiler opt-in `DS4_CUDA_QWEN_DECODE_Q8_1_R8_PROFILE=1`.
- `tests/qwen_numerics_probe.c`: oracle CPU R8 e probe reali Q4/Q5/Q6.
- `tools/perf_harness.py`, `tests/test_perf_harness.py` e
  `performance/README.md`: parity, triangolazione e gate completo riusabili.
- Ledger, registro ipotesi e indice progetto aggiornati.

Tutte queste modifiche, insieme ai due documenti iniziali già presenti, sono
state inserite nello stage Git. Nessun commit è stato creato.

## COSA RIMANE DA FARE

1. Estendere la suite slow R8 già confermata con misure periodiche 2K/8K/16K;
   il workflow `r8-long` rende questo controllo ripetibile senza cambiare la
   baseline di produzione.
2. R8 è ora default per decisione esplicita dell'utente; mantenere il rollback
   `DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8=1` per bisect e regressioni.
3. Colmare il gap `32,69→~40 tok/s`. Il quantizzatore non è il collo: nel Q6
   output head vale circa 0,005 ms contro 2,254 ms di MMVQ. Priorità a layout,
   mapping lane/load e riuso del decode nel MMVQ, mantenendo il gate completo.
4. Installare o rendere disponibile Nsight Compute/CUPTI per misurare DRAM
   throughput, achieved bandwidth, occupancy reale, durata e launch count con
   contatori hardware; fino ad allora questi campi restano non verificati.
5. Valutare W4A16/Marlin-like soltanto se R8 non può raggiungere velocità
   competitiva dopo le ottimizzazioni MMVQ o fallisce la conferma slow. Non
   cambiare il formato Q4_K/Q5_K losslessly senza prima un simulatore/probe.
6. Dopo ogni nuovo candidato: primitive probe → 8/8 e 512/512 → almeno tre
   run; promozione default soltanto dopo slow a cinque run e documentazione.

## AGGIORNAMENTO 2026-08-12 — VISIBILITÀ DEL CLIENT E TRASFERIMENTO vLLM

1. È stata verificata la sorgente vLLM/Marlin al commit pinned
   `87668ab69b3e2c849a607ece36e8a43bde7c7ee5`. I principi applicabili a DS4
   sono repack/preprocessing nel layout di consumo, metadata separati e pipeline
   dei load; il formato GPTQ/AWQ non sostituisce direttamente GGUF Q4_K_S/R8.
2. Due tentativi di precalcolare le somme Q8_1 consumate da Q4/Q5 sono stati
   respinti pur con probe `PASS`: metadata inline (−0,58% medio, stride 36→44 B)
   e sidecar (−1,50% medio). Entrambe le patch CUDA sono state annullate.
3. È stata trovata la causa per cui l'utente non vedeva i guadagni R8: eseguiva
   `./ds4` compilato l'11 agosto, mentre kernel e `ds4-bench` erano del 12 agosto.
   Dopo il relink, lo smoke breve interattivo è passato da 17,97 a **32,09 t/s**,
   coerente con 32,05 t/s del benchmark breve corrente.
4. `tools/perf-qwen-r8.sh` automatizza ora relink congiunto di client/benchmark,
   probe e A/B F32-vs-R8 (`build`, `direction`, `slow`). Il workflow è esposto
   da `perf_harness.py`; la suite `r8-slow` usa cinque run residenti a 128/2K.
5. `perf_harness.py doctor` segnala se `ds4` o `ds4-bench` sono più vecchi dei
   rispettivi sorgenti/oggetti. La documentazione chiarisce inoltre che
   `--ctx 32768` è capacità, non posizione corrente: CLI e harness vanno
   confrontati a parità di fase, token e contesto effettivamente accumulato.
6. Il workflow `r8-slow` ha prodotto R8 stabile a 31,75 t/s (CV 2,79%) a 128 e
   14,19 t/s (CV 3,92%) a 2K, con gate `PASS` e +84,27%/+34,50% su F32. Il
   verdetto globale resta `NEED_MORE_DATA` perché il baseline di questa singola
   sessione è instabile (CV 5,94%/6,16%); non viene promosso artificiosamente a
   `KEEP`.
7. Il crollo sotto 10 t/s a 10.666 token è stato riprodotto forzando il percorso
   full-attention seriale: 239,554 ms/token, con 202,409 ms nel core. Lo split-K
   corrente misura 38,193/9,445 ms; un run sotto 10 appartiene quindi alla
   configurazione/binario errati, non al decadimento normale del build corrente.
8. Il riuso GQA di una riga K/V fra due query head è stato promosso a default
   dello split-K: +8,08% medio a 8K/12K/16K, +4,67% nella conferma 8K da sette
   run/64 token, 16/16 token identici e top-20 overlap 1,0 a tutte le frontiere.
   Il core profilato a 10.666 scende 9,445→5,999 ms.
9. Gruppi GQA 3 e 6 non sono mantenuti come default; il tentativo shared-memory
   per evitare `expf` ridondanti è stato annullato dopo −2,83% medio.
10. `doctor` e `tools/perf-qwen-r8.sh` coprono ora anche `ds4-server`; l'azione
    `workflow --name r8-long --id ID` esegue build, probe e A/B GQA1/GQA2 con
    finestre da 64 token a 8K/12K/16K.
11. Prossimo collo misurato con R8/GQA2 a 10.666: FFN 14,486 ms, GDN attention
    8,095 ms e full-attention totale 11,181 ms. Per superare stabilmente 30 t/s
    a 10–12K va ridotto il matvec residuo senza riaprire il gate numerico.
12. La baseline richiesta per locale e server è ora senza env: R8 fuso,
    split-K32 automatico da 8K e GQA2. `doctor` mostra questi valori e verifica
    che `ds4`, `ds4-bench` e `ds4-server` siano tutti rilinkati allo stesso
    `ds4_cuda.o`.
13. Il run no-env `r8-default-noenv-20260812` è bit-identico al precedente
    artefatto R8 a 128/2K (logits, top-20 e 8/8 token). Le prestazioni direction
    a due run sono solo diagnostiche perché il punto 2K è instabile; le mediane
    canoniche restano quelle slow documentate nel ledger.
14. Split-K64 dopo GQA2 è respinto: il direction +1,53% era instabile; nella
    conferma 7×64 token a 10.666 K32 misura 29,014 t/s medi contro 28,713 di
    K64 (−1,04%). Il limite resta 32 partizioni.
