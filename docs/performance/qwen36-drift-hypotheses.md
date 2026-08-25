# Registro delle ipotesi sul drift Qwen3.6

> Questo file conserva la diagnosi del drift. Le decisioni sulle ottimizzazioni
> di inferenza sono consolidate in `docs/performance/qwen36-performance-ledger.md`.

Questo documento e' il registro operativo per portare Qwen3.6 27B Q4_K_M a
generare almeno 32 token greedy coerenti e ripetibili. llama.cpp sullo stesso
GGUF misura l'equivalenza numerica, ma non e' da solo un oracle semantico:
CPU e CUDA possono produrre continuazioni diverse e perfino non coerenti.
Ogni ipotesi deve essere verificata isolando una sola variabile. Un risultato
non eseguito e' `NON VERIFICATO`, mai `PASS`.

> Correzione 2026-08-05: il converter GGUF riordina le teste value da grouped
> a tiled. Gli esperimenti che avevano patchato llama.cpp verso
> `repeat_interleave` confrontavano il layout sbagliato; sono conservati come
> storia numerica ma non validano la semantica del GGUF.

## Contratto di successo

Il gate minimo usa lo stesso GGUF, gli stessi token di prompt e la stessa
sequenza teacher-forced per entrambi i motori. Richiede:

- 32 passi greedy coerenti e uguali all'oracle semantico upstream, salvo EOS
  atteso e motivato dalla fixture;
- 32 posizioni teacher-forced confrontabili;
- token ID e byte decodificati uguali;
- nessun NaN o Inf;
- determinismo su tre esecuzioni DS4 identiche;
- overlap top-20 almeno 0,95, rank agreement top-20 almeno 0,98 e MAE delle
  logprob dei token oracle non oltre 0,05, come definito in `todo.md`;
- confronto separato contro llama.cpp CPU e CUDA sullo stesso GGUF, senza
  promuovere automaticamente uno dei due a verita' semantica;
- registrazione di commit, hash GGUF, driver, CUDA, build flags, modalita'
  matematica, comando e report.

L'uguaglianza bit-exact e' richiesta tra percorsi DS4 matematicamente
equivalenti. Fra DS4 e llama.cpp, che possono usare ordini di riduzione e
kernel diversi, si usano le metriche e i gate sopra; un argmax uguale da solo
non basta.

## Metodo

1. Congelare input, artefatto e ambiente.
2. Misurare la baseline senza cambiare codice.
3. Cambiare una sola variabile.
4. Confrontare il primo tensore divergente, non soltanto i logits finali.
5. Annotare evidenza, comando, artefatto e conclusione qui.
6. Modificare un kernel soltanto quando la divergenza e' localizzata al suo
   ingresso/uscita e non e' spiegata dall'inviluppo numerico dell'oracle.
7. Dopo una correzione, rieseguire il corpus corto completo e i test
   DeepSeek/GLM applicabili.

## Stato sintetico

| ID | Ipotesi | Stato | Evidenza o prossimo esperimento |
| --- | --- | --- | --- |
| H01 | L'oracle llama.cpp originale implementava il broadcast GDN errato | RESPINTA DOPO AUDIT DEL CONVERTER | Il modulo e' corretto sul layout GGUF tiled. La patch `repeat_interleave` ignorava il riordino value-side fatto in conversione; DS4 `/3` era il vero errore di layout ed e' stato corretto in `%16`. |
| H02 | Il drift deriva da contaminazione fra casi o riuso della sessione DS4 | RESPINTA | `short_fact_english` isolato diverge ancora a greedy 4; prefisso 18 fresh e incrementale DS4 produce logits bit-exact. |
| H03 | Esiste un singolo layer con una discontinuita' macroscopica | RESPINTA NELLA FORMA FORTE | A posizione 17 la MAE del residuo cresce da 6,09e-5 al layer 0 a 1,217 al layer 63; a posizione 16 arriva gia' a 0,953. E' accumulo, non un salto unico. |
| H04 | Output norm o output head amplificano in modo anomalo il drift gia' presente | RESPINTA | Iniettando il residuo llama layer 63 nella testata DS4: MAE logits 0,00877, coseno 0,999990, top-20 1,00 e stesso top-1. La testata preserva quasi interamente l'input. |
| H05 | Parte del drift e' il normale inviluppo numerico llama.cpp CPU-vs-CUDA | CONFERMATA | Al layer 63, posizione 17, llama CPU-vs-CUDA ha MAE 1,167; DS4-vs-CUDA 1,217 e DS4-vs-CPU 0,431. L'inviluppo e' abbastanza grande da cambiare la generazione. |
| H06 | TF32 o `--use_fast_math` sono la causa primaria | RESPINTA NELLA FORMA FORTE | Quality/no-TF32 e release sono bit-exact nella trace Qwen. Senza fast-math il residuo finale migliora 1,198->1,171, ma i logits peggiorano 0,452->0,469 contro CPU e il top-1 resta errato. |
| H07 | Stream/workspace cuBLAS o atomiche rendono il risultato non deterministico | RESPINTA SUL CASO MINIMO | Tre run release identici del prompt completo hanno 963/963 tensori F32 con SHA-256 uguale. Riaprire per context/batch differenti. |
| H08 | Il matvec quantizzato e il suo ordine di riduzione contribuiscono al drift | CONFERMATA COME FONTE, NON COME FIX UNICA | Un dot Q4_K x Q8_K diagnostico con raggruppamento CPU migliora i logits CPU da MAE 0,452 a 0,420 e top-20 0,85->0,90, ma lascia EOS top-1. Non va promosso. |
| H09 | Lo stato GDN/conv ricorrente accumula un errore semantico | RESPINTA SUL KERNEL DECODE ISOLATO | Dopo canonicalizzazione lo stato reale DS4 e' molto piu' vicino a llama CUDA di quanto CPU e CUDA lo siano fra loro. L'oracle sintetico indipendente passa quattro update: stato max `2,33e-9`, heads max `2,24e-8`, conv state bit-exact. |
| H10 | RoPE/KV/full-attention causa il salto al token 18 | APERTA, PRIORITA' 3 | Separare layer ricorrenti e full-attention; confrontare posizioni 16/17 intorno ai layer full-attention, incluse Q/K normalizzate e cache. |
| H11 | Quantizzazione, layout o offset dei pesi differiscono fra i motori | BASSA MA APERTA | Stesso GGUF e embedding uguale riducono la probabilita'. Verificare checksum/offset del primo peso associato allo stadio realmente divergente. |
| H12 | Instabilita' hardware, temperatura, overclock o errori memoria causano il drift | BASSA MA APERTA | Il drift osservato e' strutturato e riproducibile. Registrare clock/power/temperature durante tre run; eliminare overclock per il gate finale. La RTX 3090 GeForce non espone ECC in `nvidia-smi`. |
| H13 | Tokenizer/chat template o token speciali causano la divergenza | RESPINTA PER IL CONFRONTO NUMERICO, APERTA PER LA QUALITA' SEMANTICA | I token sono congelati e identici. Tuttavia CPU produce solo sette token testuali poi cicla sui token chat, mentre CUDA/DS4 scelgono EOS: serve un oracle upstream per stabilire il comportamento atteso. |
| H14 | Ollama CUDA costituisce un oracle indipendente da llama.cpp | RESPINTA DAL SORGENTE | Ollama fissa e costruisce llama.cpp su CUDA. MLX usa `/3` sui pesi upstream; llama CUDA usa `%16` sui pesi GGUF riordinati. Eseguire Ollama su NVIDIA non aggiungerebbe indipendenza. |
| H15 | Il flip Q8_1 DS4 deriva da mapping/unpack/riduzione differenti dal MMVQ storico | RESPINTA | Con identici pesi e Q8_1 packed, Q4/Q5/Q6 DS4 e llama.cpp `1a064ab09` producono output bit-exact su 257×5.120. |
| H16 | Rounding o quantizer Q8_1 differente causa il flip del vero step 3 | RESPINTA SUL CASO REALE | Sul vero `output_norm` F32: 160 scale, 5.120 `qs` e 160 `qsum` coincidono. Differisce solo `ds.y`, non consumato dal MMVQ auditato. |
| H17 | Il cast FP16 della scala Q8_1 domina il flip dell'output head | RESPINTA | A hidden congelato, mantenere la scala F32 sposta il margine di appena +0,000124 e lascia argmax 728; il delta totale F32→Q8_1 è −0,022694. |
| H18 | Una sola famiglia di peso causa tutta la sensibilità Q8_1 | RESPINTA | Q4-only e Q6-only sono ciascuno sufficienti a invertire il margine critico; Q5-only lo riduce ma conserva token 310. Le primitive restano bit-exact. |
| H19 | Granularità più fine o activation MAE minore implicano Δmargin migliore | RESPINTA | Q8_32 ha activation MAE minore di Q8_1 ma Δmargin più negativo; Q8_128 conserva il top-1 con logit MAE 0,472 e grande compensazione differenziale. |
| H20 | Un passo LSQ Q8_1-32 corregge il margine critico | RESPINTA | Activation RMSE migliora 0,94%, ma sul frozen output head il logit MAE sale 0,009979→0,010206 e il margine peggiora −0,009998→−0,016533. |
| H21 | Il residuo diffuso richiede due stadi Q8_1-R8 per conservare il gate | CONFERMATA | Sul tensore critico R8 riduce activation MAE a 3,24e-5; il kernel reale passa i probe Q4/Q5/Q6, 8/8 sequenze e 512/512 argmax. Con weight decode fuso il direction benchmark è 17,41→32,69 tok/s a 128 e 10,71→14,93 a 2K. |

## Evidenze consolidate

### 2026-08-04 - corpus corto corretto

- Otto casi, context 2048, 32 greedy e 32 teacher-forced per caso.
- Nessun NaN/Inf.
- Argmax agreement complessivo 68,55%; overlap top-20 medio 0,7208; MAE
  teacher-forced delle logprob oracle 0,21145.
- Tre casi mantengono tutti i 32 token greedy; gli altri divergono alle
  posizioni 4, 0, 0, 0 e 1.
- Artefatto: `gguf-tools/quality-testing/staging/oracles/corrected-short-equivalence.json`.

Conclusione: il broadcast corretto ha rimosso un errore dell'oracle, ma il gate
di 32 token non e' ancora soddisfatto.

### 2026-08-04 - bisezione per prefisso

- `short_fact_english` coincide in argmax fino a 17 token.
- Aggiungendo `<|im_start|>` al prefisso 18 la MAE centrata sale da 0,6450 a
  2,7811; dopo `assistant`, prefisso 19, cambia l'argmax.
- Fresh e incremental DS4 al prefisso 18 sono bit-exact.

Conclusione: non e' contaminazione di sessione; il token 18 amplifica uno stato
gia' divergente.

### 2026-08-04 - bisezione per layer

- Posizione 17: embedding e attention norm del layer 0 coincidono.
- MAE `layer_out`: 6,09e-5 al layer 0, 0,0126 al layer 19, 0,167 al layer 50,
  1,217 al layer 63; nessun NaN/Inf.
- Posizione 16: MAE `layer_out` del layer 63 pari a 0,953.
- Artefatti: `gguf-tools/quality-testing/staging/qwen-layer-trace-018/`.

Conclusione: cercare prima l'inviluppo CPU/CUDA e l'amplificazione di output;
non esiste ancora evidenza sufficiente per correggere un kernel specifico.

### 2026-08-05 - testata, matrice numerica e inviluppo CPU/CUDA

- Alla posizione 17 il residuo finale DS4-vs-llama CUDA ha MAE 1,217. Dopo
  output norm i logits hanno MAE centrata 0,532, coseno 0,959 e overlap top-20
  0,60.
- Alimentando la testata DS4 con il residuo finale llama, i logits scendono a
  MAE 0,00877, coseno 0,999990 e overlap top-20 1,00. Release e quality mode
  producono lo stesso risultato bit per bit.
- La trace Qwen release e quality/no-TF32 e' bit-exact sui 29 tensori comuni.
- La build senza `--use_fast_math` riduce la MAE del residuo layer 63 contro
  CPU da 1,198 a 1,171, ma peggiora la MAE logits da 0,452 a 0,469 e non cambia
  il top-1.
- llama CPU-vs-CUDA e' gia' molto distante: al layer 63 MAE 1,167; DS4 e'
  piu' vicino a CPU (0,431) che a CUDA (1,217) alla posizione 17.

Conclusione: H04 e la forma forte di H06 sono respinte; H05 e' confermata.
Il drift nasce e si amplifica nel corpo del modello.

### 2026-08-05 - gate reale 32+32 contro llama CPU

- Stesso prompt di 24 token, 32 righe teacher-forced e 32 passi greedy.
- Teacher-forced DS4-vs-CPU: argmax 30/32, overlap top-20 0,9234, MAE logits
  0,22287, MAE centrata 0,17247 e MAE logprob target 0,18266.
- Greedy: longest common prefix 0; nessuno dei 32 argmax coincide dopo che i
  due motori hanno imboccato storie diverse.
- CPU inizia con `The sky appears blue because of the`, poi entra in un ciclo
  di token speciali; DS4 inizia direttamente con `<|im_end|>` e ripete la
  struttura chat. Anche llama CUDA sceglie EOS al primo passo.
- Durata osservata: circa 1405 s llama CPU e 52 s DS4, esclusa ogni pretesa di
  benchmark controllato.
- Artefatti ignorati: `gguf-tools/quality-testing/staging/qwen-cpu-32/`.

Conclusione: il gate di 32 token coerenti e' `FAIL`. CPU e CUDA sullo stesso
GGUF sono utili come inviluppo numerico, ma nessuno dei due e' un oracle
semantico sufficiente per questo caso.

### 2026-08-05 - esperimenti matvec quantizzato

- Layer 0: embedding esatto e attention norm circa 2,4e-8; i primi errori
  misurabili sono QKV 2,19e-4, attention output 1,83e-5 e FFN output 1,64e-4.
- Un percorso diagnostico `DS4_CUDA_QWEN_Q8_ACT_DIAG=1` quantizza le
  attivazioni Q4_K in Q8_K e usa il raggruppamento intero dell'oracle CPU.
  Al layer 0 la MAE FFN scende da 1,64e-4 a 9,79e-5; al layer 63 il residuo
  scende da 1,198 a 1,121 e i logits da 0,452 a 0,420, ma il top-1 resta EOS e
  il margine aumenta da 0,516 a 0,857.
- Arrotondare separatamente le scale delle attivazioni Q8_0 a FP16 con
  `DS4_CUDA_QWEN_Q8_0_CPU_SCALE_DIAG=1`, poi combinarlo col test Q4, peggiora
  QKV da 2,19e-4 a 9,25e-4 e i logits a MAE 0,461. Questa variante e'
  respinta.
- I flag sono solo diagnostici; il percorso predefinito non cambia.
- Artefatti ignorati: `qwen-full-prompt-trace/ds4-q8-cpu-dot-diag/` e
  `qwen-full-prompt-trace/ds4-cpu-arith-diag/` sotto la staging di quality
  testing.

Conclusione: H08 contribuisce quantitativamente, ma inseguire globalmente
l'aritmetica CPU non risolve la decisione semantica e non e' una correzione.

### 2026-08-05 - determinismo release

Tre processi separati, stesso build `sm_86`, stesso GGUF, prompt e posizione
23, hanno prodotto 963 file F32 ciascuno. Tutti i 963 SHA-256 coincidono fra i
tre run. H07 e' respinta per questo caso corto; andra' ripetuta quando cambiano
batch, context, stream o algoritmi.
Artefatti ignorati: `qwen-full-prompt-trace/ds4-default-regression{,-2}/`.

### 2026-08-05 - laboratorio numerico e audit Ollama

- Il nuovo probe sintetico confronta quattro passi GDN DS4 CUDA con un oracle
  CPU indipendente a input identici. Conv state bit-exact; errori massimi:
  `5,59e-9` conv SiLU, `2,33e-9` stato ricorrente, `2,24e-8` heads.
- Il matvec Q4_K CUDA coincide con la dequantizzazione F32 entro `1,53e-5`
  massimo. Riquantizzare soltanto l'attivazione a Q8_K sposta invece l'output
  sintetico di MAE `0,2016`: differenza di policy, non difetto Q4_K.
- Lo stato llama e' stato canonicalizzato trasponendo le dimensioni key/value.
  Al layer 0 DS4-vs-llama CUDA ha RMSE `1,58e-7`, mentre llama CPU-vs-CUDA ha
  RMSE `1,61e-5`.
- Su 723 stadi reali: 1 match esatto, 721 entro 2x la scala RMSE CPU-CUDA e una
  sola eccezione, `alpha` al layer 22 (`0,0820` contro envelope `0,0392`).
- Sorgente Ollama `43983edf...`: su NVIDIA il runner usa llama.cpp fissato a
  `b10242`. Il successivo audit del converter ha mostrato che il mapping modulo
  e' abbinato al riordino fisico value-side del GGUF; MLX usa la divisione sui
  pesi upstream non riordinati.

Conclusione aggiornata: la formula GDN era corretta, ma il mapping delle teste
DS4 era applicato al layout upstream anziche' al layout GGUF. Correggere `%16`
e rigenerare trace/oracle prima di riaprire le proiezioni quantizzate.

### 2026-08-05 - correzione layout tiled confermata end-to-end

- Il prompt italiano senza system contiene 26 token, identici al tokenizer
  Hugging Face fissato.
- Prima della correzione DS4 generava nuovamente i primi token del prompt:
  `Rispondi con una sola parola: qual`.
- Sostituendo il mapping GDN fisico `value_head / 3` con
  `value_head % 16`, senza altre modifiche, la risposta greedy diventa `Blu`.
- Il probe indipendente aggiornato continua a essere `PASS`: conv state
  bit-exact, stato/heads entro roundoff e matvec Q4_K entro roundoff.

Conclusione: bug DS4 confermato e corretto. La prossima baseline va rigenerata
con llama.cpp non patchato e col medesimo layout GGUF tiled.

### 2026-08-05 - suite corta tiled rigenerata

- `ds4-q4-cuda-tiled-short-001` completa gli otto casi a context 2048, con 32
  passi greedy e 32 teacher-forced per caso; verifica struttura/checksum `PASS`.
- I 256 argmax greedy e tutte le sequenze di token coincidono con llama.cpp
  non patchato `7d442abf`; nessun NaN o Inf.
- Overlap top-20 medio `0,97207`, rank agreement medio `0,96994`, coseno medio
  `0,9999900`, MAE teacher-forced delle logprob oracle `0,005927`.
- Il nuovo oracolo `llama-q4-cuda-special-001` usa detokenizzazione
  `special=True`: tutti gli otto casi hanno token e byte uguali a DS4.
- Il profilo cross-engine v2 è calibrato e il confronto numerico ha zero
  failure. I gate top-20 corti sono aggregati; token/byte/argmax/non-finite
  restano rigidi per posizione.
- Tre run DS4 indipendenti (`001`, `002`, `003`) hanno rendering e token
  identici. I confronti full-vocabulary `001-vs-002` e `001-vs-003` riportano
  entrambi zero float differenti su 512 righe, quindi sono bit-exact. Lo stato
  formale `NOT_VERIFIED` deriva solo da `generated_unreviewed` negli index.
- I run numerici nuovi sono revisionati. Il report resta `NOT_VERIFIED` solo
  perché il renderer DS4 completo multi-turn/tool non è ancora verificato in
  modo indipendente (`native_rendering_status=tokenizer_only`).
- Con il system prompt CLI predefinito presente (`You are a helpful assistant`)
  il test italiano a 36 token produce `Blu`; `--system ''` resta disponibile
  per disabilitarlo.

Conclusione aggiornata: gate semantico/numerico corto, decoding speciale e
ripetibilità sono soddisfatti. Restano il renderer chat indipendente e il gate
prestazionale che blocca correttamente la matrice long-context.

## Sequenza degli esperimenti

### E01 - ultimo residuo, output norm e output head

Stato: `COMPLETATO, H04 RESPINTA`.

Input: i prefissi 17 e 18 gia' congelati. Dump richiesti a posizione 16 e 17:
`layer_out` layer 63, `output_norm`, `logits/result_output`. Metriche: MAE, RMSE,
errore massimo, coseno, MAE centrata logits, top-20 overlap/rank e margine
top-1/top-2. Il risultato decide se approfondire H04 oppure risalire ai layer.

### E02 - inviluppo llama.cpp CPU/CUDA

Stato: `COMPLETATO, H05 CONFERMATA`.

Questa misura storica usava llama.cpp con la patch `repeat_interleave`; dopo
l'audit del converter è noto che quella patch era incompatibile col layout
GGUF tiled. Per le nuove baseline usare llama.cpp non patchato, stesso GGUF e
stessi token.

### E03 - matrice numerica DS4

Stato: `COMPLETATO; TF32 BIT-EXACT, FAST-MATH NON RISOLUTIVO`.

Confrontare, una variabile per volta:

1. build corrente in quality mode;
2. `DS4_CUDA_NO_TF32=1` esplicito;
3. build `sm_86` senza `--use_fast_math`;
4. soltanto se necessario, build con FMA controllato e divisione/sqrt precisa.

Conservare anche prestazioni, ma non usarle per promuovere un risultato
numericamente errato.

### E04 - prima primitiva divergente

Stato: `IN CORSO; GDN DECODE RESPINTA, POLICY MATVEC QUANTIZZATO ISOLATA`.

Sul layer 0, confrontare in ordine input norm, proiezioni, GDN/attention,
residuo attention, FFN e residuo finale. Una volta trovato il primo output che
esce dall'inviluppo llama CPU/CUDA, costruire un oracle isolato CPU per quella
primitiva. Solo allora valutare una modifica a `ds4_cuda.cu`.

### E05 - gate 32 token

Stato: `GATE CORTO CALIBRATO, ZERO FAILURE; RENDERER NATIVO NON VERIFICATO`.

Dopo ogni correzione candidata: caso minimo, `short_fact_english`, corpus corto
completo, tre run deterministiche, quindi regressioni CUDA locali e famiglie
DeepSeek/GLM applicabili. Il task e' completato soltanto quando il contratto di
successo iniziale e' interamente verde.

### E06 - oracle semantico e prossima localizzazione

Stato: `DA ESEGUIRE`.

1. Ottenere logits/continuazione da Transformers o altro runtime upstream sui
   pesi originali e sulla stessa sequenza di token, registrando ogni differenza
   di dtype e pesi rispetto al GGUF.
2. Canonicalizzare il layout di `recurrent_state/new_state` prima di calcolare
   metriche sullo stato GDN.
3. Solo se l'oracle upstream sceglie la continuazione testuale con margine
   robusto, isolare la prima primitiva che esce sia dall'inviluppo CPU/CUDA sia
   dalla direzione dell'oracle upstream.
4. Non promuovere `DS4_CUDA_QWEN_Q8_ACT_DIAG`: il test migliora alcune MAE ma
   fallisce il gate decisionale.

## Diario delle decisioni

- Non si modifica un kernel in base alla sola crescita della MAE fra layer.
- TF32 e fast-math sono ipotesi diverse: quality mode governa cuBLAS, mentre
  `--use_fast_math` e' una proprieta' di compilazione dei kernel custom.
- La RTX 3090 e' il sistema sotto test, non l'oracle semantico. Il confronto
  CPU/CUDA di llama.cpp serve a stimare l'effetto della diversa aritmetica,
  non a giustificare divergenze greedy.
- Le prove 4K/16K restano bloccate finche' il gate corto e il gate prestazionale
  definiti in `todo.md` non sono verdi.
