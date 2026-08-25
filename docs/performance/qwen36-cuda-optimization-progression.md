# Progressione di ottimizzazione CUDA — Qwen3.6 27B Q4_K_S su RTX 3090

> Checklist storica. I verdetti consolidati e gli esperimenti successivi sono
> registrati in `docs/performance/qwen36-performance-ledger.md`.

Data di creazione: 6 agosto 2026.

Questo documento è la checklist di riferimento per completare le ottimizzazioni
non ancora sviluppate descritte in
[`qwen36-lmstudio-decoding-analysis-2026-08-06.md`](qwen36-lmstudio-decoding-analysis-2026-08-06.md).
L'ordine è vincolante: ogni punto ha un gate di correttezza e prestazioni e non
si sommano due cambiamenti non ancora isolati. L'ultimo punto è sempre il
confronto con la CPU.

## Stato sintetico

| # | Stato | Punto | Dipende da |
|---:|:---:|---|---|
| 0 | [x] | Congelare artefatto, baseline e budget VRAM | — |
| 1 | no check | Portare Q8_1 e MMVQ per Q4_K/Q5_K | 0 |
| 2 | no check | Portare il percorso Q6_K × Q8_1 dell'output head | 1 |
| 3 | no check | Riutilizzare Q8_1 e fondere gate/up/SwiGLU | 1 |
| 4 | [ ] | Allineare e profilare il Gated Delta Net decode fuso | 0 |
| 5 | [ ] | Preallocare tutto lo scratch del decode | 1–4 |
| 6 | [ ] | Reintrodurre CUDA Graph sul percorso stabile | 5 |
| 7 | [ ] | Autotuning controllato per saturare la bandwidth | 1–6 |
| 8 | [ ] | Matrice context/VRAM e validazione finale GPU | 7 |
| 9 | [ ] | Valutazione e confronto finale con la CPU | 8 |

Le caselle vanno marcate solo quando sono presenti nel registro del punto:
commit/build, comando, SHA-256 del modello, report numerico, profilo, picco VRAM
e decisione `PROMOTE`, `REWORK` o `REJECT`.

## Invarianti comuni

- Modello target: **Qwen3.6 27B Q4_K_S**, file locale da 851 tensori con mix
  F32=449, Q4_K=341, Q5_K=60 e Q6_K=1. Il suo SHA-256 verificato è
  `ff857ba9f2184d8be408e8cabda12c89ba5adb202fddc1a88b3774d7bb232aca`.
- GPU target: **RTX 3090 GA102, sm_86, 24.576 MiB**, driver locale 610.62.
  Compilare nativamente per `sm_86`; non valutare un PTX generico come build
  finale.
- MTP/speculative decoding resta disattivato: il target è il forward
  autoregressivo ordinario. La KV cache non viene quantizzata per inseguire il
  risultato LM Studio, perché quel confronto usa K/V non quantizzate.
- Il modello deve rimanere interamente residente sulla GPU. Nessun fallback
  silenzioso a CPU e nessun trasferimento PCIe per layer.
- Correttezza prima della velocità: un candidato non diventa default se perde
  una sequenza greedy o un argmax della suite corta, anche se resta entro una
  soglia media larga.
- Un errore CUDA, un non-finito o un peggioramento inatteso impone di tornare
  all'ultima build verde, acquisire la prima divergenza/profilo e studiare la
  causa. Non si provano combinazioni di flag o geometrie casuali.
- Un solo processo modello pesante alla volta. Prima di ogni run reale,
  verificare la VRAM libera; per la matrice 16K conservare il guard esistente di
  **22.528 MiB liberi prima dell'avvio**.

## Cosa significa “occupare tutta la bandwidth”

La RTX 3090 dichiara 936 GB/s di banda GDDR6X (circa 871,7 GiB/s). Non è
corretto richiedere il 100% in ogni istante o sul tempo end-to-end: attention,
riduzioni, launch e dipendenze possono essere limitate da altre risorse. Il
criterio operativo è che i matvec che attraversano i pesi siano nel regime
memory-bound e usino la massima banda sostenuta riproducibile, senza spill,
letture non coalescenti o SM lasciati senza lavoro.

Per ogni profilo registrare almeno:

```text
tok/s steady
ms/token
GiB/s utili minimi = 14,76 GiB × tok/s
dram__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum
l1tex/lts sectors per request e hit rate
sm__warps_active.avg.pct_of_peak_sustained_active
launch__occupancy_limit_registers
registri/thread, local-memory spill, blocchi/SM
launch count e CPU launch time per token
picco VRAM MiB
```

I 37,66 token/s osservati in LM Studio equivalgono al limite inferiore di circa
556 GiB/s di soli pesi. È il primo target comparabile; il profilo Nsight deve
poi spiegare il traffico reale, che è maggiore della sola dimensione del GGUF.
Le metriche si interpretano insieme: occupancy più alta non è un successo se
peggiora coalescing, registri o banda DRAM.

Fonti: [NVIDIA GA102 whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf),
[Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html),
[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html),
[Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html).

## Protocollo di rollback e diagnosi

Ogni punto segue la stessa sequenza:

1. salvare baseline, commit/worktree, ambiente e comando;
2. cambiare una sola famiglia di primitive;
3. eseguire il probe sintetico e confrontare con un oracolo della stessa
   politica aritmetica;
4. eseguire teacher-forced e trovare la prima divergenza prima del greedy;
5. solo a correttezza verde misurare velocità, banda e VRAM;
6. se fallisce, ripristinare il selettore affidabile, conservare il candidato
   soltanto in una patch/branch diagnostica e classificare la causa fra layout,
   quantizzazione, riduzione, race, spill, occupancy, allocazione o capture;
7. leggere la primitiva upstream e la documentazione relativa alla causa prima
   della modifica successiva.

Una frontiera di layer o un flag per proiezione può localizzare l'errore, ma
non può diventare una lista permanente di eccezioni del modello.

## 0. Congelare artefatto, baseline e budget VRAM

**Gap.** Le misure storiche non sono tutte artifact-identical: LM Studio ha
usato anche un GGUF con blocco MTP integrato, mentre il file target DS4 non lo
esegue. Senza identità di file, build e contesto, un guadagno non è attribuibile
al kernel.

**Metodo scelto.** Usare il Q4_K_S target con SHA-256 sopra indicato per DS4,
llama.cpp CPU e llama.cpp CUDA. Conservare il commit LM Studio `1a064ab09` come
riferimento di replica e una build upstream corrente come controllo separato,
mai nella stessa colonna. Ollama non costituisce un terzo kernel indipendente:
il percorso GGUF moderno integra llama.cpp, quindi va registrata la revisione
llama.cpp effettivamente inclusa.

**Azioni.**

- [x] Verificare hash, dimensione, metadata, 64 layer target e assenza/inerzia
  MTP tramite l'ispettore già presente.
- [x] Fissare `context=5206`, prompt iniziale 128, 128 token greedy,
  `batch=2048`, `ubatch=512`, Flash Attention attiva e KV F16/F32 corrente.
- [x] Registrare baseline float warp8, candidato Q8_K diagnostico, llama.cpp al
  commit LM Studio e llama.cpp upstream corrente.
- [x] Salvare `nvidia-smi`, clock/power state, temperatura, processi GPU e picco
  VRAM. Scartare run con throttling o altri carichi.
- [x] Allocare un ledger VRAM per pesi, KV full-attention, stato GDN, tensori
  persistenti, Q8_1, scratch, graph executable, profiler e almeno 1.024 MiB di
  margine WDDM/driver durante i run brevi.

**Gate.** Tre ripetizioni dopo warm-up, mediana e dispersione; nessun OOM,
fallback o differenza di configurazione. Questa fase non modifica il percorso
di release.

**check —** Eseguito come riportato: Q4_K_S verificato (64 layer, 851 tensori, nessun MTP), mediana DS4 17,20 tok/s e Q8_K diagnostico 24,65 tok/s, llama.cpp `1a064ab09` 40,00 tok/s medi, picchi 18.758/16.565 MiB con oltre 1 GiB di margine.

## 1. Portare Q8_1 e MMVQ per Q4_K e Q5_K

**Gap.** DS4 possiede Q8_K a 256 valori e un Q8_32 ingenuo, ma non il blocco
Q8_1 più il mapping MMVQ completo. Il probe locale del 6 agosto 2026 conferma
che Q8_K implementa correttamente la propria politica, non che la politica sia
adatta: su Q4 la differenza F32→Q8_K è `MAE=0,201628`, `max=0,281075`,
`cosine=0,999992`. Il summary del probe è comunque `PASS`, quindi la causa è
numerica, non un bug di packing.

**Ricerca e decisione.** Il percorso llama.cpp/Ollama quantizza le attivazioni
in blocchi Q8_1 da 32, scrive quattro int8 con un accesso `char4`, conserva
scala e somma e li consuma direttamente nei vec-dot Q4_K/Q5_K. La soluzione va
portata come unità: formato, quantizer, vec-dot e geometria MMVQ. Copiare solo
la granularità ha già prodotto 16,45 token/s in DS4.

La letteratura W4A8 (Atom/QServe) conferma che a batch piccolo il traffico dei
pesi domina e che quantizzazione e layout del kernel vanno co-progettati. Non
si adotta però QServe stock: richiede una quantizzazione W4A8/KV4 differente e
modificherebbe sia l'artefatto Q4_K_S sia la KV, rendendo il confronto non
equivalente. Per questo modello il miglior punto di partenza senza cambio
semantico resta MMVQ di llama.cpp.

Fonti: [quantizer Q8_1 CUDA](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/quantize.cu),
[vec-dot al commit LM Studio](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/ggml/src/ggml-cuda/vecdotq.cuh),
[MMVQ al commit LM Studio](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/ggml/src/ggml-cuda/mmvq.cu),
[QServe](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf),
[Atom](https://arxiv.org/abs/2310.19102).

**Implementazione controllata.**

- [x] Definire un layout Q8_1 locale verificato con `static_assert`, allineato
  agli accessi vectorized del kernel; non riusare il nome di Q8_K.
- [x] Portare il quantizer a 32 valori, inclusi `amax`, scala, somma e regola di
  round/saturazione esatta del commit scelto.
- [x] Aggiungere vec-dot distinti Q4_K×Q8_1 e Q5_K×Q8_1, senza estrattori bit
  generici nel loop caldo.
- [x] Conservare mapping lane/`iqs`, `dp4a`, shuffle e numero di righe per warp
  di MMVQ; rendere parametriche solo le geometrie che il profiler giustifica.
- [x] Quantizzare una volta per singola proiezione in un buffer preallocabile;
  in questa fase non fondere ancora gate e up.

**Micro-test obbligatorio.** Blocchi sintetici con zero, massimi, minimi,
outlier per ogni gruppo da 32 e pattern che esercitano tutti i bit Q5. Richiedere
identità degli int8 e delle somme contro oracolo CPU Q8_1; finish float entro
roundoff. Poi Q4 soltanto, quindi Q4+Q5 sulla suite corta.

**Gate di promozione.** 8/8 sequenze, 512/512 argmax, nessun non-finito;
Q4+Q5 deve migliorare la mediana decode senza aumentare il picco VRAM oltre il
ledger. Il profilo deve mostrare più banda utile e nessuno spill locale. Se il
Q8_1 è corretto ma lento, studiare prima geometria, metadati e registri: non
allargare il gruppo di quantizzazione.

**no check —** Implementazione e probe completati (Q4 max `1,34e-5`, Q5 max `9,16e-5`, 40 registri, zero spill, `34,50 tok/s`, picco `18.697 MiB`), ma 7/8 e 484/512: al layer 10 Q8_1 ha MAE `7,87e-4` vs F32, Kahan `7,75e-4` e no-FMA `7,75e-4`, quindi la causa è la politica di quantizzazione e non lo shuffle; candidato `REJECT`, default F32 invariato.

## 2. Portare Q6_K × Q8_1 per l'output head

**Gap.** L'unico tensore Q6_K è l'output head da 248.320 righe. Il precedente
Q6×Q8_K ricostruiva ripetutamente i sei bit nel loop ed è regredito da 26,45 a
25,82 token/s; è stato correttamente rimosso.

**Ricerca e decisione.** Portare la decomposizione registrata e il vec-dot
Q6_K×Q8_1 di llama.cpp, mantenendo word allineate, maschere e accumuli in
registri. Marlin W4A16 è ottimizzato per Ampere e può essere superiore fino a
batch medi, ma non accetta direttamente il layout affine Q6_K né l'output head
Q6 del GGUF. Requantizzare a INT4 cambierebbe il modello: è escluso. Un
microbenchmark Marlin ha senso solo in un harness sintetico futuro e non può
sostituire il tensore reale senza un gate semantico nuovo.

Fonti: [vec-dot Q6_K di llama.cpp](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/ggml/src/ggml-cuda/vecdotq.cuh),
[Marlin paper](https://arxiv.org/abs/2408.11743),
[Marlin reference kernels](https://github.com/IST-DASLab/marlin).

**Azioni e gate.**

- [x] Aggiungere il probe Q6_K×Q8_1 separato e confrontarlo con lo stesso
  oracolo CPU Q8_1 usato al punto 1.
- [x] Profilare output head isolato: registri, istruzioni integer, banda, tempo
  e occupancy; confrontare con il Q6_K float warp8 bit-exact.
- [x] Eseguire A/B Q4+Q5 contro Q4+Q5+Q6 senza altri cambiamenti.
- [x] Promuovere solo con 512/512 argmax e guadagno end-to-end positivo. Se
  l'estrazione bit o i registri dominano, mantenere Q6 float e chiudere il punto
  come `REJECT` documentato, non compensare cambiando precisione del peso.

**no check —** Port completato e corretto (`max_abs=3,81e-5`, 36 registri, zero spill), mediana `18,06 tok/s` (`+5,0%`), ma 7/8 sequenze e 484/512 argmax: candidato `REJECT` e output head Q6 F32 mantenuto.

## 3. Riutilizzare Q8_1 e fondere gate/up/SwiGLU

**Gap.** Gate e up leggono la stessa attivazione normalizzata, ma oggi ciascun
matvec rialloca/riquantizza e SwiGLU legge due intermedi dalla VRAM.

**Ricerca e decisione.** Riutilizzare un solo Q8_1 per le due proiezioni. La
prima variante deve lanciare due MMVQ sullo stesso buffer, così separa il
risparmio di quantizzazione dalla fusione. La seconda variante può produrre
tile gate/up e applicare SwiGLU prima di scrivere `ffn_mid`, eliminando gli
intermedi globali. La lezione di Marlin è utile per scheduling/pipelining, ma
non giustifica il suo packing GPTQ sul Q4_K_S; si prendono i principi di
coalescing e sovrapposizione, non si cambia formato.

**Azioni.**

- [ ] Esporre un'API interna pair Q4/Q5 che riceve due pesi e un solo Q8_1.
- [ ] Misurare prima `quantize once + 2 launch` contro `quantize twice`.
- [ ] Aggiungere una fusione tiled gate/up/SwiGLU solo se il primo A/B è verde;
  evitare di trattenere un intero vettore FFN nei registri.
- [ ] Scrivere direttamente `ffn_mid`; mantenere gate/up materializzabili solo
  nel percorso di trace diagnostica fuori dal benchmark.

**Gate.** Identità/roundoff del `ffn_mid` contro la politica Q8_1 non fusa,
8/8 e 512/512 end-to-end. Il numero di launch e i byte globali devono scendere;
il tempo FFN totale deve migliorare. Se la fusione aumenta registri fino a
ridurre warp residenti o banda, conservare soltanto il riuso Q8_1.

**no check —** Non eseguito: il prerequisito Q8_1 Q4/Q5 è `REJECT`; fondere gate/up avrebbe sommato un cambiamento già fallito e violato l'ordine vincolante e il protocollo di rollback.

## 4. Allineare e profilare il Gated Delta Net decode fuso

**Gap.** DS4 ha già kernel GDN dedicati e il probe numerico è verde, ma avere
un kernel “fuso” non dimostra che traffico dello stato, geometria e pressione
registri eguaglino llama.cpp. Nel grafo DS4 il decode lancia ancora separatamente
convolution e GDN.

**Evidenza locale.** Il micro-test del 6 agosto 2026 sulla RTX 3090 ha prodotto
stato convolution bit-exact, stato ricorrente e heads con solo roundoff
(`cosine=1`, massimo `2,24e-8`); il percorso multi-riga a quattro token è
anch'esso verde.

**Ricerca e decisione.** Confrontare il kernel CUDA dell'operazione
`GATED_DELTA_NET` upstream aggiunta in llama.cpp e la sua selezione per Qwen,
con particolare attenzione allo stato per thread. L'evidenza upstream su altre
GPU mostra che la stessa fusione può fallire per pressione registri: il port va
profilato su sm_86, non copiato ciecamente.

Fonti: [llama.cpp Gated Delta Net fusion](https://github.com/ggml-org/llama.cpp/pull/19504),
[discussione su registri e backend](https://github.com/ggml-org/llama.cpp/issues/20354),
[Nsight Compute occupancy](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy).

**Azioni e gate.**

- [x] Misurare separatamente convolution, GDN e relative letture/scritture di
  stato per token nei 48 layer ricorrenti.
- [x] Confrontare mapping 48 head × 128, registri/thread, spill e blocchi/SM con
  llama.cpp al commit scelto e upstream corrente.
- [x] Provare la fusione convolution+GDN solo se elimina traffico/launch senza
  duplicare il grande stato ricorrente nei registri o nella local memory.
- [x] Richiedere i gate GDN esistenti exact/roundoff e il gate end-to-end. Se il
  kernel fuso è più lento, mantenere i due kernel e ottimizzare solo la
  geometria dimostrata dal profilo.

**no check —** La variante llama.cpp warp-per-column passa probe e gate semantico (8/8, 512/512), ma stato+norma sale da circa `0,013` a `0,049 ms/layer`: le 1.536 CTA e la norma ripetuta superano il risparmio teorico di 144 MiB/token, quindi resta solo diagnostica e il default originale è mantenuto.

## 5. Preallocare tutto lo scratch del decode

**Gap.** I candidati Q8 correnti usano `cuda_tmp_alloc_on` durante il matmul;
questo impedisce una capture stabile e rende opachi picco VRAM e lifetime.

**Metodo scelto.** I buffer Q8_1 massimi per le dimensioni Qwen (almeno 5.120,
6.144, 10.240, 12.288 e 17.408 elementi, più output head) appartengono alla
sessione/grafo e vengono riusati. Si dimensiona per il massimo realmente
contemporaneo, usando alias solo fra lifetime dimostrabilmente disgiunti.

**Azioni e gate.**

- [ ] Disegnare una tabella lifetime per ogni tensore e calcolare byte esatti
  con overflow check.
- [ ] Preallocare uno o più arena Q8_1 all'apertura della sessione; warm-up e
  lookup dei pesi non devono far crescere cache dopo la preparazione.
- [ ] Fallire esplicitamente prima della generazione se budget o indirizzi non
  sono disponibili; vietato il fallback CPU.
- [ ] Verificare con Nsight Systems che non esistano alloc/free nel token
  steady e che il picco resti sotto il ledger con almeno 1.024 MiB di margine.
- [ ] Richiedere equivalenza bit/roundoff al punto 1 e nessuna regressione
  misurabile. La preallocazione è pronta quando gli indirizzi sono stabili per
  tutta la sessione.

**no check —** Audit completato: i Q8_1 richiederebbero un’unica arena aliasabile da `5.760–19.584 byte`, ma sono usati soltanto dai candidati `REJECT` dei punti 1–2; preallocarla nel percorso F32 stabile consumerebbe VRAM senza beneficio, quindi nessuna modifica viene promossa.

## 6. Reintrodurre CUDA Graph sul percorso stabile

**Gap.** Il prototipo precedente riusava il graph ma non accelerava i matvec e
la capture Q8 tentava allocazioni. Dopo i punti 1–5 il costo dei launch può
diventare visibile e tutti gli indirizzi devono essere stabili.

**Ricerca e decisione.** CUDA Graph riduce il costo host definendo e
istanziando una volta il workflow; aggiornare pochi parametri con le API di
node update è meno costoso di ricatturare. Conservare una chiave di topologia
finita (decode, posizione/context bucket se realmente necessario), distruggere
o riusare esplicitamente le istanze e non creare una cache per token: un issue
llama.cpp recente documenta OOM proprio da graph cache senza riuso di nodi.

Fonti: [CUDA Graphs Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
[llama.cpp CUDA graph lifecycle/OOM](https://github.com/ggml-org/llama.cpp/issues/20315),
[build option upstream](https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt).

**Azioni e gate.**

- [ ] Warm-up di ogni kernel, risoluzione pesi e allocazione scratch prima
  della capture.
- [ ] Mantenere token, posizione e input dinamici in buffer device stabili;
  aggiornare valori in place o i soli parametri dei nodi necessari.
- [ ] Contare capture, instantiate, update e replay; dopo il warm-up ci deve
  essere un replay per token senza crescita di memoria.
- [ ] Misurare prima/dopo con gli stessi kernel MMVQ. Promuovere solo se
  migliora mediana e p95 senza cambiare logits o picco VRAM oltre il ledger.
- [ ] In caso di update/capture failure, disabilitare il graph e diagnosticare
  puntatore/topologia/allocazione specifica; non cambiare stream globali o
  rilassare la correttezza per farlo catturare.

**no check —** Non reintrodotto: Q8_1 e relativa preallocazione sono `REJECT`, mentre il precedente graph F32 era realmente riusato ma misurava circa `17,74 tok/s` contro `17,92` e richiedeva `--default-stream per-thread`; resta disabilitato per evitare complessità senza guadagno.

## 7. Autotuning controllato per saturare la bandwidth

**Gap.** La geometria upstream è un ottimo punto iniziale, ma la RTX 3090
sm_86, le shape esatte di Qwen3.6 e il mix Q4/Q5/Q6 possono avere un optimum
diverso. Varianti casuali renderebbero impossibile attribuire causa ed effetto.

**Metodo scelto.** Usare una matrice piccola e predefinita di geometrie lecite,
derivata da MMVQ e dall'occupancy calculator. Compilarle come specializzazioni,
non come flag permanenti; il benchmark seleziona la vincente per famiglia e la
release conserva un solo percorso.

**Azioni.**

- [ ] Profilare per famiglia: attention projections, GDN projections, FFN
  gate/up/down e output head.
- [ ] Variare una sola dimensione fra righe/warp, warp/blocco e tile K; massimo
  2–3 candidati giustificati dai limiti registri/occupancy.
- [ ] Bloccare clock/power policy quando possibile e registrare temperatura;
  usare warm-up e almeno tre run interlacciati A/B/A.
- [ ] Eliminare subito candidati con spill, transazioni non coalescenti,
  occupancy limitata dai registri senza aumento compensativo di banda, o
  regressione numerica.
- [ ] Scegliere in base a tempo end-to-end e banda sostenuta, non a occupancy
  isolata. Il target minimo comparabile è superare 556 GiB/s utili stimati o
  spiegare con contatori perché l'overhead non-pesi impedisce quel valore.

**Valutazione delle alternative di letteratura.** Marlin e QServe vanno
considerati soltanto con un microbenchmark isolato delle shape 5.120×17.408 e
17.408×5.120 su sm_86. Devono includere costo di packing/quantizzazione,
metadati asimmetrici e batch=1; una misura solo GEMM a batch 16 non è valida per
il decode singolo. Si adottano al posto di MMVQ solo se:

1. usano gli stessi valori Q4_K_S/Q5_K/Q6_K senza requantizzazione lossy;
2. passano gli stessi oracle e 512/512 argmax;
3. riducono il tempo end-to-end, non solo il kernel isolato;
4. rispettano 24 GB con il margine VRAM.

Con i formati pubblicati, queste condizioni non sono soddisfatte: Marlin usa
packing/scale GPTQ INT4 e QServe una co-progettazione W4A8KV4. Restano fonti di
tecniche di scheduling; MMVQ llama.cpp resta il candidato implementativo.

## 8. Matrice context/VRAM e validazione finale GPU

**Gap.** Un percorso veloce a context 128 può perdere vantaggio o superare i
24 GB quando cresce attention/KV. L'ottimizzazione deve restare valida al
context LM Studio 5.206 e nelle frontiere DS4 supportate.

**Metodo scelto.** Eseguire in sequenza, mai in parallelo, context 128, 2.048,
5.206, 16K e solo dopo le frontiere superiori previste dal progetto. Separare
decode da prefill e non confondere eval batch con il batch autoregressivo.

**Azioni e gate.**

- [ ] Per 128/2.048/5.206 registrare pp/tg, ms/token, traffico, launch, VRAM e
  quota tempo pesi/attention/GDN/output.
- [ ] Eseguire la matrice chunk 128/512/2.048/4.096 tramite l'harness esistente,
  con il guard di VRAM prima di ogni processo.
- [ ] Verificare canary, confini di chunk, 8/8, 512/512 e assenza di drift
  crescente; solo dopo eseguire 16K.
- [ ] Se la VRAM prevista lascia meno di 1.024 MiB al driver nei run brevi o
  fallisce il guard 22.528 MiB prima dei long run, ridurre scratch/chunk o
  serializzare workspace. Non quantizzare automaticamente la KV.
- [ ] Considerare KV Q8 solo come progetto separato se il context, non il
  decode dei pesi, diventa il limite; richiede nuovi oracle e non entra nel
  confronto LM Studio MTP-off corrente.

**Completato quando.** La build finale supera tutti i gate GPU, mantiene full
residency, non cresce in VRAM durante il replay, raggiunge la massima banda
sostenuta documentata e nessun contesto supportato causa fallback o OOM.

## 9. Valutazione e confronto finale con la CPU

Questo è obbligatoriamente l'ultimo punto. La CPU è un riferimento per layout,
formula e politica aritmetica, non un oracle semantico assoluto: backend diversi
possono avere ordini di riduzione differenti. Quando CPU e CUDA divergono,
serve triangolare con llama.cpp CUDA e, se necessario, con l'oracle upstream
BF16/FP32.

**Protocollo.**

- [ ] Usare lo stesso GGUF Q4_K_S e lo stesso SHA-256, tokenizer, chat template,
  prompt, token teacher-forced, context e sampling greedy.
- [ ] Confrontare primitive Q8_1/Q4/Q5/Q6 e GDN: int8/somme exact, stato
  exact/roundoff, finish float entro soglie motivate.
- [ ] Eseguire DS4 CPU, DS4 CUDA finale, llama.cpp CPU e llama.cpp CUDA sul
  corpus corto; aggiungere BF16/FP32 solo per decidere quale lato è più vicino
  all'upstream quando l'inviluppo CPU↔CUDA è ampio.
- [ ] Registrare sequenze, 512 argmax, overlap/rank top-20, logprob, cosine,
  MAE/max error, prima divergenza teacher-forced e stato GDN canonicalizzato.
- [ ] Misurare anche pp/tg e memoria, ma usare le prestazioni CPU come rapporto
  informativo, non come target per scegliere un kernel GPU.
- [ ] Eseguire il confronto a context 128 e 5.206; i run CPU più lunghi si
  fermano se diventano instabili o rischiosi, in accordo con le regole del
  progetto.

**Decisione finale.**

- `PROMOTE`: primitive verdi, 8/8, 512/512, nessun drift/non-finito, full
  residency sotto 24 GB, banda e throughput migliori della baseline.
- `REWORK`: formula corretta ma prima divergenza, spill, banda insufficiente,
  graph non riusato o memoria senza margine; tornare al relativo punto, non
  aggiungere eccezioni.
- `REJECT`: alternativa incompatibile con Q4_K_S, richiede requantizzazione/KV
  diversa, regredisce end-to-end o non offre un vantaggio riproducibile.

La progressione è conclusa solo quando il report finale spiega sia la qualità
rispetto alla CPU sia la distanza residua dal riferimento LM Studio/llama.cpp
di 37,66 token/s, con contatori di banda e non con sole ipotesi.

## Riferimenti principali

- [Analisi locale LM Studio/DS4](qwen36-lmstudio-decoding-analysis-2026-08-06.md)
- [llama.cpp, commit usato da LM Studio](https://github.com/ggml-org/llama.cpp/commit/1a064ab09)
- [llama.cpp CUDA Q8_1 quantizer](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/quantize.cu)
- [llama.cpp CUDA vec-dot K-quant](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/ggml/src/ggml-cuda/vecdotq.cuh)
- [llama.cpp CUDA MMVQ](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/ggml/src/ggml-cuda/mmvq.cu)
- [Ollama: integrazione GGUF/llama.cpp](https://ollama.com/blog/improved-performance-and-model-support-with-gguf)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [QServe: W4A8KV4](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf)
- [Atom: low-bit serving](https://arxiv.org/abs/2310.19102)
- [Marlin: mixed-precision autoregressive kernels](https://arxiv.org/abs/2408.11743)
