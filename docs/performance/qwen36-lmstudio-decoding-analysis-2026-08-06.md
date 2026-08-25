# Perché LM Studio raggiunge 37,66 token/s con Qwen3.6 27B e come replicarlo in DS4

> Analisi storica e fonte di ipotesi. Per lo stato canonico delle soluzioni
> mantenute e scartate vedere `docs/performance/qwen36-performance-ledger.md`.

Data dell'analisi: 6 agosto 2026.

Questa nota descrive il confronto sul Qwen3.6 27B Q4_K_S e sulla stessa RTX
3090. Le proiezioni target hanno la stessa quantizzazione; prima di definire i
due file *byte-identical* va però confrontato lo SHA-256, perché il log LM
Studio identifica anche un layer MTP integrato (65 layer offloaded), mentre il
percorso DS4 esegue i 64 layer target e non consuma MTP. Con MTP disattivato il
forward target resta il confronto rilevante, ma questa distinzione deve rimanere
esplicita. L'obiettivo non è copiare alla cieca le impostazioni della UI,
ma capire quale lavoro esegue davvero la GPU, quali ottimizzazioni sono già
presenti in DS4, quali sono state provate in questa sessione e perché alcune
varianti veloci non hanno superato i gate di correttezza.

Una precisazione importante: **nessuna prova qui descritta forza la KV cache a
8 bit**. Nelle schermate di LM Studio la quantizzazione K/V è disattivata. Il
Q8 citato nelle sezioni sui matvec è una quantizzazione temporanea delle
attivazioni di una singola moltiplicazione matrice-vettore; non cambia il
formato persistente della KV cache.

## 1. Risultato osservato e configurazione confrontata

LM Studio ha prodotto 329 token a **37,66 token/s**, cioè circa **26,56 ms per
token**, con speculative decoding/MTP disattivato. Dalle schermate e dal log
locale risultano:

- runtime CUDA llama.cpp 2.27.1, build b10099, commit `1a064ab09`;
- modello interamente residente sulla RTX 3090, con 65 layer offloaded;
- context allocato 5.206 token;
- eval batch 2.048 e physical/ubatch 512;
- Flash Attention attiva;
- KV cache residente sulla GPU, ma K e V non quantizzate a 8 bit;
- modello mantenuto in memoria, `mmap` disattivato;
- speculative decoding disattivato;
- riuso del grafo osservato in 327 passi su 329.

Il dato di 37,66 token/s non è quindi spiegato da MTP. È il throughput del
percorso autoregressivo ordinario del runtime CUDA di llama.cpp.

I parametri `batch=2048` e `ubatch=512` sono molto importanti per il prefill,
quando centinaia di righe possono essere elaborate come GEMM. Durante il
decoding di una singola richiesta, invece, entra un solo nuovo token per passo:
il problema dominante torna a essere una sequenza di moltiplicazioni
matrice-vettore, o MMV/MMVQ. Aumentare il batch di caricamento non trasforma da
solo quel lavoro in una grande GEMM.

## 2. Il modello mentale corretto per la RTX 3090

### 2.1 Perché il decoding è soprattutto limitato dalla memoria

Per ogni nuovo token, un modello denso deve attraversare quasi tutti i pesi di
ogni layer. Una moltiplicazione matrice-vettore riusa poco ciascun peso: il peso
viene letto, moltiplicato per una componente dell'attivazione e normalmente non
serve più fino al token successivo. L'intensità aritmetica è quindi bassa e la
banda della GDDR6X conta più del picco teorico in FLOP/s.

Il modello residente occupa circa 14,76 GiB. Usando questa dimensione come
semplice indicatore di banda utile, non come misura hardware esatta:

- 17,92 token/s corrispondono a circa 264 GiB/s di pesi attraversati;
- 26,45 token/s corrispondono a circa 390 GiB/s;
- 37,66 token/s corrispondono a circa 556 GiB/s.

Il calcolo trascura letture ripetute, cache, attivazioni e operazioni non
matriciali, ma rende visibile il problema: LM Studio utilizza molto meglio la
banda disponibile. Il collo di bottiglia principale non è la capacità della
VRAM e non è la dimensione della KV a context 5.206; è quanto efficientemente i
kernel attraversano e consumano i pesi quantizzati.

### 2.2 Warp, coalescing e occupazione

Una RTX 3090 esegue i thread in warp da 32. Un buon kernel di decoding deve:

1. far leggere ai lane indirizzi contigui, così le richieste diventano poche
   transazioni global-memory coalescenti;
2. distribuire il dot product fra i lane senza lasciarne molti inattivi;
3. usare registri sufficienti per evitare memoria temporanea, ma non così tanti
   da ridurre eccessivamente il numero di warp residenti su ogni SM;
4. ridurre i risultati con shuffle di warp, evitando sincronizzazioni e shared
   memory non necessarie;
5. produrre abbastanza blocchi concorrenti da nascondere la latenza della
   memoria.

Questo spiega perché due kernel che implementano la stessa formula possono
avere throughput molto diversi. Anche una lettura vettoriale `uint4` può
peggiorare il risultato se aumenta troppo la pressione sui registri. La guida
CUDA raccomanda di misurare insieme accessi coalescenti, occupazione e latenza,
non di ottimizzare una sola metrica isolata.

Riferimenti: [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html),
[Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).

## 3. Cosa fa llama.cpp nel percorso veloce

Il sorgente della build usata da LM Studio è stato ispezionato al commit esatto
[`1a064ab09`](https://github.com/ggml-org/llama.cpp/commit/1a064ab09).

### 3.1 Pesi compressi e attivazioni Q8_1 temporanee

I pesi rimangono Q4_K, Q5_K o Q6_K in VRAM. llama.cpp non espande l'intera
matrice in float prima di ogni token. Quantizza invece il vettore di attivazione
in piccoli blocchi Q8_1 e calcola direttamente prodotti come:

```text
Q4_K × Q8_1
Q5_K × Q8_1
Q6_K × Q8_1
```

Le implementazioni sono visibili in
[`vecdotq.cuh`](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/ggml/src/ggml-cuda/vecdotq.cuh)
e vengono usate dai kernel
[`mmvq.cu`](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/ggml/src/ggml-cuda/mmvq.cu).

Per un blocco di 32 float `x[i]`, la quantizzazione concettuale è:

```text
amax  = max(abs(x[i]))
d     = amax / 127
q[i]  = clamp(round(x[i] / d), -127, 127)
x[i] ≈ d * q[i]
```

Il prodotto fra quattro coppie di byte signed può poi essere accumulato con
un'istruzione tipo `dp4a`:

```text
acc += qweight[0] * qact[0]
     + qweight[1] * qact[1]
     + qweight[2] * qact[2]
     + qweight[3] * qact[3]
```

Quattro moltiplicazioni intere e quattro somme vengono così espresse con una
sola istruzione GPU, mentre le scale float vengono applicate a gruppi di somme,
non a ogni singolo elemento.

Q4_K e Q5_K sono quantizzazioni affini: oltre alla scala contengono un termine
minimo. Per questo Q8_1 conserva anche la somma dei quantizzati del blocco. Il
dot product può correggere il minimo senza ricostruire 32 float:

```text
dot ≈ d_act * d_weight * scale_group * dot_int
      - d_act * dmin_weight * min_group * sum(q_act)
```

Per Q5_K, il quinto bit è conservato separatamente in `qh`; il kernel lo
ricompone come contributo `16 * high_bit` e lo accumula nello stesso dot
product intero.

Il punto essenziale è la granularità: **Q8_1 usa una scala per 32 valori**, in
accordo con i gruppi consumati dai kernel K-quant. Un valore anomalo influenza
al massimo gli altri 31 valori del gruppo.

### 3.2 MMVQ specializzato

llama.cpp non usa un loop scalare generico sopra questi blocchi. I lane del warp
caricano porzioni precise di pesi e attivazioni, eseguono più `dp4a`, tengono
scale e somme in registri e riducono secondo una geometria scelta per il tipo
quantizzato. Q4_K, Q5_K e Q6_K hanno percorsi distinti perché il layout dei bit
e delle scale è diverso.

Questo risolve due problemi del percorso float:

- evita di scrivere una matrice dequantizzata molto più grande in scratch;
- riduce il numero e il costo delle istruzioni per elemento, consumando i byte
  quasi alla velocità con cui arrivano dalla VRAM.

### 3.3 Full offload e memoria residente

Con tutti i layer sulla RTX 3090, il token non attraversa PCIe fra CPU e GPU a
ogni layer. `Keep Model in Memory` impedisce inoltre che il sistema debba
ricaricare pagine del modello fra richieste. Nel log storico di LM Studio, la
presenza anche di un solo layer CPU disabilitava il GDN fuso e faceva crollare
il throughput; quando l'offload diventava completo, il salto era molto grande.

In DS4 questa parte è già implementata: il modello Q4_K_S viene copiato
integralmente in VRAM e il benchmark riporta circa 14,76 GiB residenti.

### 3.4 Flash Attention

Flash Attention fonde la formazione dei punteggi, softmax e prodotto con V in
tile che restano il più possibile on-chip. Riduce letture e scritture
intermedie e diventa particolarmente utile con context lunghi e durante il
prefill.

DS4 possiede già il percorso attention multi-riga/GEMM per il prefill e un
percorso CUDA dedicato per il decode. A context 128, tuttavia, l'attention non è
il costo principale rispetto all'attraversamento di 14,76 GiB di pesi; per
questo Flash Attention da sola non può spiegare il divario 18→38 token/s.

Il grafo Qwen di llama.cpp costruisce esplicitamente l'operazione Flash
Attention in
[`llama-graph.cpp`](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/src/llama-graph.cpp).

### 3.5 Gated Delta Net fusa

Qwen3.6 alterna full attention e layer ricorrenti Gated Delta Net. llama.cpp
prova automaticamente sia il GDN autoregressivo fuso sia quello chunked. Una
fusione mantiene stato, normalizzazione, gate e aggiornamento il più vicino
possibile nei registri/shared memory, evitando round trip in VRAM e molte
launch separate. La selezione è visibile in
[`llama-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/src/llama-context.cpp).

DS4 ha già kernel GDN Qwen dedicati. In questa serie di modifiche sono stati
inoltre mantenuti in registri la colonna dello stato ricorrente e i quattro
campioni della convoluzione durante il percorso multi-riga. I probe numerici
GDN sono rimasti entro roundoff. Rimane da profilare la geometria del decode e
confrontare il numero di launch e il traffico dello stato con la fusione della
build llama.cpp: avere un kernel "fuso" non garantisce automaticamente la
stessa occupazione o lo stesso traffico.

### 3.6 Riuso del grafo CUDA

CUDA Graph registra una sequenza di kernel e la ripresenta con una sola
operazione host. Risolve il costo di centinaia di launch CPU per token e riduce
jitter e sincronizzazioni. LM Studio ha riusato il grafo in 327 dei 329 passi
osservati.

Il vantaggio è grande solo se il runtime:

- prealloca tutto lo scratch prima della capture;
- mantiene stabili indirizzi, dimensioni e topologia;
- aggiorna soltanto input e parametri dinamici;
- non sincronizza il device fra nodi del grafo.

Riferimento: [CUDA Graphs, CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).

## 4. Cosa è già implementato in DS4

### 4.1 Supporto completo del GGUF Q4_K_S

Il loader ora riconosce e valida il layout misto del Q4_K_S, inclusi i tensori
Q5_K presenti solo in alcuni layer/proiezioni. Sono stati aggiunti:

- layout `block_q5_K` verificato con `static_assert`;
- dequantizzazione Q5_K;
- matvec Q5_K float warp8;
- dispatch Q5_K nei percorsi CUDA Qwen;
- controlli espliciti per distinguere Q4_K_S e Q4_K_M.

Senza questa copertura non sarebbe possibile confrontare correttamente i
tensori target Q4_K_S usati nel test LM Studio. L'identità dell'intero file,
inclusi gli eventuali tensori MTP non eseguiti, richiede comunque il controllo
SHA-256 indicato sopra.

### 4.2 Modello interamente residente e layer-major prefill

DS4 copia il modello sulla GPU, conserva la KV sulla GPU e usa prefill
layer-major. Per batch multi-riga, i pesi Q4_K vengono dequantizzati in scratch
e passati a cuBLAS quando questo è più conveniente del matvec riga per riga.
Questa ottimizzazione ha già portato il prefill del precedente Q4_K_M da circa
205 a oltre 500 token/s.

### 4.3 Kernel float warp8 affidabili

Q4_K, Q5_K e Q6_K hanno percorsi warp8 che riducono drasticamente il numero di
blocchi rispetto al vecchio kernel con un blocco CUDA per riga. Il Q6_K usato
dall'output head è bit-exact contro il block256 nel probe sintetico. Questi
kernel restano il default affidabile, ma ricostruiscono valori float durante il
dot product e quindi non raggiungono l'efficienza MMVQ di llama.cpp.

### 4.4 Q4_K/Q5_K × Q8_K diagnostico

È stato esteso il percorso sperimentale
`DS4_CUDA_QWEN_DECODE_Q4_Q8=1` da Q4_K anche a Q5_K. L'attivazione viene
quantizzata al volo in un `block_q8_K` DS4:

```c
typedef struct {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
} block_q8_K;
```

La procedura è:

1. trovare il valore di massimo modulo nei 256 float;
2. derivare una sola scala `d` per tutto il blocco;
3. arrotondare e saturare i 256 valori a byte signed;
4. calcolare 16 somme, una ogni 16 byte, per la correzione dei minimi Q4/Q5;
5. eseguire i dot product con `dp4a`;
6. ricomporre il quinto bit Q5 da `qh` e aggiungerlo come `16 * high`;
7. applicare scale e minimi in float soltanto alle somme intere.

Quindi i float non vengono convertiti permanentemente. Il tensore originale e
la KV restano float; il buffer Q8 vive solo per quella proiezione del token.

Il probe Q5 confronta il kernel GPU con la stessa politica Q8 calcolata da un
oracolo dequantizzato:

```text
MAE       1,1563e-5
max error 4,5776e-5
cosine    1,0
classe    roundoff
```

Questo dimostra che il kernel Q5 implementa correttamente **la politica Q8_K
da 256 valori**. Non dimostra che quella politica sia sufficientemente fedele
per 64 layer autoregressivi.

### 4.5 Selettori per isolare le proiezioni

I flag diagnostici permettono di applicare Q8 soltanto a:

- gate/up FFN: `DS4_CUDA_QWEN_DECODE_Q4_Q8_GATE_UP=1`;
- down FFN: `DS4_CUDA_QWEN_DECODE_Q4_Q8_DOWN=1`;
- un intervallo di layer con `DS4_CUDA_QWEN_DECODE_ADAPTIVE_Q8_LAYERS` e
  `DS4_CUDA_QWEN_DECODE_ADAPTIVE_Q8_START`.

In questa sessione il filtro di layer è stato esteso anche ai flag gate/up e
down. Resta diagnostico: la prova a 32 layer ha misurato 18,69 token/s ma il
gate completo è stato interrotto prima del risultato.

## 5. Esperimenti scartati o non promossi

### 5.1 CUDA Graph DS4

È stato costruito un prototipo che:

- spostava token e posizione in tensori device;
- catturava l'intero passo autoregressivo;
- riusava lo stesso eseguibile CUDA Graph nei token successivi.

Il grafo veniva realmente riusato, ma il throughput restava circa 17,7 token/s,
praticamente uguale alla baseline. Inoltre il percorso Q8 allocava scratch
durante la capture, operazione non permessa da CUDA.

Conclusione: il problema corrente è soprattutto nel costo dei matvec, non nella
sola launch latency. Il prototipo è stato rimosso per non imporre
`--default-stream per-thread` a tutto il backend senza un guadagno misurabile.
Un nuovo tentativo ha senso **dopo** MMVQ e soltanto con scratch preallocato.

### 5.2 Q8 per gruppi da 32 implementato in modo ingenuo

DS4 possedeva già un esperimento `Q8_32`, concettualmente più vicino a Q8_1:
una scala ogni 32 valori. Ha però raggiunto solo 16,45 token/s, peggio del float.

Non significa che Q8_1 sia lento. Significa che il solo formato corretto non
basta: il kernel DS4 esegue più loop, carica più metadati e non usa la stessa
distribuzione MMVQ fra lane di llama.cpp. È un importante risultato negativo:
**granularità numerica e geometria GPU devono essere progettate insieme**.

### 5.3 Q6_K × Q8_K artigianale

È stato provato un kernel Q6 per accelerare l'output head da 248.320 righe. La
ricostruzione dei sei bit veniva effettuata ripetutamente dentro il loop prima
di ogni `dp4a`; il costo di bit extraction e la pressione sui registri hanno
superato il risparmio aritmetico. Il throughput è sceso da 26,45 a 25,82
token/s. Il kernel è stato rimosso.

La soluzione corretta è portare la decomposizione registrata e la geometria
Q6_K×Q8_1 di `vecdotq.cuh`, non chiamare quattro volte una funzione generica di
estrazione per costruire ogni word.

### 5.4 KV cache a 8 bit

Non è stata implementata né forzata. Ridurrebbe soprattutto l'occupazione VRAM
e il traffico dell'attention a context lunghi. A context 128-5.206, con un
modello da 14,76 GiB attraversato a ogni token, non risolve il collo di
bottiglia dominante. Introduce inoltre un'altra sorgente di quantizzazione
senza essere necessaria per replicare il test LM Studio.

### 5.5 MTP/speculative decoding

Non è stato usato. Il risultato LM Studio di riferimento ha speculative
decoding su `Off`; il confronto è quindi volutamente sul decoding ordinario.

## 6. Risultati prestazionali della sessione

Benchmark comune:

```text
RTX 3090, ds4flash.gguf Q4_K_S, modello residente 14,76 GiB
context iniziale 128, context allocato 5.206, 128 token generati
```

| Variante | Decode steady | Esito |
|---|---:|---|
| Baseline float warp8 | 17,92 tok/s | default affidabile |
| CUDA Graph riusato | ~17,74 tok/s | nessun vantaggio, rimosso |
| Q8 per gruppi da 32, solo Q4 | 16,45 tok/s | più lento |
| Q4×Q8_K, Q5 ancora float | 25,62 tok/s | diagnostico |
| Q4/Q5×Q8_K completo | **26,45 tok/s** | veloce ma fallisce la suite |
| Q4/Q5/Q6×Q8_K artigianale | 25,82 tok/s | regressione, Q6 rimosso |
| Q8_K solo FFN gate/up+down | 21,96 tok/s | fallisce una continuazione |
| Q8_K solo FFN down | 17,15 tok/s | nessun vantaggio utile |
| Q8_K solo FFN gate/up | 20,34 tok/s | quasi verde, ma diverge |
| Gate/up Q8_K, primi 32 layer | 18,69 tok/s | gate completo non terminato |
| LM Studio/llama.cpp, MTP off | **37,66 tok/s** | riferimento osservato |

Il massimo candidato DS4 recupera circa il 50% del divario fra baseline e LM
Studio, ma non è stato promosso perché il guadagno non può prevalere sulla
correttezza.

## 7. Gate numerici ed end-to-end

### 7.1 Probe di primitive: riusciti

La suite `qwen_numerics_probe` è PASS:

- stato convoluzionale GDN: exact;
- stato ricorrente e heads GDN: entro roundoff float32;
- Q5_K warp8 contro block256: exact;
- Q6_K warp8 contro block256: exact;
- Q5_K×Q8_K contro l'oracolo della stessa politica Q8: cosine 1 e solo
  roundoff.

Questi test localizzano bug di layout, bit packing, scale, somme e riduzioni.
Hanno permesso, per esempio, di correggere il termine minimo Q5 affinché usasse
entrambe le somme da 16 valori di ciascun gruppo da 32.

### 7.2 Un singolo prompt: riuscito ma insufficiente

Su `short_fact_english`, Q4/Q5×Q8_K completo contro la baseline float ha dato:

```text
sequenza greedy: identica
argmax greedy:   32/32
argmax teacher:  32/32
top-20 overlap:  0,975
cosine:          almeno 0,999615
MAE logits:      circa 0,075
```

Il risultato sembrava promuovibile. La suite più ampia ha mostrato perché non
si deve validare un'ottimizzazione numerica su una sola continuazione.

### 7.3 Suite corta completa: fallimenti

La suite contiene otto casi e confronta 32 righe greedy più 32 righe
teacher-forced per caso, cioè 512 decisioni.

| Variante | Sequenze greedy identiche | Argmax | Top-20 overlap | MAE logits | Max error |
|---|---:|---:|---:|---:|---:|
| Q8 su tutte le proiezioni Q4/Q5 | 6/8 | 458/512 | 0,8630 | 0,3350 | 34,10 |
| Q8 su gate/up+down FFN | 7/8 | 479/512 | 0,9122 | 0,2108 | 33,66 |
| Q8 solo gate/up FFN | 7/8 | 508/512 | 0,9709 | 0,0733 | 26,77 |

Il gate/up isolato è vicino: sette casi sono completamente stabili e soltanto
`system_thinking_off` cambia continuazione. Tuttavia 508/512 non è 512/512 e
la divergenza autoregressiva rende il percorso inadatto come default.

I grandi errori massimi non descrivono tutte le righe: compaiono dopo la prima
decisione diversa, quando i due run consumano token differenti e quindi non
stanno più valutando lo stesso stato. Per giudicare la causa iniziale bisogna
guardare soprattutto le righe teacher-forced e la prima divergenza, non la MAE
dopo che le sequenze si sono separate.

### 7.4 Perché il probe passa ma il modello fallisce

Il probe risponde alla domanda:

> Il kernel calcola correttamente ciò che abbiamo definito?

La suite risponde alla domanda diversa:

> Ciò che abbiamo definito è una buona approssimazione per l'intero modello?

Con una scala unica su 256 attivazioni, un outlier determina il passo di
quantizzazione per tutto il blocco. Valori piccoli possono collassare sullo
stesso intero. L'errore di una proiezione entra nel residuo, attraversa RMSNorm,
gate non lineari, GDN ricorrente e altri layer. In una riga con margine ridotto
fra i primi due token, un piccolo cambiamento dei logits può invertire l'argmax;
da quel momento il greedy segue un'altra traiettoria.

Non è quindi probabile che il fallimento derivi dal bit Q5 ricomposto male: il
probe Q5 dedicato lo esclude entro il suo dominio. La causa più plausibile è la
granularità Q8_K/256, combinata con l'accumulo attraverso molte proiezioni.

## 8. Come rifare meglio la conversione float→int8

### 8.1 Portare Q8_1, non rinominare Q8_K

Il prossimo candidato dovrebbe adottare esattamente un blocco da 32:

```c
struct q8_1_block {
    half2 ds;     // scala d e somma scalata s
    int8_t qs[32];
};
```

I dettagli del layout devono corrispondere al kernel scelto. Non basta dividere
il vecchio blocco da 256 in otto strutture e lasciare invariato il loop.

Vantaggio numerico: otto scale indipendenti per 256 valori limitano l'effetto
degli outlier.

Costo: più scale e somme da leggere. llama.cpp compensa il costo tenendole in
registri e facendo corrispondere ciascun blocco Q8_1 ai gruppi Q4/Q5/Q6 già
consumati dal warp.

### 8.2 Portare insieme il vec-dot e la geometria MMVQ

La sequenza consigliata è:

1. portare `block_q8_1` e il quantizer da 32 valori;
2. portare separatamente `vec_dot_q4_K_q8_1`, `vec_dot_q5_K_q8_1` e
   `vec_dot_q6_K_q8_1` dal commit verificato;
3. conservare lo stesso mapping di `iqs`, lane e gruppi usato da MMVQ;
4. evitare funzioni generiche di estrazione bit nel loop interno;
5. caricare word da 32 bit allineate e usare maschere/shuffle in registri;
6. misurare registri per thread, occupancy, global-load efficiency e achieved
   bandwidth con Nsight Compute;
7. fare A/B fra più geometrie solo dopo che il test numerico è verde.

Questo è il punto in cui una ricerca/port mirato dal codice llama.cpp è più
utile di una nuova approssimazione inventata localmente.

### 8.3 Quantizzare una volta e riusare per gate/up

Gate e up leggono la stessa attivazione normalizzata. Il percorso corrente
quantizza il vettore separatamente per le due proiezioni. Un'API pair può:

- quantizzare `x` una sola volta;
- lanciare gate e up nello stesso kernel o in due kernel che condividono lo
  stesso buffer Q8_1;
- fondere SwiGLU quando non servono i tensori intermedi gate/up.

Il risparmio di byte delle attivazioni è piccolo rispetto ai pesi, ma elimina
quantizer, launch e scritture intermedie per 64 layer. È anche un punto naturale
per il riuso del grafo CUDA.

### 8.4 Preallocare lo scratch prima di CUDA Graph

Il buffer Q8_1 massimo necessario per ciascuna dimensione deve appartenere al
grafo Qwen ed essere allocato all'apertura della sessione. La capture non deve
chiamare `cudaMalloc`, far crescere cache o risolvere per la prima volta un
peso. Procedura:

1. warm-up di tutte le varianti kernel;
2. risoluzione e pinning dei puntatori peso;
3. allocazione Q8_1 per 5.120 e 17.408 elementi;
4. capture con token e posizione in buffer device stabili;
5. replay e aggiornamento dei soli input;
6. misura separata prima/dopo MMVQ, perché il grafo non corregge un kernel
   memory/compute-bound inefficiente.

### 8.5 Usare una frontiera di layer solo come strumento diagnostico

Ridurre Q8 ai primi o agli ultimi layer aiuta a localizzare la sensibilità, ma
non è la soluzione ideale: crea una politica specifica del modello e recupera
solo una parte del throughput. Può essere usato per costruire una mappa:

```text
layer/proiezione → errore logits → variazione del margine top-1/top-2
```

La soluzione finale dovrebbe essere Q8_1 sufficientemente fedele da passare
l'intera suite, non una lista fragile di eccezioni.

## 9. Relazione con il riferimento CPU

Il backend CPU rimane un riferimento utile per layout, formula e ordine
concettuale, ma non è un oracle semantico assoluto. Nel lavoro precedente,
llama.cpp CPU e CUDA sullo stesso GGUF mostravano già un inviluppo ampio e in un
caso la CPU entrava in un ciclo di token chat mentre CUDA e DS4 terminavano con
EOS.

Per questo i limiti adottati sono stratificati:

1. **primitive matematiche:** exact o solo roundoff contro CPU/dequant float;
2. **DS4 contro DS4:** stessa sequenza e tutti gli argmax sui casi corti prima
   di promuovere una politica aritmetica nuova;
3. **stesso GGUF contro llama.cpp CUDA:** overlap, rank, logprob e continuazione;
4. **oracle upstream BF16/FP32:** necessario per decidere quale backend è
   semanticamente migliore quando CPU e CUDA divergono.

Il Q8_K completo è dentro alcune soglie larghe CPU↔CUDA, ma fallisce il più
forte controllo interno DS4↔DS4. È corretto lasciarlo diagnostico.

## 10. Piano di lavoro consigliato

Ordine suggerito, con un gate dopo ogni passo:

1. aggiungere probe Q8_1 da 32 per Q4_K, Q5_K e Q6_K;
2. portare il vec-dot esatto dal commit llama.cpp b10099;
3. confrontare ogni kernel con un oracolo CPU che quantizza gli stessi blocchi
   Q8_1, richiedendo exact sugli interi e solo roundoff sul finish float;
4. eseguire la suite corta teacher-forced e localizzare la prima divergenza;
5. richiedere 8/8 sequenze e 512/512 argmax contro la baseline affidabile;
6. misurare Q4, poi Q4+Q5, poi Q4+Q5+Q6, senza sommare modifiche non isolate;
7. fondere quantizzazione gate/up e SwiGLU;
8. soltanto dopo preallocare scratch e reintrodurre CUDA Graph;
9. profilare con Nsight Compute e registrare banda, occupancy, register count e
   tempo per famiglia di proiezione;
10. ripetere a context 128, 2.048 e 5.206 per separare costo pesi e costo
    attention/KV.

La replica credibile dei 37,66 token/s non passa quindi da KV Q8 o da MTP. Il
percorso più promettente è replicare fedelmente **Q8_1 + vec-dot K-quant +
geometria MMVQ**, poi aggiungere fusioni e CUDA Graph su buffer completamente
preallocati. Gli esperimenti di questa sessione mostrano sia il potenziale
(17,92→26,45 token/s) sia il motivo per cui il primo tentativo non può ancora
essere il default (perdita di precisione dovuta alla scala condivisa da 256 e
divergenze autoregressive nella suite completa).

## Riferimenti principali

- [llama.cpp commit usato da LM Studio](https://github.com/ggml-org/llama.cpp/commit/1a064ab09)
- [Dot product quantizzati CUDA di llama.cpp](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/ggml/src/ggml-cuda/vecdotq.cuh)
- [Kernel MMVQ CUDA di llama.cpp](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/ggml/src/ggml-cuda/mmvq.cu)
- [Costruzione del grafo Qwen/Flash Attention](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/src/llama-graph.cpp)
- [Selezione graph reuse e GDN fuso](https://github.com/ggml-org/llama.cpp/blob/1a064ab09/src/llama-context.cpp)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
