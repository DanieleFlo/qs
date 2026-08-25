# Indice: performance

Indice generato; non modificare a mano.

## [Contratto operativo per kernel engineering](agent_performance_contract.md)

Una modifica è utile soltanto se migliora il workload dichiarato di DS4 senza

- righe 6–18: **Obiettivo** — Una modifica è utile soltanto se migliora il workload dichiarato di DS4 senza
- righe 19–38: **Ciclo obbligatorio** — 1.
- righe 39–54: **Significatività minima** — Il default è cinque ripetizioni dopo un warm-up per workload.
- righe 55–65: **Correttezza** — Il harness non sostituisce gli scorer Qwen già presenti in
- righe 66–73: **Misurazione hardware** — I tempi finali vengono dal percorso sincronizzato di ds4-bench.

## [Progressione di ottimizzazione CUDA — Qwen3.6 27B Q4_K_S su RTX 3090](qwen36-cuda-optimization-progression.md)

Le caselle vanno marcate solo quando sono presenti nel registro del punto:

- righe 15–33: **Stato sintetico** — Le caselle vanno marcate solo quando sono presenti nel registro del punto:
- righe 34–56: **Invarianti comuni** — F32=449, Q4_K=341, Q5_K=60 e Q6_K=1.
- righe 57–92: **Cosa significa “occupare tutta la bandwidth”** — La RTX 3090 dichiara 936 GB/s di banda GDDR6X (circa 871,7 GiB/s).
- righe 93–111: **Protocollo di rollback e diagnosi** — Ogni punto segue la stessa sequenza:
- righe 112–145: **0. Congelare artefatto, baseline e budget VRAM** — **Gap.** Le misure storiche non sono tutte artifact-identical: LM Studio ha
- righe 146–199: **1. Portare Q8_1 e MMVQ per Q4_K e Q5_K** — **Gap.** DS4 possiede Q8_K a 256 valori e un Q8_32 ingenuo, ma non il blocco
- righe 200–230: **2. Portare Q6_K × Q8_1 per l'output head** — **Gap.** L'unico tensore Q6_K è l'output head da 248.320 righe.
- righe 231–259: **3. Riutilizzare Q8_1 e fondere gate/up/SwiGLU** — **Gap.** Gate e up leggono la stessa attivazione normalizzata, ma oggi ciascun
- righe 260–295: **4. Allineare e profilare il Gated Delta Net decode fuso** — **Gap.** DS4 ha già kernel GDN dedicati e il probe numerico è verde, ma avere
- righe 296–321: **5. Preallocare tutto lo scratch del decode** — **Gap.** I candidati Q8 correnti usano `cuda_tmp_alloc_on` durante il matmul;
- righe 322–354: **6. Reintrodurre CUDA Graph sul percorso stabile** — **Gap.** Il prototipo precedente riusava il graph ma non accelerava i matvec e
- righe 355–395: **7. Autotuning controllato per saturare la bandwidth** — **Gap.** La geometria upstream è un ottimo punto iniziale, ma la RTX 3090
- righe 396–424: **8. Matrice context/VRAM e validazione finale GPU** — **Gap.** Un percorso veloce a context 128 può perdere vantaggio o superare i
- righe 425–463: **9. Valutazione e confronto finale con la CPU** — Questo è obbligatoriamente l'ultimo punto.
- righe 464–478: **Riferimenti principali** — Sezione strutturale; consultare il contenuto locale indicato.

## [Registro delle ipotesi sul drift Qwen3.6](qwen36-drift-hypotheses.md)

Il gate minimo usa lo stesso GGUF, gli stessi token di prompt e la stessa

- righe 18–40: **Contratto di successo** — Il gate minimo usa lo stesso GGUF, gli stessi token di prompt e la stessa
- righe 41–52: **Metodo** — 1.
- righe 53–78: **Stato sintetico** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 79–243: **Evidenze consolidate** — teacher-forced delle logprob oracle 0,21145.
  - righe 81–93: **2026-08-04 - corpus corto corretto** — teacher-forced delle logprob oracle 0,21145.
  - righe 94–103: **2026-08-04 - bisezione per prefisso** — 2,7811; dopo `assistant`, prefisso 19, cambia l'argmax.
  - righe 104–114: **2026-08-04 - bisezione per layer** — 1,217 al layer 63; nessun NaN/Inf.
  - righe 115–132: **2026-08-05 - testata, matrice numerica e inviluppo CPU/CUDA** — output norm i logits hanno MAE centrata 0,532, coseno 0,959 e overlap top-20
  - righe 133–150: **2026-08-05 - gate reale 32+32 contro llama CPU** — 0,22287, MAE centrata 0,17247 e MAE logprob target 0,18266.
  - righe 151–171: **2026-08-05 - esperimenti matvec quantizzato** — misurabili sono QKV 2,19e-4, attention output 1,83e-5 e FFN output 1,64e-4.
  - righe 172–179: **2026-08-05 - determinismo release** — Tre processi separati, stesso build `sm_86`, stesso GGUF, prompt e posizione
  - righe 180–201: **2026-08-05 - laboratorio numerico e audit Ollama** — CPU indipendente a input identici.
  - righe 202–215: **2026-08-05 - correzione layout tiled confermata end-to-end** — Hugging Face fissato.
  - righe 216–243: **2026-08-05 - suite corta tiled rigenerata** — passi greedy e 32 teacher-forced per caso; verifica struttura/checksum `PASS`.
- righe 244–310: **Sequenza degli esperimenti** — Stato: `COMPLETATO, H04 RESPINTA`.
  - righe 246–254: **E01 - ultimo residuo, output norm e output head** — Stato: `COMPLETATO, H04 RESPINTA`.
  - righe 255–263: **E02 - inviluppo llama.cpp CPU/CUDA** — Stato: `COMPLETATO, H05 CONFERMATA`.
  - righe 264–277: **E03 - matrice numerica DS4** — Stato: `COMPLETATO; TF32 BIT-EXACT, FAST-MATH NON RISOLUTIVO`.
  - righe 278–286: **E04 - prima primitiva divergente** — Stato: `IN CORSO; GDN DECODE RESPINTA, POLICY MATVEC QUANTIZZATO ISOLATA`.
  - righe 287–295: **E05 - gate 32 token** — Stato: `GATE CORTO CALIBRATO, ZERO FAILURE; RENDERER NATIVO NON VERIFICATO`.
  - righe 296–310: **E06 - oracle semantico e prossima localizzazione** — Stato: `DA ESEGUIRE`.
- righe 311–320: **Diario delle decisioni** — `--use_fast_math` e' una proprieta' di compilazione dei kernel custom.

## [Perché LM Studio raggiunge 37,66 token/s con Qwen3.6 27B e come replicarlo in DS4](qwen36-lmstudio-decoding-analysis-2026-08-06.md)

LM Studio ha prodotto 329 token a **37,66 token/s**, cioè circa **26,56 ms per

- righe 25–50: **1. Risultato osservato e configurazione confrontata** — LM Studio ha prodotto 329 token a **37,66 token/s**, cioè circa **26,56 ms per
- righe 51–96: **2. Il modello mentale corretto per la RTX 3090** — Per ogni nuovo token, un modello denso deve attraversare quasi tutti i pesi di
  - righe 53–73: **2.1 Perché il decoding è soprattutto limitato dalla memoria** — Per ogni nuovo token, un modello denso deve attraversare quasi tutti i pesi di
  - righe 74–96: **2.2 Warp, coalescing e occupazione** — Una RTX 3090 esegue i thread in warp da 32.
- righe 97–232: **3. Cosa fa llama.cpp nel percorso veloce** — Il sorgente della build usata da LM Studio è stato ispezionato al commit esatto
  - righe 102–158: **3.1 Pesi compressi e attivazioni Q8_1 temporanee** — I pesi rimangono Q4_K, Q5_K o Q6_K in VRAM.
  - righe 159–172: **3.2 MMVQ specializzato** — llama.cpp non usa un loop scalare generico sopra questi blocchi.
  - righe 173–183: **3.3 Full offload e memoria residente** — Con tutti i layer sulla RTX 3090, il token non attraversa PCIe fra CPU e GPU a
  - righe 184–199: **3.4 Flash Attention** — Flash Attention fonde la formazione dei punteggi, softmax e prodotto con V in
  - righe 200–216: **3.5 Gated Delta Net fusa** — Qwen3.6 alterna full attention e layer ricorrenti Gated Delta Net.
  - righe 217–232: **3.6 Riuso del grafo CUDA** — CUDA Graph registra una sequenza di kernel e la ripresenta con una sola
- righe 233–320: **4. Cosa è già implementato in DS4** — Il loader ora riconosce e valida il layout misto del Q4_K_S, inclusi i tensori
  - righe 235–250: **4.1 Supporto completo del GGUF Q4_K_S** — Il loader ora riconosce e valida il layout misto del Q4_K_S, inclusi i tensori
  - righe 251–258: **4.2 Modello interamente residente e layer-major prefill** — DS4 copia il modello sulla GPU, conserva la KV sulla GPU e usa prefill
  - righe 259–266: **4.3 Kernel float warp8 affidabili** — Q4_K, Q5_K e Q6_K hanno percorsi warp8 che riducono drasticamente il numero di
  - righe 267–307: **4.4 Q4_K/Q5_K × Q8_K diagnostico** — È stato esteso il percorso sperimentale
  - righe 308–320: **4.5 Selettori per isolare le proiezioni** — I flag diagnostici permettono di applicare Q8 soltanto a:
- righe 321–374: **5. Esperimenti scartati o non promossi** — È stato costruito un prototipo che:
  - righe 323–339: **5.1 CUDA Graph DS4** — È stato costruito un prototipo che:
  - righe 340–349: **5.2 Q8 per gruppi da 32 implementato in modo ingenuo** — DS4 possedeva già un esperimento `Q8_32`, concettualmente più vicino a Q8_1:
  - righe 350–361: **5.3 Q6_K × Q8_K artigianale** — È stato provato un kernel Q6 per accelerare l'output head da 248.320 righe.
  - righe 362–369: **5.4 KV cache a 8 bit** — Non è stata implementata né forzata.
  - righe 370–374: **5.5 MTP/speculative decoding** — Non è stato usato.
- righe 375–401: **6. Risultati prestazionali della sessione** — Benchmark comune:
- righe 402–476: **7. Gate numerici ed end-to-end** — La suite `qwen_numerics_probe` è PASS:
  - righe 404–418: **7.1 Probe di primitive: riusciti** — La suite `qwen_numerics_probe` è PASS:
  - righe 419–434: **7.2 Un singolo prompt: riuscito ma insufficiente** — Su `short_fact_english`, Q4/Q5×Q8_K completo contro la baseline float ha dato:
  - righe 435–455: **7.3 Suite corta completa: fallimenti** — La suite contiene otto casi e confronta 32 righe greedy più 32 righe
  - righe 456–476: **7.4 Perché il probe passa ma il modello fallisce** — Il probe risponde alla domanda:
- righe 477–558: **8. Come rifare meglio la conversione float→int8** — Il prossimo candidato dovrebbe adottare esattamente un blocco da 32:
  - righe 479–499: **8.1 Portare Q8_1, non rinominare Q8_K** — Il prossimo candidato dovrebbe adottare esattamente un blocco da 32:
  - righe 500–516: **8.2 Portare insieme il vec-dot e la geometria MMVQ** — La sequenza consigliata è:
  - righe 517–530: **8.3 Quantizzare una volta e riusare per gate/up** — Gate e up leggono la stessa attivazione normalizzata.
  - righe 531–545: **8.4 Preallocare lo scratch prima di CUDA Graph** — Il buffer Q8_1 massimo necessario per ciascuna dimensione deve appartenere al
  - righe 546–558: **8.5 Usare una frontiera di layer solo come strumento diagnostico** — Ridurre Q8 ai primi o agli ultimi layer aiuta a localizzare la sensibilità, ma
- righe 559–578: **9. Relazione con il riferimento CPU** — Il backend CPU rimane un riferimento utile per layout, formula e ordine
- righe 579–604: **10. Piano di lavoro consigliato** — Ordine suggerito, con un gate dopo ogni passo:
- righe 605–614: **Riferimenti principali** — Sezione strutturale; consultare il contenuto locale indicato.

## [Laboratorio numerico Qwen3.6](qwen36-numerics-lab.md)

oracle CPU indipendenti.

- righe 12–31: **Componenti** — oracle CPU indipendenti.
- righe 32–49: **Comandi** — Su RTX 3090 (`sm_86`):
- righe 50–77: **Audit del sorgente Ollama** — Audit eseguito il 2026-08-05 sul commit Ollama
- righe 78–198: **Risultati ottenuti** — Il probe CUDA sintetico passa:
  - righe 108–140: **Suite corta rigenerata dopo la correzione** — Il run `ds4-q4-cuda-tiled-short-001` usa llama.cpp non patchato
  - righe 141–155: **Regressioni e long context** — Le regressioni CUDA granulari condivise sono verdi, inclusi
  - righe 156–163: **System prompt** — Il system prompt non era stato rimosso: la CLI usa per default
  - righe 164–198: **Prefill layer-major e tentativi di decode veloce** — Il collo di bottiglia originale era strutturale: `ds4_session_sync` inoltrava

## [Esperimenti prestazionali Qwen3.6 27B CUDA](qwen36-performance-experiments-2026-08-05.md)

Velocizzare il percorso CUDA di Qwen3.6 27B Q4_K_M senza degradare la

- righe 9–20: **Obiettivo** — Velocizzare il percorso CUDA di Qwen3.6 27B Q4_K_M senza degradare la
- righe 21–34: **Ambiente di prova** — `65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921`;
- righe 35–48: **Baseline iniziale** — Prima delle modifiche, il prefill inoltrava un token alla volta attraverso i
- righe 49–66: **Modifica strutturale: prefill layer-major** — È stato introdotto un grafo Qwen multi-riga che:
- righe 67–112: **Esperimenti sul prefill** — Le misure principali usano 2.048 token, chunk 2.048 e RTX 3090.
  - righe 83–99: **Profilazione del prefill** — Su un chunk da 128 token, prima dell'attention GEMM, la profilazione aggregata
  - righe 100–112: **Tentativi di prefill scartati** — 136,39 token/s.
- righe 113–148: **Esperimenti sul decode** — Le misure corte usano context 128 e 16 token generati, salvo indicazione
  - righe 132–148: **Causa del rifiuto del decode Q8** — Nel caso `short_fact_english`, il candidato Q8 coincide con l'oracolo per i
- righe 149–241: **Test di correttezza eseguiti** — Comando:
  - righe 151–173: **Probe numerici CUDA** — Comando:
  - righe 174–183: **Pattern dei layer Qwen** — Comando:
  - righe 184–198: **Test del comparatore** — Comando:
  - righe 199–241: **Suite corta full-vocabulary finale** — Run DS4 finale:
- righe 242–260: **Configurazione finale consegnata** — Il percorso predefinito conserva soltanto le modifiche considerate affidabili:
- righe 261–410: **Confronto locale con LM Studio** — Su richiesta e' stata analizzata l'installazione locale di LM Studio senza
  - righe 292–312: **Baseline di release dopo il confronto** — Con la stessa build CUDA e lo stesso Q4_K_M DS4:
  - righe 313–340: **Crossover Q4_K per il physical batch da 128** — Il primo tentativo di imitare il physical batch 512 di LM Studio aveva lasciato
  - righe 341–393: **Verifiche reali 4K e 16K dopo il crossover** — Sul caso `long_canary_ctx4096` sono stati eseguiti i chunk 128, 512, 2048 e
  - righe 394–410: **Regressioni finali** — Il `make test` senza selezione e' stato interrotto dopo oltre 16 minuti mentre
- righe 411–420: **Conclusione** — La configurazione di release misurata con la stessa build raggiunge 549,50
- righe 421–468: **Aggiornamento 2026-08-06: Ottimizzazione Register-State Caching per GDN CUDA** — In data 2026-08-06 e' stata implementata l'ottimizzazione del caching dello stato nei registri GPU nei kernel CUDA `qwen35_gdn_rows_kernel` e `qwen35_conv_rows_kernel` in `ds4_cuda.cu`.
  - righe 425–428: **Modifiche Apportate** — Sezione strutturale; consultare il contenuto locale indicato.
  - righe 429–434: **Risultati delle Misure** — Sezione strutturale; consultare il contenuto locale indicato.
  - righe 435–451: **Verifica Q4_K_S e residency del 2026-08-06** — Il GGUF Unsloth Q4_K_S verificato ha SHA-256
  - righe 452–468: **Decode long-context: warp attention e tentativo grouped GQA del 2026-08-10** — Sul Q4_K_S, con 10.666 token di prompt e 200 token greedy a context 32.768,

## [Ledger prestazionale Qwen3.6 CUDA](qwen36-performance-ledger.md)

Ogni riga segue:

- righe 8–27: **Contratto** — Ogni riga segue:
- righe 28–63: **Stato corrente** — `ff857ba9f2184d8be408e8cabda12c89ba5adb202fddc1a88b3774d7bb232aca`.
- righe 64–79: **Soluzioni mantenute** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 80–111: **Soluzioni scartate o non promosse** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 112–374: **Parity Q8_1/MMVQ sul riferimento storico, 2026-08-12** — Il checkout LM Studio esatto `1a064ab0921238c1daa397d6f4a900ef33884de2`
  - righe 114–131: **FATTO DIMOSTRATO — dispatch reale** — Il checkout LM Studio esatto `1a064ab0921238c1daa397d6f4a900ef33884de2`
  - righe 132–147: **FATTO DIMOSTRATO — A/B/C al greedy step 3** — Configurazione congelata: GGUF Q4_K_S SHA-256 `ff857ba9…232aca`, RTX 3090
  - righe 148–176: **FATTO DIMOSTRATO — primitive e quantizer parity** — Un entry point diagnostico temporaneo ha alimentato i kernel DS4 e llama.cpp
  - righe 177–189: **FATTO DIMOSTRATO — frozen output head e scala FP16** — Con lo stesso hidden pre-output-norm, il solo output head F32 riproduce
  - righe 190–198: **INFERENZA E DECISIONE** — La precisione Q8_1 può cambiare una decisione a margine basso anche quando DS4
  - righe 199–231: **FATTO DIMOSTRATO — sensibilità Q4/Q5/Q6** — Due selettori temporanei, poi rimossi, hanno separato Q4-only e Q5-only.
  - righe 232–251: **FATTO DIMOSTRATO — precision staircase** — La ricostruzione offline dello stesso `output_norm` F32 e il risultato
  - righe 252–269: **FATTO DIMOSTRATO — Q8_1 LSQ offline e proiezione** — Sul vero vettore critico, un passo LSQ per blocco da 32 riduce activation MAE
  - righe 270–374: **FATTO DIMOSTRATO — residuo diffuso e Q8_1-R8** — Il simulatore offline ha decodificato direttamente le righe 310 e 728 del vero
- righe 375–399: **Analisi del residuo a 8K, 2026-08-11** — Il conto dei soli byte di peso per operazione è: ricorrente 3,426 GB, full
- righe 400–420: **Esperimenti long-context 2026-08-11** — Fonti recuperate: guida locale `problem-to-source.md`, FlashDecoding++,
- righe 421–472: **Trasferimento vLLM/Marlin e visibilità R8, 2026-08-12** — È stato ispezionato vLLM al commit fissato
- righe 473–552: **Stabilizzazione R8 long-context e riuso GQA, 2026-08-12** — La riproduzione a 10.666 token ha separato due fenomeni che prima venivano
- righe 553–582: **Curva completa 2K–30K e soglia split-K, 2026-08-13** — La nuova suite non rapida `context-curve-full` misura 15 frontiere a passo 2K,
- righe 583–628: **MTP long-context e crossover split-K, 2026-08-13** — La curva server target-only 0–28K a passo 2K, capacità 28.737, due run per
- righe 629–651: **Coda ordinata** — 1.
- righe 652–669: **Template per il prossimo record** — ID / data / commit / dirty state:
