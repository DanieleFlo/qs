# Indice: forks

Indice generato; non modificare a mano.

## [buun-llama-cpp](buun-llama-cpp.md)

Repository.

- righe 3–6: **Classificazione** — Repository.
- righe 7–10: **Perché è rilevante** — Contiene codec KV a bitrate variabile, quantizzazione trellis, attention CUDA specializzata, GDN e molte istanze template per dtype e head dimension.
- righe 11–14: **Aree da ispezionare** — Le directory `ggml/src/ggml-cuda` includono attention MMA/WMMA/vector, `gated_delta_net.cu`, MMVQ/MMQ e codec KV.
- righe 15–18: **Idee candidabili per DS4** — Dispatch per head dimension, KV cache compressa progressivamente, attention quantizzata e cattura/iniezione per speculative decoding.
- righe 19–22: **Rischi** — Il fork modifica contemporaneamente formato KV, attention, qualità e runtime: attribuire un guadagno è difficile.
- righe 23–25: **Esperimento minimo** — Usare il fork come catalogo di ipotesi.

## [club-3090](club-3090.md)

Repository.

- righe 3–6: **Classificazione** — Repository.
- righe 7–10: **Perché è rilevante** — È specifico per 24 GiB Ampere e Qwen3.6.
- righe 11–14: **Evidenza forte** — Le schede che includono digest immagine, commit, configurazione, tre warm-up, cinque misure, CV e matrice di correctness sono buone piste riproducibili.
- righe 15–18: **Evidenza debole** — La kernel matrix contiene raccomandazioni aggregate come “backend più veloce sulla 3090”: sono orientamento, non prova, finché manca un artefatto associato allo stesso modello e workload.
- righe 19–22: **Problemi Qwen già isolati** — Il repository descrive interazioni MTP × KV quantizzata × CUDA Graph, shape Marlin sotto il minimo dopo tensor parallel, picchi VRAM del chunked prefill e ricostruzione dello stato GDN nelle cache persistenti.
- righe 23–25: **Uso corretto in DS4** — Partire dalla bench card per riprodurre un fenomeno, seguire il link alla PR primaria, quindi verificare il kernel o il runtime originale.

## [club3090 llama.cpp MoE-cache branch](club3090-llama-moe-cache.md)

Branch sorgente.

- righe 3–6: **Classificazione** — Branch sorgente.
- righe 7–10: **Problema risolto** — Su modelli MoE molto più grandi della VRAM, una cache di esperti può ridurre trasferimenti ripetuti.
- righe 11–14: **Perché non è una soluzione diretta** — Qwen3.6 27B DS4 è denso/ibrido e interamente residente in circa 14,76 GiB: non ha lo stesso collo di bottiglia degli esperti offloaded.
- righe 15–18: **Lezione trasferibile** — Sono utili il metodo di sovrapposizione, la gestione esplicita del reserve VRAM e i test che dimostrano quando una cache rimane inattiva.
- righe 19–22: **Evidenza da richiedere** — Diff contro il parent, misura PCIe, cache hit, picco VRAM, prefill/decode separati e correctness.
- righe 23–25: **Decisione DS4** — Catalogare, non portare ora.

## [Genesis vLLM patches](genesis-vllm-patches.md)

Repository.

- righe 3–6: **Classificazione** — Repository.
- righe 7–10: **Problemi affrontati** — Qwen3.6 ibrido su GPU consumer da 24 GiB: TurboQuant KV, stato GDN, MTP speculative, CUDA Graph con shape dinamiche, tool calling quantizzato e limiti di contesto.
- righe 11–14: **Valore tecnico** — Il registro patch associa versioni, anchor e lifecycle, e rimuove workaround quando upstream incorpora la correzione.
- righe 15–18: **Rischi** — Centinaia di patch applicate insieme rendono difficile attribuire un guadagno.
- righe 19–22: **Idee candidabili per DS4** — Test per interazione CUDA Graph × K+1 speculative, shape minime Marlin dopo tensor parallel, ricostruzione dello stato GDN e gate acceptance.
- righe 23–25: **Decisione DS4** — Usare il registro per trovare la patch e la issue primaria, poi isolare un solo meccanismo.

## [ik_llama.cpp](ik-llama-cpp.md)

Repository.

- righe 3–6: **Classificazione** — Repository.
- righe 7–10: **Perché è rilevante** — Dichiara di avere introdotto o sperimentato quant repacking, Gated Delta Net fuso, tensor parallel, MTP e varianti CUDA prima dell'upstream.
- righe 11–14: **Aree da ispezionare** — `ggml/src/ggml-cuda/mmvq.cu`, `mmq.cu`, configurazioni Ampere, `fattn.cu` e `gated_delta_net.cu`.
- righe 15–18: **Idee candidabili per DS4** — Quantized GEMV/MMQ per IQ/K-quant, fast attention token-generation per GQA, GDN fuso e repacking orientato alla GPU.
- righe 19–22: **Rischi** — Il progetto avverte che alcune combinazioni repacked non hanno implementazione CUDA e possono ricadere sulla CPU.
- righe 23–25: **Esperimento minimo** — Compilare il commit fissato per `sm_86`, stesso GGUF e batch 1.

## [KoboldCpp](koboldcpp.md)

Repository.

- righe 3–6: **Classificazione** — Repository.
- righe 7–10: **Perché è rilevante** — Mostra come un fork consumer mantiene compatibilità, partial offload, binari Windows e molte modalità d'uso.
- righe 11–14: **Aree da ispezionare** — Confrontare il backend CUDA con il parent llama.cpp e identificare patch realmente divergenti.
- righe 15–18: **Idee candidabili per DS4** — Rilevamento GPU, selezione offload e fallback robusti possono informare il runtime.
- righe 19–22: **Rischi** — Il grande numero di feature rende difficile attribuire prestazioni.
- righe 23–25: **Esperimento minimo** — Usarlo come controllo di configurazione GGUF solo dopo aver fissato release, parent commit e flag CUDA; non considerarlo un oracle indipendente da llama.cpp se condivide lo stesso codice numerico.
