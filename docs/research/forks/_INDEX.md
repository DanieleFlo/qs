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

## [Analisi tecnica di `syv-ai/qwen38-27b-rtx3090`: da dove viene la velocità e quanto costa in qualità](syv-ai-qwen38-27b-rtx3090.md)

**Fonte primaria:** `syv-ai/qwen38-27b-rtx3090`.

- righe 3–20: **Provenienza e perimetro** — **Fonte primaria:** `syv-ai/qwen38-27b-rtx3090`.
- righe 21–42: **Conclusione principale** — La cosa più importante da capire è questa:
  - righe 31–42: **La mia classificazione** — ---
- righe 43–62: **1. Prima di tutto: cosa significano davvero 1.000, 381 e 132 tok/s** — I numeri più appariscenti della repo non devono essere messi sullo stesso piano.
- righe 63–169: **2. Le innovazioni, ordinate per qualità** — Questa è probabilmente una delle idee migliori dell'intera repo.
- righe 65–76: **1. Quantizzazione del solo drafter MTP — Classe A** — Questa è probabilmente una delle idee migliori dell'intera repo.
- righe 77–88: **2. Quantizzazione INT4 del drafter DFlash2 — Classe A** — Stesso principio.
- righe 89–100: **3. Draft vocabulary costruito sugli output del modello — Classe A** — Il drafter non proietta più sul vocabolario completo da ~248k token, ma su circa 40k token scelti osservando ciò che Qwen stesso tende a generare.
- righe 101–119: **4. Speculative decoding MTP — Classe A come algoritmo** — Qwen3.8-27B ha MTP nativo; l'architettura ufficiale ha 64 layer, 48 Gated DeltaNet + 16 attention layer e MTP addestrato nativamente.
- righe 120–135: **5. DFlash2 — Classe A per il principio, A− per questa integrazione custom** — DFlash2 propone un intero blocco di 7 token con un solo passaggio anziché concatenare quattro predizioni MTP.
- righe 136–149: **6. Context lookup drafting — Classe A** — Questa è l'idea dietro i 260-381 tok/s quando il modello copia o modifica materiale presente nel prompt.
- righe 150–161: **7. Prefix caching dello stato ibrido KV + GDN — Classe A/A−** — Su una conversazione con documento da 24k token, il follow-up passa da circa 23 secondi a circa **0,85-1,15 s TTFT**, riusando sia KV sia stato recurrente dal confine del blocco cached.
- righe 162–169: **8. Correzione dei gruppi KV ibridi e contabilizzazione CUDA graph — Classe A** — DFlash2 introduceva spreco di KV pool perché vLLM paddingava i gruppi di layer in modo sfavorevole.
- righe 170–202: **3. Ottimizzazioni “quasi pure”, ma che cambiano l'aritmetica** — È un kernel Triton progettato specificamente per il multi-query verify dello speculative decoding.
- righe 172–190: **9. Split-KV attention per il verify step — Classe A−** — È un kernel Triton progettato specificamente per il multi-query verify dello speculative decoding.
- righe 191–202: **10. Sampler small-top-k / sort-free — Classe A−** — Invece di ordinare l'intero vocab da ~248k elementi, usa un percorso specializzato quando `top-k <= 64`, più una softmax parallelizzata.
- righe 203–224: **4. FP16 recurrent state: probabilmente il miglior trade-off numericamente lossy — Classe B** — Questo punto è particolarmente interessante per Qwen3.8.
- righe 225–245: **5. Quantizzare `embed_tokens` e `lm_head` INT8 — Classe B** — Qui invece il target cambia senza ambiguità.
- righe 246–261: **6. `lm_head` GPTQ INT4 — Classe B/C** — Nella variante single-user più veloce anche il `lm_head` del **target** diventa INT4 GPTQ.
- righe 262–298: **7. INT8 delle attivazioni: qui inizia il vero quality-for-speed** — La tabella della repo è molto istruttiva.
  - righe 273–278: **Gate/up INT8: B** — +0,9% PPL per un buon aumento di throughput.
  - righe 279–286: **Intero MLP INT8: C** — È il default batch della repo: circa **942 tok/s e2e**, ma PPL sale del **2,2%**.
  - righe 287–298: **Tutte le lineari INT8: C−** — Arriva a ~1.042 tok/s e2e / ~1.222 steady decode, ma PPL è **+3,7%**.
- righe 299–331: **8. KV cache quantizzata: il punto su cui concordo maggiormente con la tua osservazione** — Qui bisogna correggere una possibile lettura del README.
  - righe 322–331: **Quindi la mia classificazione è:** — **BF16 KV fast:** preferibile se 64k bastano.
- righe 332–352: **9. KVarN K4V2 e cache 4/2 bit — Classe C per fidelity, ottima solo se serve context estremo** — KVarN consente di spingere la capacità molto più in alto: la repo riporta 262k che entrano con margine e needle test corretti fino a 240k.
- righe 353–371: **10. Il prerequisito nascosto: il body è già W4A16** — Prima ancora delle innovazioni della repo, il modello caricato è il checkpoint **Qwen3.8-27B-W4A16-AutoRound**, non il BF16 ufficiale.
- righe 372–405: **11. I casi realmente problematici: Classe D** — Questi non sono “quantizzazione con lieve drift”.
  - righe 376–383: **W4A8 Marlin con negative scales non corretto** — Il percorso originale di vLLM interpretava come unsigned scale requantizzate che potevano essere negative; il risultato era **output garbage pur mostrando benchmark eccellenti**.
  - righe 384–391: **DFlash2 backport senza semantic fix della temperatura** — Come detto prima, senza la patch si verificavano draft contro una `q` sbagliata per temperature intermedie.
  - righe 392–397: **FlashInfer/FP8 con k=4** — La repo segnala un **illegal memory access** quando una richiesta termina mentre un'altra è ancora in generazione; per questo `CTX=long` usa tre draft anziché quattro.
  - righe 398–405: **Sticky long-verify in batch** — Il soak test della repo ha trovato che il contatore globale del long-block poteva far dipendere la lunghezza del verify dalle altre richieste del batch; cambiando block size cambia anche l'ordine floating-point e in un c…
- righe 406–442: **12. Dove DS4 è più rigoroso della suite di questa repo** — Questo è probabilmente l'aspetto più importante per valutare quanto fidarsi delle affermazioni di qualità.
- righe 443–497: **13. Come rivaluterei i test della repo usando il metro DS4** — **Evidenza syv:** ottima PPL, differenza nulla a tre decimali.
  - righe 445–452: **FP16 recurrent state** — **Evidenza syv:** ottima PPL, differenza nulla a tre decimali.
  - righe 453–462: **INT8 `embed_tokens` e `lm_head`** — **Evidenza syv:** Frobenius error + PPL/task benchmark.
  - righe 463–470: **INT8 activations** — Qui la PPL mostra già il danno.
  - righe 471–482: **KV FP8/INT8/KVarN** — Qui un test DS4 è particolarmente necessario.
  - righe 483–497: **Speculative decoding** — Qui la teoria è molto più forte.
- righe 498–520: **14. Ranking finale delle innovazioni** — Se le ordinassi per rapporto **speed / fidelity**, la mia lista sarebbe:
- righe 521–543: **15. Qual è, quindi, la vera innovazione della repo?** — Non è principalmente “quantizzare tutto”.
- righe 544–557: **16. Le configurazioni che sceglierei** — **Massima fedeltà possibile sulla 3090:** terrei BF16 KV (`CTX=fast`), speculative decoding, drafter quantizzato, draft vocabulary, split-KV, sampler ottimizzato e prefix cache.
- righe 558–577: **17. Il test che manca per poter certificare queste conclusioni alla maniera di DS4** — Per portare questa repo allo standard di evidenza DS4, farei un'unica matrice Qwen “golden” con **lo stesso checkpoint, tokenizer, template, prompt e hardware**, registrando per ogni configurazione:
