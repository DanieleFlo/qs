# Indice: platforms

Indice generato; non modificare a mano.

## [Aphrodite Engine](aphrodite-engine.md)

Combina scheduler, cache, quantizzazione, speculative decoding e kernel esterni o custom.

- righe 3–6: **Ruolo** — Repository ufficiale.
- righe 7–10: **Percorso di esecuzione rilevante** — Combina scheduler, cache, quantizzazione, speculative decoding e kernel esterni o custom.
- righe 11–14: **File e componenti da ispezionare** — Partire dalla matrice quantization e dal model runner, quindi risalire al backend kernel selezionato.
- righe 15–18: **Lezioni trasferibili alla RTX 3090** — Può mostrare integrazioni consumer non prioritarie per upstream vLLM.
- righe 19–22: **Evidenza e limiti** — Evidenza B: progetto attivo e verificabile, ma meno autorevole degli upstream che implementano direttamente il kernel.
- righe 23–25: **Domande operative** — Usarlo per scoprire backend e combinazioni già tentate, poi validare l'idea nella fonte primaria del kernel e sul microbenchmark DS4.

## [ExLlamaV3](exllamav3.md)

Usa un'estensione C++/CUDA con quantizzazione EXL3, cache KV a 2–8 bit e GEMM ispirato a Marlin.

- righe 3–6: **Ruolo** — Repository ufficiale.
- righe 7–10: **Percorso di esecuzione rilevante** — Usa un'estensione C++/CUDA con quantizzazione EXL3, cache KV a 2–8 bit e GEMM ispirato a Marlin.
- righe 11–14: **File e componenti da ispezionare** — Le aree interessanti sono l'estensione CUDA, la documentazione EXL3, i kernel GEMM e la cache quantizzata.
- righe 15–18: **Lezioni trasferibili alla RTX 3090** — È una delle fonti più vicine all'hardware consumer.
- righe 19–22: **Evidenza e limiti** — Evidenza A per codice e limitazioni dichiarate; B per benchmark 3090Ti storici di ExLlamaV2.
- righe 23–25: **Domande operative** — Studiare come vengono ammortizzati unpack, scale e attivazioni nel decode.

## [llama.cpp](llama-cpp.md)

Il grafo modello seleziona operazioni GGML; il backend CUDA decide fra GEMV/MMVQ, MMQ/GEMM, cuBLAS e kernel attention.

- righe 3–6: **Ruolo** — Repository ufficiale.
- righe 7–10: **Percorso di esecuzione rilevante** — Il grafo modello seleziona operazioni GGML; il backend CUDA decide fra GEMV/MMVQ, MMQ/GEMM, cuBLAS e kernel attention.
- righe 11–14: **File da ispezionare** — `ggml/src/ggml-cuda/mmvq.cu`, `vecdotq.cuh` e `quantize.cu` mostrano packing e dot product K-quant nel decode.
- righe 15–18: **Lezioni trasferibili alla RTX 3090** — La separazione GEMV decode/GEMM prefill, le configurazioni Ampere, il caricamento cooperativo dei blocchi quantizzati e la selezione per shape sono direttamente pertinenti.
- righe 19–22: **Evidenza e limiti** — Evidenza A per semantica GGUF e implementazione CUDA mantenuta; evidenza B per prestazioni sulla nostra macchina finché non viene compilato lo stesso commit e misurato con lo stesso GGUF.
- righe 23–25: **Domande operative** — Confrontare per primo MMVQ Q4_K/Q5_K/Q6_K, full-attention decode e GDN.

## [MLC LLM](mlc-llm.md)

La compilazione trasforma il grafo, genera kernel specializzati e costruisce un runtime distribuibile.

- righe 3–6: **Ruolo** — Repository ufficiale.
- righe 7–10: **Percorso di esecuzione rilevante** — La compilazione trasforma il grafo, genera kernel specializzati e costruisce un runtime distribuibile.
- righe 11–14: **File e componenti da ispezionare** — Esaminare le definizioni dei modelli, le quantization recipes, i passaggi compiler e gli schedule TensorIR.
- righe 15–18: **Lezioni trasferibili alla RTX 3090** — La specializzazione per shape e target può guidare il dispatch DS4.
- righe 19–22: **Evidenza e limiti** — Evidenza A per il framework di compilazione; B per schedule specifici verificati sul target.
- righe 23–25: **Domande operative** — Usare MLC come generatore di ipotesi su tile, fusione e layout; verificare poi manualmente minimum bytes, codice SASS e correttezza numerica nel microbenchmark DS4.

## [Ollama](ollama.md)

Ollama gestisce download, configurazione, processi runner, API e lifecycle.

- righe 3–6: **Ruolo** — Repository ufficiale.
- righe 7–10: **Percorso di esecuzione rilevante** — Ollama gestisce download, configurazione, processi runner, API e lifecycle.
- righe 11–14: **File e componenti da ispezionare** — Individuare il pin o submodule del runner, i flag di build, le variabili CUDA e le policy di memoria.
- righe 15–18: **Lezioni trasferibili alla RTX 3090** — È utile per packaging, rilevamento GPU e configurazioni realmente usate dagli utenti.
- righe 19–22: **Evidenza e limiti** — Evidenza A per integrazione e build; evidenza derivata per i kernel.
- righe 23–25: **Domande operative** — Registrare commit del runner, build flags, graph policy e quantizzazione prima di usare Ollama come oracle o baseline prestazionale.

## [SGLang](sglang.md)

Il server runtime orchestra batching, cache e parallelismo; `sgl-kernel` concentra molte primitive CUDA/Triton e integra backend attention esterni.

- righe 3–6: **Ruolo** — Repository ufficiale.
- righe 7–10: **Percorso di esecuzione rilevante** — Il server runtime orchestra batching, cache e parallelismo; `sgl-kernel` concentra molte primitive CUDA/Triton e integra backend attention esterni.
- righe 11–14: **File e componenti da ispezionare** — Esaminare `python/sglang/srt` per scheduling e model runner, poi il package `sgl-kernel` per quantized GEMM, MoE e attention.
- righe 15–18: **Lezioni trasferibili alla RTX 3090** — Prefix reuse, chunked prefill e overlap sono utili al runtime.
- righe 19–22: **Evidenza e limiti** — Evidenza A per il runtime mantenuto, B per kernel supportati e C per ricette non accompagnate da commit, modello e log.
- righe 23–25: **Domande operative** — Confrontare le shape e le fusioni del percorso Qwen ibrido con DS4; cercare soprattutto soluzioni a state update, attention decode e quantized matvec, separandole dalle ottimizzazioni di scheduling multiutente.

## [TensorRT-LLM](tensorrt-llm.md)

Costruisce engine specializzati combinando graph optimization, plugin e kernel CUDA.

- righe 3–6: **Ruolo** — Repository ufficiale.
- righe 7–10: **Percorso di esecuzione rilevante** — Costruisce engine specializzati combinando graph optimization, plugin e kernel CUDA.
- righe 11–14: **File e componenti da ispezionare** — Le implementazioni risiedono principalmente in `cpp/tensorrt_llm/kernels`, nei plugin e nei runtime executor.
- righe 15–18: **Lezioni trasferibili alla RTX 3090** — Sono utili la specializzazione per shape, la fusione guidata dal traffico e la distinzione fra context e generation phase.
- righe 19–22: **Evidenza e limiti** — Evidenza A per comportamento NVIDIA e requisiti dichiarati.
- righe 23–25: **Domande operative** — Usare TensorRT-LLM per capire fusioni e dispatch, quindi cercare una formulazione implementabile autonomamente in C/CUDA DS4.

## [vLLM](vllm.md)

Il runtime separa scheduler, cache manager, model executor e backend attention.

- righe 3–6: **Ruolo** — Repository ufficiale.
- righe 7–10: **Percorso di esecuzione rilevante** — Il runtime separa scheduler, cache manager, model executor e backend attention.
- righe 11–14: **File e componenti da ispezionare** — Le aree utili sono `vllm/attention`, `vllm/model_executor/layers/quantization`, `vllm/model_executor/layers/mamba` e `csrc/quantization`.
- righe 15–18: **Lezioni trasferibili alla RTX 3090** — Chunked prefill, paged KV, scheduling prefill/decode e CUDA Graph sono trasferibili come principi.
- righe 19–22: **Evidenza e limiti** — Evidenza A per architettura del serving e dispatch documentato.
- righe 23–25: **Domande operative** — Studiare come il backend Qwen ibrido conserva gli stati ricorrenti, quando spezza il prefill e quali kernel usa davvero per W4A8/AWQ/GPTQ.
