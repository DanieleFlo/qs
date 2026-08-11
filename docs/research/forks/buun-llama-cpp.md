# buun-llama-cpp

## Classificazione

[Repository](https://github.com/spiritbuun/buun-llama-cpp). Fork di ricerca molto attivo che si dichiara esplicitamente altamente sperimentale. Audit al commit `ba09e2a80bcf`. Evidenza B- per codice riproducibile, C per numeri non rieseguiti.

## Perché è rilevante

Contiene codec KV a bitrate variabile, quantizzazione trellis, attention CUDA specializzata, GDN e molte istanze template per dtype e head dimension. Pubblica misure Qwen3.6 su RTX 3090, quindi affronta esattamente vincoli GA102 e 24 GiB.

## Aree da ispezionare

Le directory `ggml/src/ggml-cuda` includono attention MMA/WMMA/vector, `gated_delta_net.cu`, MMVQ/MMQ e codec KV. La proliferazione di istanze mostra una strategia di specializzazione per shape, non un kernel universale.

## Idee candidabili per DS4

Dispatch per head dimension, KV cache compressa progressivamente, attention quantizzata e cattura/iniezione per speculative decoding. L'idea più utile può essere il criterio di dispatch, non il codec completo.

## Rischi

Il fork modifica contemporaneamente formato KV, attention, qualità e runtime: attribuire un guadagno è difficile. Le affermazioni di qualità e throughput sono auto-pubblicate e richiedono riproduzione. Non copiare codice o formati sperimentali nel release path.

## Esperimento minimo

Usare il fork come catalogo di ipotesi. Per una singola tecnica registrare commit, file, shape, minimum bytes e benchmark 3090; implementare poi una variante isolata DS4 con oracle full-logit.
