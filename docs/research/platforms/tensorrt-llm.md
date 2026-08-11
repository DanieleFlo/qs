# TensorRT-LLM

## Ruolo

[Repository ufficiale](https://github.com/NVIDIA/TensorRT-LLM). È la fonte vendor più forte per runtime NVIDIA, plugin specializzati, quantizzazione e kernel di serving. Audit al commit `6b052851c9888138da3399f32c8cfe4e0ef40b28`.

## Percorso di esecuzione rilevante

Costruisce engine specializzati combinando graph optimization, plugin e kernel CUDA. Le prestazioni dipendono da dtype, plugin, profili di shape, CUDA Graph e tactic selezionata durante la build.

## File e componenti da ispezionare

Le implementazioni risiedono principalmente in `cpp/tensorrt_llm/kernels`, nei plugin e nei runtime executor. XQA, quantized GEMM, KV-cache kernels e configurazione dei graph batch sono le aree concettualmente più vicine.

## Lezioni trasferibili alla RTX 3090

Sono utili la specializzazione per shape, la fusione guidata dal traffico e la distinzione fra context e generation phase. Le release recenti possono richiedere CUDA o GPU non disponibili sulla 3090; una tecnica è candidabile solo se possiede un percorso sm_86.

## Evidenza e limiti

Evidenza A per comportamento NVIDIA e requisiti dichiarati. I numeri H100/A100 non prevedono GA102. L'assenza di una licenza SPDX semplice nel metadata richiede inoltre prudenza: si studiano idee e interfacce, non si copiano implementazioni.

## Domande operative

Usare TensorRT-LLM per capire fusioni e dispatch, quindi cercare una formulazione implementabile autonomamente in C/CUDA DS4. Verificare sempre istruzioni generate e requisito Tensor Core.
