# llama.cpp

## Ruolo

[Repository ufficiale](https://github.com/ggml-org/llama.cpp). È il riferimento più vicino a DS4 per GGUF, K-quant, esecuzione locale consumer e backend CUDA custom. Audit fissato al commit `2468576f241235452013308597e6de1b78866996`.

## Percorso di esecuzione rilevante

Il grafo modello seleziona operazioni GGML; il backend CUDA decide fra GEMV/MMVQ, MMQ/GEMM, cuBLAS e kernel attention. Per Qwen ibrido sono rilevanti anche lo stato Gated Delta Net, la convoluzione causale e il lifetime della cache ricorrente.

## File da ispezionare

`ggml/src/ggml-cuda/mmvq.cu`, `vecdotq.cuh` e `quantize.cu` mostrano packing e dot product K-quant nel decode. `mmq.cu` copre prefill/batch. `fattn.cu` e relativi template coprono attention. `gated_delta_net.cu` è il confronto semantico diretto per i layer ricorrenti.

## Lezioni trasferibili alla RTX 3090

La separazione GEMV decode/GEMM prefill, le configurazioni Ampere, il caricamento cooperativo dei blocchi quantizzati e la selezione per shape sono direttamente pertinenti. DS4 deve conservare il proprio layout GGUF e ordine numerico: una copia meccanica del kernel non sarebbe corretta.

## Evidenza e limiti

Evidenza A per semantica GGUF e implementazione CUDA mantenuta; evidenza B per prestazioni sulla nostra macchina finché non viene compilato lo stesso commit e misurato con lo stesso GGUF. Issue o singoli commenti non sono baseline.

## Domande operative

Confrontare per primo MMVQ Q4_K/Q5_K/Q6_K, full-attention decode e GDN. Registrare geometria, registri, shared memory, numero di letture dei pesi e differenza numerica rispetto a DS4.
