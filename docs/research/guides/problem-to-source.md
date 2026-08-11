# Mappa problema → soluzione esistente

## Decode Q4_K memory-bound

Leggere Marlin, QUICK e Atom; poi confrontare llama.cpp MMVQ/vec-dot, ik_llama MMVQ e ExLlamaV3 GEMM. Cercare caricamento cooperativo, packing per lane, riuso di scale e dispatch GEMV distinto da GEMM.

## Prefill Q4_K compute-bound

Confrontare llama.cpp MMQ, kernel Marlin/QServe e schedule MLC. Cercare tile che aumentano riuso senza superare registri/shared memory di sm_86.

## Full-attention decode a context lungo

Leggere FlashDecoding++, FlashInfer e SparQ. Confrontare llama.cpp fattn, vLLM backend attention e le istanze Ampere dei fork. Misurare separatamente QK, softmax e PV prima di fondere.

## Gated Delta Net

Partire dal paper GDN e dall'implementazione autori; confrontare llama.cpp e ik_llama `gated_delta_net.cu`. Verificare layout dello stato, mapping delle teste value e ordine delle riduzioni con l'oracle DS4.

## KV cache e 24 GiB

Leggere PagedAttention, KIVI e KVQuant. Usare club-3090 e buun soltanto per failure mode consumer. Separare riduzione della capacità, traffico per token e perdita numerica.

## Launch overhead e CUDA Graph

Confrontare vLLM, TensorRT-LLM e SGLang. Le shape dinamiche di MTP e attention ibrida possono rendere errata una capture apparentemente veloce: correctness per shape prima della latenza.

## Speculative e MTP

Leggere i due paper speculative, Medusa e multi-token prediction. Confrontare llama.cpp/ik_llama e i failure report club-3090. Misurare acceptance, costo verifier e rollback dello stato GDN.

## Offload e memoria insufficiente

Leggere FlexGen e le ricette consumer, ma non confondere throughput batch con latenza batch 1. Il costo PCIe e la sovrapposizione vanno misurati sul rig reale.
