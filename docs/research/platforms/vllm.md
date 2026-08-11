# vLLM

## Ruolo

[Repository ufficiale](https://github.com/vllm-project/vllm). Runtime di serving ad alto throughput con PagedAttention, batching continuo, CUDA Graph e più famiglie di kernel. Audit al commit `87668ab69b3e2c849a607ece36e8a43bde7c7ee5`.

## Percorso di esecuzione rilevante

Il runtime separa scheduler, cache manager, model executor e backend attention. Può scegliere FlashAttention, FlashInfer, Triton, CUTLASS e componenti TensorRT-LLM; usa inoltre `torch.compile` e graph capture. Dichiara supporto per modelli ibridi e state-space, incluso Qwen3.5.

## File e componenti da ispezionare

Le aree utili sono `vllm/attention`, `vllm/model_executor/layers/quantization`, `vllm/model_executor/layers/mamba` e `csrc/quantization`. Per ogni risultato bisogna identificare il backend effettivamente selezionato: “vLLM” da solo non identifica un kernel.

## Lezioni trasferibili alla RTX 3090

Chunked prefill, paged KV, scheduling prefill/decode e CUDA Graph sono trasferibili come principi. Molti kernel sono ottimizzati per datacenter, FP8 o architetture successive; sm_86, 24 GiB e Q4_K GGUF richiedono validazione dedicata.

## Evidenza e limiti

Evidenza A per architettura del serving e dispatch documentato. Evidenza B o C per una configurazione consumer non riprodotta. I benchmark multi-request non possono essere confrontati con il decode batch 1 di DS4 senza ricostruire workload e metriche.

## Domande operative

Studiare come il backend Qwen ibrido conserva gli stati ricorrenti, quando spezza il prefill e quali kernel usa davvero per W4A8/AWQ/GPTQ. Non assumere compatibilità con i K-quant GGUF.
