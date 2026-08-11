# SGLang

## Ruolo

[Repository ufficiale](https://github.com/sgl-project/sglang). Runtime di serving con scheduler, radix/prefix cache e libreria kernel specializzata. Audit al commit `b20c375c10443e4f4a5656689a04d514194364fd`.

## Percorso di esecuzione rilevante

Il server runtime orchestra batching, cache e parallelismo; `sgl-kernel` concentra molte primitive CUDA/Triton e integra backend attention esterni. Le prestazioni derivano dall'insieme scheduler più kernel, non da un singolo componente.

## File e componenti da ispezionare

Esaminare `python/sglang/srt` per scheduling e model runner, poi il package `sgl-kernel` per quantized GEMM, MoE e attention. Identificare sempre quale backend viene selezionato per la compute capability corrente.

## Lezioni trasferibili alla RTX 3090

Prefix reuse, chunked prefill e overlap sono utili al runtime. I kernel Triton o CUTLASS possono richiedere formati diversi da GGUF e avere dispatch centrato su Ada/Hopper; il beneficio su Ampere consumer non è implicito.

## Evidenza e limiti

Evidenza A per il runtime mantenuto, B per kernel supportati e C per ricette non accompagnate da commit, modello e log. Una patch di compatibilità Qwen non è automaticamente una patch prestazionale.

## Domande operative

Confrontare le shape e le fusioni del percorso Qwen ibrido con DS4; cercare soprattutto soluzioni a state update, attention decode e quantized matvec, separandole dalle ottimizzazioni di scheduling multiutente.
