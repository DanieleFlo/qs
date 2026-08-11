# club3090 llama.cpp MoE-cache branch

## Classificazione

[Branch sorgente](https://github.com/noonghunna/llama.cpp/tree/stack/club3090-moe-cache). Fork llama.cpp specializzato per cache degli esperti MoE e overlap su Ampere; commit `e0a20894906a79dad7016cb3c12a237a8c5230b3`. Evidenza B specializzata.

## Problema risolto

Su modelli MoE molto più grandi della VRAM, una cache di esperti può ridurre trasferimenti ripetuti. Il ramo aggiunge scatter parallelo, overlap device-to-host, gestione del budget e batching top-k; la documentazione operativa riporta misure su due RTX 3090.

## Perché non è una soluzione diretta

Qwen3.6 27B DS4 è denso/ibrido e interamente residente in circa 14,76 GiB: non ha lo stesso collo di bottiglia degli esperti offloaded. Copiare il sistema di cache aggiungerebbe complessità senza rimuovere traffico dei pesi residenti.

## Lezione trasferibile

Sono utili il metodo di sovrapposizione, la gestione esplicita del reserve VRAM e i test che dimostrano quando una cache rimane inattiva. Questi principi possono servire per prefetch, KV persistente o multi-GPU, non per il primo MMVQ denso.

## Evidenza da richiedere

Diff contro il parent, misura PCIe, cache hit, picco VRAM, prefill/decode separati e correctness. I numeri DeepSeek MoE non entrano nel roofline Qwen denso.

## Decisione DS4

Catalogare, non portare ora. Tornare a questo ramo solo se il profiler mostra trasferimenti o cache miss, oppure quando DS4 ottimizzerà MoE/SSD streaming.
