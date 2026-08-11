# MLC LLM

## Ruolo

[Repository ufficiale](https://github.com/mlc-ai/mlc-llm). Compila modelli e operatori attraverso TVM per CUDA, Vulkan e altre piattaforme. Audit al commit `2f78caa4db0f90730a11ee3bb5cbd5f23bf67f9f`.

## Percorso di esecuzione rilevante

La compilazione trasforma il grafo, genera kernel specializzati e costruisce un runtime distribuibile. Schedule e layout sono scelti in funzione del target e possono essere autotunati.

## File e componenti da ispezionare

Esaminare le definizioni dei modelli, le quantization recipes, i passaggi compiler e gli schedule TensorIR. Il valore principale non è copiare un kernel generato, ma capire quali dimensioni e fusioni il compilatore considera.

## Lezioni trasferibili alla RTX 3090

La specializzazione per shape e target può guidare il dispatch DS4. Un risultato generato per Vulkan, mobile o tensor core con dtype diverso non è direttamente comparabile al percorso CUDA F32/Q4_K.

## Evidenza e limiti

Evidenza A per il framework di compilazione; B per schedule specifici verificati sul target. I log di autotuning senza hardware e commit sono transitori e non costituiscono una baseline.

## Domande operative

Usare MLC come generatore di ipotesi su tile, fusione e layout; verificare poi manualmente minimum bytes, codice SASS e correttezza numerica nel microbenchmark DS4.
