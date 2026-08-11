# Ollama

## Ruolo

[Repository ufficiale](https://github.com/ollama/ollama). Distribuzione locale, model manager e server; per GGUF/CUDA riusa principalmente il runtime llama.cpp fissato dalla propria build. Audit al commit `96fb6d2fa9944bb76e2c5c3f73086a12457b301b`.

## Percorso di esecuzione rilevante

Ollama gestisce download, configurazione, processi runner, API e lifecycle. Le primitive CUDA non costituiscono normalmente un'implementazione indipendente: bisogna risalire al commit llama.cpp vendorizzato.

## File e componenti da ispezionare

Individuare il pin o submodule del runner, i flag di build, le variabili CUDA e le policy di memoria. Per un confronto kernel-level, aprire poi il sorgente del commit upstream esatto.

## Lezioni trasferibili alla RTX 3090

È utile per packaging, rilevamento GPU e configurazioni realmente usate dagli utenti. Non va contato come seconda conferma indipendente quando esegue lo stesso kernel llama.cpp.

## Evidenza e limiti

Evidenza A per integrazione e build; evidenza derivata per i kernel. Un benchmark “Ollama contro llama.cpp” può confrontare configurazioni, non implementazioni, se entrambi risolvono allo stesso codice CUDA.

## Domande operative

Registrare commit del runner, build flags, graph policy e quantizzazione prima di usare Ollama come oracle o baseline prestazionale.
