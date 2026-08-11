# KoboldCpp

## Classificazione

[Repository](https://github.com/LostRuins/koboldcpp). Fork diretto di llama.cpp con distribuzione self-contained, GUI e molte API; AGPL-3.0. Audit al commit `bf9b9bd45512`. Evidenza B per integrazione, C per una presunta ottimizzazione kernel non ricondotta a diff e benchmark.

## Perché è rilevante

Mostra come un fork consumer mantiene compatibilità, partial offload, binari Windows e molte modalità d'uso. È utile per problemi di packaging e configurazioni reali su hardware domestico.

## Aree da ispezionare

Confrontare il backend CUDA con il parent llama.cpp e identificare patch realmente divergenti. Molte funzionalità sono lato server/UI e non incidono sul kernel Qwen.

## Idee candidabili per DS4

Rilevamento GPU, selezione offload e fallback robusti possono informare il runtime. Per i kernel, la fonte primaria rimane il diff rispetto al parent o la pull request upstream.

## Rischi

Il grande numero di feature rende difficile attribuire prestazioni. La licenza e il rapporto di fork impongono di studiare il comportamento senza copiare implementazioni. Un binario più semplice da eseguire non è necessariamente un kernel più veloce.

## Esperimento minimo

Usarlo come controllo di configurazione GGUF solo dopo aver fissato release, parent commit e flag CUDA; non considerarlo un oracle indipendente da llama.cpp se condivide lo stesso codice numerico.
