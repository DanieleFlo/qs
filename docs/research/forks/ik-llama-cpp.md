# ik_llama.cpp

## Classificazione

[Repository](https://github.com/ikawrakow/ik_llama.cpp). Fork storico di llama.cpp ora evoluto indipendentemente, MIT, attivo e focalizzato su quantizzazioni e prestazioni. Audit al commit `26ceed9d4091`. Evidenza B+: codice verificabile e ampia attività, ma non upstream canonico.

## Perché è rilevante

Dichiara di avere introdotto o sperimentato quant repacking, Gated Delta Net fuso, tensor parallel, MTP e varianti CUDA prima dell'upstream. Supporta CUDA Turing o successiva, quindi include esplicitamente Ampere.

## Aree da ispezionare

`ggml/src/ggml-cuda/mmvq.cu`, `mmq.cu`, configurazioni Ampere, `fattn.cu` e `gated_delta_net.cu`. Le pull request conservate nel repository documentano regressioni, shape problematiche e motivazioni delle varianti.

## Idee candidabili per DS4

Quantized GEMV/MMQ per IQ/K-quant, fast attention token-generation per GQA, GDN fuso e repacking orientato alla GPU. Ogni idea deve essere rimappata ai blocchi Q4_K_S e alla policy Q8_K/F32 di DS4.

## Rischi

Il progetto avverte che alcune combinazioni repacked non hanno implementazione CUDA e possono ricadere sulla CPU. Divergenza dall'upstream, graph split e offload parziale possono cambiare semantica e prestazioni.

## Esperimento minimo

Compilare il commit fissato per `sm_86`, stesso GGUF e batch 1. Profilare MMVQ, GDN e attention; confrontare logits e non soltanto token/s. Una vittoria del formato IQ non prova una vittoria del layout GGUF K-quant.
