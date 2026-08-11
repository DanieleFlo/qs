# Genesis vLLM patches

## Classificazione

[Repository](https://github.com/Sandermage/genesis-vllm-patches). Overlay runtime su una wheel vLLM fissata, non fork permanente; commit `34e269301cc3df71ae4b0da00a0a159b16b4e5d8`. Evidenza B operativa, C finché i numeri non sono riprodotti localmente.

## Problemi affrontati

Qwen3.6 ibrido su GPU consumer da 24 GiB: TurboQuant KV, stato GDN, MTP speculative, CUDA Graph con shape dinamiche, tool calling quantizzato e limiti di contesto. Il rig principale dichiarato usa due RTX A5000 Ampere sm_86; include preset 3090.

## Valore tecnico

Il registro patch associa versioni, anchor e lifecycle, e rimuove workaround quando upstream incorpora la correzione. Bench e correctness includono CV, acceptance MTP e test di tool calling: è un buon modello di provenance per esperimenti dinamici.

## Rischi

Centinaia di patch applicate insieme rendono difficile attribuire un guadagno. A5000 e 3090 condividono sm_86 ma differiscono per SM, clock, bandwidth e topologia; AWQ/AutoRound/TurboQuant non corrispondono a GGUF Q4_K_S.

## Idee candidabili per DS4

Test per interazione CUDA Graph × K+1 speculative, shape minime Marlin dopo tensor parallel, ricostruzione dello stato GDN e gate acceptance. Sono soprattutto failure test e criteri di dispatch.

## Decisione DS4

Usare il registro per trovare la patch e la issue primaria, poi isolare un solo meccanismo. Non importare l'overlay né considerare il claim aggregato 1,5× una previsione per DS4.
