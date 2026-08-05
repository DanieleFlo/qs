# Numerica CUDA e determinismo

Questo documento traduce la documentazione NVIDIA in controlli concreti per il
drift DS4/llama.cpp.

## Floating point non associativo

Ogni somma/prodotto in precisione finita arrotonda. Quindi `(a+b)+c` puo'
differire da `a+(b+c)` e una riduzione parallela puo' divergere da una
sequenziale pur implementando la stessa formula. FMA calcola moltiplicazione e
somma con un solo arrotondamento, mentre due istruzioni ne fanno due; il
risultato e' spesso leggermente diverso.

Fonti primarie:

- [CUDA Best Practices, Numerical Accuracy and Precision](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#numerical-accuracy-and-precision)
- [NVIDIA Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)

Conseguenza: DS4-vs-llama.cpp non deve richiedere bit-exact se usa kernel o
ordini diversi, ma la differenza deve restare nell'inviluppo misurato e non
cambiare i 32 token greedy del gate.

## TF32, FP32 e kernel custom

TF32 usa Tensor Core con input a precisione ridotta rispetto a FP32 e
accumulatore float. In DS4:

- release: `CUBLAS_TF32_TENSOR_OP_MATH` salvo override;
- quality mode o `DS4_CUDA_NO_TF32`: `CUBLAS_DEFAULT_MATH`;
- kernel custom: compilati separatamente, attualmente con `--use_fast_math`.

Disabilitare TF32 non disabilita fast-math. La matrice diagnostica deve
compilare una variante senza fast-math, senza trasformarla in un secondo
percorso semantico permanente.

Fonte primaria:
[CUDA Programming Guide, TF32](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#alternate-floating-point).

## Effetti di `--use_fast_math`

Il flag aggrega opzioni di compilazione che privilegiano throughput e usa
intrinsic rapide per alcune operazioni. E' particolarmente rilevante per
divisione, radice, esponenziali e denormal; puo' influenzare RMSNorm, sigmoid,
SiLU, softmax e Gated DeltaNet anche se le GEMM cuBLAS restano FP32.

Durante la diagnosi si confrontano cubin ottenuti dalla stessa toolchain e con
il solo flag variato. Se la variante precisa risolve il gate, si restringe poi
il cambiamento alla primitiva responsabile invece di disabilitare globalmente
le ottimizzazioni senza misura.

Riferimento del compilatore:
[NVCC Compiler Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math).

## Riproducibilita' cuBLAS

NVIDIA garantisce in generale risultati bitwise ripetibili per la stessa
versione del toolkit, stessa architettura e stesso numero di SM quando e'
attivo un solo stream. La garanzia puo' venir meno con stream concorrenti che
condividono workspace, versioni diverse, algoritmi diversi o routine basate su
atomiche. Le mitigazioni documentate includono workspace separati, un handle
per stream, cuBLASLt con workspace posseduto dall'applicazione o
`CUBLAS_WORKSPACE_CONFIG`.

Fonte primaria:
[cuBLAS, Results Reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility).

Controllo DS4:

1. tre run identiche e SHA-256 di logits/trace;
2. registrare toolkit, driver, GPU e numero di stream;
3. se differiscono, identificare handle, stream, workspace e atomiche prima di
   confrontare semanticamente con llama.cpp;
4. non accettare una media di run non deterministiche come oracle.

## Errori attesi contro errori sospetti

Differenze di pochi ULP distribuite e stabili possono essere compatibili con
FMA o riduzioni. Sono sospetti:

- crescita molto superiore all'inviluppo llama CPU/CUDA;
- bias coerente, permutazioni o blocchi ripetuti;
- cambio improvviso di shape/layout;
- NaN/Inf;
- divergenza che dipende dalla storia della sessione quando fresh e replay
  dovrebbero coincidere;
- variazione fra run identiche;
- top-1 che cambia con margine ampio.

Per questo il confronto conserva sia metriche vettoriali sia decisionale:
MAE/RMSE/max, coseno, errore centrato, top-k overlap/rank, margine top-1/top-2,
NLL e longest common prefix greedy.

## Checklist prima di modificare un kernel

- input e pesi dello stadio hanno stessi token, shape, layout e checksum;
- output DS4 e llama sono prelevati allo stesso confine semantico;
- trace fresh e incrementale DS4 coincidono quando previsto;
- DS4 e' deterministico su tre run;
- e' noto l'inviluppo llama CPU/CUDA con la stessa revisione;
- TF32 e fast-math sono stati isolati separatamente;
- esiste un oracle piccolo della primitiva o una formula indipendente;
- la correzione proposta migliora il primo stadio divergente, non soltanto i
  logits finali.
