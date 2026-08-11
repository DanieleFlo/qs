# Fonti per il kernel engineering di DS4/Qwen

Questa pagina è un indice ragionato, non una raccolta di codice copiato. Le
fonti sono ordinate per affidabilità e per uso. Prima di trasferire un'idea in
DS4 vanno sempre verificati layout GGUF, shape reali, politica numerica e
risultato sulla RTX 3090.

La raccolta completa, archiviata e navigabile per intervalli di riga è in
[`docs/research/INDEX.md`](../docs/research/INDEX.md). Questa pagina resta la
selezione minima da leggere durante un esperimento.

## Tier A: contratti e strumenti primari

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) —
  fonte normativa per modello di esecuzione, memoria, occupancy e capacità
  compute. È preferibile a blog e tabelle aggregate.
- [Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) —
  fonte primaria per limiti e funzionalità delle compute capability 8.x. Il
  probe la usa soltanto per i campi architetturali che `nvidia-smi` non espone.
- [GeForce RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/) —
  scheda prodotto NVIDIA usata insieme ai clock letti dal driver per identificare
  GA102, bus e picchi nominali; i valori statici restano marcati nel JSON.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) —
  riferimento per capture NVTX, timeline e CLI. Il formato nativo nsys-rep è
  forward-compatible; gli export SQLite possono cambiare schema.
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/) —
  riferimento per metriche, section set, replay, filtri kernel/range ed export.
- [NVIDIA GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) — implementazione
  degli autori del lavoro ICLR 2025. È una fonte semantica forte per la
  ricorrenza, non una drop-in implementation per layout GGUF o GA102.
- [Transformers Qwen3.5](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md) —
  descrive lo stack 3:1 linear/full attention e i fallback. È utile come oracle
  semantico; non è una baseline prestazionale per DS4.

## Tier B: implementazioni mature da studiare, non copiare

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — confronto più vicino per
  GGUF e quantizzazioni K. Usare commit esatti, test backend e sorgenti CUDA;
  issue e commenti isolati restano evidenza debole finché non riprodotti.
- [vLLM](https://github.com/vllm-project/vllm) — fonte utile per scheduling,
  batching, CUDA Graph e kernel moderni. I target tipici A100/H100 e formati
  AWQ/GPTQ non sono automaticamente trasferibili a Q4_K_S su RTX 3090.
- [Marlin](https://github.com/IST-DASLab/marlin) — riferimento mirato per
  kernel autoregressivi mixed-precision. Le sue assunzioni di packing e shape
  devono essere confrontate con i blocchi GGUF di DS4.

Non si considera una repository affidabile solo perché pubblica token/s. Servono
commit, modello e hash, quantizzazione, context, batch, prompt/token count,
warm-up, ripetizioni, clock/power e gate numerico. Fork senza test, gist,
benchmark screenshot e issue non riprodotte sono piste, non baseline.

## Letteratura con versione HTML

- [Gated Delta Networks](https://arxiv.org/html/2412.06464) — semantica e
  motivazione della ricorrenza usata dalla famiglia ibrida.
- [Marlin](https://arxiv.org/html/2408.11743) — modello memory-bound,
  parallelismo e kernel mixed-precision per decoding autoregressivo.
- [LLM inference e roofline](https://arxiv.org/html/2402.16363) — quadro per
  separare limiti di memoria e calcolo; le stime vanno validate con contatori.
- [QUICK](https://arxiv.org/html/2402.10076) — interleaving e conflitti per
  inferenza quantizzata; utile come pattern, non come prova per i layout DS4.
- [FlashDecoding++](https://arxiv.org/html/2311.01282) — parallelizzazione del
  decode attention a context lungo.

I paper forniscono modelli e trasformazioni candidate, non verdetti. Il harness
accetta una tecnica soltanto dopo correctness, microbenchmark sulle shape reali
e benchmark end-to-end fuori dal profiler.
