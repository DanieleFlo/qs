# Fonti primarie CUDA

## Scopo e versione

Registro delle fonti NVIDIA usate dalle schede CUDA. Consultazione verificata
il 2026-08-11. Il toolkit locale è CUDA 12.4 (`nvcc`/`ptxas` 12.4.131,
`cuobjdump` 12.4.127); le pagine correnti possono descrivere funzioni più
recenti, quindi ogni comando va verificato con `--help` sulla macchina.

## Documentazione ufficiale

| Area | Fonte NVIDIA | Uso nella knowledge base |
|---|---|---|
| Modello e API | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | esecuzione asincrona, eventi, graph, pipeline e cooperative groups |
| Ottimizzazione generale | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | coalescing, gerarchia memoria, shared bank e metodologia |
| GA10x / `sm_86` | [Ampere Tuning Guide 12.3](https://docs.nvidia.com/cuda/archive/12.3.0/ampere-tuning-guide/) | limiti di occupancy e primitive Ampere trasferibili alla RTX 3090 |
| Compilatore | [NVCC Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) | `-lineinfo`, resource usage, warning spill/local e controllo `ptxas` |
| Binari | [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) | `cuobjdump`, `nvdisasm`, SASS, CFG e live range |
| ISA virtuale | [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | semantica di shuffle, barrier, load/store e async copy |
| Timeline | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) | launch gap, API CUDA, NVTX e statistiche temporali |
| Kernel metrics | [Nsight Compute](https://docs.nvidia.com/nsight-compute/) | replay, occupancy, SOL, memoria, sorgente e roofline |
| Libreria GEMM | [cuBLAS](https://docs.nvidia.com/cuda/cublas/) | comportamento numerico, algoritmi, stream, graph e cuBLASLt |

## Gerarchia dell'evidenza

Queste fonti sono Tier A per semantica e strumenti. Non dimostrano che una
tecnica migliori DS4: la promozione richiede profilo locale, previsione
falsificabile, gate numerico e benchmark end-to-end sulla RTX 3090.
