# Toolchain e ispezione binaria CUDA

## Domanda operativa

Prima di cambiare tile, numero di warp o shared memory bisogna sapere cosa ha
effettivamente prodotto `ptxas`: registri per thread, stack, spill, memoria
locale/shared e SASS. Il sorgente CUDA e il PTX non bastano a descrivere il
codice eseguito da GA102.

Fonti: [registro NVIDIA](sources.md).

## Inventario locale 2026-08-11

| Strumento | Stato | Versione/ruolo |
|---|---|---|
| `/usr/local/cuda/bin/nvcc` | presente | 12.4.131, driver di compilazione |
| `/usr/local/cuda/bin/ptxas` | presente | 12.4.131, PTX → cubin/SASS |
| `/usr/local/cuda/bin/cuobjdump` | presente | 12.4.127, legge cubin e host binary |
| `nvdisasm` | assente | disassembly avanzato, CFG e live range |
| `ncu` | assente | metriche hardware per kernel |
| `nsys` | assente | timeline CPU/CUDA/NVTX |

L'assenza dei tre ultimi strumenti è una capacità mancante, non una ragione per
inventare conclusioni. `cuobjdump` consente già un audit statico riproducibile.

## Comandi canonici

Mostrare versioni e opzioni realmente disponibili:

```sh
/usr/local/cuda/bin/nvcc --version
/usr/local/cuda/bin/ptxas --version
/usr/local/cuda/bin/cuobjdump --version
/usr/local/cuda/bin/cuobjdump --help
```

Compilare per il target reale conservando correlazione sorgente e chiedendo a
`ptxas` statistiche e warning:

```sh
/usr/local/cuda/bin/nvcc -O3 -g -lineinfo --use_fast_math -arch=sm_86 \
  -Xcompiler -march=native -Xcompiler -pthread \
  --resource-usage \
  -Xptxas=-warn-spills,-warn-lmem-usage \
  -c -o /tmp/ds4_cuda.inspect.o ds4_cuda.cu
```

`-G` non va usato per benchmark: la documentazione NVCC segnala che può
disabilitare le ottimizzazioni device. Per profiling usare `-lineinfo`.

Leggere tutte le risorse dal binario realmente benchmarkato:

```sh
/usr/local/cuda/bin/cuobjdump --dump-resource-usage ./ds4-bench
/usr/local/cuda/bin/cuobjdump --dump-sass ./ds4-bench
```

Filtrare è solo una presentazione; l'artefatto grezzo va conservato insieme a
SHA-256 del binario e flags di build:

```sh
/usr/local/cuda/bin/cuobjdump --dump-resource-usage ./ds4-bench \
  | grep -A2 qwen35_
```

Quando `nvdisasm` sarà disponibile, estrarre prima un cubin con `cuobjdump` e
usare `--print-line-info`, `--print-life-ranges` e `--output-control-flow-graph`.
Il tool richiede relocation complete per l'analisi di controllo; in caso
contrario ricompilare con `ptxas --preserve-relocs` o disabilitare la dataflow
analysis solo dichiarandone il limite.

## Lettura corretta delle risorse

- `REG` è per thread. Moltiplicarlo per i thread del blocco e confrontarlo con
  i 65.536 registri/SM; l'allocazione avviene a granularità hardware.
- `STACK` e spill non sono sinonimi. Confermare spill load/store dal report
  `ptxas` o dalle metriche/SASS.
- Occupancy teorica non è throughput: serve a falsificare geometrie impossibili,
  non a scegliere da sola il kernel.
- Una riduzione dei registri può peggiorare il tempo se introduce recompute o
  memoria locale. `--maxrregcount` è un esperimento, mai una cura automatica.
- PTX espone una ISA virtuale; cache operator e istruzioni finali vanno
  verificati nel SASS del cubin `sm_86`.

## Evidenza DS4 già raccolta

`cuobjdump --dump-resource-usage ./ds4-bench` sul binario corrente riporta:

| Kernel | REG | STACK | SHARED | Interpretazione iniziale |
|---|---:|---:|---:|---|
| Q4_K F32 matvec warp8 | 40 | 0 | 0 | 256 thread: i registri consentono 6 CTA/SM; il limite thread dà 48 warp, quindi non è register-occupancy-bound |
| split-K partial | 30 | 0 | 32 B | 768 CTA con 32 partizioni coprono ampiamente 82 SM |
| split-K merge | 36 | 0 | 0 | solo 24 CTA, ma il lavoro per CTA è piccolo |
| GDN state warp4 | 32 | 0 | 1.552 B | nessuna evidenza statica di spill nel decode |
| GDN rows | 255 | 96 B | 2.064 B | allarme per il percorso multi-riga/prefill, non attribuirlo al decode senza verificare il dispatch |

Questa misura respinge l'ipotesi semplice “warp8 Q4 è lento perché i 40
registri riducono l'occupancy”: su `sm_86` la geometria può raggiungere il
massimo architetturale di 48 warp/SM. Il prossimo discriminante deve essere
instruction throughput, latenza dipendente o traffico memoria misurato.

## Output da memorizzare per esperimento

```text
toolkit e driver:
comando nvcc completo:
SHA-256 binario:
simbolo kernel e demangling:
REG / STACK / spill load-store / SHARED:
block, grid e occupancy teorica:
opcode o live range sospetti:
ipotesi che la misura supporta o falsifica:
```
