# Comandi CUDA per misurare prima di ottimizzare

## Ordine degli strumenti

1. Harness DS4 fuori profiler: stabilisce tok/s, correttezza e rumore.
2. Timer CUDA/eventi o profiler interno: attribuisce fasi senza cercare ancora
   una causa microarchitetturale.
3. Nsight Systems: dimostra launch gap, sincronizzazioni, memcpy e inattività.
4. Nsight Compute: profila soltanto i kernel già classificati come hotspot.
5. `cuobjdump`/`nvdisasm`: collega metriche a risorse, SASS e dipendenze.

Invertire l'ordine induce facilmente a ottimizzare un kernel visibile ma
irrilevante end-to-end. Fonti: [registro NVIDIA](sources.md).

## Timing con eventi

Gli eventi registrano timestamp nella stream e sono appropriati per kernel o
sequenze GPU. Il pattern canonico è record start, lavoro, record stop,
sincronizzazione dello stop e `cudaEventElapsedTime`. Non inserire una
`cudaDeviceSynchronize()` fra ogni operazione in un benchmark end-to-end:
serializza il grafo e misura una workload diversa.

I timer dettagliati DS4 inseriscono sincronizzazioni intenzionali e servono
solo all'attribuzione. I tok/s dell'harness restano il criterio prestazionale.

## Nsight Systems: timeline prima delle metriche

Verificare installazione:

```sh
nsys --version
nsys status --environment
```

Traccia limitata a CUDA/NVTX, senza sampling CPU non necessario:

```sh
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --stats=true --force-overwrite=true \
  -o performance-results/qwen-longctx-timeline \
  ./ds4-bench [argomenti harness]
```

Riassumere un report esistente senza rilanciare:

```sh
nsys stats performance-results/qwen-longctx-timeline.nsys-rep
```

Cercare: gap host fra kernel, `cudaDeviceSynchronize`, allocazioni nel loop,
memcpy H↔D, kernel minuscoli ripetuti e assenza di overlap. Una timeline non
misura cache hit, stall reason o transazioni: per quelli serve `ncu`.

## Nsight Compute: un hotspot, poche sezioni

Prima elencare ciò che la versione locale supporta:

```sh
ncu --version
ncu --list-sets
ncu --list-sections
ncu --query-metrics-mode suffix --query-metrics
```

Primo pass focalizzato, con correlazione sorgente dal `-lineinfo`:

```sh
ncu --kernel-name-base function \
  --kernel-name regex:qwen35_q4_k_f32_matvec_warp8_kernel \
  --launch-count 1 \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section SourceCounters \
  --export performance-results/q4-f32-warp8 \
  ./ds4-bench [argomenti minimi riproducibili]
```

Per split-K cambiare il filtro al kernel partial e poi al merge: non sommare
metriche di due kernel come se fossero una singola launch.

## Replay, cache e clock

`ncu` può richiedere più replay per raccogliere le metriche. Per default può
svuotare le cache fra replay e controllare i clock: questo aumenta
riproducibilità interna ma non replica necessariamente il decode residente.
Registrare sempre `--cache-control` e `--clock-control`; confrontare candidati
con impostazioni identiche e non trasferire il tempo profilato ai tok/s reali.

Usare il set `full` solo dopo avere isolato una launch: raccogliere tutto su un
prefill lungo può moltiplicare il runtime e perturbare memoria/stato.

## Albero di classificazione

| Evidenza | Classificazione | Prossima domanda |
|---|---|---|
| timeline con gap host consistenti | launch-bound | graph/fusione riducono davvero ms/token? |
| DRAM throughput alto e stall memory | bandwidth-bound | byte richiesti sono utili o amplificati? accessi coalesced? |
| bassa DRAM ma L1/L2 saturi | cache/on-chip bound | layout, riuso e working set per CTA |
| issue slot alto, memoria non satura | instruction/compute-bound | unpack, conversioni, dipendenze, special function |
| occupancy bassa per registri/shared | resource-bound | tile alternativo migliora latency hiding senza spill? |
| occupancy alta ma eligible warp basso | dependency-bound | catene seriali, barrier, riduzioni o scoreboard |

## Regola di memoria

Salvare report grezzo, comando, versione tool, filtro del kernel, invocation,
clock/cache control, binario e SHA-256. Una tabella copiata dalla GUI senza
questi dati è Tier C e non decide KEEP/REJECT.
