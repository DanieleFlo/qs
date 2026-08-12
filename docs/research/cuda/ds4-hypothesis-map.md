# Mappa CUDA → prossimi esperimenti DS4

## Stato misurato da cui partire

A 10.666 token split-K32 porta il decode da 3,51 a 14,29 tok/s con 16/16 token
greedy identici. La conferma residente a cinque run misura 15,49 tok/s a 8K,
14,70 a 12K e 13,94 a 16K, contro 4,56/3,31/2,60; token, top-20 e gate dei
logits passano su tutte le frontiere. Split-K32 è quindi default automatico
da 8K, con legacy sotto soglia. Il token profilato attribuisce 46,40 ms a FFN,
18,92 ms all'attenzione ricorrente, 18,00 ms al ramo full-attention (11,78 ms
core), 1,84 ms all'output e 0,42 ms al readback.

Fonti e comandi: [fonti NVIDIA](sources.md),
[profiling](profiling-commands.md), [toolchain](toolchain-binary-inspection.md)
e [primitive sm_86](ampere-sm86-primitives.md).

## Classificazione aggiornata

Il problema iniziale era parallelismo insufficiente del core full-attention:
24 CTA seriali sulla sequenza. Split-K lo ha corretto esponendo 768 CTA.
Il residuo è prevalentemente il percorso pesi Q4_K F32 comune a FFN e alle
proiezioni GDN/full-attention, non la scansione KV.

L'audit statico falsifica una prima spiegazione: il matvec Q4_K warp8 usa 40
registri/thread, zero stack/shared; con blocchi da 256 thread può risiedere a 6
CTA/SM e raggiungere 48 warp/SM. Provare warp4 solo per “alzare occupancy” non ha
quindi una previsione valida.

## Esperimenti ordinati

| Priorità | Ipotesi falsificabile | Misura richiesta prima del codice | Previsione | Gate |
|---:|---|---|---|---|
| 1 | Q4_K F32 è limitato da instruction/unpack o dipendenze, non occupancy | `ncu` SOL, memory workload, source counters; SASS/resource usage | una sola classe deve spiegare >50% degli stall/throughput | probe matvec + logits decode + tok/s |
| 2 | Il mapping lane→byte Q4_K amplifica transazioni dei pesi | transazioni/byte utili e coalescing per warp | layout cooperativo riduce settori L1/L2/DRAM senza quantizzare x | stessa aritmetica F32; ≥3% micro e end-to-end |
| 3 | Una geometria che elabora più righe cooperativamente riusa metadata/activation meglio di warp8 | opcode/load count e cache hit; non occupancy nominale | meno istruzioni/load per output con REG sotto il limite per 48 warp | reject se spill o <1% |
| 4 | Il ramo GDN ha un hotspot diverso dalle proiezioni | timer separati qkv/z/alpha/beta/state/out e filtro kernel | ottimizzare solo state se supera 20% dei 18,92 ms | oracle stato/heads multi-step |
| 5 | Il merge split-K o 32 partizioni non è più ottimale dopo il nuovo residuo | profilo partial/merge separato a 8K/12K/16K | al massimo pochi punti percentuali; non può da solo raggiungere 20 tok/s | suite slow, stessa qualità |

## Idee sospese finché manca evidenza

- `cp.async` sui pesi Q4_K: ogni peso GEMV è usato una volta; senza tile
  condiviso il costo di shared/barrier può superare il beneficio.
- attivazione in shared: riduce un working set di 20–68 KiB già cacheabile, non
  i pesi unici; microbenchmark soltanto dopo metriche cache.
- CUDA Graph: già respinto sul percorso F32; riaprire solo con launch gap `nsys`.
- `--maxrregcount`: il kernel Q4 raggiunge già il limite warp teorico e non
  spilla; forzare meno registri rischia local memory/recompute.
- TF32/FP16/Q8: hanno già fallito gate qualitativi in configurazioni storiche;
  non sono scorciatoie ammesse per il target “nessuna regressione”.

## Nuova intuizione concreta

Il prossimo passo non è cambiare il numero di warp. Serve rendere osservabile
il rapporto fra byte utili Q4_K e istruzioni di unpack per riga. Senza `ncu`, il
primo incremento sicuro dell'harness è salvare automaticamente resource usage
e un istogramma SASS del simbolo Q4_K; quando `ncu` sarà disponibile, aggiungere
settori global/L1/L2, issue slot, eligible warp e stall scoreboard allo stesso
record. Solo allora scegliere fra layout cooperativo, predecode metadata o
pipeline shared.

## Criterio verso 20 tok/s

14,29 tok/s corrisponde a circa 70 ms/token fuori profiler; 20 tok/s richiede
meno di 50 ms. Eliminare interamente il core full-attention profilato non basta:
la prossima soluzione deve ridurre anche il percorso pesi comune. Una candidate
che migliora soltanto il core KV ma lascia FFN/GDN invariati non può soddisfare
la previsione e va classificata come ottimizzazione locale, non come soluzione
del target.
