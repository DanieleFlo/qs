# Indice: cuda

Indice generato; non modificare a mano.

## [Primitive Ampere `sm_86` e trasferibilità a DS4](ampere-sm86-primitives.md)

La guida Ampere NVIDIA distingue `sm_80` da `sm_86`.

- righe 3–11: **Limiti della RTX 3090 rilevanti** — La guida Ampere NVIDIA distingue `sm_80` da `sm_86`.
- righe 12–29: **Coalescing e layout** — Da compute capability 6.0 gli accessi di un warp vengono serviti dal numero di
- righe 30–45: **Shuffle e riduzioni warp** — `shfl.sync` scambia registri fra lane del warp e sincronizza il member mask.
- righe 46–67: **Async copy e pipeline** — Ampere accelera copie global→shared asincrone, evita il registro intermedio e
- righe 68–74: **L1/shared carveout e cache operator** — Ampere unifica L1, texture e shared con carveout configurabile.
- righe 75–82: **CUDA Graph** — I graph ammortizzano overhead host quando una topologia viene rilanciata molte
- righe 83–94: **Occupancy come vincolo, non obiettivo** — Calcolo minimo per un kernel con `T` thread e `R` registri/thread:

## [Mappa CUDA → prossimi esperimenti DS4](ds4-hypothesis-map.md)

A 10.666 token split-K32 porta il decode da 3,51 a 14,29 tok/s con 16/16 token

- righe 3–16: **Stato misurato da cui partire** — A 10.666 token split-K32 porta il decode da 3,51 a 14,29 tok/s con 16/16 token
- righe 17–28: **Classificazione aggiornata** — Il problema iniziale era parallelismo insufficiente del core full-attention:
- righe 29–38: **Esperimenti ordinati** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 39–50: **Idee sospese finché manca evidenza** — condiviso il costo di shared/barrier può superare il beneficio.
- righe 51–60: **Nuova intuizione concreta** — Il prossimo passo non è cambiare il numero di warp.
- righe 61–68: **Criterio verso 20 tok/s** — 14,29 tok/s corrisponde a circa 70 ms/token fuori profiler; 20 tok/s richiede

## [Comandi CUDA per misurare prima di ottimizzare](profiling-commands.md)

1.

- righe 3–14: **Ordine degli strumenti** — 1.
- righe 15–25: **Timing con eventi** — Gli eventi registrano timestamp nella stream e sono appropriati per kernel o
- righe 26–53: **Nsight Systems: timeline prima delle metriche** — Verificare installazione:
- righe 54–79: **Nsight Compute: un hotspot, poche sezioni** — Prima elencare ciò che la versione locale supporta:
- righe 80–90: **Replay, cache e clock** — `ncu` può richiedere più replay per raccogliere le metriche.
- righe 91–101: **Albero di classificazione** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 102–106: **Regola di memoria** — Salvare report grezzo, comando, versione tool, filtro del kernel, invocation,

## [Fonti primarie CUDA](sources.md)

Registro delle fonti NVIDIA usate dalle schede CUDA.

- righe 3–9: **Scopo e versione** — Registro delle fonti NVIDIA usate dalle schede CUDA.
- righe 10–23: **Documentazione ufficiale** — Sezione strutturale; consultare il contenuto locale indicato.
- righe 24–28: **Gerarchia dell'evidenza** — Queste fonti sono Tier A per semantica e strumenti.

## [Toolchain e ispezione binaria CUDA](toolchain-binary-inspection.md)

Prima di cambiare tile, numero di warp o shared memory bisogna sapere cosa ha

- righe 3–11: **Domanda operativa** — Prima di cambiare tile, numero di warp o shared memory bisogna sapere cosa ha
- righe 12–25: **Inventario locale 2026-08-11** — L'assenza dei tre ultimi strumenti è una capacità mancante, non una ragione per
- righe 26–71: **Comandi canonici** — Mostrare versioni e opzioni realmente disponibili:
- righe 72–84: **Lettura corretta delle risorse** — i 65.536 registri/SM; l'allocazione avviene a granularità hardware.
- righe 85–101: **Evidenza DS4 già raccolta** — `cuobjdump --dump-resource-usage ./ds4-bench` sul binario corrente riporta:
- righe 102–113: **Output da memorizzare per esperimento** — toolkit e driver:
