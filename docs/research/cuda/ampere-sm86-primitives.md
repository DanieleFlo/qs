# Primitive Ampere `sm_86` e trasferibilità a DS4

## Limiti della RTX 3090 rilevanti

La guida Ampere NVIDIA distingue `sm_80` da `sm_86`. Per compute capability
8.6: massimo 48 warp concorrenti/SM, 64K registri a 32 bit, 255 registri per
thread, 16 blocchi/SM, 100 KiB shared/SM e fino a 99 KiB shared per blocco.
Usare i valori `sm_80` di A100 (64 warp, più shared) produce previsioni errate.

Fonti: [registro NVIDIA](sources.md).

## Coalescing e layout

Da compute capability 6.0 gli accessi di un warp vengono serviti dal numero di
transazioni da 32 byte necessario a coprire gli indirizzi richiesti. Per DS4
questo implica:

- misurare le transazioni reali dei byte Q4_K/Q5_K, non soltanto la dimensione
  nominale dei pesi;
- evitare che le lane leggano blocchi di righe lontane quando una mappatura
  cooperativa può leggere segmenti contigui;
- distinguere broadcast delle attivazioni, spesso servito da cache, da accessi
  ai pesi che dominano i byte unici;
- verificare allineamento e tail per le shape 5.120, 6.144 e 17.408.

Shared memory può riordinare load coalesced, ma bank conflict serializzano la
richiesta. Ogni proposta shared deve prevedere byte caricati una volta, bank
mapping, barrier e occupancy risultante.

## Shuffle e riduzioni warp

`shfl.sync` scambia registri fra lane del warp e sincronizza il member mask.
È utile per somme/riduzioni senza shared, come il kernel attention warp e il
matvec warp8. Vincoli:

- il mask deve rappresentare le lane realmente partecipanti;
- una riduzione warp non aumenta il numero di CTA disponibili;
- una catena di cinque shuffle resta una dipendenza seriale;
- `redux.sync` Ampere accelera riduzioni intere 32-bit, non la somma F32 usata
  dal percorso di qualità Q4_K F32.

Quindi sostituire automaticamente una riduzione F32 con primitive intere
cambierebbe politica numerica o richiederebbe una nuova rappresentazione: non è
una micro-ottimizzazione gratuita.

## Async copy e pipeline

Ampere accelera copie global→shared asincrone, evita il registro intermedio e
può bypassare L1. È trasferibile quando un tile globale viene riusato da più
operazioni prima di essere espulso. Non è automaticamente utile per GEMV batch
1 se ogni peso viene consumato una sola volta.

Checklist prima di `cp.async`/`cuda::pipeline`:

1. identificare un tile riusato (per esempio attivazioni condivise da più righe
   o un tile K/V usato da più query);
2. dimostrare con metriche che load latency e non unpack/FP32 domina;
3. calcolare shared per CTA e blocchi residenti su 100 KiB/SM;
4. costruire double buffer con stage espliciti;
5. mantenere la stessa aritmetica e confrontare i logits;
6. misurare fuori profiler.

Nel Q4_K warp8 corrente le otto warp condividono la stessa attivazione, che è
piccola e cacheabile, mentre ogni riga legge pesi diversi. Copiare l'attivazione
in shared può ridurre richieste cache ma non i ~14,76 GiB di pesi per token;
va quindi trattato come microbenchmark a beneficio previsto modesto.

## L1/shared carveout e cache operator

Ampere unifica L1, texture e shared con carveout configurabile. Aumentare shared
può ridurre cache L1; forzare `.cg`, `.ca` o bypass non è neutro. Prima di
cambiare operatori di cache servono hit rate L1/L2 e transazioni `ncu` sulla
shape esatta. Il SASS, non il solo PTX, conferma l'operatore finale.

## CUDA Graph

I graph ammortizzano overhead host quando una topologia viene rilanciata molte
volte. Non accelerano un kernel dominante né eliminano dipendenze dati. DS4 ha
già misurato 17,74 contro 17,92 tok/s sul percorso F32 corto: REJECT per quella
shape. Riaprire l'ipotesi solo se `nsys` mostra gap di launch materiali nel
nuovo percorso e la topologia per context/shape è stabile.

## Occupancy come vincolo, non obiettivo

Calcolo minimo per un kernel con `T` thread e `R` registri/thread:

```text
blocchi_per_SM <= min(1536 / T, 65536 / (T * R), 16, limite_shared)
warp_per_SM = blocchi_per_SM * T / 32 <= 48
```

Arrotondamenti di allocazione possono abbassare il risultato; confermare con
occupancy API o `ncu`. Dopo aver raggiunto warp sufficienti, più occupancy può
non migliorare bandwidth o una catena dipendente.
