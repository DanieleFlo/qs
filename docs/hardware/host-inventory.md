# Inventario della macchina di validazione

Rilevazione locale: 2026-08-05, Europa/Roma. I valori dinamici sono una
fotografia, non una specifica permanente.

## Componenti osservate

| Componente | Valore osservato | Origine |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 3090, Ampere, compute capability 8.6 | `nvidia-smi -q` |
| Device PCI | `10DE:2204`, bus `0000:01:00.0`, subsystem `1043:87AF` | `nvidia-smi -q` |
| VRAM | 24.576 MiB totali; ECC non disponibile/esposto | `nvidia-smi -q` |
| Driver | KMD 610.62, modello WDDM | `nvidia-smi -q` |
| Runtime esposto dal driver | CUDA UMD 13.3 | `nvidia-smi -q` |
| VBIOS | `94.02.42.00.a9` | `nvidia-smi -q` |
| Limite potenza | 390 W corrente/default; intervallo esposto 100-480 W | `nvidia-smi -q` |
| Clock massimi esposti | SM 2130 MHz, memoria 9751 MHz | `nvidia-smi -q` |
| PCIe supportato | device Gen4 x16; host dichiara fino a Gen5 | `nvidia-smi -q` |
| CPU | Intel Core i9-14900K | registro hardware Windows |
| Sistema operativo | Windows 10 Pro; inferenza/toolchain CUDA eseguite anche via WSL | ambiente locale e artefatti di staging |

Al momento della fotografia la scheda era anche display-attached, in P8, con
link PCIe Gen1 x2 e circa 1,5 GiB usati da processi grafici. Sono valori di idle
e power management: durante un benchmark vanno ricampionati sotto carico prima
di diagnosticare un collo di bottiglia PCIe. Il modello WDDM e l'uso grafico
condiviso possono aumentare rumore di latenza e memoria disponibile, ma non
spiegano da soli un drift numerico deterministico.

## Implicazioni per Qwen3.6 27B Q4_K_M

- Il tensor span osservato dal progetto e' circa 17,77 GiB e l'arena CUDA circa
  18,50 GiB. Il margine rispetto ai 24 GiB fisici e' ridotto da memoria
  riservata, desktop e workspace; ogni run deve registrare VRAM libera.
- La build deve targettizzare esplicitamente `sm_86`. Il Makefile del progetto
  richiede `CUDA_ARCH=sm_86` per il gate riproducibile.
- La scheda non offre un contatore ECC utilizzabile tramite `nvidia-smi`; la
  ripetibilita' e i checksum sono quindi parte essenziale del controllo.
- Clock, temperatura, power cap e processi grafici vanno registrati per le
  prestazioni. Una divergenza bit-identica su piu' run punta prima al software
  o all'aritmetica, non a instabilita' casuale dell'hardware.

## Snapshot consigliato per ogni run

Conservare in un file di testo accanto al report:

```powershell
nvidia-smi -q
nvidia-smi --query-gpu=name,uuid,driver_version,vbios_version,pci.bus_id,memory.total,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu,utilization.gpu,compute_cap --format=csv
git rev-parse HEAD
git diff --stat
```

Nel manifest del test aggiungere comando completo, hash SHA-256 del GGUF,
toolkit/nvcc effettivo dentro WSL, build flags, variabili `DS4_CUDA_*`, backend
llama.cpp e numero di stream.

## Limiti dell'inventario

Modello esatto della scheda partner, RAM e motherboard non sono ancora stati
acquisiti con un comando non privilegiato affidabile. Non vanno dedotti dal
solo subsystem ID. Questi dati sono secondari per il drift, ma utili per una
baseline prestazionale completa.
