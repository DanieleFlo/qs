# Hardware e runtime di riferimento

Questa directory raccoglie la conoscenza hardware necessaria per diagnosticare
correttezza e prestazioni senza affidarsi alla memoria della sessione.

- `host-inventory.md`: fotografia riproducibile della macchina effettiva.
- `rtx-3090.md`: architettura GA102/compute capability 8.6 e conseguenze per i
  kernel di inferenza.
- `cuda-numerics.md`: precisione, riduzioni, TF32, fast-math e determinismo.

Le specifiche del produttore sono distinte dalle osservazioni locali. Una
lettura di clock o PCIe mentre la GPU e' in idle non va interpretata come valore
sotto carico. Prima di ogni report definitivo si aggiorna l'inventario con
`nvidia-smi -q` e si registra l'ambiente insieme agli artefatti numerici.
