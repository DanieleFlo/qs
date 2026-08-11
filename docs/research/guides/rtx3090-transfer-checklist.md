# Checklist di trasferimento RTX 3090

## Compatibilità hardware

Confermare percorso sm_86, niente FP8 nativo, shared memory opt-in entro 99 KiB, registri e massimo 1.536 thread/SM. Una tecnica Hopper con TMA/WGMMA non è una candidate primaria.

## Compatibilità dei dati

Verificare shape Qwen3.6, Q4_K_S/Q5_K/Q6_K, group size, scale, layout GGUF e dtype delle attivazioni. AWQ/GPTQ/EXL3 non sono alias di K-quant.

## Compatibilità numerica

Definire ordine di accumulo, fast math, conversione Q8, tolleranza e oracle. Confrontare output completo della primitiva e logits di frontiera.

## Modello prestazionale

Calcolare FLOP, minimum bytes, arithmetic intensity e limite roofline; prevedere anche registri, shared memory e numero di CTA. Dichiarare quale metrica end-to-end deve muoversi.

## Esperimento isolato

Una sola tecnica, 2–3 shape reali, warm-up residente, campioni grezzi e baseline nello stesso processo quando possibile. Sotto profiler non si prende il tempo finale.

## Promozione

Prima PASS numerico, poi microbenchmark, suite direction e infine slow. Regressioni dominanti o guadagni inferiori al rumore producono REJECT/NEED_MORE_DATA.
