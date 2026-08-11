# Contratto operativo per kernel engineering

Questo documento è il gate operativo del performance harness. Va letto insieme
ad AGENT.md, PROJECT_INDEX.md e todo.md prima di modificare kernel CUDA.

## Obiettivo

Una modifica è utile soltanto se migliora il workload dichiarato di DS4 senza
rompere la correttezza. Occupancy, utilizzo SM, TFLOP e bandwidth sono evidenze
diagnostiche, non obiettivi finali. Le metriche finali sono TTFT, prefill
token/s, decode token/s e ms/token, picco VRAM, stabilità numerica e correttezza
del modello.

Prefill e decode sono workload distinti. Ogni risultato deve nominare almeno
fase, batch, context length, shape/dtype/quantizzazione rilevanti, artefatto del
modello, hardware e build. «Il kernel è più veloce» senza queste coordinate non
è un risultato riproducibile.

## Ciclo obbligatorio

1. Congelare una baseline e il relativo experiment.json.
2. Profilare e formulare una diagnosi basata su più metriche.
3. Cercare in `docs/research/INDEX.md` chi ha già affrontato lo stesso problema;
   distinguere fonte primaria, fork sperimentale e benchmark non riprodotto.
4. Scrivere un'ipotesi falsificabile, la fonte che la motiva, le differenze di
   shape/formato/hardware e la metrica finale attesa.
5. Modificare una variabile attribuibile per esperimento.
6. Eseguire prima la correctness della primitiva e poi i gate numerici del modello.
7. Eseguire microbenchmark e benchmark end-to-end fuori dal profiler.
8. Confrontare baseline e candidate; emettere KEEP, REJECT o NEED_MORE_DATA.
9. Conservare risultato, fonti e spiegazione anche quando l'esperimento fallisce.

tools/perf_harness.py run rifiuta esperimenti senza ID univoco implicito o
esplicito, ipotesi, workload e metrica target. Non sovrascrive directory di
esperimenti esistenti. Il rollback del codice usa Git; gli artefatti in
performance-results/ sono dati locali e non devono essere promossi a golden
automaticamente.

## Significatività minima

Il default è cinque ripetizioni dopo un warm-up per workload. Il report include
minimo, p10, mediana, media, p90, massimo, deviazione standard e coefficiente di
variazione. Un coefficiente sopra il 5% marca il risultato instabile: differenze
minori o comparabili al rumore richiedono altri campioni (NEED_MORE_DATA). Per
una decisione automatica futura andranno usati intervalli appaiati sulla stessa
macchina e nello stesso stato termico; non si confrontano run sotto Nsight con
run liberi.

Durante lo sviluppo si usa prima la suite direction: due workload, due campioni
e nessun processo di warm-up separato. Serve soltanto a scartare rapidamente una
direzione sbagliata o scegliere quale variante approfondire. Non può promuovere
una candidate. KEEP richiede la suite slow, warm-up esplicito, almeno cinque
ripetizioni, correctness PASS e confronto sulla stessa macchina.

## Correttezza

Il harness non sostituisce gli scorer Qwen già presenti in
gguf-tools/quality-testing/. Una candidate non è accettabile finché i test
della primitiva, i gate full-vocabulary/teacher-forced pertinenti e almeno un
controllo end-to-end non sono verdi. Test saltati sono NOT_VERIFIED, mai PASS.
Ogni baseline performance conserva inoltre i logits alle frontiere canoniche;
la candidate li confronta automaticamente prima del verdetto. Argmax diverso,
overlap top-20 sotto 0,95, cosine sotto 0,999 o valori non finiti bloccano la
candidate. Questi gate rapidi non sostituiscono la suite full-vocabulary lenta.

## Misurazione hardware

I tempi finali vengono dal percorso sincronizzato di ds4-bench. Nsight Systems
serve a trovare gap, copie, sincronizzazioni e hotspot; Nsight Compute serve a
spiegare traffico, scheduler, stalls e risorse. I suoi tempi sotto replay non
sono usati come benchmark finale. Bandwidth utile e fisica vanno distinte:
minimum_required_bytes / time misura il lavoro utile, mentre
actual_dram_bytes / time misura il traffico osservato.
