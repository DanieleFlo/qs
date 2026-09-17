# Qualificazione numerica Qwen3.8 — 17 settembre 2026

Seguito dell'[audit dell'inferenza](qwen-inference-audit-2026-09-09.md).
Questa sessione affronta i due controlli rimasti aperti: oracolo completo
same-GGUF e precisione FP16 del prefill già presente. Non misura nuovi
guadagni prestazionali e non modifica soglie o golden.

## Ambiente e metodo

RTX 3090 24 GB, WSL Ubuntu 24.04, CUDA 12.4, sm_86, driver 610.62.
Modello Qwen3.8-27B-UD-Q4_K_S, SHA-256 verificato:
`75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017`.
MTP disabilitato; contesto 24576, 32 posizioni greedy e 32 teacher-forced,
248320 logits F32 per posizione. DS4 usa prefill chunk 2048; llama.cpp usa
batch 2048 e microbatch 512. Non si confrontano i tempi di questi run.

llama.cpp è compilato dalla revisione fissata nel manifest,
`2468576f241235452013308597e6de1b78866996`, in un worktree pulito dedicato,
con CUDA, Release e GGML_NATIVE. Lo scorer DS4 include la correzione
`6984390`: una doppia dichiarazione di `quality` impediva la compilazione;
rimossi anche il ramo duplicato di parsing e nessun comportamento utile.
La successiva integrazione `a1808b8` preserva le modifiche locali al loader
già pubblicate su origin; il confronto normalizzato degli otto file salvati
prima del merge conferma che sono identici, salvo il titolo nuovo del README.

Il corpus condiviso contiene soltanto input: non vengono riutilizzati output
o logits Qwen3.6. Tokenizer e template sono quelli Qwen3.8 alla revisione
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, verificati contro il manifest.
Le continuazioni teacher-forced sono i 32 token greedy di un bootstrap DS4,
fissati una volta e identici per tutti i confronti; non sono una golden esterna.

| Caso | Token del prompt |
|---|---:|
| single_token_ascii | 13 |
| short_fact_english | 24 |
| unicode_multilingual | 44 |
| system_thinking_off | 39 |
| system_thinking_on | 75 |
| multi_turn_preserve_thinking | 97 |
| code_completion_c | 37 |
| tool_call_weather | 283 |
| long_canary_4096 | 21338 |

Il caso lungo viene eseguito integralmente, senza troncamento nonostante
il nome storico. I run sono residenti e sequenziali, senza processi modello
concorrenti. Ogni inventario viene verificato con `verify_run`, con checksum
dei logits completi. Il rendering nativo degli scorer resta `tokenizer_only`:
queste prove verificano la tokenizzazione di byte canonici, non certificano
l'intero percorso di rendering chat di ciascun engine.

Artefatti locali: `performance-results/qwen38-qualification-20260917/`.
`prepare.py` prepara gli input; `run.py NOME` esegue e inventaria ogni run;
`compare.py LEFT RIGHT MODE [SUITE]` invoca il comparatore del repository.
I comandi esatti, le variabili diagnostiche e gli hash dei binari sono nei
file `*.command.json`; `model.sha256` e `tokenizer-checksums.json` identificano
gli input. I risultati diagnostici non possono diventare PASS per il solo
fatto che i processi siano terminati correttamente.

## Oracolo completo: esito FAIL, nessuna promozione

DS4 default e llama.cpp completano tutti i nove casi, con inventari validi
e nessun logit non finito nelle 576 posizioni confrontate. Le sequenze
greedy coincidono per tutti i 32 token in sei casi, compreso il caso lungo.
I prefissi identici degli altri tre sono 24 token per `short_fact_english`,
20 per `system_thinking_off` e 30 per `tool_call_weather`.
Il teacher-forcing, che conserva gli stessi input dopo la divergenza greedy,
mostra 285/288 argmax uguali e MAE delle log-probabilità 0,00770125.
Questi dati non superano il requisito di identità e non giustificano
la promozione dell'oracolo né l'allargamento delle tolleranze.

Nel caso Unicode i 44 token canonici includono la normalizzazione NFC del
tokenizer ufficiale; entrambi i tokenizer nativi ne producono 45 sui byte
originali, per la sequenza `e` + accento combinante. Il confronto numerico
usa comunque i medesimi 44 token canonici in entrambi gli engine.
L'accordo tra tokenizer nativi non dimostra quindi l'accordo con l'upstream.
Il comparatore ora espone `left_matches_canonical` e
`right_matches_canonical` e impedisce PASS se un rendering dichiarato
verificato differisce dall'input canonico, anche quando entrambi gli engine
commettono la stessa differenza. Il caso `tokenizer_only` resta NOT_VERIFIED.
La regressione dedicata e gli altri test del comparatore passano: 21/21.

Il manifest resta NOT_VERIFIED. La ripetizione indipendente dell'oracolo,
la verifica del rendering nativo e la localizzazione delle divergenze restano
da completare; non sono errori corretti dal solo aggiornamento del comparatore.
Report: `llama-1-vs-default-all.json` e relativo riepilogo.
