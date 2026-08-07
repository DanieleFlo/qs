# Validazione checkpoint agentici Qwen3.6 — 2026-08-07

## Esito

`PASS` / `PROMOTE` per i task 10 e 11 sul percorso CUDA singola GPU.

- GPU: NVIDIA RTX 3090, `sm_86`, 24 GiB, driver 610.62.
- Target: `gguf/Qwen3.6-27B-Q4_K_S.gguf`.
- SHA-256 target:
  `ff857ba9f2184d8be408e8cabda12c89ba5adb202fddc1a88b3774d7bb232aca`.
- Sidecar MTP: `gguf/mtp-Qwen3.6-27B-Q4_0.gguf`, uno stadio, draft 4.
- Backend: CUDA, modello target interamente residente.

Il checkpoint v3 include logits target, stato ricorrente Gated DeltaNet,
timeline/posizione, fingerprint di modello e sessione e, quando presente, la
frontiera MTP completa: pending hidden, logits draft, token draft, validità e
lunghezza della cache MTP. Checksum, dimensione e fingerprint vengono validati
prima di qualsiasi scrittura sullo stato GPU vivo.

## Comandi riproducibili

```bash
make tests/test_agentic_checkpoint ds4-server ds4_test

DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q4_K_S.gguf \
DS4_AGENTIC_TEST_DIR=tests \
./tests/test_agentic_checkpoint

DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q4_K_S.gguf \
DS4_TEST_MTP=gguf/mtp-Qwen3.6-27B-Q4_0.gguf \
DS4_AGENTIC_TEST_DIR=tests \
./tests/test_agentic_checkpoint

./ds4_test --server

DS4_AGENTIC_BASE_URL=http://127.0.0.1:18081 \
DS4_AGENTIC_MODEL=deepseek-v4-flash \
DS4_AGENTIC_TEST_CHECKPOINT_ROOT=tests/agentic-live-kv \
DS4_AGENTIC_MODEL_REPORT=tests/agentic-checkpoint.out.log \
python3 -m unittest tests.test_agentic_api -v
```

`tests/run_agentic_checkpoint.sh` esegue le due varianti model-backed e salva
un report JSON line-oriented con suffisso `-mtp` per il sidecar.

## Gate lungo Q4_K_S target-only

Il caso usa una frontiera parent da 10.000 token, seguita dalla SkillCall.

| Misura | Risultato |
|---|---:|
| Parent prefill | 10.000 token, 16.027,961 ms |
| Skill instructions | 500 token, 8.507,308 ms |
| Child scartato | 2.000 token |
| SkillResult prefill | 200 token, 3.535,905 ms |
| Checkpoint | 159.852.624 byte |
| GPU → RAM staging | 183,180 ms |
| Scrittura payload | 346,341 ms |
| Lettura SSD → RAM | 464,518 ms |
| Restore RAM → GPU | 56,407 ms |
| Logits confrontati | 248.320 float32, bit-exact |

Il costo al return è quindi il restore fisso più il prefill dei 200 token del
risultato. Il parent da 10k e il child da 2k non vengono riprefillati. La
sessione canonica di confronto è nuova, ricostruisce la stessa frontiera parent
e applica lo stesso SkillResult: timeline, top-1 e tutti i 248.320 logits sono
bit-exact.

## Gate MTP

Il gate MTP usa una frontiera corta mirata, perché il sidecar aggiorna la propria
frontiera sequenzialmente e non serve ripetere su 10k il gate prestazionale del
target. Il target resta lo stesso Q4_K_S.

| Misura | Risultato |
|---|---:|
| Parent / instructions / child / result | 256 / 64 / 128 / 32 token |
| Checkpoint MTP | 160.866.384 byte |
| GPU → RAM staging | 177,348 ms |
| Scrittura payload | 311,122 ms |
| Lettura SSD → RAM | 468,610 ms |
| Restore RAM → GPU | 58,229 ms |
| Picco VRAM osservato | 22.073 MiB |
| Logits target dopo return | 248.320 float32, bit-exact |

Il confronto copre anche tre frame annidate, nomi ricorsivi, rollback MTP e
frontiera a un token dal limite di contesto.

## Robustezza e lifetime

Il gate model-backed verifica che i seguenti errori non modifichino token o
logits della sessione viva:

- checkpoint di un'altra sessione;
- checksum corrotto;
- file troncato;
- cancellazione prima dello staging o prima del restore GPU;
- ID/file mancante attraverso l'API.

Tre livelli annidati vengono ripristinati in ordine inverso e confrontati ogni
volta con una sessione canonica fresca. Il caso context-boundary termina a 511
token in un contesto da 512, preservando il token richiesto per la generazione.

La suite HTTP contiene 15 test e passa integralmente in 152,270 secondi. Copre
inoltre richieste senza `agentic`, liste vuote, nomi con prefisso comune,
registry invalido, cambio delle capability sulla KV viva, tool senza file,
skill annidate e ricorsive, ID ignoti/consumati, return del parent con child
irraggiungibile, stop a metà call e file SSD mancanti/corrotti/troncati.

I log dimostrano:

- `skill_checkpoint_live_bytes` cresce da 159.852.624 a 319.705.248 byte con
  due skill attive e torna a zero durante i return;
- `skill_checkpoint_count` e `skill_return_count` sono contatori storici,
  separati dallo spazio vivo;
- un tool normale non crea `SKILL_CHECKPOINT`;
- i return osservati prefillingano 13–15 token di risultato e non il parent;
- al riavvio sono stati rimossi due checkpoint orfani per 319.705.248 byte;
- su DrvFS la cancellazione usa retry limitato e segnala esplicitamente un
  eventuale file non eliminato.

I test unitari C (`ds4_test --server`) restano verdi e includono parser,
registry, filtro token-aware e pulizia sicura degli orfani.
