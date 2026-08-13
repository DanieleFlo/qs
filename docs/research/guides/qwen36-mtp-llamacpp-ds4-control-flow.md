# Qwen3.6 MTP: workflow llama.cpp, confronto DS4 e audit CUDA

## Ruolo e perimetro

Questa scheda descrive il percorso MTP realmente eseguito per Qwen3.6, dal
carry dell'hidden target fino al rollback dopo la verifica. L'audit upstream e'
fissato a llama.cpp commit
`1a064ab0921238c1daa397d6f4a900ef33884de2` (2026-07-22). Le misure DS4 usano
una RTX 3090 24 GiB (`sm_86`), target `Qwen3.6-27B-Q4_K_S.gguf` e sidecar
`mtp-Qwen3.6-27B-Q4_0.gguf`.

L'obiettivo e' depth 2. DS4 usa depth 2 di default e applica un hard cap 4;
questo lavoro non valuta ne' abilita profondita' superiori a 4.

## Fonti primarie

- [llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673):
  integrazione MTP, context target/draft separati, hidden NextN e rollback
  ricorrente.
- [`common/speculative.cpp` al commit auditato](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/common/speculative.cpp):
  `common_speculative_impl_draft_mtp::{process,draft,accept}`.
- [`server-context.cpp` al commit auditato](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/tools/server/server-context.cpp):
  costruzione del batch target, sampling, accept e trim.
- [`common/sampling.cpp` al commit auditato](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/common/sampling.cpp):
  `common_sampler_sample_and_accept_n`, cioe' sampling autorevole di ogni riga
  target, confronto col draft e bonus sample dopo full acceptance.
- [`qwen35.cpp` al commit auditato](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/src/models/qwen35.cpp):
  trunk ibrido, `h_nextn` e grafo del blocco MTP.
- [`llama-memory-recurrent.cpp` al commit auditato](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/src/llama-memory-recurrent.cpp):
  snapshot bounded `n_rs_seq` e rollback parziale.
- [`quantize.cu` al commit auditato](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/ggml/src/ggml-cuda/quantize.cu)
  e [`mmvq.cu`](https://github.com/ggml-org/llama.cpp/blob/1a064ab0921238c1daa397d6f4a900ef33884de2/ggml/src/ggml-cuda/mmvq.cu):
  geometria CUDA multi-riga usata come riferimento.
- [Documentazione speculative decoding llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md):
  opzioni e famiglie di drafter.

## Semantica NextN

Il trunk target espone `h_nextn` dopo `output_norm`. Per una posizione `p`, il
sidecar riceve la coppia spostata `(token[p], h_target[p-1])`:

```text
Embedding(token[p]) --enorm--\
                              concat -> eh_proj -> decoder full-attention
h_target[p-1] ------hnorm---/          -> FFN -> shared norm/head
                                                     |
                                                     +-> token[p+1]
```

Lo shift e' obbligatorio. L'hidden target di `token[p]` diventa il carry della
posizione successiva solo dopo che il target ha realmente valutato quel token.
llama.cpp chiama il tensore normalizzato `h_nextn`; DS4 usa lo stesso punto del
grafo, non il residuo pre-norm.

## Workflow llama.cpp

### Prefill e frontiera

Il context target conserva logits, KV e memoria ricorrente. `pending_h[seq]`
conserva l'ultimo `h_nextn`. Il context MTP ha KV propri. A ogni batch target,
`process()` sposta gli hidden di una riga e fa un solo `llama_decode()` batch
nel context MTP; il primo elemento usa il carry del batch precedente.

### Draft

Il token appena campionato non e' ancora nel target state. `draft()` esegue il
sidecar su `(id_last, pending_h)` e poi autoregressivamente sugli hidden MTP
prodotti, fino a `n_max` o alla soglia `p_min`.

### Verifica target

Il server costruisce un unico batch:

```text
[sampled, draft0, draft1, ...]
```

I logits della riga 0 verificano `draft0`, quelli della riga 1 verificano
`draft1`; i logits dell'ultima riga restano la nuova frontiera. Il numero di
righe target non diminuisce rispetto a `seed + drafts`: cambia la forma del
batch, che permette di riusare i pesi del target tra colonne.

### Accept e rollback

`process()` conserva tutti gli hidden target in `verify_h`. Dopo il confronto
dei token, `accept()` seleziona `verify_h[n_accepted]` come nuovo `pending_h`.
Le celle full-attention future vengono troncate o sovrascritte. Per i layer
ricorrenti `n_rs_seq` mantiene snapshot per-token: `seq_rm()` seleziona lo
snapshot corrispondente quando il rollback rientra nella finestra bounded;
solo rollback piu' lunghi richiedono il checkpoint generale.

### Pseudocodice

```text
sampled = target_sampler(target_logits)
drafts  = mtp_autoregressive(sampled, pending_h, depth)

rows = target_decode([sampled] + drafts)
mtp_process([sampled] + drafts, shifted(rows.h_nextn))

accepted, pending = sample_target_rows_and_match(rows.logits, drafts)
rollback_target_and_draft_suffix(accepted)
pending_h = rows.h_nextn[accepted]
emit(sampled + accepted drafts)
```

In greedy mode `target_sampler` and the row checks sono argmax. Con temperatura
positiva il sampler target e' clonato/checkpointato, consuma le righe verifier
in ordine e produce il mismatch o il bonus finale che apre il ciclo seguente.
Il sidecar continua a proporre il proprio top-1: non decide mai la distribuzione
dei token emessi.

## DS4 prima dell'audit

Il vecchio ciclo eseguiva `target_seed(1)` e poi `verify(drafts)`. Su partial
accept copiava circa 157 MiB di stato GDN, ripristinava lo stato pre-batch e
rigiocava il prefisso accettato. Il verifier 2--4 righe usava kernel F32 che
dequantizzavano i pesi per ogni token. Il catch-up sidecar calcolava anche Q,
attention e FFN pur dovendo soltanto materializzare K/V.

Questa forma era corretta, ma non poteva raggiungere il budget depth-2:
`target_seed` costava circa un decode intero e il replay aggiungeva un altro
decode a ogni partial accept.

Un secondo problema era nel frontend. CLI e server invocavano il ciclo MTP solo
con `temperature <= 0`, mentre il default di entrambi e' `temperature=1`. Di
conseguenza `--mtp` caricava e manteneva il sidecar, incluso il catch-up K/V a
ogni token, ma la generazione restava target-only: nessun draft veniva verificato
e il costo aggiuntivo poteva sembrare una regressione. Questo spiega la
differenza tra i benchmark greedy precedenti e l'avvio locale/server ordinario.

## DS4 finale

Con `--mtp`, il percorso predefinito e':

1. proporre due draft dal sidecar usando `first_token` e il carry target;
2. verificare `[first_token, draft0, draft1]` in un solo target batch V(3);
3. in greedy produrre top-1 per riga sul device; con temperatura positiva,
   applicare il sampler ai logits target completi di ogni riga in ordine;
4. salvare dentro i kernel GDN gli stati dopo riga 0 e riga 1; lo stato dopo
   l'ultima riga e' gia' quello live;
5. su reject ripristinare lo snapshot riga 0, su partial lo snapshot riga 1;
   non esiste replay del target nel percorso normale;
6. selezionare i logits della medesima riga accettata;
7. aggiornare il sidecar con un percorso K/V-only, saltando Q, attention,
   output attention, FFN e LM head;
8. committare `first_token` e i draft accettati;
9. con sampling, conservare il mismatch target o il bonus della riga finale
   come token pendente del ciclo successivo.

`DS4_MTP_SPLIT_TARGET=1` conserva il vecchio layout soltanto come diagnostica
greedy; il sampling forza il batch fused perche' necessita di tutte le righe.
Depth 2 e il batch fused sono i default; l'engine rifiuta silenziosamente
profondita' Qwen oltre 4 clampandole a 4.

### Sampling, masking e doppio controllo

La soluzione segue `common_sampler_sample_and_accept_n` del llama.cpp locale di
LM Studio: temperature, top-k, top-p e min-p sono applicati una volta sola alla
riga target autorevole. Il token ottenuto viene confrontato col top-1 MTP. Se
coincide il draft e' accettato; altrimenti quel token target viene portato al
ciclo successivo. Dopo full acceptance viene campionata allo stesso modo la
riga bonus finale.

Applicare gli stessi mask ai logits MTP non rimuove questo confronto. I filtri
standard conservano per costruzione il top-1 della distribuzione a cui sono
applicati, mentre il set ammesso dal target puo' essere diverso. Filtrare anche
il sidecar aggiungerebbe quindi lavoro ma il target dovrebbe comunque campionare
e verificare. I mask grammar/JSON/tool possono eliminare proposte impossibili,
ma sono stateful e devono avanzare dopo ogni token accettato; fino a quando tale
stato non e' checkpointabile dentro il batch, MTP stocastico viene escluso per
richieste tool e response-schema. Q8_1+R8 non e' stato modificato.

## Problemi trovati e ritorno a llama.cpp

| Problema DS4 | Verifica in llama.cpp | Decisione |
|---|---|---|
| Seed target separato | Upstream verifica `[sampled]+drafts` in un batch | Batch fused reso default |
| Replay dopo partial accept | Upstream conserva `n_rs_seq` frontiere ricorrenti | Snapshot GDN per riga e restore diretto; replay 0 ms |
| Catch-up sidecar calcolava l'intera rete | Upstream separa chiaramente `process()` dal drafting, ma il grafo generico esegue ancora piu' lavoro del necessario | DS4 usa K/V-only: ottimizzazione locale oltre upstream |
| Una quantizzazione CUDA per ogni riga verifier | `quantize_row_q8_1_cuda` copre tutte le righe in una sola griglia | Un solo launch su `blocks * n_tokens` |
| Kernel rows2 scriveva la seconda riga da lane 1 | `mmvq.cu` usa una riduzione il cui risultato e' disponibile ai lane writer; `warp_sum_f32` DS4 usa `shfl_down` e completa solo lane 0 | Entrambe le righe sono pubblicate da lane 0 |
| Head Q4_0 sidecar dequantizzato scalarmente | Upstream Q4_0 x Q8_1 usa dot integer/DP4A | Head Q4_0 Q8_1+R8 DP4A warp-8 |
| MTP bypassato con il default `temperature=1` | `sample_and_accept_n` campiona ogni riga target e confronta il draft | Stesso sampler DS4 per riga, mismatch/bonus pendente |
| Mancava telemetria posizionale | Upstream distingue draft/accept per ciclo | Contatori p1/p2, tempi seed/propose/snapshot/batch/rollback/replay/catch-up |
| Il CLI upstream sembrava bloccato dopo un token | Il frontend era rientrato nel prompt interattivo e stampava righe vuote; il modello e MTP non erano bloccati | Benchmark upstream eseguiti con `--single-turn` e timeout |

Il bug lane 1 e' il caso piu' importante. Il kernel sembrava veloce ma poteva
accettare token sulla base di una somma parziale. Disabilitare `rows2` rendeva
il testo corretto ma scendeva a 30.77 tok/s. Correggere il writer ha ripristinato
la sequenza greedy e la regressione raw-copy ora misura gap argmax 0.000.

## Audit dei kernel MTP

### Target Q4_K/Q5_K/Q6_K

Il verifier usa attivazioni Q8_1 piu' un secondo residuo Q8_1. Il residuo e'
necessario per la gate di qualita' DS4: la variante Q8_1 semplice era piu'
veloce, ma aveva gia' fallito il corpus completo (7/8 casi, 484/512 token) e
non e' stata mantenuta.

Per V(2)/V(3):

- una sola quantizzazione copre tutte le righe contigue;
- ogni CTA MMVQ mantiene tutte le colonne in registri e carica i pesi una sola
  volta per applicarli a 2 o 3 token;
- due output rows per CTA seguono la geometria generica upstream;
- quattro warp per CTA sono risultati piu' veloci sulla RTX 3090;
- Q4_K, Q5_K e Q6_K conservano due accumulatori separati per Q8_1 e residuo.

Una variante corretta a 8 warp, divisa in due gruppi indipendenti da 4 warp,
ha misurato 33.33 tok/s contro 37.12 tok/s della geometria 4-warp sul prompt
naturale ed e' stata scartata.

### Sidecar Q4_0

Il shared head usa Q4_0 x Q8_1+R8 con DP4A e otto output rows per CTA. Il
catch-up non esegue il shared head: normalizza K, applica RoPE e memorizza
K/V. La proposta continua invece a eseguire l'intero blocco, perche' hidden e
logits servono allo step autoregressivo successivo.

### Rollback GDN

Le 48 convolution state e recurrent state vengono snapshotate direttamente
durante V(3). Servono solo due frontiere a depth 2: dopo il token obbligatorio
e dopo il primo draft. La frontiera full-accept resta nello stato live. Le 16
cache full-attention sono position-addressed e non vengono copiate.

## Esperimenti scartati

- **Q8_1 senza residuo**: 57.85 tok/s in un run preliminare, ma gate di
  qualita' fallita; non presente nel percorso finale.
- **rows2 con lane 1 writer**: numeri fino a 50+ tok/s, invalidi per divergenza
  greedy; tutti i claim ottenuti con quel writer sono ignorati.
- **rows2 8-warp/CTA**: corretto ma 33.33 tok/s, piu' lento del 4-warp.
- **verifier one-row/CTA**: corretto, ma 30.77 tok/s e acceptance 70/111 sul
  prompt naturale.
- **adaptive widening**: rimosso. Depth 2 e' stabile; l'hard cap resta 4.

## Validazione finale

### Suite model-backed

Comando:

```sh
DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q4_K_S.gguf \
DS4_TEST_MTP=gguf/mtp-Qwen3.6-27B-Q4_0.gguf \
DS4_TEST_MTP_CTX=768 \
DS4_TEST_MTP_PREFILL_CHUNK=16 \
DS4_TEST_QWEN_MTP_PATHS=1 \
DS4_MTP_STATS=1 \
./ds4_test --mtp-verify-depth
```

Risultato:

```text
natural: cycles=80 accepted=152/159, committed=232, replay=0,
         net=47.358 tok/s, worst_argmax_gap=0.000
forced reject: accepted=0, max_chunk=1, gap=0.000
forced partial: accepted=15, max_chunk=2, replay=0, gap=0.000
raw copy: accepted=40/47, max_chunk=3, gap=0.000
sampled temp=1: 128/128 token uguali al target-only, max_chunk=3
prompt logits MTP on/off: 248320 float bit-exact, top1=71093
fallbacks: 0
```

### CLI Q4_K_S

Context 768, prefill chunk 16, 128 token, greedy:

| Workload | Target-only | `--mtp` | Speedup | Acceptance |
|---|---:|---:|---:|---:|
| Copia semplice ripetitiva | 30.48 tok/s | 46.39 tok/s | **1.52x** | 83/89 (93.3%) |
| Copia naturale | 32.39 tok/s | 36.20 tok/s | 1.12x | 70/111 (63.1%) |

I due output CLI sono stati confrontati sullo stesso prompt; la suite raw-copy
aggiunge la prova teacher-forced per ogni token committato. La differenza tra i
workload e' acceptance, non prefill o profondita'.

Con il default sampled, seed 123, `top_p=1` e `min_p=0.05`:

| Workload sampled | Target-only | `--mtp` | Speedup | Acceptance |
|---|---:|---:|---:|---:|
| Copia semplice ripetitiva | 31.64 tok/s | 48.74 tok/s | **1.54x** | 84/87 (96.6%) |

Gli output sono byte-identici. La suite model-backed separata confronta inoltre
128 token ID con seed `0x123456789abcdef0`, ottiene uguaglianza esatta e
`max_chunk=3`, escludendo che la parita' derivi da un fallback one-token.

### Confronto locale llama.cpp

Il binario CUDA upstream al commit auditato e' stato eseguito sulla stessa RTX
3090, con lo stesso target Q4_K_S, la stessa sidecar Q4_0, lo stesso file di
prompt, 128 token greedy e `--spec-draft-n-max 2`. Il diff degli output
target-only e MTP e' byte-identico eccetto la riga dei tempi:

| Runtime | Target-only | MTP depth 2 | Speedup |
|---|---:|---:|---:|
| llama.cpp `1a064ab09` | 40.9 tok/s | 72.8 tok/s | **1.78x** |
| DS4 finale | 30.48 tok/s | 46.39 tok/s | **1.52x** |

Sul valore assoluto DS4 e' quindi circa 25.5% sotto upstream target-only e
36.3% sotto upstream con MTP; detto nell'altra direzione, llama.cpp e' 1.34x e
1.57x piu' veloce. La perdita aggiuntiva MTP e' coerente con il costo ancora
visibile di proposta sidecar, snapshot GDN e catch-up. Il batch verifier DS4,
pero', non rilegge piu' i pesi per ogni colonna e non rigioca il target sui
partial accept.

Questa e' una misura end-to-end, non un microbenchmark di un singolo kernel:
llama.cpp applica il proprio frontend conversazionale mentre il comando DS4
canonico usa `--raw-prompt`, quindi il file sorgente e l'hardware coincidono ma
il token stream di prefill puo' differire. Il confronto di speedup interno a
ciascun runtime resta il dato piu' robusto.

### Server reale

Quattro richieste `/v1/chat/completions`, una warm-up e tre misurate, 180 token
di prompt chat e 128 token generati. Mediane delle tre richieste finali:

| Metrica | Target-only | `--mtp` | Speedup |
|---|---:|---:|---:|
| Generazione server | 31.47 tok/s | 51.07 tok/s | **1.62x** |
| HTTP end-to-end | 9.566 s | 7.905 s | 1.21x |

Le risposte hanno stessa lunghezza (567 caratteri), stesso prefisso e
`finish=length`. I contatori server MTP aggregati sono:

```text
cycles=172 proposed=340 accepted=340 full=172 partial=0 rejected=0
fallbacks=0 committed=512 replay_ms=0 net_tok_s=51.221
```

Il prefill resta circa 4.9 s in entrambi i modi e domina una parte del tempo
HTTP. Non e' necessario modificarlo per il target di throughput decode; un
progetto separato puo' affrontare il riuso della live/disk KV cache.

Una richiesta sampled separata (`temperature=1`, `top_p=1`, `min_p=0.05`, seed
123, 128 token) conferma il percorso usato dall'avvio normale:

| Metrica sampled | Target-only | `--mtp` | Speedup |
|---|---:|---:|---:|
| Generazione server | 34.55 tok/s | 50.15 tok/s | **1.45x** |
| Server-internal end-to-end | 4.526 s | 3.391 s | **1.34x** |
| HTTP osservato | 5.173 s | 3.964 s | **1.31x** |

La risposta MTP termina a 128 token con `finish=length`, accetta 82/88 draft e
registra zero fallback. Le richieste tool/schema restano sul percorso
conservativo nelle fasi a sampling stateful quando la temperatura e' positiva;
nelle fasi DSML rese greedy dal decoder possono usare MTP in sicurezza.

Il gate finale DSML a `--ctx 32768`, tools attivi, `temperature=0.7` e seed
fisso ha restituito la stessa `list_files` valida nei due modi. Il decode e'
passato da 25.95 a 37.75 tok/s (1.45x) e il tempo server-internal da 13.957 a
13.411 s, limitato dal prefill di 390 token. Sullo stesso server, una risposta
naturale sampled di 128 token e' passata da 34.76 a 41.97 tok/s (1.21x), con
lo stesso hash dell'output e zero fallback/replay.

### Correzione VRAM 24 GiB

Il vecchio report `15.02 GiB planned` era incompleto: trattava il KV Qwen come
la piccola cache generica compressa, mentre i 16 layer full-attention allocano
circa 4 GiB a context 32768; ometteva inoltre workspace Qwen, stato MTP e i
1.55 GiB della sidecar. Il report ora conta queste classi e dichiara
esplicitamente che cache dinamiche dell'acceleratore e driver restano escluse.

La riduzione persistente non cambia il prefill e non rimuove kernel veloci:

- snapshot rollback per-riga: 5 -> 2 frontiere, circa 454.5 MiB;
- logits verifier: 16 -> 3 righe, circa 12.3 MiB;
- workspace proiezioni recurrent/full-attention condiviso: 224 MiB;
- output SwiGLU in-place sul gate ormai morto: 272 MiB.

Il risparmio totale e' circa 963 MiB con un prefill esplicito da 4096 righe e
circa 715 MiB con il default Qwen preesistente da 2048; l'ottimizzazione non
cambia quel default. Dopo i test server finali `nvidia-smi` riportava 23,893
MiB usati e 434 MiB liberi, senza il paging patologico osservato prima della
correzione.

## Nota vLLM

L'assunzione che vLLM non implementi Qwen MTP non e' piu' valida. Il ramo
corrente contiene [`Qwen3_5MTP`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)
e la relativa [configurazione speculative](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py).
E' un riferimento semantico PyTorch/Hugging Face, non una baseline diretta per
GGUF Q4_K_S e i kernel K-quant DS4.

## Limiti dell'evidenza

Il workflow e l'audit CUDA llama.cpp sono fissati al commit indicato. I claim
prestazionali della PR upstream non sono stati riusati: la tabella locale viene
dal binario `1a064ab09` sulla RTX 3090 osservata. Il confronto piu' controllato
resta target-only DS4 contro lo stesso binario DS4 con il solo flag `--mtp`;
il confronto assoluto tra runtime include anche differenze di frontend,
scheduler, allocator e kernel non-MTP.
