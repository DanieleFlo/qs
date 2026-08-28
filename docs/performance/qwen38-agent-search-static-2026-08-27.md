# Qwen3.8 agentico: partizione statica DSML `SEARCH`

## Obiettivo e invarianti

Questo ciclo avvicina il decode del server agentico Qwen3.8 alla velocità del
server non agentico senza modificare kernel CUDA, API, formati DSML/JSON, KV su
SSD o semantica del sampling. Qwen3.8-27B-UD-Q4_K_S è il riferimento primario;
Qwen3.6 resta coperto come compatibilità.

La candidate riusa la partizione statica DSML già impiegata nelle stringhe
libere anche nello stato opzionale `SEARCH`, soltanto quando il tracker è
sincronizzato e la finestra trattenuta non contiene `<`. La base accetta solo
piece non vuoti, privi di `<` e non sensibili all'identità. Stop, controlli di
thinking e token potenzialmente strutturali restano nella frontiera dinamica.
`tool_choice=required`, marker parziali e stati strutturali continuano sul
percorso fail-closed completo.

## Baseline prima della modifica

Hardware: RTX 3090, build CUDA `sm_86`, contesto allocato 22.593, prompt
agentico realmente occupato 10.814 token, KV disk da 8 GiB. Il vero
`DeepAgent` `bootstrap-wiki` ha tool disponibili con `tool_choice=auto`, ma
deve produrre una storia normale tra 250 e 300 parole. Ogni configurazione usa
un warm-up e due campioni misurati; tutte le risposte contengono 432 token e 298
parole e terminano con `stop` in una sola richiesta modello.

| Configurazione | Decode mediano | CV | Masking mediano |
|---|---:|---:|---:|
| Target-only baseline | 11,739 tok/s | 0,95% | 17.293,25 ms |
| MTP caricato, baseline constrained target-only | 11,524 tok/s | 2,76% | 17.633,14 ms |

Nel primo campione target-only, 107.523.426 candidati sono stati filtrati in
17,21 s, contro 19,42 s di eval. Il masking CPU spiegava quindi quasi metà del
decode della storia, non una fluttuazione dell'inferenza GPU.

Artifact:

- `performance-results/q38-agent-plain-story-target-base-20260827-002/experiment.json`;
- `performance-results/q38-agent-plain-story-mtp-base-20260827/experiment.json`.

## Risultato appaiato

| Configurazione | Decode mediano | CV | Guadagno | Masking mediano |
|---|---:|---:|---:|---:|
| Target-only candidate | 23,457 tok/s | 1,65% | +99,82% | 136,34 ms |
| MTP caricato, candidate constrained target-only | 23,417 tok/s | 0,15% | +103,21% | 140,05 ms |

Il masking target-only scende del 99,21%. In entrambi i campioni target-only la
risposta ha lo stesso SHA-256 della baseline, i 433 passi entrano nella
partizione `SEARCH`, i fallback sono zero e la frontiera dinamica totale è
258.934 token. I checkpoint post-risposta sono ricostruiti da
`memory-request` in VRAM senza reload SSD nella finestra di pubblicazione.

Con il sidecar MTP caricato, tutti i 433 passi sono dichiarati dal nuovo
contatore `mtp_constrained_target_only_steps`: la grammatica continua ad
avanzare un token per volta e nessun guadagno viene attribuito allo speculative
decoding. Le curve non vincolate restano separate da questo risultato.

Artifact:

- `performance-results/q38-agent-plain-story-target-search-static-candidate-20260827/experiment.json`;
- `performance-results/q38-agent-plain-story-mtp-search-static-candidate-20260827/experiment.json`.

## Gate di correttezza

La storia completa è stata ripetuta con `compare_new_vs_oracle`: 433 confronti
full-vocabulary, zero divergenze e hash identico alla baseline. La suite
strutturale `direction` ha poi verificato DSML obbligatorio e JSON Schema
annidato con output deterministici, schema validi, prefix probe validi, output
uguali alla baseline e zero divergenze. Il verdict prestazionale della suite
rapida è `NEED_MORE_DATA` perché il confronto doppio è rumoroso; il gate di
correttezza è `PASS`.

Sono inoltre passati `ds4_server_test`, `ds4_test --constraint-trie`, 61 test
dell'harness e 16 test di compatibilità Qwen3.8. Agent SSD è passato nei gruppi
system, HDS e deep-skill in entrambe le direzioni di cambio modalità: cold
target-only → warm MTP e cold MTP → warm target-only. Metal e distributed non
sono verificati su questo host.

La direzione MTP → target-only ha inizialmente scoperto una corruzione reale
del checkpoint system residente su host: il payload binario veniva scritto con
`fmemopen(..., payload_bytes, "wb")`, senza spazio fisico per il NUL aggiunto
alla chiusura dello stream. La capacità è ora `payload_bytes + 1`, mentre la
lunghezza logica letta e sottoposta a checksum resta `payload_bytes`. Un test
unitario conserva esplicitamente un ultimo byte non nullo; il gate live
successivo non registra più `corrupt or trailing`, ricostruisce da
`memory-request` e non rilegge il system prompt da SSD dopo la risposta. La
stessa correzione è applicata allo snapshot di sessione generico che usava lo
stesso pattern.

Artifact:

- `performance-results/q38-agent-plain-story-search-static-compare-oracle-20260827/experiment.json`;
- `performance-results/q38-constrained-search-static-candidate-compare-20260827-002/experiment.json`.

## Matrice temperatura, thinking e MTP

Il gate `tests/test_server_sampling_matrix.py` esplicita il prodotto cartesiano
richiesto invece di affidarsi ai default del modello:

| Caso | Temperatura | `top_p` | `top_k` Chat | Thinking |
|---|---:|---:|---:|---:|
| `greedy_nothink` | 0 | 0,80 | 20 | no |
| `sampled_nothink` | 0,70 | 0,80 | 20 | no |
| `greedy_think` | 0 | 0,95 | 20 | sì |
| `sampled_think` | 0,60 | 0,95 | 20 | sì |

Ogni caso attraversa tre superfici: Chat senza tool, Responses con capability
agentica opzionale e risposta testuale, Responses con tool obbligatorio. Ogni
superficie gira target-only e con sidecar MTP, per 24 richieste live totali. Il
test usa Chat per il testo libero, dove DS4 espone anche `seed`, `top_k` e
`min_p`; usa Responses per `agentic`, perché il server rifiuta intenzionalmente
questa estensione su Chat. Responses espone `temperature` e `top_p`, ma non
finge di avere un seed o i knob non supportati.

Il gate controlla anche il percorso interno:

- il testo libero MTP deve produrre cicli speculativi;
- le fasi DSML con sidecar caricato devono dichiarare i passi target-only;
- `tool_choice=auto` senza thinking deve usare `search_static_steps` senza
  fallback;
- con thinking, il `</think>` trattenuto nella finestra SEARCH deve causare
  prima fallback dinamici fail-closed e poi passi statici;
- il tool obbligatorio deve attraversare il filtro strutturale completo e
  restituire una sola call con argomenti esatti.

Il 28 agosto 2026 sono passati 12/12 casi target e 12/12 casi MTP. Il confronto
appaiato passa in 6 casi con identità semantica completa e in 6 con identità del
contratto pubblico. Chat ha seed fisso e resta byte-identico. Responses non
espone seed; inoltre il verifier batch può scegliere un diverso near-argmax nel
reasoning nascosto. In questi casi il gate consente differenze nel reasoning o
nel testo libero campionato, ma non in presenza/assenza del thinking, scelta del
tool, argomenti, finish o esposizione di marker strutturali.

Artifact validati con la stessa revisione del test:

- `performance-results/q38-server-sampling-matrix-search-static-20260828-007/target/matrix.json`;
- `performance-results/q38-server-sampling-matrix-search-static-20260828-006/mtp/matrix.json`.

Il runner ripetibile avvia entrambi i server e applica lo stesso confronto:

```powershell
./tests/run_server_sampling_matrix.ps1 -Mode both
```

Il controllo model-free è incluso in `make test` ed è disponibile separatamente
con `make test-server-sampling-matrix-static`.

## Fonti primarie e scelte implementative

I valori campionati non sono inventati dal benchmark. La guida ufficiale
[Qwen3 quickstart](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md)
raccomanda `temperature=0.6`, `top_p=0.95`, `top_k=20` per thinking e
`temperature=0.7`, `top_p=0.8`, `top_k=20` per non-thinking; avverte inoltre che
il greedy thinking può degradare o ripetersi. La matrice mantiene comunque il
caso thinking a temperatura zero perché è un asse di compatibilità richiesto,
ma non lo usa come raccomandazione di qualità. La guida ufficiale
[Qwen function calling](https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md)
copre esplicitamente tool calling con thinking abilitato e disabilitato.

Per MTP sono stati confrontati il flusso upstream di
[llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md),
la documentazione [vLLM MTP](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md),
il modello Qwen MTP di
[vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)
e il regression test vLLM che mantiene la grammatica dopo la chiusura del
reasoning dentro una finestra speculativa
([test MTP structured output](https://github.com/vllm-project/vllm/blob/main/tests/v1/spec_decode/test_mtp_structured_output.py)).
I fondamenti sono *Fast Inference from Transformers via Speculative Decoding*
([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)) e *Accelerating Large
Language Model Decoding with Speculative Sampling*
([arXiv:2302.01318](https://arxiv.org/abs/2302.01318)): il target resta
autorevole e gli scostamenti numerici del microbatch non autorizzano mai una
violazione del contratto pubblico o della grammatica.

Per il checkpoint residente è stato seguito il contratto POSIX/glibc di
[`fmemopen`](https://man7.org/linux/man-pages/man3/fmemopen.3.html): uno stream
aperto in scrittura aggiunge un NUL su flush/close quando c'è spazio, e la
documentazione richiede al chiamante un byte extra nella capacità. Il payload
binario non dipende dal NUL e viene riaperto in lettura con la sua lunghezza
esatta.

## Test live ripetibile

`tests/run_agent_story_live.ps1` avvia in sequenza target-only e MTP, abilita il
phase profiling, assegna KV SSD e workroot univoci, esegue un warm-up e due run
misurate e conserva risposta, hash, token usage, phase profile e gate
funzionali. Il profiler invia esplicitamente `reasoning.effort=none`: senza quel
campo Qwen3.8 consumava il budget nel reasoning e Agent Wiki effettuava una
seconda richiesta di recupero.

```powershell
./tests/run_agent_story_live.ps1 -Mode both
```

Il test richiede almeno 400 token di output e 250 parole, `finish=stop`, una
sola richiesta modello, zero tool call, nessun marker DSML esposto e un
checkpoint post-risposta residente senza rilettura SSD.
