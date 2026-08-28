# Indice: architecture

Indice generato; non modificare a mano.

## [Qwen3.6 MTP implementation](qwen36-mtp-design.md)

DS4 supports native Qwen3.6 Multi-Token Prediction on the single-GPU CUDA

- righe 3–25: **Scope and contract** — DS4 supports native Qwen3.6 Multi-Token Prediction on the single-GPU CUDA
- righe 26–54: **GGUF and NextN semantics** — The support model is selected structurally, not by filename:
- righe 55–70: **Session state** — Each enabled Qwen session owns:
- righe 71–107: **Production adaptive-depth cycle** — Below 2K, the configured depth-two cycle is:
- righe 108–137: **Sampling and logits masks** — DS4 follows llama.cpp's current sample-and-match contract.
- righe 138–166: **CUDA paths** — Q4_K, Q5_K, and Q6_K verifier matmuls use Q8_1 plus a second Q8_1 residual.
  - righe 140–159: **Target verifier** — Q4_K, Q5_K, and Q6_K verifier matmuls use Q8_1 plus a second Q8_1 residual.
  - righe 160–166: **Support network** — The Q4_0 shared head uses Q8_1+residual DP4A and schedules eight output rows per
- righe 167–178: **Failure policy** — diagnostic.
- righe 179–249: **Validation gate** — The production model-backed command is:

## [Qwen3.8 27B UD-Q4_K_S compatibility audit](qwen38-compatibility.md)

DS4 recognizes only the audited Unsloth pair at repository revision

- righe 3–15: **Pinned artifacts** — DS4 recognizes only the audited Unsloth pair at repository revision
- righe 16–44: **Architecture and GGUF contract** — The runtime architecture is the existing `qwen35` implementation: 64 target
- righe 45–73: **Frontend behavior** — Both `ds4` and `ds4-server` recognize the stable model ID
- righe 74–100: **Test and oracle isolation** — `make test-qwen38` validates the pinned metadata snapshots, embedded and
- righe 101–111: **CUDA runtime status** — The Qwen3.6 graph is reused, while Q3_K, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,

## [Qwen3.6-27B Q4_K_S — Architettura completa, memoria ricorrente e quantizzazione](qwen_architecture.md)

Modello:

- righe 29–59: **1. Identità del modello** — Modello:
- righe 60–87: **2. Parametri principali** — ---
- righe 88–150: **3. Visione globale del modello** — Per una sequenza di testo:
- righe 151–209: **4. Ordine esatto dei 64 layer** — Il pattern è:
- righe 210–270: **5. Struttura generale di ogni decoder layer** — Ogni decoder layer mantiene la dimensione:
- righe 271–332: **6. Token embedding** — Il vocabolario contiene:
- righe 333–380: **7. RMSNorm** — La normalizzazione principale usa:
- righe 381–481: **8. FFN: tipo e dimensioni** — Tutti i 64 decoder layer contengono lo stesso tipo di FFN.
  - righe 439–481: **Matrici** — In convenzione PyTorch `[out_features, in_features]`:
- righe 482–529: **9. Gated DeltaNet: panoramica** — 48 dei 64 layer usano un **Gated DeltaNet** al posto della
- righe 530–611: **10. Proiezioni del Gated DeltaNet** — Da un hidden state da 5120:
- righe 541–561: **Q/K/V** — in_proj_qkv:
- righe 562–575: **Gate z** — in_proj_z:
- righe 576–585: **Gate b** — in_proj_b:
- righe 586–595: **Gate a** — in_proj_a:
- righe 596–611: **Output** — Dopo il DeltaNet:
- righe 612–689: **11. Gated DeltaNet completo** — h_t
- righe 690–743: **12. Depthwise causal Conv1D** — Prima della Delta Rule, Q/K/V attraversano una convolution temporale.
- righe 744–819: **13. Primo stato persistente: Conv state** — Durante il decoding token-per-token non avrebbe senso ricalcolare
- righe 820–896: **14. Secondo stato persistente: matrice ricorrente DeltaNet** — La memoria più importante è:
- righe 897–949: **15. Come funziona matematicamente S_t** — Consideriamo una singola testa.
- righe 916–949: **Passaggio 1 — decay della memoria** — S̄_t = exp(g_t) · S_(t-1)
- righe 950–978: **16. Lettura predittiva dalla memoria** — La key corrente interroga lo stato:
- righe 979–1016: **17. Calcolo dell'errore / Delta** — Si confronta ciò che la memoria prevedeva con il vero `v_t`:
- righe 1017–1071: **18. Scrittura nella memoria** — La memoria viene aggiornata mediante un prodotto esterno:
- righe 1072–1092: **19. Lettura finale dalla nuova memoria** — Dopo aver scritto il nuovo dato, la query legge lo stato aggiornato:
- righe 1093–1163: **20. DeltaNet completo in un singolo diagramma didattico** — STATO DAL PASSATO
- righe 1164–1251: **21. Da dove arrivano g_t e β_t?** — Il modello produce due vettori da 48 elementi:
- righe 1252–1303: **22. Q/K vengono replicati 3 volte nel DeltaNet** — Originariamente:
- righe 1304–1349: **23. RMSNormGated all'uscita del DeltaNet** — L'output ricorrente non viene inviato direttamente a `out_proj`.
- righe 1350–1391: **24. Full Attention: i restanti 16 layer** — Ogni quarto layer usa una normale causal self-attention,
- righe 1392–1473: **25. Proiezioni della Full Attention** — L'input è:
- righe 1401–1445: **Query + output gate** — Qwen usa una particolarità:
- righe 1446–1453: **K** — k_proj:
- righe 1454–1461: **V** — v_proj:
- righe 1462–1473: **Output** — Dopo l'attention:
- righe 1474–1540: **26. Full Attention completa** — h_t
- righe 1541–1584: **27. Partial RoPE** — La dimensione di ciascuna attention head è:
- righe 1585–1650: **28. Terzo stato persistente: KV cache** — I 16 Full Attention layer NON usano la matrice ricorrente `S_t`.
- righe 1651–1702: **29. Differenza fondamentale: S_t contro KV cache** — Non conserva tutti i token.
- righe 1653–1673: **Gated DeltaNet** — Non conserva tutti i token.
- righe 1674–1702: **Full Attention** — Conserva K e V di ogni posizione:
- righe 1703–1716: **30. Le tre memorie temporali di Qwen3.6-27B** — Nel decoding esistono quindi tre tipi distinti di stato.
- righe 1717–1802: **31. È quindi autoregressivo oppure ricorrente?** — **Entrambi**, ma le parole descrivono aspetti differenti.
- righe 1721–1755: **Autoregressivo** — Significa che la distribuzione del token successivo è:
- righe 1756–1802: **Ricorrente** — Significa che internamente alcuni layer hanno uno stato:
- righe 1803–1837: **32. Cosa attraversa realmente il confine fra due token** — Alla fine dell'iterazione `t` abbiamo:
- righe 1838–1941: **33. Vista completa token-per-token** — ╔══════════════════════════════════════════════════════════════╗
- righe 1942–1986: **34. Un errore concettuale comune** — Una descrizione troppo semplice di un Transformer dice:
- righe 1987–2047: **35. Perché combinare DeltaNet e Full Attention?** — Le due memorie hanno proprietà complementari.
- righe 1991–2009: **DeltaNet** — Vantaggi:
- righe 2010–2027: **Full Attention** — Vantaggio:
- righe 2028–2047: **Architettura ibrida** — Qwen usa:
- righe 2048–2084: **36. Dimensione dello stato ricorrente DeltaNet** — Per un Gated DeltaNet:
- righe 2085–2109: **37. Dimensione del Conv state** — Per un GDN:
- righe 2110–2153: **38. Crescita della KV cache** — Per un Full Attention layer e un token:
- righe 2154–2219: **39. Final RMSNorm e LM Head** — Dopo il decoder layer 64:
- righe 2220–2250: **40. Vision Encoder** — Qwen3.6-27B è multimodale.
- righe 2251–2301: **41. Patch embedding visuale** — Il patch embedding è realizzato con una Conv3D:
- righe 2302–2339: **42. Vision Transformer** — Sono presenti:
- righe 2340–2404: **43. Vision merger** — Il merger spaziale usa:
- righe 2405–2447: **44. Pipeline multimodale semplificata** — IMMAGINE / VIDEO
- righe 2448–2483: **45. Quantizzazione Q4_K_S: cosa significa** — Ora separiamo completamente due concetti:
- righe 2484–2559: **46. Pesi statici contro stato dinamico** — Questa distinzione è fondamentale.
- righe 2488–2521: **Pesi statici** — Esempi:
- righe 2522–2559: **Stato dinamico** — Esempi:
- righe 2560–2605: **47. Grafico: dove agisce Q4_K_S** — MODELLO SU DISCO
- righe 2606–2641: **48. Cos'è il primitive Q4_K** — Il tipo di base del preset Q4_K_S è:
- righe 2642–2729: **49. Struttura di block_q4_K** — Un super-block contiene:
- righe 2730–2783: **50. Perché Q4_K_S non equivale esattamente a 4.5 bpw** — Il nome completo interno è:
- righe 2784–2828: **51. Alcune promozioni del preset Q4_K_S** — Nel codice corrente di llama.cpp,
- righe 2829–2870: **52. `S` in Q4_K_S** — È importante non interpretare `S` come:
- righe 2871–2913: **53. Q4_K_S non modifica lo stato DeltaNet** — Questa distinzione merita di essere visualizzata.
- righe 2914–2965: **54. Q4_K_S non modifica le equazioni del modello** — BF16 e Q4_K_S rappresentano lo stesso modello matematico
- righe 2919–2930: **BF16** — h
- righe 2931–2965: **Q4_K_S** — h
- righe 2966–3017: **55. Riassunto completo delle matrici principali** — ---
- righe 2968–2977: **Embedding/output** — ---
- righe 2978–2987: **FFN — tutti i 64 layer** — ---
- righe 2988–3003: **Gated DeltaNet — 48 layer** — ---
- righe 3004–3017: **Full Attention — 16 layer** — ---
- righe 3018–3107: **56. Mappa mentale finale** — Qwen3.6-27B
- righe 3108–3180: **57. Il modello in una sola immagine concettuale** — PASSATO
- righe 3181–3270: **58. Frase conclusiva tecnicamente precisa** — Qwen3.6-27B può essere descritto così:
- righe 3271–3312: **59. Fonti primarie di riferimento** — I dati architetturali vanno verificati principalmente in:
- righe 3290–3312: **Nota importante sulla certezza di Q4_K_S** — `Q4_K_S` definisce un preset di quantizzazione, non la lista
