# Qwen3.6-27B Q4_K_S — Architettura completa, memoria ricorrente e quantizzazione

> **Obiettivo di questo documento**
>
> Descrivere Qwen3.6-27B dalla sequenza di input fino ai logits di output,
> spiegando:
>
> - dimensioni di input e output;
> - embedding;
> - struttura esatta dei 64 layer;
> - Gated DeltaNet;
> - Full Gated Attention;
> - FFN / SwiGLU;
> - RMSNorm;
> - Vision Encoder;
> - flusso autoregressivo;
> - stati interni persistenti tra un token e il successivo;
> - differenza fra stato ricorrente e KV cache;
> - cosa significa realmente `Q4_K_S`;
> - cosa viene quantizzato e cosa NON viene automaticamente quantizzato.
>
> Il punto concettuale più importante è:
>
> **Qwen3.6-27B è autoregressivo nell'output, ma la sua memoria interna è
> ibrida: ricorrente nei Gated DeltaNet + KV cache nei layer di full attention.**

---

# 1. Identità del modello

Modello:

**Qwen3.6-27B**

Architettura interna dichiarata nel checkpoint:

```text
Qwen3_5ForConditionalGeneration
model_type = qwen3_5
```

Questo non è un errore: il modello pubblico si chiama Qwen3.6-27B,
ma riutilizza l'implementazione architetturale `qwen3_5` di Transformers.

Il modello è:

- decoder causale/autoregressivo;
- multimodale testo + immagini/video;
- denso, non MoE;
- 64 layer linguistici;
- architettura ibrida:
  - 48 Gated DeltaNet;
  - 16 Full Attention;
- hidden size = **5120**;
- vocabolario = **248320**;
- contesto massimo dichiarato = **262144 token**.

---

# 2. Parametri principali

| Parametro | Valore |
|---|---:|
| Vocabulary size | 248320 |
| Hidden size | 5120 |
| Decoder layers | 64 |
| Gated DeltaNet layers | 48 |
| Full Attention layers | 16 |
| Full Attention interval | 4 |
| FFN intermediate size | 17408 |
| Activation FFN | SiLU |
| RMSNorm epsilon | 1e-6 |
| Attention Q heads | 24 |
| Attention KV heads | 4 |
| Attention head dimension | 256 |
| DeltaNet K heads | 16 |
| DeltaNet V heads | 48 |
| DeltaNet K head dimension | 128 |
| DeltaNet V head dimension | 128 |
| DeltaNet Conv1D kernel | 4 |
| Native context | 262144 |
| RoPE theta | 10000000 |
| Partial rotary factor | 0.25 |
| Input/output embeddings tied | No |

---

# 3. Visione globale del modello

Per una sequenza di testo:

```text
Token IDs
[B, T]
   │
   ▼
┌───────────────────────────────────┐
│ TOKEN EMBEDDING                   │
│                                   │
│ 248320 token                      │
│          ↓                        │
│ 5120 feature per token            │
└───────────────────────────────────┘
   │
   │ [B, T, 5120]
   ▼
┌───────────────────────────────────┐
│ 64 DECODER LAYER                  │
│                                   │
│ pattern ripetuto 16 volte:        │
│                                   │
│ ┌──────────────┐                  │
│ │ GatedDelta 1 │                  │
│ └──────┬───────┘                  │
│        ▼                          │
│ ┌──────────────┐                  │
│ │ GatedDelta 2 │                  │
│ └──────┬───────┘                  │
│        ▼                          │
│ ┌──────────────┐                  │
│ │ GatedDelta 3 │                  │
│ └──────┬───────┘                  │
│        ▼                          │
│ ┌──────────────┐                  │
│ │ Full Attn    │                  │
│ └──────────────┘                  │
│                                   │
│       × 16 gruppi                 │
└───────────────────────────────────┘
   │
   │ [B, T, 5120]
   ▼
┌───────────────────────────────────┐
│ FINAL RMSNORM                     │
│ 5120 → 5120                       │
└───────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────┐
│ LM HEAD                           │
│ 5120 → 248320                     │
└───────────────────────────────────┘
   │
   ▼
Logits
[B, T, 248320]
```

---

# 4. Ordine esatto dei 64 layer

Il pattern è:

```text
GDN
GDN
GDN
FULL ATTENTION
```

ripetuto **16 volte**.

In numerazione umana 1-based:

```text
Layer  1  = Gated DeltaNet
Layer  2  = Gated DeltaNet
Layer  3  = Gated DeltaNet
Layer  4  = Full Attention

Layer  5  = Gated DeltaNet
Layer  6  = Gated DeltaNet
Layer  7  = Gated DeltaNet
Layer  8  = Full Attention

Layer  9  = Gated DeltaNet
Layer 10  = Gated DeltaNet
Layer 11  = Gated DeltaNet
Layer 12  = Full Attention

...

Layer 61  = Gated DeltaNet
Layer 62  = Gated DeltaNet
Layer 63  = Gated DeltaNet
Layer 64  = Full Attention
```

I Full Attention layer sono quindi:

```text
4, 8, 12, 16,
20, 24, 28, 32,
36, 40, 44, 48,
52, 56, 60, 64
```

In indici 0-based:

```text
3, 7, 11, 15,
19, 23, 27, 31,
35, 39, 43, 47,
51, 55, 59, 63
```

---

# 5. Struttura generale di ogni decoder layer

Ogni decoder layer mantiene la dimensione:

```text
5120 → 5120
```

e usa una struttura Pre-Norm.

Schema:

```text
                   residual
                      │
                      │─────────────────────────────┐
                      │                             │
                      ▼                             │
                ┌───────────┐                       │
                │ RMSNorm   │                       │
                │   5120    │                       │
                └─────┬─────┘                       │
                      │                             │
                      ▼                             │
              ┌─────────────────┐                   │
              │  TOKEN MIXER    │                   │
              │                 │                   │
              │ Gated DeltaNet  │                   │
              │       oppure    │                   │
              │ Full Attention  │                   │
              └────────┬────────┘                   │
                       │                            │
                       ▼                            │
                    ADD ◄───────────────────────────┘
                       │
                       │ residual
                       │────────────────────────────┐
                       ▼                            │
                ┌───────────┐                       │
                │ RMSNorm   │                       │
                │   5120    │                       │
                └─────┬─────┘                       │
                      │                             │
                      ▼                             │
                ┌───────────┐                       │
                │ SwiGLU    │                       │
                │   FFN     │                       │
                └─────┬─────┘                       │
                      │                             │
                      ▼                             │
                    ADD ◄───────────────────────────┘
                      │
                      ▼
                output 5120
```

Quindi il token mixer cambia fra i layer, ma l'FFN è presente
in tutti i 64 layer.

---

# 6. Token embedding

Il vocabolario contiene:

```text
248320 token
```

Ogni token viene trasformato in un vettore:

```text
5120-dimensional
```

Concettualmente:

```text
token ID
   │
   │ indice 0 ... 248319
   ▼

Embedding table

      5120 feature
   ◄────────────────────►

┌─────────────────────────┐
│ token 0                 │
├─────────────────────────┤
│ token 1                 │
├─────────────────────────┤
│ ...                     │
├─────────────────────────┤
│ token 248319            │
└─────────────────────────┘

shape matrice:

[248320, 5120]
```

Per un batch:

```text
input IDs:
[B,T]

        ↓ embedding

hidden states:
[B,T,5120]
```

Gli embedding di input e l'LM head finale **non condividono i pesi**:

```text
tie_word_embeddings = false
```

---

# 7. RMSNorm

La normalizzazione principale usa:

```text
RMSNorm
epsilon = 1e-6
```

Per i decoder layer:

```text
input_layernorm:
5120 → 5120

post_attention_layernorm:
5120 → 5120
```

La normalizzazione può essere vista concettualmente come:

```text
                  x
                  │
                  ▼
       media dei quadrati
                  │
                  ▼
        sqrt(mean(x²)+ε)
                  │
                  ▼
          x / RMS(x)
                  │
                  ▼
            peso appreso
                  │
                  ▼
             output
```

Nell'implementazione Qwen la parametrizzazione è centrata su 1:

```text
output = normalized_x × (1 + weight)
```

---

# 8. FFN: tipo e dimensioni

Tutti i 64 decoder layer contengono lo stesso tipo di FFN.

Dimensioni:

```text
input hidden:
5120
```

Due proiezioni parallele:

```text
gate_proj:
5120 → 17408

up_proj:
5120 → 17408
```

Poi:

```text
SiLU(gate_proj(x))
       │
       │
       × element-wise
       │
up_proj(x)
       │
       ▼
    17408
       │
       ▼
down_proj
17408 → 5120
```

Formula:

```text
FFN(x) =
    W_down(
        SiLU(W_gate x)
        ⊙
        W_up x
    )
```

Dove:

```text
⊙ = moltiplicazione elemento per elemento
```

È quindi una MLP gated di tipo **SwiGLU-style**.

### Matrici

In convenzione PyTorch `[out_features, in_features]`:

| Tensor | Mapping | Weight shape |
|---|---:|---:|
| gate_proj | 5120 → 17408 | [17408,5120] |
| up_proj | 5120 → 17408 | [17408,5120] |
| down_proj | 17408 → 5120 | [5120,17408] |

Le tre proiezioni non hanno bias.

Grafico:

```text
                         x [5120]
                       /          \
                      /            \
                     ▼              ▼
             gate_proj           up_proj
           5120 → 17408       5120 → 17408
                  │                 │
                  ▼                 │
                SiLU                │
                  │                 │
                  └───────┬─────────┘
                          │
                          ▼
                  element-wise ×
                          │
                          ▼
                       [17408]
                          │
                          ▼
                    down_proj
                  17408 → 5120
                          │
                          ▼
                       [5120]
```

---

# 9. Gated DeltaNet: panoramica

48 dei 64 layer usano un **Gated DeltaNet** al posto della
self-attention standard.

Il suo scopo è poter trasportare informazione dal passato
al presente attraverso uno **stato ricorrente di dimensione fissa**.

Configurazione:

```text
hidden size             = 5120

K heads iniziali        = 16
K head dim              = 128

Q heads iniziali        = 16
Q head dim              = 128

V heads                 = 48
V head dim              = 128

Conv1D kernel           = 4
```

Da cui:

```text
Q dimension:
16 × 128 = 2048

K dimension:
16 × 128 = 2048

V dimension:
48 × 128 = 6144
```

Concatenando:

```text
QKV dimension =
2048 + 2048 + 6144
= 10240
```

---

# 10. Proiezioni del Gated DeltaNet

Da un hidden state da 5120:

```text
h_t
[5120]
```

vengono prodotte diverse quantità.

## Q/K/V

```text
in_proj_qkv:

5120 → 10240
```

che successivamente viene separato:

```text
                 10240
                    │
        ┌───────────┼───────────────┐
        │           │               │
        ▼           ▼               ▼
     Q 2048      K 2048          V 6144
        │           │               │
   16 × 128    16 × 128         48 × 128
```

## Gate z

```text
in_proj_z:

5120 → 6144
```

cioè:

```text
48 × 128
```

## Gate b

```text
in_proj_b:

5120 → 48
```

un valore per ciascuna delle 48 V-head.

## Gate a

```text
in_proj_a:

5120 → 48
```

anche questo produce un valore per testa.

## Output

Dopo il DeltaNet:

```text
6144 → 5120
```

tramite:

```text
out_proj
```

---

# 11. Gated DeltaNet completo

```text
                           h_t
                          [5120]
                             │
                             ▼
                        RMSNorm
                             │
            ┌────────────────┼──────────────────┐
            │                │                  │
            ▼                ▼                  ▼
     in_proj_qkv         in_proj_z          a_t , b_t
     5120→10240          5120→6144         5120→48
            │
            ▼
  Depthwise Causal Conv1D
  channels = 10240
  kernel   = 4
            │
            ▼
   ┌────────┼──────────┐
   │        │          │
   ▼        ▼          ▼
 Q=2048   K=2048     V=6144
 16×128   16×128     48×128
   │        │          │
   │        │          │
   └─── repeat 3× ─────┤
            │
            ▼
     Q = 48×128
     K = 48×128
     V = 48×128
            │
            │
            │           S_(t-1)
            │      [48,128,128]
            │             │
            ▼             ▼
      ┌────────────────────────┐
      │   GATED DELTA RULE     │
      │                        │
      │ aggiorna memoria S_t   │
      └───────────┬────────────┘
                  │
                  ▼
            48 × 128
                  │
                  │
          z_t ────┤
                  ▼
          RMSNorm Gated
             + SiLU
                  │
                  ▼
               6144
                  │
                  ▼
             out_proj
           6144 → 5120
                  │
                  ▼
               residual
                  │
                  ▼
              RMSNorm
                  │
                  ▼
             SwiGLU FFN
       5120 → 17408 → 5120
                  │
                  ▼
               residual
```

---

# 12. Depthwise causal Conv1D

Prima della Delta Rule, Q/K/V attraversano una convolution temporale.

Dimensione:

```text
10240 canali
```

Kernel:

```text
4
```

Groups:

```text
10240
```

Dato che:

```text
groups = channels
```

è una **depthwise convolution**:

ogni canale ha il proprio filtro temporale indipendente.

Schema:

```text
           tempo →

canale 0:  x[t-3] x[t-2] x[t-1] x[t]
             └──────── kernel 4 ───────┘

canale 1:  x[t-3] x[t-2] x[t-1] x[t]
             └──────── kernel 4 ───────┘

...

canale 10239:
            x[t-3] x[t-2] x[t-1] x[t]
             └──────── kernel 4 ───────┘
```

Questo introduce una memoria locale di breve periodo.

---

# 13. Primo stato persistente: Conv state

Durante il decoding token-per-token non avrebbe senso ricalcolare
ogni volta gli ultimi quattro vettori.

Per questo ciascun Gated DeltaNet conserva un **conv state**.

Shape concettuale/runtime:

```text
C_t^ℓ

[B, 10240, 4]
```

dove:

```text
B      = batch size
10240  = canali QKV concatenati
4      = kernel temporale
ℓ      = indice del Gated DeltaNet layer
```

Il passaggio temporale è:

```text
ITERAZIONE t-1

C_(t-1)
   │
   │ memoria locale
   ▼

────────────────────────────────────
        confine temporale
────────────────────────────────────

ITERAZIONE t

nuovo QKV_t
   │
   │
   ├───────────────┐
   │               │
   ▼               ▼
C_(t-1)        nuovo valore
   │               │
   └───────┬───────┘
           ▼
   causal Conv1D
      kernel 4
           │
           ├────────────► Q_t,K_t,V_t
           │
           ▼
          C_t
           │
           │ salvato
           ▼

────────────────────────────────────
        confine temporale
────────────────────────────────────

ITERAZIONE t+1

C_t viene riutilizzato
```

Questa è una memoria corta.

Non è la memoria principale a lungo termine del DeltaNet.

---

# 14. Secondo stato persistente: matrice ricorrente DeltaNet

La memoria più importante è:

```text
S_t
```

Per ciascun Gated DeltaNet layer:

```text
S_t^ℓ
=
[B, 48, 128, 128]
```

Ignorando il batch, ogni layer possiede:

```text
48 matrici da 128 × 128
```

cioè una matrice per ciascuna V-head.

Grafico:

```text
Gated DeltaNet layer ℓ

┌───────────────────────────────────────────────────────────┐
│                                                           │
│ Head 0                                                    │
│ S_t[0] = matrice 128 × 128                                │
│                                                           │
├───────────────────────────────────────────────────────────┤
│ Head 1                                                    │
│ S_t[1] = matrice 128 × 128                                │
│                                                           │
├───────────────────────────────────────────────────────────┤
│ ...                                                       │
├───────────────────────────────────────────────────────────┤
│ Head 47                                                   │
│ S_t[47] = matrice 128 × 128                               │
│                                                           │
└───────────────────────────────────────────────────────────┘

Totale:

48 × 128 × 128
=
786432 valori per layer
```

E ci sono 48 Gated DeltaNet:

```text
786432 × 48
=
37748736 valori
```

di recurrent state per sequenza/batch element.

La cosa fondamentale:

```text
dimensione S dopo 100 token
=
dimensione S dopo 10000 token
=
dimensione S dopo 200000 token
```

La memoria DeltaNet è **O(1) rispetto alla lunghezza del contesto**.

---

# 15. Come funziona matematicamente S_t

Consideriamo una singola testa.

Al passo temporale `t` abbiamo:

```text
q_t
k_t
v_t

S_(t-1)

g_t
β_t
```

Lo stato precedente viene prima fatto decadere.

## Passaggio 1 — decay della memoria

```text
S̄_t = exp(g_t) · S_(t-1)
```

Dato che `g_t` viene costruito in modo da essere non positivo:

```text
0 < exp(g_t) ≤ 1
```

Quindi il gate decide quanto del passato mantenere.

Visualmente:

```text
S_(t-1)
   │
   │ × exp(g_t)
   ▼
  S̄_t

g_t molto vicino a 0
→ exp(g_t) ≈ 1
→ ricordo quasi tutto

g_t molto negativo
→ exp(g_t) ≈ 0
→ dimentico fortemente
```

---

# 16. Lettura predittiva dalla memoria

La key corrente interroga lo stato:

```text
v_hat_t = k_tᵀ S̄_t
```

Interpretazione didattica:

```text
       k_t
        │
        ▼
┌─────────────────┐
│ memoria S̄_t    │
└────────┬────────┘
         │
         ▼
      v_hat_t
```

Ovvero:

> "Dato questo tipo di key, quale value crede già di conoscere
> la memoria?"

---

# 17. Calcolo dell'errore / Delta

Si confronta ciò che la memoria prevedeva con il vero `v_t`:

```text
errore =
v_t - v_hat_t
```

poi il gate β decide quanto apprendere:

```text
δ_t =
β_t · (v_t - v_hat_t)
```

Grafico:

```text
                    vero valore
                       v_t
                        │
                        │
                        ▼
                    ┌───────┐
memoria ──► v_hat_t │   -   │
                    └───┬───┘
                        │
                        ▼
                       errore
                        │
                        │ × β_t
                        ▼
                       δ_t
```

---

# 18. Scrittura nella memoria

La memoria viene aggiornata mediante un prodotto esterno:

```text
S_t =
S̄_t
+
k_t δ_tᵀ
```

Il termine:

```text
k_t δ_tᵀ
```

ha dimensione:

```text
128 × 128
```

ed è quindi compatibile con la matrice di memoria.

Schema:

```text
                 S_(t-1)
                    │
                    │ decay
                    ▼
                   S̄_t
                    │
                    │
              ┌─────┴──────┐
              │            │
              │            │
             k_t          δ_t
              │            │
              └─────┬──────┘
                    │
                    ▼
             prodotto esterno
               k_t δ_tᵀ
                    │
                    ▼
           S̄_t + k_t δ_tᵀ
                    │
                    ▼
                   S_t
```

---

# 19. Lettura finale dalla nuova memoria

Dopo aver scritto il nuovo dato, la query legge lo stato aggiornato:

```text
o_t =
q_tᵀ S_t
```

ottenendo un vettore da 128 elementi per testa.

Con 48 head:

```text
48 × 128
=
6144
```

---

# 20. DeltaNet completo in un singolo diagramma didattico

```text
                    STATO DAL PASSATO
                         S_(t-1)
                    [48,128,128]
                            │
                            │
                  ┌─────────▼──────────┐
                  │ DECAY              │
                  │                    │
                  │ × exp(g_t)         │
                  └─────────┬──────────┘
                            │
                            ▼
                           S̄_t
                            │
                 ┌──────────┴───────────┐
                 │                      │
                 │                      │
                 │ k_tᵀ S̄_t            │
                 │                      │
                 ▼                      │
              v_hat_t                   │
                 │                      │
                 │                      │
       v_t ──────┤                      │
                 ▼                      │
         v_t - v_hat_t                  │
                 │                      │
                 │ × β_t                │
                 ▼                      │
                δ_t                     │
                 │                      │
                 │                      │
         k_t ────┤                      │
                 ▼                      │
             k_t δ_tᵀ                   │
                 │                      │
                 └─────────┬────────────┘
                           │
                           ▼
                  S_t = S̄_t + k_tδ_tᵀ
                           │
             ┌─────────────┴──────────────┐
             │                            │
             │                            │
             ▼                            ▼
       q_tᵀ S_t                    SALVA S_t
             │                     per token t+1
             ▼
          output_t
        [48 × 128]
             │
             ▼
           [6144]
```

Questa è una vera equazione ricorrente:

```text
S_t = f(S_(t-1), x_t)
```

Perciò l'informazione del passato non arriva al token seguente
soltanto attraverso il token appena generato.

Arriva anche attraverso lo stato interno `S_t`.

---

# 21. Da dove arrivano g_t e β_t?

Il modello produce due vettori da 48 elementi:

```text
a_t = in_proj_a(h_t)
b_t = in_proj_b(h_t)
```

con:

```text
5120 → 48
```

Il gate di scrittura è:

```text
β_t = sigmoid(b_t)
```

Quindi:

```text
0 < β_t < 1
```

Interpretazione:

```text
β_t piccolo
→ modifica poco la memoria

β_t grande
→ corregge fortemente la memoria
```

Il decay viene costruito come:

```text
g_t =
-exp(A_log) · softplus(a_t + dt_bias)
```

Quindi:

```text
g_t ≤ 0
```

e pertanto:

```text
0 < exp(g_t) ≤ 1
```

Interpretazione:

```text
exp(g_t)
≈ fattore di conservazione della memoria.
```

In forma molto semplificata:

```text
             h_t
              │
     ┌────────┴────────┐
     │                 │
     ▼                 ▼
   b_proj            a_proj
  5120→48           5120→48
     │                 │
     ▼                 ▼
 sigmoid            softplus
     │                 │
     ▼                 ▼
    β_t               g_t
     │                 │
     ▼                 ▼
"quanto             "quanto
 scrivere"           ricordare/
                     dimenticare"
```

---

# 22. Q/K vengono replicati 3 volte nel DeltaNet

Originariamente:

```text
Q:
16 head × 128

K:
16 head × 128

V:
48 head × 128
```

Per eseguire la Delta Rule sulle 48 V-head:

```text
48 / 16 = 3
```

quindi Q e K vengono ripetuti tre volte.

```text
Q originali:

Q0 Q1 Q2 ... Q15
 │
 │ repeat_interleave × 3
 ▼

Q0 Q0 Q0
Q1 Q1 Q1
Q2 Q2 Q2
...
Q15 Q15 Q15

totale = 48 head
```

Stessa cosa per K.

Alla Delta Rule arrivano quindi:

```text
Q = [B,T,48,128]
K = [B,T,48,128]
V = [B,T,48,128]
```

---

# 23. RMSNormGated all'uscita del DeltaNet

L'output ricorrente non viene inviato direttamente a `out_proj`.

Il Gated DeltaNet produce anche:

```text
z_t
```

da:

```text
in_proj_z:
5120 → 6144
```

Il risultato della Delta Rule viene combinato con `z_t`
tramite la normalizzazione gated.

Concettualmente:

```text
Delta Rule output
      │
      ▼
   RMSNorm
      │
      │
      ├─────────────┐
      │             │
      │             ▼
      │          SiLU(z)
      │             │
      └──────×──────┘
             │
             ▼
           6144
             │
             ▼
          out_proj
        6144 → 5120
```

---

# 24. Full Attention: i restanti 16 layer

Ogni quarto layer usa una normale causal self-attention,
ma con:

- Grouped Query Attention;
- Q/K RMSNorm;
- partial RoPE;
- output gate.

Configurazione:

```text
Q heads  = 24
KV heads = 4

head_dim = 256
```

Quindi:

```text
Q dimension:
24 × 256 = 6144

K dimension:
4 × 256 = 1024

V dimension:
4 × 256 = 1024
```

Rapporto GQA:

```text
24 / 4 = 6
```

quindi ciascuna K/V head serve 6 query heads.

---

# 25. Proiezioni della Full Attention

L'input è:

```text
h_t
5120
```

## Query + output gate

Qwen usa una particolarità:

`q_proj` produce contemporaneamente:

- Q;
- un gate G per l'output.

Perciò:

```text
q_proj:

5120 → 12288
```

perché:

```text
12288
=
6144 Q
+
6144 gate
```

Schema:

```text
h_t [5120]
     │
     ▼
 q_proj
5120 → 12288
     │
     ▼
┌────┴────┐
│         │
▼         ▼
Q         G
6144      6144
24×256    24×256
```

## K

```text
k_proj:

5120 → 1024
```

## V

```text
v_proj:

5120 → 1024
```

## Output

Dopo l'attention:

```text
6144 → 5120
```

tramite `o_proj`.

---

# 26. Full Attention completa

```text
                           h_t
                          [5120]
                             │
                             ▼
                         RMSNorm
                             │
                ┌────────────┼───────────────┐
                │            │               │
                ▼            ▼               ▼
             q_proj        k_proj          v_proj
          5120→12288     5120→1024       5120→1024
                │            │               │
             ┌──┴───┐        │               │
             │      │        │               │
             ▼      ▼        ▼               ▼
             Q      Gate     K               V
           6144     6144    1024            1024
             │               │
             ▼               ▼
          Q RMSNorm       K RMSNorm
             │               │
             └──────┬────────┘
                    │
                    ▼
                partial RoPE
                    │
                    │
                    │        KV cache passata
                    │             │
                    ▼             ▼
          ┌────────────────────────────┐
          │ CAUSAL GROUPED ATTENTION   │
          │                            │
          │ 24 Q heads                 │
          │ 4 KV heads                 │
          │ GQA ratio = 6              │
          └─────────────┬──────────────┘
                        │
                        ▼
                      6144
                        │
                        │
                 × sigmoid(Gate)
                        │
                        ▼
                      6144
                        │
                        ▼
                     o_proj
                   6144→5120
                        │
                        ▼
                     residual
                        │
                        ▼
                     RMSNorm
                        │
                        ▼
                    SwiGLU FFN
              5120→17408→5120
```

---

# 27. Partial RoPE

La dimensione di ciascuna attention head è:

```text
256
```

ma:

```text
partial_rotary_factor = 0.25
```

quindi le dimensioni che ricevono RoPE sono:

```text
256 × 0.25
=
64
```

Per ciascuna Q/K head:

```text
┌──────────────────────────── 256 ────────────────────────────┐
│                                                             │
│  64 dimensioni                 192 dimensioni                │
│  con RoPE                      pass-through                  │
│  ◄────────►                    ◄─────────────────────────►   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

RoPE theta:

```text
10000000
```

Il modello usa inoltre MRoPE per la gestione multimodale.

---

# 28. Terzo stato persistente: KV cache

I 16 Full Attention layer NON usano la matrice ricorrente `S_t`.

Conservano invece esplicitamente:

```text
K_1, K_2, ..., K_t
V_1, V_2, ..., V_t
```

Per ciascun Full Attention layer:

```text
K cache:
[B,4,T,256]

V cache:
[B,4,T,256]
```

dove `T` cresce con la sequenza.

Grafico:

```text
tempo:

 t=1
 ┌─────┐
 │ K1  │
 │ V1  │
 └─────┘

 t=2
 ┌─────┬─────┐
 │ K1  │ K2  │
 │ V1  │ V2  │
 └─────┴─────┘

 t=3
 ┌─────┬─────┬─────┐
 │ K1  │ K2  │ K3  │
 │ V1  │ V2  │ V3  │
 └─────┴─────┴─────┘

 ...

 t=T
 ┌─────┬─────┬─────┬───────────┬─────┐
 │  1  │  2  │  3  │    ...    │  T  │
 └─────┴─────┴─────┴───────────┴─────┘
```

Quindi:

```text
DeltaNet recurrent memory:
O(1) rispetto a T

KV cache:
O(T)
```

---

# 29. Differenza fondamentale: S_t contro KV cache

## Gated DeltaNet

Non conserva tutti i token.

Comprende il passato in una matrice:

```text
passato arbitrariamente lungo
         │
         ▼
┌─────────────────────┐
│ stato compatto S_t  │
│                     │
│ 48 × 128 × 128      │
└─────────────────────┘
```

Dimensione costante.

---

## Full Attention

Conserva K e V di ogni posizione:

```text
token 1 ──► K1,V1
token 2 ──► K2,V2
token 3 ──► K3,V3
...
token T ──► KT,VT

        │
        ▼

┌─────────────────────────────────┐
│ KV cache                        │
│                                 │
│ posizione 1                     │
│ posizione 2                     │
│ posizione 3                     │
│ ...                             │
│ posizione T                     │
└─────────────────────────────────┘
```

Dimensione crescente.

---

# 30. Le tre memorie temporali di Qwen3.6-27B

Nel decoding esistono quindi tre tipi distinti di stato.

| Stato | Layer | Shape per layer | Cresce con T? | Funzione |
|---|---:|---:|---:|---|
| Conv state C | 48 GDN | [B,10240,4] | No | memoria locale |
| Recurrent state S | 48 GDN | [B,48,128,128] | No | memoria ricorrente lunga |
| KV cache | 16 Attention | K,V=[B,4,T,256] | Sì | accesso esplicito ai token passati |

Questo è uno dei concetti più importanti dell'intera architettura.

---

# 31. È quindi autoregressivo oppure ricorrente?

**Entrambi**, ma le parole descrivono aspetti differenti.

## Autoregressivo

Significa che la distribuzione del token successivo è:

```text
P(x_(t+1) | x_1, x_2, ..., x_t)
```

e non può utilizzare token futuri.

Quindi:

```text
x_1
 │
 ▼
x_2
 │
 ▼
x_3
 │
 ▼
...
 │
 ▼
x_t
 │
 ▼
x_(t+1)
```

Il modello resta causale.

---

## Ricorrente

Significa che internamente alcuni layer hanno uno stato:

```text
S_t =
f(S_(t-1), input_t)
```

che viene passato direttamente all'iterazione successiva.

Quindi la computazione è più simile a:

```text
                         ┌──────────────────────┐
                         │                      │
                         │      S_(t-1)         │
                         │                      │
                         └──────────┬───────────┘
                                    │
                                    ▼
x_t ─────────────────────────► MODELLO
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
                   logits_t                    S_t
                       │                         │
                       ▼                         │
                   x_(t+1)                      │
                       │                         │
                       │                         │
                       └─────────────┐           │
                                     │           │
                                     ▼           ▼
                                  MODELLO ◄──────┘
```

Quindi:

> Il nuovo token è solo UNA delle informazioni che passano
> dall'iterazione t all'iterazione t+1.

Passano anche gli stati interni.

---

# 32. Cosa attraversa realmente il confine fra due token

Alla fine dell'iterazione `t` abbiamo:

```text
1. token scelto:
   x_(t+1)

2. per ciascuno dei 48 Gated DeltaNet:
   C_t^ℓ
   S_t^ℓ

3. per ciascuno dei 16 Full Attention:
   KV_(1:t)^m
```

L'iterazione seguente riceve quindi:

```text
                     ITERAZIONE t
                          │
           ┌──────────────┼───────────────┐
           │              │               │
           ▼              ▼               ▼
       x_(t+1)         C_t,S_t        KV_(1:t)
           │              │               │
           │              │               │
           └──────────────┼───────────────┘
                          │
                          ▼
                    ITERAZIONE t+1
```

---

# 33. Vista completa token-per-token

```text
╔══════════════════════════════════════════════════════════════╗
║                         TOKEN t                              ║
╚══════════════════════════════════════════════════════════════╝

                           x_t
                            │
                            ▼
                     token embedding
                            │
                            ▼
                        h_t^0
                            │
                            ▼

┌─────────────────────────────────────────────────────────────┐
│ GDN layer 1                                                 │
│                                                             │
│ input corrente:      h_t^0                                  │
│ stato dal passato:   C_(t-1)^1 , S_(t-1)^1                  │
│                                                             │
│ produce:             h_t^1                                  │
│ nuovo stato:         C_t^1 , S_t^1                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼

┌─────────────────────────────────────────────────────────────┐
│ GDN layer 2                                                 │
│                                                             │
│ input corrente:      h_t^1                                  │
│ stato dal passato:   C_(t-1)^2 , S_(t-1)^2                  │
│                                                             │
│ produce:             h_t^2                                  │
│ nuovo stato:         C_t^2 , S_t^2                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼

┌─────────────────────────────────────────────────────────────┐
│ GDN layer 3                                                 │
│                                                             │
│ input corrente:      h_t^2                                  │
│ stato dal passato:   C_(t-1)^3 , S_(t-1)^3                  │
│                                                             │
│ produce:             h_t^3                                  │
│ nuovo stato:         C_t^3 , S_t^3                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼

┌─────────────────────────────────────────────────────────────┐
│ Full Attention layer 4                                      │
│                                                             │
│ input corrente:      h_t^3                                  │
│ stato dal passato:   KV_(1:t-1)^4                           │
│                                                             │
│ produce:             h_t^4                                  │
│ nuova cache:         KV_(1:t)^4                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼

                  stesso pattern × 16

                             │
                             ▼
                        layer 64
                             │
                             ▼
                       Final RMSNorm
                             │
                             ▼
                          LM Head
                             │
                             ▼
                     248320 logits
                             │
                             ▼
                   sampling / argmax
                             │
                             ▼
                         x_(t+1)


              STATI CHE VENGONO CONSERVATI
              ═════════════════════════════

       48 × C_t^ℓ
       48 × S_t^ℓ
       16 × KV_(1:t)^m

                             │
                             ▼

╔══════════════════════════════════════════════════════════════╗
║                       TOKEN t+1                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

# 34. Un errore concettuale comune

Una descrizione troppo semplice di un Transformer dice:

```text
token precedente
      │
      ▼
64 layer
      │
      ▼
nuovo token
      │
      ▼
64 layer
```

Per Qwen3.6-27B questa rappresentazione è incompleta.

Quella più corretta è:

```text
                           nuovo token
                               │
                               ▼
                 ┌────────────────────────┐
                 │                        │
                 │     64-layer model     │
                 │                        │
                 └───────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
          nuovo token      48 stati       16 KV
                           DeltaNet        cache
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                      iterazione dopo
```

---

# 35. Perché combinare DeltaNet e Full Attention?

Le due memorie hanno proprietà complementari.

## DeltaNet

Vantaggi:

```text
memoria a dimensione fissa
+
calcolo lineare rispetto alla sequenza
+
stato ricorrente compatto
```

Ma il passato viene **compresso** nello stato.

Il modello non possiede necessariamente una copia esplicita
di ogni vecchio K/V in questi layer.

---

## Full Attention

Vantaggio:

```text
può accedere direttamente
alle rappresentazioni K/V
delle posizioni precedenti
```

ma:

```text
KV cache ∝ lunghezza contesto
```

---

## Architettura ibrida

Qwen usa:

```text
3 DeltaNet
+
1 Full Attention
```

così il modello combina:

```text
memoria ricorrente efficiente
             +
accesso esplicito periodico al passato
```

---

# 36. Dimensione dello stato ricorrente DeltaNet

Per un Gated DeltaNet:

```text
48 × 128 × 128
=
786432 valori
```

Per 48 layer:

```text
786432 × 48
=
37748736 valori
```

Nel percorso di riferimento della Gated Delta Rule di Transformers,
la computazione ricorrente viene portata in `float32`.

Se lo stato è mantenuto in FP32:

```text
37748736 × 4 byte
=
150994944 byte

≈ 144 MiB
```

per batch element, solo per le matrici `S`.

Questo valore rimane sostanzialmente costante rispetto a T.

---

# 37. Dimensione del Conv state

Per un GDN:

```text
10240 × 4
=
40960 valori
```

Per 48 GDN:

```text
40960 × 48
=
1966080 valori
```

È molto più piccolo dello stato matriciale `S`.

La sua funzione è soprattutto mantenere il piccolo contesto
necessario alla causal Conv1D.

---

# 38. Crescita della KV cache

Per un Full Attention layer e un token:

```text
K:
4 × 256 = 1024 valori

V:
4 × 256 = 1024 valori

totale:
2048 valori / token / layer
```

Con 16 Full Attention:

```text
2048 × 16
=
32768 valori / token
```

Quindi la sola KV cache dei 16 layer cresce linearmente:

```text
T token
→ 32768 × T valori
```

prima di considerare dtype, overhead del runtime, batching, ecc.

Contrasto:

```text
Delta state:
costante rispetto a T

KV cache:
lineare rispetto a T
```

---

# 39. Final RMSNorm e LM Head

Dopo il decoder layer 64:

```text
hidden:
[B,T,5120]
```

passa attraverso:

```text
Final RMSNorm
5120 → 5120
```

poi:

```text
LM Head
5120 → 248320
```

Quindi:

```text
logits:
[B,T,248320]
```

Per il decoding normalmente serve soprattutto l'ultima posizione:

```text
[B,248320]
```

ossia un logit per ciascun token del vocabolario.

Schema:

```text
hidden finale del token t
        [5120]
           │
           ▼
      Final RMSNorm
           │
           ▼
         [5120]
           │
           ▼
         LM Head
      5120 → 248320
           │
           ▼
       248320 logits
           │
           ▼
     softmax / sampling
           │
           ▼
      token t+1 scelto
```

---

# 40. Vision Encoder

Qwen3.6-27B è multimodale.

Il vision encoder ufficiale usa:

| Parametro | Valore |
|---|---:|
| Vision layers | 27 |
| Hidden size | 1152 |
| Attention heads | 16 |
| Head dimension | 72 |
| Intermediate FFN | 4304 |
| Input channels | 3 |
| Spatial patch | 16×16 |
| Temporal patch | 2 |
| Spatial merge | 2 |
| Output hidden | 5120 |

Poiché:

```text
1152 / 16
=
72
```

ogni attention head visuale ha dimensione 72.

---

# 41. Patch embedding visuale

Il patch embedding è realizzato con una Conv3D:

```text
input channels:
3

kernel:
2 × 16 × 16

stride:
2 × 16 × 16

output channels:
1152
```

Un patch contiene quindi:

```text
2 × 16 × 16 × 3
=
1536 valori di input
```

e viene trasformato in:

```text
1152 feature
```

Schema:

```text
2 frame temporali
×
16 × 16 pixel
×
3 canali RGB
       │
       ▼
     Conv3D
       │
       ▼
1152-dimensional
visual token
```

---

# 42. Vision Transformer

Sono presenti:

```text
27 Vision Transformer block
```

Ogni blocco:

```text
input 1152
   │
   ▼
LayerNorm
   │
   ▼
Vision Attention
16 head × 72
   │
   ▼
residual
   │
   ▼
LayerNorm
   │
   ▼
MLP
1152 → 4304 → 1152
   │
   ▼
residual
```

L'MLP visuale usa GELU.

---

# 43. Vision merger

Il merger spaziale usa:

```text
spatial_merge_size = 2
```

quindi combina:

```text
2 × 2 = 4
```

visual token.

Dato che ciascuno è 1152-dimensional:

```text
4 × 1152
=
4608
```

Il merger esegue concettualmente:

```text
4 visual token
4 × 1152
     │
     ▼
   4608
     │
     ▼
linear
4608 → 4608
     │
     ▼
   GELU
     │
     ▼
linear
4608 → 5120
     │
     ▼
visual token compatibile
con il language model
```

Questo è importante perché:

```text
vision output hidden
=
5120

language hidden
=
5120
```

I token visuali possono quindi entrare nello stesso decoder linguistico.

---

# 44. Pipeline multimodale semplificata

```text
                       IMMAGINE / VIDEO
                              │
                              ▼
                     patch extraction
                              │
                              ▼
                         Conv3D
                        → 1152
                              │
                              ▼
                     27 Vision blocks
                              │
                              ▼
                     spatial merger
                     4608 → 5120
                              │
                              │
                              ▼
                   visual embeddings
                        [*,5120]
                              │
                              │
                              ├───────────────┐
                              │               │
                              │               │
TEXT ──► token embedding ─────┘               │
          [*,5120]                            │
                                              ▼
                                sequenza multimodale
                                      hidden=5120
                                              │
                                              ▼
                                     64 decoder layer
                                              │
                                              ▼
                                          LM head
```

---

# 45. Quantizzazione Q4_K_S: cosa significa

Ora separiamo completamente due concetti:

```text
ARCHITETTURA
```

e:

```text
QUANTIZZAZIONE DEI PESI
```

La versione `Q4_K_S` mantiene la stessa architettura.

Non cambia:

```text
hidden_size             = 5120
num_layers              = 64
FFN                     = 17408
GDN layers              = 48
Attention layers        = 16
Q/K/V dimensions
S_t dimensions
Conv state dimensions
KV-cache dimensions logiche
vocabulary              = 248320
```

Cambia principalmente il **modo in cui i pesi statici vengono
memorizzati nel file GGUF**.

---

# 46. Pesi statici contro stato dinamico

Questa distinzione è fondamentale.

## Pesi statici

Esempi:

```text
embedding.weight

gate_proj.weight
up_proj.weight
down_proj.weight

q_proj.weight
k_proj.weight
v_proj.weight
o_proj.weight

in_proj_qkv.weight
in_proj_z.weight
in_proj_a.weight
in_proj_b.weight

out_proj.weight

lm_head.weight
```

Sono parametri appresi.

Non cambiano mentre il modello genera una risposta.

`Q4_K_S` riguarda principalmente questi tensori.

---

## Stato dinamico

Esempi:

```text
hidden state h_t

Q_t
K_t
V_t

DeltaNet:
C_t
S_t

Attention:
KV cache

logits
```

Queste quantità vengono create/modificate durante l'inferenza.

`Q4_K_S` **non significa automaticamente** che queste quantità
siano anch'esse memorizzate come `Q4_K`.

La precisione runtime di:

```text
KV cache
recurrent state
activation
```

dipende dal runtime e dalle sue opzioni.

---

# 47. Grafico: dove agisce Q4_K_S

```text
                        MODELLO SU DISCO
                         Q4_K_S GGUF
                              │
                              │
           ┌──────────────────┴──────────────────┐
           │                                     │
           ▼                                     ▼
    PESI QUANTIZZATI                       METADATI
           │
           │
           ▼
┌─────────────────────────────┐
│ embedding                   │
│ attention matrices          │
│ DeltaNet matrices           │
│ FFN matrices                │
│ output matrices             │
│ ...                         │
└──────────────┬──────────────┘
               │
               │ caricamento / dequantizzazione
               │ durante i kernel
               ▼
        CALCOLO INFERENZA
               │
        ┌──────┴────────┐
        │               │
        ▼               ▼
   activations      memoria runtime
                    ┌──────────────┐
                    │ Conv state   │
                    │ S_t          │
                    │ KV cache     │
                    └──────────────┘

Q4_K_S
   │
   └────► NON implica automaticamente
          "tutto il calcolo è a 4 bit".
```

---

# 48. Cos'è il primitive Q4_K

Il tipo di base del preset Q4_K_S è:

```text
GGML_TYPE_Q4_K
```

`Q4_K` usa super-block da:

```text
256 pesi
```

organizzati come:

```text
8 blocchi × 32 pesi
```

Per ciascun peso il formato rappresenta concettualmente:

```text
x = a × q + b
```

dove:

```text
q = valore quantizzato a 4 bit
a = scale
b = offset/minimum
```

---

# 49. Struttura di block_q4_K

Un super-block contiene:

```text
256 pesi
```

I quantizzati veri e propri occupano:

```text
256 × 4 bit
=
1024 bit
=
128 byte
```

Più:

```text
12 byte
per scale/min quantizzati

2 byte
super-block scale d

2 byte
super-block dmin
```

Totale:

```text
128
+ 12
+ 2
+ 2
=
144 byte
```

per:

```text
256 pesi
```

Quindi:

```text
144 × 8 / 256
=
4.5 bit per peso
```

per il primitive `block_q4_K`.

Grafico:

```text
Q4_K SUPER-BLOCK
256 pesi
144 byte totali

┌───────────────────────────────────────────────┐
│ quant values                                 │
│ 256 × 4 bit                                  │
│                                              │
│ 128 byte                                     │
├───────────────────────────────────────────────┤
│ packed scales + mins                         │
│                                              │
│ 12 byte                                      │
├──────────────────────┬────────────────────────┤
│ d                    │ dmin                   │
│ FP16                 │ FP16                   │
│ 2 byte               │ 2 byte                │
└──────────────────────┴────────────────────────┘

Totale:
144 byte / 256 valori

= 4.5 bpw
```

---

# 50. Perché Q4_K_S non equivale esattamente a 4.5 bpw

Il nome completo interno è:

```text
LLAMA_FTYPE_MOSTLY_Q4_K_S
```

La parola importante è:

```text
MOSTLY
```

Il preset di llama.cpp non impone necessariamente Q4_K
a ogni singolo tensor.

Alcuni tensori possono essere:

```text
Q5_K
Q6_K
F16/F32
o altri tipi appropriati
```

in funzione:

- della categoria del tensor;
- dell'architettura;
- della shape;
- della versione di llama.cpp;
- di eventuali override;
- dell'uso di importance matrix;
- degli argomenti del quantizzatore.

Quindi:

```text
Q4_K_S
≠
"ogni singolo parametro del file è Q4_K"
```

Più corretto:

```text
Q4_K_S
=
preset MOSTLY_Q4_K con strategia S
```

---

# 51. Alcune promozioni del preset Q4_K_S

Nel codice corrente di llama.cpp,
il tipo di base di `Q4_K_S` è `Q4_K`.

Esistono però regole che possono aumentare la precisione di
tensori considerati sensibili.

Fra le regole del preset corrente:

```text
alcuni attention-V tensor iniziali
Q4_K → Q5_K
```

e:

```text
una prima frazione dei FFN_DOWN tensor
Q4_K → Q5_K
```

Il punto da ricordare non è tanto memorizzare la singola eccezione,
quanto capire la struttura:

```text
             Q4_K_S
                │
                ▼
       tipo predefinito Q4_K
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
  tensor normali     tensor sensibili
      Q4_K           possono essere
                     promossi a Q5_K,
                     Q6_K, ecc.
```

Di conseguenza la dimensione media reale del GGUF può essere
superiore ai 4.5 bit/peso del singolo primitive Q4_K.

---

# 52. `S` in Q4_K_S

È importante non interpretare `S` come:

```text
Small model
```

o:

```text
State
```

o:

```text
Sparse
```

Nel contesto dei preset K-quant è semplicemente la variante
`S` del preset.

Esempi della famiglia:

```text
Q3_K_S
Q3_K_M
Q3_K_L

Q4_K_S
Q4_K_M

Q5_K_S
Q5_K_M
```

`Q4_K_S` è la variante più orientata alla dimensione rispetto
a `Q4_K_M`.

---

# 53. Q4_K_S non modifica lo stato DeltaNet

Questa distinzione merita di essere visualizzata.

```text
                         Q4_K_S
                            │
                            ▼
                   PESI DEL MODELLO
                            │
                            ▼
        ┌─────────────────────────────────┐
        │ W_Q, W_K, W_V                  │
        │ W_gate                         │
        │ W_up                           │
        │ W_down                         │
        │ DeltaNet projections           │
        │ embedding                      │
        │ output weights                 │
        └─────────────────────────────────┘


                    DURANTE L'INFERENZA
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
      Conv state           S_t             KV cache
    [10240,4]       [48,128,128]        [4,T,256]
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
                            ▼
              precisione determinata
              dal runtime/configurazione

NON:

Q4_K_S → S_t automaticamente Q4_K
```

---

# 54. Q4_K_S non modifica le equazioni del modello

BF16 e Q4_K_S rappresentano lo stesso modello matematico
approssimato con precisione diversa dei pesi.

## BF16

```text
h
 │
 ▼
W_BF16 · h
 │
 ▼
output
```

## Q4_K_S

```text
h
 │
 │
 ▼
peso memorizzato
in forma Q4_K/Q5_K/...
 │
 │ kernel quantizzato /
 │ dequantizzazione necessaria
 ▼
W_approx · h
 │
 ▼
output approssimato
```

Le shape restano le stesse:

```text
5120 → 17408
17408 → 5120

5120 → 10240
6144 → 5120

5120 → 12288
5120 → 1024
6144 → 5120
```

---

# 55. Riassunto completo delle matrici principali

## Embedding/output

| Componente | Input | Output |
|---|---:|---:|
| Token embedding | token ID | 5120 |
| Final RMSNorm | 5120 | 5120 |
| LM head | 5120 | 248320 |

---

## FFN — tutti i 64 layer

| Componente | Input | Output |
|---|---:|---:|
| gate_proj | 5120 | 17408 |
| up_proj | 5120 | 17408 |
| down_proj | 17408 | 5120 |

---

## Gated DeltaNet — 48 layer

| Componente | Input | Output |
|---|---:|---:|
| in_proj_qkv | 5120 | 10240 |
| Q | 5120 | 2048 |
| K | 5120 | 2048 |
| V | 5120 | 6144 |
| in_proj_z | 5120 | 6144 |
| in_proj_a | 5120 | 48 |
| in_proj_b | 5120 | 48 |
| Delta output | — | 6144 |
| out_proj | 6144 | 5120 |

---

## Full Attention — 16 layer

| Componente | Input | Output |
|---|---:|---:|
| q_proj | 5120 | 12288 |
| Q dopo split | — | 6144 |
| Output gate dopo split | — | 6144 |
| k_proj | 5120 | 1024 |
| v_proj | 5120 | 1024 |
| Attention output | — | 6144 |
| o_proj | 6144 | 5120 |

---

# 56. Mappa mentale finale

```text
Qwen3.6-27B
│
├── INPUT
│   │
│   └── vocab 248320
│       └── embedding → 5120
│
├── LANGUAGE MODEL
│   │
│   └── 64 layer
│       │
│       └── 16 ×
│           │
│           ├── Gated DeltaNet
│           │   ├── Conv state C
│           │   │   └── [10240,4]
│           │   │
│           │   └── recurrent state S
│           │       └── [48,128,128]
│           │
│           ├── Gated DeltaNet
│           │   ├── C
│           │   └── S
│           │
│           ├── Gated DeltaNet
│           │   ├── C
│           │   └── S
│           │
│           └── Full Attention
│               └── KV cache
│                   ├── K [4,T,256]
│                   └── V [4,T,256]
│
├── FFN IN OGNI LAYER
│   │
│   └── SwiGLU
│       ├── 5120 → 17408 gate
│       ├── 5120 → 17408 up
│       └── 17408 → 5120 down
│
├── OUTPUT
│   │
│   ├── Final RMSNorm
│   │   └── 5120
│   │
│   └── LM Head
│       └── 5120 → 248320 logits
│
├── VISION
│   │
│   ├── Conv3D patching
│   │   └── → 1152
│   │
│   ├── 27 Vision blocks
│   │   └── 16 heads × 72
│   │
│   ├── FFN
│   │   └── 1152 → 4304 → 1152
│   │
│   └── merger
│       └── 4608 → 5120
│
└── Q4_K_S
    │
    ├── non cambia l'architettura
    │
    ├── quantizza principalmente i pesi statici
    │
    ├── tipo di base = Q4_K
    │
    ├── Q4_K primitive:
    │   ├── 256 pesi/super-block
    │   ├── 8 × 32
    │   └── 4.5 bpw
    │
    ├── "MOSTLY Q4_K"
    │   └── alcuni tensor possono avere precisione maggiore
    │
    └── NON implica automaticamente:
        ├── KV cache Q4_K
        ├── recurrent state S Q4_K
        ├── Conv state Q4_K
        └── activation Q4_K
```

---

# 57. Il modello in una sola immagine concettuale

```text
                                     PASSATO
                                        │
             ┌──────────────────────────┴─────────────────────────────┐
             │                                                        │
             │                                                        │
      MEMORIA COMPRESSA                                        MEMORIA ESPLICITA
       Gated DeltaNet                                            Full Attention
             │                                                        │
             │                                                        │
     ┌───────┴─────────┐                                      ┌───────┴────────┐
     │                 │                                      │                │
     ▼                 ▼                                      ▼                ▼
Conv state          S state                                  K cache          V cache
[10240,4]       [48,128,128]                              [4,T,256]        [4,T,256]
     │                 │                                      │                │
     └────────┬────────┘                                      └───────┬────────┘
              │                                                       │
              └──────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
                                TOKEN CORRENTE
                                     x_t
                                      │
                                      ▼
                                  embedding
                                     5120
                                      │
                                      ▼
                        ┌────────────────────────┐
                        │                        │
                        │  64 DECODER LAYER      │
                        │                        │
                        │  48 GDN + 16 Attn      │
                        │                        │
                        └───────────┬────────────┘
                                    │
                                    ▼
                                  5120
                                    │
                                    ▼
                               Final RMSNorm
                                    │
                                    ▼
                                 LM Head
                            5120 → 248320
                                    │
                                    ▼
                              248320 logits
                                    │
                                    ▼
                                  sample
                                    │
                                    ▼
                                  x_t+1
                                    │
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
             nuovo token                   stati aggiornati
                                            C_t, S_t, KV_t
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                             ITERAZIONE t+1
```

---

# 58. Frase conclusiva tecnicamente precisa

Qwen3.6-27B può essere descritto così:

> **Un decoder causale autoregressivo ibrido di 64 layer, nel quale
> 48 layer utilizzano Gated DeltaNet con stato ricorrente a dimensione
> costante e causal Conv1D, mentre 16 layer utilizzano causal Grouped
> Query Attention con KV cache crescente. Ogni layer contiene un
> SwiGLU FFN 5120→17408→5120. Nella variante GGUF Q4_K_S
> l'architettura non cambia: viene principalmente modificata la
> rappresentazione dei pesi statici tramite un preset MOSTLY_Q4_K,
> mentre recurrent state, convolution state e KV cache sono stati
> runtime distinti e non diventano automaticamente Q4_K.**

In forma matematica molto compatta:

```text
OUTPUT AUTOREGRESSIVO:

x_(t+1)
~
P(
    x_(t+1)
    |
    x_≤t
)


MEMORIA GATED DELTANET:

C_t = ConvStateUpdate(C_(t-1), x_t)

S_t =
exp(g_t) S_(t-1)
+
k_t [
    β_t(
        v_t
        -
        k_tᵀ exp(g_t) S_(t-1)
    )
]ᵀ


OUTPUT DEL DELTANET:

o_t =
q_tᵀ S_t


MEMORIA FULL ATTENTION:

KV_t =
concat(
    KV_(t-1),
    K_t,
    V_t
)


MODELLO COMPLETO:

x_t
+
{C_(t-1)^ℓ, S_(t-1)^ℓ}_{48 GDN}
+
{KV_(1:t-1)^m}_{16 Attention}

        ↓

64 decoder layer

        ↓

logits_t ∈ R^248320

        ↓

x_(t+1)

        +

nuovi stati:
{C_t^ℓ,S_t^ℓ}
e
{KV_(1:t)^m}
```

---

# 59. Fonti primarie di riferimento

I dati architetturali vanno verificati principalmente in:

1. checkpoint/config ufficiale:
   `Qwen/Qwen3.6-27B`

2. implementazione Hugging Face:
   `transformers/models/qwen3_5/modeling_qwen3_5.py`

3. implementazione cache:
   `transformers/cache_utils.py`

4. implementazione quantizzazione:
   `ggml-org/llama.cpp/src/llama-quant.cpp`

5. struttura dei K-quants:
   `ggml-org/llama.cpp/ggml/src/ggml-common.h`

## Nota importante sulla certezza di Q4_K_S

`Q4_K_S` definisce un preset di quantizzazione, non la lista
immutabile dei tipi di ogni tensor di qualsiasi file GGUF esistente.

Due file denominati:

`Qwen3.6-27B-Q4_K_S.gguf`

possono differire se chi li ha prodotti ha usato, per esempio:

- importance matrix;
- override per tensor;
- diversa versione di llama.cpp;
- opzioni specifiche per embedding/output;
- conversione multimodale differente.

Per sapere con certezza il dtype GGML di **ogni singolo tensor**
di uno specifico file Q4_K_S è necessario ispezionare proprio
quel file GGUF.

L'architettura matematica del Qwen3.6-27B, invece,
rimane quella descritta sopra.