# Qwen3.6 MTP implementation

## Scope and contract

DS4 supports native Qwen3.6 Multi-Token Prediction on the single-GPU CUDA
path with:

- target `Qwen3.6-27B-Q4_K_S.gguf` or `Qwen3.6-27B-Q4_K_M.gguf`;
- support `mtp-Qwen3.6-27B-Q4_0.gguf`, one NextN block at `blk.64`.

MTP is only a drafter. The target model is authoritative for every committed
token. Greedy MTP output must remain target-greedy at every committed frontier,
and stochastic MTP with a fixed seed must remain identical to ordinary target
sampling. Any verifier, rollback, sampler, or support-cache failure must not
expose speculative state as accepted state.

Qwen is configured with draft depth two. Below 2K occupied tokens the cycle
uses both drafts (target V(3)); from 2K onward it caps the current cycle to one
draft (target V(2)), because verifier attention grows with the occupied KV.
The implementation never widens depth adaptively and hard-caps Qwen at four.

The full upstream audit, source links, rejected experiments, and benchmark
evidence are in
[`../research/guides/qwen36-mtp-llamacpp-ds4-control-flow.md`](../research/guides/qwen36-mtp-llamacpp-ds4-control-flow.md).

## GGUF and NextN semantics

The support model is selected structurally, not by filename:

1. `general.architecture=qwen35`;
2. `qwen35.nextn_predict_layers=1` and 65 total blocks;
3. tokenizer, vocabulary, and shape metadata match the target;
4. the support file supplies the expected top-level shared tensors and the
   18-tensor `blk.64` layout.

At position `p`, NextN consumes `(token[p], h_target[p-1])`. The target hidden
is the normalized final `h_nextn` row, matching llama.cpp's Qwen graph:

```text
e = RMSNorm(Embedding(token[p]), nextn.enorm)
h = RMSNorm(h_target[p-1],       nextn.hnorm)
u = nextn.eh_proj(concat(e, h))

a = u + GatedFullAttention(RMSNorm(u), mtp_KV, p)
y = a + SwiGLU(RMSNorm(a))

h_mtp  = RMSNorm(y, nextn.shared_head_norm)
logits = shared_output(h_mtp)
```

Recursive proposals temporarily use the preceding MTP hidden. After target
verification, the support frontier is overwritten using authoritative target
hidden rows.

## Session state

Each enabled Qwen session owns:

- support-block scratch and independent full-attention K/V;
- one normalized target-hidden carry row and two MTP hidden ping-pong rows;
- a bounded verifier logits buffer and GPU row argmax output;
- pre-batch safety snapshots of the 48 Gated DeltaNet states for V(3);
- per-verifier-row GDN snapshots used for direct partial rollback;
- a logical MTP cache frontier;
- aggregate and positional timing/acceptance counters.

Full-attention target KV is position-addressed, so future rejected rows are
left invisible and overwritten. GDN state is recurrent and must be restored to
the exact accepted row.

## Production adaptive-depth cycle

Below 2K, the configured depth-two cycle is:

1. sample `first_token` from the current target logits;
2. draft `draft0` and `draft1` autoregressively from the support model;
3. evaluate `[first_token, draft0, draft1]` as one target V(3) batch;
4. greedy decode compares `row_top[0]` with `draft0` and `row_top[1]` with
   `draft1`; stochastic decode samples those full target rows in order and
   compares the samples with the corresponding drafts;
5. on full acceptance, keep the live state after row 2;
6. on zero acceptance, restore the GDN snapshots after row 0 and select row-0
   logits;
7. on one accepted draft, restore the snapshots after row 1 and select row-1
   logits;
8. catch the support K/V up to the accepted frontier using the shifted target
   hidden rows;
9. commit `first_token` and the accepted draft prefix;
10. for stochastic decode, carry the sampled mismatch token, or the sample from
    the final bonus row after full acceptance, into the next cycle.

The row snapshots are written by the convolution and GDN kernels while the
states are already in registers. The final row is not copied because it is the
live full-accept state. Partial acceptance does not replay a target token.

At 2K and above, the same cycle verifies only `[first_token, draft0]` as V(2).
The row-0 snapshot is sufficient for an ordinary rejection, so V(2) omits the
redundant pre-batch copy of every recurrent state. A batch/backend failure
invalidates the session instead of exposing speculative state;
`DS4_MTP_QWEN_V2_SAFE_SNAPSHOT=1` restores the diagnostic safety copy.
`DS4_MTP_QWEN_FORCE_DRAFT2=1` forces V(3), while
`DS4_MTP_QWEN_DRAFT1=1` forces V(2), for resident A/B tests.

`DS4_MTP_SPLIT_TARGET=1` retains the older `target_seed + verifier` layout as a
diagnostic switch for greedy runs. Stochastic verification always uses the
fused layout because every verifier row is required by the sampler.

## Sampling and logits masks

DS4 follows llama.cpp's current sample-and-match contract. The MTP sidecar
proposes its top token. Temperature, top-k, top-p, and min-p are applied once to
each authoritative target row, not to the sidecar logits. A matching target
sample accepts the proposal; the first mismatch is already the next output
token. Full acceptance samples the final target row as a pending bonus token.

Applying the ordinary sampling mask to the MTP logits cannot replace target
verification. Top-k/top-p/min-p retain the proposal's own top token by
construction, while the target distribution can retain a different set. The
target filter and RNG draw are therefore still required to preserve the target
sampling distribution; masking both models would add work without removing the
authoritative check.

Grammar, JSON-schema, and tool masks are different because they encode external
state. They could reject impossible MTP proposals early, but the mask must be
advanced after every accepted token and the target row must still be filtered.
Until that state can be checkpointed and advanced inside a speculative batch,
stochastic MTP is disabled for tool and response-schema requests. Greedy
constrained phases retain the established behavior.

## CUDA paths

### Target verifier

Q4_K, Q5_K, and Q6_K verifier matmuls use Q8_1 plus a second Q8_1 residual.
For two or three rows:

- one activation-quantization grid covers every contiguous row;
- one MMVQ CTA keeps all token columns in registers;
- weights are decoded once and applied to every token column;
- two output rows share a four-warp CTA;
- only lane zero publishes results because DS4's `warp_sum_f32` is a
  `shfl_down` reduction.

The last rule is a correctness requirement. Publishing the second output row
from lane one caused partial sums to enter target verification. The model-backed
raw-copy regression exists specifically to prevent that failure from returning.

The established Q8_1+residual representation remains mandatory. A plain-Q8
experiment was faster but failed the full quality corpus and is not present in
the production branch.

### Support network

The Q4_0 shared head uses Q8_1+residual DP4A and schedules eight output rows per
CTA. Proposal argmax stays on GPU. Support catch-up is K/V-only: it evaluates
embedding and the input projections needed for K/V, applies K norm and RoPE,
and skips Q, attention, attention output, FFN, normalization, and LM head.

## Failure policy

- Invalid or incompatible support GGUF fails engine creation with a structural
  diagnostic.
- Proposal failure before target mutation falls back to the mandatory target
  token and disables future MTP use for the session.
- Target evaluation or rollback failure invalidates the session.
- Support catch-up failure preserves committed target state and disables MTP
  for later cycles.
- TP, distributed, and unsupported streaming ownership combinations do not
  silently share MTP state.

## Validation gate

The production model-backed command is:

```sh
DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q4_K_S.gguf \
DS4_TEST_MTP=gguf/mtp-Qwen3.6-27B-Q4_0.gguf \
DS4_TEST_MTP_CTX=768 \
DS4_TEST_MTP_PREFILL_CHUNK=16 \
DS4_TEST_QWEN_MTP_PATHS=1 \
DS4_MTP_STATS=1 \
./ds4_test --mtp-verify-depth
```

It covers natural full acceptance, forced reject, forced partial acceptance,
raw-copy teacher-forced replay, maximum returned chunk, fixed-seed stochastic
sampling, and MTP-on/off prompt logits. The final RTX 3090 Q4_K_S run passed
with zero fallback and zero replay; all committed-token argmax gaps were 0.000,
the 248,320 prompt logits were bit-exact with MTP disabled, and all 128 sampled
token IDs matched the target-only run. The stochastic run returned chunks up to
three tokens, so it exercised speculation rather than the one-token fallback.

The simple repeated-copy benchmark measured 30.48 token/s target-only versus
46.39 token/s with `--mtp` in the CLI (1.52x). A real server run measured
median generation throughput 31.47 versus 51.07 token/s (1.62x). A less
predictable natural-copy prompt accepted only 63.1% of drafts and measured
1.12x, so acceptance remains workload-dependent.

With `temperature=1`, `top_p=1`, `min_p=0.05`, and seed 123, the same 128-token
CLI workload measured 31.64 token/s target-only versus 48.74 token/s with MTP
(1.54x), with byte-identical output and 84/87 accepted proposals. A real server
request using those sampled settings measured 34.55 versus 50.15 token/s decode
(1.45x); server-internal end-to-end time fell from 4.526 s to 3.391 s (1.34x).

The final `--ctx 32768` server validation also covered DSML. With tools,
`temperature=0.7`, and a fixed seed, target-only decode measured 25.95 token/s
and MTP measured 37.75 token/s (1.45x); both returned the same valid
`list_files` call. A 128-token natural sampled request measured 34.76 versus
41.97 token/s (1.21x), with the same output hash and no fallback or replay.

The 24 GiB memory fix does not reduce the prefill chunk or disable a fast
kernel. At depth two, recurrent rollback storage now reserves two row
frontiers instead of the hard-cap maximum of five (about 454.5 MiB saved), and
verifier logits reserve three rows instead of sixteen (about 12.3 MiB). Qwen's
mutually exclusive recurrent/full-attention projection buffers share one
workspace (224 MiB at 4096 rows), and the element-wise SwiGLU output reuses its
dead gate input (272 MiB). The total persistent saving is about 963 MiB with
an explicit 4096-row prefill and about 715 MiB with the existing 2048-row
Qwen default; the optimization does not change that default. Startup memory
accounting now includes Qwen's real full-context attention KV, graph
workspaces, MTP state, and support-model weights; dynamic accelerator caches
and driver allocations remain explicitly labelled as excluded.

For context, the audited local llama.cpp binary measured 40.9 versus 72.8
token/s (1.78x) at depth two on the same GPU, models, source prompt, and output
length. Its target-only and MTP outputs were byte-identical except for the
timing line. This is an end-to-end directional comparison rather than an
isolated kernel comparison because llama.cpp and DS4 use different frontend
prompt construction. DS4 remains about 25.5% slower target-only and 36.3%
slower with MTP in those runs.

Short prefill was not changed. In the server gate it remained about 4.9
seconds in both modes; generation, not prefill, was the MTP optimization target.

Long-context verification now shares the target-only split-K attention path.
The split partials and merge are indexed by verifier row, so the fused V(3)
batch enters split-K at the same occupied-context position as ordinary decode.
An RTX 3090 serial/split bisection selected 96 tokens as the common automatic
crossover (64 was tied for MTP). A second resident search selected V(3) below
2K and V(2) from 2K; the MTP context allocation remains capped below 30K in
the 0–28K harness suite.
