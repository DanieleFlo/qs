# Qwen3.6 MTP implementation project

## Scope and correctness contract

This project adds native Qwen3.6 27B Multi-Token Prediction (MTP) to the
single-GPU CUDA path. The supported pair is:

- target: `Qwen3.6-27B-Q4_K_S.gguf` or `Qwen3.6-27B-Q4_K_M.gguf`
  (64-layer Qwen35 trunk);
- support: `mtp-Qwen3.6-27B-Q4_0.gguf` (one NextN block at `blk.64`).

MTP is only a drafter. The target model remains the sole authority for every
committed token. With greedy sampling, enabling MTP must produce the same token
stream and the same target logits at every committed frontier as normal Qwen
decoding. A support-model error disables speculation for the cycle; it must
never turn a draft into an accepted token.

The original implementation deliberately used the normal one-token Qwen target
path as its verifier. That established the correctness baseline, but could not
accelerate decoding because every accepted draft still paid one complete target
decode. The production CUDA path now verifies a short suffix layer-major, owns
bounded snapshots of the 48 recurrent Gated DeltaNet states, and replays only
the accepted prefix after a partial acceptance. Full-attention future KV rows
need no snapshot because they are position-addressed and overwritten before
they become visible again.

## Evidence from LM Studio, Unsloth and llama.cpp

LM Studio and Unsloth do not provide an independent Qwen MTP compute kernel.
Both expose the implementation in their llama.cpp runtime:

- Unsloth detects the authoritative GGUF key
  `<arch>.nextn_predict_layers > 0` and enables `--spec-type draft-mtp`.
  Its GPU preset permits up to six draft tokens, but this is scheduling policy,
  not a property of the single trained NextN block. See
  [Unsloth PR #5527](https://github.com/unslothai/unsloth/pull/5527).
- LM Studio issue reports and launch examples show the same llama.cpp
  `draft-mtp` path. They also show why loader detection and hybrid-cache
  rollback must be explicit rather than inferred from a filename. See
  [LM Studio issue #1941](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1941)
  and [LM Studio issue #1597](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1597).
- The actual dense-Qwen graph is implemented in
  [llama.cpp `qwen35.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp),
  while proposal state and target-hidden alignment live in
  [llama.cpp `common/speculative.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp).
  The original integration history is
  [llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673).
- llama.cpp states the essential performance condition directly: draft tokens
  are useful when the target scores them in one batch rather than in sequential
  decode calls. Its current controls also expose a draft ceiling and a minimum
  draft probability rather than assuming one fixed depth is optimal. See
  [llama.cpp speculative decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).
- Hybrid Qwen rollback is not ordinary KV truncation. Upstream work explicitly
  adds partial removal for recurrent GDN memory; without that capability the
  target context rejects speculative decoding. See
  [llama.cpp issue #20039](https://github.com/ggml-org/llama.cpp/issues/20039)
  and the related partial-removal work referenced there.

Two upstream failures shape the DS4 gates. The target embedding, rather than an
MTP-produced approximation, must be fed back after verification; otherwise
acceptance degrades over long runs. Also, draft widths must never change a
greedy stream, as investigated in
[llama.cpp issue #23335](https://github.com/ggml-org/llama.cpp/issues/23335).

The primary literature gives the same systems constraint. Speculative decoding
is exact and faster only when scoring a short continuation has latency close to
one target step, not `k` target steps. See
[Leviathan, Kalman and Matias (2023)](https://arxiv.org/abs/2211.17192) and
[Chen et al. (2023)](https://arxiv.org/abs/2302.01318). Multi-token training can
raise draft quality, but it does not remove verifier or hardware costs; see
[Gloeckle et al. (2024)](https://arxiv.org/abs/2404.19737). This distinction is
important here: a good Qwen NextN head is necessary, while a true target
microbatch and cheap rollback are what turn acceptance into throughput.

## Audited GGUF structure

The repository snapshots in
`gguf-tools/quality-testing/data/qwen36-metadata/` are the loader oracle.

| Property | Target Q4_K_S/Q4_K_M | MTP support Q4_0 |
| --- | ---: | ---: |
| Architecture | `qwen35` | `qwen35` |
| Main/total blocks | 64 | 65 |
| `nextn_predict_layers` | absent | 1 |
| Tensor count | 851 | 18 |
| Vocabulary | 248,320 | 248,320 |
| Hidden width | 5,120 | 5,120 |
| Attention heads / KV heads | 24 / 4 | 24 / 4 |
| Head width | 256 | 256 |
| Dense FFN width | 17,408 | 17,408 |

The support file contains shared `token_embd`, `output`, and `output_norm`
tensors plus the following `blk.64` tensors:

- gated full attention: Q, K, V, Q/K norms, attention norm and output;
- dense SwiGLU FFN: gate, up, down and post-attention norm;
- NextN adapter: `nextn.enorm`, `nextn.hnorm`, `nextn.eh_proj` and
  `nextn.shared_head_norm`.

There is exactly one trained NextN block. It can be applied autoregressively
more than once, so configured draft depth and trained stage count are distinct.
The support file omits `nextn.embed_tokens` and `nextn.shared_head_head`; the
top-level support embedding and output tensors are therefore the required
fallbacks.

## Kernel semantics

For target position `p`, let `x_p` be its token and let `h_(p-1)` be the
target's normalized final hidden row from the preceding position. One MTP step
is:

```text
e = RMSNorm(Embedding(x_p), nextn.enorm)
h = RMSNorm(h_(p-1),       nextn.hnorm)
u = nextn.eh_proj([e ; h])

a = u + Attention(RMSNorm(u, attn_norm), mtp_KV, position=p)
y = a + ffn_down(SiLU(ffn_gate(RMSNorm(a))) * ffn_up(RMSNorm(a)))

h_mtp  = RMSNorm(y, nextn.shared_head_norm)
logits = output(h_mtp)
```

Attention is the Qwen35 full-attention block: the Q projection contains query
and sigmoid gate channels; Q and K receive per-head RMS normalization and
partial multi-section RoPE; causal attention is gated before the output
projection. The MTP cache contains only this one full-attention layer. It is
independent of the target's 16 full-attention caches and 48 Gated DeltaNet
states.

Alignment is the subtle invariant. The pair at MTP position `p` is
`(x_p, h_(p-1))`, not `(x_p, h_p)`. During recursive drafting, the next step
temporarily uses the preceding MTP hidden row. After target verification, DS4
overwrites accepted MTP positions using target hidden rows; rejected future
rows become invisible by reducing the logical MTP frontier and are overwritten
on the next cycle.

## DS4 architecture

### Loader and binding

Add a Qwen NextN support kind selected by structure, never filename:

1. require `general.architecture=qwen35`;
2. require `qwen35.nextn_predict_layers=1` and `qwen35.block_count=65`;
3. validate tokenizer/vocabulary and all shape-defining metadata against the
   already loaded target;
4. require the exact 18-tensor support layout and accepted Q4_0/F32 types;
5. bind `blk.64` without changing the global 64-layer target shape.

The normal target loader stays unchanged. Loading the target with MTP disabled
therefore traverses the exact existing Qwen Q4_K_S/Q4_K_M path.

### Session state

Each Qwen MTP session owns:

- support-block scratch tensors;
- K/V cache for `blk.64` with the same context capacity as the target;
- a normalized target-hidden carry row;
- two MTP hidden rows for autoregressive ping-pong;
- a 16-row verifier-logit buffer and GPU argmax output;
- snapshots of only the 48 GDN convolution and recurrent states (about 157 MiB
  for Qwen3.6 27B); the 16 full-attention caches are not copied;
- a logical cache frontier;
- counters for cycles, proposed, verified, accepted, full/partial/rejected
  cycles and proposal/verification time.

The state is reset with the Qwen target state. Prompt synchronization rebuilds
the support KV using shifted target hidden rows. Prefix extension continues
from the live frontier; any non-prefix sync resets both target and support
state.

### Proposal and verification cycle

1. Save the normalized target hidden row for the current committed frontier.
2. Evaluate the caller's normal target token. This commits the mandatory first
   token exactly as non-MTP decoding does.
3. Run `blk.64` on `(first_token, saved_hidden)` and greedily propose a suffix;
   recursive proposals use the preceding MTP hidden row.
4. Verify the first draft from the already resident target logits. A first-token
   miss avoids both snapshot and target batch.
5. Snapshot the recurrent GDN states and evaluate the draft suffix in one
   layer-major target microbatch. Produce all row argmaxes on GPU and copy only
   the last full-vocabulary row needed by the next sampling frontier.
6. On full acceptance, keep the resulting target state. On a partial
   acceptance, restore the recurrent snapshot and replay only the accepted
   prefix. Position-addressed full-attention KV rows beyond the logical
   frontier remain invisible and are overwritten later.
7. Rebuild/overwrite the support KV for accepted positions using target hidden
   rows, then cap its logical frontier at the accepted prefix.
8. Return the mandatory target token plus accepted drafts. EOS, output limit,
   accepted-buffer capacity and context capacity cap the cycle before every
   write.

If snapshot or microbatch setup fails before a safe mutation, DS4 retains the
one-token correctness path. If rollback itself fails, the session is
invalidated rather than exposing speculative recurrent state.

## Failure and fallback policy

- Invalid or incompatible support GGUF: fail engine creation with a structural
  diagnostic.
- Runtime proposal failure before target mutation: keep the mandatory target
  token and disable drafts for that cycle.
- Target evaluation failure: invalidate the session exactly as normal decode.
- MTP catch-up failure after a committed target token: invalidate MTP state,
  continue future target-only decoding, and report the fallback.
- TP, distributed and SSD-streaming combinations remain rejected or target-only
  until their support state has an explicit owner and rollback protocol.

## Test plan and release gates

### Offline and unit tests

- detect the 18-tensor Qwen support model and reject missing/wrong metadata,
  tensor, dimension or type;
- validate shifted `(token, hidden)` alignment;
- acceptance helper cases: full, zero and partial acceptance;
- logical MTP rollback after rejection;
- EOS, stop/output cap and context boundary;
- proposal failure leaves target acceptance unchanged;
- stats never report more accepted than verified or proposed tokens.

### Model-backed CUDA tests on RTX 3090

Use `Qwen3.6-27B-Q4_K_S.gguf` for the strict correctness gate and repeat the
performance gate with `Qwen3.6-27B-Q4_K_M.gguf`, matching the deployment model.

1. Run the same prompt with MTP disabled and enabled at greedy temperature 0.
2. Compare every committed token and full-vocabulary target logits after every
   cycle; any divergence is a hard failure.
3. Exercise draft depths 1, 2 and 6, plus forced full/partial/reject fixtures
   where practical.
4. Repeat across EOS, explicit stop sequence, one-token context room and
   support failure/fallback.
5. Record cycles, proposed/verified/accepted tokens, acceptance rate, target
   decode time, proposal time and net tokens/s. Performance never relaxes the
   equivalence gate.

### Regression

Run the normal build and unit suite. The Qwen target-only Q4_K_S/Q4_K_M paths
must be unchanged. Inspect CUDA compilation and run the existing Qwen numerical tests.
Metal, ROCm, SSD streaming and distributed paths receive no new execution
branch; their build surfaces still need validation in the release matrix.

## Why the correctness baseline was slower

Let `T` be one target decode, `D(k)` the cost of drafting `k` tokens, `V(k)`
the target microbatch verifier, and `A` the accepted draft count. The original
sequential cycle cost approximately

```text
T_sequential = T + D(k) + A*T + catchup(A)
tokens        = 1 + A
```

It therefore performed the same number of target passes as ordinary decoding,
then added the drafter, full-vocabulary PCIe readback for every proposal, and
support-cache catch-up. Acceptance could improve amortization of application
overhead but could not remove target work. On the RTX 3090 reproducer this was
18.54 token/s target-only versus 15.87 token/s with depth four: a 14.4% loss.
The MTP counters localized 3.755 s of a 6.039 s generation window in 68 serial
target verifications, with 61.8% draft acceptance.

The fast cycle instead has the approximate cost

```text
T_batched = T + D(k) + V(k)
          + P(partial) * (restore + replay(A)) + catchup(A)
tokens    = 1 + A
```

This exposes the actual break-even rule: `V(k)` must be much smaller than
`k*T`, the drafter must be cheap, and the chosen depth must not create enough
partial replays to consume the saved target work. A first implementation of the
batched graph still failed this rule because the CUDA Q4_K/Q5_K/Q6_K dispatcher
used its generic small-prefill kernel for 2--4 rows. That kernel reread and
dequantized weights independently for every token and produced only 7.25
token/s. The final microbatch kernel loads each quantized value once, applies it
to a specialized two- or four-token tile, and preserves the established F32
accumulation order independently for each target row.

The NextN Q4_0 drafter had the analogous launch problem: one 256-thread block
per output row. Its warp-8 kernel now schedules eight output rows per block.
Draft reduction order may differ because proposals are advisory, but every
committed token remains target-verified. Proposal argmax also stays on GPU, so
each step returns four bytes instead of the complete 248,320-float vocabulary.

Finally, `--mtp-draft` is treated as a ceiling. The scheduler starts at depth
two, widens to at most four only after eight consecutive full accepts, and
returns to two immediately on a partial or rejected cycle. This follows the
measured and upstream observation that optimal depth depends on acceptance,
context, model and hardware; a larger fixed ceiling can be slower even with a
competent head. llama.cpp reports the same failure mode when draft evaluation
dominates or acceptance falls; see
[issue #23752](https://github.com/ggml-org/llama.cpp/issues/23752) and its
[adaptive-depth discussion](https://github.com/ggml-org/llama.cpp/discussions/23738).
Loading a structurally valid Qwen NextN sidecar therefore defaults to ceiling
two when the caller leaves the legacy value at one; omitting `--mtp` remains
the unambiguous way to request target-only decoding.

For `ds4` and `ds4-server`, `--mtp` takes no value and loads
`gguf/mtp-Qwen3.6-27B-Q4_0.gguf`. Use `--mtp-model FILE` only to override that
known sidecar path. For example:

```sh
./ds4 -m gguf/Qwen3.6-27B-Q4_K_S.gguf --mtp --temp 0
./ds4-server -m gguf/Qwen3.6-27B-Q4_K_M.gguf --mtp --ctx 32768
```

## Implemented fast path and measured gates (2026-08-07)

The loader and baseline remain in `ds4.c`, with simultaneous target/support
residency in `ds4_cuda.cu`. The optimized implementation adds:

- a layer-major target verifier returning row argmaxes on device;
- bounded GDN snapshot/restore plus accepted-prefix replay;
- row-wise target-hidden catch-up for the MTP cache;
- exact F32 Q4_K/Q5_K/Q6_K microbatch warp-8 kernels for 2--4 rows;
- a Q4_0 warp-8 drafter kernel and backend proposal argmax;
- conservative adaptive depth with a user-provided maximum.

Production Qwen remains single-GPU CUDA and continues to reject TP,
distributed and SSD-streaming combinations.

The model-backed gate used:

```sh
DS4_TEST_MODEL=gguf/Qwen3.6-27B-Q4_K_S.gguf \
DS4_TEST_MTP=gguf/mtp-Qwen3.6-27B-Q4_0.gguf \
DS4_TEST_MTP_CTX=768 \
DS4_TEST_MTP_PREFILL_CHUNK=16 \
DS4_TEST_QWEN_MTP_PATHS=1 \
DS4_MTP_STATS=1 \
./ds4_test --mtp-verify-depth
```

On the local RTX 3090 build (`sm_86`) the updated gate passed with:

- natural full acceptance: 182/182 drafts, 50/50 full cycles, maximum returned
  chunk 5, replay worst argmax gap 0.000, 25.138 net committed token/s;
- forced full rejection: 0/61 drafts, 31 rejected cycles, maximum chunk 1,
  replay worst argmax gap 0.000;
- forced partial acceptance: 16/31 drafts, 15 partial cycles plus one naturally
  full tail cycle, maximum chunk 2, replay worst argmax gap 0.000;
- MTP-on versus target-only prefill: all 248,320 float logits bit-exact, with
  the same top-1 token 71093;
- no runtime fallback in any of the three speculative runs.

The short Q4_K_M reproduction used the same 33-token rendered prompt, context
5000, greedy temperature zero and 96 generated tokens. On the final binary it
measured 18.13 token/s target-only and 19.17 token/s with the Q4_0 sidecar and
`--mtp-draft 4` (a ceiling; effective depth was two for this mixed-confidence
sample), a 5.7% speedup. The drafter time fell from 741 ms in the old depth-four
run to 212 ms, while all 96 committed tokens remained identical in the shown
output. This is a passing hardware-specific gate, not a universal speed claim:
long-context and multi-prompt sweeps remain necessary when changing kernels or
scheduler thresholds.

Dedicated full-vocabulary comparisons at every speculative frontier and the
EOS/explicit-stop/context-boundary/runtime-fallback matrix remain release
gates; they are not implied by the passing tests above.

## Remaining optimization boundary

The largest remaining penalty is partial acceptance: restore plus accepted-
prefix replay is intentionally conservative. Future work may capture a bounded
prefix state inside the GDN row kernel, or use confidence from the draft top-2
to avoid low-value second proposals. Either change must preserve target logits,
the shifted `(token, hidden)` alignment, and the exact rollback gates above.
