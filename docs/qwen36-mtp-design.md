# Qwen3.6 MTP implementation project

## Scope and correctness contract

This project adds native Qwen3.6 27B Multi-Token Prediction (MTP) to the
single-GPU CUDA path. The supported pair is:

- target: `Qwen3.6-27B-Q4_K_S.gguf` (64-layer Qwen35 trunk);
- support: `mtp-Qwen3.6-27B-Q4_0.gguf` (one NextN block at `blk.64`).

MTP is only a drafter. The target model remains the sole authority for every
committed token. With greedy sampling, enabling MTP must produce the same token
stream and the same target logits at every committed frontier as normal Qwen
decoding. A support-model error disables speculation for the cycle; it must
never turn a draft into an accepted token.

The first implementation deliberately uses the normal one-token Qwen target
path as its verifier. This makes rejection side-effect free and establishes a
small correctness baseline before a later batched verifier adds recurrent-state
snapshots. It does not claim a speedup: proposal and acceptance metrics are
reported separately from target equivalence.

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

Two upstream failures shape the DS4 gates. The target embedding, rather than an
MTP-produced approximation, must be fed back after verification; otherwise
acceptance degrades over long runs. Also, draft widths must never change a
greedy stream, as investigated in
[llama.cpp issue #23335](https://github.com/ggml-org/llama.cpp/issues/23335).

## Audited GGUF structure

The repository snapshots in
`gguf-tools/quality-testing/data/qwen36-metadata/` are the loader oracle.

| Property | Target Q4_K_S | MTP support Q4_0 |
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
therefore traverses the exact existing Qwen Q4_K_S path.

### Session state

Each Qwen MTP session owns:

- support-block scratch tensors;
- K/V cache for `blk.64` with the same context capacity as the target;
- a normalized target-hidden carry row;
- two MTP hidden rows for autoregressive ping-pong;
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
4. For each proposal, compare it with the current target argmax before target
   evaluation. Stop at the first mismatch.
5. Evaluate only matching proposals with the ordinary one-token Qwen path.
   This verifier cannot contaminate target state with a rejected token.
6. Rebuild/overwrite the support KV for accepted positions using target hidden
   rows, then cap its logical frontier at the accepted prefix.
7. Return the mandatory target token plus accepted drafts. EOS, output limit,
   accepted-buffer capacity and context capacity cap the cycle before every
   write.

This sequential verifier is the correctness baseline. A future fast verifier
may call the existing multi-row Qwen path only after it can snapshot and restore
all 48 recurrent Gated DeltaNet states at each possible acceptance frontier.

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

Use `Qwen3.6-27B-Q4_K_S.gguf` with the local Q4_0 support model.

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

Run the normal build and unit suite. The Qwen target-only Q4_K_S path must be
unchanged. Inspect CUDA compilation and run the existing Qwen numerical tests.
Metal, ROCm, SSD streaming and distributed paths receive no new execution
branch; their build surfaces still need validation in the release matrix.

## Follow-up optimization boundary

The correctness implementation intentionally does not call itself faster than
baseline. The next performance step is a batched target verifier with bounded
Gated DeltaNet snapshots, followed by a row-wise MTP prompt catch-up kernel and
backend argmax. Those changes must preserve this document's alignment,
frontier and full-vocabulary equivalence contracts.

## Implemented baseline and measured gate (2026-08-07)

The correctness baseline is now implemented in `ds4.c`, with CUDA Q4_0
embedding/matvec support and simultaneous target/support residency in
`ds4_cuda.cu`. The loader validates the exact tensor layout, semantic Qwen
metadata, and every tokenizer string against the target before binding the
sidecar. Production Qwen remains single-GPU CUDA and continues to reject TP,
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

On the local RTX 3090 build (`sm_86`) it passed with:

- natural full acceptance: 185/185 drafts, 47/47 full cycles, maximum returned
  chunk 5, replay worst argmax gap 0.000;
- forced full rejection: 0/118 drafts, 31 rejected cycles, maximum chunk 1,
  replay worst argmax gap 0.000;
- forced partial acceptance: 16/60 drafts, 15 partial cycles plus one naturally
  full tail cycle, maximum chunk 2, replay worst argmax gap 0.000;
- MTP-on versus target-only prefill: all 248,320 float logits bit-exact, with
  the same top-1 token 71093;
- no runtime fallback in any of the three speculative runs.

The natural run reported 13.760 net committed tokens/s for this deliberately
sequential verifier. This is a correctness measurement, not a speedup claim.
Dedicated full-vocabulary comparisons at every speculative frontier and the
EOS/explicit-stop/context-boundary/runtime-fallback matrix remain release
gates; they are not implied by the passing test above.
