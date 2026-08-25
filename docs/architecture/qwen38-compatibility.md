# Qwen3.8 27B UD-Q4_K_S compatibility audit

## Pinned artifacts

DS4 recognizes only the audited Unsloth pair at repository revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`:

- `Qwen3.8-27B-UD-Q4_K_S.gguf`: 15,358,213,024 bytes, SHA-256
  `75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017`;
- `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`: 1,369,590,656 bytes, SHA-256
  `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e`.

`download_model.sh qwen38-q4-mtp` downloads both files, flattens the MTP path
into `gguf/`, verifies the checksums, and links `ds4flash.gguf` to the target.

## Architecture and GGUF contract

The runtime architecture is the existing `qwen35` implementation: 64 target
layers, hidden size 5120, 24 query heads, 4 KV heads, head dimension 256,
intermediate size 17408, vocabulary 248320, and 262144-token metadata context.
The Qwen3.8 target contains a 65th embedded NextN block and 866 tensors. The
external MTP GGUF contains the same block 64 as an 18-tensor sidecar.

The target validation pins the observed dynamic-quant inventory:

| GGML type | Tensor count |
| --- | ---: |
| F32 | 360 |
| Q8_0 | 99 |
| Q3_K | 13 |
| Q4_K | 95 |
| Q5_K | 80 |
| Q6_K | 18 |
| IQ2_XS | 1 |
| IQ2_S | 1 |
| IQ3_XXS | 5 |
| IQ3_S | 15 |
| IQ4_NL | 7 |
| IQ4_XS | 172 |

The external MTP inventory is 8 F32, 2 Q3_K, 4 Q4_K, and 4 Q6_K tensors. DS4
also requires the target and sidecar tokenizer tables to be byte-identical and
rejects a Qwen3.6 sidecar paired with a Qwen3.8 target (or the inverse).

## Frontend behavior

Both `ds4` and `ds4-server` recognize the stable model ID
`Qwen3.8-27B-UD-Q4_K_S.gguf`. `--mtp` selects
`gguf/mtp-Qwen3.8-27B-Q4_0.gguf` only after the target generation has been
validated; `--mtp-model FILE` remains the explicit override. Qwen3.6 and
Qwen3.8 use separate managed KV-cache directories.

Qwen3.6 and Qwen3.8 are rendered with the audited GGUF ChatML contract:
`<|im_start|>{role}\n...<|im_end|>`, the Qwen thinking prefix, and grouped
`<tool_response>` user turns. DS4 deliberately retains its constrained DSML
tool-call wire format instead of exposing Qwen's native `<function=...>` text:
the existing parser, marker protection, JSON-schema validation, checkpointing
and fail-closed token mask therefore remain one shared security boundary.

The server selects a distinct Qwen syntax path rather than reusing DeepSeek
role markers. Per-turn agentic allowlists are attached to the latest user turn,
outside the durable system-cache boundary, so they remain recent after archived
history. A narrow reminder is also added for a required single zero-argument
tool; it prevents Qwen3.6 from consuming a short reasoning budget in prose
without perturbing multi-tool calls. `ds4-agent` wraps its trusted DSML grammar
in a real ChatML system message and reuses the same rule for long-session system
prompt reminders.

The agent's fixed system-prompt snapshot treats audited dense Qwen Q4_K_S as a
Q4 persistent-payload identity. It must not ask the routed-expert quantization
helper for a model that has no routed experts; doing so previously disabled the
otherwise-supported Qwen3.8 warm system-prompt cache.

## Test and oracle isolation

`make test-qwen38` validates the pinned metadata snapshots, embedded and
external MTP layouts, exact artifact hashes when requested, automatic MTP
selection, target/sidecar generation mismatch rejection, model IDs, cache and
agentic gates, plus live target-only/MTP generation when
`DS4_TEST_QWEN38_LIVE=1`. The
Qwen3.8 oracle generator, verifier, scorer and equivalence entrypoints write
only `ds4-qwen38-*` formats and a `.ds4-qwen38-staging` marker. They may reuse
the Qwen3.6 input cases, but never its continuations, logits or expected token
IDs.

The live constrained API matrix covers optional and required calls, multiple
calls, agent skills with and without archived history, nested arguments,
structured JSON output, hostile protocol/JSON payloads, reasoning decoys and
unsupported schemas. On 2026-08-25 it passed 20/20 on both Qwen3.8 and Qwen3.6;
Qwen3.8 also passed the affected zero-argument, multi-call and auto-skill cases
after the final prompt refinement. A real `ds4-agent` Qwen3.8 one-shot emitted
one valid DSML `read` call, consumed the ChatML `<tool_response>`, and returned
the expected first heading from `PROJECT_INDEX.md`.

The Qwen3.8 matrix was additionally run with
`DS4_CONSTRAINT_MODE=compare_new_vs_oracle`: every observed optimized/oracle
comparison reported zero allowed-token divergences. The trie-mode harness
artifacts use one warm-up plus five measured repetitions and require identical,
deterministic semantic output.

## CUDA runtime status

The Qwen3.6 graph is reused, while Q3_K, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
IQ4_NL, and IQ4_XS have dedicated CUDA block layouts, embedding accessors,
prefill dequantization and MMVQ decode paths. `qwen_numerics_probe` compares all
seven implementations with an independent scalar codebook oracle. Production
decode uses the verified Q8_1+R8 residual policy; prefill uses F16 GEMM from 128
rows and retains F32 below that crossover. The implementation roadmap records
the current numerical and performance evidence. MTP, server and SSD acceptance
have dedicated live gates and pass independently of kernel readiness; only the
reviewed full-corpus cross-engine oracle promotion remains `NOT_VERIFIED`.
