# Official-Continuation Quality Testing

This directory contains the prompts, tracked official fixtures, and scripts used
to compare local GGUF variants against hosted-model continuations.

The metric is target-token negative log likelihood: collect a deterministic
official continuation, then ask each local GGUF how much probability it assigns
to that exact continuation token by token.  This avoids judging quality from one
sampled answer.

## 1. Tracked Fixture Sets

Curated fixtures are kept in the repository so release QA can run without
calling hosted APIs:

- `data/glm52-openrouter-100`: 100 GLM 5.2 continuations collected through
  OpenRouter `z-ai/glm-5.2` with `top_logprobs=20`.
- `data/flash`: 100 DeepSeek V4 Flash continuations collected from the official
  DeepSeek API with `top_logprobs=20`.
- `data/pro`: 100 DeepSeek V4 PRO continuations collected from the official
  DeepSeek API with `top_logprobs=20`.

DeepSeek V4 Flash also has tracked official smoke vectors in
`tests/test-vectors/`.  Those vectors drive `./ds4_test --logprob-vectors` and
include short prompts plus long-prompt attention cases.

The hosted APIs expose output-token logprobs and top-logprob alternatives, not
full vocabulary logits.

## 2. Collect Official Continuations

```sh
export DEEPSEEK_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/flash \
  --count 100 \
  --max-tokens 24
```

For GLM 5.2 through OpenRouter:

```sh
export OPENROUTER_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --model z-ai/glm-5.2 \
  --endpoint https://openrouter.ai/api/v1/chat/completions \
  --api-key-env OPENROUTER_API_KEY \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/glm52-openrouter \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20 \
  --token-limit-field max_tokens \
  --provider-order parasail/fp8 \
  --require-parameters \
  --thinking omit \
  --reasoning-effort none
```

Use one output directory per official model.  The default model is Flash, so
`data/flash` is the recommended path for Flash continuations.  For PRO:

```sh
python3 gguf-tools/quality-testing/collect_official.py \
  --model deepseek-v4-pro \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/pro \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20
```

The script writes:

- `data/<model>/prompts/case_*.txt`
- `data/<model>/continuations/case_*.txt`
- `data/<model>/responses/case_*.json`
- `data/<model>/manifest.tsv`

The prompt list is tracked in `prompts.jsonl`.  Curated fixture directories are
also tracked after review; ad-hoc API collection directories should stay
untracked until they are intentionally promoted into the release QA set.

## 3. Build The Local Scorer

```sh
make -C gguf-tools quality-score
```

The scorer links against the DS4 runtime and uses Metal by default.

## 4. Score GGUF Variants

```sh
gguf-tools/quality-testing/score_official \
  ../deepseek-v4-quants/gguf/OLD.gguf \
  gguf-tools/quality-testing/data/pro/manifest.tsv \
  /tmp/old.tsv \
  4096

gguf-tools/quality-testing/score_official \
  ../deepseek-v4-quants/gguf/NEW.gguf \
  gguf-tools/quality-testing/data/pro/manifest.tsv \
  /tmp/new.tsv \
  4096
```

Use `data/flash/manifest.tsv` for Flash GGUFs,
`data/glm52-openrouter-100/manifest.tsv` for GLM 5.2 GGUFs, and
`data/pro/manifest.tsv` for PRO GGUFs.  The scorer and comparator do not care
which model produced the manifest; the manifest path selects the continuation
set.

For a full-residency vs SSD-streaming comparison, score the same model twice and
add the streaming flags to one run:

```sh
gguf-tools/quality-testing/score_official \
  /path/to/model.gguf \
  gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv \
  /tmp/streaming.tsv \
  4096 \
  --ssd-streaming
```

## 5. Compare

```sh
python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

Output fields:

- `avg_nll`: average negative log likelihood; lower is better.
- `delta_new_minus_old`: negative means the new GGUF fits the official
  continuation better.
- `case_wins_new_old_ties`: per-prompt NLL wins.
- `first_token_matches`: how often the local greedy first token matches the
  official first token.
- `avg_greedy_lcp`: average greedy longest common prefix against the official
  continuation.
- `api_target_mae`: when the manifest includes `response_file`, absolute
  local-vs-API logprob delta for aligned official output tokens.
- `api_top_coverage`: fraction of API top-logprob alternatives that map exactly
  to one local tokenizer token.
- `api_top1_rate`: how often the API top alternative equals the local greedy
  token.
- `api_topn_recall`: fraction of mapped API top-N alternatives found in the
  local top-N for the same position.
- `api_top_mae`: local-vs-API logprob MAE over mapped API top alternatives.
- `api_pair_rate`: pairwise ordering agreement among mapped API alternatives.

## 6. Qwen3.6 Fixture Staging

Qwen3.6 uses the same data-set layout as the existing quality fixtures.  The
tracked manifests live under `data/qwen36-27b` and `data/qwen36-27b-mtp`, with
`prompts`, `continuations`, and `responses` directories.  Shared scripts stay
in this directory alongside `collect_official.py` and the existing scorers.

Validate both manifests without loading a model:

```sh
python3 gguf-tools/quality-testing/validate_qwen36.py \
  gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  gguf-tools/quality-testing/data/qwen36-27b-mtp/manifest.json

make -C gguf-tools quality-qwen36-test
```

Inspect local GGUF headers and regenerate the tracked, deterministic metadata
snapshots.  Local model paths are command-line inputs and are never stored in
the manifests:

```sh
python3 gguf-tools/quality-testing/inspect_qwen36_gguf.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --artifact target=gguf/Qwen3.6-27B-Q4_K_M.gguf \
  --snapshot-dir gguf-tools/quality-testing/data/qwen36-metadata

python3 gguf-tools/quality-testing/inspect_qwen36_gguf.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b-mtp/manifest.json \
  --artifact mtp=gguf/mtp-Qwen3.6-27B-Q4_0.gguf \
  --snapshot-dir gguf-tools/quality-testing/data/qwen36-metadata
```

The inspector validates the complete artifact SHA-256 and size, GGUF v3
structure, semantic metadata, tokenizer/chat-template hashes, tensor names,
shapes, offsets and quantization types.  Duplicate or truncated directory
entries are rejected.

Materialize the deterministic long prompt into an ignored staging directory:

```sh
python3 gguf-tools/quality-testing/generate_qwen36_prompt.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --case long_canary_4096 \
  --staging-dir gguf-tools/quality-testing/staging/local
```

Generate an oracle candidate by choosing `llama.cpp`, `transformers`, or
`vllm`.  Every run requires complete provenance and writes the familiar
`prompts/`, `continuations/`, `responses/`, and `manifest.tsv` structure:

```sh
python3 gguf-tools/quality-testing/generate_qwen36_oracle.py \
  --manifest /path/to/local-manifest.json \
  --oracle llama.cpp \
  --staging-dir /path/to/qwen36-staging \
  --run-id llama-q4-cuda-001 \
  --engine-commit LLAMA_CPP_FULL_COMMIT \
  --build-flags '-DGGML_CUDA=ON' \
  --backend CUDA \
  --hardware 'NVIDIA GPU model and driver' \
  --dtype 'GGUF Q4_K_M'
```

For the pinned local target, pass the model without editing the manifest and
size the runtime context for the selected corpus:

```sh
python3 gguf-tools/quality-testing/generate_qwen36_oracle.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --oracle llama.cpp \
  --artifact-path target=gguf/Qwen3.6-27B-Q4_K_M.gguf \
  --staging-dir gguf-tools/quality-testing/staging/oracles \
  --run-id llama-q4-cuda-001 \
  --steps 32 --top-k 20 --context 24576 --n-gpu-layers -1 \
  --engine-commit LLAMA_CPP_FULL_COMMIT \
  --build-flags 'EXACT BUILD FLAGS' \
  --backend CUDA --hardware 'EXACT GPU AND DRIVER' --dtype 'GGUF Q4_K_M'
```

llama.cpp and Transformers save separate greedy and teacher-forced
full-vocabulary float32 streams.  Every stream records shape, byte size and
SHA-256.  After producing a second run, validate both candidates and their
render/token determinism:

```sh
python3 gguf-tools/quality-testing/verify_qwen36_run.py \
  gguf-tools/quality-testing/staging/oracles/llama-q4-cuda-001 \
  --repeat gguf-tools/quality-testing/staging/oracles/llama-q4-cuda-002 \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json
```

The llama.cpp adapter records full-vocabulary logits against the exact pinned
GGUF.  Transformers and vLLM are labelled semantic upstream oracles because
their weights and precision differ.  vLLM records top-logprob slices because
its public generation API does not expose full logits.

Generators reject `golden` and `goldens` paths, reject non-empty unmarked
staging directories, and provide no promotion or acceptance command.  Review
and copy approved fixture data manually.  Missing models, runtimes, or hardware
remain `not_verified`; they are never treated as passing.

The Q4 target was verified on an RTX 3090 with llama-cpp-python 0.3.23,
llama.cpp commit `7d442abf5c6244117fd5a1dc51a5d19f00792491`, CUDA full offload and two
independent 32-step runs over all nine corpus categories.  The generated runs
remain ignored, unreviewed staging candidates.  Transformers, vLLM and MTP
execution are still `not_verified` on this host.

## 7. Qwen3.6 Numerical Equivalence

The structured scorer mode bypasses templates and tokenization during the
numeric pass. It consumes canonical prompt and teacher-forced token IDs from
an existing, validated oracle run and keeps the model resident while writing
greedy and teacher-forced `float32-le` logits. Python owns the same directory,
checksum and provenance format introduced in section 6:

```sh
python3 gguf-tools/quality-testing/generate_qwen36_score.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --source-run gguf-tools/quality-testing/staging/oracles/llama-q4-cuda-001 \
  --scorer gguf-tools/quality-testing/score_official \
  --model gguf/Qwen3.6-27B-Q4_K_M.gguf --engine ds4 \
  --staging-dir gguf-tools/quality-testing/staging/oracles \
  --run-id ds4-q4-cuda-001 --context 24576 --top-k 20 \
  --engine-commit DS4_FULL_COMMIT --build-flags 'EXACT BUILD FLAGS' \
  --backend CUDA --hardware 'EXACT GPU AND DRIVER' --dtype 'GGUF Q4_K_M' \
  --prefill-chunk 2048
```

For the temporary 24 GiB GPU cap, use the versioned
`data/qwen36-27b/context-profile-16k.json` together with
`manifest-16k.json`, pass `--context 16384 --context-profile ...` explicitly,
and use `run_qwen36_context_matrix.py` to expand the 4096/16384 frontiers.
The context limit is configuration data, not a compiled engine constant.
On CUDA, weight arenas default to a 256 MiB reserve granularity so a Q4_K_M
model does not strand multiple GiB of VRAM in partially used arenas; the
`DS4_CUDA_WEIGHT_ARENA_CHUNK_MB` override remains available for diagnostics.

Use `score_llama` and `--engine llama.cpp` to collect the same token-driven
layout from the C++ scorer. Existing positional TSV invocations of both
scorers remain unchanged. Until native Qwen chat rendering exists, a DS4 run
records `native_rendering_status=tokenizer_only`; this prevents a report from
passing while still allowing numeric diagnostics.

`score_official --quality` enables the runtime quality mode. For an output-head
localization test, `--output-head-hidden FILE --output-head-logits FILE` reads
one exact 5120-element F32 hidden state and writes the corresponding
248320-element DS4 output-head logits without running the transformer body.
For the patched local llama scorer, `LLAMA_QWEN_CPU=1` forces zero GPU layers
and disables KQV/operation offload; verify the startup buffers before labeling
the result a CPU oracle.

Compare two complete runs with one command:

```sh
python3 gguf-tools/quality-testing/compare_qwen36_equivalence.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --mode ds4-vs-llama \
  --left-run gguf-tools/quality-testing/staging/oracles/llama-q4-cuda-001 \
  --right-run gguf-tools/quality-testing/staging/oracles/ds4-q4-cuda-001 \
  --report gguf-tools/quality-testing/staging/oracles/ds4-vs-llama.json
```

Task 4 uses the same comparator and run layout with the fixed short-message
profile declared in the manifest:

```sh
python3 gguf-tools/quality-testing/compare_qwen36_equivalence.py \
  --manifest gguf-tools/quality-testing/data/qwen36-27b/manifest.json \
  --mode ds4-vs-llama --suite short --top-k 20 \
  --left-run gguf-tools/quality-testing/staging/oracles/llama-q4-cuda-001 \
  --right-run gguf-tools/quality-testing/staging/oracles/ds4-q4-cuda-001 \
  --report gguf-tools/quality-testing/staging/oracles/ds4-vs-llama-short.json
```

The short suite selects every corpus category except `long-canary`, requires
32 greedy and teacher-forced positions, and blocks prompt/token/decoded-byte
differences and non-finite logits per position.  The calibrated v2 profile
also requires aggregate top-20 overlap and rank agreement of at least 0.95,
and aggregate teacher-forced oracle-logprob MAE no greater than 0.05.
Per-position cross-engine metric envelopes remain versioned separately in
`equivalence_thresholds.cross_engine`. Top-1/top-2 margins at or below 0.10
are reported but do not fail the run. Native rendered bytes and greedy decoded
bytes are stored as compact hex in the response JSON; full logits remain
checksummed `float32-le` files.

`ds4-vs-ds4` additionally requires every float32 bit pattern to be identical.
For a long run whose reference contains more cases than the candidate,
`--case CASE_ID` selects an explicit common case and may be repeated; it cannot
be combined with `--suite short` and never relaxes provenance or metric gates.
Exit statuses are `0` for `PASS`, `1` for a failed gate, `2` for invalid input,
and `3` for `NOT_VERIFIED`. `--diagnostic` permits nonstandard engine roles
for tooling checks but can never produce `PASS`. Unreviewed runs or missing
cross-engine calibration likewise remain `NOT_VERIFIED`; neither collection
nor comparison can promote golden files or update thresholds automatically.

### Qwen layer trace and long-run speed gate

The short-prefix diagnostic can capture the final token at every layer without
changing model semantics. First dump only `layer_out` (DS4) and `l_out`
(llama.cpp) with `DS4_QWEN_TRACE_*` and `LLAMA_QWEN_TRACE_*`; use layer `all`
for the boundary bisection, then repeat without a stage filter on the first bad
layer. `compare_qwen36_trace.py` associates embedding, attention/GDN,
recurrent state, FFN and residual stages and writes the first float32
divergence. Trace directories must already exist. These reports are diagnostic
and do not calibrate or update cross-engine thresholds.

`DS4_CUDA_QWEN_Q8_ACT_DIAG=1` is an intentionally slow diagnostic path. It
tests CPU-like Q8_K activation quantization and Q4_K integer grouping.
`DS4_CUDA_QWEN_Q8_0_CPU_SCALE_DIAG=1` separately rounds Q8_0 activation scales
to FP16. Neither changes the default path; do not combine them unless the
experiment explicitly calls for two variables, and do not promote either on
the basis of lower MAE alone. The 2026-08-05 experiments did not fix the greedy
top-1. The decision log is `docs/performance/qwen36-drift-hypotheses.md`.

`run_qwen36_context_matrix.py --execute` is blocked unless
`--performance-report` names a `ds4-qwen36-speed-v1` JSON report for the same
model SHA-256, backend and hardware. The report records `engine_commit`,
`build_flags`, exact `command` argv, `context`, `prefill_tokens`,
`decode_tokens`, both measured token/s values and `peak_memory_mib`. The runner
refuses missing/non-comparable measurements, prefill below 500 token/s, or
decode below 20 token/s before starting a long-context model process.

## 8. Qwen3.8 Generation-Isolated Quality Path

Qwen3.8 reuses the audited Qwen input corpus, never Qwen3.6 continuations or
logits. Its manifest, staging marker, run format, thresholds, snapshots and
same-GGUF oracle namespace are all `ds4-qwen38-*`:

```sh
python3 gguf-tools/quality-testing/validate_qwen38.py \
  gguf-tools/quality-testing/data/qwen38-27b/manifest.json
make -C gguf-tools quality-qwen38-test
```

The generation-aware tools have explicit Qwen3.8 entrypoints:

```sh
python3 gguf-tools/quality-testing/generate_qwen38_oracle.py \
  --manifest gguf-tools/quality-testing/data/qwen38-27b/manifest.json \
  --oracle llama.cpp --artifact-path target=gguf/Qwen3.8-27B-UD-Q4_K_S.gguf \
  --staging-dir /path/to/qwen38-staging --run-id llama-qwen38-001 \
  --steps 32 --top-k 20 --context 8192 --n-gpu-layers -1 \
  --engine-commit COMMIT --build-flags FLAGS --backend CUDA \
  --hardware HARDWARE --dtype "GGUF UD-Q4_K_S"

python3 gguf-tools/quality-testing/verify_qwen38_run.py \
  /path/to/qwen38-staging/llama-qwen38-001 \
  --manifest gguf-tools/quality-testing/data/qwen38-27b/manifest.json

python3 gguf-tools/quality-testing/compare_qwen38_equivalence.py \
  --manifest gguf-tools/quality-testing/data/qwen38-27b/manifest.json \
  --mode ds4-vs-llama --left-run /path/to/llama-run \
  --right-run /path/to/ds4-run --report /path/to/report.json --top-k 20
```

`generate_qwen38_score.py` and `generate_qwen38_prompt.py` package DS4 scorer
output and long inputs using the same Qwen3.8-only staging contract. The
Qwen3.8 speed-report formats are `ds4-qwen38-speed-v1/v2`; they retain the same
production minimums of 500 prefill and 15 decode tokens/s. Runtime numerical
and speed execution is enabled because every quant format in the pinned UD
artifact now has a CUDA path. Use `DS4_TEST_QWEN38_LIVE=1 make test-qwen38` for
the target/MTP CLI gate and `DS4_TEST_QWEN38_SHA=1` for the separate 16.7 GB
checksum gate. A generated oracle remains `NOT_VERIFIED` until its full corpus
and provenance are reviewed; a one-case diagnostic is not a promotion.
