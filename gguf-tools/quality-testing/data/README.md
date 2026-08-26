# Official Quality Fixtures

This directory contains curated hosted-model continuation fixtures that are
safe to commit and use in release QA.

- `glm52-openrouter-100`: 100 GLM 5.2 OpenRouter continuations with API
  top-logprob slices.
- `flash`: 100 DeepSeek V4 Flash 0731 continuations from the official DeepSeek
  API, with API top-logprob slices.
- `pro`: 100 DeepSeek V4 PRO preview continuations with API top-logprob slices.
- `pro-0813`: 100 DeepSeek V4 PRO 0813 continuations with API top-logprob
  slices.
- `qwen36-27b`: shared English Qwen3.6 input corpus and pinned target
  manifest; reviewed output uses the same prompts/continuations/responses
  layout as the existing sets.
- `qwen36-27b-mtp`: pinned target-plus-MTP manifest, reusing the exact target
  prompt set.
- `qwen38-27b`: pinned Qwen3.8 UD-Q4_K_S target-plus-MTP manifest. It reuses
  the input corpus only; numerical oracles and output gates have a distinct
  `ds4-qwen38-*` namespace and never compare outputs against Qwen3.6.

Each fixture directory contains:

- `prompts/case_*.txt`: exact user prompts.
- `continuations/case_*.txt`: deterministic hosted-model continuations.
- `responses/case_*.json`: raw hosted responses, including logprob slices.
- `manifest.tsv`: paths consumed by `score_official`.

DeepSeek V4 Flash smoke vectors are also tracked in `tests/test-vectors/` and
are run by `./ds4_test --logprob-vectors`.
