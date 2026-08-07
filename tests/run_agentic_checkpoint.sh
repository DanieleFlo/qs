#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

model="${DS4_TEST_MODEL:-gguf/Qwen3.6-27B-Q4_K_S.gguf}"
test_dir="${DS4_AGENTIC_TEST_DIR:-$repo_dir/tests}"
suffix=""
if [[ "${1:-}" == "--mtp" ]]; then suffix="-mtp"; fi
out="$test_dir/agentic-checkpoint${suffix}.out.log"
err="$test_dir/agentic-checkpoint${suffix}.err.log"
status="$test_dir/agentic-checkpoint${suffix}.exit"

rm -f "$out" "$err" "$status"
if [[ "${1:-}" == "--mtp" ]]; then
    mtp="${DS4_TEST_MTP:-gguf/mtp-Qwen3.6-27B-Q4_0.gguf}"
    DS4_TEST_MODEL="$model" DS4_TEST_MTP="$mtp" \
        DS4_AGENTIC_TEST_DIR="$test_dir" \
        ./tests/test_agentic_checkpoint >"$out" 2>"$err"
else
    DS4_TEST_MODEL="$model" DS4_AGENTIC_TEST_DIR="$test_dir" \
        ./tests/test_agentic_checkpoint >"$out" 2>"$err"
fi
rc=$?
printf '%s\n' "$rc" >"$status"
exit "$rc"
