#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

model="${DS4_TEST_MODEL:-gguf/Qwen3.8-27B-UD-Q4_K_S.gguf}"
test_dir="${DS4_AGENTIC_TEST_DIR:-$repo_dir/tests}"
test_binary="${DS4_AGENTIC_CHECKPOINT_BIN:-$repo_dir/tests/test_agentic_checkpoint}"
suffix=""
if [[ "${1:-}" == "--mtp" ]]; then suffix="-mtp"; fi
out="$test_dir/agentic-checkpoint${suffix}.out.log"
err="$test_dir/agentic-checkpoint${suffix}.err.log"
status="$test_dir/agentic-checkpoint${suffix}.exit"

rm -f "$out" "$err" "$status"
if [[ ! -x "$test_binary" ]]; then
    printf 'error: agentic checkpoint test binary is not executable: %s\n' \
        "$test_binary" >&2
    exit 2
fi
if [[ "${1:-}" == "--mtp" ]]; then
    if [[ -n "${DS4_TEST_MTP:-}" ]]; then
        mtp="$DS4_TEST_MTP"
    elif [[ "$(basename "$model")" == "Qwen3.8-27B-UD-Q4_K_S.gguf" ]]; then
        mtp="gguf/mtp-Qwen3.8-27B-Q4_0.gguf"
    else
        mtp="gguf/mtp-Qwen3.6-27B-Q4_0.gguf"
    fi
    DS4_TEST_MODEL="$model" DS4_TEST_MTP="$mtp" \
        DS4_AGENTIC_TEST_DIR="$test_dir" \
        "$test_binary" >"$out" 2>"$err"
else
    DS4_TEST_MODEL="$model" DS4_AGENTIC_TEST_DIR="$test_dir" \
        "$test_binary" >"$out" 2>"$err"
fi
rc=$?
printf '%s\n' "$rc" >"$status"
exit "$rc"
