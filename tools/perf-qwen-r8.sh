#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

action=${1:-}
experiment_id=${2:-}
if [ -z "$action" ]; then
    echo "usage: $0 build|direction|slow|long [EXPERIMENT_ID] [--env NAME=VALUE ...]" >&2
    exit 2
fi
if [ "$action" != "build" ] && [ -z "$experiment_id" ]; then
    echo "$action requires a new EXPERIMENT_ID" >&2
    exit 2
fi
if [ -n "$experiment_id" ]; then
    shift 2
else
    shift
fi

cuda_arch=${CUDA_ARCH:-sm_86}
model=${PERF_MODEL:-gguf/Qwen3.6-27B-Q4_K_S.gguf}
prompt=${PERF_PROMPT:-tests/long_context_story_prompt.txt}
results=${PERF_RESULTS:-performance-results}
harness="python3 tools/perf_harness.py"

build_runtime() {
    # Relink every user-facing consumer of ds4_cuda.o. Benchmark-only builds
    # previously hid measured gains behind stale CLI/server executables.
    make -j2 ds4 ds4-bench ds4-server CUDA_ARCH="$cuda_arch"
}

run_long_gqa_pair() {
    repetitions=$1
    warmup=$2
    shift 2
    $harness run \
        --id "$experiment_id-gqa1" --suite long-context-slow \
        --repetitions "$repetitions" --warmup "$warmup" \
        --model "$model" --prompt "$prompt" \
        --hypothesis "freeze scalar per-query-head split-K with R8 enabled" \
        --metric gen_steady_tps --baseline-run --results "$results" \
        --env DS4_CUDA_QWEN_NO_GQA_GROUP_ATTN=1
    $harness run \
        --id "$experiment_id-gqa2" --suite long-context-slow \
        --repetitions "$repetitions" --warmup "$warmup" \
        --model "$model" --prompt "$prompt" \
        --hypothesis "reuse each GQA K/V load across two query heads" \
        --metric gen_steady_tps "$@" \
        --baseline "$results/$experiment_id-gqa1/experiment.json" \
        --results "$results"
}

run_pair() {
    suite=$1
    repetitions=$2
    warmup=$3
    shift 3
    $harness run \
        --id "$experiment_id-baseline" --suite "$suite" \
        --repetitions "$repetitions" --warmup "$warmup" \
        --model "$model" --prompt "$prompt" \
        --hypothesis "freeze the exact F32 rollback baseline" \
        --metric gen_steady_tps --baseline-run --results "$results" \
        --env DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8=1
    $harness run \
        --id "$experiment_id-r8" --suite "$suite" \
        --repetitions "$repetitions" --warmup "$warmup" \
        --model "$model" --prompt "$prompt" \
        --hypothesis "confirm the default fused residual R8 Q8_1 decode" \
        --metric gen_steady_tps "$@" \
        --baseline "$results/$experiment_id-baseline/experiment.json" \
        --results "$results"
}

case "$action" in
    build)
        build_runtime
        make qwen-numerics CUDA_ARCH="$cuda_arch"
        ;;
    direction)
        build_runtime
        run_pair direction 2 never "$@"
        ;;
    slow)
        build_runtime
        make qwen-numerics CUDA_ARCH="$cuda_arch"
        run_pair r8-slow 5 always "$@"
        ;;
    long)
        build_runtime
        make qwen-numerics CUDA_ARCH="$cuda_arch"
        run_long_gqa_pair 5 always "$@"
        ;;
    *)
        echo "unknown action: $action" >&2
        exit 2
        ;;
esac
