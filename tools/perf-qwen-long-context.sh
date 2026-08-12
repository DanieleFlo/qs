#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

action=${1:-}
experiment_id=${2:-}
if [ -z "$action" ] || [ -z "$experiment_id" ]; then
    echo "usage: $0 profile|direction|slow EXPERIMENT_ID [--env NAME=VALUE ...]" >&2
    exit 2
fi
shift 2

model=${PERF_MODEL:-gguf/Qwen3.6-27B-Q4_K_S.gguf}
prompt=${PERF_PROMPT:-tests/long_context_story_prompt.txt}
results=${PERF_RESULTS:-performance-results}
harness="python3 tools/perf_harness.py"

case "$action" in
    profile)
        $harness profile-network \
            --model "$model" --prompt "$prompt" \
            --context 10666 --generation-tokens 2 --prefill-chunk 2048 \
            --env DS4_CUDA_QWEN_NO_SPLIT_K_ATTN=1 \
            --output "$results/$experiment_id-baseline-network.json"
        $harness profile-network \
            --model "$model" --prompt "$prompt" \
            --context 10666 --generation-tokens 2 --prefill-chunk 2048 \
            "$@" \
            --baseline "$results/$experiment_id-baseline-network.json" \
            --output "$results/$experiment_id-candidate-network.json"
        ;;
    direction)
        $harness run \
            --id "$experiment_id-baseline" --suite long-context-direction \
            --repetitions 2 --warmup never \
            --model "$model" --prompt "$prompt" \
            --hypothesis "freeze the long-context baseline" \
            --metric gen_steady_tps --baseline-run --results "$results" \
            --env DS4_CUDA_QWEN_NO_SPLIT_K_ATTN=1
        $harness run \
            --id "$experiment_id-candidate" --suite long-context-direction \
            --repetitions 2 --warmup never \
            --model "$model" --prompt "$prompt" \
            --hypothesis "parallel KV partitions reduce long-context attention latency" \
            --metric gen_steady_tps "$@" \
            --baseline "$results/$experiment_id-baseline/experiment.json" \
            --results "$results"
        ;;
    slow)
        $harness run \
            --id "$experiment_id-baseline" --suite long-context-slow \
            --repetitions 5 --warmup always \
            --model "$model" --prompt "$prompt" \
            --hypothesis "freeze the confirmed long-context baseline" \
            --metric gen_steady_tps --baseline-run --results "$results" \
            --env DS4_CUDA_QWEN_NO_SPLIT_K_ATTN=1
        $harness run \
            --id "$experiment_id-candidate" --suite long-context-slow \
            --repetitions 5 --warmup always \
            --model "$model" --prompt "$prompt" \
            --hypothesis "confirm parallel KV attention in one resident 8K/12K/16K sweep" \
            --metric gen_steady_tps "$@" \
            --baseline "$results/$experiment_id-baseline/experiment.json" \
            --results "$results"
        ;;
    *)
        echo "unknown action: $action" >&2
        exit 2
        ;;
esac
