#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

cuda_arch=${CUDA_ARCH:-sm_86}

make -j2 cuda CUDA_ARCH="$cuda_arch"
make -j2 qwen-numerics CUDA_ARCH="$cuda_arch"
python3 -m unittest \
    tests.test_perf_harness \
    tests.test_qwen36_equivalence \
    tests.test_qwen36_numerics \
    tests.test_qwen36_safety
./ds4_test --qwen35-layer-pattern
