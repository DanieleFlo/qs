/* Bitwise regression for the cooperative IQ4_XS decoder against the scalar
 * production decoder. This includes finite half scales, signs, zeros,
 * nibble/scale extremes and partial CTAs; no model file is required. */
#include "../ds4_cuda.cu"

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); exit(1); \
} } while (0)
#define CUDA(c) CHECK((c) == cudaSuccess)

template<typename T> static void check(uint64_t blocks) {
    const uint64_t count = blocks * CUDA_QK_K;
    const size_t weight_bytes = blocks * sizeof(cuda_block_iq4_xs);
    const size_t bytes = count * sizeof(T);
    cuda_block_iq4_xs *host = (cuda_block_iq4_xs *)malloc(weight_bytes), *weights;
    T *out;
    void *reference = malloc(bytes), *actual = malloc(bytes);
    CHECK(host && reference && actual);
    uint32_t seed = 12345;
    for (size_t i = 0; i < weight_bytes; i++) {
        seed = 1664525u * seed + 1013904223u;
        ((uint8_t *)host)[i] = (uint8_t)(seed >> 24);
    }
    for (uint64_t i = 0; i < blocks; i++) {
        uint16_t scale = (uint16_t)i;
        /* Exclude Inf/NaN weights; keep both signs and every finite scale. */
        if ((scale & 0x7c00u) == 0x7c00u) scale = 0;
        host[i].d = scale;
        if (i % 7u == 0u) {
            host[i].scales_h = i & 1u ? 0xffffu : 0;
            memset(host[i].scales_l, i & 1u ? 0xff : 0, sizeof(host[i].scales_l));
            memset(host[i].qs, (int)(i & 255u), sizeof(host[i].qs));
        }
    }
    CUDA(cudaMalloc(&weights, weight_bytes));
    CUDA(cudaMalloc(&out, bytes));
    CUDA(cudaMemcpy(weights, host, weight_bytes, cudaMemcpyHostToDevice));
    if (sizeof(T) == sizeof(float)) {
        qwen38_ud_dequant_f32_kernel<<<(count + 255u) / 256u, 256>>>(
            (float *)out, (const unsigned char *)weights, 23u, count);
    } else {
        qwen38_ud_dequant_f16_kernel<<<(count + 255u) / 256u, 256>>>(
            (__half *)out, (const unsigned char *)weights, 23u, count);
    }
    CUDA(cudaMemcpy(reference, out, bytes, cudaMemcpyDeviceToHost));
    for (int vector = 0; vector < 2; vector++) {
        if (vector) {
            qwen38_iq4_xs_dequant_kernel<T, true><<<(blocks + 7u) / 8u, 256>>>(
                out, weights, blocks);
        } else {
            qwen38_iq4_xs_dequant_kernel<T, false><<<(blocks + 7u) / 8u, 256>>>(
                out, weights, blocks);
        }
        CUDA(cudaMemcpy(actual, out, bytes, cudaMemcpyDeviceToHost));
        CHECK(memcmp(reference, actual, bytes) == 0);
    }
    printf("{\"blocks\":%llu,\"bytes_per_value\":%zu,\"both_variants_bit_exact\":true}\n",
           (unsigned long long)blocks, sizeof(T));
    CUDA(cudaFree(weights)); CUDA(cudaFree(out));
    free(host); free(reference); free(actual);
}

int main(void) {
    CUDA(cudaSetDevice(0));
    for (uint64_t blocks : {1ull, 7ull, 8ull, 9ull, 17ull, 65537ull}) {
        check<float>(blocks);
        check<__half>(blocks);
    }
    return 0;
}
