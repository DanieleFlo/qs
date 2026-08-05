#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} test_block_q6_k;

static float test_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t frac = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) bits = sign;
    else if (exp == 31) bits = sign | 0x7f800000u | (frac << 13);
    else bits = sign | ((exp + 112u) << 23) | (frac << 13);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float test_q6_value(const test_block_q6_k *b, uint32_t i) {
    const uint32_t half = i >> 7;
    const uint32_t within = i & 127u;
    const uint32_t lane = within & 31u;
    const uint8_t qh = b->qh[half * 32u + lane];
    uint8_t q;
    uint32_t scale;
    if (within < 32u) {
        q = (b->ql[half * 64u + lane] & 15u) | ((qh & 3u) << 4);
        scale = half * 8u + (lane >> 4);
    } else if (within < 64u) {
        q = (b->ql[half * 64u + lane + 32u] & 15u) | (((qh >> 2) & 3u) << 4);
        scale = half * 8u + 2u + (lane >> 4);
    } else if (within < 96u) {
        q = (b->ql[half * 64u + lane] >> 4) | (((qh >> 4) & 3u) << 4);
        scale = half * 8u + 4u + (lane >> 4);
    } else {
        q = (b->ql[half * 64u + lane + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
        scale = half * 8u + 6u + (lane >> 4);
    }
    return test_f16_to_f32(b->d) * (float)b->scales[scale] * ((float)q - 32.0f);
}

static int check_q6k_matmul(void) {
    test_block_q6_k weights[2];
    float input[256];
    float expected[2] = {0.0f, 0.0f};
    float actual[2] = {0.0f, 0.0f};
    memset(weights, 0, sizeof(weights));
    for (uint32_t row = 0; row < 2; row++) {
        weights[row].d = 0x3c00u;
        for (uint32_t i = 0; i < 128; i++) weights[row].ql[i] = (uint8_t)(i * 13u + row * 17u);
        for (uint32_t i = 0; i < 64; i++) weights[row].qh[i] = (uint8_t)(i * 29u + row * 7u);
        for (uint32_t i = 0; i < 16; i++) weights[row].scales[i] = (int8_t)((int)i - 7 + (int)row);
    }
    for (uint32_t i = 0; i < 256; i++) input[i] = ((int)(i % 23u) - 11) * 0.03125f;
    for (uint32_t row = 0; row < 2; row++) {
        for (uint32_t i = 0; i < 256; i++) expected[row] += test_q6_value(&weights[row], i) * input[i];
    }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(actual));
    int rc = 1;
    if (x && out && ds4_gpu_set_model_map(weights, sizeof(weights)) &&
        ds4_gpu_tensor_write(x, 0, input, sizeof(input)) &&
        ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0, 14u,
                                    256u, 2u, x, 1u) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(out, 0, actual, sizeof(actual))) {
        rc = 0;
        for (uint32_t row = 0; row < 2; row++) {
            const float err = actual[row] - expected[row];
            if (err < -1.0e-3f || err > 1.0e-3f) {
                fprintf(stderr, "Q6_K matmul mismatch row=%u got=%f expected=%f\n",
                        row, (double)actual[row], (double)expected[row]);
                rc = 1;
            }
        }
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    return rc;
}

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double getenv_seconds(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = NULL;
    const double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        /* Exclude one-time CUDA module/kernel setup from the throughput guard. */
        if (!ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !ds4_gpu_synchronize()) {
            rc = 1;
            goto cleanup;
        }
        const double t0 = monotonic_seconds();
        if (ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            ds4_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = ds4_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        const double max_seconds = getenv_seconds("DS4_CUDA_TOPK_REGRESSION_SEC", 2.0);
        fprintf(stderr, "cuda-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

cleanup:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8100;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (heads && q && raw && comp &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                              comp,
                                              0,
                                              n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "attention fallback ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_q6k_matmul() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
