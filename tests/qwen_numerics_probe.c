#define _POSIX_C_SOURCE 200809L

#include "ds4_gpu.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GDN_KEY_HEADS = 16,
    GDN_VALUE_HEADS = 48,
    GDN_DIM = 128,
    GDN_QKV = 10240,
    GDN_HEADS = 6144,
    GDN_CONV = 4,
    GDN_STEPS = 4,
};

typedef struct {
    double mae;
    double rmse;
    double max_abs;
    double max_ref;
    double cosine;
} probe_metrics;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} probe_block_q4_k;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
} probe_block_q5_k;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} probe_block_q6_k;

typedef struct {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
} probe_block_q3_k;

typedef struct {
    uint16_t d;
    uint16_t qs[32];
    uint8_t scales[8];
} probe_block_iq2_xs;

typedef struct {
    uint16_t d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t scales[8];
} probe_block_iq2_s;

typedef struct {
    uint16_t d;
    uint8_t qs[96];
} probe_block_iq3_xxs;

typedef struct {
    uint16_t d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t signs[32];
    uint8_t scales[4];
} probe_block_iq3_s;

typedef struct {
    uint16_t d;
    uint8_t qs[16];
} probe_block_iq4_nl;

typedef struct {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
} probe_block_iq4_xs;

_Static_assert(sizeof(probe_block_q3_k) == 110, "Q3_K block layout");
_Static_assert(sizeof(probe_block_iq2_xs) == 74, "IQ2_XS block layout");
_Static_assert(sizeof(probe_block_iq2_s) == 82, "IQ2_S block layout");
_Static_assert(sizeof(probe_block_iq3_xxs) == 98, "IQ3_XXS block layout");
_Static_assert(sizeof(probe_block_iq3_s) == 110, "IQ3_S block layout");
_Static_assert(sizeof(probe_block_iq4_nl) == 18, "IQ4_NL block layout");
_Static_assert(sizeof(probe_block_iq4_xs) == 136, "IQ4_XS block layout");

/* Reuse only the immutable GGML codebooks.  The host dequantizers below are
 * separate scalar implementations, so the test still catches packing,
 * indexing, sign, scale, dispatch, and CUDA-reduction mistakes. */
#define DS4_IQ2_TABLE_STORAGE static const __attribute__((unused))
#include "ds4_iq2_tables_cuda.inc"
#undef DS4_IQ2_TABLE_STORAGE
#define DS4_IQ_UD_TABLE_STORAGE static const
#include "ds4_iq_ud_tables_cuda.inc"
#undef DS4_IQ_UD_TABLE_STORAGE

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t frac = h & 0x3ffu;
    uint32_t bits = exp == 0 ? sign :
                    exp == 31 ? sign | 0x7f800000u | (frac << 13) :
                    sign | ((exp + 112u) << 23) | (frac << 13);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float f32_round_to_f16(float value) {
    const _Float16 half = (_Float16)value;
    return (float)half;
}

static void quantize_dequantize_q8_1(const float *input, float *output,
                                     uint32_t count) {
    for (uint32_t base = 0; base < count; base += 32u) {
        float maximum = 0.0f;
        for (uint32_t i = 0; i < 32u; i++)
            maximum = fmaxf(maximum, fabsf(input[base + i]));
        const float d = maximum > 0.0f ? maximum / 127.0f : 0.0f;
        const float stored_d = f32_round_to_f16(d);
        for (uint32_t i = 0; i < 32u; i++) {
            int q = d > 0.0f ? (int)nearbyintf(input[base + i] / d) : 0;
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            output[base + i] = stored_d * (float)q;
        }
    }
}

static void quantize_dequantize_q8_1_r8(const float *input, float *output,
                                        uint32_t count) {
    for (uint32_t base = 0; base < count; base += 32u) {
        float maximum = 0.0f;
        for (uint32_t i = 0; i < 32u; i++)
            maximum = fmaxf(maximum, fabsf(input[base + i]));
        const float d0 = maximum > 0.0f ? maximum / 127.0f : 0.0f;
        const float stored_d0 = f32_round_to_f16(d0);
        int q0[32];
        float residual[32];
        maximum = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            q0[i] = d0 > 0.0f ? (int)nearbyintf(input[base + i] / d0) : 0;
            if (q0[i] < -127) q0[i] = -127;
            if (q0[i] > 127) q0[i] = 127;
            residual[i] = input[base + i] - stored_d0 * (float)q0[i];
            maximum = fmaxf(maximum, fabsf(residual[i]));
        }
        const float d1 = maximum > 0.0f ? maximum / 127.0f : 0.0f;
        const float stored_d1 = f32_round_to_f16(d1);
        for (uint32_t i = 0; i < 32u; i++) {
            int q1 = d1 > 0.0f ? (int)nearbyintf(residual[i] / d1) : 0;
            if (q1 < -127) q1 = -127;
            if (q1 > 127) q1 = 127;
            output[base + i] = stored_d0 * (float)q0[i] + stored_d1 * (float)q1;
        }
    }
}

static void q4_k_scale_min(uint32_t group, const uint8_t *packed,
                           uint8_t *scale, uint8_t *minimum) {
    if (group < 4) {
        *scale = packed[group] & 63u;
        *minimum = packed[group + 4] & 63u;
    } else {
        *scale = (packed[group + 4] & 15u) | ((packed[group - 4] >> 6) << 4);
        *minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
    }
}

static float q4_k_value(const probe_block_q4_k *block, uint32_t i) {
    const uint32_t group = i / 32u;
    const uint32_t byte = (group >> 1) * 32u + (i & 31u);
    const uint32_t shift = (group & 1u) * 4u;
    uint8_t scale, minimum;
    q4_k_scale_min(group, block->scales, &scale, &minimum);
    return f16_to_f32(block->d) * (float)scale *
           (float)((block->qs[byte] >> shift) & 15u) -
           f16_to_f32(block->dmin) * (float)minimum;
}

static float q5_k_value(const probe_block_q5_k *block, uint32_t i) {
    const uint32_t group = i / 32u;
    const uint32_t byte = (group >> 1) * 32u + (i & 31u);
    const uint32_t shift = (group & 1u) * 4u;
    uint8_t scale, minimum;
    q4_k_scale_min(group, block->scales, &scale, &minimum);
    const uint32_t high = (block->qh[i & 31u] >> group) & 1u;
    const uint32_t q = ((block->qs[byte] >> shift) & 15u) | (high << 4u);
    return f16_to_f32(block->d) * (float)scale * (float)q -
           f16_to_f32(block->dmin) * (float)minimum;
}

static float q6_k_value(const probe_block_q6_k *block, uint32_t i) {
    const uint32_t half = i >> 7u;
    const uint32_t within = i & 127u;
    const uint32_t lane = within & 31u;
    const uint8_t high = block->qh[half * 32u + lane];
    uint8_t q;
    uint32_t scale;
    if (within < 32u) {
        q = (block->ql[half * 64u + lane] & 15u) | ((high & 3u) << 4u);
        scale = half * 8u + (lane >> 4u);
    } else if (within < 64u) {
        q = (block->ql[half * 64u + lane + 32u] & 15u) |
            (((high >> 2u) & 3u) << 4u);
        scale = half * 8u + 2u + (lane >> 4u);
    } else if (within < 96u) {
        q = (block->ql[half * 64u + lane] >> 4u) |
            (((high >> 4u) & 3u) << 4u);
        scale = half * 8u + 4u + (lane >> 4u);
    } else {
        q = (block->ql[half * 64u + lane + 32u] >> 4u) |
            (((high >> 6u) & 3u) << 4u);
        scale = half * 8u + 6u + (lane >> 4u);
    }
    return f16_to_f32(block->d) * (float)block->scales[scale] *
           (float)((int)q - 32);
}

static uint8_t u64_byte(uint64_t value, uint32_t index) {
    return (uint8_t)(value >> (8u * index));
}

static uint16_t load_u16_le(const uint8_t *bytes) {
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8u);
}

static float q3_k_value(const probe_block_q3_k *block, uint32_t i) {
    const uint32_t n = i >> 7u;
    const uint32_t within = i & 127u;
    const uint32_t j = within >> 5u;
    const uint32_t l = within & 31u;
    const uint32_t is0 = l >> 4u;
    const uint32_t is = 8u * n + 2u * j + is0;
    uint8_t scale;
    if (is < 4u) {
        scale = (block->scales[is] & 0x0fu) |
                (((block->scales[is + 8u] >> 0u) & 3u) << 4u);
    } else if (is < 8u) {
        scale = (block->scales[is] & 0x0fu) |
                (((block->scales[is + 4u] >> 2u) & 3u) << 4u);
    } else if (is < 12u) {
        scale = (block->scales[is - 8u] >> 4u) |
                (((block->scales[is] >> 4u) & 3u) << 4u);
    } else {
        scale = (block->scales[is - 8u] >> 4u) |
                (((block->scales[is - 4u] >> 6u) & 3u) << 4u);
    }
    const uint32_t mask = 1u << (4u * n + j);
    const int q = (int)((block->qs[32u * n + l] >> (2u * j)) & 3u) -
                  ((block->hmask[l] & mask) ? 0 : 4);
    return f16_to_f32(block->d) * (float)((int)scale - 32) * (float)q;
}

static float iq2_xs_value(const probe_block_iq2_xs *block, uint32_t i) {
    const uint32_t ib = i >> 5u;
    const uint32_t il = (i & 31u) >> 3u;
    const uint32_t j = i & 7u;
    const uint16_t q = block->qs[4u * ib + il];
    const uint8_t grid = u64_byte(cuda_iq2xs_grid[q & 511u], j);
    const uint8_t signs = cuda_ksigns_iq2xs[q >> 9u];
    const uint32_t scale =
        (block->scales[ib] >> (4u * (il >> 1u))) & 0x0fu;
    const float d = f16_to_f32(block->d) * (0.5f + (float)scale) * 0.25f;
    return d * (float)grid * ((signs & (1u << j)) ? -1.0f : 1.0f);
}

static float iq2_s_value(const probe_block_iq2_s *block, uint32_t i) {
    const uint32_t ib = i >> 5u;
    const uint32_t il = (i & 31u) >> 3u;
    const uint32_t j = i & 7u;
    const uint32_t grid_index = (uint32_t)block->qs[4u * ib + il] |
        (((uint32_t)block->qh[ib] << (8u - 2u * il)) & 0x300u);
    const uint8_t grid = u64_byte(cuda_iq2s_grid[grid_index], j);
    const uint8_t signs = block->qs[32u + 4u * ib + il];
    const uint32_t scale =
        (block->scales[ib] >> (4u * (il >> 1u))) & 0x0fu;
    const float d = f16_to_f32(block->d) * (0.5f + (float)scale) * 0.25f;
    return d * (float)grid * ((signs & (1u << j)) ? -1.0f : 1.0f);
}

static float iq3_xxs_value(const probe_block_iq3_xxs *block, uint32_t i) {
    const uint32_t ib = i >> 5u;
    const uint32_t il = (i & 31u) >> 3u;
    const uint32_t j = i & 7u;
    const uint8_t *q3 = block->qs + 8u * ib;
    const uint8_t *gas = block->qs + 64u + 4u * ib;
    const uint32_t aux = (uint32_t)load_u16_le(gas) |
                         ((uint32_t)load_u16_le(gas + 2u) << 16u);
    const uint32_t code = q3[2u * il + (j >> 2u)];
    const uint8_t grid =
        (uint8_t)(cuda_iq3xxs_grid[code] >> (8u * (j & 3u)));
    const uint8_t signs = cuda_ksigns_iq2xs[(aux >> (7u * il)) & 127u];
    const float d = f16_to_f32(block->d) *
                    (0.5f + (float)(aux >> 28u)) * 0.5f;
    return d * (float)grid * ((signs & (1u << j)) ? -1.0f : 1.0f);
}

static float iq3_s_value(const probe_block_iq3_s *block, uint32_t i) {
    const uint32_t ib = i >> 5u;
    const uint32_t il = (i & 31u) >> 3u;
    const uint32_t j = i & 7u;
    const uint8_t *qs = block->qs + 8u * ib;
    const uint32_t code = j < 4u ?
        ((uint32_t)qs[2u * il] |
         (((uint32_t)block->qh[ib] << (8u - 2u * il)) & 0x100u)) :
        ((uint32_t)qs[2u * il + 1u] |
         (((uint32_t)block->qh[ib] << (7u - 2u * il)) & 0x100u));
    const uint8_t grid =
        (uint8_t)(cuda_iq3s_grid[code] >> (8u * (j & 3u)));
    const uint8_t signs = block->signs[4u * ib + il];
    const uint32_t scale =
        (block->scales[ib >> 1u] >> (4u * (ib & 1u))) & 0x0fu;
    const float d = f16_to_f32(block->d) * (float)(1u + 2u * scale);
    return d * (float)grid * ((signs & (1u << j)) ? -1.0f : 1.0f);
}

static float iq4_nl_value(const probe_block_iq4_nl *block, uint32_t i) {
    const uint8_t q = block->qs[i & 15u];
    const uint8_t code = i < 16u ? (q & 0x0fu) : (q >> 4u);
    return f16_to_f32(block->d) * (float)cuda_kvalues_iq4nl[code];
}

static float iq4_xs_value(const probe_block_iq4_xs *block, uint32_t i) {
    const uint32_t ib = i >> 5u;
    const uint32_t j = i & 31u;
    const uint8_t q = block->qs[16u * ib + (j & 15u)];
    const uint8_t code = j < 16u ? (q & 0x0fu) : (q >> 4u);
    const uint32_t scale =
        ((block->scales_l[ib >> 1u] >> (4u * (ib & 1u))) & 0x0fu) |
        (((block->scales_h >> (2u * ib)) & 3u) << 4u);
    return f16_to_f32(block->d) * (float)((int)scale - 32) *
           (float)cuda_kvalues_iq4nl[code];
}

static float pattern(uint32_t index, uint32_t salt, float scale) {
    uint32_t x = index ^ (salt * 0x9e3779b9u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return ((int32_t)(x % 2001u) - 1000) * scale;
}

static float tree_sum_128(const float *values) {
    float work[GDN_DIM];
    memcpy(work, values, sizeof(work));
    for (uint32_t stride = GDN_DIM / 2; stride; stride >>= 1) {
        for (uint32_t i = 0; i < stride; i++) work[i] += work[i + stride];
    }
    return work[0];
}

static probe_metrics measure(const float *got, const float *ref, size_t n) {
    long double abs_sum = 0.0L, sq_sum = 0.0L;
    long double dot = 0.0L, got_sq = 0.0L, ref_sq = 0.0L;
    double max_abs = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double g = got[i], r = ref[i], d = g - r;
        const double ad = fabs(d), ar = fabs(r);
        abs_sum += ad;
        sq_sum += d * d;
        dot += g * r;
        got_sq += g * g;
        ref_sq += r * r;
        if (ad > max_abs) max_abs = ad;
        if (ar > max_ref) max_ref = ar;
    }
    probe_metrics m = {
        .mae = (double)(abs_sum / (long double)n),
        .rmse = sqrt((double)(sq_sum / (long double)n)),
        .max_abs = max_abs,
        .max_ref = max_ref,
        .cosine = (got_sq && ref_sq) ? (double)(dot / sqrtl(got_sq * ref_sq))
                                     : (max_abs == 0.0 ? 1.0 : 0.0),
    };
    return m;
}

static const char *classify(probe_metrics m, double abs_floor, double rel) {
    if (m.max_abs == 0.0) return "exact";
    if (m.max_abs <= abs_floor + rel * m.max_ref && m.cosine >= 0.999999)
        return "roundoff";
    return "outside_roundoff_envelope";
}

static int emit_metric(const char *tensor, uint32_t step,
                       const float *got, const float *ref, size_t n,
                       double abs_floor, double rel) {
    const probe_metrics m = measure(got, ref, n);
    const char *kind = classify(m, abs_floor, rel);
    printf("{\"probe\":\"gdn\",\"step\":%u,\"tensor\":\"%s\","
           "\"values\":%zu,\"mae\":%.9g,\"rmse\":%.9g,"
           "\"max_abs\":%.9g,\"max_ref\":%.9g,\"cosine\":%.12g,"
           "\"classification\":\"%s\"}\n",
           step, tensor, n, m.mae, m.rmse, m.max_abs, m.max_ref,
           m.cosine, kind);
    return strcmp(kind, "outside_roundoff_envelope") == 0;
}

static void cpu_gdn_step(float *heads, float *qkv, const float *z,
                         const float *alpha, const float *beta_in,
                         float *conv_state, float *state,
                         const float *a, const float *conv,
                         const float *dt, const float *norm) {
    for (uint32_t c = 0; c < GDN_QKV; c++) {
        float *s = conv_state + (uint64_t)c * GDN_CONV;
        s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = qkv[c];
        float y = 0.0f;
        for (uint32_t i = 0; i < GDN_CONV; i++)
            y += s[i] * conv[(uint64_t)c * GDN_CONV + i];
        qkv[c] = y / (1.0f + expf(-y));
    }

    for (uint32_t vh = 0; vh < GDN_VALUE_HEADS; vh++) {
        /* The production GGUF stores value-side tensors in tiled head order,
         * so physical value head vh is paired with key head vh % Hk. */
        const uint32_t kh = vh % GDN_KEY_HEADS;
        float q[GDN_DIM], k[GDN_DIM], squares[GDN_DIM], out[GDN_DIM];
        for (uint32_t j = 0; j < GDN_DIM; j++) {
            q[j] = qkv[(uint64_t)kh * GDN_DIM + j];
            k[j] = qkv[2048u + (uint64_t)kh * GDN_DIM + j];
            squares[j] = q[j] * q[j];
        }
        const float qscale = 1.0f / sqrtf(tree_sum_128(squares) + 1.0e-6f);
        for (uint32_t j = 0; j < GDN_DIM; j++) {
            q[j] *= qscale;
            squares[j] = k[j] * k[j];
        }
        const float kscale = 1.0f / sqrtf(tree_sum_128(squares) + 1.0e-6f);
        for (uint32_t j = 0; j < GDN_DIM; j++) k[j] *= kscale;

        const float beta = 1.0f / (1.0f + expf(-beta_in[vh]));
        const float x = alpha[vh] + dt[vh];
        const float softplus = x > 20.0f ? x : log1pf(expf(x));
        const float decay = expf(a[vh] * softplus);
        for (uint32_t d = 0; d < GDN_DIM; d++) {
            const float value = qkv[4096u + (uint64_t)vh * GDN_DIM + d];
            float pred = 0.0f;
            for (uint32_t j = 0; j < GDN_DIM; j++)
                pred += decay * state[((uint64_t)vh * GDN_DIM + j) * GDN_DIM + d] * k[j];
            float y = 0.0f;
            for (uint32_t j = 0; j < GDN_DIM; j++) {
                float *cell = state + ((uint64_t)vh * GDN_DIM + j) * GDN_DIM + d;
                *cell = decay * *cell + beta * k[j] * (value - pred);
                y += *cell * q[j] * 0.08838834764831845f;
            }
            out[d] = y;
            squares[d] = y * y;
        }
        const float rms = 1.0f / sqrtf(tree_sum_128(squares) / GDN_DIM + 1.0e-6f);
        for (uint32_t d = 0; d < GDN_DIM; d++) {
            const float gate = z[(uint64_t)vh * GDN_DIM + d];
            heads[(uint64_t)vh * GDN_DIM + d] =
                out[d] * rms * norm[d] * (gate / (1.0f + expf(-gate)));
        }
    }
}

static int run_gdn_probe(void) {
    const size_t qkv_n = GDN_QKV;
    const size_t heads_n = GDN_HEADS;
    const size_t conv_n = (size_t)GDN_QKV * GDN_CONV;
    const size_t state_n = (size_t)GDN_VALUE_HEADS * GDN_DIM * GDN_DIM;
    const size_t map_n = (GDN_VALUE_HEADS + conv_n + GDN_VALUE_HEADS + GDN_DIM);
    float *map = calloc(map_n, sizeof(float));
    float *qkv = malloc(qkv_n * sizeof(float));
    float *qkv_ref = malloc(qkv_n * sizeof(float));
    float *z = malloc(heads_n * sizeof(float));
    float *alpha = malloc(GDN_VALUE_HEADS * sizeof(float));
    float *beta = malloc(GDN_VALUE_HEADS * sizeof(float));
    float *conv_state = malloc(conv_n * sizeof(float));
    float *conv_ref = malloc(conv_n * sizeof(float));
    float *state = malloc(state_n * sizeof(float));
    float *state_ref = malloc(state_n * sizeof(float));
    float *heads = malloc(heads_n * sizeof(float));
    float *heads_ref = malloc(heads_n * sizeof(float));
    if (!map || !qkv || !qkv_ref || !z || !alpha || !beta ||
        !conv_state || !conv_ref || !state || !state_ref || !heads || !heads_ref)
        return 1;

    const uint64_t a_off = 0;
    const uint64_t conv_off = GDN_VALUE_HEADS * sizeof(float);
    const uint64_t dt_off = conv_off + conv_n * sizeof(float);
    const uint64_t norm_off = dt_off + GDN_VALUE_HEADS * sizeof(float);
    float *a = map + a_off / sizeof(float);
    float *conv = map + conv_off / sizeof(float);
    float *dt = map + dt_off / sizeof(float);
    float *norm = map + norm_off / sizeof(float);
    for (uint32_t i = 0; i < GDN_VALUE_HEADS; i++) {
        a[i] = -0.04f - 0.0007f * (float)i;
        dt[i] = -2.4f + 0.013f * (float)i;
    }
    for (size_t i = 0; i < conv_n; i++) conv[i] = pattern((uint32_t)i, 41, 0.00035f);
    for (uint32_t i = 0; i < GDN_DIM; i++) norm[i] = 0.85f + 0.002f * (float)i;
    for (size_t i = 0; i < conv_n; i++) conv_state[i] = pattern((uint32_t)i, 71, 0.00002f);
    for (size_t i = 0; i < state_n; i++) state[i] = pattern((uint32_t)i, 97, 0.0000005f);
    memcpy(conv_ref, conv_state, conv_n * sizeof(float));
    memcpy(state_ref, state, state_n * sizeof(float));

    ds4_gpu_tensor *t_qkv = ds4_gpu_tensor_alloc(GDN_STEPS * qkv_n * sizeof(float));
    ds4_gpu_tensor *t_z = ds4_gpu_tensor_alloc(GDN_STEPS * heads_n * sizeof(float));
    ds4_gpu_tensor *t_alpha = ds4_gpu_tensor_alloc(
        GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
    ds4_gpu_tensor *t_beta = ds4_gpu_tensor_alloc(
        GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
    ds4_gpu_tensor *t_conv = ds4_gpu_tensor_alloc(conv_n * sizeof(float));
    ds4_gpu_tensor *t_state = ds4_gpu_tensor_alloc(state_n * sizeof(float));
    ds4_gpu_tensor *t_heads = ds4_gpu_tensor_alloc(GDN_STEPS * heads_n * sizeof(float));
    int failed = !t_qkv || !t_z || !t_alpha || !t_beta || !t_conv || !t_state || !t_heads;
    if (failed) {
        fprintf(stderr, "qwen numerics probe: GPU tensor allocation failed\n");
        goto cleanup;
    }
    failed |= !ds4_gpu_set_model_map(map, map_n * sizeof(float));
    failed |= !ds4_gpu_tensor_write(t_conv, 0, conv_state, conv_n * sizeof(float));
    failed |= !ds4_gpu_tensor_write(t_state, 0, state, state_n * sizeof(float));

    for (uint32_t step = 0; step < GDN_STEPS && !failed; step++) {
        for (size_t i = 0; i < qkv_n; i++) qkv[i] = pattern((uint32_t)i, 101 + step, 0.00008f);
        for (size_t i = 0; i < heads_n; i++) z[i] = pattern((uint32_t)i, 151 + step, 0.0004f);
        for (uint32_t i = 0; i < GDN_VALUE_HEADS; i++) {
            alpha[i] = pattern(i, 181 + step, 0.003f);
            beta[i] = pattern(i, 211 + step, 0.003f);
        }
        memcpy(qkv_ref, qkv, qkv_n * sizeof(float));
        cpu_gdn_step(heads_ref, qkv_ref, z, alpha, beta, conv_ref, state_ref,
                     a, conv, dt, norm);

        failed |= !ds4_gpu_tensor_write(t_qkv, 0, qkv, qkv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_z, 0, z, heads_n * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_alpha, 0, alpha, GDN_VALUE_HEADS * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_beta, 0, beta, GDN_VALUE_HEADS * sizeof(float));
        failed |= !ds4_gpu_qwen35_gated_delta_net_tensor(
            t_heads, t_qkv, t_z, t_alpha, t_beta, t_conv, t_state,
            map, map_n * sizeof(float), a_off, conv_off, dt_off, norm_off);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(t_qkv, 0, qkv, qkv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_conv, 0, conv_state, conv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_state, 0, state, state_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_heads, 0, heads, heads_n * sizeof(float));
        if (!failed) {
            failed |= emit_metric("conv_silu", step, qkv, qkv_ref, qkv_n, 2e-6, 2e-5);
            failed |= emit_metric("conv_state", step, conv_state, conv_ref, conv_n, 0.0, 0.0);
            failed |= emit_metric("recurrent_state", step, state, state_ref, state_n, 2e-6, 3e-5);
            failed |= emit_metric("heads", step, heads, heads_ref, heads_n, 5e-6, 5e-5);
        }
    }

    /* Exercise the optimized rows kernel in the way prefill uses it: all
     * tokens in one launch while convolution and recurrent state stay local
     * across the token loop.  Repeating the one-row API above cannot catch a
     * missing or misordered state update inside that loop. */
    float *qkv_rows = malloc(GDN_STEPS * qkv_n * sizeof(float));
    float *qkv_rows_ref = malloc(GDN_STEPS * qkv_n * sizeof(float));
    float *z_rows = malloc(GDN_STEPS * heads_n * sizeof(float));
    float *alpha_rows = malloc(GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
    float *beta_rows = malloc(GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
    float *heads_rows = malloc(GDN_STEPS * heads_n * sizeof(float));
    float *heads_rows_ref = malloc(GDN_STEPS * heads_n * sizeof(float));
    if (!qkv_rows || !qkv_rows_ref || !z_rows || !alpha_rows || !beta_rows ||
        !heads_rows || !heads_rows_ref) {
        failed = 1;
    }
    if (!failed) {
        for (size_t i = 0; i < conv_n; i++)
            conv_state[i] = pattern((uint32_t)i, 71, 0.00002f);
        for (size_t i = 0; i < state_n; i++)
            state[i] = pattern((uint32_t)i, 97, 0.0000005f);
        memcpy(conv_ref, conv_state, conv_n * sizeof(float));
        memcpy(state_ref, state, state_n * sizeof(float));
        for (uint32_t step = 0; step < GDN_STEPS; step++) {
            float *qr = qkv_rows + (size_t)step * qkv_n;
            float *qref = qkv_rows_ref + (size_t)step * qkv_n;
            float *zr = z_rows + (size_t)step * heads_n;
            float *ar = alpha_rows + (size_t)step * GDN_VALUE_HEADS;
            float *br = beta_rows + (size_t)step * GDN_VALUE_HEADS;
            for (size_t i = 0; i < qkv_n; i++)
                qr[i] = pattern((uint32_t)i, 101 + step, 0.00008f);
            for (size_t i = 0; i < heads_n; i++)
                zr[i] = pattern((uint32_t)i, 151 + step, 0.0004f);
            for (uint32_t i = 0; i < GDN_VALUE_HEADS; i++) {
                ar[i] = pattern(i, 181 + step, 0.003f);
                br[i] = pattern(i, 211 + step, 0.003f);
            }
            memcpy(qref, qr, qkv_n * sizeof(float));
            cpu_gdn_step(heads_rows_ref + (size_t)step * heads_n,
                         qref, zr, ar, br, conv_ref, state_ref,
                         a, conv, dt, norm);
        }
        failed |= !ds4_gpu_tensor_write(t_qkv, 0, qkv_rows,
                                         GDN_STEPS * qkv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_z, 0, z_rows,
                                         GDN_STEPS * heads_n * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_alpha, 0, alpha_rows,
                                         GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_beta, 0, beta_rows,
                                         GDN_STEPS * GDN_VALUE_HEADS * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_conv, 0, conv_state,
                                         conv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_write(t_state, 0, state,
                                         state_n * sizeof(float));
        failed |= !ds4_gpu_qwen35_gated_delta_net_rows_tensor(
            t_heads, t_qkv, t_z, t_alpha, t_beta, t_conv, t_state,
            NULL, NULL,
            map, map_n * sizeof(float), a_off, conv_off, dt_off, norm_off,
            GDN_STEPS);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(t_qkv, 0, qkv_rows,
                                        GDN_STEPS * qkv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_conv, 0, conv_state,
                                        conv_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_state, 0, state,
                                        state_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_heads, 0, heads_rows,
                                        GDN_STEPS * heads_n * sizeof(float));
        if (!failed) {
            failed |= emit_metric("rows_conv_silu", GDN_STEPS,
                                  qkv_rows, qkv_rows_ref,
                                  GDN_STEPS * qkv_n, 2e-6, 2e-5);
            failed |= emit_metric("rows_conv_state", GDN_STEPS,
                                  conv_state, conv_ref, conv_n, 0.0, 0.0);
            failed |= emit_metric("rows_recurrent_state", GDN_STEPS,
                                  state, state_ref, state_n, 2e-6, 3e-5);
            failed |= emit_metric("rows_heads", GDN_STEPS,
                                  heads_rows, heads_rows_ref,
                                  GDN_STEPS * heads_n, 5e-6, 5e-5);
        }
    }
    free(heads_rows_ref); free(heads_rows); free(beta_rows); free(alpha_rows);
    free(z_rows); free(qkv_rows_ref); free(qkv_rows);

cleanup:
    if (t_heads) ds4_gpu_tensor_free(t_heads);
    if (t_state) ds4_gpu_tensor_free(t_state);
    if (t_conv) ds4_gpu_tensor_free(t_conv);
    if (t_beta) ds4_gpu_tensor_free(t_beta);
    if (t_alpha) ds4_gpu_tensor_free(t_alpha);
    if (t_z) ds4_gpu_tensor_free(t_z);
    if (t_qkv) ds4_gpu_tensor_free(t_qkv);
    free(heads_ref); free(heads); free(state_ref); free(state);
    free(conv_ref); free(conv_state); free(beta); free(alpha); free(z);
    free(qkv_ref); free(qkv); free(map);
    return failed;
}

static void cpu_qwen_attention(float *out, const float *q_gate,
                               const float *key_cache,
                               const float *value_cache, uint32_t position) {
    for (uint32_t qh = 0; qh < 24u; qh++) {
        const uint32_t kvh = qh / 6u;
        const float *q = q_gate + (uint64_t)qh * 512u;
        float acc[256] = {0};
        float online_m = -INFINITY;
        float online_l = 0.0f;
        for (uint32_t t = 0; t <= position; t++) {
            const uint64_t base = ((uint64_t)t * 4u + kvh) * 256u;
            float dot = 0.0f;
            for (uint32_t d = 0; d < 256u; d++)
                dot += q[d] * key_cache[base + d];
            const float score = dot * (1.0f / 16.0f);
            const float next_m = fmaxf(online_m, score);
            const float alpha = expf(online_m - next_m);
            const float beta = expf(score - next_m);
            online_l = online_l * alpha + beta;
            online_m = next_m;
            for (uint32_t d = 0; d < 256u; d++)
                acc[d] = acc[d] * alpha + value_cache[base + d] * beta;
        }
        for (uint32_t d = 0; d < 256u; d++) {
            const float gate = 1.0f / (1.0f + expf(-q[256u + d]));
            out[(uint64_t)qh * 256u + d] = (acc[d] / online_l) * gate;
        }
    }
}

static int run_attention_variant(ds4_gpu_tensor *t_out,
                                 ds4_gpu_tensor *t_q,
                                 ds4_gpu_tensor *t_k,
                                 ds4_gpu_tensor *t_v,
                                 ds4_gpu_tensor *t_key_cache,
                                 ds4_gpu_tensor *t_value_cache,
                                 const float *map, size_t map_n,
                                 const float *q, const float *k,
                                 const float *v, const float *key_cache,
                                 const float *value_cache,
                                 uint32_t position, uint32_t context,
                                 float *out) {
    int failed = 0;
    failed |= !ds4_gpu_tensor_write(t_q, 0, q, 24u * 512u * sizeof(float));
    failed |= !ds4_gpu_tensor_write(t_k, 0, k, 4u * 256u * sizeof(float));
    failed |= !ds4_gpu_tensor_write(t_v, 0, v, 4u * 256u * sizeof(float));
    failed |= !ds4_gpu_tensor_write(t_key_cache, 0, key_cache,
                                    (size_t)context * 4u * 256u * sizeof(float));
    failed |= !ds4_gpu_tensor_write(t_value_cache, 0, value_cache,
                                    (size_t)context * 4u * 256u * sizeof(float));
    failed |= !ds4_gpu_qwen35_full_attention_tensor(
        t_out, t_q, t_k, t_v, t_key_cache, t_value_cache,
        map, map_n * sizeof(float), 0, 256u * sizeof(float),
        position, context);
    failed |= !ds4_gpu_synchronize();
    failed |= !ds4_gpu_tensor_read(t_out, 0, out,
                                   24u * 256u * sizeof(float));
    return failed;
}

static int run_full_attention_probe(void) {
    enum {
        Q_HEADS = 24,
        KV_HEADS = 4,
        HEAD_DIM = 256,
        Q_STRIDE = 512,
        POSITION = 255,
        CONTEXT = 256,
    };
    const size_t q_n = (size_t)Q_HEADS * Q_STRIDE;
    const size_t kv_n = (size_t)KV_HEADS * HEAD_DIM;
    const size_t out_n = (size_t)Q_HEADS * HEAD_DIM;
    const size_t cache_n = (size_t)CONTEXT * KV_HEADS * HEAD_DIM;
    float *map = malloc(2u * HEAD_DIM * sizeof(float));
    float *q = malloc(q_n * sizeof(float));
    float *k = malloc(kv_n * sizeof(float));
    float *v = malloc(kv_n * sizeof(float));
    float *key_cache = malloc(cache_n * sizeof(float));
    float *value_cache = malloc(cache_n * sizeof(float));
    float *prepared_q = malloc(q_n * sizeof(float));
    float *prepared_key_cache = malloc(cache_n * sizeof(float));
    float *prepared_value_cache = malloc(cache_n * sizeof(float));
    float *cpu = malloc(out_n * sizeof(float));
    float *legacy = malloc(out_n * sizeof(float));
    float *warp = malloc(out_n * sizeof(float));
    float *split_k = malloc(out_n * sizeof(float));
    int failed = !map || !q || !k || !v || !key_cache || !value_cache ||
                 !prepared_q || !prepared_key_cache || !prepared_value_cache ||
                 !cpu || !legacy || !warp || !split_k;
    if (failed) goto cleanup;

    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        map[d] = 0.9f + 0.0007f * (float)d;
        map[HEAD_DIM + d] = 1.1f - 0.0005f * (float)d;
    }
    for (size_t i = 0; i < q_n; i++) q[i] = pattern((uint32_t)i, 1201, 0.001f);
    for (size_t i = 0; i < kv_n; i++) {
        k[i] = pattern((uint32_t)i, 1213, 0.0012f);
        v[i] = pattern((uint32_t)i, 1223, 0.0008f);
    }
    for (size_t i = 0; i < cache_n; i++) {
        key_cache[i] = pattern((uint32_t)i, 1231, 0.0009f);
        value_cache[i] = pattern((uint32_t)i, 1237, 0.0007f);
    }

    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc(out_n * sizeof(float));
    ds4_gpu_tensor *t_q = ds4_gpu_tensor_alloc(q_n * sizeof(float));
    ds4_gpu_tensor *t_k = ds4_gpu_tensor_alloc(kv_n * sizeof(float));
    ds4_gpu_tensor *t_v = ds4_gpu_tensor_alloc(kv_n * sizeof(float));
    ds4_gpu_tensor *t_key_cache = ds4_gpu_tensor_alloc(cache_n * sizeof(float));
    ds4_gpu_tensor *t_value_cache = ds4_gpu_tensor_alloc(cache_n * sizeof(float));
    failed = !t_out || !t_q || !t_k || !t_v || !t_key_cache || !t_value_cache;
    if (!failed) failed |= !ds4_gpu_set_model_map(map, 2u * HEAD_DIM * sizeof(float));

    unsetenv("DS4_CUDA_QWEN_WARP_ATTN");
    unsetenv("DS4_CUDA_QWEN_SPLIT_K_ATTN");
    if (!failed) failed |= run_attention_variant(
        t_out, t_q, t_k, t_v, t_key_cache, t_value_cache,
        map, 2u * HEAD_DIM, q, k, v, key_cache, value_cache,
        POSITION, CONTEXT, legacy);
    if (!failed) {
        /* Q/K normalization, RoPE, and the current cache write are shared by
         * both CUDA variants. Read their prepared values back so this oracle
         * isolates the attention reduction and online-softmax policy. */
        failed |= !ds4_gpu_tensor_read(t_q, 0, prepared_q, q_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_key_cache, 0, prepared_key_cache,
                                       cache_n * sizeof(float));
        failed |= !ds4_gpu_tensor_read(t_value_cache, 0, prepared_value_cache,
                                       cache_n * sizeof(float));
    }
    if (!failed) cpu_qwen_attention(cpu, prepared_q, prepared_key_cache,
                                    prepared_value_cache, POSITION);

    setenv("DS4_CUDA_QWEN_WARP_ATTN", "1", 1);
    if (!failed) failed |= run_attention_variant(
        t_out, t_q, t_k, t_v, t_key_cache, t_value_cache,
        map, 2u * HEAD_DIM, q, k, v, key_cache, value_cache,
        POSITION, CONTEXT, warp);
    unsetenv("DS4_CUDA_QWEN_WARP_ATTN");
    setenv("DS4_CUDA_QWEN_SPLIT_K_ATTN", "1", 1);
    if (!failed) failed |= run_attention_variant(
        t_out, t_q, t_k, t_v, t_key_cache, t_value_cache,
        map, 2u * HEAD_DIM, q, k, v, key_cache, value_cache,
        POSITION, CONTEXT, split_k);
    if (!failed) {
        const probe_metrics legacy_cpu = measure(legacy, cpu, out_n);
        const probe_metrics warp_cpu = measure(warp, cpu, out_n);
        const probe_metrics split_k_cpu = measure(split_k, cpu, out_n);
        const char *legacy_kind = classify(legacy_cpu, 2e-6, 2e-5);
        const char *warp_kind = classify(warp_cpu, 2e-6, 2e-5);
        const char *split_k_kind = classify(split_k_cpu, 2e-6, 2e-5);
        printf("{\"probe\":\"full_attention\",\"comparison\":\"legacy_vs_cpu\","
               "\"values\":%zu,\"mae\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               out_n, legacy_cpu.mae, legacy_cpu.max_abs,
               legacy_cpu.cosine, legacy_kind);
        printf("{\"probe\":\"full_attention\",\"comparison\":\"warp_vs_cpu\","
               "\"values\":%zu,\"mae\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               out_n, warp_cpu.mae, warp_cpu.max_abs,
               warp_cpu.cosine, warp_kind);
        printf("{\"probe\":\"full_attention\",\"comparison\":\"split_k_vs_cpu\","
               "\"values\":%zu,\"mae\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               out_n, split_k_cpu.mae, split_k_cpu.max_abs,
               split_k_cpu.cosine, split_k_kind);
        failed |= strcmp(legacy_kind, "outside_roundoff_envelope") == 0;
        failed |= strcmp(warp_kind, "outside_roundoff_envelope") == 0;
        failed |= strcmp(split_k_kind, "outside_roundoff_envelope") == 0;
    }

    if (t_value_cache) ds4_gpu_tensor_free(t_value_cache);
    if (t_key_cache) ds4_gpu_tensor_free(t_key_cache);
    if (t_v) ds4_gpu_tensor_free(t_v);
    if (t_k) ds4_gpu_tensor_free(t_k);
    if (t_q) ds4_gpu_tensor_free(t_q);
    if (t_out) ds4_gpu_tensor_free(t_out);
cleanup:
    unsetenv("DS4_CUDA_QWEN_WARP_ATTN");
    unsetenv("DS4_CUDA_QWEN_SPLIT_K_ATTN");
    free(split_k); free(warp); free(legacy); free(cpu);
    free(prepared_value_cache); free(prepared_key_cache); free(prepared_q);
    free(value_cache); free(key_cache); free(v); free(k); free(q); free(map);
    return failed;
}

static int run_q4_k_probe(void) {
    enum { INPUTS = 256, ROWS = 8 };
    probe_block_q4_k weights[ROWS];
    float input[INPUTS], input_q8[INPUTS], input_q8_1[INPUTS], input_r8[INPUTS];
    float direct[ROWS] = {0}, gpu[ROWS] = {0}, q8_policy[ROWS] = {0};
    float q8_1_policy[ROWS] = {0}, q8_1_gpu[ROWS] = {0};
    float r8_policy[ROWS] = {0}, r8_gpu[ROWS] = {0};
    memset(weights, 0, sizeof(weights));
    for (uint32_t row = 0; row < ROWS; row++) {
        weights[row].d = 0x2400u;    /* 0.015625 */
        weights[row].dmin = 0x2000u; /* 0.0078125 */
        for (uint32_t i = 0; i < 12; i++)
            weights[row].scales[i] = (uint8_t)(17u * i + 29u * row + 3u);
        for (uint32_t i = 0; i < 128; i++)
            weights[row].qs[i] = (uint8_t)(37u * i + 11u * row + 19u);
    }
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < INPUTS; i++) {
        input[i] = pattern(i, 307, 0.0009f);
        if (fabsf(input[i]) > max_abs) max_abs = fabsf(input[i]);
    }
    const float q8_scale = max_abs / 127.0f;
    for (uint32_t i = 0; i < INPUTS; i++)
        input_q8[i] = nearbyintf(input[i] / q8_scale) * q8_scale;
    quantize_dequantize_q8_1(input, input_q8_1, INPUTS);
    quantize_dequantize_q8_1_r8(input, input_r8, INPUTS);
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t i = 0; i < INPUTS; i++) {
            const float w = q4_k_value(&weights[row], i);
            direct[row] += w * input[i];
            q8_policy[row] += w * input_q8[i];
            q8_1_policy[row] += w * input_q8_1[i];
            r8_policy[row] += w * input_r8[i];
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(gpu));
    int failed = !x || !out;
    if (!failed) {
        failed |= !ds4_gpu_set_model_map(weights, sizeof(weights));
        failed |= !ds4_gpu_tensor_write(x, 0, input, sizeof(input));
        setenv("DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                12u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, gpu, sizeof(gpu));
        setenv("DS4_CUDA_QWEN_DECODE_Q8_1", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                12u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, q8_1_gpu, sizeof(q8_1_gpu));
        unsetenv("DS4_CUDA_QWEN_DECODE_Q8_1");
        unsetenv("DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8");
        setenv("DS4_CUDA_QWEN_DECODE_Q8_1_R8", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                12u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, r8_gpu, sizeof(r8_gpu));
        unsetenv("DS4_CUDA_QWEN_DECODE_Q8_1_R8");
    }
    if (!failed) {
        const probe_metrics implementation = measure(gpu, direct, ROWS);
        const probe_metrics policy = measure(direct, q8_policy, ROWS);
        const probe_metrics q8_1_implementation =
            measure(q8_1_gpu, q8_1_policy, ROWS);
        const probe_metrics r8_implementation = measure(r8_gpu, r8_policy, ROWS);
        const char *kind = classify(implementation, 2e-5, 2e-5);
        printf("{\"probe\":\"q4_k_matvec\",\"comparison\":\"gpu_vs_dequant_f32_oracle\"," 
               "\"values\":%u,\"mae\":%.9g,\"rmse\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               ROWS, implementation.mae, implementation.rmse,
               implementation.max_abs, implementation.cosine, kind);
        printf("{\"probe\":\"q4_k_matvec\",\"comparison\":\"f32_vs_q8_k_activation_policy\"," 
               "\"values\":%u,\"mae\":%.9g,\"rmse\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"arithmetic_policy_delta\"}\n",
               ROWS, policy.mae, policy.rmse, policy.max_abs, policy.cosine);
        const char *q8_1_kind = classify(q8_1_implementation, 2e-5, 2e-5);
        printf("{\"probe\":\"q4_k_matvec\",\"comparison\":\"q8_1_mmvq_vs_q8_1_policy\"," 
               "\"values\":%u,\"mae\":%.9g,\"rmse\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               ROWS, q8_1_implementation.mae, q8_1_implementation.rmse,
               q8_1_implementation.max_abs, q8_1_implementation.cosine,
               q8_1_kind);
        failed |= strcmp(kind, "outside_roundoff_envelope") == 0;
        failed |= strcmp(q8_1_kind, "outside_roundoff_envelope") == 0;
        const char *r8_kind = classify(r8_implementation, 2e-5, 2e-5);
        printf("{\"probe\":\"q4_k_matvec\",\"comparison\":\"q8_1_r8_mmvq_vs_r8_policy\","
               "\"values\":%u,\"mae\":%.9g,\"rmse\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               ROWS, r8_implementation.mae, r8_implementation.rmse,
               r8_implementation.max_abs, r8_implementation.cosine, r8_kind);
        failed |= strcmp(r8_kind, "outside_roundoff_envelope") == 0;
    }
    if (out) ds4_gpu_tensor_free(out);
    if (x) ds4_gpu_tensor_free(x);
    unsetenv("DS4_CUDA_QWEN_DECODE_Q8_1");
    unsetenv("DS4_CUDA_QWEN_DECODE_Q8_1_R8");
    unsetenv("DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8");
    return failed;
}

static int run_q6_k_warp8_probe(void) {
    enum { INPUTS = 256, ROWS = 16 };
    probe_block_q6_k weights[ROWS];
    float input[INPUTS], input_q8_1[INPUTS], input_r8[INPUTS];
    float reference[ROWS] = {0}, warp8[ROWS] = {0};
    float q8_1_policy[ROWS] = {0}, q8_1_gpu[ROWS] = {0};
    float r8_policy[ROWS] = {0}, r8_gpu[ROWS] = {0};
    memset(weights, 0, sizeof(weights));
    for (uint32_t row = 0; row < ROWS; row++) {
        weights[row].d = 0x2400u;
        for (uint32_t i = 0; i < 128; i++)
            weights[row].ql[i] = (uint8_t)(41u * i + 13u * row + 7u);
        for (uint32_t i = 0; i < 64; i++)
            weights[row].qh[i] = (uint8_t)(23u * i + 19u * row + 11u);
        for (uint32_t i = 0; i < 16; i++)
            weights[row].scales[i] = (int8_t)((17u * i + 5u * row) % 63u - 31);
    }
    for (uint32_t i = 0; i < INPUTS; i++)
        input[i] = pattern(i, 911, 0.0007f);
    quantize_dequantize_q8_1(input, input_q8_1, INPUTS);
    quantize_dequantize_q8_1_r8(input, input_r8, INPUTS);
    for (uint32_t row = 0; row < ROWS; row++)
        for (uint32_t i = 0; i < INPUTS; i++)
            q8_1_policy[row] += q6_k_value(&weights[row], i) * input_q8_1[i];
    for (uint32_t row = 0; row < ROWS; row++)
        for (uint32_t i = 0; i < INPUTS; i++)
            r8_policy[row] += q6_k_value(&weights[row], i) * input_r8[i];

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(reference));
    int failed = !x || !out;
    if (!failed) {
        failed |= !ds4_gpu_set_model_map(weights, sizeof(weights));
        failed |= !ds4_gpu_tensor_write(x, 0, input, sizeof(input));
        setenv("DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8", "1", 1);
        setenv("DS4_CUDA_QWEN_NO_DECODE_F32_WARP8", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                14u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, reference, sizeof(reference));
        unsetenv("DS4_CUDA_QWEN_NO_DECODE_F32_WARP8");
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                14u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, warp8, sizeof(warp8));
        setenv("DS4_CUDA_QWEN_DECODE_Q6_Q8_1", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                14u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, q8_1_gpu, sizeof(q8_1_gpu));
        unsetenv("DS4_CUDA_QWEN_DECODE_Q6_Q8_1");
        unsetenv("DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8");
        setenv("DS4_CUDA_QWEN_DECODE_Q8_1_R8", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                14u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, r8_gpu, sizeof(r8_gpu));
        unsetenv("DS4_CUDA_QWEN_DECODE_Q8_1_R8");
    }
    if (!failed) {
        const probe_metrics metric = measure(warp8, reference, ROWS);
        const char *kind = metric.max_abs == 0.0 ? "exact" : "mismatch";
        printf("{\"probe\":\"q6_k_matvec\",\"comparison\":\"warp8_vs_block256\"," 
               "\"values\":%u,\"mae\":%.9g,\"rmse\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               ROWS, metric.mae, metric.rmse, metric.max_abs,
               metric.cosine, kind);
        failed |= metric.max_abs != 0.0;
        const probe_metrics q8_1_metric =
            measure(q8_1_gpu, q8_1_policy, ROWS);
        const char *q8_1_kind = classify(q8_1_metric, 2e-5, 2e-5);
        printf("{\"probe\":\"q6_k_matvec\",\"comparison\":\"q8_1_mmvq_vs_q8_1_policy\"," 
               "\"values\":%u,\"mae\":%.9g,\"rmse\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               ROWS, q8_1_metric.mae, q8_1_metric.rmse,
               q8_1_metric.max_abs, q8_1_metric.cosine, q8_1_kind);
        failed |= strcmp(q8_1_kind, "outside_roundoff_envelope") == 0;
        const probe_metrics r8_metric = measure(r8_gpu, r8_policy, ROWS);
        const char *r8_kind = classify(r8_metric, 2e-5, 2e-5);
        printf("{\"probe\":\"q6_k_matvec\",\"comparison\":\"q8_1_r8_mmvq_vs_r8_policy\","
               "\"values\":%u,\"mae\":%.9g,\"rmse\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               ROWS, r8_metric.mae, r8_metric.rmse, r8_metric.max_abs,
               r8_metric.cosine, r8_kind);
        failed |= strcmp(r8_kind, "outside_roundoff_envelope") == 0;
    }
    if (out) ds4_gpu_tensor_free(out);
    if (x) ds4_gpu_tensor_free(x);
    unsetenv("DS4_CUDA_QWEN_NO_DECODE_F32_WARP8");
    unsetenv("DS4_CUDA_QWEN_DECODE_Q6_Q8_1");
    unsetenv("DS4_CUDA_QWEN_DECODE_Q8_1_R8");
    unsetenv("DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8");
    return failed;
}

static int run_q5_k_warp8_probe(void) {
    enum { INPUTS = 256, ROWS = 16 };
    probe_block_q5_k weights[ROWS];
    float input[INPUTS], input_q8[INPUTS], input_q8_1[INPUTS], input_r8[INPUTS];
    float direct[ROWS] = {0}, block256[ROWS] = {0}, warp8[ROWS] = {0};
    float q8_policy[ROWS] = {0}, q8_gpu[ROWS] = {0};
    float q8_1_policy[ROWS] = {0}, q8_1_gpu[ROWS] = {0};
    float r8_policy[ROWS] = {0}, r8_gpu[ROWS] = {0};
    memset(weights, 0, sizeof(weights));
    for (uint32_t row = 0; row < ROWS; row++) {
        weights[row].d = 0x2400u;
        weights[row].dmin = 0x2000u;
        for (uint32_t i = 0; i < 12; i++)
            weights[row].scales[i] = (uint8_t)(17u * i + 29u * row + 3u);
        for (uint32_t i = 0; i < 32; i++)
            weights[row].qh[i] = (uint8_t)(23u * i + 19u * row + 11u);
        for (uint32_t i = 0; i < 128; i++)
            weights[row].qs[i] = (uint8_t)(37u * i + 11u * row + 19u);
    }
    float signed_max = 0.0f;
    for (uint32_t i = 0; i < INPUTS; i++) {
        input[i] = pattern(i, 1009, 0.0007f);
        if (fabsf(input[i]) > fabsf(signed_max)) signed_max = input[i];
    }
    const float inverse_scale = signed_max != 0.0f ? -127.0f / signed_max : 0.0f;
    const float q8_scale = inverse_scale != 0.0f ? 1.0f / inverse_scale : 0.0f;
    for (uint32_t i = 0; i < INPUTS; i++) {
        int q = inverse_scale != 0.0f ? (int)nearbyintf(inverse_scale * input[i]) : 0;
        if (q > 127) q = 127;
        input_q8[i] = (float)q * q8_scale;
    }
    quantize_dequantize_q8_1(input, input_q8_1, INPUTS);
    quantize_dequantize_q8_1_r8(input, input_r8, INPUTS);
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t i = 0; i < INPUTS; i++) {
            direct[row] += q5_k_value(&weights[row], i) * input[i];
            q8_policy[row] += q5_k_value(&weights[row], i) * input_q8[i];
            q8_1_policy[row] += q5_k_value(&weights[row], i) * input_q8_1[i];
            r8_policy[row] += q5_k_value(&weights[row], i) * input_r8[i];
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(input));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(block256));
    int failed = !x || !out;
    if (!failed) {
        failed |= !ds4_gpu_set_model_map(weights, sizeof(weights));
        failed |= !ds4_gpu_tensor_write(x, 0, input, sizeof(input));
        setenv("DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8", "1", 1);
        setenv("DS4_CUDA_QWEN_NO_DECODE_F32_WARP8", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                13u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, block256, sizeof(block256));
        unsetenv("DS4_CUDA_QWEN_NO_DECODE_F32_WARP8");
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                13u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, warp8, sizeof(warp8));
        setenv("DS4_CUDA_QWEN_DECODE_Q4_Q8", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                13u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, q8_gpu, sizeof(q8_gpu));
        unsetenv("DS4_CUDA_QWEN_DECODE_Q4_Q8");
        setenv("DS4_CUDA_QWEN_DECODE_Q8_1", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                13u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, q8_1_gpu, sizeof(q8_1_gpu));
        unsetenv("DS4_CUDA_QWEN_DECODE_Q8_1");
        unsetenv("DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8");
        setenv("DS4_CUDA_QWEN_DECODE_Q8_1_R8", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(out, weights, sizeof(weights), 0,
                                                13u, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, r8_gpu, sizeof(r8_gpu));
        unsetenv("DS4_CUDA_QWEN_DECODE_Q8_1_R8");
    }
    if (!failed) {
        const probe_metrics implementation = measure(block256, direct, ROWS);
        const probe_metrics optimized = measure(warp8, block256, ROWS);
        const probe_metrics q8_implementation = measure(q8_gpu, q8_policy, ROWS);
        const probe_metrics q8_1_implementation =
            measure(q8_1_gpu, q8_1_policy, ROWS);
        const probe_metrics r8_implementation = measure(r8_gpu, r8_policy, ROWS);
        const char *implementation_kind = classify(implementation, 2e-5, 2e-5);
        const char *optimized_kind = optimized.max_abs == 0.0 ? "exact" : "mismatch";
        printf("{\"probe\":\"q5_k_matvec\",\"comparison\":\"block256_vs_dequant_f32_oracle\","
               "\"values\":%u,\"max_abs\":%.9g,\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               ROWS, implementation.max_abs, implementation.cosine, implementation_kind);
        printf("{\"probe\":\"q5_k_matvec\",\"comparison\":\"warp8_vs_block256\"," 
               "\"values\":%u,\"max_abs\":%.9g,\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               ROWS, optimized.max_abs, optimized.cosine, optimized_kind);
        const char *q8_kind = classify(q8_implementation, 2e-5, 2e-5);
        printf("{\"probe\":\"q5_k_matvec\",\"comparison\":\"q8_gpu_vs_q8_policy\"," 
               "\"values\":%u,\"mae\":%.9g,\"max_abs\":%.9g,\"cosine\":%.12g,"
               "\"classification\":\"%s\"}\n",
               ROWS, q8_implementation.mae, q8_implementation.max_abs,
               q8_implementation.cosine, q8_kind);
        const char *q8_1_kind = classify(q8_1_implementation, 2e-5, 2e-5);
        printf("{\"probe\":\"q5_k_matvec\",\"comparison\":\"q8_1_mmvq_vs_q8_1_policy\"," 
               "\"values\":%u,\"mae\":%.9g,\"max_abs\":%.9g,\"cosine\":%.12g,"
               "\"classification\":\"%s\"}\n",
               ROWS, q8_1_implementation.mae, q8_1_implementation.max_abs,
               q8_1_implementation.cosine, q8_1_kind);
        failed |= strcmp(implementation_kind, "outside_roundoff_envelope") == 0;
        failed |= optimized.max_abs != 0.0;
        failed |= strcmp(q8_kind, "outside_roundoff_envelope") == 0;
        failed |= strcmp(q8_1_kind, "outside_roundoff_envelope") == 0;
        const char *r8_kind = classify(r8_implementation, 2e-5, 2e-5);
        printf("{\"probe\":\"q5_k_matvec\",\"comparison\":\"q8_1_r8_mmvq_vs_r8_policy\","
               "\"values\":%u,\"mae\":%.9g,\"rmse\":%.9g,\"max_abs\":%.9g,"
               "\"cosine\":%.12g,\"classification\":\"%s\"}\n",
               ROWS, r8_implementation.mae, r8_implementation.rmse,
               r8_implementation.max_abs, r8_implementation.cosine, r8_kind);
        failed |= strcmp(r8_kind, "outside_roundoff_envelope") == 0;
    }
    if (out) ds4_gpu_tensor_free(out);
    if (x) ds4_gpu_tensor_free(x);
    unsetenv("DS4_CUDA_QWEN_NO_DECODE_F32_WARP8");
    unsetenv("DS4_CUDA_QWEN_DECODE_Q4_Q8");
    unsetenv("DS4_CUDA_QWEN_DECODE_Q8_1");
    unsetenv("DS4_CUDA_QWEN_DECODE_Q8_1_R8");
    unsetenv("DS4_CUDA_QWEN_NO_DECODE_Q8_1_R8");
    return failed;
}

static uint32_t ud_block_size(uint32_t type) {
    switch (type) {
    case 11u: return sizeof(probe_block_q3_k);
    case 17u: return sizeof(probe_block_iq2_xs);
    case 18u: return sizeof(probe_block_iq3_xxs);
    case 20u: return sizeof(probe_block_iq4_nl);
    case 21u: return sizeof(probe_block_iq3_s);
    case 22u: return sizeof(probe_block_iq2_s);
    case 23u: return sizeof(probe_block_iq4_xs);
    default: return 0u;
    }
}

static uint32_t ud_quant_size(uint32_t type) {
    return type == 20u ? 32u : 256u;
}

static const char *ud_type_name(uint32_t type) {
    switch (type) {
    case 11u: return "q3_k";
    case 17u: return "iq2_xs";
    case 18u: return "iq3_xxs";
    case 20u: return "iq4_nl";
    case 21u: return "iq3_s";
    case 22u: return "iq2_s";
    case 23u: return "iq4_xs";
    default: return "unknown";
    }
}

static float ud_value(uint32_t type, const uint8_t *row, uint32_t i) {
    const uint32_t qk = ud_quant_size(type);
    const uint32_t block_size = ud_block_size(type);
    const uint8_t *block = row + (i / qk) * block_size;
    const uint32_t qi = i % qk;
    switch (type) {
    case 11u: return q3_k_value((const probe_block_q3_k *)block, qi);
    case 17u: return iq2_xs_value((const probe_block_iq2_xs *)block, qi);
    case 18u: return iq3_xxs_value((const probe_block_iq3_xxs *)block, qi);
    case 20u: return iq4_nl_value((const probe_block_iq4_nl *)block, qi);
    case 21u: return iq3_s_value((const probe_block_iq3_s *)block, qi);
    case 22u: return iq2_s_value((const probe_block_iq2_s *)block, qi);
    case 23u: return iq4_xs_value((const probe_block_iq4_xs *)block, qi);
    default: return 0.0f;
    }
}

static void ud_fill_weights(uint32_t type, uint8_t *weights,
                            uint32_t rows, uint32_t inputs) {
    const uint32_t qk = ud_quant_size(type);
    const uint32_t block_size = ud_block_size(type);
    const uint32_t blocks_per_row = inputs / qk;
    const uint32_t row_bytes = blocks_per_row * block_size;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t i = 0; i < row_bytes; i++) {
            weights[(size_t)row * row_bytes + i] =
                (uint8_t)(37u * i + 53u * row + 19u * type + 11u);
        }
        for (uint32_t block = 0; block < blocks_per_row; block++) {
            uint8_t *ptr = weights + (size_t)row * row_bytes +
                           (size_t)block * block_size;
            /* Small, exactly representable positive per-block scales keep the
             * oracle sensitive without overflowing adversarial byte patterns. */
            const uint16_t d = (uint16_t)(0x1800u +
                ((row + 3u * block + type) & 3u) * 0x0100u);
            switch (type) {
            case 11u: ((probe_block_q3_k *)ptr)->d = d; break;
            case 17u: ((probe_block_iq2_xs *)ptr)->d = d; break;
            case 18u: ((probe_block_iq3_xxs *)ptr)->d = d; break;
            case 20u: ((probe_block_iq4_nl *)ptr)->d = d; break;
            case 21u: ((probe_block_iq3_s *)ptr)->d = d; break;
            case 22u: ((probe_block_iq2_s *)ptr)->d = d; break;
            case 23u: ((probe_block_iq4_xs *)ptr)->d = d; break;
            default: break;
            }
        }
    }
}

static float tree_dot_256(uint32_t type, const uint8_t *row,
                          const float *input) {
    float work[256];
    for (uint32_t i = 0; i < 256u; i++)
        work[i] = ud_value(type, row, i) * input[i];
    for (uint32_t stride = 128u; stride; stride >>= 1u)
        for (uint32_t i = 0; i < stride; i++)
            work[i] += work[i + stride];
    return work[0];
}

static int run_ud_format_probe(uint32_t type) {
    enum { INPUTS = 256, ROWS = 7, TOKENS = 8 };
    const uint32_t row_bytes =
        (INPUTS / ud_quant_size(type)) * ud_block_size(type);
    const size_t weight_bytes = (size_t)ROWS * row_bytes;
    uint8_t *weights = malloc(weight_bytes);
    float *input = malloc((size_t)TOKENS * INPUTS * sizeof(float));
    float *input_q8 = malloc((size_t)TOKENS * INPUTS * sizeof(float));
    float *input_r8 = malloc((size_t)TOKENS * INPUTS * sizeof(float));
    float *oracle = calloc((size_t)TOKENS * ROWS, sizeof(float));
    float *oracle_q8 = calloc(ROWS, sizeof(float));
    float *oracle_r8 = calloc(ROWS, sizeof(float));
    float *oracle_r8_rows = calloc(
        (size_t)TOKENS * ROWS, sizeof(float));
    float *decode = calloc(ROWS, sizeof(float));
    float *decode_q8 = calloc(ROWS, sizeof(float));
    float *decode_r8 = calloc(ROWS, sizeof(float));
    float *verify2_r8 = calloc(2u * ROWS, sizeof(float));
    float *verify3_r8 = calloc(3u * ROWS, sizeof(float));
    float *prefill = calloc((size_t)TOKENS * ROWS, sizeof(float));
    float *prefill_f16 = calloc((size_t)TOKENS * ROWS, sizeof(float));
    int failed = !weights || !input || !input_q8 || !input_r8 || !oracle ||
                 !oracle_q8 || !oracle_r8 || !oracle_r8_rows || !decode ||
                 !decode_q8 || !decode_r8 || !verify2_r8 || !verify3_r8 ||
                 !prefill || !prefill_f16;
    if (failed) goto cleanup;

    ud_fill_weights(type, weights, ROWS, INPUTS);
    for (uint32_t tok = 0; tok < TOKENS; tok++) {
        for (uint32_t i = 0; i < INPUTS; i++) {
            input[(size_t)tok * INPUTS + i] =
                pattern(i + 257u * tok, 1709u + 13u * type, 0.0006f);
        }
        quantize_dequantize_q8_1(
            input + (size_t)tok * INPUTS,
            input_q8 + (size_t)tok * INPUTS, INPUTS);
        quantize_dequantize_q8_1_r8(
            input + (size_t)tok * INPUTS,
            input_r8 + (size_t)tok * INPUTS, INPUTS);
        for (uint32_t row = 0; row < ROWS; row++) {
            oracle[(size_t)tok * ROWS + row] = tree_dot_256(
                type, weights + (size_t)row * row_bytes,
                input + (size_t)tok * INPUTS);
            oracle_r8_rows[(size_t)tok * ROWS + row] = tree_dot_256(
                type, weights + (size_t)row * row_bytes,
                input_r8 + (size_t)tok * INPUTS);
        }
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        oracle_q8[row] = tree_dot_256(
            type, weights + (size_t)row * row_bytes, input_q8);
        oracle_r8[row] = tree_dot_256(
            type, weights + (size_t)row * row_bytes, input_r8);
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(
        (size_t)TOKENS * INPUTS * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
        (size_t)TOKENS * ROWS * sizeof(float));
    failed = !x || !out;
    if (!failed) {
        failed |= !ds4_gpu_set_model_map(weights, weight_bytes);
        failed |= !ds4_gpu_tensor_write(
            x, 0, input, (size_t)TOKENS * INPUTS * sizeof(float));
        setenv("DS4_CUDA_QWEN38_NO_DECODE_Q8_1_R8", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(
            out, weights, weight_bytes, 0, type, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(out, 0, decode, ROWS * sizeof(float));

        setenv("DS4_CUDA_QWEN38_DECODE_Q8_1", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(
            out, weights, weight_bytes, 0, type, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(
            out, 0, decode_q8, ROWS * sizeof(float));
        unsetenv("DS4_CUDA_QWEN38_DECODE_Q8_1");

        unsetenv("DS4_CUDA_QWEN38_NO_DECODE_Q8_1_R8");
        setenv("DS4_CUDA_QWEN38_DECODE_Q8_1_R8", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(
            out, weights, weight_bytes, 0, type, INPUTS, ROWS, x, 1u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(
            out, 0, decode_r8, ROWS * sizeof(float));

        failed |= !ds4_gpu_matmul_quant_tensor(
            out, weights, weight_bytes, 0, type, INPUTS, ROWS, x, 2u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(
            out, 0, verify2_r8, 2u * ROWS * sizeof(float));
        failed |= !ds4_gpu_matmul_quant_tensor(
            out, weights, weight_bytes, 0, type, INPUTS, ROWS, x, 3u);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(
            out, 0, verify3_r8, 3u * ROWS * sizeof(float));
        unsetenv("DS4_CUDA_QWEN38_DECODE_Q8_1_R8");

        setenv("DS4_CUDA_QWEN_PREFILL_PEDANTIC", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(
            out, weights, weight_bytes, 0, type, INPUTS, ROWS, x, TOKENS);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(
            out, 0, prefill, (size_t)TOKENS * ROWS * sizeof(float));
        unsetenv("DS4_CUDA_QWEN_PREFILL_PEDANTIC");

        setenv("DS4_CUDA_QWEN38_PREFILL_F16", "1", 1);
        failed |= !ds4_gpu_matmul_quant_tensor(
            out, weights, weight_bytes, 0, type, INPUTS, ROWS, x, TOKENS);
        failed |= !ds4_gpu_synchronize();
        failed |= !ds4_gpu_tensor_read(
            out, 0, prefill_f16,
            (size_t)TOKENS * ROWS * sizeof(float));
        unsetenv("DS4_CUDA_QWEN38_PREFILL_F16");
    }
    if (!failed) {
        const probe_metrics decode_metric = measure(decode, oracle, ROWS);
        const probe_metrics q8_metric = measure(decode_q8, oracle_q8, ROWS);
        const probe_metrics r8_metric = measure(decode_r8, oracle_r8, ROWS);
        const probe_metrics verify2_metric =
            measure(verify2_r8, oracle_r8_rows, 2u * ROWS);
        const probe_metrics verify3_metric =
            measure(verify3_r8, oracle_r8_rows, 3u * ROWS);
        const probe_metrics prefill_metric =
            measure(prefill, oracle, (size_t)TOKENS * ROWS);
        const probe_metrics prefill_f16_metric =
            measure(prefill_f16, oracle, (size_t)TOKENS * ROWS);
        const char *decode_kind = classify(decode_metric, 2e-5, 3e-5);
        const char *q8_kind = classify(q8_metric, 3e-5, 4e-5);
        const char *r8_kind = classify(r8_metric, 5e-5, 5e-5);
        const char *verify2_kind = classify(verify2_metric, 5e-5, 5e-5);
        const char *verify3_kind = classify(verify3_metric, 5e-5, 5e-5);
        const char *prefill_kind = classify(prefill_metric, 3e-5, 4e-5);
        const char *prefill_f16_kind =
            classify(prefill_f16_metric, 3e-3, 4e-3);
        printf("{\"probe\":\"qwen38_ud_matmul\",\"format\":\"%s\","
               "\"comparison\":\"decode_vs_scalar_codebook_oracle\","
               "\"values\":%u,\"max_abs\":%.9g,\"cosine\":%.12g,"
               "\"classification\":\"%s\"}\n",
               ud_type_name(type), ROWS, decode_metric.max_abs,
               decode_metric.cosine, decode_kind);
        printf("{\"probe\":\"qwen38_ud_matmul\",\"format\":\"%s\","
               "\"comparison\":\"q8_1_mmvq_vs_q8_1_policy\","
               "\"values\":%u,\"max_abs\":%.9g,\"cosine\":%.12g,"
               "\"classification\":\"%s\"}\n",
               ud_type_name(type), ROWS, q8_metric.max_abs,
               q8_metric.cosine, q8_kind);
        printf("{\"probe\":\"qwen38_ud_matmul\",\"format\":\"%s\","
               "\"comparison\":\"q8_1_r8_mmvq_vs_r8_policy\","
               "\"values\":%u,\"max_abs\":%.9g,\"cosine\":%.12g,"
               "\"classification\":\"%s\"}\n",
               ud_type_name(type), ROWS, r8_metric.max_abs,
               r8_metric.cosine, r8_kind);
        printf("{\"probe\":\"qwen38_ud_matmul\",\"format\":\"%s\","
               "\"comparison\":\"verify2_q8_1_r8_vs_r8_policy\","
               "\"values\":%u,\"max_abs\":%.9g,\"cosine\":%.12g,"
               "\"classification\":\"%s\"}\n",
               ud_type_name(type), 2u * ROWS, verify2_metric.max_abs,
               verify2_metric.cosine, verify2_kind);
        printf("{\"probe\":\"qwen38_ud_matmul\",\"format\":\"%s\","
               "\"comparison\":\"verify3_q8_1_r8_vs_r8_policy\","
               "\"values\":%u,\"max_abs\":%.9g,\"cosine\":%.12g,"
               "\"classification\":\"%s\"}\n",
               ud_type_name(type), 3u * ROWS, verify3_metric.max_abs,
               verify3_metric.cosine, verify3_kind);
        printf("{\"probe\":\"qwen38_ud_matmul\",\"format\":\"%s\","
               "\"comparison\":\"prefill_dequant_gemm_vs_scalar_codebook_oracle\","
               "\"values\":%u,\"max_abs\":%.9g,\"cosine\":%.12g,"
               "\"classification\":\"%s\"}\n",
               ud_type_name(type), TOKENS * ROWS, prefill_metric.max_abs,
               prefill_metric.cosine, prefill_kind);
        printf("{\"probe\":\"qwen38_ud_matmul\",\"format\":\"%s\","
               "\"comparison\":\"prefill_f16_gemm_vs_scalar_codebook_oracle\","
               "\"values\":%u,\"max_abs\":%.9g,\"cosine\":%.12g,"
               "\"classification\":\"%s\"}\n",
               ud_type_name(type), TOKENS * ROWS, prefill_f16_metric.max_abs,
               prefill_f16_metric.cosine, prefill_f16_kind);
        failed |= strcmp(decode_kind, "outside_roundoff_envelope") == 0;
        failed |= strcmp(q8_kind, "outside_roundoff_envelope") == 0;
        failed |= strcmp(r8_kind, "outside_roundoff_envelope") == 0;
        failed |= strcmp(verify2_kind, "outside_roundoff_envelope") == 0;
        failed |= strcmp(verify3_kind, "outside_roundoff_envelope") == 0;
        failed |= strcmp(prefill_kind, "outside_roundoff_envelope") == 0;
        failed |= strcmp(prefill_f16_kind,
                         "outside_roundoff_envelope") == 0;
    }
    if (out) ds4_gpu_tensor_free(out);
    if (x) ds4_gpu_tensor_free(x);

cleanup:
    unsetenv("DS4_CUDA_QWEN_PREFILL_PEDANTIC");
    unsetenv("DS4_CUDA_QWEN38_DECODE_Q8_1");
    unsetenv("DS4_CUDA_QWEN38_DECODE_Q8_1_R8");
    unsetenv("DS4_CUDA_QWEN38_NO_DECODE_Q8_1_R8");
    unsetenv("DS4_CUDA_QWEN38_PREFILL_F16");
    free(prefill_f16);
    free(prefill);
    free(verify3_r8);
    free(verify2_r8);
    free(decode_r8);
    free(decode_q8);
    free(decode);
    free(oracle_r8);
    free(oracle_r8_rows);
    free(oracle_q8);
    free(oracle);
    free(input_r8);
    free(input_q8);
    free(input);
    free(weights);
    return failed;
}

static int run_qwen38_ud_probes(void) {
    static const uint32_t types[] = {11u, 17u, 18u, 20u, 21u, 22u, 23u};
    int failed = 0;
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++)
        failed |= run_ud_format_probe(types[i]);
    return failed;
}

int main(void) {
    if (!ds4_gpu_init()) return 2;
    int failed = run_gdn_probe();
    failed |= run_full_attention_probe();
    failed |= run_q4_k_probe();
    failed |= run_q5_k_warp8_probe();
    failed |= run_q6_k_warp8_probe();
    failed |= run_qwen38_ud_probes();
    ds4_gpu_cleanup();
    printf("{\"probe\":\"summary\",\"status\":\"%s\"}\n",
           failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
