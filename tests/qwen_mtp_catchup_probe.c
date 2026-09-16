/* Model-backed regression for the shifted MTP carry. Include the runtime to
 * inspect hidden/KV without a public test API or tracing (which forces logits
 * and used to hide this bug). Run with target GGUF and MTP sidecar arguments. */
#include "../ds4.c"

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); exit(1); \
} } while (0)

static float *read_tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    float *p = malloc(bytes);
    CHECK(p && ds4_gpu_tensor_read(t, 0, p, bytes));
    for (uint64_t i = 0; i < bytes / sizeof(float); i++) CHECK(isfinite(p[i]));
    return p;
}

static void check_tensor(const ds4_gpu_tensor *t, const float *ref,
                         uint64_t bytes) {
    float *p = read_tensor(t, bytes);
    CHECK(memcmp(p, ref, bytes) == 0);
    free(p);
}

static void check_step(ds4_session *s, uint32_t token, uint32_t pos) {
    ds4_engine *e = s->engine;
    ds4_qwen_gpu_graph *g = &s->qwen_graph;
    const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *expected = ds4_gpu_tensor_alloc(bytes);
    CHECK(expected);
    /* Poison stale output scratch and verify that the expensive output head
     * remains skipped while the target hidden is refreshed. */
    CHECK(ds4_gpu_tensor_fill_f32(g->output_norm, -17.0f, DS4_N_EMBD));
    CHECK(ds4_gpu_tensor_fill_f32(g->logits, -19.0f, DS4_N_VOCAB));
    CHECK(qwen_graph_forward_token_mode(g, &e->model, &e->weights,
          token, pos, DS4_QWEN_STAGE_PREFILL, NULL, false));
    CHECK(ds4_gpu_rms_norm_weight_tensor(expected, g->cur,
          e->model.map, e->model.size, e->weights.output_norm->abs_offset,
          DS4_N_EMBD, DS4_RMS_EPS));
    float *ref = read_tensor(expected, bytes);
    check_tensor(g->output_norm, ref, bytes);
    float *logits = read_tensor(g->logits, (uint64_t)DS4_N_VOCAB * sizeof(float));
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) CHECK(logits[i] == -19.0f);
    free(logits);

    /* Independent expected cache entry from (token[p], saved target_h[p-1]).
     * The production catch-up must use this pair and only then advance carry. */
    CHECK(qwen_graph_mtp_step(g, &e->mtp_model, &e->qwen_mtp_weights,
          token, pos, g->mtp_pending_h, NULL, NULL, NULL,
          DS4_QWEN_STAGE_MTP_CATCHUP));
    const uint64_t kv_bytes = (uint64_t)(pos + 1) * DS4_N_HEAD_KV *
                              DS4_N_HEAD_DIM * sizeof(float);
    float *key = read_tensor(g->mtp_key_cache, kv_bytes);
    float *value = read_tensor(g->mtp_value_cache, kv_bytes);
    CHECK(qwen_graph_mtp_catchup(g, &e->mtp_model, &e->qwen_mtp_weights, token, pos));
    check_tensor(g->mtp_key_cache, key, kv_bytes);
    check_tensor(g->mtp_value_cache, value, kv_bytes);
    check_tensor(g->mtp_pending_h, ref, bytes);
    CHECK(g->mtp_cache_len == pos + 1);
    free(key); free(value); free(ref);
    ds4_gpu_tensor_free(expected);
    printf("{\"test\":\"mtp_carry_without_logits_or_trace\",\"pos\":%u,"
           "\"bit_exact\":true,\"output_head_skipped\":true}\n", pos);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s TARGET.gguf MTP.gguf\n", argv[0]);
        return 2;
    }
    CHECK(getenv("DS4_QWEN_TRACE_DIR") == NULL);
    ds4_engine *e = NULL;
    ds4_session *s = NULL;
    ds4_engine_options opts = {
        .model_path = argv[1], .mtp_path = argv[2],
        .backend = DS4_BACKEND_CUDA, .n_threads = 8,
        .context_size = 1024, .prefill_chunk = 512, .mtp_draft_tokens = 2,
    };
    CHECK(ds4_engine_open(&e, &opts) == 0 && ds4_engine_has_mtp(e));
    CHECK(ds4_session_create(&s, e, opts.context_size) == 0);
    ds4_tokens pattern = {0}, tokens = {0};
    ds4_tokenize_text(e, "The capital of Italy is Rome. Write a Python function.\n", &pattern);
    CHECK(pattern.len > 0);
    for (int i = 0; i < 514; i++) ds4_tokens_push(&tokens, pattern.v[i % pattern.len]);
    CHECK(qwen_graph_reset(&s->qwen_graph));
    check_step(s, tokens.v[0], 0);
    check_step(s, tokens.v[1], 1);
    CHECK(qwen_graph_reset(&s->qwen_graph));
    CHECK(qwen_graph_forward_rows(&s->qwen_graph, &e->model, &e->weights,
          tokens.v, 0, 512, DS4_QWEN_STAGE_PREFILL, s->logits, NULL));
    CHECK(qwen_graph_mtp_catchup_rows(&s->qwen_graph, &e->model, &e->weights,
          &e->mtp_model, &e->qwen_mtp_weights, tokens.v, 0, 512));
    check_step(s, tokens.v[512], 512);
    check_step(s, tokens.v[513], 513);
    ds4_tokens_free(&pattern);
    ds4_tokens_free(&tokens);
    ds4_session_free(s);
    ds4_engine_close(e);
    return 0;
}
