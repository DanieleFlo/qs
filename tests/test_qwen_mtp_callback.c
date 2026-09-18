/* The server's constrained sampler may decline a verifier row at a
 * thinking/tool boundary. Exercise both the draft and bonus row at V(2). */
#include "../ds4.c"

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL line %d: %s (%s)\n", __LINE__, #c, err); exit(1); \
} } while (0)

typedef struct { int draft; int abort_row; int calls; } callback_state;
static int sample_row(void *ud, const float *logits, int vocab,
                      const int *prefix, int len, uint64_t *rng) {
    (void)logits; (void)prefix;
    callback_state *state = ud;
    state->calls++;
    if (state->abort_row == 0) return (state->draft + 1) % vocab;
    *rng += 123; /* A declined row must also roll back sampling state. */
    return len == state->abort_row ? -1 : state->draft;
}

static int save_state(ds4_session *s, ds4_session_snapshot *snap,
                      char *err, size_t errlen) {
    FILE *fp = tmpfile();
    if (!fp) return -1;
    int rc = ds4_session_save_payload(s, fp, err, errlen);
    snap->len = ds4_session_payload_bytes(s);
    snap->ptr = malloc(snap->len);
    if (rc || !snap->ptr || fseek(fp, 0, SEEK_SET) ||
        fread(snap->ptr, 1, snap->len, fp) != snap->len) rc = -1;
    fclose(fp);
    return rc;
}
static int load_state(ds4_session *s, const ds4_session_snapshot *snap,
                      char *err, size_t errlen) {
    FILE *fp = fmemopen(snap->ptr, snap->len, "rb");
    if (!fp) return -1;
    int rc = ds4_session_load_payload(s, fp, snap->len, err, errlen);
    fclose(fp);
    return rc;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char err[256] = {0};
    ds4_engine *e = NULL;
    ds4_session *s = NULL;
    ds4_engine_options options = {
        .model_path = argv[1], .mtp_path = argv[2],
        .backend = DS4_BACKEND_CUDA, .context_size = 4096,
        .prefill_chunk = 2048, .mtp_draft_tokens = 2,
    };
    CHECK(getenv("DS4_MTP_QWEN_V2_SAFE_SNAPSHOT") == NULL);
    CHECK(ds4_engine_open(&e, &options) == 0 && ds4_engine_has_mtp(e));
    CHECK(ds4_session_create(&s, e, options.context_size) == 0);
    ds4_tokens pattern = {0}, prompt = {0};
    ds4_tokenize_text(e, "Test a skill and its tool, then return the result.\n", &pattern);
    CHECK(pattern.len > 0);
    for (unsigned i = 0; i < DS4_QWEN_MTP_DEPTH1_CONTEXT + 16; i++)
        ds4_tokens_push(&prompt, pattern.v[i % pattern.len]);
    CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    CHECK(s->qwen_graph.mtp_ready);
    ds4_session_snapshot saved = {0};
    CHECK(save_state(s, &saved, err, sizeof(err)) == 0);
    int first = ds4_session_argmax(s), draft = -1;
    ds4_qwen_gpu_graph *g = &s->qwen_graph;
    CHECK(qwen_graph_mtp_step(g, &e->mtp_model, &e->qwen_mtp_weights,
          first, prompt.len, g->mtp_pending_h, g->mtp_draft_h[0], NULL,
          &draft, DS4_QWEN_STAGE_MTP_DRAFT));
    ds4_session_snapshot expected = {0};
    for (int abort_row = 0; abort_row <= 2; abort_row++) {
        CHECK(load_state(s, &saved, err, sizeof(err)) == 0);
        callback_state state = {.draft = draft, .abort_row = abort_row};
        uint64_t rng = 42;
        int accepted[3] = {0}, next = -1;
        int count = ds4_session_eval_speculative_constrained_sample(
            s, first, 1.0f, 0, 1.0f, 0.0f, &rng, 3, -1,
            accepted, 3, &next, sample_row, &state, err, sizeof(err));
        CHECK(state.calls == (abort_row == 2 ? 2 : 1));
        CHECK(count == 1 && accepted[0] == first);
        CHECK(ds4_session_pos(s) == prompt.len + 1);
        CHECK(rng == 42);
        CHECK(g->mtp_ready && g->mtp_cache_len == (uint32_t)prompt.len + 1);
        if (abort_row) CHECK(next == -1);
        /* mtp_logits is verifier scratch while the draft-valid flag is false.
         * Canonicalize only that unused host buffer before comparing payloads. */
        CHECK(!s->mtp_draft_valid);
        memset(s->mtp_logits, 0, (size_t)DS4_N_VOCAB * sizeof(float));
        ds4_session_snapshot actual = {0};
        CHECK(save_state(s, &actual, err, sizeof(err)) == 0);
        if (!abort_row) expected = actual;
        else {
            /* Full state, not just output: recurrent, attention, logits and MTP
             * must equal an ordinary rejected-draft commit at the same row. */
            CHECK(actual.len == expected.len);
            CHECK(memcmp(actual.ptr, expected.ptr, actual.len) == 0);
            ds4_session_snapshot_free(&actual);
        }
    }
    fprintf(stderr, "PASS: MTP callback cancellation preserves KV, MTP and RNG\n");
    ds4_session_snapshot_free(&expected);
    ds4_session_snapshot_free(&saved);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&pattern);
    ds4_session_free(s);
    ds4_engine_close(e);
    return 0;
}
