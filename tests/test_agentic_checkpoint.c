#include "ds4.h"
#include "ds4_kvstore.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define CHECK(c, ...) do {                                                   \
    if (!(c)) {                                                              \
        fprintf(stderr, "FAIL line %d: ", __LINE__);                       \
        fprintf(stderr, __VA_ARGS__);                                        \
        fputc('\n', stderr);                                                 \
        goto fail;                                                           \
    }                                                                        \
} while (0)

typedef struct {
    bool cancelled;
} cancel_state;

static bool cancel_now(void *ud) {
    cancel_state *state = ud;
    return state && state->cancelled;
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void append_tokens(ds4_tokens *dst, const ds4_tokens *pattern, int n) {
    for (int i = 0; i < n; i++) {
        ds4_tokens_push(dst, pattern->v[i % pattern->len]);
    }
}

static void append_all(ds4_tokens *dst, const ds4_tokens *src) {
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}

static bool tokens_equal(const ds4_tokens *a, const ds4_tokens *b) {
    return a && b && a->len == b->len &&
           (a->len == 0 || !memcmp(a->v, b->v,
                                   (size_t)a->len * sizeof(a->v[0])));
}

static int sync_session(ds4_session *session, const ds4_tokens *tokens,
                        const char *label) {
    char err[256] = {0};
    int rc = ds4_session_sync(session, tokens, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "FAIL: %s sync rc=%d: %s\n", label, rc, err);
        return 1;
    }
    return 0;
}

static int compare_with_canonical(ds4_engine *engine, ds4_session *actual,
                                  const ds4_tokens *prefix,
                                  const ds4_tokens *expected, int ctx,
                                  const char *label) {
    ds4_session *canonical = NULL;
    float *a = NULL;
    float *b = NULL;
    int rc = 1;
    const int vocab = ds4_engine_vocab_size(engine);
    if (ds4_session_create(&canonical, engine, ctx) != 0) goto done;
    if (prefix && sync_session(canonical, prefix, label) != 0) goto done;
    if (sync_session(canonical, expected, label) != 0) goto done;
    if (!tokens_equal(ds4_session_tokens(actual), expected)) {
        fprintf(stderr, "FAIL: %s token timeline mismatch\n", label);
        goto done;
    }
    a = malloc((size_t)vocab * sizeof(*a));
    b = malloc((size_t)vocab * sizeof(*b));
    if (!a || !b) goto done;
    if (ds4_session_copy_logits(actual, a, vocab) != vocab ||
        ds4_session_copy_logits(canonical, b, vocab) != vocab) goto done;
    if (memcmp(a, b, (size_t)vocab * sizeof(*a)) != 0) {
        size_t mismatches = 0;
        float max_abs = 0.0f;
        for (int i = 0; i < vocab; i++) {
            if (memcmp(&a[i], &b[i], sizeof(float)) != 0) mismatches++;
            float d = fabsf(a[i] - b[i]);
            if (d > max_abs) max_abs = d;
        }
        fprintf(stderr,
                "FAIL: %s logits not bit-exact mismatches=%zu max_abs=%g\n",
                label, mismatches, max_abs);
        goto done;
    }
    fprintf(stderr, "agentic-checkpoint: %s bit-exact tokens=%d logits=%d\n",
            label, expected->len, vocab);
    rc = 0;
done:
    free(a);
    free(b);
    ds4_session_free(canonical);
    return rc;
}

static int save_checkpoint(ds4_session *session, const char *path,
                           ds4_skill_state_metrics *metrics) {
    char err[256] = {0};
    FILE *fp = fopen(path, "wb");
    if (!fp) return 1;
    uint64_t bytes = 0;
    int rc = ds4_session_save_skill_state(session, fp, &bytes, metrics,
                                          err, sizeof(err));
    if (rc == 0 && fflush(fp) != 0) rc = 1;
    if (rc == 0 && fsync(fileno(fp)) != 0) rc = 1;
    if (fclose(fp) != 0 && rc == 0) rc = 1;
    if (rc != 0) fprintf(stderr, "checkpoint save rc=%d: %s\n", rc, err);
    if (metrics && metrics->checkpoint_bytes != bytes) return 1;
    return rc;
}

static int load_checkpoint(ds4_session *session, const char *path,
                           const ds4_tokens *frontier,
                           ds4_skill_state_metrics *metrics) {
    char err[256] = {0};
    FILE *fp = fopen(path, "rb");
    if (!fp) return errno == ENOENT ? -ENOENT : -errno;
    int rc = ds4_session_load_skill_state(session, fp, frontier, metrics,
                                          err, sizeof(err));
    fclose(fp);
    if (rc != 0) fprintf(stderr, "checkpoint load rc=%d: %s\n", rc, err);
    return rc;
}

static int save_session_payload(ds4_session *session, const char *path,
                                uint64_t *payload_bytes) {
    char err[256] = {0};
    const uint64_t expected = ds4_session_payload_bytes(session);
    FILE *fp = fopen(path, "wb");
    if (!fp || expected == 0) {
        if (fp) fclose(fp);
        return 1;
    }
    int rc = ds4_session_save_payload(session, fp, err, sizeof(err));
    if (rc == 0 && fflush(fp) != 0) rc = 1;
    if (rc == 0 && fsync(fileno(fp)) != 0) rc = 1;
    if (fclose(fp) != 0 && rc == 0) rc = 1;
    if (rc != 0) fprintf(stderr, "session payload save rc=%d: %s\n", rc, err);
    if (rc == 0 && payload_bytes) *payload_bytes = expected;
    return rc;
}

static int load_session_payload(ds4_session *session, const char *path,
                                uint64_t payload_bytes) {
    char err[256] = {0};
    FILE *fp = fopen(path, "rb");
    if (!fp) return 1;
    int rc = ds4_session_load_payload(session, fp, payload_bytes,
                                      err, sizeof(err));
    fclose(fp);
    if (rc != 0) fprintf(stderr, "session payload load rc=%d: %s\n", rc, err);
    return rc;
}

static int copy_file(const char *src, const char *dst, bool truncate_half,
                     bool corrupt) {
    FILE *in = fopen(src, "rb");
    FILE *out = fopen(dst, "wb");
    if (!in || !out) {
        if (in) fclose(in);
        if (out) fclose(out);
        return 1;
    }
    fseek(in, 0, SEEK_END);
    long size = ftell(in);
    rewind(in);
    long limit = truncate_half ? size / 2 : size;
    unsigned char buf[65536];
    long copied = 0;
    while (copied < limit) {
        size_t want = sizeof(buf);
        if ((long)want > limit - copied) want = (size_t)(limit - copied);
        size_t got = fread(buf, 1, want, in);
        if (!got) break;
        if (corrupt && copied <= 4096 && copied + (long)got > 4096) {
            buf[4096 - copied] ^= 0x5a;
            corrupt = false;
        }
        if (fwrite(buf, 1, got, out) != got) break;
        copied += (long)got;
    }
    int rc = copied == limit ? 0 : 1;
    if (fclose(in) != 0) rc = 1;
    if (fclose(out) != 0) rc = 1;
    return rc;
}

static int assert_unchanged(ds4_session *session, const ds4_tokens *tokens,
                            const float *logits, int vocab,
                            const char *label) {
    float *now = malloc((size_t)vocab * sizeof(*now));
    if (!now) return 1;
    int rc = 0;
    if (!tokens_equal(ds4_session_tokens(session), tokens) ||
        ds4_session_copy_logits(session, now, vocab) != vocab ||
        memcmp(now, logits, (size_t)vocab * sizeof(*now)) != 0) {
        fprintf(stderr, "FAIL: %s changed live state\n", label);
        rc = 1;
    }
    free(now);
    return rc;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    const char *mtp = getenv("DS4_TEST_MTP");
    const char *base = getenv("DS4_AGENTIC_TEST_DIR");
    const bool fast = getenv("DS4_AGENTIC_FAST") != NULL;
    if (!model || !model[0]) {
        fprintf(stderr, "DS4_TEST_MODEL is required\n");
        return 2;
    }
    char dir_template[4096];
    snprintf(dir_template, sizeof(dir_template), "%s/agentic-model.XXXXXX",
             base && base[0] ? base : "/tmp");
    char *dir = mkdtemp(dir_template);
    if (!dir) {
        perror("mkdtemp");
        return 2;
    }

    ds4_engine *engine = NULL;
    ds4_session *live = NULL;
    ds4_session *isolated = NULL;
    ds4_tokens pattern = {0}, call = {0}, result_pattern = {0};
    ds4_tokens frontier = {0}, child = {0}, returned = {0};
    float *frontier_logits = NULL;
    float *child_logits = NULL;
    int rc = 1;
    char checkpoint[4096], corrupt_path[4096], truncated_path[4096];
    char session_path[4096];
    snprintf(checkpoint, sizeof(checkpoint), "%s/frontier.dsk", dir);
    snprintf(corrupt_path, sizeof(corrupt_path), "%s/corrupt.dsk", dir);
    snprintf(truncated_path, sizeof(truncated_path), "%s/truncated.dsk", dir);
    snprintf(session_path, sizeof(session_path), "%s/session.dsv", dir);

    ds4_engine_options options = {
        .model_path = model,
        .mtp_path = mtp && mtp[0] ? mtp : NULL,
        .mtp_draft_tokens = mtp && mtp[0] ? 4 : 0,
        .backend = DS4_BACKEND_CUDA,
        .prefill_chunk = 2048,
    };
    CHECK(ds4_engine_open(&engine, &options) == 0, "engine open failed");
    const bool mtp_enabled = ds4_engine_has_mtp(engine);
    const int parent_count = fast ? 128 : (mtp_enabled ? 256 : 10000);
    const int instruction_count = fast ? 32 : (mtp_enabled ? 64 : 500);
    const int child_count = fast ? 64 : (mtp_enabled ? 128 : 2000);
    const int result_count = fast ? 16 : (mtp_enabled ? 32 : 200);
    const int main_ctx = fast ? 1024 : (mtp_enabled ? 4096 : 16384);
    CHECK(ds4_session_create(&live, engine, main_ctx) == 0,
          "live session create failed");
    ds4_tokenize_text(engine, " parent-memory deterministic block", &pattern);
    ds4_tokenize_text(engine, "<tool-call name=skill-A call-id=recursive>", &call);
    ds4_tokenize_text(engine, " skill-result deterministic value", &result_pattern);
    CHECK(pattern.len > 0 && call.len > 0 && result_pattern.len > 0,
          "token patterns are empty");

    append_tokens(&frontier, &pattern, parent_count);
    append_all(&frontier, &call);
    const double parent_t0 = monotonic_ms();
    CHECK(sync_session(live, &frontier, "parent") == 0,
          "parent sync failed");
    const double parent_ms = monotonic_ms() - parent_t0;

    ds4_skill_state_metrics save_metrics = {0};
    CHECK(save_checkpoint(live, checkpoint, &save_metrics) == 0,
          "checkpoint save failed");
    const int vocab = ds4_engine_vocab_size(engine);
    frontier_logits = malloc((size_t)vocab * sizeof(*frontier_logits));
    child_logits = malloc((size_t)vocab * sizeof(*child_logits));
    CHECK(frontier_logits && child_logits, "logit allocation failed");
    CHECK(ds4_session_copy_logits(live, frontier_logits, vocab) == vocab,
          "frontier logits copy failed");

    ds4_tokens_copy(&child, &frontier);
    append_tokens(&child, &pattern, instruction_count);
    const double instructions_t0 = monotonic_ms();
    CHECK(sync_session(live, &child, "skill instructions") == 0,
          "instruction sync failed");
    const double instructions_ms = monotonic_ms() - instructions_t0;
    CHECK(ds4_session_pos(live) - frontier.len == instruction_count,
          "expected exactly %d instruction prefill tokens", instruction_count);

    ds4_skill_state_metrics load_metrics = {0};
    CHECK(load_checkpoint(live, checkpoint, &frontier, &load_metrics) == 0,
          "restore after instructions failed");
    CHECK(ds4_session_pos(live) == frontier.len,
          "restore did not reset frontier length");
    CHECK(ds4_session_copy_logits(live, child_logits, vocab) == vocab &&
          !memcmp(frontier_logits, child_logits,
                  (size_t)vocab * sizeof(*child_logits)),
          "frontier logits were not restored bit-exact");

    ds4_tokens_free(&child);
    ds4_tokens_copy(&child, &frontier);
    append_tokens(&child, &pattern, child_count);
    CHECK(sync_session(live, &child, "child") == 0, "child sync failed");
    const int discarded_child = ds4_session_pos(live) - frontier.len;
    CHECK(discarded_child == child_count,
          "expected %d child tokens, got %d", child_count, discarded_child);
    CHECK(load_checkpoint(live, checkpoint, &frontier, &load_metrics) == 0,
          "restore after 2k child failed");
    ds4_tokens_copy(&returned, &frontier);
    append_tokens(&returned, &result_pattern, result_count);
    const double result_t0 = monotonic_ms();
    CHECK(sync_session(live, &returned, "result") == 0,
          "result sync failed");
    const double result_ms = monotonic_ms() - result_t0;
    CHECK(ds4_session_pos(live) - frontier.len == result_count,
          "expected exactly %d result prefill tokens", result_count);
    CHECK(compare_with_canonical(engine, live, &frontier, &returned, main_ctx,
                                 "return canonical") == 0,
          "long canonical comparison failed");

    /* Cross-session replay must fail before touching the other session. */
    CHECK(ds4_session_create(&isolated, engine, main_ctx) == 0,
          "isolated session create failed");
    CHECK(sync_session(isolated, &child, "isolated child") == 0,
          "isolated sync failed");
    ds4_tokens isolated_tokens = {0};
    ds4_tokens_copy(&isolated_tokens, ds4_session_tokens(isolated));
    CHECK(ds4_session_copy_logits(isolated, child_logits, vocab) == vocab,
          "isolated logits copy failed");
    CHECK(load_checkpoint(isolated, checkpoint, &frontier, NULL) != 0,
          "cross-session checkpoint unexpectedly loaded");
    CHECK(assert_unchanged(isolated, &isolated_tokens, child_logits, vocab,
                           "cross-session rejection") == 0,
          "cross-session rejection mutated state");
    ds4_tokens_free(&isolated_tokens);
    ds4_session_free(isolated);
    isolated = NULL;

    /* Corruption and truncation are rejected transactionally. */
    CHECK(copy_file(checkpoint, corrupt_path, false, true) == 0,
          "corrupt fixture copy failed");
    CHECK(copy_file(checkpoint, truncated_path, true, false) == 0,
          "truncated fixture copy failed");
    CHECK(ds4_session_copy_logits(live, child_logits, vocab) == vocab,
          "live logits copy failed");
    ds4_tokens live_tokens = {0};
    ds4_tokens_copy(&live_tokens, ds4_session_tokens(live));
    CHECK(load_checkpoint(live, corrupt_path, &frontier, NULL) != 0,
          "corrupt checkpoint unexpectedly loaded");
    CHECK(assert_unchanged(live, &live_tokens, child_logits, vocab,
                           "corrupt rejection") == 0,
          "corrupt rejection mutated state");
    CHECK(load_checkpoint(live, truncated_path, &frontier, NULL) != 0,
          "truncated checkpoint unexpectedly loaded");
    CHECK(assert_unchanged(live, &live_tokens, child_logits, vocab,
                           "truncated rejection") == 0,
          "truncated rejection mutated state");

    cancel_state cancel = {.cancelled = true};
    ds4_session_set_cancel(live, cancel_now, &cancel);
    CHECK(load_checkpoint(live, checkpoint, &frontier, NULL) ==
              DS4_SESSION_SYNC_INTERRUPTED,
          "cancelled restore did not report interruption");
    CHECK(assert_unchanged(live, &live_tokens, child_logits, vocab,
                           "cancelled restore") == 0,
          "cancelled restore mutated state");
    FILE *cancel_fp = fopen("/dev/null", "wb");
    uint64_t cancel_bytes = 0;
    char cancel_err[160] = {0};
    CHECK(cancel_fp != NULL, "cancel save stream open failed");
    CHECK(ds4_session_save_skill_state(live, cancel_fp, &cancel_bytes, NULL,
                                       cancel_err, sizeof(cancel_err)) ==
              DS4_SESSION_SYNC_INTERRUPTED,
          "cancelled save did not report interruption");
    fclose(cancel_fp);
    ds4_session_set_cancel(live, NULL, NULL);
    ds4_tokens_free(&live_tokens);

    /* Three nested recursive frames, restored in reverse order. */
    ds4_tokens nested = {0};
    append_tokens(&nested, &pattern, 256);
    const char *nested_paths[3] = {0};
    char nested_buf[3][4096];
    ds4_tokens nested_frontier[3] = {{0}};
    for (int level = 0; level < 3; level++) {
        append_all(&nested, &call); /* same skill name: recursive instances */
        CHECK(sync_session(live, &nested, "nested open") == 0,
              "nested open failed");
        ds4_tokens_copy(&nested_frontier[level], &nested);
        snprintf(nested_buf[level], sizeof(nested_buf[level]),
                 "%s/nested-%d.dsk", dir, level);
        nested_paths[level] = nested_buf[level];
        CHECK(save_checkpoint(live, nested_paths[level], NULL) == 0,
              "nested checkpoint %d save failed", level);
        append_tokens(&nested, &pattern, 32);
    }
    append_tokens(&nested, &pattern, 64);
    CHECK(sync_session(live, &nested, "nested leaf") == 0,
          "nested leaf sync failed");
    for (int level = 2; level >= 0; level--) {
        CHECK(load_checkpoint(live, nested_paths[level],
                              &nested_frontier[level], NULL) == 0,
              "nested restore %d failed", level);
        ds4_tokens expected = {0};
        ds4_tokens_copy(&expected, &nested_frontier[level]);
        append_tokens(&expected, &result_pattern, 7 + level);
        CHECK(sync_session(live, &expected, "nested return") == 0,
              "nested return %d failed", level);
        CHECK(compare_with_canonical(engine, live, &nested_frontier[level],
                                     &expected, main_ctx,
                                     "nested return canonical") == 0,
              "nested canonical %d failed", level);
        ds4_tokens_free(&expected);
    }
    for (int level = 0; level < 3; level++) {
        ds4_tokens_free(&nested_frontier[level]);
        unlink(nested_paths[level]);
    }
    ds4_tokens_free(&nested);

    /* The general Qwen Q4_K_S payload is portable across session objects and
     * restores all continuation-dependent state, not only target KV rows. */
    CHECK(ds4_engine_is_qwen36_q4_k_s(engine),
          "agentic checkpoint gate requires Qwen3.6 Q4_K_S");
    uint64_t session_payload_bytes = 0;
    CHECK(save_session_payload(live, session_path, &session_payload_bytes) == 0,
          "portable session save failed");
    CHECK(ds4_session_create(&isolated, engine, main_ctx) == 0,
          "portable destination create failed");
    CHECK(load_session_payload(isolated, session_path,
                               session_payload_bytes) == 0,
          "portable session load failed");
    ds4_tokens portable_expected = {0};
    ds4_tokens_copy(&portable_expected, ds4_session_tokens(live));
    CHECK(ds4_session_copy_logits(live, frontier_logits, vocab) == vocab,
          "portable source logits copy failed");
    CHECK(assert_unchanged(isolated, &portable_expected, frontier_logits, vocab,
                           "portable restore") == 0,
          "portable restore was not bit-exact");
    append_tokens(&portable_expected, &result_pattern, 16);
    CHECK(sync_session(live, &portable_expected, "portable source continuation") == 0,
          "portable source continuation failed");
    CHECK(sync_session(isolated, &portable_expected,
                       "portable restored continuation") == 0,
          "portable restored continuation failed");
    CHECK(ds4_session_copy_logits(live, frontier_logits, vocab) == vocab &&
          ds4_session_copy_logits(isolated, child_logits, vocab) == vocab &&
          !memcmp(frontier_logits, child_logits,
                  (size_t)vocab * sizeof(*child_logits)),
          "portable continuation logits were not bit-exact");

    /* Exercise the actual content-addressed cache path. This catches Qwen's
     * dense Q4_K_S quant tag (there is no routed-expert tensor to inspect) and
     * proves that a fresh/page-reloaded session can restore the visible key. */
    ds4_kvstore store = {0};
    ds4_kvstore_options store_options = ds4_kvstore_default_options();
    store_options.min_tokens = 1;
    CHECK(ds4_kvstore_open(&store, dir, 1024, true, store_options,
                           "agentic-reload-test", NULL, NULL),
          "portable kvstore open failed");
    const ds4_tokens *portable_live = ds4_session_tokens(live);
    CHECK(ds4_kvstore_store_live_prefix(&store, engine, live, portable_live,
                                        portable_live->len, "agent-session",
                                        NULL, NULL, 0),
          "Qwen visible session cache store failed");
    size_t visible_len = 0;
    char *visible_text = ds4_kvstore_render_tokens_text(engine, portable_live,
                                                        &visible_len);
    CHECK(visible_text && visible_len > 0,
          "Qwen visible session key render failed");
    ds4_session_free(isolated);
    isolated = NULL;
    CHECK(ds4_session_create(&isolated, engine, main_ctx) == 0,
          "reload destination create failed");
    ds4_tokens reloaded_prompt = {0};
    ds4_kvstore_load_result reload_result = {0};
    CHECK(ds4_kvstore_try_load_text(&store, engine, isolated, visible_text,
                                    &reloaded_prompt, &reload_result,
                                    NULL, true,
                                    DS4_KVSTORE_REASON_ALL) == portable_live->len,
          "Qwen visible session cache reload failed");
    CHECK(!reload_result.consumed,
          "persistent Qwen session cache was consumed on reload");
    CHECK(tokens_equal(ds4_session_tokens(isolated), portable_live),
          "reloaded Qwen visible session tokens differ");
    CHECK(assert_unchanged(isolated, portable_live, frontier_logits, vocab,
                           "visible session disk reload") == 0,
          "reloaded Qwen visible session logits differ");
    if (reload_result.path) unlink(reload_result.path);
    ds4_kvstore_load_result_free(&reload_result);
    ds4_tokens_free(&reloaded_prompt);
    free(visible_text);
    ds4_kvstore_close(&store);
    ds4_tokens_free(&portable_expected);
    ds4_session_free(isolated);
    isolated = NULL;

    /* Near-boundary restore: child fills the context, result still resumes at
     * the parent frontier and lands exactly on the context limit. */
    ds4_session_free(live);
    live = NULL;
    CHECK(ds4_session_create(&live, engine, 512) == 0,
          "boundary session create failed");
    ds4_tokens boundary_frontier = {0}, boundary_child = {0};
    append_tokens(&boundary_frontier, &pattern, 480);
    CHECK(sync_session(live, &boundary_frontier, "boundary frontier") == 0,
          "boundary frontier sync failed");
    CHECK(save_checkpoint(live, checkpoint, NULL) == 0,
          "boundary checkpoint failed");
    ds4_tokens_copy(&boundary_child, &boundary_frontier);
    append_tokens(&boundary_child, &pattern, 31);
    CHECK(sync_session(live, &boundary_child, "boundary child") == 0,
          "boundary child sync failed");
    CHECK(load_checkpoint(live, checkpoint, &boundary_frontier, NULL) == 0,
          "boundary restore failed");
    ds4_tokens_free(&boundary_child);
    ds4_tokens_copy(&boundary_child, &boundary_frontier);
    append_tokens(&boundary_child, &result_pattern, 31);
    CHECK(sync_session(live, &boundary_child, "boundary result") == 0,
          "boundary result sync failed");
    CHECK(ds4_session_pos(live) == 511,
          "boundary did not preserve one generation slot");
    CHECK(compare_with_canonical(engine, live, &boundary_frontier,
                                 &boundary_child, 512,
                                 "context boundary canonical") == 0,
          "boundary canonical comparison failed");
    ds4_tokens_free(&boundary_frontier);
    ds4_tokens_free(&boundary_child);

    fprintf(stdout,
            "AGENTIC_CHECKPOINT_REPORT {\"model\":\"%s\",\"mtp\":%s,\"fast\":%s,"
            "\"parent_tokens\":%d,\"instruction_prefill_tokens\":%d,"
            "\"result_prefill_tokens\":%d,\"discarded_child_tokens\":%d,"
            "\"checkpoint_bytes\":%llu,\"checkpoint_stage_ms\":%.3f,"
            "\"checkpoint_write_ms\":%.3f,\"checkpoint_read_ms\":%.3f,"
            "\"checkpoint_restore_ms\":%.3f,\"parent_prefill_ms\":%.3f,"
            "\"instruction_prefill_ms\":%.3f,\"result_prefill_ms\":%.3f,"
            "\"canonical_logits_bit_exact\":true,"
            "\"nested_three_levels_bit_exact\":true,"
            "\"session_isolation_rejected\":true,"
            "\"cancel_save_rejected\":true,"
            "\"cancel_restore_rejected\":true,"
            "\"portable_session_bytes\":%llu,"
            "\"portable_session_continuation_bit_exact\":true,"
            "\"context_boundary_bit_exact\":true}\n",
            model, mtp_enabled ? "true" : "false", fast ? "true" : "false",
            parent_count, instruction_count, result_count, discarded_child,
            (unsigned long long)save_metrics.checkpoint_bytes,
            save_metrics.stage_ms, save_metrics.write_ms,
            load_metrics.read_ms, load_metrics.restore_ms,
            parent_ms, instructions_ms, result_ms,
            (unsigned long long)session_payload_bytes);
    rc = 0;

fail:
    free(frontier_logits);
    free(child_logits);
    ds4_tokens_free(&pattern);
    ds4_tokens_free(&call);
    ds4_tokens_free(&result_pattern);
    ds4_tokens_free(&frontier);
    ds4_tokens_free(&child);
    ds4_tokens_free(&returned);
    ds4_session_free(live);
    ds4_session_free(isolated);
    ds4_engine_close(engine);
    unlink(checkpoint);
    unlink(corrupt_path);
    unlink(truncated_path);
    unlink(session_path);
    rmdir(dir);
    return rc;
}
