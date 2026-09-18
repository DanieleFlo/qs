#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

/* A new client can reuse a system frontier after a failed request reset KV. */
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_engine_options options = {
        .model_path = argv[1],
        .mtp_path = getenv("DS4_TEST_MTP"),
        .mtp_draft_tokens = getenv("DS4_TEST_MTP") ? 4 : 0,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .prefill_chunk = 2048,
    };
    resident_kv_frontier frontier = {0}, local = {0};
    char skill_dir[] = "/tmp/qs-skill-regression.XXXXXX";
    server srv = {0};
    server_slot slot = {0};
    pthread_mutex_init(&srv.tool_mu, NULL);
    bool skill_dir_created = false;
    ds4_tokens prefix = {0}, tail = {0}, prompt = {0}, unrelated = {0};
    float *expected = NULL, *actual = NULL;
    char err[256] = {0};
    int rc = 1;
#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL line %d: %s (%s)\n", __LINE__, #c, err); \
    goto done; } } while (0)
    REQUIRE(ds4_engine_open(&engine, &options) == 0);
    REQUIRE(ds4_session_create(&session, engine, 1024) == 0);
    buf system = {0};
    buf_puts(&system, "<|im_start|>system\n");
    for (int i = 0; i < 12; i++)
        buf_puts(&system, "The registered tool is mock-tool. Invoke it when asked. ");
    buf_puts(&system, "<|im_end|>\n");
    ds4_tokenize_rendered_chat(engine, system.ptr, &prefix);
    buf_free(&system);
    ds4_tokenize_text(engine,
        "<|im_start|>user\nTest mock-tool.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n", &tail);
    ds4_tokenize_rendered_chat(engine,
        "<|im_start|>system\nYou translate poems."
        "<|im_end|>\n<|im_start|>user\nTranslate hello."
        "<|im_end|>\n<|im_start|>assistant\n", &unrelated);
    /* Exercise the production minimum system boundary, not just a tiny prompt. */
    REQUIRE(prefix.len >= RESIDENT_SYSTEM_MIN_TOKENS);
    ds4_tokens_copy(&prompt, &prefix);
    for (int i = 0; i < tail.len; i++) ds4_tokens_push(&prompt, tail.v[i]);
    REQUIRE(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    REQUIRE(resident_kv_frontier_capture(session, &frontier, false,
                                         err, sizeof(err)));
    REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    int vocab = ds4_engine_vocab_size(engine);
    expected = xmalloc((size_t)vocab * sizeof(float));
    actual = xmalloc((size_t)vocab * sizeof(float));
    REQUIRE(ds4_session_copy_logits(session, expected, vocab) == vocab);
    for (int attempt = 0; attempt < 3; attempt++) {
        ds4_session_invalidate(session);
        if (attempt == 1)
            REQUIRE(ds4_session_sync(session, &unrelated, err, sizeof(err)) == 0);
        REQUIRE(resident_kv_frontier_restore(session, &frontier, err, sizeof(err)));
        REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab);
        REQUIRE(memcmp(expected, actual, (size_t)vocab * sizeof(float)) == 0);
    }
    /* Request-local response cleanup still uses a compact accelerator frontier. */
    REQUIRE(resident_kv_frontier_restore(session, &frontier, err, sizeof(err)));
    REQUIRE(resident_kv_frontier_capture(session, &local, true, err, sizeof(err)));
    REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    REQUIRE(resident_kv_frontier_restore(session, &local, err, sizeof(err)));
    REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab);
    REQUIRE(memcmp(expected, actual, (size_t)vocab * sizeof(float)) == 0);
    /* A skill is saved before post-response canonicalization. Rewriting the
     * sampled response destroys attention rows that a compact frame omits. */
    ds4_session_invalidate(session);
    REQUIRE(!resident_kv_frontier_restore(session, &local, err, sizeof(err)));
    REQUIRE(ds4_session_pos(session) == 0);
    err[0] = '\0';
    REQUIRE(mkdtemp(skill_dir) != NULL);
    skill_dir_created = true;
    srv.skill_dir = skill_dir;
    srv.slot_count = 1;
    srv.slots = &slot;
    slot.session = session;
    REQUIRE(resident_kv_frontier_restore(session, &frontier, err, sizeof(err)));
    tool_call call = {.id = "call_regression", .name = "mock-skill"};
    REQUIRE(skill_frame_save(&srv, &slot, &call, NULL, err, sizeof(err)));
    /* Corrupt bytes must be rejected before any mutation; repairing the file
     * must then allow the same frame to be restored. */
    FILE *checkpoint = fopen(slot.skills->checkpoint_path, "r+b");
    REQUIRE(checkpoint != NULL);
    REQUIRE(fseek(checkpoint, -1, SEEK_END) == 0);
    int last_byte = fgetc(checkpoint);
    REQUIRE(last_byte != EOF);
    REQUIRE(fseek(checkpoint, -1, SEEK_END) == 0);
    REQUIRE(fputc(last_byte ^ 1, checkpoint) != EOF);
    REQUIRE(fclose(checkpoint) == 0);
    REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    REQUIRE(!skill_frame_restore(&slot, slot.skills, NULL, err, sizeof(err)));
    REQUIRE(ds4_session_pos(session) == prompt.len);
    REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab);
    REQUIRE(memcmp(expected, actual, (size_t)vocab * sizeof(float)) == 0);
    checkpoint = fopen(slot.skills->checkpoint_path, "r+b");
    REQUIRE(checkpoint != NULL);
    REQUIRE(fseek(checkpoint, -1, SEEK_END) == 0);
    REQUIRE(fputc(last_byte, checkpoint) != EOF);
    REQUIRE(fclose(checkpoint) == 0);
    err[0] = '\0';
    REQUIRE(ds4_session_sync(session, &unrelated, err, sizeof(err)) == 0);
    REQUIRE(skill_frame_restore(&slot, slot.skills, NULL, err, sizeof(err)));
    REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_copy_logits(session, actual, vocab) == vocab);
    REQUIRE(memcmp(expected, actual, (size_t)vocab * sizeof(float)) == 0);
    fprintf(stderr, "PASS: system/skill restore, corruption rejection, and compact rollback are bit-exact\n");
    rc = 0;
done:
    skill_frames_clear_locked(&srv, &slot);
    if (skill_dir_created) rmdir(skill_dir);
    pthread_mutex_destroy(&srv.tool_mu);
    free(expected);
    free(actual);
    resident_kv_frontier_clear(&local);
    resident_kv_frontier_clear(&frontier);
    ds4_tokens_free(&prefix);
    ds4_tokens_free(&tail);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&unrelated);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return rc;
}
