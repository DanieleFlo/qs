#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct qwen_trace_config {
    std::string directory;
    int position = -1;
    int layer = -1;
    std::string stage;
    bool active = false;
    bool selected_is_last = false;
};

static bool qwen_trace_stage(const std::string &name, std::string *stage, int *layer) {
    static const char *const names[] = {
        "model.input_embed", "attn_norm", "linear_attn_qkv_mixed", "z",
        "alpha", "beta", "conv_output_silu", "final_output", "new_state",
        "linear_attn_out", "attn_output", "attn_residual", "attn_post_norm",
        "ffn_out", "post_ffn", "l_out", "result_norm", "result_output",
    };
    for (const char *candidate : names) {
        const std::string prefix = std::string(candidate) + "-";
        if (name == candidate) {
            *stage = candidate;
            *layer = name.rfind("result_", 0) == 0 ? 63 : 0;
            return true;
        }
        if (name.rfind(prefix, 0) == 0) {
            char *end = nullptr;
            long value = std::strtol(name.c_str() + prefix.size(), &end, 10);
            if (end && *end == '\0' && value >= 0 && value <= INT_MAX) {
                *stage = candidate;
                *layer = (int)value;
                return true;
            }
        }
    }
    return false;
}

static int64_t qwen_trace_width(const std::string &stage) {
    if (stage == "linear_attn_qkv_mixed" || stage == "conv_output_silu") return 10240;
    if (stage == "z" || stage == "final_output") return 6144;
    if (stage == "alpha" || stage == "beta") return 48;
    if (stage == "new_state") return 48ll * 128ll * 128ll;
    if (stage == "result_output") return 248320;
    return 5120;
}

static bool qwen_trace_callback(struct ggml_tensor *tensor, bool ask, void *user_data) {
    auto *config = static_cast<qwen_trace_config *>(user_data);
    std::string stage;
    int layer = -1;
    if (!config || !config->active || config->directory.empty() || config->position < 0 ||
        !qwen_trace_stage(tensor->name, &stage, &layer) ||
        (config->layer >= 0 && layer != config->layer) ||
        (!config->stage.empty() && stage != config->stage)) {
        return false;
    }
    if (ask) return true;
    if (tensor->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "score_llama: trace tensor %s is not float32\n", tensor->name);
        return true;
    }

    const int64_t width = qwen_trace_width(stage);
    size_t offset = 0;
    size_t bytes = (size_t)width * sizeof(float);
    if (stage == "new_state") {
        if (ggml_nelements(tensor) < width) return true;
    } else {
        const int token_axis = stage == "beta" ? 2 : 1;
        int row = config->position;
        if (tensor->ne[token_axis] == 1 && config->selected_is_last) row = 0;
        if (tensor->ne[token_axis] <= row) return true;
        offset = (size_t)row * tensor->nb[token_axis];
        if (offset > ggml_nbytes(tensor) || bytes > ggml_nbytes(tensor) - offset) return true;
    }

    std::vector<float> values((size_t)width);
    ggml_backend_tensor_get(tensor, values.data(), offset, bytes);
    std::string path = config->directory + "/llama-pos" +
        std::to_string(config->position) + "-layer" + std::to_string(layer) +
        "-" + stage + ".f32";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "score_llama: cannot write trace %s\n", path.c_str());
        return true;
    }
    out.write(reinterpret_cast<const char *>(values.data()), (std::streamsize)bytes);
    return true;
}

static void die(const char *msg) {
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

static std::string read_file(const char *path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "open %s: %s\n", path, std::strerror(errno));
        std::exit(1);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void strip_newline(std::string &s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
}

static std::vector<std::string> split_tab(const std::string &line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            out.push_back(line.substr(start));
            return out;
        }
        out.push_back(line.substr(start, tab - start));
        start = tab + 1;
    }
}

static std::vector<llama_token> load_token_array(const char *path) {
    const std::string json = read_file(path);
    const char *p = json.c_str();
    while (*p && std::isspace((unsigned char)*p)) p++;
    if (*p++ != '[') die("token file must contain a JSON array");
    std::vector<llama_token> out;
    for (;;) {
        while (*p && std::isspace((unsigned char)*p)) p++;
        if (*p == ']') return out;
        char *end = nullptr;
        errno = 0;
        long value = std::strtol(p, &end, 10);
        if (errno || end == p || value < 0 || value > std::numeric_limits<llama_token>::max()) {
            die("invalid token ID in JSON array");
        }
        out.push_back((llama_token)value);
        p = end;
        while (*p && std::isspace((unsigned char)*p)) p++;
        if (*p == ',') p++;
        else if (*p != ']') die("invalid token JSON array separator");
    }
}

static void write_token_array(std::ostream &out, const std::vector<llama_token> &tokens) {
    out << '[';
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i) out << ',';
        out << tokens[i];
    }
    out << ']';
}

static std::string hex_bytes(const std::string &bytes) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (unsigned char byte : bytes) {
        out.push_back(hex[byte >> 4]);
        out.push_back(hex[byte & 15]);
    }
    return out;
}

static std::string detokenize(
        const llama_vocab *vocab,
        const std::vector<llama_token> &tokens) {
    std::vector<char> bytes(tokens.size() * 16 + 16);
    int n = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(),
                             bytes.data(), (int32_t)bytes.size(), false, true);
    if (n < 0) {
        bytes.resize((size_t)-n);
        n = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(),
                             bytes.data(), (int32_t)bytes.size(), false, true);
    }
    if (n < 0) die("llama_detokenize failed");
    return std::string(bytes.data(), (size_t)n);
}

static std::vector<llama_token> tokenize(
        const llama_vocab *vocab,
        const std::string &text,
        bool add_special,
        bool parse_special) {
    int n = llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                           nullptr, 0, add_special, parse_special);
    if (n < 0) n = -n;
    if (n == 0) return {};

    std::vector<llama_token> tokens((size_t)n);
    int got = llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                             tokens.data(), n, add_special, parse_special);
    if (got < 0) die("llama_tokenize failed");
    tokens.resize((size_t)got);
    return tokens;
}

static std::string render_glm_ds4_prompt(const std::string &prompt) {
    return std::string("[gMASK]<sop><|user|>") + prompt +
           "<|assistant|><think></think>";
}

static std::string render_template_prompt(
        const char *tmpl,
        const std::string &prompt,
        bool *ok) {
    llama_chat_message msg = {"user", prompt.c_str()};
    int n = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
    if (n < 0) {
        *ok = false;
        return {};
    }
    std::vector<char> buf((size_t)n + 1);
    int got = llama_chat_apply_template(tmpl, &msg, 1, true,
                                        buf.data(), (int32_t)buf.size());
    if (got < 0) {
        *ok = false;
        return {};
    }
    *ok = true;
    return std::string(buf.data(), (size_t)got);
}

static bool decode_chunk(
        llama_context *ctx,
        llama_batch &batch,
        const llama_token *tokens,
        int n_tokens,
        int pos,
        bool logits_last) {
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (logits_last && i == n_tokens - 1) ? 1 : 0;
    }
    return llama_decode(ctx, batch) == 0;
}

static bool decode_tokens(
        llama_context *ctx,
        llama_batch &batch,
        const std::vector<llama_token> &tokens,
        int start_pos,
        int n_batch,
        bool logits_last) {
    int off = 0;
    while (off < (int)tokens.size()) {
        int n = std::min(n_batch, (int)tokens.size() - off);
        bool want_logits = logits_last && off + n == (int)tokens.size();
        if (!decode_chunk(ctx, batch, tokens.data() + off, n,
                          start_pos + off, want_logits)) {
            return false;
        }
        off += n;
    }
    return true;
}

static double token_logprob(
        const float *logits,
        int n_vocab,
        llama_token token,
        llama_token *greedy_out) {
    float max_logit = -std::numeric_limits<float>::infinity();
    llama_token greedy = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            greedy = (llama_token)i;
        }
    }

    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        sum += std::exp((double)logits[i] - (double)max_logit);
    }
    *greedy_out = greedy;
    return (double)logits[token] - ((double)max_logit + std::log(sum));
}

static int score_token_manifest(
        llama_context *ctx,
        const llama_vocab *vocab,
        int n_vocab,
        int n_batch,
        const char *manifest_path,
        int ctx_size,
        qwen_trace_config *trace) {
    std::ifstream manifest(manifest_path, std::ios::binary);
    if (!manifest) die("failed to open token manifest");
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::string line;
    int cases = 0;
    while (std::getline(manifest, line)) {
        strip_newline(line);
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> cols = split_tab(line);
        if (cols.size() != 7) die("bad token manifest row");
        const std::string &id = cols[0];
        const std::string rendered = read_file(cols[1].c_str());
        const std::vector<llama_token> prompt = load_token_array(cols[2].c_str());
        const std::vector<llama_token> target = load_token_array(cols[3].c_str());
        const std::vector<llama_token> native_prompt = tokenize(vocab, rendered, false, true);
        if (prompt.empty() || target.empty() || (int)(prompt.size() + target.size() + 1) >= ctx_size) {
            die("empty token sequence or context exceeded");
        }
        std::ofstream greedy_out(cols[4], std::ios::binary);
        std::ofstream teacher_out(cols[5], std::ios::binary);
        if (!greedy_out || !teacher_out) die("failed to open logits output");

        llama_memory_clear(llama_get_memory(ctx), true);
        if (trace) {
            trace->selected_is_last = trace->position == (int)prompt.size() - 1;
            trace->active = trace->position < (int)prompt.size();
        }
        if (!decode_tokens(ctx, batch, prompt, 0, n_batch, true)) die("prompt decode failed");
        if (trace) trace->active = false;
        std::vector<llama_token> greedy;
        for (size_t i = 0; i < target.size(); i++) {
            const float *logits = llama_get_logits_ith(ctx, -1);
            if (!logits) die("greedy logits unavailable");
            greedy_out.write(reinterpret_cast<const char *>(logits), (std::streamsize)n_vocab * sizeof(float));
            llama_token token = 0;
            (void)token_logprob(logits, n_vocab, target[i], &token);
            greedy.push_back(token);
            if (!decode_chunk(ctx, batch, &token, 1, (int)prompt.size() + (int)i, true)) {
                die("greedy token decode failed");
            }
        }
        greedy_out.close();

        llama_memory_clear(llama_get_memory(ctx), true);
        if (!decode_tokens(ctx, batch, prompt, 0, n_batch, true)) die("teacher prompt decode failed");
        std::vector<double> teacher_logprobs;
        for (size_t i = 0; i < target.size(); i++) {
            const float *logits = llama_get_logits_ith(ctx, -1);
            if (!logits) die("teacher logits unavailable");
            teacher_out.write(reinterpret_cast<const char *>(logits), (std::streamsize)n_vocab * sizeof(float));
            llama_token ignored = 0;
            teacher_logprobs.push_back(token_logprob(logits, n_vocab, target[i], &ignored));
            if (!decode_chunk(ctx, batch, &target[i], 1, (int)prompt.size() + (int)i, true)) {
                die("teacher token decode failed");
            }
        }
        teacher_out.close();

        std::ofstream response(cols[6], std::ios::binary);
        if (!response) die("failed to open response output");
        response << "{\n  \"engine\": \"llama.cpp\",\n  \"canonical_prompt_token_ids\": ";
        write_token_array(response, prompt);
        response << ",\n  \"native_prompt_token_ids\": ";
        write_token_array(response, native_prompt);
        response << ",\n  \"native_rendered_bytes_hex\": \"" << hex_bytes(rendered) << '"';
        response << ",\n  \"native_rendering_status\": \"tokenizer_only\",\n  \"prompt_token_ids\": ";
        write_token_array(response, prompt);
        response << ",\n  \"greedy_token_ids\": ";
        write_token_array(response, greedy);
        response << ",\n  \"greedy_bytes_hex\": \"" << hex_bytes(detokenize(vocab, greedy)) << '"';
        response << ",\n  \"teacher_forced_source\": \"canonical token manifest\",\n  \"teacher_forced\": [";
        for (size_t i = 0; i < target.size(); i++) {
            if (i) response << ',';
            response << "{\"token_id\":" << target[i] << ",\"logprob\":" << teacher_logprobs[i] << '}';
        }
        response << "]\n}\n";
        std::fprintf(stderr, "%s cases=%d prompt=%zu target=%zu vocab=%d token_manifest=1\n",
                     id.c_str(), ++cases, prompt.size(), target.size(), n_vocab);
    }
    llama_batch_free(batch);
    return 0;
}

int main(int argc, char **argv) {
    const bool token_manifest_mode = argc > 1 && std::strcmp(argv[1], "--token-manifest") == 0;
    const int base = token_manifest_mode ? 2 : 1;
    if ((!token_manifest_mode && argc != 4 && argc != 5 && argc != 6) ||
        (token_manifest_mode && argc != 4 && argc != 5)) {
        std::fprintf(stderr,
                     "usage: %s MODEL manifest.tsv OUT.tsv [ctx] [auto|glm-ds4]\n"
                     "       %s --token-manifest MODEL cases.tsv [ctx]\n",
                     argv[0],
                     argv[0]);
        return 2;
    }

    const char *model_path = argv[base];
    const char *manifest_path = argv[base + 1];
    const char *out_path = token_manifest_mode ? nullptr : argv[base + 2];
    int ctx_size = argc > base + (token_manifest_mode ? 2 : 3) ? std::atoi(argv[base + (token_manifest_mode ? 2 : 3)]) : 4096;
    if (ctx_size < 1024) ctx_size = 1024;
    const std::string template_mode = !token_manifest_mode && argc == 6 ? argv[5] : "auto";
    if (template_mode != "auto" && template_mode != "glm-ds4") {
        die("template mode must be auto or glm-ds4");
    }

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = std::getenv("LLAMA_QWEN_CPU") ? 0 : -1;
    model_params.use_mmap = true;

    llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (!model) die("failed to open model");
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const char *tmpl = llama_model_chat_template(model, nullptr);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t)ctx_size;
    ctx_params.n_batch = 2048;
    ctx_params.n_ubatch = 512;
    ctx_params.n_seq_max = 1;
    ctx_params.no_perf = true;
    if (std::getenv("LLAMA_QWEN_CPU")) {
        ctx_params.offload_kqv = false;
        ctx_params.op_offload = false;
    }

    qwen_trace_config trace;
    const char *trace_dir = std::getenv("LLAMA_QWEN_TRACE_DIR");
    const char *trace_position = std::getenv("LLAMA_QWEN_TRACE_POSITION");
    const char *trace_layer = std::getenv("LLAMA_QWEN_TRACE_LAYER");
    const char *trace_stage = std::getenv("LLAMA_QWEN_TRACE_STAGE");
    if (trace_dir && trace_dir[0]) {
        if (!trace_position || !trace_layer) {
            die("LLAMA_QWEN_TRACE_DIR requires LLAMA_QWEN_TRACE_POSITION and LLAMA_QWEN_TRACE_LAYER");
        }
        trace.directory = trace_dir;
        trace.position = std::atoi(trace_position);
        trace.layer = std::strcmp(trace_layer, "all") == 0 ? -1 : std::atoi(trace_layer);
        trace.stage = trace_stage ? trace_stage : "";
        if (trace.position < 0 || trace.layer < -1) die("Qwen trace position/layer is invalid");
        ctx_params.cb_eval = qwen_trace_callback;
        ctx_params.cb_eval_user_data = &trace;
    }

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) die("failed to create context");
    const int n_batch = std::min<int>((int)llama_n_batch(ctx), 2048);
    if (token_manifest_mode) {
        const int rc = score_token_manifest(ctx, vocab, n_vocab, n_batch, manifest_path,
                                            ctx_size, trace_dir && trace_dir[0] ? &trace : nullptr);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return rc;
    }
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    std::ifstream mf(manifest_path, std::ios::binary);
    if (!mf) {
        std::fprintf(stderr, "open %s: %s\n", manifest_path, std::strerror(errno));
        return 1;
    }
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "open %s: %s\n", out_path, std::strerror(errno));
        return 1;
    }
    out << "id\tprompt_tokens\ttarget_tokens\tnll\tavg_nll\tfirst_match\tgreedy_lcp\n";

    std::string line;
    int case_n = 0;
    double total_nll = 0.0;
    long total_tokens = 0;
    long total_lcp = 0;
    long first_matches = 0;
    bool warned_template_fallback = false;

    while (std::getline(mf, line)) {
        strip_newline(line);
        if (line.empty() || line[0] == '#') continue;

        std::vector<std::string> cols = split_tab(line);
        if (cols.size() < 3) die("bad manifest row");
        const std::string &id = cols[0];
        const std::string &prompt_path = cols[1];
        const std::string &cont_path = cols[2];

        std::string prompt_text = read_file(prompt_path.c_str());
        std::string cont_text = read_file(cont_path.c_str());

        std::string rendered;
        bool used_template = false;
        if (template_mode == "auto" && tmpl) {
            rendered = render_template_prompt(tmpl, prompt_text, &used_template);
        }
        if (!used_template) {
            if (template_mode == "auto" && !warned_template_fallback) {
                std::fprintf(stderr,
                             "score_llama: llama.cpp chat template unavailable; "
                             "using DS4 GLM prompt fallback\n");
                warned_template_fallback = true;
            }
            rendered = render_glm_ds4_prompt(prompt_text);
        }

        std::vector<llama_token> prompt =
            tokenize(vocab, rendered, false, true);
        std::vector<llama_token> target =
            tokenize(vocab, cont_text, false, false);

        if (prompt.empty()) die("empty prompt tokenization");
        if (target.empty()) die("empty continuation tokenization");
        if ((int)prompt.size() + (int)target.size() + 1 >= ctx_size) {
            std::fprintf(stderr, "%s exceeds ctx=%d\n", id.c_str(), ctx_size);
            return 1;
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        if (!decode_tokens(ctx, batch, prompt, 0, n_batch, true)) {
            std::fprintf(stderr, "%s prompt decode failed\n", id.c_str());
            return 1;
        }

        double nll = 0.0;
        int lcp = 0;
        bool still_matching = true;
        bool first_match = false;

        for (int i = 0; i < (int)target.size(); i++) {
            const float *logits = llama_get_logits_ith(ctx, -1);
            if (!logits) {
                std::fprintf(stderr, "%s logits unavailable at target token %d\n",
                             id.c_str(), i);
                return 1;
            }

            llama_token greedy = 0;
            double lp = token_logprob(logits, n_vocab, target[(size_t)i], &greedy);
            if (i == 0) first_match = (greedy == target[(size_t)i]);
            if (still_matching && greedy == target[(size_t)i]) lcp++;
            else still_matching = false;
            nll += -lp;

            if (!decode_chunk(ctx, batch, &target[(size_t)i], 1,
                              (int)prompt.size() + i, true)) {
                std::fprintf(stderr, "%s target decode failed at token %d\n",
                             id.c_str(), i);
                return 1;
            }
        }

        const double avg = nll / (double)target.size();
        out << id << '\t'
            << prompt.size() << '\t'
            << target.size() << '\t'
            << nll << '\t'
            << avg << '\t'
            << (first_match ? 1 : 0) << '\t'
            << lcp << '\n';
        out.flush();

        case_n++;
        total_nll += nll;
        total_tokens += (long)target.size();
        total_lcp += lcp;
        first_matches += first_match ? 1 : 0;
        std::fprintf(stderr,
                     "%s cases=%d prompt=%zu target=%zu avg_nll=%.6f lcp=%d\n",
                     id.c_str(), case_n, prompt.size(), target.size(), avg, lcp);
    }

    std::fprintf(stderr,
                 "summary cases=%d tokens=%ld avg_nll=%.9f first_match=%ld avg_lcp=%.3f\n",
                 case_n,
                 total_tokens,
                 total_tokens ? total_nll / (double)total_tokens : 0.0,
                 first_matches,
                 case_n ? (double)total_lcp / (double)case_n : 0.0);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
