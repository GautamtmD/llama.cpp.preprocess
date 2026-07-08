// multimodal-server — slice 2: session-oriented HTTP LLM API with streaming.
//
// Endpoints (see docs/ipc-protocol.md):
//   POST   /sessions              -> create a session, returns {session_id}
//   POST   /sessions/{id}/inject  -> body {text}; tokenize + decode into KV cache (no gen)
//   POST   /sessions/{id}/generate-> body {max_tokens, stream, ...}
//       stream=false (default): returns slice-1 JSON {text, tokens, ...}
//       stream=true : text/event-stream; one event per token + a final usage event
//   DELETE /sessions/{id}         -> free the session
//   GET    /health                -> liveness
//
// All MultiModalAgent engine code lives under engine/multimodal/ — we do NOT
// modify the upstream llama.cpp fork. One model is loaded and shared; each
// session is its own llama_context (owning its KV cache).

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>

#include "httplib.h"
#include "nlohmann/json.hpp"

#include "util.h"

#include "common.h"
#include "chat.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

// 'json' (nlohmann::ordered_json) is provided by common/chat.h.

// Read an entire file into a string (for --chat-template-file). Throws on error.
static std::string read_file_contents(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

namespace {

struct ServerConfig {
    std::string model_path;
    std::string mmproj_path;   // multimodal projector gguf (empty = text-only)
    int  port          = 8080;
    int  n_gpu_layers  = 99;
    int  ctx_size      = 4096;
    int  n_batch       = 2048;

    // Chat-template handling (mirrors llama-server / common/arg.cpp). We reuse
    // common/'s templating; these just feed it the same inputs llama-server does.
    std::string chat_template;                                // --chat-template / --chat-template-file (override; empty = model's default)
    bool        use_jinja            = true;                  // --jinja / --no-jinja (default true, like llama-server)
    bool        enable_chat_template = true;                  // --no-chat-template disables the 'messages' inject path
    std::map<std::string, std::string> chat_template_kwargs;  // --chat-template-kwargs (key -> JSON value serialized as a string)
    std::string system_prompt;                                // --system-prompt (prepended as a system message to every conversation)
};

// One model loaded once; many sessions (each its own context / KV cache).
struct AppState {
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr chat_templates;  // built from the model; applies its chat template
    mtmd_context * mtmd_ctx = nullptr;         // multimodal projector (may be null)
    int n_ctx_per_session = 4096;
    int n_batch = 2048;
    // Chat-template options (copied from ServerConfig at startup) — passed to
    // common_chat_templates_apply so our rendering matches llama-server.
    bool                                 use_jinja            = true;
    bool                                 enable_chat_template = true;
    std::map<std::string, std::string>   chat_template_kwargs;
    std::string                          system_prompt;
    std::mutex mu;
    std::map<int64_t, llama_context *> sessions;  // owns the contexts
    // Last token in each session's cache (seq 0). Used to refresh logits after a
    // fork restores KV via llama_state_seq_set_data (the restore carries KV but
    // NOT the output/logits buffer, so the forked session must re-decode its last
    // token before it can generate). LLAMA_TOKEN_NULL when unknown (e.g. a session
    // whose last inject ended in a media/audio embedding chunk — see fork handler).
    std::map<int64_t, llama_token> last_tokens;
    std::atomic<int64_t> next_id{1};
};

struct GenParams {
    int   max_tokens = 256;
    float temp       = 0.8f;
    float top_p      = 1.0f;
    int   seed       = -1;
};

struct GenResult {
    std::string text;
    std::vector<int64_t> ids;
    double gen_s = 0.0;
};

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

llama_sampler * make_sampler(float temp, float top_p, int seed) {
    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (top_p < 1.0f) llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    if (temp > 0)     llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
    const uint32_t s = (seed < 0) ? LLAMA_DEFAULT_SEED : (uint32_t)seed;
    llama_sampler_chain_add(chain, llama_sampler_init_dist(s));
    return chain;
}

// Find a session by id under the lock. Returns nullptr if absent. Does NOT hold
// the lock on return (each session's context is single-threaded; the caller
// drives generation without the global lock).
llama_context * lookup_session(AppState & app, int64_t n) {
    std::lock_guard<std::mutex> lk(app.mu);
    auto it = (n > 0) ? app.sessions.find(n) : app.sessions.end();
    return (it == app.sessions.end()) ? nullptr : it->second;
}

// Create a new session context with the server's standard params. Returns
// nullptr on failure. (Shared by POST /sessions and POST /sessions/{id}/fork.)
llama_context * create_session_ctx(const AppState & app) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = app.n_ctx_per_session;
    cp.n_batch = std::min<int>(app.n_batch, app.n_ctx_per_session);
    cp.no_perf = true;
    return llama_init_from_model(app.model, cp);
}

// Register a pre-built context as a new session under the lock; returns its id.
int64_t register_session(AppState & app, llama_context * ctx) {
    const int64_t n = app.next_id.fetch_add(1);
    std::lock_guard<std::mutex> lk(app.mu);
    app.sessions[n] = ctx;
    app.last_tokens[n] = LLAMA_TOKEN_NULL;
    return n;
}

// Remember the last token decoded into a session (for fork logits refresh).
void set_last_token(AppState & app, int64_t n, llama_token t) {
    std::lock_guard<std::mutex> lk(app.mu);
    app.last_tokens[n] = t;
}

// The last token decoded into a session, or LLAMA_TOKEN_NULL if unknown.
llama_token get_last_token(AppState & app, int64_t n) {
    std::lock_guard<std::mutex> lk(app.mu);
    auto it = app.last_tokens.find(n);
    return (it == app.last_tokens.end()) ? LLAMA_TOKEN_NULL : it->second;
}

// Seed an empty session with BOS so there are logits to sample from.
bool seed_if_empty(const AppState & app, llama_context * ctx) {
    if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) >= 0) return true;
    llama_token bos = llama_vocab_bos(app.vocab);
    llama_batch b = llama_batch_get_one(&bos, 1);
    return llama_decode(ctx, b) == 0;
}

// Shared generation loop. Calls `on_token(piece, id)` per produced token (after
// the token's text is known and it has been fed back into the cache). The
// callback may return false to stop generation early (e.g. client disconnect).
// Errors are surfaced by returning a non-empty `error` string.
//
// Both the streaming and non-streaming handlers use this so generation logic
// lives in exactly one place.
GenResult run_generation(
    const AppState & app, llama_context * ctx, const GenParams & p,
    std::function<bool(const std::string & piece, int64_t id)> on_token
) {
    GenResult r;
    if (!seed_if_empty(app, ctx)) return r;

    llama_sampler * smpl = make_sampler(p.temp, p.top_p, p.seed);
    const int n_ctx = llama_n_ctx(ctx);
    const double t0 = now_s();
    bool keep_going = true;
    for (int step = 0; step < p.max_tokens && keep_going; ++step) {
        const int used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        if (used + 1 > n_ctx) break;
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(app.vocab, id)) break;
        char buf[64];
        const int n = llama_token_to_piece(app.vocab, id, buf, sizeof(buf), 0, true);
        std::string piece = (n > 0) ? std::string(buf, n) : std::string{};
        r.ids.push_back((int64_t)id);
        r.text += piece;
        if (on_token) keep_going = on_token(piece, (int64_t)id);
        llama_batch batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, batch) != 0) break;
    }
    r.gen_s = now_s() - t0;
    llama_sampler_free(smpl);
    return r;
}

} // namespace

// ---- multimodal helpers (outside anon namespace so they can be forward-declared) ----

// Decode image bytes (PNG/JPEG/BMP/...) into an mtmd_bitmap via the mtmd
// helper. We use the public helper (not stb directly) because the stb_image
// implementation is compiled statically into the mtmd library and its symbols
// are not exported from mtmd.dll. The helper also auto-detects audio files,
// which sets up slice 3b (audio inject) for free.
// Returns nullptr on failure. Caller owns the result (free with mtmd_bitmap_free).
// Decode image OR audio bytes into an mtmd_bitmap via the mtmd helper. The
// helper auto-detects the media type by magic bytes: images via stb_image
// (jpg/png/bmp/...), audio via miniaudio (wav/mp3/flac) resampled to the
// projector's sample rate (mtmd_get_audio_sample_rate). Returns nullptr on
// failure. Caller owns the result (free with mtmd_bitmap_free).
static mtmd_bitmap * bitmap_from_media_bytes(mtmd_context * mtmd_ctx,
                                             const std::string & bytes) {
    mtmd_helper_bitmap_wrapper wrap = mtmd_helper_bitmap_init_from_buf(
        mtmd_ctx,
        reinterpret_cast<const unsigned char *>(bytes.data()),
        bytes.size(),
        /*placeholder*/ false);
    // video_ctx is non-null only for video input, which we don't support here.
    // (Image/audio decode produces a bitmap with a null video_ctx.)
    return wrap.bitmap;  // may be nullptr on failure
}

// Run the mtmd tokenize + per-chunk eval path: turns the marker-containing text
// + bitmaps into chunks and decodes each into the session's KV cache.
// Returns false on error. Does NOT free the bitmaps (caller owns).
static bool mtmd_inject(mtmd_context * mtmd_ctx, const llama_vocab * /*vocab*/,
                        llama_context * ctx, const std::string & text,
                        const std::vector<mtmd_bitmap *> & bitmaps) {
    mtmd_input_text input_text;
    input_text.text          = text.c_str();
    input_text.add_special   = true;
    input_text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    std::vector<const mtmd_bitmap *> bptrs;
    bptrs.reserve(bitmaps.size());
    for (auto * b : bitmaps) bptrs.push_back(b);

    int32_t rc = mtmd_tokenize(mtmd_ctx, chunks, &input_text,
                               bptrs.data(), bptrs.size());
    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        return false;
    }

    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    llama_pos n_past = 0;
    // start from the current cache position (so multi-turn inject composes)
    llama_pos cur_max = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (cur_max >= 0) n_past = cur_max + 1;

    bool ok = true;
    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        llama_pos new_n_past = n_past;
        int32_t r = mtmd_helper_eval_chunk_single(
            mtmd_ctx, ctx, chunk, n_past, /*seq_id*/ 0, /*n_batch*/ 512,
            /*logits_last*/ (i == n_chunks - 1), &new_n_past);
        if (r != 0) { ok = false; break; }
        n_past = new_n_past;
    }
    mtmd_input_chunks_free(chunks);
    return ok;
}

int main(int argc, char ** argv) {
    ServerConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string{};
        };
        if      (a == "--port")          cfg.port = std::atoi(next().c_str());
        else if (a == "--model" || a == "-m") cfg.model_path = next();
        else if (a == "--mmproj")        cfg.mmproj_path = next();
        else if (a == "--n-gpu-layers" || a == "-ngl") cfg.n_gpu_layers = std::atoi(next().c_str());
        else if (a == "--ctx-size" || a == "-c") cfg.ctx_size = std::atoi(next().c_str());
        else if (a == "--n-batch")      cfg.n_batch = std::atoi(next().c_str());
        else if (a == "--chat-template")      cfg.chat_template = next();
        else if (a == "--chat-template-file") cfg.chat_template = read_file_contents(next());
        else if (a == "--jinja")              cfg.use_jinja = true;
        else if (a == "--no-jinja")           cfg.use_jinja = false;
        else if (a == "--no-chat-template")   cfg.enable_chat_template = false;
        else if (a == "--system-prompt")      cfg.system_prompt = next();
        else if (a == "--chat-template-kwargs") {
            // Parse a JSON object string, e.g. '{"k":"v"}'. Each value is stored
            // as its JSON serialization (mirrors common/arg.cpp); re-parsed by
            // common_chat_templates_apply when rendering.
            std::string v = next();
            json parsed = json::parse(v);
            if (!parsed.is_object()) {
                std::cerr << "error: --chat-template-kwargs must be a JSON object\n";
                return 2;
            }
            for (auto it = parsed.begin(); it != parsed.end(); ++it) {
                cfg.chat_template_kwargs[it.key()] = it.value().dump();
            }
        }
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "multimodal-server [options]\n"
                "  -m, --model PATH          model gguf (required)\n"
                "      --mmproj PATH         multimodal projector gguf (enables image/audio)\n"
                "      --port N              HTTP port (default 8080)\n"
                "  -ngl,--n-gpu-layers N     GPU layers (default 99)\n"
                "  -c, --ctx-size N          context per session (default 4096)\n"
                "      --n-batch N           batch size (default 2048)\n"
                "      --chat-template TPL   Jinja chat template override (else model default)\n"
                "      --chat-template-file F  read Jinja chat template override from a file\n"
                "      --jinja / --no-jinja  use the Jinja template engine (default: enabled)\n"
                "      --no-chat-template    disable templating: 'messages' inject is rejected\n"
                "      --system-prompt TEXT  system prompt prepended to every conversation\n"
                "      --chat-template-kwargs JSON  extra Jinja vars, e.g. '{\"k\":\"v\"}'\n";
            return 0;
        }
    }
    if (cfg.model_path.empty()) {
        std::cerr << "error: --model is required (see --help)\n";
        return 2;
    }

    // Validate a user-supplied chat template up front (fail fast), mirroring
    // common/arg.cpp. Without --jinja only commonly-used templates are accepted.
    if (!cfg.chat_template.empty() &&
        !common_chat_verify_template(cfg.chat_template, cfg.use_jinja)) {
        std::cerr << "error: the supplied chat template is not supported: "
                  << cfg.chat_template << "\n";
        if (cfg.use_jinja) {
            std::cerr << "(template failed Jinja validation)\n";
        } else {
            std::cerr << "note: started without --jinja, only commonly used templates are accepted\n";
        }
        return 1;
    }

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) std::cerr << text;
    }, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = cfg.n_gpu_layers;
    std::cerr << "loading model: " << cfg.model_path << " ...\n";
    AppState app;
    app.model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
    if (!app.model) { std::cerr << "error: failed to load model\n"; return 1; }
    app.vocab = llama_model_get_vocab(app.model);
    app.chat_templates = common_chat_templates_init(app.model, /* override */ cfg.chat_template);
    app.use_jinja            = cfg.use_jinja;
    app.enable_chat_template = cfg.enable_chat_template;
    app.chat_template_kwargs = cfg.chat_template_kwargs;
    app.system_prompt        = cfg.system_prompt;
    if (app.chat_templates) {
        std::cerr << "chat template loaded (use_jinja=" << (app.use_jinja ? "true" : "false")
                  << ", enabled=" << (app.enable_chat_template ? "true" : "false") << ").\n";
    } else {
        std::cerr << "warning: no chat template in model; 'messages' inject will fail.\n";
    }
    app.n_ctx_per_session = cfg.ctx_size;
    app.n_batch = cfg.n_batch;
    std::cerr << "model loaded.\n";

    // Load the multimodal projector (if given). Encoder-free models like Gemma 4
    // 12B have a tiny projector (no heavy ViT), so this is cheap.
    if (!cfg.mmproj_path.empty()) {
        std::cerr << "loading mmproj: " << cfg.mmproj_path << " ...\n";
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu = (cfg.n_gpu_layers > 0);
        mp.warmup  = true;
        app.mtmd_ctx = mtmd_init_from_file(cfg.mmproj_path.c_str(), app.model, mp);
        if (!app.mtmd_ctx) {
            std::cerr << "error: failed to load mmproj\n";
            return 1;
        }
        std::cerr << "mmproj loaded (vision=" << mtmd_support_vision(app.mtmd_ctx)
                  << " audio=" << mtmd_support_audio(app.mtmd_ctx) << ").\n";
    }

    // Warm up CUDA / kernel JIT by running a tiny generation. Without this the
    // first real request pays ~10s+ of CUDA initialization (graph capture, kernel
    // JIT) as its TTFT.
    {
        std::cerr << "warming up...\n";
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = 512; cp.no_perf = true;
        if (llama_context * wctx = llama_init_from_model(app.model, cp)) {
            GenParams gp{5, 0.0f, 1.0f, 0};
            run_generation(app, wctx, gp, nullptr);
            llama_free(wctx);
            std::cerr << "warmup done.\n";
        }
    }

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // ---- GET /info : capabilities (what the model + projector support) ----
    svr.Get("/info", [&](const httplib::Request &, httplib::Response &res) {
        json body = {
            {"model_loaded", app.model != nullptr},
            {"supports_vision", app.mtmd_ctx ? mtmd_support_vision(app.mtmd_ctx) : false},
            {"supports_audio", app.mtmd_ctx ? mtmd_support_audio(app.mtmd_ctx) : false},
            {"audio_sample_rate", app.mtmd_ctx ? mtmd_get_audio_sample_rate(app.mtmd_ctx) : 0},
        };
        res.set_content(body.dump(), "application/json");
    });

    // ---- POST /sessions : create a session ----
    svr.Post("/sessions", [&](const httplib::Request &, httplib::Response &res) {
        llama_context * ctx = create_session_ctx(app);
        if (!ctx) {
            res.status = 500;
            res.set_content(error_body("failed to create context", 500).dump(), "application/json");
            return;
        }
        const int64_t n = register_session(app, ctx);
        json body = {{"session_id", make_session_id(n)}};
        res.set_content(body.dump(), "application/json");
    });

    // ---- POST /sessions/{id}/fork : copy this session's KV into a NEW session ----
    //
    // Approach A' (see docs/decisions/0004-fork-copy-semantics.md): snapshot the
    // source sequence's KV via llama_state_seq_get_data and restore it into a
    // fresh context (llama_state_seq_set_data). The forked session owns an
    // independent K/V copy, so source and fork generate independently. Forkable
    // at any point — it snapshots whatever the source cache currently holds
    // (after a text inject, an audio inject, or a generate). The source session
    // is untouched. (Slice 5 will swap this for llama_memory_seq_cp once
    // sessions become sequences in a pooled context — see the ADR.)
    svr.Post(R"(/sessions/[^/]+/fork)", [&](const httplib::Request &req, httplib::Response &res) {
        std::string sid = extract_session_id(req.path, "fork");
        llama_context * src = lookup_session(app, parse_session_id_num(sid));
        if (!src) {
            res.status = 404;
            res.set_content(error_body("unknown session", 404).dump(), "application/json");
            return;
        }

        const double t0 = now_s();
        // Snapshot source seq 0's KV into a host buffer, then restore into a new ctx.
        const size_t seq_size = llama_state_seq_get_size(src, /*seq_id*/ 0);
        std::vector<uint8_t> buf(seq_size);
        llama_state_seq_get_data(src, buf.data(), buf.size(), /*seq_id*/ 0);
        llama_context * dst = create_session_ctx(app);
        if (!dst) {
            res.status = 500;
            res.set_content(error_body("failed to create forked context", 500).dump(), "application/json");
            return;
        }
        llama_state_seq_set_data(dst, buf.data(), buf.size(), /*dest_seq_id*/ 0);
        // The restore carries KV but NOT the output/logits buffer, so the forked
        // session can't sample until logits are recomputed. Re-decode the source's
        // last token at its existing position (same token -> identical K/V, plus a
        // fresh logits row). When the last token is unknown (a session whose last
        // inject ended in an image/audio embedding chunk), there is no discrete
        // token to re-decode; the client should inject one text token before
        // generating from such a fork (the chat protocol always closes a turn with
        // text markers, so this is the normal flow).
        const llama_token src_last = get_last_token(app, parse_session_id_num(sid));
        if (src_last != LLAMA_TOKEN_NULL) {
            llama_memory_t dmem = llama_get_memory(dst);
            const llama_pos pmax = llama_memory_seq_pos_max(dmem, 0);
            // Re-decoding at an already-cached position won't produce a logits row,
            // so free the last cell then re-evaluate the source's last token there:
            // it recomputes identical K/V and yields a fresh logits row, making the
            // forked session immediately generation-ready.
            llama_memory_seq_rm(dmem, 0, pmax, pmax + 1);
            llama_batch b = llama_batch_init(/*n_tokens*/ 1, /*embd*/ 0, /*n_seq_max*/ 1);
            b.n_tokens    = 1;
            b.token[0]    = src_last;
            b.pos[0]      = pmax;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0]= 0;
            b.logits[0]   = 1;  // request logits for this token
            llama_decode(dst, b);
            llama_batch_free(b);
        }
        const double dt = now_s() - t0;

        const int64_t new_n = register_session(app, dst);
        set_last_token(app, new_n, src_last);  // the fork inherits the source's last token
        const int dst_size  = llama_memory_seq_pos_max(llama_get_memory(dst), 0) + 1;
        json body = {
            {"session_id", make_session_id(new_n)},
            {"forked_from", sid},
            {"cache_size", dst_size},
            {"fork_ms", (int)(dt * 1000)},
        };
        res.set_content(body.dump(), "application/json");
    });

    // ---- POST /sessions/{id}/inject : text -> KV cache (no generation) ----
    svr.Post(R"(/sessions/[^/]+/inject)", [&](const httplib::Request &req, httplib::Response &res) {
        std::string sid = extract_session_id(req.path, "inject");
        const int64_t sid_num = parse_session_id_num(sid);
        llama_context * ctx = lookup_session(app, sid_num);
        if (!ctx) {
            res.status = 404;
            res.set_content(error_body("unknown session", 404).dump(), "application/json");
            return;
        }
        std::string text;
        bool used_template = false;
        bool used_multimodal = false;
        bool return_prompt = false;  // opt-in: echo the rendered prompt (chat-template parity tests)
        try {
            auto j = json::parse(req.body);
            if (j.contains("messages")) {
                if (!app.enable_chat_template) {
                    res.status = 400;
                    res.set_content(error_body(
                        "chat template disabled (started with --no-chat-template); "
                        "use 'text' instead of 'messages'", 400).dump(), "application/json");
                    return;
                }
                // Apply the model's chat template to a list of {role, content} msgs.
                // content may be a string OR an array of parts (text/image/audio).
                return_prompt = j.value("return_prompt", false);
                if (!app.chat_templates) {
                    res.status = 500;
                    res.set_content(error_body("model has no chat template", 500).dump(), "application/json");
                    return;
                }
                // First pass: collect any media parts across all messages, replacing
                // each with the mtmd media marker in the text content.
                std::vector<mtmd_bitmap *> bitmaps;  // owned; freed below
                std::string media_marker = mtmd_default_marker();
                auto replace_media_in_content = [&](const json & content) -> std::string {
                    if (content.is_string()) {
                        return content.get<std::string>();
                    }
                    // array of parts
                    std::string out;
                    for (const auto & part : content) {
                        std::string ptype = part.value("type", "");
                        if (ptype == "text") {
                            out += part.value("text", "");
                        } else if (ptype == "image" || ptype == "image_url") {
                            if (!app.mtmd_ctx) {
                                throw std::runtime_error("image part requires --mmproj to be loaded");
                            }
                            // accept {"data": "<base64>"} or {"image_url": {"url": "data:image/png;base64,<..>"}}
                            std::string b64;
                            if (part.contains("data")) {
                                b64 = part["data"].get<std::string>();
                            } else if (part.contains("image_url")) {
                                b64 = part["image_url"].value("url", "");
                            }
                            // strip optional data-URL prefix
                            size_t comma = b64.find(',');
                            if (b64.rfind("data:", 0) == 0 && comma != std::string::npos) {
                                b64 = b64.substr(comma + 1);
                            }
                            std::string bytes = base64_decode(b64);
                            mtmd_bitmap * bmp = bitmap_from_media_bytes(app.mtmd_ctx, bytes);
                            if (!bmp) {
                                throw std::runtime_error("failed to decode image");
                            }
                            bitmaps.push_back(bmp);
                            out += media_marker;
                            used_multimodal = true;
                        } else if (ptype == "audio" || ptype == "input_audio") {
                            if (!app.mtmd_ctx) {
                                throw std::runtime_error("audio part requires --mmproj to be loaded");
                            }
                            if (!mtmd_support_audio(app.mtmd_ctx)) {
                                throw std::runtime_error("audio part requires a projector with audio support");
                            }
                            // accept {"data": "<b64>"} or {"input_audio": {"data": "<data-url or b64>"}}
                            std::string b64;
                            if (part.contains("data")) {
                                b64 = part["data"].get<std::string>();
                            } else if (part.contains("input_audio")) {
                                b64 = part["input_audio"].value("data", "");
                            }
                            // strip optional data-URL prefix (data:audio/wav;base64,...)
                            size_t comma = b64.find(',');
                            if (b64.rfind("data:", 0) == 0 && comma != std::string::npos) {
                                b64 = b64.substr(comma + 1);
                            }
                            std::string bytes = base64_decode(b64);
                            mtmd_bitmap * bmp = bitmap_from_media_bytes(app.mtmd_ctx, bytes);
                            if (!bmp) {
                                throw std::runtime_error("failed to decode audio");
                            }
                            bitmaps.push_back(bmp);
                            out += media_marker;
                            used_multimodal = true;
                        }
                    }
                    return out;
                };

                common_chat_templates_inputs inputs;
                inputs.add_generation_prompt = j.value("add_generation_prompt", true);
                inputs.use_jinja             = app.use_jinja;
                inputs.chat_template_kwargs  = app.chat_template_kwargs;
                // Prepend a global --system-prompt if configured.
                if (!app.system_prompt.empty()) {
                    common_chat_msg sys;
                    sys.role    = "system";
                    sys.content = app.system_prompt;
                    inputs.messages.push_back(std::move(sys));
                }
                for (const auto & m : j["messages"]) {
                    common_chat_msg msg;
                    msg.role    = m.value("role", "user");
                    msg.content = replace_media_in_content(m["content"]);
                    inputs.messages.push_back(std::move(msg));
                }
                common_chat_params cp = common_chat_templates_apply(app.chat_templates.get(), inputs);
                text = cp.prompt;
                used_template = true;

                // If we have media, run the mtmd tokenize+eval path instead of the
                // plain text tokenize+decode below.
                if (used_multimodal) {
                    double t0 = now_s();
                    bool ok = mtmd_inject(app.mtmd_ctx, app.vocab, ctx, text, bitmaps);
                    const double dt = now_s() - t0;
                    for (auto * b : bitmaps) mtmd_bitmap_free(b);
                    if (!ok) {
                        res.status = 500;
                        res.set_content(error_body("mtmd tokenize/decode failed", 500).dump(), "application/json");
                        return;
                    }
                    const int new_size = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
                    json body = {
                        {"session_id", sid},
                        {"cache_size", new_size},
                        {"inject_ms", (int)(dt * 1000)},
                        {"chat_template_applied", used_template},
                        {"used_multimodal", true},
                        {"n_media", bitmaps.size()},
                    };
                    if (return_prompt) body["prompt"] = text;
                    res.set_content(body.dump(), "application/json");
                    return;
                }
            } else if (j.contains("audio")) {
                std::string b64 = j["audio"].get<std::string>();
                size_t comma = b64.find(',');
                if (b64.rfind("data:", 0) == 0 && comma != std::string::npos) {
                    b64 = b64.substr(comma + 1);
                }
                std::string bytes = base64_decode(b64);

                if (bytes.empty()) {
                    res.status = 400;
                    res.set_content(error_body("audio buffer cannot be empty", 400).dump(), "application/json");
                    return;
                }

                if (bytes.size() % sizeof(float) != 0) {
                    res.status = 400;
                    res.set_content(error_body("audio buffer size must be a multiple of 4 bytes (sizeof(float))", 400).dump(), "application/json");
                    return;
                }

                size_t n_samples = bytes.size() / sizeof(float);
                if (n_samples % 640 != 0) {
                    res.status = 400;
                    res.set_content(error_body("audio sample count must be a multiple of 640 (Gemma 4 audio frame size)", 400).dump(), "application/json");
                    return;
                }

                if (!app.mtmd_ctx) {
                    res.status = 400;
                    res.set_content(error_body("audio requires --mmproj to be loaded", 400).dump(), "application/json");
                    return;
                }

                const float * samples = reinterpret_cast<const float *>(bytes.data());
                mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio(n_samples, samples);

                if (!bmp) {
                    res.status = 400;
                    res.set_content(error_body("failed to decode audio bitmap", 400).dump(), "application/json");
                    return;
                }

                std::string dummy_prompt = mtmd_default_marker();
                mtmd_input_text input_text;
                input_text.text          = dummy_prompt.c_str();
                input_text.add_special   = false;
                input_text.parse_special = true;

                mtmd_input_chunks * chunks = mtmd_input_chunks_init();
                const mtmd_bitmap * bptrs[1] = {bmp};
                int32_t rc = mtmd_tokenize(app.mtmd_ctx, chunks, &input_text, bptrs, 1);
                if (rc != 0) {
                    mtmd_bitmap_free(bmp);
                    mtmd_input_chunks_free(chunks);
                    res.status = 500;
                    res.set_content(error_body("mtmd_tokenize failed", 500).dump(), "application/json");
                    return;
                }

                const size_t n_chunks = mtmd_input_chunks_size(chunks);
                bool ok = true;
                bool has_audio = false;
                llama_pos new_n_past = 0;
                for (size_t i = 0; i < n_chunks; ++i) {
                    const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
                    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                        has_audio = true;
                        llama_pos n_past = 0;
                        llama_pos cur_max = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
                        if (cur_max >= 0) n_past = cur_max + 1;

                        int32_t r = mtmd_helper_eval_chunk_single(
                            app.mtmd_ctx, ctx, chunk, n_past, /*seq_id*/ 0, /*n_batch*/ 512,
                            /*logits_last*/ true, &new_n_past);
                        if (r != 0) {
                            ok = false;
                            break;
                        }
                    }
                }
                if (!has_audio) {
                    ok = false;
                }

                mtmd_bitmap_free(bmp);
                mtmd_input_chunks_free(chunks);

                if (!ok) {
                    res.status = 500;
                    res.set_content(error_body("failed to decode streaming audio chunk", 500).dump(), "application/json");
                    return;
                }

                const int new_size = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
                json body = {
                    {"session_id", sid},
                    {"cache_size", new_size},
                    {"chat_template_applied", false},
                    {"used_multimodal", true},
                    {"n_media", 1}
                };
                res.set_content(body.dump(), "application/json");
                return;
            } else {
                text = j.value("text", "");
            }
        }
        catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_body(std::string("bad request: ") + e.what(), 400).dump(), "application/json");
            return;
        }
        if (text.empty()) {
            res.status = 400;
            res.set_content(error_body("'text' or 'messages' is required", 400).dump(), "application/json");
            return;
        }
        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;
        const int n_needed = -llama_tokenize(app.vocab, text.c_str(), text.size(),
                                             nullptr, 0, is_first, true);
        if (n_needed < 0) {
            res.status = 500;
            res.set_content(error_body("tokenize failed", 500).dump(), "application/json");
            return;
        }
        std::vector<llama_token> toks(n_needed);
        if (llama_tokenize(app.vocab, text.c_str(), text.size(), toks.data(), toks.size(),
                           is_first, true) < 0) {
            res.status = 500;
            res.set_content(error_body("tokenize failed", 500).dump(), "application/json");
            return;
        }
        const int n_ctx = llama_n_ctx(ctx);
        const int used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        if (used + (int)toks.size() > n_ctx) {
            res.status = 409;
            res.set_content(error_body("session context full", 409).dump(), "application/json");
            return;
        }
        const double t0 = now_s();
        llama_batch batch = llama_batch_get_one(toks.data(), (int)toks.size());
        if (llama_decode(ctx, batch) != 0) {
            res.status = 500;
            res.set_content(error_body("llama_decode failed", 500).dump(), "application/json");
            return;
        }
        const double dt = now_s() - t0;
        if (!toks.empty()) set_last_token(app, sid_num, toks.back());  // for fork logits refresh
        const int new_size = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        json body = {
            {"session_id", sid},
            {"tokens_injected", toks.size()},
            {"cache_size", new_size},
            {"inject_ms", (int)(dt * 1000)},
            {"chat_template_applied", used_template},
        };
        if (return_prompt) body["prompt"] = text;
        res.set_content(body.dump(), "application/json");
    });

    // ---- POST /sessions/{id}/generate : streaming (SSE) or JSON ----
    svr.Post(R"(/sessions/[^/]+/generate)", [&](const httplib::Request &req, httplib::Response &res) {
        std::string sid = extract_session_id(req.path, "generate");
        const int64_t sid_num = parse_session_id_num(sid);
        llama_context * ctx = lookup_session(app, sid_num);
        if (!ctx) {
            res.status = 404;
            res.set_content(error_body("unknown session", 404).dump(), "application/json");
            return;
        }
        GenParams gp;
        bool stream = false;
        try {
            auto j = json::parse(req.body);
            gp.max_tokens = j.value("max_tokens", 256);
            gp.temp       = j.value("temperature", 0.8f);
            gp.top_p      = j.value("top_p", 1.0f);
            gp.seed       = j.value("seed", -1);
            stream        = j.value("stream", false);
        } catch (...) {}

        if (!stream) {
            // slice-1 behavior: whole response as JSON.
            GenResult r = run_generation(app, ctx, gp, nullptr);
            if (!r.ids.empty()) set_last_token(app, sid_num, (llama_token)r.ids.back());
            const double tok_s = (r.gen_s > 0) ? (r.ids.size() / r.gen_s) : 0.0;
            json body = {
                {"session_id", sid},
                {"text", r.text},
                {"tokens", r.ids},
                {"n_tokens", r.ids.size()},
                {"gen_ms", (int)(r.gen_s * 1000)},
                {"tokens_per_s", tok_s},
            };
            res.set_content(body.dump(), "application/json");
            return;
        }

        // Streaming: text/event-stream. Generate tokens directly inside the
        // chunked provider callback — httplib calls it on its worker thread, and
        // each token piece is written to the DataSink the instant it is produced.
        // No separate thread, no lifetime issues. The provider blocks httplib's
        // worker until generation finishes, which is fine (one request at a time).
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider(
            "text/event-stream",
            [&app, ctx, gp, sid, sid_num](size_t, httplib::DataSink & ds) -> bool {
                GenResult r = run_generation(app, ctx, gp, [&ds](const std::string & piece, int64_t id) {
                    std::string ev = sse_event(json{
                        {"type", "token"}, {"token", piece}, {"id", id}
                    });
                    return ds.write(ev.data(), ev.size());
                });
                if (!r.ids.empty()) set_last_token(app, sid_num, (llama_token)r.ids.back());
                const double tok_s = (r.gen_s > 0) ? (r.ids.size() / r.gen_s) : 0.0;
                std::string done = sse_event(json{
                    {"type", "done"},
                    {"n_tokens", r.ids.size()},
                    {"gen_ms", (int)(r.gen_s * 1000)},
                    {"tokens_per_s", tok_s},
                });
                ds.write(done.data(), done.size());
                ds.done();
                return true;
            }
        );
    });

    // ---- DELETE /sessions/{id} : free the session ----
    svr.Delete(R"(/sessions/[^/]+)", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string prefix = "/sessions/";
        if (req.path.rfind(prefix, 0) != 0 || req.path.find('/', prefix.size()) != std::string::npos) {
            res.status = 404;
            res.set_content(error_body("not found", 404).dump(), "application/json");
            return;
        }
        std::string sid = req.path.substr(prefix.size());
        int64_t n = parse_session_id_num(sid);
        {
            std::lock_guard<std::mutex> lk(app.mu);
            auto it = (n > 0) ? app.sessions.find(n) : app.sessions.end();
            if (it == app.sessions.end()) {
                res.status = 404;
                res.set_content(error_body("unknown session", 404).dump(), "application/json");
                return;
            }
            llama_free(it->second);
            app.sessions.erase(it);
            app.last_tokens.erase(n);
        }
        res.set_content(json{{"session_id", sid}, {"deleted", true}}.dump(), "application/json");
    });

    std::cerr << "multimodal-server listening on 0.0.0.0:" << cfg.port << "\n";
    if (!svr.listen("0.0.0.0", cfg.port)) {
        std::cerr << "error: failed to listen on port " << cfg.port << "\n";
        llama_model_free(app.model);
        llama_backend_free();
        return 1;
    }

    {
        std::lock_guard<std::mutex> lk(app.mu);
        for (auto & [_, ctx] : app.sessions) llama_free(ctx);
        app.sessions.clear();
    }
    llama_model_free(app.model);
    llama_backend_free();
    return 0;
}
