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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

#include "common.h"
#include "chat.h"
#include "ggml.h"
#include "llama.h"

// 'json' (nlohmann::ordered_json) is provided by common/chat.h.

namespace {

struct ServerConfig {
    std::string model_path;
    int  port          = 8080;
    int  n_gpu_layers  = 99;
    int  ctx_size      = 4096;
    int  n_batch       = 2048;
};

// One model loaded once; many sessions (each its own context / KV cache).
struct AppState {
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr chat_templates;  // built from the model; applies its chat template
    int n_ctx_per_session = 4096;
    int n_batch = 2048;
    std::mutex mu;
    std::map<int64_t, llama_context *> sessions;  // owns the contexts
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

std::string make_session_id(int64_t n) {
    std::ostringstream os;
    os << "s_" << n;
    return os.str();
}

// Parse the {id} out of a "/sessions/{id}/..." path. Returns empty on miss.
std::string extract_session_id(const std::string & path, const std::string & action) {
    const std::string prefix = "/sessions/";
    if (path.rfind(prefix, 0) != 0) return "";
    const size_t rest = path.find('/', prefix.size());
    if (rest == std::string::npos) return "";
    const std::string id = path.substr(prefix.size(), rest - prefix.size());
    const std::string act = path.substr(rest);
    if (act != "/" + action) return "";
    return id;
}

int64_t parse_session_id_num(const std::string & id) {
    const std::string pfx = "s_";
    if (id.rfind(pfx, 0) != 0) return -1;
    try { return std::stoll(id.substr(pfx.size())); }
    catch (...) { return -1; }
}

json error_body(const std::string & msg, int code) {
    return json{{"error", msg}, {"code", code}};
}

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

std::string sse_event(const json & j) {
    std::ostringstream os;
    os << "data: " << j.dump() << "\n\n";
    return os.str();
}

} // namespace

int main(int argc, char ** argv) {
    ServerConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string{};
        };
        if      (a == "--port")          cfg.port = std::atoi(next().c_str());
        else if (a == "--model" || a == "-m") cfg.model_path = next();
        else if (a == "--n-gpu-layers" || a == "-ngl") cfg.n_gpu_layers = std::atoi(next().c_str());
        else if (a == "--ctx-size" || a == "-c") cfg.ctx_size = std::atoi(next().c_str());
        else if (a == "--n-batch")      cfg.n_batch = std::atoi(next().c_str());
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "multimodal-server [options]\n"
                "  -m, --model PATH          model gguf (required)\n"
                "      --port N              HTTP port (default 8080)\n"
                "  -ngl,--n-gpu-layers N     GPU layers (default 99)\n"
                "  -c, --ctx-size N          context per session (default 4096)\n"
                "      --n-batch N           batch size (default 2048)\n";
            return 0;
        }
    }
    if (cfg.model_path.empty()) {
        std::cerr << "error: --model is required (see --help)\n";
        return 2;
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
    app.chat_templates = common_chat_templates_init(app.model, /* override */ "");
    if (app.chat_templates) {
        std::cerr << "chat template loaded.\n";
    } else {
        std::cerr << "warning: no chat template in model; 'messages' inject will fail.\n";
    }
    app.n_ctx_per_session = cfg.ctx_size;
    app.n_batch = cfg.n_batch;
    std::cerr << "model loaded.\n";

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

    // ---- POST /sessions : create a session ----
    svr.Post("/sessions", [&](const httplib::Request &, httplib::Response &res) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = app.n_ctx_per_session;
        cp.n_batch = std::min<int>(app.n_batch, app.n_ctx_per_session);
        cp.no_perf = true;
        llama_context * ctx = llama_init_from_model(app.model, cp);
        if (!ctx) {
            res.status = 500;
            res.set_content(error_body("failed to create context", 500).dump(), "application/json");
            return;
        }
        const int64_t n = app.next_id.fetch_add(1);
        {
            std::lock_guard<std::mutex> lk(app.mu);
            app.sessions[n] = ctx;
        }
        json body = {{"session_id", make_session_id(n)}};
        res.set_content(body.dump(), "application/json");
    });

    // ---- POST /sessions/{id}/inject : text -> KV cache (no generation) ----
    svr.Post(R"(/sessions/[^/]+/inject)", [&](const httplib::Request &req, httplib::Response &res) {
        std::string sid = extract_session_id(req.path, "inject");
        llama_context * ctx = lookup_session(app, parse_session_id_num(sid));
        if (!ctx) {
            res.status = 404;
            res.set_content(error_body("unknown session", 404).dump(), "application/json");
            return;
        }
        std::string text;
        bool used_template = false;
        try {
            auto j = json::parse(req.body);
            if (j.contains("messages")) {
                // Apply the model's chat template to a list of {role, content} msgs.
                if (!app.chat_templates) {
                    res.status = 500;
                    res.set_content(error_body("model has no chat template", 500).dump(), "application/json");
                    return;
                }
                common_chat_templates_inputs inputs;
                inputs.add_generation_prompt = j.value("add_generation_prompt", true);
                for (const auto & m : j["messages"]) {
                    common_chat_msg msg;
                    msg.role    = m.value("role", "user");
                    msg.content = m.value("content", "");
                    inputs.messages.push_back(std::move(msg));
                }
                common_chat_params cp = common_chat_templates_apply(app.chat_templates.get(), inputs);
                text = cp.prompt;
                used_template = true;
            } else {
                text = j.value("text", "");
            }
        }
        catch (...) {
            res.status = 400;
            res.set_content(error_body("invalid JSON body", 400).dump(), "application/json");
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
        const int new_size = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        json body = {
            {"session_id", sid},
            {"tokens_injected", toks.size()},
            {"cache_size", new_size},
            {"inject_ms", (int)(dt * 1000)},
            {"chat_template_applied", used_template},
        };
        res.set_content(body.dump(), "application/json");
    });

    // ---- POST /sessions/{id}/generate : streaming (SSE) or JSON ----
    svr.Post(R"(/sessions/[^/]+/generate)", [&](const httplib::Request &req, httplib::Response &res) {
        std::string sid = extract_session_id(req.path, "generate");
        llama_context * ctx = lookup_session(app, parse_session_id_num(sid));
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
            [&app, ctx, gp, sid](size_t, httplib::DataSink & ds) -> bool {
                GenResult r = run_generation(app, ctx, gp, [&ds](const std::string & piece, int64_t id) {
                    std::string ev = sse_event(json{
                        {"type", "token"}, {"token", piece}, {"id", id}
                    });
                    return ds.write(ev.data(), ev.size());
                });
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
