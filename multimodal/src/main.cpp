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

#include <algorithm>
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
#include <filesystem>

#include "httplib.h"
#include "nlohmann/json.hpp"

#include "util.h"

#include "common.h"
#include "chat.h"
#include "sampling.h"
#include "json-schema-to-grammar.h"
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

static std::string detect_config_path(const std::string & model_path) {
    try {
        std::filesystem::path mp(model_path);
        if (!std::filesystem::exists(mp)) {
            return "";
        }
        // 1. model_path_without_ext + ".json"
        std::filesystem::path p1 = mp;
        p1.replace_extension(".json");
        if (std::filesystem::exists(p1)) return p1.string();

        // 2. model_path_without_ext + "_config.json"
        std::filesystem::path p2 = mp;
        p2.replace_extension("");
        p2 += "_config.json";
        if (std::filesystem::exists(p2)) return p2.string();

        // 3. model_dir / "config.json"
        std::filesystem::path dir = mp.parent_path();
        std::filesystem::path p3 = dir / "config.json";
        if (std::filesystem::exists(p3)) return p3.string();

        // 4. model_dir / "model_config.json"
        std::filesystem::path p4 = dir / "model_config.json";
        if (std::filesystem::exists(p4)) return p4.string();
    } catch (...) {
        // ignore filesystem exceptions and return empty
    }
    return "";
}

static ModelConfig load_config_for_model(const std::string & model_path, const std::string & explicit_config_path, const struct llama_model * model) {
    std::string path_to_load;

    if (!explicit_config_path.empty()) {
        if (std::filesystem::exists(explicit_config_path)) {
            path_to_load = explicit_config_path;
        } else {
            std::cerr << "warning: explicit config file not found: " << explicit_config_path << "\n";
            std::cerr << "falling back to metadata auto-detection...\n";
        }
    } else {
        path_to_load = detect_config_path(model_path);
    }

    ModelConfig cfg;
    bool loaded = false;
    if (!path_to_load.empty()) {
        std::cerr << "loading model config from: " << path_to_load << " ...\n";
        try {
            std::string content = read_file_contents(path_to_load);
            auto j = nlohmann::ordered_json::parse(content);
            cfg = ModelConfig::from_json(j);
            loaded = true;
        } catch (const std::exception & e) {
            std::cerr << "warning: failed to load or parse config from " << path_to_load << ": " << e.what() << "\n";
            std::cerr << "falling back to metadata auto-detection...\n";
        }
    }

    if (!loaded) {
        // Auto-detect / Fallback based on model metadata
        int32_t n_embd = llama_model_n_embd(model);
        char desc[512] = "";
        llama_model_desc(model, desc, sizeof(desc));
        std::string desc_str(desc);

        std::cerr << "info: no config file loaded, auto-detecting model properties from GGUF metadata...\n";
        std::cerr << "  model embedding length: " << n_embd << "\n";
        std::cerr << "  model description: " << desc_str << "\n";

        if (n_embd == 2560 || desc_str.find("E4B") != std::string::npos || desc_str.find("4B") != std::string::npos) {
            cfg.audio_frame_size = 1; // E4B / gemma4a models do not require a divisor constraint in server
            std::cerr << "  detected Gemma 4 E4B (4B) model. Setting audio_frame_size = 1.\n";
        } else {
            cfg.audio_frame_size = 640; // Default Gemma 4 12B audio frame size
            std::cerr << "  defaulting to Gemma 4 12B model. Setting audio_frame_size = 640.\n";
        }
    }

    // Guard audio_frame_size against invalid non-positive values
    if (cfg.audio_frame_size <= 0) {
        std::cerr << "warning: invalid audio_frame_size (" << cfg.audio_frame_size << ") in config, falling back to 640.\n";
        cfg.audio_frame_size = 640;
    }

    return cfg;
}

struct ServerConfig {
    std::string model_path;
    std::string mmproj_path;   // multimodal projector gguf (empty = text-only)
    int  port          = 8080;
    int  n_gpu_layers  = 99;
    int  ctx_size      = 4096;
    int  n_batch       = 2048;
    bool allow_cpu     = false;  // GPU model offload is required unless explicitly opted out

    // Chat-template handling (mirrors llama-server / common/arg.cpp). We reuse
    // common/'s templating; these just feed it the same inputs llama-server does.
    std::string chat_template;                                // --chat-template / --chat-template-file (override; empty = model's default)
    bool        use_jinja            = true;                  // --jinja / --no-jinja (default true, like llama-server)
    bool        enable_chat_template = true;                  // --no-chat-template disables the 'messages' inject path
    std::map<std::string, std::string> chat_template_kwargs;  // --chat-template-kwargs (key -> JSON value serialized as a string)
    std::string system_prompt;                                // --system-prompt (prepended as a system message to every conversation)
    std::string config_path;                                  // --config (optional path to model config JSON file)
};

double now_s();

// A session parked in host RAM (offloaded from VRAM). The live llama_context
// has been freed (that is what actually releases VRAM — llama_memory_seq_rm only
// clears logical cells, not the pre-allocated KV buffer); this holds the
// serialized seq-0 state so the session can be loaded back into a fresh
// context unchanged. `state` is the per-sequence serializable state (KV +, for
// hybrid models like Qwen3.5/MiniCPM-V-4.6, the SSM/Mamba recurrent state).
struct OffloadedState {
    std::vector<uint8_t> state;                       // llama_state_seq_get_data output
    llama_token          last_token = LLAMA_TOKEN_NULL; // for logits refresh on load
    int                  cache_size = 0;              // tokens at offload time
    size_t               state_bytes = 0;             // == state.size(); symmetric offload/load
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
    ModelConfig                          model_cfg;
    std::mutex mu;
    std::map<int64_t, llama_context *> sessions;  // owns the contexts
    // Last token in each session's cache (seq 0). Used to refresh logits after a
    // fork restores KV via llama_state_seq_set_data (the restore carries KV but
    // NOT the output/logits buffer, so the forked session must re-decode its last
    // token before it can generate). LLAMA_TOKEN_NULL when unknown (e.g. a session
    // whose last inject ended in a media/audio embedding chunk — see fork handler).
    std::map<int64_t, llama_token> last_tokens;
    // Sessions parked in host RAM (KV offloaded from VRAM). A session id is in
    // exactly one of `sessions` (live, VRAM) or `offloaded_sessions` (RAM).
    std::map<int64_t, OffloadedState> offloaded_sessions;
    std::atomic<int64_t> next_id{1};
    std::mutex generations_mu;
    std::map<std::string, std::shared_ptr<std::atomic<bool>>> active_generations;
};

struct GenerationRegistration {
    AppState & app;
    std::string sid;
    std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);

    GenerationRegistration(AppState & app, std::string sid) : app(app), sid(std::move(sid)) {
        std::lock_guard<std::mutex> lk(app.generations_mu);
        app.active_generations[this->sid] = cancelled;
    }

    bool finish(llama_context * ctx, llama_pos p_start, llama_token last_token,
                bool disconnected, double & rewind_s) {
        std::lock_guard<std::mutex> lk(app.generations_mu);
        const bool must_rewind = disconnected || cancelled->load(std::memory_order_relaxed);
        if (must_rewind) {
            const double rewind_start = now_s();
            llama_memory_t mem = llama_get_memory(ctx);
            llama_memory_seq_rm(mem, 0, p_start, -1);
            const llama_pos pmax = llama_memory_seq_pos_max(mem, 0);
            if (last_token != LLAMA_TOKEN_NULL && pmax >= 0) {
                llama_memory_seq_rm(mem, 0, pmax, pmax + 1);
                llama_batch b = llama_batch_init(/*n_tokens*/ 1, /*embd*/ 0, /*n_seq_max*/ 1);
                b.n_tokens     = 1;
                b.token[0]     = last_token;
                b.pos[0]       = pmax;
                b.n_seq_id[0]  = 1;
                b.seq_id[0][0] = 0;
                b.logits[0]    = 1;
                llama_decode(ctx, b);
                llama_batch_free(b);
            }
            rewind_s = now_s() - rewind_start;
        }
        const auto it = app.active_generations.find(sid);
        if (it != app.active_generations.end() && it->second == cancelled) {
            app.active_generations.erase(it);
        }
        return must_rewind;
    }

    ~GenerationRegistration() {
        std::lock_guard<std::mutex> lk(app.generations_mu);
        const auto it = app.active_generations.find(sid);
        if (it != app.active_generations.end() && it->second == cancelled) {
            app.active_generations.erase(it);
        }
    }
};

struct GenParams {
    int   max_tokens = 256;
    float temp       = 0.8f;
    float top_p      = 0.95f;
    int   top_k      = 40;
    float min_p      = 0.05f;
    int   seed       = -1;
    bool  ignore_eos = false;
    std::vector<std::string> stop;
    json response_format;          // {type: json_object|json_schema|text, ...}
    std::string grammar;           // raw GBNF (XOR response_format)
    json tools;                    // OpenAI tool schemas array
    std::string tool_choice = "auto";  // none | auto | required
    json sampling;                 // extra common_params_sampling overrides
};

struct GenResult {
    std::string text;
    std::vector<int64_t> ids;
    double gen_s = 0.0;
    double rewind_s = 0.0;
    bool cancelled = false;
    json tool_calls = json::array();
};

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Apply the optional `sampling` object (extra common_params_sampling fields:
// penalties, dry, mirostat, top_n_sigma, min_keep, n_probs, …) onto sparams.
// Only known scalar fields are mapped; unknown keys are ignored.
void apply_sampling_overrides(common_params_sampling & sparams, const json & s) {
    if (!s.is_object()) return;
    auto getf = [&](const char * k, float def) { return s.value(k, def); };
    auto geti = [&](const char * k, int   def) { return s.value(k, def); };
    if (s.contains("top_k"))       sparams.top_k       = geti("top_k", sparams.top_k);
    if (s.contains("top_p"))       sparams.top_p       = getf("top_p", sparams.top_p);
    if (s.contains("min_p"))       sparams.min_p       = getf("min_p", sparams.min_p);
    if (s.contains("top_n_sigma")) sparams.top_n_sigma = getf("top_n_sigma", sparams.top_n_sigma);
    if (s.contains("typical_p"))   sparams.typ_p       = getf("typical_p", sparams.typ_p);
    if (s.contains("xtc_probability")) sparams.xtc_probability = getf("xtc_probability", sparams.xtc_probability);
    if (s.contains("xtc_threshold"))   sparams.xtc_threshold   = getf("xtc_threshold", sparams.xtc_threshold);
    if (s.contains("min_keep"))    sparams.min_keep    = geti("min_keep", sparams.min_keep);
    if (s.contains("n_probs"))     sparams.n_probs     = geti("n_probs", sparams.n_probs);
    if (s.contains("penalty_last_n"))  sparams.penalty_last_n  = geti("penalty_last_n", sparams.penalty_last_n);
    if (s.contains("penalty_repeat"))  sparams.penalty_repeat  = getf("penalty_repeat", sparams.penalty_repeat);
    if (s.contains("penalty_freq"))    sparams.penalty_freq    = getf("penalty_freq", sparams.penalty_freq);
    if (s.contains("penalty_present")) sparams.penalty_present = getf("penalty_present", sparams.penalty_present);
    if (s.contains("dry_multiplier"))  sparams.dry_multiplier  = getf("dry_multiplier", sparams.dry_multiplier);
    if (s.contains("dry_base"))        sparams.dry_base        = getf("dry_base", sparams.dry_base);
    if (s.contains("dry_allowed_length")) sparams.dry_allowed_length = geti("dry_allowed_length", sparams.dry_allowed_length);
    if (s.contains("dry_penalty_last_n")) sparams.dry_penalty_last_n = geti("dry_penalty_last_n", sparams.dry_penalty_last_n);
    if (s.contains("mirostat"))     sparams.mirostat     = geti("mirostat", sparams.mirostat);
    if (s.contains("mirostat_tau")) sparams.mirostat_tau = getf("mirostat_tau", sparams.mirostat_tau);
    if (s.contains("mirostat_eta")) sparams.mirostat_eta = getf("mirostat_eta", sparams.mirostat_eta);
    if (s.contains("dynatemp_range"))   sparams.dynatemp_range  = getf("dynatemp_range", sparams.dynatemp_range);
    if (s.contains("dynatemp_exponent")) sparams.dynatemp_exponent = getf("dynatemp_exponent", sparams.dynatemp_exponent);
    if (s.contains("dry_sequence_breakers") && s["dry_sequence_breakers"].is_array()) {
        sparams.dry_sequence_breakers = s["dry_sequence_breakers"].get<std::vector<std::string>>();
    }
}

// Build a common_params_sampling from GenParams: maps the request-level decoding
// fields, applies the `sampling` overrides, and seeds the EOG logit-bias table
// (copied into the active bias set when ignore_eos is set). The grammar / tools
// derivation is done separately in run_generation (it needs the chat templates).
common_params_sampling build_sampling_params(const AppState & app, const GenParams & p) {
    common_params_sampling sparams;
    sparams.temp       = p.temp;
    sparams.top_p      = p.top_p;
    sparams.top_k      = p.top_k;
    sparams.min_p      = p.min_p;
    sparams.seed       = (p.seed < 0) ? LLAMA_DEFAULT_SEED : (uint32_t) p.seed;
    sparams.ignore_eos = p.ignore_eos;
    sparams.no_perf    = true;
    if (p.sampling.is_object()) apply_sampling_overrides(sparams, p.sampling);

    // Pre-compute the EOG logit-bias table once (-INFINITY on every EOG token),
    // mirroring common/common.cpp. When ignore_eos is set, fold it into the
    // active bias set so the logit-bias sampler suppresses EOG tokens entirely.
    if (sparams.logit_bias_eog.empty()) {
        const int n = llama_vocab_n_tokens(app.vocab);
        for (llama_token i = 0; i < n; ++i) {
            if (llama_vocab_is_eog(app.vocab, i)) {
                sparams.logit_bias_eog.push_back({i, -INFINITY});
            }
        }
    }
    if (p.ignore_eos) {
        sparams.logit_bias.insert(sparams.logit_bias.end(),
                                  sparams.logit_bias_eog.begin(),
                                  sparams.logit_bias_eog.end());
    }
    return sparams;
}

// Streaming stop-sequence buffering. Tokens accumulate in `buf`; this flushes
// the longest safe prefix to the SSE stream, holding back any tail that is a
// partial prefix of a stop sequence (so raw stop characters never leak). Returns
// true (and clears the matched part) if a FULL stop sequence is present, meaning
// generation should stop. `emit` writes one chunk to the stream.
struct stop_match { bool matched = false; };
stop_match flush_stream_buffer(std::string & buf, const std::vector<std::string> & stops,
                               const std::function<bool(const std::string &)> & emit) {
    // 1. Full stop present? Truncate the buffer at it and stop.
    for (const auto & s : stops) {
        if (s.empty()) continue;
        auto pos = buf.find(s);
        if (pos != std::string::npos) {
            if (pos > 0) emit(buf.substr(0, pos));
            buf.clear();
            return {true};
        }
    }
    // 2. Hold back the longest tail that is a proper prefix of some stop.
    size_t hold = 0;
    for (const auto & s : stops) {
        const size_t maxp = std::min(buf.size(), s.size() - 1);
        for (size_t l = maxp; l > hold; --l) {
            if (std::equal(s.begin(), s.begin() + (std::ptrdiff_t) l, buf.end() - (std::ptrdiff_t) l)) {
                hold = l;
                break;
            }
        }
    }
    const size_t emit_len = buf.size() - hold;
    if (emit_len > 0) emit(buf.substr(0, emit_len));
    buf.erase(0, emit_len);
    return {false};
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

// True iff session n is currently parked in host RAM (offloaded).
bool is_offloaded(AppState & app, int64_t n) {
    std::lock_guard<std::mutex> lk(app.mu);
    return app.offloaded_sessions.count(n) > 0;
}

// Resolve a live (VRAM) session by id. Sets res to 409 (offloaded) or 404
// (unknown) and returns nullptr when not live; otherwise returns the ctx. Used
// by inject/generate/fork so a parked (RAM) session cannot be silently used —
// the client must POST /sessions/{id}/load first.
llama_context * require_live_session(AppState & app, int64_t sid_num,
                                     httplib::Response & res) {
    llama_context * ctx = lookup_session(app, sid_num);
    if (!ctx) {
        if (is_offloaded(app, sid_num)) {
            res.status = 409;
            res.set_content(error_body("session is offloaded; POST /sessions/{id}/load first", 409).dump(),
                            "application/json");
        } else {
            res.status = 404;
            res.set_content(error_body("unknown session", 404).dump(), "application/json");
        }
        return nullptr;
    }
    return ctx;
}

// Restore a serialized seq-0 state into a fresh context and refresh logits by
// re-decoding last_token at its (freed) position. Shared by fork_context (state
// copied from a source ctx) and load (state from an offloaded RAM buffer).
// last_token == LLAMA_TOKEN_NULL means the state ended in an image/audio
// embedding chunk (no discrete token); KV/SSM is restored but logits are NOT
// refreshed — the caller must inject one text token before generating.
void restore_state_and_refresh(llama_context * dst, const uint8_t * data, size_t size,
                               llama_token last_token) {
    llama_state_seq_set_data(dst, data, size, /*dest_seq_id*/ 0);
    if (last_token != LLAMA_TOKEN_NULL) {
        llama_memory_t dmem = llama_get_memory(dst);
        const llama_pos pmax = llama_memory_seq_pos_max(dmem, 0);
        // Re-decoding at an already-cached position yields no logits row, so free
        // the last cell then re-evaluate last_token there: identical K/V + logits.
        llama_memory_seq_rm(dmem, 0, pmax, pmax + 1);
        llama_batch b = llama_batch_init(/*n_tokens*/ 1, /*embd*/ 0, /*n_seq_max*/ 1);
        b.n_tokens     = 1;
        b.token[0]     = last_token;
        b.pos[0]       = pmax;
        b.n_seq_id[0]  = 1;
        b.seq_id[0][0] = 0;
        b.logits[0]    = 1;  // request logits for this token
        llama_decode(dst, b);
        llama_batch_free(b);
    }
}

// Core fork copy (approach A', see docs/decisions/0004-fork-copy-semantics.md):
// deep-copy src's seq-0 KV into a fresh context and refresh logits by re-decoding
// src_last at its (freed) position. Returns the new context (caller owns) or
// nullptr on failure. Shared by POST /sessions/{id}/fork and the startup warm-up
// (so the first real fork isn't a cold CUDA-JIT hit).
//
// src_last == LLAMA_TOKEN_NULL means the source's last inject ended in an
// image/audio EMBEDDING chunk (no discrete token to re-decode). The KV is still
// copied, but logits are NOT refreshed — the forked session must inject one text
// token before it can generate (the chat protocol always closes a turn with text
// markers, so the normal audio flow ends in text and is fully forkable).
llama_context * fork_context(const AppState & app, llama_context * src, llama_token src_last) {
    llama_context * dst = create_session_ctx(app);
    if (!dst) return nullptr;
    const size_t seq_size = llama_state_seq_get_size(src, /*seq_id*/ 0);
    std::vector<uint8_t> buf(seq_size);
    llama_state_seq_get_data(src, buf.data(), seq_size, /*seq_id*/ 0);
    restore_state_and_refresh(dst, buf.data(), seq_size, src_last);
    return dst;
}

// Seed an empty session with BOS so there are logits to sample from.
bool seed_if_empty(const AppState & app, llama_context * ctx) {
    if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) >= 0) return true;
    llama_token bos = llama_vocab_bos(app.vocab);
    llama_batch b = llama_batch_get_one(&bos, 1);
    return llama_decode(ctx, b) == 0;
}

// Shared generation loop, now built on llama.cpp's `common_sampler` (ADR 0005).
//
// `on_event` receives full SSE-event JSON objects. When non-null (streaming),
// it is called with {"type":"token","token":...,"id":N} or
// {"type":"tool_call","tool_call":{...}} per produced delta and may return false
// to stop early (client disconnect). When null (non-streaming), everything is
// accumulated into GenResult. Both handlers go through this one path.
//
// Grammar / response_format / tools are derived here (they need the chat
// templates): tools → common_chat_templates_apply (grammar + lazy triggers +
// parser); response_format → json_schema_to_grammar; raw grammar → GBNF.
GenResult run_generation(
    const AppState & app, llama_context * ctx, const GenParams & p,
    std::function<bool(const json & event)> on_event,
    GenerationRegistration * generation = nullptr,
    llama_token rewind_token = LLAMA_TOKEN_NULL
) {
    GenResult r;
    if (!seed_if_empty(app, ctx)) return r;
    const llama_pos p_start = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;

    common_params_sampling sparams = build_sampling_params(app, p);

    bool tool_calling_active = false;
    common_chat_parser_params parser_params;

    // ---- derive grammar / parser / triggers from tools or response_format ----
    if (p.tools.is_array() && !p.tools.empty() && app.chat_templates) {
        // Tool-calling path (ADR 0006): parse the OpenAI tool defs, then let
        // common_chat_templates_apply derive the GBNF grammar, lazy triggers,
        // preserved tokens, and the per-format PEG parser — exactly as
        // llama-server does. A dummy user message is supplied only because the
        // template requires a non-empty message list; common_chat_templates_apply
        // derives grammar/triggers/parser solely from tools + tool_choice + the
        // template definition, NOT from message content, so "hello" is harmless.
        common_chat_templates_inputs inputs;
        inputs.use_jinja            = app.use_jinja;
        inputs.chat_template_kwargs = app.chat_template_kwargs;
        inputs.tools                = common_chat_tools_parse_oaicompat(p.tools);
        inputs.tool_choice          = common_chat_tool_choice_parse_oaicompat(p.tool_choice);
        inputs.add_generation_prompt = true;
        common_chat_msg dummy;
        dummy.role    = "user";
        dummy.content = "hello";
        inputs.messages.push_back(std::move(dummy));

        common_chat_params cp = common_chat_templates_apply(app.chat_templates.get(), inputs);
        if (!cp.grammar.empty()) {
            sparams.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, cp.grammar};
        }
        sparams.grammar_lazy = cp.grammar_lazy;
        // Preserved tokens + triggers: tokenize each (single-token results become
        // TOKEN triggers / preserved ids), mirroring the server's schema handler.
        for (const auto & s : cp.preserved_tokens) {
            auto ids = common_tokenize(app.vocab, s, false, true);
            if (ids.size() == 1) sparams.preserved_tokens.insert(ids[0]);
        }
        for (auto trig : cp.grammar_triggers) {
            if (trig.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                auto ids = common_tokenize(app.vocab, trig.value, false, true);
                if (ids.size() == 1) { trig.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN; trig.token = ids[0]; }
            }
            sparams.grammar_triggers.push_back(std::move(trig));
        }
        parser_params = common_chat_parser_params(cp);
        if (!cp.parser.empty()) parser_params.parser.load(cp.parser);
        tool_calling_active = true;
        // NOTE: generation_prompt is intentionally left empty. For models like
        // Gemma 4 the chat handler leaves cp.generation_prompt empty in the
        // normal (non-continuation) case and the grammar's optional `start`
        // rule absorbs the assistant turn marker that is already in the KV cache
        // (injected via /inject with add_generation_prompt=true). Pre-filling it
        // would wrongly advance the grammar past tokens the model must generate.
    } else if (p.response_format.is_object()) {
        // response_format → GBNF via json_schema_to_grammar (ADR 0007).
        std::string rf_type = p.response_format.value("type", std::string{});
        if (rf_type == "json_object") {
            json schema = p.response_format.value("schema", json::object());
            sparams.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(schema)};
        } else if (rf_type == "json_schema") {
            json schema = p.response_format.value("json_schema", json::object()).value("schema", json::object());
            sparams.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(schema)};
        }
        // type == "text" applies no constraint (no grammar set).
    } else if (!p.grammar.empty()) {
        sparams.grammar = {COMMON_GRAMMAR_TYPE_USER, p.grammar};
    }

    common_sampler * smpl = common_sampler_init(app.model, sparams);
    const int n_ctx = llama_n_ctx(ctx);
    const double t0 = now_s();
    bool keep_going = true;

    if (tool_calling_active) {
        // ---- tool-calling loop: parse the accumulated text each step ----
        std::string acc;                          // full accumulated generation
        common_chat_msg prev_msg;                 // previous parse (for diffing)
        std::vector<std::string> tc_ids_cache;    // stable ids across re-parses
        int tc_counter = 0;
        auto gen_tc_id = [&]() { return std::to_string(++tc_counter); };
        auto emit_diffs = [&](const std::vector<common_chat_msg_diff> & diffs) {
            for (const auto & d : diffs) {
                if (!keep_going) break;
                if (d.tool_call_index == std::string::npos) {
                    // content / reasoning delta → token event
                    if (!d.content_delta.empty() && on_event) {
                        keep_going = on_event(json{{"type", "token"}, {"token", d.content_delta}});
                        if (!keep_going) r.cancelled = true;
                    }
                    if (keep_going && !d.reasoning_content_delta.empty() && on_event) {
                        keep_going = on_event(json{{"type", "token"}, {"token", d.reasoning_content_delta}});
                        if (!keep_going) r.cancelled = true;
                    }
                } else if (on_event) {
                    json tc = {{"index", (int) d.tool_call_index}};
                    if (!d.tool_call_delta.id.empty())        tc["id"]        = std::string("fc_") + d.tool_call_delta.id;
                    if (!d.tool_call_delta.name.empty())      tc["name"]      = d.tool_call_delta.name;
                    if (!d.tool_call_delta.arguments.empty()) tc["arguments"] = d.tool_call_delta.arguments;
                    keep_going = on_event(json{{"type", "tool_call"}, {"tool_call", std::move(tc)}});
                    if (!keep_going) r.cancelled = true;
                }
            }
        };
        for (int step = 0; step < p.max_tokens && keep_going; ++step) {
            if (generation && generation->cancelled->load(std::memory_order_relaxed)) {
                r.cancelled = true;
                break;
            }
            const int used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (used + 1 > n_ctx) break;
            llama_token id = common_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(app.vocab, id)) break;
            std::string piece = common_token_to_piece(app.vocab, id, true);
            common_sampler_accept(smpl, id, true);
            r.ids.push_back((int64_t) id);
            acc += piece;
            r.text += piece;
            // TECH DEBT: re-parse the ENTIRE accumulated text on each step —
            // O(n²) over generation length. Incremental parsing is future work.
            auto new_msg = common_chat_parse(acc, /*is_partial*/ true, parser_params);
            if (!new_msg.empty()) {
                new_msg.set_tool_call_ids(tc_ids_cache, gen_tc_id);
                auto diffs = common_chat_msg_diff::compute_diffs(prev_msg, new_msg);
                prev_msg = new_msg;
                emit_diffs(diffs);
            }
            if (!keep_going) break;
            llama_batch batch = llama_batch_get_one(&id, 1);
            if (llama_decode(ctx, batch) != 0) break;
        }
        // Final non-partial parse: emit remaining diffs and extract tool_calls.
        auto final_msg = common_chat_parse(acc, /*is_partial*/ false, parser_params);
        if (!final_msg.empty()) {
            final_msg.set_tool_call_ids(tc_ids_cache, gen_tc_id);
            emit_diffs(common_chat_msg_diff::compute_diffs(prev_msg, final_msg));
            for (const auto & tc : final_msg.tool_calls) {
                r.tool_calls.push_back({
                    {"id", std::string("fc_") + tc.id},
                    {"type", "function"},
                    {"function", {{"name", tc.name}, {"arguments", tc.arguments}}},
                });
            }
        }
    } else {
        // ---- plain loop with streaming stop-sequence buffering ----
        std::string stream_buf;
        for (int step = 0; step < p.max_tokens && keep_going; ++step) {
            if (generation && generation->cancelled->load(std::memory_order_relaxed)) {
                r.cancelled = true;
                break;
            }
            const int used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (used + 1 > n_ctx) break;
            llama_token id = common_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(app.vocab, id)) break;
            std::string piece = common_token_to_piece(app.vocab, id, true);
            common_sampler_accept(smpl, id, true);
            r.ids.push_back((int64_t) id);
            r.text += piece;
            if (on_event) {
                stream_buf += piece;
                auto m = flush_stream_buffer(stream_buf, p.stop, [&](const std::string & chunk) {
                    bool ok = on_event(json{{"type", "token"}, {"token", chunk}, {"id", (int64_t) id}});
                    if (!ok) {
                        keep_going = false;
                        r.cancelled = true;
                    }
                    return ok;
                });
                if (m.matched) keep_going = false;
            }
            llama_batch batch = llama_batch_get_one(&id, 1);
            if (llama_decode(ctx, batch) != 0) break;
        }
        // Flush any held-back tail (a partial stop that never completed).
        if (on_event && !stream_buf.empty() && keep_going) {
            const bool delivered = on_event(json{{"type", "token"}, {"token", stream_buf},
                                                 {"id", r.ids.empty() ? (int64_t) 0 : r.ids.back()}});
            if (!delivered) {
                keep_going = false;
                r.cancelled = true;
            }
        }
        // Truncate the final text at the first stop sequence (non-streaming
        // result, and the authoritative text for streaming too).
        for (const auto & s : p.stop) {
            auto pos = r.text.find(s);
            if (pos != std::string::npos) { r.text = r.text.substr(0, pos); break; }
        }
    }

    if (generation) {
        r.cancelled = generation->finish(ctx, p_start, rewind_token, r.cancelled, r.rewind_s);
    }

    r.gen_s = now_s() - t0;
    common_sampler_free(smpl);
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
        else if (a == "--allow-cpu")     cfg.allow_cpu = true;
        else if (a == "--chat-template")      cfg.chat_template = next();
        else if (a == "--chat-template-file") cfg.chat_template = read_file_contents(next());
        else if (a == "--jinja")              cfg.use_jinja = true;
        else if (a == "--no-jinja")           cfg.use_jinja = false;
        else if (a == "--no-chat-template")   cfg.enable_chat_template = false;
        else if (a == "--system-prompt")      cfg.system_prompt = next();
        else if (a == "--config")             cfg.config_path = next();
        else if (a == "--chat-template-kwargs") {
            // Parse a JSON object string, e.g. '{"k":"v"}'. Each value is stored
            // as its JSON serialization (mirrors common/arg.cpp); re-parsed by
            // common_chat_templates_apply when rendering.
            std::string v = next();
            try {
                json parsed = json::parse(v);
                if (!parsed.is_object()) {
                    std::cerr << "error: --chat-template-kwargs must be a JSON object\n";
                    return 2;
                }
                for (auto it = parsed.begin(); it != parsed.end(); ++it) {
                    cfg.chat_template_kwargs[it.key()] = it.value().dump();
                }
            } catch (const std::exception & e) {
                std::cerr << "error: --chat-template-kwargs must be valid JSON: " << e.what() << "\n";
                return 2;
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
                "      --allow-cpu           explicitly permit CPU-only execution (default: GPU required)\n"
                "      --chat-template TPL   Jinja chat template override (else model default)\n"
                "      --chat-template-file F  read Jinja chat template override from a file\n"
                "      --jinja / --no-jinja  use the Jinja template engine (default: enabled)\n"
                "      --no-chat-template    disable templating: 'messages' inject is rejected\n"
                "      --system-prompt TEXT  system prompt prepended to every conversation\n"
                "      --chat-template-kwargs JSON  extra Jinja vars, e.g. '{\"k\":\"v\"}'\n"
                "      --config PATH         path to model config JSON file\n";
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

    int gpu_devices = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            ++gpu_devices;
        }
    }
    if (!cfg.allow_cpu && (gpu_devices == 0 || cfg.n_gpu_layers == 0)) {
        std::cerr << "error: " << gpu_execution_error("multimodal-server") << "\n";
        llama_backend_free();
        return 3;
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = cfg.n_gpu_layers;
    std::cerr << "loading model: " << cfg.model_path << " ...\n";
    AppState app;
    app.model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
    if (!app.model) {
        std::cerr << "error: failed to load model\n";
        if (!cfg.allow_cpu) std::cerr << "error: " << gpu_execution_error("multimodal-server") << "\n";
        llama_backend_free();
        return 1;
    }
    const int gpu_model_layers = (gpu_devices > 0 && cfg.n_gpu_layers != 0)
        ? (cfg.n_gpu_layers < 0
            ? llama_model_n_layer(app.model) + 1
            : std::min(cfg.n_gpu_layers, llama_model_n_layer(app.model) + 1))
        : 0;
    if (!gpu_execution_allowed(cfg.allow_cpu, gpu_model_layers)) {
        std::cerr << "error: " << gpu_execution_error("multimodal-server") << "\n";
        llama_model_free(app.model);
        llama_backend_free();
        return 3;
    }
    std::cerr << "execution: " << gpu_model_layers << " model layer(s) assigned to GPU"
              << (cfg.allow_cpu ? " (--allow-cpu enabled)" : "") << ".\n";
    app.model_cfg = load_config_for_model(cfg.model_path, cfg.config_path, app.model);
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
            GenParams gp;  // greedy warmup: temp=0 set explicitly below
            gp.max_tokens = 5; gp.temp = 0.0f; gp.seed = 0;
            run_generation(app, wctx, gp, nullptr);
            llama_free(wctx);
            std::cerr << "warmup done.\n";
        }
    }

    // Warm up the fork path (KV deep-copy + last-token re-decode) so the first
    // real fork doesn't pay the one-time CUDA graph/JIT cost (~hundreds of ms).
    {
        std::cerr << "warming up fork path...\n";
        if (llama_context * wsrc = create_session_ctx(app)) {
            llama_token bos = llama_vocab_bos(app.vocab);
            llama_batch b = llama_batch_get_one(&bos, 1);
            if (llama_decode(wsrc, b) == 0) {
                if (llama_context * wdst = fork_context(app, wsrc, bos)) llama_free(wdst);
            }
            llama_free(wsrc);
            std::cerr << "fork warmup done.\n";
        }
    }

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // ---- GET /info : capabilities (what the model + projector support) ----
    svr.Get("/info", [&](const httplib::Request &, httplib::Response &res) {
        // Reconcile the model-config-declared modalities with what is actually
        // loaded: a modality is available iff declared AND its projector is
        // loaded (text is always available). When the config declares nothing,
        // infer from the projector (backward compat). This is the source of truth
        // tests use to select by capability (e.g. skip audio tests on a
        // vision+text model).
        const bool v = app.mtmd_ctx && mtmd_support_vision(app.mtmd_ctx);
        const bool a = app.mtmd_ctx && mtmd_support_audio(app.mtmd_ctx);
        std::vector<std::string> in_mods;
        if (app.model_cfg.input_modalities.empty()) {
            in_mods = {"text"};
            if (v) in_mods.push_back("image");
            if (a) in_mods.push_back("audio");
        } else {
            for (const auto & m : app.model_cfg.input_modalities) {
                if (m == "text" || (m == "image" && v) || (m == "audio" && a)) in_mods.push_back(m);
            }
            if (in_mods.empty()) in_mods = {"text"};
        }
        std::vector<std::string> out_mods = app.model_cfg.output_modalities;
        if (out_mods.empty()) out_mods = {"text"};
        json in_arr = json::array(), out_arr = json::array();
        for (const auto & m : in_mods)  in_arr.push_back(m);
        for (const auto & m : out_mods) out_arr.push_back(m);
        json body = {
            {"model_loaded", app.model != nullptr},
            {"supports_vision", std::find(in_mods.begin(), in_mods.end(), "image") != in_mods.end()},
            {"supports_audio",  std::find(in_mods.begin(), in_mods.end(), "audio")  != in_mods.end()},
            {"audio_sample_rate", app.mtmd_ctx ? mtmd_get_audio_sample_rate(app.mtmd_ctx) : 0},
            {"input_modalities", in_arr},
            {"output_modalities", out_arr},
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
        const int64_t sid_num = parse_session_id_num(sid);
        llama_context * src = require_live_session(app, sid_num, res);
        if (!src) return;

        const double t0 = now_s();
        const llama_token src_last = get_last_token(app, parse_session_id_num(sid));
        llama_context * dst = fork_context(app, src, src_last);
        if (!dst) {
            res.status = 500;
            res.set_content(error_body("failed to create forked context", 500).dump(), "application/json");
            return;
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
        llama_context * ctx = require_live_session(app, sid_num, res);
        if (!ctx) return;
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
                // Optional tools/tool_choice: render the tool instructions into the
                // prompt at inject time (ADR 0006), so the cached conversation
                // already carries the tool contract for the following /generate.
                if (j.contains("tools") && j["tools"].is_array() && !j["tools"].empty()) {
                    inputs.tools = common_chat_tools_parse_oaicompat(j["tools"]);
                    inputs.tool_choice =
                        common_chat_tool_choice_parse_oaicompat(j.value("tool_choice", std::string("auto")));
                }
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
                    // last inject ended in an image/audio embedding chunk: no
                    // discrete last token, so clear any stale value (a fork must
                    // not re-decode a wrong token at a media cell — see ADR 0004).
                    set_last_token(app, sid_num, LLAMA_TOKEN_NULL);
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
                if (n_samples % app.model_cfg.audio_frame_size != 0) {
                    res.status = 400;
                    std::ostringstream os;
                    os << "audio sample count must be a multiple of " << app.model_cfg.audio_frame_size << " (model audio frame size)";
                    res.set_content(error_body(os.str(), 400).dump(), "application/json");
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
                // streaming-audio inject ends in an audio embedding chunk: clear
                // any stale last token so a later fork doesn't re-decode it at the
                // wrong (media) position. The chat protocol follows audio with a
                // text turn-close, which sets a valid last token again.
                set_last_token(app, sid_num, LLAMA_TOKEN_NULL);
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

    // ---- POST /sessions/{id}/cancel : signal an active generation ----
    svr.Post(R"(/sessions/[^/]+/cancel)", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string sid = extract_session_id(req.path, "cancel");
        const int64_t sid_num = parse_session_id_num(sid);
        if (!lookup_session(app, sid_num)) {
            if (is_offloaded(app, sid_num)) {
                // No active generation on a parked (RAM) session; cancel is a no-op.
                res.set_content(json{{"session_id", sid}, {"cancelled", false}}.dump(), "application/json");
                return;
            }
            res.status = 404;
            res.set_content(error_body("unknown session", 404).dump(), "application/json");
            return;
        }
        bool cancelled = false;
        {
            std::lock_guard<std::mutex> lk(app.generations_mu);
            const auto it = app.active_generations.find(sid);
            if (it != app.active_generations.end()) {
                it->second->store(true, std::memory_order_relaxed);
                cancelled = true;
            }
        }
        res.set_content(json{{"session_id", sid}, {"cancelled", cancelled}}.dump(), "application/json");
    });

    // ---- POST /sessions/{id}/generate : streaming (SSE) or JSON ----
    svr.Post(R"(/sessions/[^/]+/generate)", [&](const httplib::Request &req, httplib::Response &res) {
        std::string sid = extract_session_id(req.path, "generate");
        const int64_t sid_num = parse_session_id_num(sid);
        llama_context * ctx = require_live_session(app, sid_num, res);
        if (!ctx) return;
        GenParams gp;
        bool stream = false;
        try {
            auto j = json::parse(req.body);
            gp.max_tokens = j.value("max_tokens", 256);
            gp.temp       = j.value("temperature", app.model_cfg.temperature);
            gp.top_p      = j.value("top_p", app.model_cfg.top_p);
            gp.top_k      = j.value("top_k", app.model_cfg.top_k);
            gp.min_p      = j.value("min_p", app.model_cfg.min_p);
            gp.seed       = j.value("seed", -1);
            gp.ignore_eos = j.value("ignore_eos", false);
            stream        = j.value("stream", false);
            // stop: accept an array or a single string.
            if (j.contains("stop")) {
                if (j["stop"].is_array()) {
                    for (const auto & s : j["stop"]) gp.stop.push_back(s.get<std::string>());
                } else if (j["stop"].is_string()) {
                    gp.stop.push_back(j["stop"].get<std::string>());
                }
            }
            if (j.contains("grammar"))         gp.grammar         = j["grammar"].get<std::string>();
            if (j.contains("response_format")) gp.response_format = j["response_format"];
            if (j.contains("tools"))           gp.tools           = j["tools"];
            if (j.contains("tool_choice"))     gp.tool_choice     = j["tool_choice"].get<std::string>();
            if (j.contains("sampling"))        gp.sampling        = j["sampling"];
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_body(std::string("bad request: ") + e.what(), 400).dump(), "application/json");
            return;
        }

        // response_format XOR grammar (both → 400). tools take precedence over
        // both (the chat template derives its own grammar), so the conflict only
        // applies when tools are absent.
        const bool has_rf = gp.response_format.is_object();
        const bool has_gr = !gp.grammar.empty();
        const bool has_tools = gp.tools.is_array() && !gp.tools.empty();
        if (!has_tools && has_rf && has_gr) {
            res.status = 400;
            res.set_content(error_body("cannot specify both response_format and grammar", 400).dump(), "application/json");
            return;
        }
        if (has_rf) {
            std::string rf_type = gp.response_format.value("type", std::string{});
            if (rf_type != "json_object" && rf_type != "json_schema" && rf_type != "text") {
                res.status = 400;
                res.set_content(error_body("invalid response_format.type (expected json_object, json_schema, or text)", 400).dump(), "application/json");
                return;
            }
        }
        const bool was_empty = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;
        if (!seed_if_empty(app, ctx)) {
            res.status = 500;
            res.set_content(error_body("failed to seed generation", 500).dump(), "application/json");
            return;
        }
        llama_token pre_generation_last = get_last_token(app, sid_num);
        if (was_empty && pre_generation_last == LLAMA_TOKEN_NULL) {
            pre_generation_last = llama_vocab_bos(app.vocab);
        }
        auto generation = std::make_shared<GenerationRegistration>(app, sid);


        if (!stream) {
            // Non-streaming: whole response as JSON (incl. tool_calls if any).
            GenResult r = run_generation(app, ctx, gp, nullptr, generation.get(), pre_generation_last);
            if (!r.cancelled && !r.ids.empty()) set_last_token(app, sid_num, (llama_token) r.ids.back());
            if (r.cancelled) {
                std::cerr << "generation cancelled for " << sid << "; rewind=" << r.rewind_s * 1000.0 << " ms\n";
            }
            const double tok_s = (r.gen_s > 0) ? (r.ids.size() / r.gen_s) : 0.0;
            json body = {
                {"session_id", sid},
                {"text", r.text},
                {"tokens", r.ids},
                {"n_tokens", r.ids.size()},
                {"gen_ms", (int)(r.gen_s * 1000)},
                {"tokens_per_s", tok_s},
            };
            if (!r.tool_calls.empty()) body["tool_calls"] = r.tool_calls;
            res.set_content(body.dump(), "application/json");
            return;
        }

        // Streaming: text/event-stream. Generate tokens directly inside the
        // chunked provider callback — httplib calls it on its worker thread, and
        // each event is written to the DataSink the instant it is produced.
        // No separate thread, no lifetime issues. The provider blocks httplib's
        // worker until generation finishes, which is fine (one request at a time).
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        const auto is_connection_closed = req.is_connection_closed;
        res.set_chunked_content_provider(
            "text/event-stream",
            [&app, ctx, gp, sid, sid_num, generation, pre_generation_last, is_connection_closed](size_t, httplib::DataSink & ds) -> bool {
                GenResult r = run_generation(app, ctx, gp, [&ds, &is_connection_closed](const json & event) {
                    if (is_connection_closed()) return false;
                    std::string ev = sse_event(event);
                    return ds.write(ev.data(), ev.size());
                }, generation.get(), pre_generation_last);
                if (!r.cancelled && !r.ids.empty()) set_last_token(app, sid_num, (llama_token) r.ids.back());
                if (r.cancelled) {
                    std::cerr << "generation cancelled for " << sid << "; rewind=" << r.rewind_s * 1000.0 << " ms\n";
                }
                const double tok_s = (r.gen_s > 0) ? (r.ids.size() / r.gen_s) : 0.0;
                json done_body = {
                    {"type", "done"},
                    {"n_tokens", r.ids.size()},
                    {"gen_ms", (int)(r.gen_s * 1000)},
                    {"tokens_per_s", tok_s},
                };
                if (!r.tool_calls.empty()) done_body["tool_calls"] = r.tool_calls;
                std::string done = sse_event(done_body);
                ds.write(done.data(), done.size());
                ds.done();
                return true;
            }
        );
    });

    // ---- GET /sessions/usage : aggregate VRAM/RAM footprint (EUS-6) ----
    // MUST be registered before GET /sessions/{id} so the literal "usage" is not
    // shadowed by the {id} regex (cpp-httplib matches in registration order).
    svr.Get("/sessions/usage", [&](const httplib::Request &, httplib::Response &res) {
        json vram_sessions = json::array();
        size_t vram_total = 0;
        json ram_sessions  = json::array();
        size_t ram_total  = 0;
        {
            std::lock_guard<std::mutex> lk(app.mu);
            for (const auto & [n, ctx] : app.sessions) {
                const size_t sb = llama_state_seq_get_size(ctx, /*seq_id*/ 0);
                vram_sessions.push_back({{"session_id", make_session_id(n)}, {"state_bytes", sb}});
                vram_total += sb;
            }
            for (const auto & [n, st] : app.offloaded_sessions) {
                ram_sessions.push_back({{"session_id", make_session_id(n)}, {"state_bytes", st.state_bytes}});
                ram_total += st.state_bytes;
            }
        }
        json body = {
            {"vram", {{"n_sessions", vram_sessions.size()}, {"total_state_bytes", vram_total}, {"sessions", vram_sessions}}},
            {"ram",  {{"n_sessions", ram_sessions.size()},  {"total_state_bytes", ram_total},  {"sessions", ram_sessions}}},
        };
        res.set_content(body.dump(), "application/json");
    });

    // ---- GET /sessions/{id} : session status (location + footprint) (EUS-6) ----
    svr.Get(R"(/sessions/[^/]+)", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string prefix = "/sessions/";
        if (req.path.rfind(prefix, 0) != 0 || req.path.find('/', prefix.size()) != std::string::npos) {
            res.status = 404;
            res.set_content(error_body("not found", 404).dump(), "application/json");
            return;
        }
        const int64_t sid_num = parse_session_id_num(req.path.substr(prefix.size()));
        std::lock_guard<std::mutex> lk(app.mu);
        auto it = (sid_num > 0) ? app.sessions.find(sid_num) : app.sessions.end();
        if (it != app.sessions.end()) {
            const int cache_size = llama_memory_seq_pos_max(llama_get_memory(it->second), 0) + 1;
            const size_t state_bytes = llama_state_seq_get_size(it->second, /*seq_id*/ 0);
            json body = {
                {"session_id", make_session_id(sid_num)}, {"location", "vram"},
                {"cache_size", cache_size}, {"state_bytes", state_bytes},
            };
            res.set_content(body.dump(), "application/json");
            return;
        }
        auto oit = app.offloaded_sessions.find(sid_num);
        if (oit != app.offloaded_sessions.end()) {
            json body = {
                {"session_id", make_session_id(sid_num)}, {"location", "ram"},
                {"cache_size", oit->second.cache_size}, {"state_bytes", oit->second.state_bytes},
            };
            res.set_content(body.dump(), "application/json");
            return;
        }
        res.status = 404;
        res.set_content(error_body("unknown session", 404).dump(), "application/json");
    });

    // ---- POST /sessions/{id}/offload : move KV state VRAM -> host RAM (EUS-6) ----
    // Serialize the session's seq-0 state to a host buffer (the GPU->host copy is
    // done OUTSIDE app.mu so it cannot serialize another session's decode loop),
    // then free the live context under app.mu (that is what actually releases
    // VRAM) and park the bytes in offloaded_sessions. Idempotent on a RAM session.
    svr.Post(R"(/sessions/[^/]+/offload)", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string sid = extract_session_id(req.path, "offload");
        const int64_t sid_num = parse_session_id_num(sid);
        // Refuse to offload a session with an active generation (use-after-free guard).
        {
            std::lock_guard<std::mutex> lk(app.generations_mu);
            if (app.active_generations.count(sid)) {
                res.status = 409;
                res.set_content(error_body("session has an active generation; cancel or wait for it to finish before offloading", 409).dump(),
                                "application/json");
                return;
            }
        }
        llama_context * ctx = nullptr;
        llama_token last = LLAMA_TOKEN_NULL;
        {
            std::lock_guard<std::mutex> lk(app.mu);
            auto oit = app.offloaded_sessions.find(sid_num);
            if (oit != app.offloaded_sessions.end()) {
                // idempotent: already RAM
                res.set_content(json{{"session_id", sid}, {"location", "ram"},
                                     {"cache_size", oit->second.cache_size},
                                     {"state_bytes", oit->second.state_bytes},
                                     {"offload_ms", 0}}.dump(), "application/json");
                return;
            }
            auto it = (sid_num > 0) ? app.sessions.find(sid_num) : app.sessions.end();
            if (it == app.sessions.end()) {
                res.status = 404;
                res.set_content(error_body("unknown session", 404).dump(), "application/json");
                return;
            }
            ctx = it->second;
            auto lt = app.last_tokens.find(sid_num);
            if (lt != app.last_tokens.end()) last = lt->second;
        }
        // Serialize lock-free (this ctx is exclusively ours for this session).
        // Per-sequence state (KV +, for hybrid models, the SSM recurrent state)
        // — the same primitive fork uses (ADR 0004 A'). state_bytes is symmetric
        // across offload/load and is what /usage sums.
        const double t0 = now_s();
        const size_t seq_size = llama_state_seq_get_size(ctx, /*seq_id*/ 0);
        std::vector<uint8_t> buf(seq_size);
        llama_state_seq_get_data(ctx, buf.data(), seq_size, /*seq_id*/ 0);
        const int cache_size = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        {
            std::lock_guard<std::mutex> lk(app.mu);
            auto it = app.sessions.find(sid_num);
            if (it != app.sessions.end()) {
                llama_free(it->second);
                app.sessions.erase(it);
            }
            app.last_tokens.erase(sid_num);
            OffloadedState st;
            st.state = std::move(buf);
            st.last_token = last;
            st.cache_size = cache_size;
            st.state_bytes = seq_size;
            app.offloaded_sessions[sid_num] = std::move(st);
        }
        const double dt = (now_s() - t0) * 1000.0;
        res.set_content(json{{"session_id", sid}, {"location", "ram"},
                             {"cache_size", cache_size}, {"state_bytes", seq_size},
                             {"offload_ms", (int) dt}}.dump(), "application/json");
    });

    // ---- POST /sessions/{id}/load : restore KV state host RAM -> VRAM (EUS-6) ----
    // Recreate a fresh context, restore the seq-0 state + refresh logits, and put
    // the session back in `sessions`. Idempotent on a VRAM session.
    svr.Post(R"(/sessions/[^/]+/load)", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string sid = extract_session_id(req.path, "load");
        const int64_t sid_num = parse_session_id_num(sid);
        const double t0 = now_s();
        OffloadedState st;
        bool was_offloaded = false;
        {
            std::lock_guard<std::mutex> lk(app.mu);
            if ((sid_num > 0) && app.sessions.count(sid_num)) {
                // idempotent: already VRAM — report current footprint.
                llama_context * ctx = app.sessions[sid_num];
                const int cache_size = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
                const size_t state_bytes = llama_state_seq_get_size(ctx, /*seq_id*/ 0);
                res.set_content(json{{"session_id", sid}, {"location", "vram"},
                                     {"cache_size", cache_size}, {"state_bytes", state_bytes},
                                     {"load_ms", 0}}.dump(), "application/json");
                return;
            }
            auto oit = app.offloaded_sessions.find(sid_num);
            if (oit == app.offloaded_sessions.end()) {
                res.status = 404;
                res.set_content(error_body("unknown session", 404).dump(), "application/json");
                return;
            }
            st = std::move(oit->second);
            app.offloaded_sessions.erase(oit);
            was_offloaded = true;
        }
        llama_context * ctx = create_session_ctx(app);
        if (!ctx) {
            // Re-park so the session (and its state) is not lost.
            std::lock_guard<std::mutex> lk(app.mu);
            app.offloaded_sessions[sid_num] = std::move(st);
            res.status = 500;
            res.set_content(error_body("failed to create context", 500).dump(), "application/json");
            return;
        }
        // Restore the per-sequence state and refresh logits by re-decoding the
        // last token (the restore carries KV/SSM but NOT the logits buffer) —
        // the same mechanism fork uses (ADR 0004 A'). CAVEAT: the redecode frees
        // the last cell then re-evaluates there, which requires seq_rm to rewind
        // the last position; that holds for pure-attention models (e.g. Gemma 4)
        // but NOT hybrid SSM/M-RoPE models (e.g. MiniCPM-V-4.6 / Qwen3.5), where
        // seq_rm can't rewind the recurrent state, so generate-after-restore is
        // unsupported on hybrids today (a fork-level limitation; see worklog).
        restore_state_and_refresh(ctx, st.state.data(), st.state_bytes, st.last_token);
        {
            std::lock_guard<std::mutex> lk(app.mu);
            app.sessions[sid_num] = ctx;
            app.last_tokens[sid_num] = st.last_token;
        }
        const double dt = (now_s() - t0) * 1000.0;
        res.set_content(json{{"session_id", sid}, {"location", "vram"},
                             {"cache_size", st.cache_size}, {"state_bytes", st.state_bytes},
                             {"load_ms", (int) dt}}.dump(), "application/json");
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
            if (it != app.sessions.end()) {
                llama_free(it->second);
                app.sessions.erase(it);
                app.last_tokens.erase(n);
            } else {
                auto oit = app.offloaded_sessions.find(n);
                if (oit != app.offloaded_sessions.end()) {
                    app.offloaded_sessions.erase(oit);  // frees the host-RAM buffer
                } else {
                    res.status = 404;
                    res.set_content(error_body("unknown session", 404).dump(), "application/json");
                    return;
                }
            }
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
