// multimodal-server — slice 2: session-oriented HTTP LLM API with streaming.
//
// Endpoints (see docs/ipc-protocol.md):
//   POST   /sessions              -> create a session, returns {session_id}
//   POST   /sessions/{id}/inject  -> body {text}; tokenize + decode into KV cache (no gen)
//   POST   /sessions/{id}/generate-> body {max_tokens, stream, ...}
//       stream=false (default): returns slice-1 JSON {text, tokens, ...}
//       stream=true : SSE token/tool-call events, then exactly one terminal done/error event
//   DELETE /sessions/{id}         -> free the session
//   GET    /health                -> liveness
//
// All MultiModalAgent engine code stays under engine/multimodal/; upstream
// llama.cpp remains untouched. One model and one pooled context are shared;
// external sessions map to scheduler-owned sequence IDs.

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

#include "inference_scheduler.h"
#include "server_cli.h"
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


double now_s();

// A session parked in host RAM (offloaded from the pooled context). Its internal
// sequence has been released, reclaiming a live-sequence slot and logical KV
// capacity; the process-wide pooled buffers remain allocated. `state` holds the
// serialized target-sequence state so load can restore it into a newly reserved
// pooled sequence. This includes KV plus, for hybrid models such as
// Qwen3.5/MiniCPM-V-4.6, the SSM/Mamba recurrent state.
struct OffloadedState {
    std::vector<uint8_t> state;                       // llama_state_seq_get_data output
    llama_token          last_token = LLAMA_TOKEN_NULL; // for logits refresh on load
    int                  cache_size = 0;              // tokens at offload time
    size_t               state_bytes = 0;             // == state.size(); symmetric offload/load
    bool                 loading = false;             // RAM -> loading -> VRAM reservation
};

struct LiveSession {
    llama_seq_id seq_id = -1;
    llama_token last_token = LLAMA_TOKEN_NULL;
    bool busy = false;
    uint32_t readers = 0;
    bool releasing = false;
};

// One model and one explicitly sized pooled context. External sessions map to
// reusable internal sequence IDs owned exclusively by InferenceScheduler.
struct AppState {
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr chat_templates;  // built from the model; applies its chat template
    mtmd_context * mtmd_ctx = nullptr;         // multimodal projector (may be null)
    bool supports_vision = false;
    bool supports_audio = false;
    int audio_sample_rate = 0;
    std::string media_marker;
    int n_ctx_per_session = 4096;
    // Chat-template options (copied from ServerConfig at startup) — passed to
    // common_chat_templates_apply so our rendering matches llama-server.
    bool                                 use_jinja            = true;
    bool                                 enable_chat_template = true;
    std::map<std::string, std::string>   chat_template_kwargs;
    std::string                          system_prompt;
    ModelConfig                          model_cfg;
    int max_sequences = 8;
    size_t pool_footprint_bytes = 0;
    size_t model_gpu_bytes = 0;
    std::unique_ptr<InferenceScheduler> scheduler;
    std::mutex mu;
    std::condition_variable sessions_cv;
    std::map<int64_t, LiveSession> sessions;
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

    bool finish(llama_seq_id seq_id, llama_pos p_start, llama_token last_token,
                uint64_t generation_start_family, bool disconnected, double & rewind_s) {
        std::lock_guard<std::mutex> lk(app.generations_mu);
        const bool must_rewind = disconnected || cancelled->load(std::memory_order_relaxed);
        if (must_rewind) {
            const double rewind_start = now_s();
            app.scheduler->rewind(
                seq_id, p_start, last_token, generation_start_family);
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
    std::string error;
    int cache_size = 0;
    json tool_calls = json::array();
};

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

size_t gpu_free_bytes() {
    size_t result = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        result += free;
    }
    return result;
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


int64_t register_session(AppState & app, llama_seq_id seq_id,
                         llama_token last_token = LLAMA_TOKEN_NULL) {
    const int64_t n = app.next_id.fetch_add(1);
    std::lock_guard<std::mutex> lk(app.mu);
    app.sessions[n] = LiveSession{seq_id, last_token, false};
    return n;
}

void set_last_token(AppState & app, int64_t n, llama_token token) {
    std::lock_guard<std::mutex> lk(app.mu);
    auto it = app.sessions.find(n);
    if (it != app.sessions.end()) it->second.last_token = token;
}

std::optional<LiveSession> reserve_live_session(
    AppState & app, int64_t sid_num, httplib::Response & res,
    bool will_release_sequence = false) {
    std::lock_guard<std::mutex> lk(app.mu);
    auto live = (sid_num > 0) ? app.sessions.find(sid_num) : app.sessions.end();
    if (live == app.sessions.end()) {
        if (app.offloaded_sessions.count(sid_num) > 0) {
            res.status = 409;
            res.set_content(error_body(
                "session is offloaded; POST /sessions/{id}/load first", 409).dump(),
                "application/json");
        } else {
            res.status = 404;
            res.set_content(error_body("unknown session", 404).dump(), "application/json");
        }
        return std::nullopt;
    }
    if (live->second.busy || live->second.readers > 0) {
        res.status = 409;
        res.set_content(error_body("session already has an active operation", 409).dump(),
                        "application/json");
        return std::nullopt;
    }
    live->second.busy = true;
    live->second.releasing = will_release_sequence;
    return live->second;
}

void clear_session_busy(AppState & app, int64_t n) {
    {
        std::lock_guard<std::mutex> lk(app.mu);
        auto it = app.sessions.find(n);
        if (it != app.sessions.end()) {
            it->second.busy = false;
            it->second.releasing = false;
        }
    }
    app.sessions_cv.notify_all();
}

void release_live_reader(AppState & app, int64_t n) {
    {
        std::lock_guard<std::mutex> lk(app.mu);
        auto it = app.sessions.find(n);
        if (it != app.sessions.end() && it->second.readers > 0) --it->second.readers;
    }
    app.sessions_cv.notify_all();
}

struct BusyGuard {
    AppState & app;
    int64_t session;
    bool active;
    BusyGuard(AppState & app, int64_t session, bool active = true)
        : app(app), session(session), active(active) {}
    BusyGuard(const BusyGuard &) = delete;
    BusyGuard & operator=(const BusyGuard &) = delete;
    ~BusyGuard() { if (active) clear_session_busy(app, session); }
};

struct LiveReadGuard {
    AppState & app;
    int64_t session;
    LiveReadGuard(AppState & app, int64_t session) : app(app), session(session) {}
    LiveReadGuard(const LiveReadGuard &) = delete;
    LiveReadGuard & operator=(const LiveReadGuard &) = delete;
    ~LiveReadGuard() { release_live_reader(app, session); }
};

struct LiveReadersGuard {
    AppState & app;
    std::vector<int64_t> sessions;
    LiveReadersGuard(AppState & app, std::vector<int64_t> sessions)
        : app(app), sessions(std::move(sessions)) {}
    LiveReadersGuard(const LiveReadersGuard &) = delete;
    LiveReadersGuard & operator=(const LiveReadersGuard &) = delete;
    ~LiveReadersGuard() {
        {
            std::lock_guard<std::mutex> lk(app.mu);
            for (int64_t session : sessions) {
                auto it = app.sessions.find(session);
                if (it != app.sessions.end() && it->second.readers > 0) --it->second.readers;
            }
        }
        app.sessions_cv.notify_all();
    }
};

struct SchedulerGenerationGuard {
    InferenceScheduler & scheduler;
    llama_seq_id seq_id;
    SchedulerGenerationCheckpoint checkpoint;
    bool checkpoint_registered = false;

    SchedulerGenerationGuard(InferenceScheduler & scheduler, llama_seq_id seq_id)
        : scheduler(scheduler), seq_id(seq_id) {
        scheduler.begin_generation();
        try {
            checkpoint = scheduler.generation_checkpoint(seq_id);
            checkpoint_registered = true;
        } catch (...) {
            scheduler.end_generation();
            throw;
        }
    }
    SchedulerGenerationGuard(const SchedulerGenerationGuard &) = delete;
    SchedulerGenerationGuard & operator=(const SchedulerGenerationGuard &) = delete;
    ~SchedulerGenerationGuard() {
        if (checkpoint_registered) {
            try {
                scheduler.finish_generation(seq_id);
            } catch (const std::exception & e) {
                std::cerr << "failed to release generation checkpoint: "
                          << e.what() << "\n";
            } catch (...) {
                std::cerr << "failed to release generation checkpoint\n";
            }
        }
        scheduler.end_generation();
    }
};

struct GenerationRegistrationGuard {
    GenerationRegistration * registration;
    llama_seq_id seq_id;
    llama_pos start_position;
    llama_token rewind_token;
    uint64_t checkpoint_family;
    bool finished = false;

    ~GenerationRegistrationGuard() {
        if (!registration || finished) return;
        double rewind_s = 0.0;
        try {
            registration->finish(
                seq_id, start_position, rewind_token, checkpoint_family,
                /*disconnected*/ true, rewind_s);
        } catch (const std::exception & e) {
            std::cerr << "failed to rewind exceptional generation: "
                      << e.what() << "\n";
        } catch (...) {
            std::cerr << "failed to rewind exceptional generation\n";
        }
    }

    void finish(GenResult & result) {
        if (registration) {
            result.cancelled = registration->finish(
                seq_id, start_position, rewind_token, checkpoint_family,
                result.cancelled, result.rewind_s);
        }
        finished = true;
    }
};

struct PreparedGeneration {
    common_sampler_ptr sampler;
    common_chat_parser_params parser_params;
    bool tool_calling_active = false;
    uint8_t canonical_greedy_policy = 0;
};

PreparedGeneration prepare_generation(AppState & app, const GenParams & p) {
    PreparedGeneration prepared;
    common_params_sampling sparams = build_sampling_params(app, p);

    if (p.tools.is_array() && !p.tools.empty() && app.chat_templates) {
        // Tool-calling path (ADR 0006): derive the grammar, lazy triggers,
        // preserved tokens, and PEG parser exactly as llama-server does.
        common_chat_templates_inputs inputs;
        inputs.use_jinja              = app.use_jinja;
        inputs.chat_template_kwargs   = app.chat_template_kwargs;
        inputs.tools                  = common_chat_tools_parse_oaicompat(p.tools);
        inputs.tool_choice            = common_chat_tool_choice_parse_oaicompat(p.tool_choice);
        inputs.add_generation_prompt  = true;
        common_chat_msg dummy;
        dummy.role    = "user";
        dummy.content = "hello";
        inputs.messages.push_back(std::move(dummy));

        common_chat_params cp =
            common_chat_templates_apply(app.chat_templates.get(), inputs);
        if (!cp.grammar.empty()) {
            sparams.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, cp.grammar};
        }
        sparams.grammar_lazy = cp.grammar_lazy;
        for (const auto & token_text : cp.preserved_tokens) {
            auto ids = common_tokenize(app.vocab, token_text, false, true);
            if (ids.size() == 1) sparams.preserved_tokens.insert(ids[0]);
        }
        for (auto trigger : cp.grammar_triggers) {
            if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                auto ids = common_tokenize(app.vocab, trigger.value, false, true);
                if (ids.size() == 1) {
                    trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                    trigger.token = ids[0];
                }
            }
            sparams.grammar_triggers.push_back(std::move(trigger));
        }
        prepared.parser_params = common_chat_parser_params(cp);
        if (!cp.parser.empty()) prepared.parser_params.parser.load(cp.parser);
        prepared.tool_calling_active = true;
        // generation_prompt intentionally stays empty: the injected assistant
        // turn marker is consumed by the grammar's optional start rule.
    } else if (p.response_format.is_object()) {
        const std::string rf_type = p.response_format.at("type").get<std::string>();
        if (rf_type == "json_object") {
            const json schema = p.response_format.value("schema", json::object());
            sparams.grammar = {
                COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
                json_schema_to_grammar(schema),
            };
        } else if (rf_type == "json_schema") {
            const json & schema = p.response_format.at("json_schema").at("schema");
            sparams.grammar = {
                COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
                json_schema_to_grammar(schema),
            };
        }
    } else if (!p.grammar.empty()) {
        sparams.grammar = {COMMON_GRAMMAR_TYPE_USER, p.grammar};
    }

    const std::string & grammar = common_grammar_value(sparams.grammar);
    if (!grammar.empty()) {
        // common_sampler_init throws after allocating its sampler chain when
        // grammar parsing fails. Parse once with an RAII-owned probe first so
        // malformed client input cannot leak constructor state or reach SSE.
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> grammar_probe{
            llama_sampler_init_grammar(app.vocab, grammar.c_str(), "root"),
            llama_sampler_free,
        };
        if (!grammar_probe) {
            const std::string source =
                p.response_format.is_object() ? "response_format" : "grammar";
            throw std::invalid_argument("invalid " + source + ": failed to parse grammar");
        }
    }

    prepared.sampler.reset(common_sampler_init(app.model, sparams));
    if (!prepared.sampler) {
        throw std::runtime_error("failed to initialize sampler");
    }
    const bool canonical_greedy =
        p.temp <= 0.0f && p.grammar.empty() && p.response_format.is_null() &&
        (!p.tools.is_array() || p.tools.empty()) &&
        (!p.sampling.is_object() || p.sampling.empty());
    prepared.canonical_greedy_policy =
        canonical_greedy ? (p.ignore_eos ? 2 : 1) : 0;
    return prepared;
}

// Shared generation loop, now built on llama.cpp's `common_sampler` (ADR 0005).
//
// `on_event` receives full SSE-event JSON objects. When non-null (streaming),
// it is called with {"type":"token","token":...,"id":N} or
// {"type":"tool_call","tool_call":{...}} per produced delta and may return false
// to stop early (client disconnect). When null (non-streaming), everything is
// accumulated into GenResult. Both handlers go through this one path.
//
// Grammar, response-format, tool-parser, and sampler construction are preflighted
// by prepare_generation before this function registers scheduler state or emits
// streaming headers.
GenResult run_generation(
    AppState & app, llama_seq_id seq_id, const GenParams & p,
    PreparedGeneration & prepared,
    std::function<bool(const json & event)> on_event,
    GenerationRegistration * generation = nullptr,
    llama_token rewind_token = LLAMA_TOKEN_NULL
) {
    GenResult r;
    SchedulerGenerationGuard scheduler_generation{*app.scheduler, seq_id};
    const SchedulerGenerationCheckpoint & checkpoint =
        scheduler_generation.checkpoint;
    const llama_pos pmax = checkpoint.position;
    // Rewind starts at the first position absent from the checkpoint. For an
    // empty sequence pmax is -1, so cancellation removes the temporary BOS at 0.
    const llama_pos p_start = pmax + 1;
    GenerationRegistrationGuard registration_guard{
        generation, seq_id, p_start, rewind_token, checkpoint.family};
    r.cache_size = pmax < 0 ? 0 : pmax + 1;
    llama_token boundary_token = rewind_token;
    common_sampler * smpl = prepared.sampler.get();
    common_chat_parser_params & parser_params = prepared.parser_params;
    const bool tool_calling_active = prepared.tool_calling_active;
    const uint8_t canonical_greedy_policy = prepared.canonical_greedy_policy;
    const int n_ctx = app.n_ctx_per_session;
    const double t0 = now_s();
    bool keep_going = true;

    StopSequenceMatcher stop_matcher;
    if (tool_calling_active) {
        // Tool parsing consumes only stop-safe text. Holding a possible stop
        // prefix here also prevents partial stop bytes from leaking as SSE
        // content/tool deltas.
        std::string acc;
        common_chat_msg prev_msg;
        std::vector<std::string> tc_ids_cache;
        int tc_counter = 0;
        auto gen_tc_id = [&]() { return std::to_string(++tc_counter); };
        auto emit_diffs = [&](const std::vector<common_chat_msg_diff> & diffs) {
            for (const auto & d : diffs) {
                if (!keep_going) break;
                if (d.tool_call_index == std::string::npos) {
                    if (!d.content_delta.empty() && on_event) {
                        keep_going = on_event(
                            json{{"type", "token"}, {"token", d.content_delta}});
                        if (!keep_going) r.cancelled = true;
                    }
                    if (keep_going &&
                        !d.reasoning_content_delta.empty() && on_event) {
                        keep_going = on_event(json{
                            {"type", "token"},
                            {"token", d.reasoning_content_delta},
                        });
                        if (!keep_going) r.cancelled = true;
                    }
                } else if (on_event) {
                    json tc = {{"index", static_cast<int>(d.tool_call_index)}};
                    if (!d.tool_call_delta.id.empty()) {
                        tc["id"] = std::string("fc_") + d.tool_call_delta.id;
                    }
                    if (!d.tool_call_delta.name.empty()) {
                        tc["name"] = d.tool_call_delta.name;
                    }
                    if (!d.tool_call_delta.arguments.empty()) {
                        tc["arguments"] = d.tool_call_delta.arguments;
                    }
                    keep_going = on_event(json{
                        {"type", "tool_call"},
                        {"tool_call", std::move(tc)},
                    });
                    if (!keep_going) r.cancelled = true;
                }
            }
        };
        auto parse_partial = [&]() {
            // TECH DEBT: re-parse the entire safe text on each emitted chunk.
            auto next = common_chat_parse(
                acc, /*is_partial*/ true, parser_params);
            if (next.empty()) return;
            next.set_tool_call_ids(tc_ids_cache, gen_tc_id);
            emit_diffs(common_chat_msg_diff::compute_diffs(prev_msg, next));
            prev_msg = std::move(next);
        };
        auto append_safe = [&](std::string_view text) {
            acc.append(text.data(), text.size());
            r.text.append(text.data(), text.size());
            return true;
        };

        for (int step = 0; step < p.max_tokens && keep_going; ++step) {
            if (generation &&
                generation->cancelled->load(std::memory_order_relaxed)) {
                r.cancelled = true;
                break;
            }
            const int cells_needed = r.cache_size == 0 ? 2 : 1;
            if (r.cache_size + cells_needed > n_ctx) {
                r.error = "session context full";
                break;
            }
            auto step_result = app.scheduler->step(
                seq_id, smpl, boundary_token, canonical_greedy_policy,
                static_cast<uint32_t>(step)).get();
            if (!step_result.error.empty()) {
                r.error = step_result.error;
                break;
            }
            if (step_result.eog) break;

            const llama_token id = step_result.token;
            r.cache_size = step_result.cache_size;
            const std::string piece =
                common_token_to_piece(app.vocab, id, true);
            common_sampler_accept(smpl, id, true);
            boundary_token = id;
            r.ids.push_back(static_cast<int64_t>(id));

            const size_t safe_size_before = acc.size();
            const StopMatchResult stop =
                stop_matcher.append(piece, p.stop, append_safe);
            if (acc.size() != safe_size_before) parse_partial();
            if (!keep_going) break;
            if (stop.matched) {
                keep_going = false;
                break;
            }
        }

        if (keep_going && !stop_matcher.matched()) {
            const size_t safe_size_before = acc.size();
            stop_matcher.finish(append_safe);
            if (acc.size() != safe_size_before) parse_partial();
        }

        // An intentional stop may leave a partial tool serialization. Parse it
        // in partial mode; completed tool calls remain available, while an
        // incomplete call is not promoted to a final result.
        if (!acc.empty()) {
            auto final_msg = common_chat_parse(
                acc, /*is_partial*/ stop_matcher.matched(), parser_params);
            if (!final_msg.empty()) {
                final_msg.set_tool_call_ids(tc_ids_cache, gen_tc_id);
                emit_diffs(
                    common_chat_msg_diff::compute_diffs(prev_msg, final_msg));
                for (const auto & tc : final_msg.tool_calls) {
                    r.tool_calls.push_back({
                        {"id", std::string("fc_") + tc.id},
                        {"type", "function"},
                        {"function", {
                            {"name", tc.name},
                            {"arguments", tc.arguments},
                        }},
                    });
                }
            }
        }
    } else {
        int64_t emitted_token_id = 0;
        auto emit_safe = [&](std::string_view text) {
            r.text.append(text.data(), text.size());
            if (!on_event) return true;
            const bool delivered = on_event(json{
                {"type", "token"},
                {"token", std::string(text)},
                {"id", emitted_token_id},
            });
            if (!delivered) {
                keep_going = false;
                r.cancelled = true;
            }
            return delivered;
        };

        for (int step = 0; step < p.max_tokens && keep_going; ++step) {
            if (generation &&
                generation->cancelled->load(std::memory_order_relaxed)) {
                r.cancelled = true;
                break;
            }
            const int cells_needed = r.cache_size == 0 ? 2 : 1;
            if (r.cache_size + cells_needed > n_ctx) {
                r.error = "session context full";
                break;
            }
            auto step_result = app.scheduler->step(
                seq_id, smpl, boundary_token, canonical_greedy_policy,
                static_cast<uint32_t>(step)).get();
            if (!step_result.error.empty()) {
                r.error = step_result.error;
                break;
            }
            if (step_result.eog) break;

            const llama_token id = step_result.token;
            r.cache_size = step_result.cache_size;
            const std::string piece =
                common_token_to_piece(app.vocab, id, true);
            common_sampler_accept(smpl, id, true);
            boundary_token = id;
            r.ids.push_back(static_cast<int64_t>(id));
            emitted_token_id = static_cast<int64_t>(id);

            const StopMatchResult stop =
                stop_matcher.append(piece, p.stop, emit_safe);
            if (stop.emission_failed) {
                keep_going = false;
                r.cancelled = true;
            } else if (stop.matched) {
                keep_going = false;
            }
        }

        // Flush a tail that only partially matched when generation ended for a
        // reason other than a completed stop or disconnected output.
        if (keep_going && !stop_matcher.matched()) {
            const StopMatchResult final = stop_matcher.finish(emit_safe);
            if (final.emission_failed) {
                keep_going = false;
                r.cancelled = true;
            }
        }
    }

    registration_guard.finish(r);
    r.cache_size = app.scheduler->invoke_preserving_logits([seq_id](llama_context * ctx) {
        return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
    });
    r.gen_s = now_s() - t0;
    return r;
}

} // namespace

// ---- multimodal helpers (outside anon namespace so they can be forward-declared) ----

// Decode image or audio bytes into an independently owned mtmd bitmap. The
// upstream helper is thread-safe; mtmd's official `bitmap_ptr` wrapper releases
// the result with the context-free `mtmd_bitmap_free` function.
static mtmd_bitmap * bitmap_from_media_bytes(mtmd_context * mtmd_ctx,
                                             const std::string & bytes) {
    mtmd_helper_bitmap_wrapper wrap = mtmd_helper_bitmap_init_from_buf(
        mtmd_ctx,
        reinterpret_cast<const unsigned char *>(bytes.data()),
        bytes.size(),
        /*placeholder*/ false);
    mtmd::bitmap_ptr bitmap{wrap.bitmap};
    mtmd_helper::video_ptr video{wrap.video_ctx};
    if (video) return nullptr;  // video is outside this endpoint's contract
    return bitmap.release();  // may be nullptr on failure
}

static bool decode_tokens_in_chunks(
    llama_context * ctx, llama_seq_id seq_id, llama_pos start_position,
    const std::vector<llama_token> & tokens) {
    const uint32_t batch_capacity = llama_n_batch(ctx);
    if (batch_capacity == 0) return false;

    for (size_t offset = 0; offset < tokens.size(); offset += batch_capacity) {
        const size_t chunk_size =
            std::min<size_t>(batch_capacity, tokens.size() - offset);
        llama_batch batch =
            llama_batch_init(static_cast<int32_t>(chunk_size), 0, 1);
        for (size_t i = 0; i < chunk_size; ++i) {
            const size_t token_index = offset + i;
            batch.token[i] = tokens[token_index];
            batch.pos[i] = start_position + static_cast<llama_pos>(token_index);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i] = token_index + 1 == tokens.size();
        }
        batch.n_tokens = static_cast<int32_t>(chunk_size);
        const bool decoded = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        if (!decoded) return false;
    }
    return true;
}

// Run the mtmd tokenize + per-chunk eval path: turns the marker-containing text
// + bitmaps into chunks and decodes each into the session's KV cache. Preserve
// the final discrete token when the rendered prompt ends in text: the pooled
// scheduler needs it to recreate sequence-specific boundary logits.
enum class InjectStatus {
    ok,
    context_full,
    decode_failed,
};

struct MtmdInjectResult {
    InjectStatus status = InjectStatus::decode_failed;
    llama_token last_token = LLAMA_TOKEN_NULL;
};

static MtmdInjectResult mtmd_inject(mtmd_context * mtmd_ctx, llama_context * ctx,
                                    llama_seq_id seq_id, const std::string & text,
                                    const std::vector<mtmd::bitmap_ptr> & bitmaps,
                                    llama_pos context_limit) {
    mtmd_input_text input_text;
    input_text.text          = text.c_str();
    input_text.add_special   = true;
    input_text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    std::vector<const mtmd_bitmap *> bptrs;
    bptrs.reserve(bitmaps.size());
    for (const auto & bitmap : bitmaps) bptrs.push_back(bitmap.get());

    int32_t rc = mtmd_tokenize(mtmd_ctx, chunks, &input_text,
                               bptrs.data(), bptrs.size());
    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        return {};
    }

    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    llama_token last_token = LLAMA_TOKEN_NULL;
    if (n_chunks > 0) {
        const mtmd_input_chunk * final_chunk = mtmd_input_chunks_get(chunks, n_chunks - 1);
        if (mtmd_input_chunk_get_type(final_chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(final_chunk, &n_tokens);
            if (tokens && n_tokens > 0) last_token = tokens[n_tokens - 1];
        }
    }
    llama_pos n_past = 0;
    // start from the current cache position (so multi-turn inject composes)
    llama_memory_t memory = llama_get_memory(ctx);
    llama_pos cur_max = llama_memory_seq_pos_max(memory, seq_id);
    if (cur_max >= 0) n_past = cur_max + 1;
    if (n_past + mtmd_helper_get_n_pos(chunks) > context_limit) {
        mtmd_input_chunks_free(chunks);
        return {InjectStatus::context_full, LLAMA_TOKEN_NULL};
    }
    const llama_pos original_n_past = n_past;

    bool ok = true;
    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        llama_pos new_n_past = n_past;
        int32_t r = mtmd_helper_eval_chunk_single(
            mtmd_ctx, ctx, chunk, n_past, seq_id,
            static_cast<int32_t>(llama_n_batch(ctx)),
            /*logits_last*/ (i == n_chunks - 1), &new_n_past);
        if (r != 0) { ok = false; break; }
        n_past = new_n_past;
    }
    mtmd_input_chunks_free(chunks);
    if (!ok) {
        llama_memory_seq_rm(memory, seq_id, original_n_past, -1);
        return {InjectStatus::decode_failed, LLAMA_TOKEN_NULL};
    }
    return {InjectStatus::ok, last_token};
}

int main(int argc, char ** argv) {
    ServerCliResult cli = parse_server_arguments(argc, argv);
    if (cli.show_help) {
        std::cout << server_help_text();
        return 0;
    }
    if (!cli.error.empty()) {
        std::cerr << "error: " << cli.error << "\n";
        return 2;
    }
    ServerConfig cfg = std::move(cli.config);
    if (!cfg.chat_template_file.empty()) {
        try {
            cfg.chat_template = read_file_contents(cfg.chat_template_file);
        } catch (const std::exception & exception) {
            std::cerr << "error: cannot read --chat-template-file '"
                      << cfg.chat_template_file << "': " << exception.what() << "\n";
            return 2;
        }
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
    const size_t gpu_free_before_model = gpu_free_bytes();
    AppState app;
    app.model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
    if (!app.model) {
        std::cerr << "error: failed to load model\n";
        if (!cfg.allow_cpu) std::cerr << "error: " << gpu_execution_error("multimodal-server") << "\n";
        llama_backend_free();
        return 1;
    }
    const size_t gpu_free_after_model = gpu_free_bytes();
    app.model_gpu_bytes = gpu_allocation_delta(gpu_free_before_model, gpu_free_after_model);
    if (!gpu_execution_allowed(cfg.allow_cpu, app.model_gpu_bytes)) {
        std::cerr << "error: " << gpu_execution_error("multimodal-server") << "\n";
        llama_model_free(app.model);
        llama_backend_free();
        return 3;
    }
    std::cerr << "execution: actual GPU model allocation=" << app.model_gpu_bytes << " bytes"
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
    app.max_sequences = cfg.max_sequences;
    std::cerr << "model loaded.\n";

    // Load the multimodal projector (if given). Encoder-free models like Gemma 4
    // 12B have a tiny projector (no heavy ViT), so this is cheap.
    if (!cfg.mmproj_path.empty()) {
        std::cerr << "loading mmproj: " << cfg.mmproj_path << " ...\n";
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu = (cfg.n_gpu_layers != 0);
        mp.warmup  = true;
        app.mtmd_ctx = mtmd_init_from_file(cfg.mmproj_path.c_str(), app.model, mp);
        if (!app.mtmd_ctx) {
            std::cerr << "error: failed to load mmproj\n";
            return 1;
        }
        app.supports_vision = mtmd_support_vision(app.mtmd_ctx);
        app.supports_audio = mtmd_support_audio(app.mtmd_ctx);
        app.audio_sample_rate = mtmd_get_audio_sample_rate(app.mtmd_ctx);
        app.media_marker = mtmd_default_marker();
        std::cerr << "mmproj loaded (vision=" << app.supports_vision
                  << " audio=" << app.supports_audio << ").\n";
    }

    // Allocate the one process-wide context. --ctx-size remains per sequence;
    // llama receives the explicitly multiplied pool size and sequence capacity.
    const size_t gpu_free_before_pool = gpu_free_bytes();
    llama_context_params pooled_params = llama_context_default_params();
    const uint64_t requested_pool_size =
        static_cast<uint64_t>(cfg.ctx_size) * static_cast<uint64_t>(cfg.max_sequences);
    pooled_params.n_ctx = static_cast<uint32_t>(requested_pool_size);
    pooled_params.n_batch =
        static_cast<uint32_t>(std::min<uint64_t>(cfg.n_batch, requested_pool_size));
    // Keep two internal IDs available for the disposable startup copy/removal
    // probe even when public live capacity is configured as one.
    pooled_params.n_seq_max = std::max(cfg.max_sequences, 2);
    // Output reservation is keyed to n_seq_max even when a smaller n_batch
    // decodes the two disposable probe sequences in separate calls.
    pooled_params.n_outputs_max = pooled_params.n_seq_max;
    pooled_params.kv_unified = true;  // required for tokens coupled to multiple fork sequences
    pooled_params.no_perf = true;
    llama_context * pooled_ctx = llama_init_from_model(app.model, pooled_params);
    if (!pooled_ctx) {
        std::cerr << "error: failed to create pooled inference context\n";
        return 1;
    }
    if (llama_n_ctx(pooled_ctx) < pooled_params.n_ctx) {
        std::cerr << "error: pooled context provides only " << llama_n_ctx(pooled_ctx)
                  << " total tokens; requested " << pooled_params.n_ctx << "\n";
        llama_free(pooled_ctx);
        return 1;
    }
    if (llama_n_batch(pooled_ctx) < static_cast<uint32_t>(cfg.max_sequences)) {
        std::cerr << "error: pooled context batch capacity " << llama_n_batch(pooled_ctx)
                  << " is smaller than --max-sequences " << cfg.max_sequences
                  << "; increase --n-batch or reduce --max-sequences\n";
        llama_free(pooled_ctx);
        if (app.mtmd_ctx) mtmd_free(app.mtmd_ctx);
        llama_model_free(app.model);
        llama_backend_free();
        return 2;
    }
    app.scheduler = std::make_unique<InferenceScheduler>(
        pooled_ctx, app.vocab, cfg.max_sequences);
    std::string probe_error;
    if (!app.scheduler->probe_sequence_capabilities(probe_error)) {
        std::cerr << "error: pooled sequence compatibility probe failed: " << probe_error << "\n";
        app.scheduler.reset();
        if (app.mtmd_ctx) mtmd_free(app.mtmd_ctx);
        llama_model_free(app.model);
        llama_backend_free();
        return 1;
    }
    std::cerr << "pool: capacity=" << cfg.max_sequences
              << " per_sequence_ctx=" << cfg.ctx_size
              << " total_ctx=" << llama_n_ctx(pooled_ctx)
              << " decode_batch_capacity=" << llama_n_batch(pooled_ctx)
              << " preallocated_state_bytes=" << app.scheduler->preallocated_bytes() << "\n";

    // Warm the pooled decode and same-stream seq_cp paths without creating a
    // second context. The temporary sequence is fully released before serving.
    std::cerr << "warming up pooled scheduler...\n";
    const int warm_seq = app.scheduler->allocate_sequence();
    if (warm_seq < 0) {
        std::cerr << "error: failed to reserve warmup sequence\n";
        return 1;
    }
    GenParams warm_params;
    warm_params.max_tokens = 5;
    warm_params.temp = 0.0f;
    warm_params.seed = 0;
    PreparedGeneration warm_prepared = prepare_generation(app, warm_params);
    run_generation(app, warm_seq, warm_params, warm_prepared, nullptr);
    if (auto warm_fork = app.scheduler->fork_sequence(warm_seq, LLAMA_TOKEN_NULL)) {
        app.scheduler->release_sequence(*warm_fork);
    }
    app.scheduler->release_sequence(warm_seq);
    app.scheduler->invoke_preserving_logits([](llama_context * ctx) { llama_synchronize(ctx); });
    const size_t gpu_free_after_pool = gpu_free_bytes();
    app.pool_footprint_bytes = gpu_free_before_pool > gpu_free_after_pool
        ? gpu_free_before_pool - gpu_free_after_pool : 0;
    std::cerr << "warmup done; pooled GPU footprint=" << app.pool_footprint_bytes << " bytes.\n";

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
        const bool v = app.supports_vision;
        const bool a = app.supports_audio;
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
            {"audio_sample_rate", app.audio_sample_rate},
            {"input_modalities", in_arr},
            {"output_modalities", out_arr},
        };
        res.set_content(body.dump(), "application/json");
    });

    // ---- POST /sessions : create a session ----
    svr.Post("/sessions", [&](const httplib::Request &, httplib::Response &res) {
        const int seq_id = app.scheduler->allocate_sequence();
        if (seq_id < 0) {
            res.status = 503;
            res.set_content(error_body("pooled sequence capacity exhausted", 503).dump(), "application/json");
            return;
        }
        const int64_t n = register_session(app, seq_id);
        json body = {{"session_id", make_session_id(n)}};
        res.set_content(body.dump(), "application/json");
    });

    // ---- POST /sessions/{id}/fork : alias this session's prefix into a NEW session ----
    //
    // Approach B (ADRs 0004 and 0009): allocate another sequence in the pooled
    // context and copy the source's sequence metadata with llama_memory_seq_cp.
    // The immutable prefix KV cells are shared; only later divergent suffixes
    // consume additional cells. Source and fork keep independent sequence,
    // sampler, and generation state. Forking snapshots whatever cache the source
    // currently holds (after text/audio injection or generation) without
    // modifying the source.
    svr.Post(R"(/sessions/[^/]+/fork)", [&](const httplib::Request &req, httplib::Response &res) {
        std::string sid = extract_session_id(req.path, "fork");
        const int64_t sid_num = parse_session_id_num(sid);
        auto src = reserve_live_session(app, sid_num, res);
        if (!src) return;
        BusyGuard busy{app, sid_num, true};

        const double t0 = now_s();
        auto destination = app.scheduler->fork_sequence(src->seq_id, src->last_token);
        if (!destination) {
            res.status = 503;
            res.set_content(error_body("pooled sequence capacity exhausted", 503).dump(), "application/json");
            return;
        }
        const double dt = now_s() - t0;

        const int64_t new_n = register_session(app, *destination, src->last_token);
        const int dst_size = app.scheduler->invoke_preserving_logits([seq = *destination](llama_context * ctx) {
            return llama_memory_seq_pos_max(llama_get_memory(ctx), seq) + 1;
        });
        json body = {
            {"session_id", make_session_id(new_n)},
            {"forked_from", sid},
            {"cache_size", dst_size},
            {"fork_ms", (int)(dt * 1000)},
            {"fork_ms_precise", dt * 1000.0},
        };
        res.set_content(body.dump(), "application/json");
    });

    // ---- POST /sessions/{id}/inject : text -> KV cache (no generation) ----
    svr.Post(R"(/sessions/[^/]+/inject)", [&](const httplib::Request &req, httplib::Response &res) {
        std::string sid = extract_session_id(req.path, "inject");
        const int64_t sid_num = parse_session_id_num(sid);
        auto session = reserve_live_session(app, sid_num, res);
        if (!session) return;
        BusyGuard busy{app, sid_num, true};
        const llama_seq_id seq_id = session->seq_id;
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
                std::vector<mtmd::bitmap_ptr> bitmaps;
                const std::string & media_marker = app.media_marker;
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
                            mtmd::bitmap_ptr bmp{
                                app.scheduler->invoke_preserving_logits([&](llama_context *) {
                                    return bitmap_from_media_bytes(app.mtmd_ctx, bytes);
                                })};
                            if (!bmp) {
                                throw std::runtime_error("failed to decode image");
                            }
                            bitmaps.push_back(std::move(bmp));
                            out += media_marker;
                            used_multimodal = true;
                        } else if (ptype == "audio" || ptype == "input_audio") {
                            if (!app.mtmd_ctx) {
                                throw std::runtime_error("audio part requires --mmproj to be loaded");
                            }
                            if (!app.supports_audio) {
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
                            mtmd::bitmap_ptr bmp{
                                app.scheduler->invoke_preserving_logits([&](llama_context *) {
                                    return bitmap_from_media_bytes(app.mtmd_ctx, bytes);
                                })};
                            if (!bmp) {
                                throw std::runtime_error("failed to decode audio");
                            }
                            bitmaps.push_back(std::move(bmp));
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
                    const uint64_t mutation_parent =
                        app.scheduler->prepare_sequence_mutation(seq_id);
                    MtmdInjectResult inject_result = app.scheduler->invoke_invalidating_logits([&](llama_context * ctx) {
                        const auto result = mtmd_inject(
                            app.mtmd_ctx, ctx, seq_id, text, bitmaps, app.n_ctx_per_session);
                        return result;
                    });
                    const double dt = now_s() - t0;
                    if (inject_result.status != InjectStatus::ok) {
                        app.scheduler->abort_sequence_mutation(
                            seq_id, mutation_parent, session->last_token);
                        res.status = inject_result.status == InjectStatus::context_full ? 409 : 500;
                        const std::string message = res.status == 409
                            ? "session context full" : "mtmd tokenize/decode failed";
                        res.set_content(error_body(message, res.status).dump(), "application/json");
                        return;
                    }
                    app.scheduler->complete_opaque_mutation(
                        seq_id, inject_result.last_token);
                    const int new_size = app.scheduler->invoke_preserving_logits([seq_id](llama_context * ctx) {
                        return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
                    });
                    json body = {
                        {"session_id", sid},
                        {"cache_size", new_size},
                        {"inject_ms", (int)(dt * 1000)},
                        {"chat_template_applied", used_template},
                        {"used_multimodal", true},
                        {"n_media", bitmaps.size()},
                    };
                    if (return_prompt) body["prompt"] = text;
                    // Text-ending rendered prompts are generation-ready. A
                    // media-ending prompt retains ADR 0004's explicit caveat.
                    set_last_token(app, sid_num, inject_result.last_token);
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

                const uint64_t mutation_parent =
                    app.scheduler->prepare_sequence_mutation(seq_id);
                InjectStatus status = app.scheduler->invoke_invalidating_logits([&](llama_context * ctx) {
                    const float * samples = reinterpret_cast<const float *>(bytes.data());
                    mtmd_bitmap * bmp = mtmd_bitmap_init_from_audio(n_samples, samples);
                    if (!bmp) return InjectStatus::decode_failed;
                    std::string dummy_prompt = mtmd_default_marker();
                    mtmd_input_text input_text;
                    input_text.text = dummy_prompt.c_str();
                    input_text.add_special = false;
                    input_text.parse_special = true;
                    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
                    const mtmd_bitmap * bptrs[1] = {bmp};
                    const int32_t rc = mtmd_tokenize(app.mtmd_ctx, chunks, &input_text, bptrs, 1);
                    llama_memory_t memory = llama_get_memory(ctx);
                    const llama_pos cur_max = llama_memory_seq_pos_max(memory, seq_id);
                    const llama_pos original_n_past = cur_max < 0 ? 0 : cur_max + 1;
                    llama_pos required = 0;
                    bool has_audio = false;
                    if (rc == 0) {
                        const size_t n_chunks = mtmd_input_chunks_size(chunks);
                        for (size_t i = 0; i < n_chunks; ++i) {
                            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
                            if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) continue;
                            has_audio = true;
                            required += mtmd_input_chunk_get_n_pos(chunk);
                        }
                    }

                    InjectStatus result = InjectStatus::decode_failed;
                    if (rc == 0 && has_audio &&
                        original_n_past + required <= app.n_ctx_per_session) {
                        bool decoded = true;
                        llama_pos n_past = original_n_past;
                        const size_t n_chunks = mtmd_input_chunks_size(chunks);
                        for (size_t i = 0; i < n_chunks; ++i) {
                            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
                            if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) continue;
                            llama_pos new_n_past = n_past;
                            if (mtmd_helper_eval_chunk_single(
                                    app.mtmd_ctx, ctx, chunk, n_past, seq_id,
                                    static_cast<int32_t>(llama_n_batch(ctx)),
                                    /*logits_last*/ true, &new_n_past) != 0) {
                                decoded = false;
                                break;
                            }
                            n_past = new_n_past;
                        }
                        if (decoded) {
                            result = InjectStatus::ok;
                        } else {
                            llama_memory_seq_rm(memory, seq_id, original_n_past, -1);
                        }
                    } else if (rc == 0 && has_audio) {
                        result = InjectStatus::context_full;
                    }
                    mtmd_bitmap_free(bmp);
                    mtmd_input_chunks_free(chunks);
                    return result;
                });

                if (status != InjectStatus::ok) {
                    app.scheduler->abort_sequence_mutation(
                        seq_id, mutation_parent, session->last_token);
                    res.status = status == InjectStatus::context_full ? 409 : 500;
                    const std::string message = res.status == 409
                        ? "session context full" : "failed to decode streaming audio chunk";
                    res.set_content(error_body(message, res.status).dump(), "application/json");
                    return;
                }

                app.scheduler->complete_opaque_mutation(seq_id, LLAMA_TOKEN_NULL);
                const int new_size = app.scheduler->invoke_preserving_logits([seq_id](llama_context * ctx) {
                    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
                });
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
        const int used = app.scheduler->invoke_preserving_logits([seq_id](llama_context * ctx) {
            return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
        });
        const bool is_first = used == 0;
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
        const int n_ctx = app.n_ctx_per_session;
        if (used + (int)toks.size() > n_ctx) {
            res.status = 409;
            res.set_content(error_body("session context full", 409).dump(), "application/json");
            return;
        }
        const double t0 = now_s();
        const uint64_t mutation_parent =
            app.scheduler->prepare_sequence_mutation(seq_id);
        const bool decoded = app.scheduler->invoke_invalidating_logits([&](llama_context * ctx) {
            const bool result = decode_tokens_in_chunks(ctx, seq_id, used, toks);
            if (!result) {
                llama_memory_seq_rm(llama_get_memory(ctx), seq_id, used, -1);
            }
            return result;
        });
        if (!decoded) {
            app.scheduler->abort_sequence_mutation(
                seq_id, mutation_parent, session->last_token);
            res.status = 500;
            res.set_content(error_body("llama_decode failed", 500).dump(), "application/json");
            return;
        }
        const double dt = now_s() - t0;
        app.scheduler->complete_text_mutation(seq_id, mutation_parent, used, toks);
        if (!toks.empty()) set_last_token(app, sid_num, toks.back());  // for fork logits refresh
        const int new_size = app.scheduler->invoke_preserving_logits([seq_id](llama_context * ctx) {
            return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
        });
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
        bool live_exists = false;
        bool offloaded_exists = false;
        {
            std::lock_guard<std::mutex> lk(app.mu);
            live_exists = app.sessions.count(sid_num) > 0;
            offloaded_exists = app.offloaded_sessions.count(sid_num) > 0;
        }
        if (!live_exists) {
            if (offloaded_exists) {
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
        auto session = reserve_live_session(app, sid_num, res);
        if (!session) return;
        auto busy = std::make_shared<BusyGuard>(app, sid_num);
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

        const bool has_tools = gp.tools.is_array() && !gp.tools.empty();
        const std::string validation_error =
            validate_generation_constraints(
                gp.response_format, gp.grammar, has_tools, gp.stop);
        if (!validation_error.empty()) {
            res.status = 400;
            res.set_content(
                error_body(validation_error, 400).dump(), "application/json");
            return;
        }

        // Derive and parse every client-selected constraint, then construct the
        // RAII-owned sampler before registering generation state or SSE headers.
        std::shared_ptr<PreparedGeneration> prepared;
        try {
            prepared = std::make_shared<PreparedGeneration>(
                prepare_generation(app, gp));
        } catch (const std::bad_alloc &) {
            res.status = 500;
            res.set_content(
                error_body("generation preflight allocation failed", 500).dump(),
                "application/json");
            return;
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(
                error_body(
                    std::string("invalid generation constraints: ") + e.what(),
                    400).dump(),
                "application/json");
            return;
        }
        const llama_seq_id seq_id = session->seq_id;
        if (gp.max_tokens > 0) {
            const int current_size = app.scheduler->invoke_preserving_logits(
                [seq_id](llama_context * ctx) {
                    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
                });
            const int cells_needed = current_size == 0 ? 2 : 1;
            if (current_size + cells_needed > app.n_ctx_per_session) {
                res.status = 409;
                res.set_content(error_body("session context full", 409).dump(),
                                "application/json");
                return;
            }
        }
        const llama_token pre_generation_last = session->last_token;
        auto generation = std::make_shared<GenerationRegistration>(app, sid);

        if (!stream) {
            // Non-streaming: whole response as JSON (incl. tool_calls if any).
            GenResult r;
            try {
                r = run_generation(
                    app, seq_id, gp, *prepared, nullptr,
                    generation.get(), pre_generation_last);
            } catch (const std::exception & e) {
                res.status = 500;
                res.set_content(
                    error_body(
                        std::string("generation failed: ") + e.what(), 500).dump(),
                    "application/json");
                return;
            } catch (...) {
                res.status = 500;
                res.set_content(
                    error_body("generation failed", 500).dump(),
                    "application/json");
                return;
            }
            const bool partial_context_full =
                r.error == "session context full" && !r.ids.empty();
            if (!r.error.empty() && !partial_context_full) {
                res.status = r.error == "session context full" ? 409 : 500;
                res.set_content(error_body(r.error, res.status).dump(), "application/json");
                return;
            }
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
                {"gen_ms_precise", r.gen_s * 1000.0},
                {"tokens_per_s", tok_s},
                {"cache_size", r.cache_size},
            };
            if (!r.tool_calls.empty()) body["tool_calls"] = r.tool_calls;
            if (partial_context_full) body["finish_reason"] = "context_full";
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
            [&app, seq_id, gp, sid, sid_num, prepared, generation, busy,
             pre_generation_last, is_connection_closed]
            (size_t, httplib::DataSink & ds) -> bool {
                auto close_stream = [&ds](const json & terminal) noexcept {
                    try {
                        const std::string event = sse_event(terminal);
                        ds.write(event.data(), event.size());
                    } catch (...) {
                    }
                    try {
                        ds.done();
                    } catch (...) {
                    }
                };
                auto close_with_error =
                    [&close_stream, &ds](const char * message) noexcept {
                        try {
                            close_stream(json{
                                {"type", "error"},
                                {"error", message},
                                {"code", 500},
                            });
                        } catch (...) {
                            try {
                                ds.done();
                            } catch (...) {
                            }
                        }
                    };

                try {
                    GenResult r = run_generation(
                        app, seq_id, gp, *prepared,
                        [&ds, &is_connection_closed](const json & event) {
                            if (is_connection_closed()) return false;
                            const std::string encoded = sse_event(event);
                            return ds.write(encoded.data(), encoded.size());
                        },
                        generation.get(), pre_generation_last);
                    if (!r.cancelled && !r.ids.empty()) {
                        set_last_token(
                            app, sid_num, static_cast<llama_token>(r.ids.back()));
                    }
                    if (r.cancelled) {
                        std::cerr << "generation cancelled for " << sid
                                  << "; rewind=" << r.rewind_s * 1000.0
                                  << " ms\n";
                    }
                    const bool partial_context_full =
                        r.error == "session context full" && !r.ids.empty();
                    json terminal;
                    if (!r.error.empty() && !partial_context_full) {
                        terminal = {
                            {"type", "error"},
                            {"error", r.error},
                            {"code", r.error == "session context full" ? 409 : 500},
                        };
                    } else {
                        const double tok_s =
                            r.gen_s > 0 ? r.ids.size() / r.gen_s : 0.0;
                        terminal = {
                            {"type", "done"},
                            {"n_tokens", r.ids.size()},
                            {"gen_ms", static_cast<int>(r.gen_s * 1000)},
                            {"gen_ms_precise", r.gen_s * 1000.0},
                            {"tokens_per_s", tok_s},
                            {"cache_size", r.cache_size},
                        };
                        if (!r.tool_calls.empty()) {
                            terminal["tool_calls"] = r.tool_calls;
                        }
                        if (partial_context_full) {
                            terminal["finish_reason"] = "context_full";
                        }
                    }
                    close_stream(terminal);
                } catch (const std::exception & e) {
                    close_with_error(e.what());
                } catch (...) {
                    close_with_error("generation failed");
                }
                return true;
            }
        );
    });

    // ---- GET /sessions/usage : aggregate VRAM/RAM footprint (EUS-6) ----
    // MUST be registered before GET /sessions/{id} so the literal "usage" is not
    // shadowed by the {id} regex (cpp-httplib matches in registration order).
    svr.Get("/sessions/usage", [&](const httplib::Request &, httplib::Response &res) {
        std::vector<std::pair<int64_t, llama_seq_id>> live;
        std::vector<int64_t> live_reader_ids;
        json ram_sessions = json::array();
        size_t ram_total = 0;
        {
            std::unique_lock<std::mutex> lk(app.mu);
            app.sessions_cv.wait(lk, [&] {
                return std::none_of(
                    app.sessions.begin(), app.sessions.end(),
                    [](const auto & entry) { return entry.second.releasing; });
            });
            for (auto & [n, session] : app.sessions) {
                ++session.readers;
                live.push_back({n, session.seq_id});
                live_reader_ids.push_back(n);
            }
            for (const auto & [n, state] : app.offloaded_sessions) {
                ram_sessions.push_back({{"session_id", make_session_id(n)}, {"state_bytes", state.state_bytes}});
                ram_total += state.state_bytes;
            }
        }
        LiveReadersGuard live_readers{app, std::move(live_reader_ids)};
        auto live_sizes = app.scheduler->invoke_preserving_logits([live](llama_context * ctx) {
            std::vector<size_t> sizes;
            sizes.reserve(live.size());
            for (const auto & [_, seq] : live) sizes.push_back(llama_state_seq_get_size(ctx, seq));
            return sizes;
        });
        json vram_sessions = json::array();
        size_t vram_total = 0;
        for (size_t i = 0; i < live.size(); ++i) {
            vram_sessions.push_back({{"session_id", make_session_id(live[i].first)},
                                     {"state_bytes", live_sizes[i]}});
            vram_total += live_sizes[i];
        }
        const int active = app.scheduler->active_sequences();
        const uint32_t total_ctx = app.scheduler->invoke_preserving_logits([](llama_context * ctx) {
            return llama_n_ctx(ctx);
        });
        const SchedulerLogicalUsage logical = app.scheduler->logical_usage();
        json body = {
            {"vram", {{"n_sessions", vram_sessions.size()}, {"total_state_bytes", vram_total}, {"sessions", vram_sessions}}},
            {"ram",  {{"n_sessions", ram_sessions.size()},  {"total_state_bytes", ram_total},  {"sessions", ram_sessions}}},
            {"pool", {
                {"contexts_created", 1},
                {"capacity", app.max_sequences},
                {"active_sequences", active},
                {"free_sequences", app.max_sequences - active},
                {"ctx_size_per_sequence", app.n_ctx_per_session},
                {"ctx_size_total", total_ctx},
                {"preallocated_bytes", app.pool_footprint_bytes},
                {"model_gpu_bytes", app.model_gpu_bytes},
                {"gpu_free_bytes", gpu_free_bytes()},
                {"logical_owned_cells", logical.owned_cells},
                {"logical_allocated_bytes", logical.estimated_bytes},
                {"sequence_capabilities_probed", app.scheduler->sequence_capabilities_probed()},
            }},
        };
        res.set_content(body.dump(), "application/json");
    });

    svr.Get("/diagnostics/batching", [&](const httplib::Request &, httplib::Response &res) {
        const auto metrics = app.scheduler->diagnostics();
        json histogram = json::object();
        for (const auto & [width, count] : metrics.decode_calls_by_sequence_count) {
            histogram[std::to_string(width)] = count;
        }
        res.set_content(json{
            {"decode_calls", metrics.decode_calls},
            {"decoded_tokens", metrics.decoded_tokens},
            {"max_sequences_per_decode", metrics.max_sequences_per_decode},
            {"decode_calls_by_sequence_count", histogram},
            {"lineage_cache", {
                {"active_families", metrics.active_lineage_families},
                {"detached_families", metrics.detached_lineage_families},
                {"transitions", metrics.lineage_transitions},
                {"canonical_tokens", metrics.canonical_greedy_tokens},
            }},
        }.dump(), "application/json");
    });

    // ---- GET /sessions/{id} : sequence-scoped status and logical footprint ----
    svr.Get(R"(/sessions/[^/]+)", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string prefix = "/sessions/";
        if (req.path.rfind(prefix, 0) != 0 || req.path.find('/', prefix.size()) != std::string::npos) {
            res.status = 404;
            res.set_content(error_body("not found", 404).dump(), "application/json");
            return;
        }
        const int64_t sid_num = parse_session_id_num(req.path.substr(prefix.size()));
        std::optional<LiveSession> live;
        std::optional<json> parked;
        {
            std::lock_guard<std::mutex> lk(app.mu);
            auto live_it = app.sessions.find(sid_num);
            if (live_it != app.sessions.end()) {
                if (live_it->second.releasing) {
                    res.status = 409;
                    res.set_content(error_body("session already has an active operation", 409).dump(),
                                    "application/json");
                    return;
                }
                ++live_it->second.readers;
                live = live_it->second;
            } else {
                auto parked_it = app.offloaded_sessions.find(sid_num);
                if (parked_it != app.offloaded_sessions.end()) {
                    parked = json{{"session_id", make_session_id(sid_num)}, {"location", "ram"},
                                  {"cache_size", parked_it->second.cache_size},
                                  {"state_bytes", parked_it->second.state_bytes},
                                  {"busy", parked_it->second.loading}};
                }
            }
        }
        if (live) {
            LiveReadGuard reader{app, sid_num};
            const auto status = app.scheduler->invoke_preserving_logits(
                [seq = live->seq_id](llama_context * ctx) {
                    return std::pair<int, size_t>{
                        llama_memory_seq_pos_max(llama_get_memory(ctx), seq) + 1,
                        llama_state_seq_get_size(ctx, seq)};
                });
            const llama_token boundary = app.scheduler->boundary_token(live->seq_id);
            res.set_content(json{{"session_id", make_session_id(sid_num)}, {"location", "vram"},
                                 {"cache_size", status.first}, {"state_bytes", status.second},
                                 {"boundary_token", boundary}, {"busy", live->busy}}.dump(),
                            "application/json");
            return;
        }
        if (parked) {
            res.set_content(parked->dump(), "application/json");
            return;
        }
        res.status = 404;
        res.set_content(error_body("unknown session", 404).dump(), "application/json");
    });

    // Offload serializes/removes one sequence. It frees logical KV capacity and
    // the slot, not the process-wide preallocated pooled buffers (ADR 0009).
    svr.Post(R"(/sessions/[^/]+/offload)", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string sid = extract_session_id(req.path, "offload");
        const int64_t sid_num = parse_session_id_num(sid);
        {
            std::lock_guard<std::mutex> lk(app.mu);
            auto parked = app.offloaded_sessions.find(sid_num);
            if (parked != app.offloaded_sessions.end()) {
                if (parked->second.loading) {
                    res.status = 409;
                    res.set_content(error_body("session already has an active operation", 409).dump(),
                                    "application/json");
                    return;
                }
                res.set_content(json{{"session_id", sid}, {"location", "ram"},
                                     {"cache_size", parked->second.cache_size},
                                     {"state_bytes", parked->second.state_bytes},
                                     {"offload_ms", 0}}.dump(), "application/json");
                return;
            }
        }
        auto live = reserve_live_session(app, sid_num, res, /*will_release_sequence*/ true);
        if (!live) return;
        BusyGuard busy{app, sid_num, true};
        const double t0 = now_s();
        OffloadedState state = app.scheduler->invoke_invalidating_logits([seq = live->seq_id, last = live->last_token](llama_context * ctx) {
            OffloadedState result;
            result.state_bytes = llama_state_seq_get_size(ctx, seq);
            result.state.resize(result.state_bytes);
            llama_state_seq_get_data(ctx, result.state.data(), result.state_bytes, seq);
            result.cache_size = llama_memory_seq_pos_max(llama_get_memory(ctx), seq) + 1;
            result.last_token = last;
            return result;
        });
        app.scheduler->release_sequence(live->seq_id);
        {
            std::lock_guard<std::mutex> lk(app.mu);
            app.sessions.erase(sid_num);
            app.offloaded_sessions[sid_num] = state;
        }
        const double elapsed_ms = (now_s() - t0) * 1000.0;
        res.set_content(json{{"session_id", sid}, {"location", "ram"},
                             {"cache_size", state.cache_size}, {"state_bytes", state.state_bytes},
                             {"offload_ms", (int) elapsed_ms}}.dump(), "application/json");
    });

    svr.Post(R"(/sessions/[^/]+/load)", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string sid = extract_session_id(req.path, "load");
        const int64_t sid_num = parse_session_id_num(sid);
        std::optional<LiveSession> live;
        OffloadedState state;
        {
            std::lock_guard<std::mutex> lk(app.mu);
            auto live_it = app.sessions.find(sid_num);
            if (live_it != app.sessions.end()) {
                if (live_it->second.busy || live_it->second.readers > 0) {
                    res.status = 409;
                    res.set_content(error_body("session already has an active operation", 409).dump(),
                                    "application/json");
                    return;
                }
                ++live_it->second.readers;
                live = live_it->second;
            } else {
                auto parked = app.offloaded_sessions.find(sid_num);
                if (parked == app.offloaded_sessions.end()) {
                    res.status = 404;
                    res.set_content(error_body("unknown session", 404).dump(), "application/json");
                    return;
                }
                if (parked->second.loading) {
                    res.status = 409;
                    res.set_content(error_body("session already has an active operation", 409).dump(),
                                    "application/json");
                    return;
                }
                parked->second.loading = true;
                state = parked->second;
            }
        }
        if (live) {
            LiveReadGuard reader{app, sid_num};
            const auto status = app.scheduler->invoke_preserving_logits(
                [seq = live->seq_id](llama_context * ctx) {
                    return std::pair<int, size_t>{
                        llama_memory_seq_pos_max(llama_get_memory(ctx), seq) + 1,
                        llama_state_seq_get_size(ctx, seq)};
                });
            res.set_content(json{{"session_id", sid}, {"location", "vram"},
                                 {"cache_size", status.first}, {"state_bytes", status.second},
                                 {"load_ms", 0}}.dump(), "application/json");
            return;
        }
        auto clear_loading_reservation = [&] {
            std::lock_guard<std::mutex> lk(app.mu);
            auto parked = app.offloaded_sessions.find(sid_num);
            if (parked != app.offloaded_sessions.end()) parked->second.loading = false;
        };
        const int seq_id = app.scheduler->allocate_sequence();
        if (seq_id < 0) {
            clear_loading_reservation();
            res.status = 503;
            res.set_content(error_body("pooled sequence capacity exhausted", 503).dump(), "application/json");
            return;
        }
        const double t0 = now_s();
        try {
            app.scheduler->invoke_when_no_generation([&](llama_context * ctx) {
                if (llama_state_seq_set_data(
                        ctx, state.state.data(), state.state_bytes, seq_id) == 0) {
                    throw std::runtime_error("failed to restore offloaded sequence state");
                }
            });
        } catch (...) {
            app.scheduler->release_sequence(seq_id);
            clear_loading_reservation();
            throw;
        }
        {
            std::lock_guard<std::mutex> lk(app.mu);
            app.offloaded_sessions.erase(sid_num);
            app.sessions[sid_num] = LiveSession{seq_id, state.last_token, false};
        }
        const double elapsed_ms = (now_s() - t0) * 1000.0;
        res.set_content(json{{"session_id", sid}, {"location", "vram"},
                             {"cache_size", state.cache_size}, {"state_bytes", state.state_bytes},
                             {"load_ms", (int) elapsed_ms}}.dump(), "application/json");
    });

    svr.Delete(R"(/sessions/[^/]+)", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string prefix = "/sessions/";
        if (req.path.rfind(prefix, 0) != 0 || req.path.find('/', prefix.size()) != std::string::npos) {
            res.status = 404;
            res.set_content(error_body("not found", 404).dump(), "application/json");
            return;
        }
        const std::string sid = req.path.substr(prefix.size());
        const int64_t sid_num = parse_session_id_num(sid);
        std::optional<llama_seq_id> sequence;
        {
            std::lock_guard<std::mutex> lk(app.mu);
            auto live = app.sessions.find(sid_num);
            if (live != app.sessions.end()) {
                if (live->second.busy || live->second.readers > 0) {
                    res.status = 409;
                    res.set_content(error_body("session already has an active operation", 409).dump(),
                                    "application/json");
                    return;
                }
                sequence = live->second.seq_id;
                app.sessions.erase(live);
            } else {
                auto parked = app.offloaded_sessions.find(sid_num);
                if (parked == app.offloaded_sessions.end()) {
                    res.status = 404;
                    res.set_content(error_body("unknown session", 404).dump(), "application/json");
                    return;
                }
                if (parked->second.loading) {
                    res.status = 409;
                    res.set_content(error_body("session already has an active operation", 409).dump(),
                                    "application/json");
                    return;
                }
                app.offloaded_sessions.erase(parked);
            }
        }
        if (sequence) app.scheduler->release_sequence(*sequence);
        res.set_content(json{{"session_id", sid}, {"deleted", true}}.dump(), "application/json");
    });

    std::cerr << "multimodal-server listening on 0.0.0.0:" << cfg.port << "\n";
    if (!svr.listen("0.0.0.0", cfg.port)) {
        std::cerr << "error: failed to listen on port " << cfg.port << "\n";
        app.scheduler.reset();
        if (app.mtmd_ctx) mtmd_free(app.mtmd_ctx);
        llama_model_free(app.model);
        llama_backend_free();
        return 1;
    }

    app.scheduler.reset();  // frees the pooled context before the model
    if (app.mtmd_ctx) mtmd_free(app.mtmd_ctx);
    llama_model_free(app.model);
    llama_backend_free();
    return 0;
}
