// Pure helpers for multimodal-server, extracted so they can be unit-tested
// without a model/GPU/projector. Header-only; depends only on the vendored
// nlohmann json header. Signatures use nlohmann::ordered_json directly to
// avoid colliding with main.cpp's `json` alias (they are the same type).
//
// Keep these functions model-free and side-effect-free so test_util.cpp can
// exercise them in isolation.

#pragma once

#include "execution_policy.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// RFC4648 base64 decode (no URL-safe; ignores whitespace and '='; skips any
// non-alphabet character).
inline std::string base64_decode(const std::string & s) {
    static int8_t tbl[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) tbl[i] = -1;
        const char * chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) tbl[(unsigned char)chars[i]] = (int8_t)i;
        init = true;
    }
    std::string out;
    out.reserve(s.size() * 3 / 4);
    int val = 0, bits = 0;
    for (char c : s) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        int8_t d = tbl[(unsigned char)c];
        if (d < 0) continue;  // skip invalid
        val = (val << 6) | d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += char((val >> bits) & 0xFF);
        }
    }
    return out;
}

// int64 session number -> "s_<n>"
inline std::string make_session_id(int64_t n) {
    std::ostringstream os;
    os << "s_" << n;
    return os.str();
}

// Parse the {id} out of a "/sessions/{id}/<action>" path. Returns "" on miss
// or if the trailing action does not match.
inline std::string extract_session_id(const std::string & path, const std::string & action) {
    const std::string prefix = "/sessions/";
    if (path.rfind(prefix, 0) != 0) return "";
    const size_t rest = path.find('/', prefix.size());
    if (rest == std::string::npos) return "";
    const std::string id = path.substr(prefix.size(), rest - prefix.size());
    const std::string act = path.substr(rest);
    if (act != "/" + action) return "";
    return id;
}

// Canonical "s_<n>" -> n, or -1 on bad format/overflow. Only ASCII decimal
// digits are accepted; zero is "s_0", while positive values have no leading zero.
inline int64_t parse_session_id_num(const std::string & id) {
    constexpr std::string_view prefix = "s_";
    if (id.size() <= prefix.size() || id.compare(0, prefix.size(), prefix) != 0) return -1;

    const std::string_view digits{id.data() + prefix.size(), id.size() - prefix.size()};
    if (digits.size() > 1 && digits.front() == '0') return -1;

    int64_t value = 0;
    for (const char c : digits) {
        if (c < '0' || c > '9') return -1;
        const int64_t digit = c - '0';
        if (value > (std::numeric_limits<int64_t>::max() - digit) / 10) return -1;
        value = value * 10 + digit;
    }
    return value;
}

// Standard JSON error body: {"error": msg, "code": code}.
inline nlohmann::ordered_json error_body(const std::string & msg, int code) {
    return nlohmann::ordered_json{{"error", msg}, {"code", code}};
}

// Validate the model-free shape and precedence rules for /generate constraints.
// Grammar syntax and JSON-schema conversion require the loaded model/vocabulary
// and are preflighted separately before response headers are committed.
inline std::string validate_generation_constraints(
    const nlohmann::ordered_json & response_format,
    const std::string & grammar,
    bool tools_active,
    const std::vector<std::string> & stops = {}) {
    if (std::any_of(stops.begin(), stops.end(), [](const std::string & stop) {
            return stop.empty();
        })) {
        return "stop sequences must not be empty";
    }
    if (response_format.is_null()) return "";
    if (!response_format.is_object()) {
        return "response_format must be an object";
    }
    const auto type = response_format.find("type");
    if (type == response_format.end() || !type->is_string()) {
        return "response_format.type must be a string";
    }
    const std::string value = type->get<std::string>();
    if (value != "json_object" && value != "json_schema" && value != "text") {
        return "invalid response_format.type (expected json_object, json_schema, or text)";
    }
    if (value == "json_object") {
        const auto schema = response_format.find("schema");
        if (schema != response_format.end() && !schema->is_object()) {
            return "response_format.schema must be an object";
        }
    }
    if (value == "json_schema") {
        const auto wrapper = response_format.find("json_schema");
        if (wrapper == response_format.end() || !wrapper->is_object()) {
            return "response_format.json_schema must be an object";
        }
        const auto schema = wrapper->find("schema");
        if (schema == wrapper->end() || !schema->is_object()) {
            return "response_format.json_schema.schema must be an object";
        }
    }
    if (!tools_active && !grammar.empty()) {
        return "cannot specify both response_format and grammar";
    }
    return "";
}

struct StopMatchResult {
    bool matched = false;
    bool emission_failed = false;
};

// Incremental byte matcher shared by JSON and SSE generation. It retains only a
// suffix that could still complete a stop sequence, so stop bytes never reach
// the response/parser. Empty stops are ignored defensively; request validation
// rejects them before this path.
class StopSequenceMatcher {
public:
    template <typename Emit>
    StopMatchResult append(
        std::string_view text,
        const std::vector<std::string> & stops,
        Emit && emit) {
        if (matched_) return {true, false};
        buffer_.append(text.data(), text.size());

        size_t earliest = std::string::npos;
        for (const std::string & stop : stops) {
            if (stop.empty()) continue;
            earliest = std::min(earliest, buffer_.find(stop));
        }
        if (earliest != std::string::npos) {
            bool emitted = true;
            if (earliest > 0) {
                emitted = emit(std::string_view(buffer_.data(), earliest));
            }
            buffer_.clear();
            matched_ = true;
            return {true, !emitted};
        }

        size_t hold = 0;
        for (const std::string & stop : stops) {
            if (stop.empty()) continue;
            const size_t maximum =
                std::min(buffer_.size(), stop.size() - 1);
            for (size_t length = maximum; length > hold; --length) {
                if (buffer_.compare(
                        buffer_.size() - length, length,
                        stop, 0, length) == 0) {
                    hold = length;
                    break;
                }
            }
        }

        const size_t emit_length = buffer_.size() - hold;
        bool emitted = true;
        if (emit_length > 0) {
            emitted = emit(std::string_view(buffer_.data(), emit_length));
            buffer_.erase(0, emit_length);
        }
        return {false, !emitted};
    }

    template <typename Emit>
    StopMatchResult finish(Emit && emit) {
        if (matched_ || buffer_.empty()) return {matched_, false};
        const bool emitted = emit(std::string_view(buffer_));
        buffer_.clear();
        return {false, !emitted};
    }

    bool matched() const {
        return matched_;
    }

private:
    std::string buffer_;
    bool matched_ = false;
};

// SSE event frame: "data: <json>\n\n".
inline std::string sse_event(const nlohmann::ordered_json & j) {
    std::ostringstream os;
    os << "data: " << j.dump() << "\n\n";
    return os.str();
}

inline bool has_builtin_gemma4_reasoning(std::string_view architecture) {
    return architecture == "gemma4";
}

struct ReasoningConfig {
    static constexpr int32_t unset_budget = -2;

    std::string start_marker;
    std::string end_marker;
    int32_t none_budget = unset_budget;
    int32_t minimal_budget = unset_budget;
    int32_t low_budget = unset_budget;
    int32_t medium_budget = unset_budget;
    int32_t high_budget = unset_budget;

    bool complete() const {
        const bool ordered_finite =
            none_budget == 0 && minimal_budget >= 0 &&
            minimal_budget < low_budget && low_budget < medium_budget;
        const bool valid_high =
            high_budget == -1 || high_budget > medium_budget;
        return !start_marker.empty() && !end_marker.empty() &&
               ordered_finite && valid_high;
    }

    std::optional<int32_t> budget_for(std::string_view effort) const {
        if (effort == "none") return none_budget;
        if (effort == "minimal") return minimal_budget;
        if (effort == "low") return low_budget;
        if (effort == "medium") return medium_budget;
        if (effort == "high") return high_budget;
        return std::nullopt;
    }
};

struct ModelConfig {
    int32_t audio_frame_size = 640;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int32_t top_k = 40;
    float min_p = 0.05f;
    ReasoningConfig reasoning;
    // Declared modalities (text/image/audio in; text out). When empty, the
    // server infers them at runtime from the loaded projector (backward compat).
    // When authored (model_config.json `modalities` block), they are the source
    // of truth — surfaced and reconciled via GET /info so tests can select by
    // capability (e.g. skip audio tests on a vision+text model).
    std::vector<std::string> input_modalities;
    std::vector<std::string> output_modalities;

    void enable_gemma4_reasoning() {
        reasoning.start_marker = "<|channel>thought";
        reasoning.end_marker = "<channel|>";
        reasoning.none_budget = 0;
        reasoning.minimal_budget = 64;
        reasoning.low_budget = 256;
        reasoning.medium_budget = 1024;
        reasoning.high_budget = -1;
    }

    static ModelConfig from_json(const nlohmann::ordered_json & j) {
        ModelConfig cfg;
        if (j.contains("audio_frame_size")) {
            cfg.audio_frame_size = j["audio_frame_size"].get<int32_t>();
        }
        if (j.contains("sampler")) {
            auto s = j["sampler"];
            if (s.is_object()) {
                if (s.contains("temperature")) cfg.temperature = s["temperature"].get<float>();
                if (s.contains("top_p"))       cfg.top_p       = s["top_p"].get<float>();
                if (s.contains("top_k"))       cfg.top_k       = s["top_k"].get<int32_t>();
                if (s.contains("min_p"))       cfg.min_p       = s["min_p"].get<float>();
            }
        }
        if (j.contains("temperature")) cfg.temperature = j["temperature"].get<float>();
        if (j.contains("top_p"))       cfg.top_p       = j["top_p"].get<float>();
        if (j.contains("top_k"))       cfg.top_k       = j["top_k"].get<int32_t>();
        if (j.contains("min_p"))       cfg.min_p       = j["min_p"].get<float>();
        if (j.contains("reasoning") && j["reasoning"].is_object()) {
            const auto & r = j["reasoning"];
            if (r.contains("start_marker")) {
                cfg.reasoning.start_marker = r["start_marker"].get<std::string>();
            }
            if (r.contains("end_marker")) {
                cfg.reasoning.end_marker = r["end_marker"].get<std::string>();
            }
            if (r.contains("effort_budgets") && r["effort_budgets"].is_object()) {
                const auto & budgets = r["effort_budgets"];
                if (budgets.contains("none")) {
                    cfg.reasoning.none_budget = budgets["none"].get<int32_t>();
                }
                if (budgets.contains("minimal")) {
                    cfg.reasoning.minimal_budget = budgets["minimal"].get<int32_t>();
                }
                if (budgets.contains("low")) {
                    cfg.reasoning.low_budget = budgets["low"].get<int32_t>();
                }
                if (budgets.contains("medium")) {
                    cfg.reasoning.medium_budget = budgets["medium"].get<int32_t>();
                }
                if (budgets.contains("high")) {
                    cfg.reasoning.high_budget = budgets["high"].get<int32_t>();
                }
            }
        }
        if (j.contains("modalities")) {
            const auto & m = j["modalities"];
            if (m.is_object()) {
                if (m.contains("input") && m["input"].is_array()) {
                    for (const auto & x : m["input"])
                        if (x.is_string()) cfg.input_modalities.push_back(x.get<std::string>());
                }
                if (m.contains("output") && m["output"].is_array()) {
                    for (const auto & x : m["output"])
                        if (x.is_string()) cfg.output_modalities.push_back(x.get<std::string>());
                }
            }
        }
        return cfg;
    }
};

struct ReasoningEffortResolution {
    bool supplied = false;
    int32_t budget_tokens = -1;
    std::string error;
};

inline ReasoningEffortResolution resolve_reasoning_effort(
    const nlohmann::ordered_json & request,
    const ModelConfig & config) {
    const auto value = request.find("reasoning_effort");
    if (value == request.end()) return {};

    ReasoningEffortResolution result;
    result.supplied = true;
    if (!value->is_string()) {
        result.error = "reasoning_effort must be a string";
        return result;
    }

    const std::string & effort = value->get_ref<const std::string &>();
    const auto budget = config.reasoning.budget_for(effort);
    if (!budget.has_value()) {
        result.error =
            "unsupported reasoning_effort '" + effort +
            "' (expected none, minimal, low, medium, or high)";
        return result;
    }
    if (!config.reasoning.complete()) {
        result.error =
            "loaded model does not support controllable reasoning; configure "
            "reasoning start_marker, end_marker, and all effort_budgets";
        return result;
    }
    result.budget_tokens = *budget;
    return result;
}
