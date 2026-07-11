// Pure helpers for multimodal-server, extracted so they can be unit-tested
// without a model/GPU/projector. Header-only; depends only on the vendored
// nlohmann json header. Signatures use nlohmann::ordered_json directly to
// avoid colliding with main.cpp's `json` alias (they are the same type).
//
// Keep these functions model-free and side-effect-free so test_util.cpp can
// exercise them in isolation.

#pragma once

#include "nlohmann/json.hpp"

#include <cstdint>
#include <sstream>
#include <string>
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

// "s_<n>" -> n, or -1 on bad format/overflow.
inline int64_t parse_session_id_num(const std::string & id) {
    const std::string pfx = "s_";
    if (id.rfind(pfx, 0) != 0) return -1;
    try { return std::stoll(id.substr(pfx.size())); }
    catch (...) { return -1; }
}

// Standard JSON error body: {"error": msg, "code": code}.
inline nlohmann::ordered_json error_body(const std::string & msg, int code) {
    return nlohmann::ordered_json{{"error", msg}, {"code", code}};
}

// SSE event frame: "data: <json>\n\n".
inline std::string sse_event(const nlohmann::ordered_json & j) {
    std::ostringstream os;
    os << "data: " << j.dump() << "\n\n";
    return os.str();
}

struct ModelConfig {
    int32_t audio_frame_size = 640;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int32_t top_k = 40;
    float min_p = 0.05f;
    // Declared modalities (text/image/audio in; text out). When empty, the
    // server infers them at runtime from the loaded projector (backward compat).
    // When authored (model_config.json `modalities` block), they are the source
    // of truth — surfaced and reconciled via GET /info so tests can select by
    // capability (e.g. skip audio tests on a vision+text model).
    std::vector<std::string> input_modalities;
    std::vector<std::string> output_modalities;

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
