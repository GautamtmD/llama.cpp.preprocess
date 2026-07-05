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
