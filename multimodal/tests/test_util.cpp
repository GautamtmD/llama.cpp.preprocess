// Unit tests for the pure multimodal-server helpers (util.h).
// Dependency-free: header-only against util.h + the vendored nlohmann json.
// Runs with the build, no model/GPU/projector, no link to llama/mtmd.
//
// Built by the `multimodal-util-tests` CMake target and run by build_engine.sh.
// Exits non-zero on any check failure (so it can gate the build).

#include "util.h"

#include <iostream>
#include <string>

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "FAIL [line " << __LINE__ << "]: " #cond << "\n";     \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        auto _va = (a);                                                        \
        auto _vb = (b);                                                        \
        if (!(_va == _vb)) {                                                   \
            std::cerr << "FAIL [line " << __LINE__ << "]: " #a " == " #b        \
                      << "\n  got=" << _va << "\n  exp=" << _vb << "\n";       \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static void test_base64_decode() {
    CHECK_EQ(base64_decode(""), std::string(""));
    CHECK_EQ(base64_decode("aGVsbG8="), std::string("hello"));
    CHECK_EQ(base64_decode("aGVsbG8gd29ybGQ="), std::string("hello world"));
    // padding optional / ignored
    CHECK_EQ(base64_decode("aGVsbG8"), std::string("hello"));
    // whitespace ignored
    CHECK_EQ(base64_decode("a G V s b G 8 ="), std::string("hello"));
    // invalid characters skipped (not in alphabet)
    CHECK_EQ(base64_decode("aGVsbG8!@#$"), std::string("hello"));
    // binary round-trip (bytes 0..255 incl. nulls)
    std::string bin;
    for (int i = 0; i < 256; ++i) bin += char(i);
    // manual encode of `bin` (we only ship a decoder; encode inline here)
    static const char * A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string enc;
    int val = 0, bits = 0;
    for (unsigned char c : bin) {
        val = (val << 8) | c;
        bits += 8;
        while (bits >= 6) {
            bits -= 6;
            enc += A[(val >> bits) & 0x3F];
        }
    }
    if (bits) enc += A[(val << (6 - bits)) & 0x3F];
    CHECK_EQ(base64_decode(enc), bin);
}

static void test_make_session_id() {
    CHECK_EQ(make_session_id(0), std::string("s_0"));
    CHECK_EQ(make_session_id(1), std::string("s_1"));
    CHECK_EQ(make_session_id(12345), std::string("s_12345"));
    CHECK_EQ(make_session_id(9223372036854775807LL), std::string("s_9223372036854775807"));
}

static void test_extract_session_id() {
    using namespace std::string_literals;
    CHECK_EQ(extract_session_id("/sessions/s_1/generate", "generate"), "s_1"s);
    CHECK_EQ(extract_session_id("/sessions/s_42/inject", "inject"), "s_42"s);
    CHECK_EQ(extract_session_id("/sessions/anything/generate", "generate"), "anything"s);
    // action mismatch -> empty
    CHECK_EQ(extract_session_id("/sessions/s_1/generate", "inject"), ""s);
    // wrong prefix
    CHECK_EQ(extract_session_id("/sessionsX/s_1/generate", "generate"), ""s);
    CHECK_EQ(extract_session_id("/health", "generate"), ""s);
    // no trailing action
    CHECK_EQ(extract_session_id("/sessions/s_1", "generate"), ""s);
    CHECK_EQ(extract_session_id("/sessions/s_1/", "generate"), ""s);
    // empty id segment
    CHECK_EQ(extract_session_id("/sessions//generate", "generate"), ""s);
}

static void test_parse_session_id_num() {
    CHECK(parse_session_id_num("s_1") == 1);
    CHECK(parse_session_id_num("s_0") == 0);
    CHECK(parse_session_id_num("s_12345") == 12345);
    // missing prefix
    CHECK(parse_session_id_num("1") == -1);
    CHECK(parse_session_id_num("xyz") == -1);
    // empty number
    CHECK(parse_session_id_num("s_") == -1);
    // overflow
    CHECK(parse_session_id_num("s_9999999999999999999999") == -1);
}

static void test_error_body() {
    auto j = error_body("oops", 400);
    CHECK(j.value("error", "") == "oops");
    CHECK(j.value("code", 0) == 400);
    // ordered_json preserves key order: error before code
    const std::string s = j.dump();
    auto perr = s.find("\"error\"");
    auto pcode = s.find("\"code\"");
    CHECK(perr != std::string::npos && pcode != std::string::npos && perr < pcode);
}

static void test_sse_event() {
    nlohmann::ordered_json j{{"token", "hi"}, {"id", 7}};
    const std::string s = sse_event(j);
    CHECK(s.rfind("data: ", 0) == 0);          // starts with "data: "
    CHECK(s.size() >= 2 && s.substr(s.size() - 2) == "\n\n");  // ends with "\n\n"
    // payload is the json dump between the prefix and the trailing newlines
    const std::string payload = s.substr(6, s.size() - 6 - 2);
    CHECK(payload == j.dump());
    CHECK(payload.find("\"token\"") != std::string::npos);
    CHECK(payload.find("\"id\"") != std::string::npos);
}

static void test_model_config() {
    // Default constructor
    ModelConfig c1;
    CHECK_EQ(c1.audio_frame_size, 640);
    CHECK(std::abs(c1.temperature - 0.8f) < 1e-5f);
    CHECK(std::abs(c1.top_p - 0.95f) < 1e-5f);
    CHECK_EQ(c1.top_k, 40);
    CHECK(std::abs(c1.min_p - 0.05f) < 1e-5f);

    // Parse empty json
    nlohmann::ordered_json j_empty = nlohmann::ordered_json::object();
    ModelConfig c2 = ModelConfig::from_json(j_empty);
    CHECK_EQ(c2.audio_frame_size, 640);

    // Parse custom values
    nlohmann::ordered_json j_custom = nlohmann::ordered_json::parse(
        "{\"audio_frame_size\": 160, \"sampler\": {\"temperature\": 0.5, \"top_k\": 20}}"
    );
    ModelConfig c3 = ModelConfig::from_json(j_custom);
    CHECK_EQ(c3.audio_frame_size, 160);
    CHECK(std::abs(c3.temperature - 0.5f) < 1e-5f);
    CHECK(std::abs(c3.top_p - 0.95f) < 1e-5f);
    CHECK_EQ(c3.top_k, 20);

    // Parse flat layout
    nlohmann::ordered_json j_flat = nlohmann::ordered_json::parse(
        "{\"audio_frame_size\": 1, \"temperature\": 0.1, \"top_p\": 0.9, \"top_k\": 10, \"min_p\": 0.01}"
    );
    ModelConfig c4 = ModelConfig::from_json(j_flat);
    CHECK_EQ(c4.audio_frame_size, 1);
    CHECK(std::abs(c4.temperature - 0.1f) < 1e-5f);
    CHECK(std::abs(c4.top_p - 0.9f) < 1e-5f);
    CHECK_EQ(c4.top_k, 10);
    CHECK(std::abs(c4.min_p - 0.01f) < 1e-5f);
}

int main() {
    test_base64_decode();
    test_make_session_id();
    test_extract_session_id();
    test_parse_session_id_num();
    test_error_body();
    test_sse_event();
    test_model_config();

    if (g_failures) {
        std::cerr << g_failures << " util test check(s) FAILED\n";
        return 1;
    }
    std::cout << "all util tests passed\n";
    return 0;
}
