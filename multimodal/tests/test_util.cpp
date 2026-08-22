// Unit tests for the pure multimodal-server helpers (util.h).
// Dependency-free: header-only against util.h + the vendored nlohmann json.
// Runs with the build, no model/GPU/projector, no link to llama/mtmd.
//
// Built by the `multimodal-util-tests` CMake target and run by build_engine.sh.
// Exits non-zero on any check failure (so it can gate the build).

#include "util.h"
#include "server_cli.h"
#include <chrono>

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
    CHECK(parse_session_id_num("s_0") == 0);
    CHECK(parse_session_id_num("s_1") == 1);
    CHECK(parse_session_id_num("s_12345") == 12345);
    CHECK(parse_session_id_num("s_9223372036854775807") == 9223372036854775807LL);

    const std::vector<std::string> malformed = {
        "",
        "1",
        "xyz",
        "s_",
        "s_00",
        "s_01",
        "s_1junk",
        "s_+1",
        "s_-1",
        "s_ 1",
        "s_1 ",
        "s_\t1",
        "s_9223372036854775808",
        "s_9999999999999999999999",
    };
    for (const auto & id : malformed) {
        CHECK(parse_session_id_num(id) == -1);
    }
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

static void test_gpu_execution_policy() {
    CHECK_EQ(gpu_allocation_delta(/*free_before*/ 1000, /*free_after*/ 400), 600u);
    CHECK_EQ(gpu_allocation_delta(/*free_before*/ 400, /*free_after*/ 1000), 0u);
    CHECK(gpu_execution_allowed(/*allow_cpu*/ false, /*actual_gpu_model_bytes*/ 1));
    CHECK(gpu_execution_allowed(/*allow_cpu*/ true,  /*actual_gpu_model_bytes*/ 0));
    CHECK(!gpu_execution_allowed(/*allow_cpu*/ false, /*actual_gpu_model_bytes*/ 0));

    const std::string server_error = gpu_execution_error("multimodal-server");
    CHECK(server_error.find("GPU load was not successful") != std::string::npos);
    CHECK(server_error.find("multimodal-server cannot run") != std::string::npos);
    CHECK(server_error.find("--allow-cpu") != std::string::npos);

    const std::string bench_error = gpu_execution_error("benchmark");
    CHECK(bench_error.find("GPU load was not successful") != std::string::npos);
    CHECK(bench_error.find("benchmark cannot run") != std::string::npos);
    CHECK(bench_error.find("--allow-cpu") != std::string::npos);
}

static void test_builtin_reasoning_architecture_detection() {
    CHECK(has_builtin_gemma4_reasoning("gemma4"));
    CHECK(!has_builtin_gemma4_reasoning(""));
    CHECK(!has_builtin_gemma4_reasoning("gemma"));
    CHECK(!has_builtin_gemma4_reasoning("gemma2"));
    CHECK(!has_builtin_gemma4_reasoning("gemma3"));
    CHECK(!has_builtin_gemma4_reasoning("gemma3n"));
    CHECK(!has_builtin_gemma4_reasoning("gemma4-assistant"));
    CHECK(!has_builtin_gemma4_reasoning("Gemma4"));
    CHECK(!has_builtin_gemma4_reasoning("custom-gemma4"));
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

static void test_reasoning_effort_resolution_and_latency() {
    using json = nlohmann::ordered_json;

    ModelConfig gemma;
    gemma.enable_gemma4_reasoning();
    CHECK(gemma.reasoning.complete());
    CHECK_EQ(gemma.reasoning.start_marker, std::string("<|channel>thought"));
    CHECK_EQ(gemma.reasoning.end_marker, std::string("<channel|>"));

    const struct {
        const char * effort;
        int32_t budget;
    } expected[] = {
        {"none", 0},
        {"minimal", 64},
        {"low", 256},
        {"medium", 1024},
        {"high", -1},
    };
    for (const auto & item : expected) {
        const auto resolved = resolve_reasoning_effort(
            json{{"reasoning_effort", item.effort}}, gemma);
        CHECK(resolved.supplied);
        CHECK(resolved.error.empty());
        CHECK_EQ(resolved.budget_tokens, item.budget);
    }

    const auto omitted = resolve_reasoning_effort(json::object(), gemma);
    CHECK(!omitted.supplied);
    CHECK(omitted.error.empty());

    const auto non_string =
        resolve_reasoning_effort(json{{"reasoning_effort", 1}}, gemma);
    CHECK(non_string.error.find("must be a string") != std::string::npos);
    const auto unknown =
        resolve_reasoning_effort(json{{"reasoning_effort", "extreme"}}, gemma);
    CHECK(unknown.error.find("unsupported reasoning_effort") != std::string::npos);

    ModelConfig unsupported;
    const auto unsupported_result =
        resolve_reasoning_effort(json{{"reasoning_effort", "none"}}, unsupported);
    CHECK(unsupported_result.error.find("does not support controllable reasoning") !=
          std::string::npos);

    const auto parsed = ModelConfig::from_json(json{
        {"reasoning", {
            {"start_marker", "<think>"},
            {"end_marker", "</think>"},
            {"effort_budgets", {
                {"none", 0},
                {"minimal", 8},
                {"low", 16},
                {"medium", 32},
                {"high", 64},
            }},
        }},
    });
    CHECK(parsed.reasoning.complete());
    CHECK_EQ(resolve_reasoning_effort(
                 json{{"reasoning_effort", "high"}}, parsed).budget_tokens,
             64);

    ModelConfig incomplete = parsed;
    incomplete.reasoning.end_marker.clear();
    CHECK(!incomplete.reasoning.complete());
    CHECK(!resolve_reasoning_effort(
               json{{"reasoning_effort", "low"}}, incomplete).error.empty());

    constexpr int iterations = 100000;
    const json request{{"reasoning_effort", "none"}};
    const auto started = std::chrono::steady_clock::now();
    int64_t budget_sum = 0;
    for (int i = 0; i < iterations; ++i) {
        budget_sum += resolve_reasoning_effort(request, gemma).budget_tokens;
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    CHECK_EQ(budget_sum, 0);
    CHECK(elapsed_ms < 500.0);
}

static void test_generation_constraint_validation() {
    using json = nlohmann::ordered_json;

    CHECK(validate_generation_constraints(json(), "", false).empty());
    CHECK(validate_generation_constraints(json{{"type", "text"}}, "", false).empty());
    CHECK(validate_generation_constraints(
              json{{"type", "json_object"}}, "", false).empty());
    CHECK(validate_generation_constraints(
              json{{"type", "json_object"}, {"schema", json::object()}}, "", false).empty());
    CHECK(validate_generation_constraints(
              json{{"type", "json_schema"},
                   {"json_schema", json{{"schema", json::object()}}}},
              "", false).empty());
    CHECK(validate_generation_constraints(
              json{{"type", "json_object"}}, R"(root ::= "x")", true).empty());
    CHECK(validate_generation_constraints(
              json(), "", false, std::vector<std::string>{"valid"}).empty());
    CHECK(validate_generation_constraints(
              json(), "", false, std::vector<std::string>{""}).find(
              "stop sequences must not be empty") != std::string::npos);

    CHECK(validate_generation_constraints(json::array(), "", false).find(
              "response_format must be an object") != std::string::npos);
    CHECK(validate_generation_constraints(json::object(), "", false).find(
              "response_format.type must be a string") != std::string::npos);
    CHECK(validate_generation_constraints(json{{"type", "yaml"}}, "", false).find(
              "invalid response_format.type") != std::string::npos);
    CHECK(validate_generation_constraints(
              json{{"type", "json_object"}, {"schema", json::array()}}, "", false).find(
              "response_format.schema must be an object") != std::string::npos);
    CHECK(validate_generation_constraints(
              json{{"type", "json_schema"}, {"json_schema", json::array()}}, "", false).find(
              "response_format.json_schema must be an object") != std::string::npos);
    CHECK(validate_generation_constraints(
              json{{"type", "json_schema"}, {"json_schema", json::object()}}, "", false).find(
              "response_format.json_schema.schema must be an object") != std::string::npos);
    CHECK(validate_generation_constraints(
              json{{"type", "json_object"}}, R"(root ::= "x")", false).find(
              "cannot specify both response_format and grammar") != std::string::npos);
}

static void test_stop_sequence_matcher() {
    std::string emitted;
    auto emit = [&emitted](std::string_view text) {
        emitted.append(text);
        return true;
    };

    StopSequenceMatcher spanning;
    const std::vector<std::string> end_stop{"END"};
    auto first = spanning.append("prefix E", end_stop, emit);
    CHECK(!first.matched);
    CHECK(!first.emission_failed);
    CHECK_EQ(emitted, std::string("prefix "));
    auto second = spanning.append("ND hidden", end_stop, emit);
    CHECK(second.matched);
    CHECK(!second.emission_failed);
    CHECK_EQ(emitted, std::string("prefix "));

    emitted.clear();
    StopSequenceMatcher earliest;
    const std::vector<std::string> ordered_late_first{"LATE", "EARLY"};
    auto ordered = earliest.append(
        "visible EARLY hidden LATE", ordered_late_first, emit);
    CHECK(ordered.matched);
    CHECK_EQ(emitted, std::string("visible "));

    emitted.clear();
    StopSequenceMatcher defensive_empty;
    const std::vector<std::string> empty_and_real{"", "STOP"};
    auto safe = defensive_empty.append("ordinary text", empty_and_real, emit);
    CHECK(!safe.matched);
    CHECK(!safe.emission_failed);
    defensive_empty.finish(emit);
    CHECK_EQ(emitted, std::string("ordinary text"));
}


static void test_server_argument_validation() {
    auto expect_error = [](std::vector<std::string> args, const std::string & expected) {
        const ServerCliResult result = parse_server_arguments(args);
        CHECK(!result.error.empty());
        CHECK(result.error.find(expected) != std::string::npos);
    };

    const ServerCliResult valid = parse_server_arguments({
        "--model", "model.gguf",
        "--port", "1234",
        "--n-gpu-layers", "-1",
        "--ctx-size", "1024",
        "--n-batch", "8",
        "--max-sequences", "8",
        "--chat-template-kwargs", R"({"custom":"value"})",
    });
    CHECK(valid.error.empty());
    CHECK(!valid.show_help);
    CHECK_EQ(valid.config.port, 1234);
    CHECK_EQ(valid.config.n_gpu_layers, -1);
    CHECK_EQ(valid.config.ctx_size, 1024);
    CHECK_EQ(valid.config.n_batch, 8);
    CHECK_EQ(valid.config.max_sequences, 8);
    CHECK_EQ(valid.config.chat_template_kwargs.at("custom"), "\"value\"");

    const ServerCliResult help = parse_server_arguments({"--help"});
    CHECK(help.error.empty());
    CHECK(help.show_help);

    expect_error({}, "--model is required");
    expect_error({"--model"}, "--model requires a path");
    expect_error({"--model", "model.gguf", "--unknown"}, "unrecognized argument '--unknown'");
    expect_error({"--model", "model.gguf", "--port", "12x"}, "--port must be an integer");
    expect_error(
        {"--model", "model.gguf", "--ctx-size", "999999999999999999999"},
        "--ctx-size must be an integer");
    expect_error({"--model", "model.gguf", "--ctx-size", "0"}, "--ctx-size must be positive");
    expect_error({"--model", "model.gguf", "--n-batch", "0"}, "--n-batch must be positive");
    expect_error(
        {"--model", "model.gguf", "--max-sequences", "0"},
        "--max-sequences must be positive");
    expect_error(
        {"--model", "model.gguf", "--n-batch", "3", "--max-sequences", "4"},
        "--n-batch (3) must be at least --max-sequences (4)");
    expect_error(
        {"--model", "model.gguf", "--ctx-size", "2147483647", "--max-sequences", "3",
         "--n-batch", "3"},
        "exceeds the maximum pooled context size");
    expect_error(
        {"--model", "model.gguf", "--chat-template-kwargs", "[]"},
        "--chat-template-kwargs must be a JSON object");
}

int main() {
    test_base64_decode();
    test_make_session_id();
    test_extract_session_id();
    test_parse_session_id_num();
    test_error_body();
    test_sse_event();
    test_gpu_execution_policy();
    test_builtin_reasoning_architecture_detection();
    test_server_argument_validation();
    test_model_config();
    test_generation_constraint_validation();
    test_reasoning_effort_resolution_and_latency();
    test_stop_sequence_matcher();

    if (g_failures) {
        std::cerr << g_failures << " util test check(s) FAILED\n";
        return 1;
    }
    std::cout << "all util tests passed\n";
    return 0;
}
