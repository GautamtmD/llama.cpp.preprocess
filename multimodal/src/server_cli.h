// Strict, model-free command-line parsing and validation for multimodal-server.
// Kept separate from main.cpp so malformed user input is rejected before any
// backend/model initialization and the complete CLI contract is unit-testable.
#pragma once

#include "nlohmann/json.hpp"

#include <charconv>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <system_error>
#include <vector>

struct ServerConfig {
    std::string model_path;
    std::string mmproj_path;
    int port = 8080;
    int n_gpu_layers = 99;
    int ctx_size = 4096;
    int n_batch = 2048;
    int max_sequences = 8;
    bool allow_cpu = false;

    std::string chat_template;
    std::string chat_template_file;
    bool use_jinja = true;
    bool enable_chat_template = true;
    std::map<std::string, std::string> chat_template_kwargs;
    std::string system_prompt;
    std::string config_path;
};

struct ServerCliResult {
    ServerConfig config;
    bool show_help = false;
    std::string error;
};

inline std::string server_help_text() {
    return
        "multimodal-server [options]\n"
        "  -m, --model PATH          model gguf (required)\n"
        "      --mmproj PATH         multimodal projector gguf (enables image/audio)\n"
        "      --port N              listen port, 1-65535 (default 8080)\n"
        "  -ngl,--n-gpu-layers N     GPU layers; negative=all, 0=CPU (default 99)\n"
        "  -c, --ctx-size N          context tokens per sequence, >0 (default 4096)\n"
        "      --n-batch N           logical decode batch; must be >= --max-sequences (default 2048)\n"
        "      --max-sequences N     pooled live-sequence capacity, >0 (default 8)\n"
        "      --allow-cpu           explicitly permit CPU-only execution (default: GPU required)\n"
        "      --chat-template TPL   Jinja chat template override (else model default)\n"
        "      --chat-template-file F  read Jinja chat template override from a file\n"
        "      --jinja / --no-jinja  use the Jinja template engine (default: enabled)\n"
        "      --no-chat-template    disable templating: 'messages' inject is rejected\n"
        "      --system-prompt TEXT  system prompt prepended to every conversation\n"
        "      --chat-template-kwargs JSON  extra Jinja vars, e.g. '{\"k\":\"v\"}'\n"
        "      --config PATH         path to model config JSON file\n";
}

inline bool parse_server_integer(
    const std::string & option, const std::string & value, int & output,
    std::string & error) {
    int parsed = 0;
    const char * first = value.data();
    const char * last = first + value.size();
    const auto conversion = std::from_chars(first, last, parsed);
    if (value.empty() || conversion.ec != std::errc{} || conversion.ptr != last) {
        error = option + " must be an integer; received '" + value + "'";
        return false;
    }
    output = parsed;
    return true;
}

inline std::string validate_server_config(const ServerConfig & config) {
    if (config.model_path.empty()) {
        return "--model is required; provide a GGUF path (see --help)";
    }
    if (config.port < 1 || config.port > 65535) {
        return "--port must be between 1 and 65535; received " + std::to_string(config.port);
    }
    if (config.ctx_size <= 0) {
        return "--ctx-size must be positive; received " + std::to_string(config.ctx_size);
    }
    if (config.n_batch <= 0) {
        return "--n-batch must be positive; received " + std::to_string(config.n_batch);
    }
    if (config.max_sequences <= 0) {
        return "--max-sequences must be positive; received " +
               std::to_string(config.max_sequences);
    }
    if (config.n_batch < config.max_sequences) {
        return "--n-batch (" + std::to_string(config.n_batch) +
               ") must be at least --max-sequences (" +
               std::to_string(config.max_sequences) +
               ") so one scheduler decode cadence can include every live sequence; "
               "increase --n-batch or reduce --max-sequences";
    }
    const uint64_t pooled_context =
        static_cast<uint64_t>(config.ctx_size) * static_cast<uint64_t>(config.max_sequences);
    if (pooled_context > std::numeric_limits<uint32_t>::max()) {
        return "--ctx-size (" + std::to_string(config.ctx_size) +
               ") multiplied by --max-sequences (" +
               std::to_string(config.max_sequences) +
               ") exceeds the maximum pooled context size (" +
               std::to_string(std::numeric_limits<uint32_t>::max()) + ")";
    }
    return {};
}

inline ServerCliResult parse_server_arguments(const std::vector<std::string> & arguments) {
    ServerCliResult result;
    auto require_value = [&](size_t & index, const std::string & option,
                             const std::string & expected) -> const std::string * {
        if (index + 1 >= arguments.size()) {
            result.error = option + " requires " + expected;
            return nullptr;
        }
        return &arguments[++index];
    };
    auto parse_integer_option = [&](size_t & index, const std::string & option,
                                    int & destination) {
        const std::string * value = require_value(index, option, "an integer value");
        return value && parse_server_integer(option, *value, destination, result.error);
    };

    for (size_t i = 0; i < arguments.size(); ++i) {
        const std::string & option = arguments[i];
        if (option == "--help" || option == "-h") {
            result.show_help = true;
            return result;
        }
        if (option == "--allow-cpu") {
            result.config.allow_cpu = true;
        } else if (option == "--jinja") {
            result.config.use_jinja = true;
        } else if (option == "--no-jinja") {
            result.config.use_jinja = false;
        } else if (option == "--no-chat-template") {
            result.config.enable_chat_template = false;
        } else if (option == "--port") {
            if (!parse_integer_option(i, "--port", result.config.port)) return result;
        } else if (option == "--n-gpu-layers" || option == "-ngl") {
            if (!parse_integer_option(i, "--n-gpu-layers", result.config.n_gpu_layers)) return result;
        } else if (option == "--ctx-size" || option == "-c") {
            if (!parse_integer_option(i, "--ctx-size", result.config.ctx_size)) return result;
        } else if (option == "--n-batch") {
            if (!parse_integer_option(i, "--n-batch", result.config.n_batch)) return result;
        } else if (option == "--max-sequences") {
            if (!parse_integer_option(i, "--max-sequences", result.config.max_sequences)) return result;
        } else if (option == "--model" || option == "-m") {
            const std::string * value = require_value(i, "--model", "a path");
            if (!value) return result;
            result.config.model_path = *value;
        } else if (option == "--mmproj") {
            const std::string * value = require_value(i, "--mmproj", "a path");
            if (!value) return result;
            result.config.mmproj_path = *value;
        } else if (option == "--chat-template") {
            const std::string * value = require_value(i, "--chat-template", "a value");
            if (!value) return result;
            result.config.chat_template = *value;
        } else if (option == "--chat-template-file") {
            const std::string * value = require_value(i, "--chat-template-file", "a path");
            if (!value) return result;
            result.config.chat_template_file = *value;
        } else if (option == "--system-prompt") {
            const std::string * value = require_value(i, "--system-prompt", "a value");
            if (!value) return result;
            result.config.system_prompt = *value;
        } else if (option == "--config") {
            const std::string * value = require_value(i, "--config", "a path");
            if (!value) return result;
            result.config.config_path = *value;
        } else if (option == "--chat-template-kwargs") {
            const std::string * value =
                require_value(i, "--chat-template-kwargs", "a JSON object value");
            if (!value) return result;
            try {
                const nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(*value);
                if (!parsed.is_object()) {
                    result.error = "--chat-template-kwargs must be a JSON object";
                    return result;
                }
                for (auto item = parsed.begin(); item != parsed.end(); ++item) {
                    result.config.chat_template_kwargs[item.key()] = item.value().dump();
                }
            } catch (const std::exception & exception) {
                result.error = std::string("--chat-template-kwargs must be valid JSON: ") +
                               exception.what();
                return result;
            }
        } else {
            result.error = "unrecognized argument '" + option + "'; run with --help to list supported options";
            return result;
        }
    }

    result.error = validate_server_config(result.config);
    return result;
}

inline ServerCliResult parse_server_arguments(int argc, char ** argv) {
    std::vector<std::string> arguments;
    arguments.reserve(argc > 1 ? static_cast<size_t>(argc - 1) : 0);
    for (int i = 1; i < argc; ++i) arguments.emplace_back(argv[i]);
    return parse_server_arguments(arguments);
}
