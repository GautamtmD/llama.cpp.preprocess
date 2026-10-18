// Pure helpers for the ADR 0010 decision readout on /sessions/{id}/inject:
// request validation and raw-logit statistics over the last decoded row.
// Header-only and model-free (token ids are int32_t, the tokenizer is a
// callback) so test_util.cpp can exercise them without llama/mtmd.

#pragma once

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

inline constexpr int kReadoutMaxCandidates = 64;
inline constexpr int kReadoutMaxTopK = 16;
inline constexpr const char * kReadoutSemantics = "raw_last_row_pre_sampler_v1";

struct ReadoutRequest {
    std::vector<std::string> texts;   // candidate strings as supplied
    std::vector<int32_t> tokens;      // their single token ids (same order)
    int top_k = 0;
};

// Tokenizes `text` with add_special=false and parse_special=true. Returns false
// when the tokenizer itself fails.
using ReadoutTokenizer = std::function<bool(const std::string &, std::vector<int32_t> &)>;

// Validates a `readout` request object. Returns "" on success, otherwise a
// client-error message (HTTP 400). Never touches session state.
inline std::string parse_readout_request(const nlohmann::ordered_json & value,
                                         const ReadoutTokenizer & tokenize,
                                         ReadoutRequest & out) {
    out = ReadoutRequest{};
    if (!value.is_object()) return "readout must be an object";
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (it.key() != "candidates" && it.key() != "top_k") {
            return "readout: unknown key '" + it.key() + "'";
        }
    }
    if (!value.contains("candidates")) return "readout.candidates is required";
    const auto & candidates = value["candidates"];
    if (!candidates.is_array()) return "readout.candidates must be an array of strings";
    if (candidates.empty() || candidates.size() > static_cast<size_t>(kReadoutMaxCandidates)) {
        return "readout.candidates must contain 1-" + std::to_string(kReadoutMaxCandidates) +
               " strings; received " + std::to_string(candidates.size());
    }
    std::map<int32_t, size_t> seen;
    std::vector<int32_t> pieces;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto & candidate = candidates[i];
        const std::string where = "readout.candidates[" + std::to_string(i) + "]";
        if (!candidate.is_string()) return where + " must be a string";
        const std::string text = candidate.get<std::string>();
        if (text.empty()) return where + " must be non-empty";
        if (!tokenize(text, pieces)) return where + " could not be tokenized";
        if (pieces.size() != 1) {
            return where + " ('" + text + "') tokenizes to " + std::to_string(pieces.size()) +
                   " tokens; each candidate must be exactly one token";
        }
        const auto [it, inserted] = seen.emplace(pieces[0], i);
        if (!inserted) {
            return where + " ('" + text + "') maps to token " + std::to_string(pieces[0]) +
                   ", already used by readout.candidates[" + std::to_string(it->second) + "]";
        }
        out.texts.push_back(text);
        out.tokens.push_back(pieces[0]);
    }
    if (value.contains("top_k")) {
        const auto & top_k = value["top_k"];
        if (!top_k.is_number_integer()) return "readout.top_k must be an integer";
        const int64_t k = top_k.get<int64_t>();
        if (k < 0 || k > kReadoutMaxTopK) {
            return "readout.top_k must be 0-" + std::to_string(kReadoutMaxTopK) +
                   "; received " + std::to_string(k);
        }
        out.top_k = static_cast<int>(k);
    }
    return "";
}

struct ReadoutEntry {
    int32_t token = -1;
    float logit = 0.0f;
    double logprob = 0.0;
};

struct ReadoutResult {
    std::vector<ReadoutEntry> candidates;  // request order
    std::vector<ReadoutEntry> top;         // by logit desc, ties by token id asc
    double candidate_mass = 0.0;           // sum exp(logprob) over candidates
    double log_normalizer = 0.0;           // full-vocab logsumexp
};

// Raw-logit statistics for one output row: full-vocab logsumexp (max-shifted),
// candidate logits/logprobs, and the top-k entries. `candidates` must be valid
// ids in [0, n_vocab) and distinct (parse_readout_request guarantees both).
inline ReadoutResult compute_readout(const float * logits, int32_t n_vocab,
                                     const std::vector<int32_t> & candidates, int top_k) {
    ReadoutResult result;
    // Three branch-light passes (max, exp-sum, top-k) so the first two
    // auto-vectorize; the top-k pass rejects almost every entry with one compare.
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < n_vocab; ++i) maximum = logits[i] > maximum ? logits[i] : maximum;
    if (std::isfinite(maximum)) {
        double sum = 0.0;
        constexpr int32_t kBlock = 4096;  // float partial sums, double across blocks
        for (int32_t b = 0; b < n_vocab; b += kBlock) {
            const int32_t e = std::min(n_vocab, b + kBlock);
            float partial = 0.0f;
            for (int32_t i = b; i < e; ++i) partial += std::exp(logits[i] - maximum);
            sum += partial;
        }
        result.log_normalizer = double(maximum) + std::log(sum);
    } else {
        result.log_normalizer = double(maximum);
    }
    const size_t k = static_cast<size_t>(std::max(0, std::min<int>(top_k, n_vocab)));
    std::vector<int32_t> best;  // indices, sorted by (logit desc, id asc)
    best.reserve(k + 1);
    for (int32_t i = 0; k > 0 && i < n_vocab; ++i) {
        const float value = logits[i];
        if (best.size() == k && !(value > logits[best.back()])) continue;
        // Strict comparison while scanning ascending ids keeps lower ids first on ties.
        size_t at = best.size();
        while (at > 0 && value > logits[best[at - 1]]) --at;
        best.insert(best.begin() + static_cast<std::ptrdiff_t>(at), i);
        if (best.size() > k) best.pop_back();
    }
    auto entry = [&](int32_t token) {
        ReadoutEntry e;
        e.token = token;
        e.logit = logits[token];
        e.logprob = double(e.logit) - result.log_normalizer;
        return e;
    };
    for (int32_t token : candidates) {
        result.candidates.push_back(entry(token));
        result.candidate_mass += std::exp(result.candidates.back().logprob);
    }
    if (result.candidate_mass > 1.0) result.candidate_mass = 1.0;  // rounding guard
    for (int32_t token : best) result.top.push_back(entry(token));
    return result;
}
