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
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

inline constexpr int kReadoutMaxCandidates = 64;
inline constexpr int kReadoutMaxTopK = 16;
inline constexpr const char * kReadoutSemantics = "raw_last_row_pre_sampler_v1";

// Length of the longest prefix of `data[0, n)` that is well-formed UTF-8.
// Truncated multi-byte sequences at the end are excluded. Pure, no allocation.
inline size_t readout_utf8_valid_prefix(const char * data, size_t n) {
    size_t i = 0;
    while (i < n) {
        const auto c = static_cast<unsigned char>(data[i]);
        if (c < 0x80) { ++i; continue; }
        size_t len = 0;
        uint32_t lo = 0x80, hi = 0xBF;  // allowed range for the second byte
        if (c >= 0xC2 && c <= 0xDF) {
            len = 2;
        } else if (c >= 0xE0 && c <= 0xEF) {
            len = 3;
            if (c == 0xE0) lo = 0xA0;       // reject overlong
            else if (c == 0xED) hi = 0x9F;  // reject surrogates
        } else if (c >= 0xF0 && c <= 0xF4) {
            len = 4;
            if (c == 0xF0) lo = 0x90;       // reject overlong
            else if (c == 0xF4) hi = 0x8F;  // reject > U+10FFFF
        } else {
            break;  // C0/C1, F5-FF, or a lone continuation byte
        }
        if (i + len > n) break;  // truncated sequence
        const auto c1 = static_cast<unsigned char>(data[i + 1]);
        if (c1 < lo || c1 > hi) break;
        bool ok = true;
        for (size_t j = 2; j < len; ++j) {
            const auto cj = static_cast<unsigned char>(data[i + j]);
            if (cj < 0x80 || cj > 0xBF) { ok = false; break; }
        }
        if (!ok) break;
        i += len;
    }
    return i;
}

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

// Fast exp() for the logsumexp sum only (never for candidate/top logprobs,
// which stay `logit - log_normalizer` in double). The argument is max-shifted
// (x <= 0); values below -87 are clamped (exp(-87) < 2^-126, so a clamped
// entry contributes < 1e-37 against a sum that is >= 1). Range reduction
// x * log2(e) -> integer part via the magic-number rounding trick, a degree-6
// polynomial for 2^f on f in [-0.5, 0.5], and the 2^n scale assembled from
// the exponent bits. Everything is plain float arithmetic; the caller stores
// the polynomial and the scale bits in separate buffers (two loops) because
// MSVC does not auto-vectorize the single-loop bit-cast form, while the
// two-loop form vectorizes. Truncation error of the polynomial is < 2e-7
// relative; with float rounding the per-element relative error stays in the
// low 1e-6 range on [-87, 0] (worst in the deep tail, whose weight in the
// sum is negligible), keeping |log_normalizer - double reference| << 1e-5.
inline float readout_fast_exp_poly(float x, uint32_t & scale_bits) {
    const float xc = x < -87.0f ? -87.0f : x;
    const float t = xc * 1.4426950408889634f;  // x * log2(e)
    const float nf = (t + 12582912.0f) - 12582912.0f;  // round(t), |t| < 2^22
    const float f = t - nf;                            // [-0.5, 0.5]
    scale_bits = static_cast<uint32_t>(static_cast<int32_t>(nf) + 127) << 23;
    return 1.0f + f * (0.6931471805599453f +
           f * (0.2402265069591007f +
           f * (0.05550410866482158f +
           f * (0.009618129107628477f +
           f * (0.0013333558146428443f +
           f * 0.00015403430350607636f)))));
}

// Raw-logit statistics for one output row: full-vocab logsumexp (max-shifted),
// candidate logits/logprobs, and the top-k entries. `candidates` must be valid
// ids in [0, n_vocab) and distinct (parse_readout_request guarantees both).
inline ReadoutResult compute_readout(const float * logits, int32_t n_vocab,
                                     const std::vector<int32_t> & candidates, int top_k) {
    ReadoutResult result;
    const int k = std::max(0, std::min<int>(std::min<int>(top_k, n_vocab), kReadoutMaxTopK));
    // One blocked pass: a vectorizable per-block max scan (which also gates
    // the scalar top-k scan: a block whose max cannot beat the current k-th
    // best holds no top-k entry), an online rescale of the double sum when a
    // block raises the running max, and the fast-exp sum. Float partial sums
    // within a block, double across blocks, as before. Top-k semantics are
    // unchanged: logit desc, ties by token id asc (strict `>` on an ascending
    // scan, and a block is skipped only when no entry can satisfy it).
    int32_t best[kReadoutMaxTopK];  // indices, sorted by (logit desc, id asc)
    int nb = 0;
    float thresh = -std::numeric_limits<float>::infinity();  // logits[best[k-1]] once full
    float maximum = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    constexpr int32_t kBlock = 4096;  // float partial sums, double across blocks
    constexpr int32_t kChunk = 1024;  // fast-exp scratch chunk (fits L1 with the source chunk)
    float exps[kChunk];
    uint32_t exp_bits[kChunk];
    float scales[kChunk];
    for (int32_t b = 0; b < n_vocab; b += kBlock) {
        const int32_t e = std::min(n_vocab, b + kBlock);
        float block_max = -std::numeric_limits<float>::infinity();
        for (int32_t i = b; i < e; ++i) block_max = logits[i] > block_max ? logits[i] : block_max;
        if (k > 0 && (nb < k || block_max > thresh)) {
            for (int32_t i = b; i < e; ++i) {
                const float value = logits[i];
                if (nb == k && !(value > thresh)) continue;
                int at = nb;
                while (at > 0 && value > logits[best[at - 1]]) --at;
                const int lim = std::min(nb, k - 1);
                for (int j = lim; j > at; --j) best[j] = best[j - 1];
                if (at < k) {
                    best[at] = i;
                    if (nb < k) ++nb;
                }
                thresh = nb == k ? logits[best[k - 1]] : -std::numeric_limits<float>::infinity();
            }
        }
        if (block_max > maximum) {
            if (std::isfinite(maximum) && std::isfinite(block_max)) {
                sum *= std::exp(double(maximum) - double(block_max));
            }
            maximum = block_max;
        }
        if (!std::isfinite(block_max) || !std::isfinite(maximum)) continue;
        float partial = 0.0f;
        for (int32_t c = b; c < e; c += kChunk) {
            const int32_t m = std::min<int32_t>(kChunk, e - c);
            const float * src = logits + c;
            for (int32_t i = 0; i < m; ++i) {
                exps[i] = readout_fast_exp_poly(src[i] - maximum, exp_bits[i]);
            }
            // Exponent bits -> float scales via memcpy (no strict-aliasing
            // violation; the whole-chunk copy is what MSVC vectorizes).
            std::memcpy(scales, exp_bits, static_cast<size_t>(m) * sizeof(float));
            for (int32_t i = 0; i < m; ++i) exps[i] *= scales[i];
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
            int32_t i = 0;
            for (; i + 8 <= m; i += 8) {
                s0 += exps[i]; s1 += exps[i + 1]; s2 += exps[i + 2]; s3 += exps[i + 3];
                s4 += exps[i + 4]; s5 += exps[i + 5]; s6 += exps[i + 6]; s7 += exps[i + 7];
            }
            for (; i < m; ++i) s0 += exps[i];
            partial += ((s0 + s4) + (s1 + s5)) + ((s2 + s6) + (s3 + s7));
        }
        sum += partial;
    }
    result.log_normalizer = std::isfinite(maximum) ? double(maximum) + std::log(sum)
                                                   : double(maximum);
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
    for (int j = 0; j < nb; ++j) result.top.push_back(entry(best[j]));
    return result;
}
