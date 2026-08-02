// multimodal-fork-bench — M2.0 measurement gate.
//
// Measures the cost of forking a session's KV cache three ways, across context
// sizes, to pick the fork copy semantics (see docs/decisions/0004 + worklog):
//
//   A  (full deep-copy):   new ctx + llama_state_get_data -> llama_state_set_data
//                          (copies KV + logits + embeddings)
//   A' (seq deep-copy):    new ctx + llama_state_seq_get_data -> llama_state_seq_set_data
//                          (copies only the sequence's KV)
//   B  (intra-context cp): llama_memory_seq_cp(mem, 0, 1, 0, -1) in the SAME context
//                          (aliases cells when src/dst share a stream — metadata only)
//
// For each (approach, context size N) it reports: median fork copy time over a
// few trials (A/A' INCLUDE the new-context creation, which is part of the fork
// cost under those approaches; B needs no new context), the forked session's
// full-state footprint (llama_state_get_size — the VRAM a fresh context holds;
// A/A' each allocate ≈ this much per fork, B reuses the pool ≈ 0), and the
// per-sequence KV size.
//
// Build & run (see scripts/build_engine.sh; this target is opt-in):
//   cmake --build engine/multimodal/build --config Release --target multimodal-fork-bench
//   CUDA_VISIBLE_DEVICES=1 engine/multimodal/build/bin/Release/multimodal-fork-bench.exe \
//       --model "/c/ML Models/Gemma4 12b/gemma-4-12b-it-qat-q4_0.gguf" -ngl 99 \
//       --ctx-size 4096 --tokens 128 1024 2048
//
// Note: each trial creates ~3 full-size contexts for the 12B (source + one dst
// per deep-copy approach), so a full sweep is slow (minutes). The n=ctx_size
// (full-context) case is especially heavy and may need a long timeout.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "execution_policy.h"
#include "ggml-backend.h"
#include "llama.h"

static size_t gpu_free_bytes() {
    size_t result = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        result += free;
    }
    return result;
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Tokenize `text` repeatedly until we have at least `n` tokens.
static std::vector<llama_token> make_n_tokens(const llama_vocab * vocab, int n, bool add_bos) {
    const std::string unit = "The quick brown fox jumps over the lazy dog. ";
    std::string text;
    std::vector<llama_token> toks;
    while ((int) toks.size() < n) {
        text += unit;
        std::vector<llama_token> buf(text.size() + 16);
        const int r = llama_tokenize(vocab, text.c_str(), (int) text.size(), buf.data(),
                                     (int) buf.size(), add_bos, true);
        if (r < 0) { text.clear(); continue; }  // grow and retry
        toks.assign(buf.begin(), buf.begin() + r);
    }
    if ((int) toks.size() > n) toks.resize(n);
    return toks;
}

// Decode `toks` into ctx's seq 0 in batches (fills the KV cache). Returns false on error.
static bool fill_cache(llama_context * ctx, const std::vector<llama_token> & toks, int n_batch) {
    size_t i = 0;
    while (i < toks.size()) {
        int n = (int) std::min<size_t>(n_batch, toks.size() - i);
        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data() + i), n);
        if (llama_decode(ctx, b) != 0) return false;
        i += n;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int ngl = 99;
    int ctx_size = 4096;
    int n_batch = 2048;
    std::vector<int> token_counts = {128, 1024, 2048};
    int trials = 3;
    bool allow_cpu = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string{}; };
        if      (a == "--model" || a == "-m") model_path = next();
        else if (a == "-ngl" || a == "--n-gpu-layers") ngl = atoi(next().c_str());
        else if (a == "-c" || a == "--ctx-size") ctx_size = atoi(next().c_str());
        else if (a == "--n-batch") n_batch = atoi(next().c_str());
        else if (a == "--tokens") {
            token_counts.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') token_counts.push_back(atoi(argv[++i]));
        } else if (a == "--trials") trials = atoi(next().c_str());
        else if (a == "--allow-cpu") allow_cpu = true;
        else if (a == "-h" || a == "--help") {
            std::printf("multimodal-fork-bench --model PATH [-ngl N] [-c N] [--tokens N ...] [--trials N] [--allow-cpu]\n"
                        "  --allow-cpu  explicitly permit CPU-only execution (default: GPU required)\n");
            return 0;
        }
    }
    if (model_path.empty()) { std::fprintf(stderr, "error: --model required\n"); return 2; }

    llama_log_set([](ggml_log_level lvl, const char * txt, void *) {
        if (lvl >= GGML_LOG_LEVEL_WARN) std::fprintf(stderr, "%s", txt);
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
    if (!allow_cpu && (gpu_devices == 0 || ngl == 0)) {
        std::fprintf(stderr, "error: %s\n", gpu_execution_error("benchmark").c_str());
        llama_backend_free();
        return 3;
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    std::fprintf(stderr, "loading model: %s ...\n", model_path.c_str());
    const size_t gpu_free_before_model = gpu_free_bytes();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "error: model load failed\n");
        if (!allow_cpu) std::fprintf(stderr, "error: %s\n", gpu_execution_error("benchmark").c_str());
        llama_backend_free();
        return 1;
    }
    const size_t gpu_model_bytes =
        gpu_allocation_delta(gpu_free_before_model, gpu_free_bytes());
    if (!gpu_execution_allowed(allow_cpu, gpu_model_bytes)) {
        std::fprintf(stderr, "error: %s\n", gpu_execution_error("benchmark").c_str());
        llama_model_free(model);
        llama_backend_free();
        return 3;
    }
    std::fprintf(stderr, "execution: actual GPU model allocation=%zu bytes%s.\n",
                 gpu_model_bytes, allow_cpu ? " (--allow-cpu enabled)" : "");
    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto make_ctx = [&]() {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = ctx_size;
        cp.n_batch = std::min<int>(n_batch, ctx_size);
        cp.no_perf = true;
        return llama_init_from_model(model, cp);
    };

    // Unbuffered stdout so partial output survives if the run is interrupted.
    setvbuf(stdout, nullptr, _IONBF, 0);

    // One-off CUDA/graph warmup so the first measurement isn't inflated by JIT.
    { llama_context * w = make_ctx(); if (w) llama_free(w); }

    std::printf("# multimodal-fork-bench  (ctx_size=%d, trials=%d, gpu_layers=%d)\n", ctx_size, trials, ngl);
    std::printf("# A/A' times INCLUDE the new-context creation (it is part of the fork cost);\n"
                "# B (seq_cp) needs no new context. m = median over trials.\n"
                "# new-sess state = forked session's full-state footprint (A/A' each allocate\n"
                "# ≈ this much per fork; B reuses the pool, ≈ 0).\n");
    std::printf("\n");
    std::printf("| N tokens | A fork (ms) | A' seq fork (ms) | B seq_cp (ms) | new-sess state (MiB) | seq KV (MiB) |\n");
    std::printf("|---:|---:|---:|---:|---:|---:|\n");

    for (int n : token_counts) {
        if (n > ctx_size) { std::fprintf(stderr, "skip N=%d > ctx_size\n", n); continue; }
        auto toks = make_n_tokens(vocab, n, /*add_bos*/ true);

        std::vector<double> tA, tAp, tB;
        size_t full_bytes = 0, seq_bytes = 0, new_sess_bytes = 0;

        for (int t = 0; t < trials; ++t) {
            std::fprintf(stderr, "  [N=%d trial %d/%d] creating source...\n", n, t + 1, trials);
            llama_context * src = make_ctx();
            if (!fill_cache(src, toks, n_batch)) { std::fprintf(stderr, "fill failed\n"); return 1; }
            full_bytes = llama_state_get_size(src);
            seq_bytes  = llama_state_seq_get_size(src, 0);

            // ---- A: full deep-copy into a new context (new ctx + roundtrip) ----
            {
                std::vector<uint8_t> buf(full_bytes);
                double t0 = now_s();
                llama_context * dst = make_ctx();
                llama_state_get_data(src, buf.data(), buf.size());
                llama_state_set_data(dst, buf.data(), buf.size());
                double t1 = now_s();
                tA.push_back((t1 - t0) * 1000.0);
                new_sess_bytes = llama_state_get_size(dst);
                llama_free(dst);
            }

            // ---- A': per-sequence deep-copy into a new context ----
            {
                std::vector<uint8_t> buf(seq_bytes);
                double t0 = now_s();
                llama_context * dst = make_ctx();
                llama_state_seq_get_data(src, buf.data(), buf.size(), /*seq_id*/ 0);
                llama_state_seq_set_data(dst, buf.data(), buf.size(), /*dest_seq_id*/ 0);
                double t1 = now_s();
                tAp.push_back((t1 - t0) * 1000.0);
                new_sess_bytes = llama_state_get_size(dst);
                llama_free(dst);
            }

            // ---- B: intra-context seq_cp (0 -> 1), no new context, ~0 extra VRAM ----
            {
                llama_memory_t mem = llama_get_memory(src);
                llama_memory_seq_rm(mem, 1, -1, -1);  // clear any prior seq 1
                double t0 = now_s();
                llama_memory_seq_cp(mem, 0, 1, /*p0*/ 0, /*p1*/ -1);
                double t1 = now_s();
                tB.push_back((t1 - t0) * 1000.0);
            }

            llama_free(src);
        }

        std::sort(tA.begin(), tA.end());
        std::sort(tAp.begin(), tAp.end());
        std::sort(tB.begin(), tB.end());
        double medA = tA[tA.size() / 2];
        double medAp = tAp[tAp.size() / 2];
        double medB = tB[tB.size() / 2];
        auto mib = [](size_t b) { return b / (1024.0 * 1024.0); };
        std::printf("| %d | %.1f | %.1f | %.3f | %.1f | %.1f |\n",
                    n, medA, medAp, medB, mib(new_sess_bytes), mib(seq_bytes));
        std::fprintf(stderr, "  [N=%d] A=%.1fms A'=%.1fms B=%.3fms (newSess=%.1fMiB seqKV=%.1fMiB)\n",
                     n, medA, medAp, medB, mib(new_sess_bytes), mib(seq_bytes));
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
