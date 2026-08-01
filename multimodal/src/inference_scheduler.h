#pragma once

#include "llama.h"
#include "sampling.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

struct SchedulerStepResult {
    llama_token token = LLAMA_TOKEN_NULL;
    bool eog = false;
    int cache_size = 0;
    std::string error;
};

struct SchedulerDiagnostics {
    uint64_t decode_calls = 0;
    uint64_t decoded_tokens = 0;
    uint32_t max_sequences_per_decode = 0;
    std::map<uint32_t, uint64_t> decode_calls_by_sequence_count;
};

// Single owner of a pooled llama_context. Commands and generation steps may be
// submitted concurrently; only this object's worker thread touches the context.
class InferenceScheduler {
public:
    InferenceScheduler(llama_context * ctx, const llama_vocab * vocab, int max_sequences,
                       std::chrono::microseconds batching_window = std::chrono::microseconds(2000));
    ~InferenceScheduler();

    InferenceScheduler(const InferenceScheduler &) = delete;
    InferenceScheduler & operator=(const InferenceScheduler &) = delete;

    template <typename Fn>
    auto invoke(Fn && fn) -> decltype(fn(static_cast<llama_context *>(nullptr))) {
        using Result = decltype(fn(static_cast<llama_context *>(nullptr)));
        auto promise = std::make_shared<std::promise<Result>>();
        auto future = promise->get_future();
        enqueue_command([this, promise, fn = std::forward<Fn>(fn)]() mutable {
            try {
                if constexpr (std::is_void_v<Result>) {
                    fn(ctx_);
                    promise->set_value();
                } else {
                    promise->set_value(fn(ctx_));
                }
                invalidate_logits();
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });
        return future.get();
    }

    int allocate_sequence();
    void release_sequence(llama_seq_id seq_id);
    std::optional<llama_seq_id> fork_sequence(llama_seq_id source, llama_token boundary_token);

    std::future<SchedulerStepResult> step(llama_seq_id seq_id, common_sampler * sampler,
                                          llama_token boundary_token);
    void rewind(llama_seq_id seq_id, llama_pos generation_start, llama_token boundary_token);

    SchedulerDiagnostics diagnostics();
    int capacity() const { return max_sequences_; }
    int active_sequences();
    size_t preallocated_bytes() const { return preallocated_bytes_; }
    size_t logical_allocated_bytes();

private:
    struct StepRequest {
        llama_seq_id seq_id;
        common_sampler * sampler;
        llama_token boundary_token;
        std::promise<SchedulerStepResult> promise;
    };

    void enqueue_command(std::function<void()> command);
    void worker_loop();
    void process_steps(std::vector<std::shared_ptr<StepRequest>> requests);
    bool decode_batch(llama_batch & batch, uint32_t distinct_sequences, std::string & error);
    bool initialize_logits(const std::vector<std::shared_ptr<StepRequest>> & requests,
                           std::string & error);
    void invalidate_logits();

    llama_context * ctx_;
    const llama_vocab * vocab_;
    int max_sequences_;
    std::chrono::microseconds batching_window_;
    size_t preallocated_bytes_ = 0;

    std::thread worker_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool stopping_ = false;
    std::deque<std::function<void()>> commands_;
    std::deque<std::shared_ptr<StepRequest>> steps_;

    // Worker-thread-owned state.
    std::set<llama_seq_id> free_sequences_;
    std::set<llama_seq_id> active_sequences_;
    std::map<llama_seq_id, int> logits_rows_;
    std::map<llama_seq_id, uint64_t> sequence_families_;
    std::map<llama_seq_id, llama_token> boundary_tokens_;
    std::set<uint64_t> detached_families_;
    uint64_t next_family_ = 1;
    SchedulerDiagnostics diagnostics_;
    uint64_t logical_cells_ = 0;
    size_t bytes_per_cell_ = 1;
};
