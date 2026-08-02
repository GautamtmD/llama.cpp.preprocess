#include "inference_scheduler.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>

InferenceScheduler::InferenceScheduler(
    llama_context * ctx, const llama_vocab * vocab, int max_sequences,
    std::chrono::microseconds batching_window)
    : ctx_(ctx),
      vocab_(vocab),
      max_sequences_(max_sequences),
      batching_window_(batching_window) {
    if (!ctx_ || max_sequences_ <= 0) {
        throw std::invalid_argument("scheduler requires a context and positive sequence capacity");
    }
    for (int i = 0; i < max_sequences_; ++i) {
        free_sequences_.insert(i);
        logits_rows_[i] = -1;
    }
    preallocated_bytes_ = llama_state_get_size(ctx_);
    bytes_per_cell_ = std::max<size_t>(1, preallocated_bytes_ / std::max<uint32_t>(1, llama_n_ctx(ctx_)));
    worker_ = std::thread(&InferenceScheduler::worker_loop, this);
}

InferenceScheduler::~InferenceScheduler() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    llama_free(ctx_);
}

void InferenceScheduler::enqueue_command(std::function<void()> command) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stopping_) {
            throw std::runtime_error("inference scheduler is stopping");
        }
        commands_.push_back(std::move(command));
    }
    cv_.notify_one();
}

void InferenceScheduler::enqueue_deferred_command(std::function<void()> command) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stopping_) {
            throw std::runtime_error("inference scheduler is stopping");
        }
        deferred_commands_.push_back(std::move(command));
    }
    cv_.notify_one();
}

void InferenceScheduler::begin_generation() {
    active_generations_.fetch_add(1, std::memory_order_acq_rel);
}

void InferenceScheduler::end_generation() {
    const int previous = active_generations_.fetch_sub(1, std::memory_order_acq_rel);
    if (previous == 1) cv_.notify_one();
}

void InferenceScheduler::invalidate_logits() {
    for (auto & [_, row] : logits_rows_) {
        row = -1;
    }
}

int InferenceScheduler::allocate_sequence() {
    return invoke([this](llama_context * ctx) {
        if (free_sequences_.empty()) {
            return -1;
        }
        const llama_seq_id seq = *free_sequences_.begin();
        free_sequences_.erase(free_sequences_.begin());
        llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
        active_sequences_.insert(seq);
        logits_rows_[seq] = -1;
        sequence_families_[seq] = next_family_++;
        boundary_tokens_[seq] = LLAMA_TOKEN_NULL;
        return static_cast<int>(seq);
    });
}

void InferenceScheduler::release_sequence(llama_seq_id seq_id) {
    invoke([this, seq_id](llama_context * ctx) {
        llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
        active_sequences_.erase(seq_id);
        free_sequences_.insert(seq_id);
        logits_rows_[seq_id] = -1;
        sequence_families_.erase(seq_id);
        boundary_tokens_.erase(seq_id);
    });
}

void InferenceScheduler::prepare_sequence_mutation(llama_seq_id seq_id) {
    invoke([this, seq_id](llama_context *) {
        if (!active_sequences_.count(seq_id)) {
            throw std::runtime_error("cannot mutate inactive sequence");
        }
        // Split exact-state identity before an external inject changes KV. If a
        // sibling generates before the inject command arrives, this can only
        // miss a safe coalescing opportunity; it can never merge unlike states.
        sequence_families_[seq_id] = next_family_++;
        logits_rows_[seq_id] = -1;
        boundary_tokens_[seq_id] = LLAMA_TOKEN_NULL;
    });
}

std::optional<llama_seq_id> InferenceScheduler::fork_sequence(
    llama_seq_id source, llama_token boundary_token) {
    return invoke([this, source, boundary_token](llama_context * ctx) -> std::optional<llama_seq_id> {
        if (!active_sequences_.count(source) || free_sequences_.empty()) {
            return std::nullopt;
        }
        const llama_seq_id destination = *free_sequences_.begin();
        free_sequences_.erase(free_sequences_.begin());
        llama_memory_t memory = llama_get_memory(ctx);
        llama_memory_seq_rm(memory, destination, -1, -1);
        llama_memory_seq_cp(memory, source, destination, 0, -1);
        active_sequences_.insert(destination);
        logits_rows_[destination] = -1;
        uint64_t fork_family = sequence_families_[source];
        if (detached_families_.count(fork_family) || boundary_tokens_[source] != boundary_token) {
            fork_family = next_family_++;
            sequence_families_[source] = fork_family;
            detached_families_.erase(fork_family);
        }
        sequence_families_[destination] = fork_family;
        boundary_tokens_[source] = boundary_token;
        boundary_tokens_[destination] = boundary_token;
        return destination;
    });
}

std::future<SchedulerStepResult> InferenceScheduler::step(
    llama_seq_id seq_id, common_sampler * sampler, llama_token boundary_token) {
    auto request = std::make_shared<StepRequest>();
    request->seq_id = seq_id;
    request->sampler = sampler;
    request->boundary_token = boundary_token;
    auto future = request->promise.get_future();
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stopping_) {
            request->promise.set_value({LLAMA_TOKEN_NULL, false, 0, "scheduler is stopping"});
            return future;
        }
        steps_.push_back(request);
    }
    cv_.notify_one();
    return future;
}

void InferenceScheduler::rewind(
    llama_seq_id seq_id, llama_pos generation_start, llama_token boundary_token) {
    invoke_preserving_logits([this, seq_id, generation_start, boundary_token](llama_context * ctx) {
        llama_memory_seq_rm(llama_get_memory(ctx), seq_id, generation_start, -1);
        // Rewind restores a prior state but the scheduler does not retain that
        // historical identity. Assign a fresh conservative identity.
        sequence_families_[seq_id] = next_family_++;
        boundary_tokens_[seq_id] = boundary_token;
        logits_rows_[seq_id] = -1;
    });
}

SchedulerDiagnostics InferenceScheduler::diagnostics() {
    return invoke_preserving_logits([this](llama_context *) { return diagnostics_; });
}

int InferenceScheduler::active_sequences() {
    return invoke_preserving_logits(
        [this](llama_context *) { return static_cast<int>(active_sequences_.size()); });
}

size_t InferenceScheduler::logical_allocated_bytes() {
    return invoke_preserving_logits(
        [this](llama_context *) { return logical_cells_ * bytes_per_cell_; });
}

void InferenceScheduler::worker_loop() {
    while (true) {
        std::function<void()> command;
        std::vector<std::shared_ptr<StepRequest>> requests;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] {
                return stopping_ || !commands_.empty() || !steps_.empty() ||
                       (!deferred_commands_.empty() && active_generations_.load() == 0);
            });
            if (stopping_ && commands_.empty() && deferred_commands_.empty() && steps_.empty()) {
                return;
            }
            if (!commands_.empty()) {
                command = std::move(commands_.front());
                commands_.pop_front();
            } else if (!steps_.empty()) {
                const auto deadline = std::chrono::steady_clock::now() + batching_window_;
                cv_.wait_until(lock, deadline, [this] { return stopping_ || !commands_.empty(); });
                if (!commands_.empty()) {
                    command = std::move(commands_.front());
                    commands_.pop_front();
                } else {
                    std::set<llama_seq_id> seen;
                    while (!steps_.empty()) {
                        auto request = std::move(steps_.front());
                        steps_.pop_front();
                        if (seen.insert(request->seq_id).second) {
                            requests.push_back(std::move(request));
                        } else {
                            request->promise.set_value(
                                {LLAMA_TOKEN_NULL, false, 0, "duplicate pending sequence step"});
                        }
                    }
                }
            } else if (!deferred_commands_.empty() &&
                       (stopping_ || active_generations_.load() == 0)) {
                command = std::move(deferred_commands_.front());
                deferred_commands_.pop_front();
            }
        }
        if (command) {
            command();
        } else if (!requests.empty()) {
            process_steps(std::move(requests));
        }
    }
}

bool InferenceScheduler::decode_batch(
    llama_batch & batch, uint32_t distinct_sequences, std::string & error) {
    if (batch.n_tokens == 0) {
        return true;
    }
    if (llama_decode(ctx_, batch) != 0) {
        error = "llama_decode failed in pooled scheduler";
        return false;
    }
    ++diagnostics_.decode_calls;
    diagnostics_.decoded_tokens += batch.n_tokens;
    diagnostics_.max_sequences_per_decode =
        std::max(diagnostics_.max_sequences_per_decode, distinct_sequences);
    ++diagnostics_.decode_calls_by_sequence_count[distinct_sequences];
    return true;
}

bool InferenceScheduler::initialize_logits(
    const std::vector<std::shared_ptr<StepRequest>> & requests, std::string & error) {
    struct InitializationPlan {
        llama_seq_id seq_id;
        llama_pos position;
        llama_token token;
        bool remove_boundary;
    };

    llama_memory_t memory = llama_get_memory(ctx_);
    std::vector<InitializationPlan> plans;
    plans.reserve(requests.size());

    // Validate the complete initialization set before mutating any sequence.
    // Otherwise one later media/inactive request can leave earlier boundaries
    // removed even though no replacement decode runs.
    for (const auto & request : requests) {
        if (!active_sequences_.count(request->seq_id)) {
            error = "sequence is no longer active";
            return false;
        }
        llama_pos position = llama_memory_seq_pos_max(memory, request->seq_id);
        llama_token token = request->boundary_token;
        const bool remove_boundary = position >= 0;
        if (!remove_boundary) {
            position = 0;
            token = llama_vocab_bos(vocab_);
        } else if (token == LLAMA_TOKEN_NULL) {
            error = "sequence ends in media embeddings; inject text before generation";
            return false;
        }
        plans.push_back({request->seq_id, position, token, remove_boundary});
    }

    std::vector<const InitializationPlan *> removed;
    removed.reserve(plans.size());
    auto rollback_removed = [&](const std::string & original_error) {
        invalidate_logits();
        if (removed.empty()) {
            error = original_error;
            return;
        }
        llama_batch rollback =
            llama_batch_init(static_cast<int32_t>(removed.size()), 0, max_sequences_);
        for (const InitializationPlan * plan : removed) {
            const int row = rollback.n_tokens++;
            rollback.token[row] = plan->token;
            rollback.pos[row] = plan->position;
            rollback.n_seq_id[row] = 1;
            rollback.seq_id[row][0] = plan->seq_id;
            rollback.logits[row] = 0;
        }
        std::string rollback_error;
        const bool restored = decode_batch(
            rollback, static_cast<uint32_t>(removed.size()), rollback_error);
        llama_batch_free(rollback);
        error = original_error;
        if (!restored) {
            error += "; boundary rollback failed: " + rollback_error;
        }
    };

    for (const auto & plan : plans) {
        if (!plan.remove_boundary) continue;
        if (!llama_memory_seq_rm(memory, plan.seq_id, plan.position, plan.position + 1)) {
            rollback_removed("model memory does not support pooled suffix removal");
            return false;
        }
        removed.push_back(&plan);
    }

    llama_batch batch = llama_batch_init(static_cast<int32_t>(plans.size()), 0, max_sequences_);
    std::vector<std::pair<llama_seq_id, int>> sequence_rows;
    sequence_rows.reserve(plans.size());
    for (const auto & plan : plans) {
        const int row = batch.n_tokens++;
        batch.token[row] = plan.token;
        batch.pos[row] = plan.position;
        batch.n_seq_id[row] = 1;
        batch.seq_id[row][0] = plan.seq_id;
        batch.logits[row] = 1;
        sequence_rows.push_back({plan.seq_id, row});
    }

    invalidate_logits();
    std::string decode_error;
    const bool ok = decode_batch(
        batch, static_cast<uint32_t>(sequence_rows.size()), decode_error);
    llama_batch_free(batch);
    if (!ok) {
        // A failed backend decode may have committed only a prefix of the batch.
        // Remove any replacement/BOS rows that could have landed before
        // restoring the original boundaries recorded above.
        for (const auto & plan : plans) {
            llama_memory_seq_rm(memory, plan.seq_id, plan.position, plan.position + 1);
        }
        rollback_removed(decode_error);
        return false;
    }
    for (const auto & [sequence, row] : sequence_rows) {
        logits_rows_[sequence] = row;
    }
    return true;
}

void InferenceScheduler::process_steps(std::vector<std::shared_ptr<StepRequest>> requests) {
    std::string error;
    for (const auto & request : requests) {
        boundary_tokens_[request->seq_id] = request->boundary_token;
    }
    const bool needs_initialization = std::any_of(
        requests.begin(), requests.end(),
        [this](const auto & request) { return logits_rows_[request->seq_id] < 0; });
    if (needs_initialization) {
        std::vector<std::shared_ptr<StepRequest>> initialization = requests;
        std::set<llama_seq_id> included;
        std::set<uint64_t> families_to_detach;
        for (const auto & request : requests) {
            included.insert(request->seq_id);
            const uint64_t family = sequence_families_[request->seq_id];
            if (!detached_families_.count(family)) families_to_detach.insert(family);
        }
        for (const auto & [sequence, family] : sequence_families_) {
            if (!families_to_detach.count(family) || included.count(sequence)) continue;
            auto sibling = std::make_shared<StepRequest>();
            sibling->seq_id = sequence;
            sibling->sampler = nullptr;
            sibling->boundary_token = boundary_tokens_[sequence];
            initialization.push_back(std::move(sibling));
        }
        if (!initialize_logits(initialization, error)) {
            for (auto & request : requests) {
                request->promise.set_value({LLAMA_TOKEN_NULL, false, 0, error});
            }
            return;
        }
        detached_families_.insert(families_to_detach.begin(), families_to_detach.end());
    }

    struct Sampled {
        std::shared_ptr<StepRequest> request;
        llama_token token;
        bool eog;
    };
    std::vector<Sampled> sampled;
    sampled.reserve(requests.size());
    for (auto & request : requests) {
        const int row = logits_rows_[request->seq_id];
        if (row < 0) {
            request->promise.set_value({LLAMA_TOKEN_NULL, false, 0, "missing sequence logits"});
            continue;
        }
        const llama_token token = common_sampler_sample(request->sampler, ctx_, row);
        sampled.push_back({request, token, llama_vocab_is_eog(vocab_, token)});
    }

    llama_batch batch = llama_batch_init(static_cast<int32_t>(sampled.size()), 0, max_sequences_);
    using StateKey = std::tuple<uint64_t, llama_pos, llama_token>;
    std::map<StateKey, int> shared_rows;
    std::map<int, uint64_t> next_row_families;
    std::vector<std::pair<llama_seq_id, int>> decoded_sequence_rows;
    for (const auto & item : sampled) {
        if (item.eog) continue;
        const llama_pos position =
            llama_memory_seq_pos_max(llama_get_memory(ctx_), item.request->seq_id) + 1;
        // Equal position/token is safe to coalesce only when scheduler lineage
        // proves the complete logical KV state is identical. Family identity is
        // shared solely by fork or a previous coalesced transition, and is split
        // before inject, rewind, restore, or a different sampled transition.
        const StateKey key{
            sequence_families_.at(item.request->seq_id), position, item.token};
        auto row_it = shared_rows.find(key);
        int row;
        if (row_it == shared_rows.end()) {
            row = batch.n_tokens++;
            shared_rows.emplace(key, row);
            next_row_families[row] = next_family_++;
            batch.token[row] = item.token;
            batch.pos[row] = position;
            batch.n_seq_id[row] = 0;
            batch.logits[row] = 1;
        } else {
            row = row_it->second;
        }
        batch.seq_id[row][batch.n_seq_id[row]++] = item.request->seq_id;
        decoded_sequence_rows.push_back({item.request->seq_id, row});
    }

    if (batch.n_tokens > 0) {
        invalidate_logits();
        if (!decode_batch(batch, static_cast<uint32_t>(decoded_sequence_rows.size()), error)) {
            for (auto & item : sampled) {
                item.request->promise.set_value({LLAMA_TOKEN_NULL, false, 0, error});
            }
            llama_batch_free(batch);
            return;
        }
        for (const auto & [sequence, row] : decoded_sequence_rows) {
            logits_rows_[sequence] = row;
            sequence_families_[sequence] = next_row_families.at(row);
        }
        logical_cells_ += batch.n_tokens;
    }
    llama_batch_free(batch);

    for (auto & item : sampled) {
        const int cache_size =
            llama_memory_seq_pos_max(llama_get_memory(ctx_), item.request->seq_id) + 1;
        item.request->promise.set_value({item.token, item.eog, cache_size, {}});
    }
}
