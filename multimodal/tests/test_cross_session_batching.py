"""Measured EUS-4 acceptance tests for pooled cross-session batching.

Run against the live GPU server. These tests intentionally assert scheduler and
pool telemetry, not throughput alone, so six independent contexts cannot pass by
merely running concurrently.
"""

from __future__ import annotations

import base64
import json
import math
import statistics
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

pytestmark = [pytest.mark.usefixtures("base", "make_session"), pytest.mark.requires("text")]

N_SEQUENCES = 6
N_STEPS = 32
FORK_SAMPLES = 50
FORK_P95_MS = 5.0
BATCHED_STEP_RATIO = 1.5


def _post(base: str, path: str, **kwargs) -> requests.Response:
    return requests.post(f"{base}{path}", timeout=180, **kwargs)


def _inject(base: str, sid: str, text: str) -> dict:
    response = _post(base, f"/sessions/{sid}/inject", json={"text": text})
    assert response.status_code == 200, response.text
    return response.json()


def _fork(base: str, sid: str) -> dict:
    response = _post(base, f"/sessions/{sid}/fork")
    assert response.status_code == 200, response.text
    return response.json()


def _generate(base: str, sid: str, max_tokens: int = N_STEPS, **extra) -> dict:
    body = {"max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True}
    body.update(extra)
    response = _post(base, f"/sessions/{sid}/generate", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _delete(base: str, sid: str) -> None:
    response = requests.delete(f"{base}/sessions/{sid}", timeout=60)
    assert response.status_code == 200, response.text


def _usage(base: str) -> dict:
    response = requests.get(f"{base}/sessions/usage", timeout=60)
    assert response.status_code == 200, response.text
    return response.json()


def _batching(base: str) -> dict:
    response = requests.get(f"{base}/diagnostics/batching", timeout=60)
    assert response.status_code == 200, response.text
    return response.json()


def _hist_delta(before: dict, after: dict, width: int) -> int:
    key = str(width)
    return int(after["decode_calls_by_sequence_count"].get(key, 0)) - int(
        before["decode_calls_by_sequence_count"].get(key, 0)
    )


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def test_pool_is_single_explicitly_sized_context(base, make_session):
    source = make_session()
    _inject(base, source, "Cross-session pooled ownership sentinel. ")
    forks = [_fork(base, source)["session_id"] for _ in range(N_SEQUENCES)]
    try:
        pool = _usage(base)["pool"]
        assert pool["contexts_created"] == 1
        assert pool["capacity"] >= N_SEQUENCES + 1
        assert pool["active_sequences"] >= N_SEQUENCES + 1
        assert pool["free_sequences"] == pool["capacity"] - pool["active_sequences"]
        assert pool["ctx_size_per_sequence"] > 0
        assert pool["ctx_size_total"] >= pool["ctx_size_per_sequence"] * pool["capacity"]
        assert pool["preallocated_bytes"] > 0
        assert pool["model_gpu_bytes"] > 0
        assert pool["gpu_free_bytes"] >= 0
        assert pool["logical_owned_cells"] >= 0
        assert pool["sequence_capabilities_probed"] is True
    finally:
        for sid in forks:
            _delete(base, sid)


def test_fork_p95_and_shared_prefix_memory(base, make_session):
    source = make_session()
    _inject(base, source, "The quick brown fox jumps over the lazy dog. " * 20)
    samples: list[float] = []
    for _ in range(FORK_SAMPLES):
        forked = _fork(base, source)
        samples.append(float(forked["fork_ms_precise"]))
        _delete(base, forked["session_id"])
    p95 = _p95(samples)
    print(
        f"  [eus4-fork] n={len(samples)} median={statistics.median(samples):.3f}ms p95={p95:.3f}ms"
    )
    assert p95 < FORK_P95_MS

    source_status = requests.get(f"{base}/sessions/{source}", timeout=60).json()
    before = _usage(base)["pool"]
    forks = [_fork(base, source)["session_id"] for _ in range(N_SEQUENCES)]
    try:
        after = _usage(base)["pool"]
        owned_delta = after["logical_owned_cells"] - before["logical_owned_cells"]
        assert owned_delta == source_status["cache_size"] * N_SEQUENCES

        gpu_delta = max(0, before["gpu_free_bytes"] - after["gpu_free_bytes"])
        print(f"  [eus4-prefix] six-fork actual GPU delta={gpu_delta / (1024 * 1024):.3f} MiB")
        assert gpu_delta < 60 * 1024 * 1024
        assert gpu_delta / N_SEQUENCES < 10 * 1024 * 1024
        assert after["preallocated_bytes"] == before["preallocated_bytes"]
    finally:
        for sid in forks:
            _delete(base, sid)

    released = _usage(base)["pool"]
    assert released["logical_owned_cells"] == before["logical_owned_cells"]


def test_logical_ownership_tracks_inject_generate_and_offload(base, make_session):
    sid = make_session()
    initial = _usage(base)["pool"]
    assert initial["logical_owned_cells"] >= 0

    injected = _inject(base, sid, "Logical ownership lifecycle sentinel. ")
    after_inject = _usage(base)["pool"]
    assert (
        after_inject["logical_owned_cells"] - initial["logical_owned_cells"]
        == injected["cache_size"]
    )

    generated = _generate(base, sid, max_tokens=4)
    after_generate = _usage(base)["pool"]
    assert (
        after_generate["logical_owned_cells"] - after_inject["logical_owned_cells"]
        == generated["n_tokens"]
    )

    offload = _post(base, f"/sessions/{sid}/offload")
    assert offload.status_code == 200, offload.text
    after_offload = _usage(base)["pool"]
    assert after_offload["logical_owned_cells"] == initial["logical_owned_cells"]


def test_six_jobs_share_decode_and_match_greedy_baseline(base, make_session):
    source = make_session()
    _inject(
        base,
        source,
        "Continue this deterministic sequence with many short tokens: 1 2 3 4 5. ",
    )

    # Warm, independently run baseline from an identical fork.
    control = _fork(base, source)["session_id"]
    baseline = _generate(base, control)
    _delete(base, control)
    assert baseline["n_tokens"] == N_STEPS
    baseline_step_ms = baseline["gen_ms_precise"] / N_STEPS

    sessions = [_fork(base, source)["session_id"] for _ in range(N_SEQUENCES)]
    barrier = threading.Barrier(N_SEQUENCES)
    telemetry_before = _batching(base)

    def run(sid: str) -> dict:
        barrier.wait(timeout=30)
        return _generate(base, sid)

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=N_SEQUENCES) as executor:
            results = list(executor.map(run, sessions))
        elapsed_ms = (time.perf_counter() - started) * 1000
        telemetry_after = _batching(base)

        for result in results:
            assert result["n_tokens"] == N_STEPS
            assert result["text"] == baseline["text"]
            assert result["tokens"] == baseline["tokens"]
            assert result["cache_size"] == baseline["cache_size"]

        six_way_calls = _hist_delta(telemetry_before, telemetry_after, N_SEQUENCES)
        assert six_way_calls >= N_STEPS, (telemetry_before, telemetry_after)
        assert telemetry_after["max_sequences_per_decode"] >= N_SEQUENCES

        batched_step_ms = elapsed_ms / N_STEPS
        ratio = batched_step_ms / baseline_step_ms
        aggregate_tps = N_SEQUENCES * N_STEPS / (elapsed_ms / 1000)
        print(
            f"  [eus4-batch] n=1 {baseline_step_ms:.3f}ms/step; n=6 "
            f"{batched_step_ms:.3f}ms/step ratio={ratio:.3f}; aggregate={aggregate_tps:.2f}tok/s"
        )
        assert ratio <= BATCHED_STEP_RATIO
    finally:
        for sid in sessions:
            _delete(base, sid)


def test_divergent_histories_never_merge_equal_position_token_rows(base, make_session):
    """Equal `(position, token)` is insufficient proof that KV ancestry matches.

    The one-token grammar forces all jobs to sample the same token at the same
    position after different one-token histories. Besides one boundary-logit
    initialization row per sequence, the scheduler must decode one generated row
    per sequence—not one row carrying unrelated sequence IDs.
    """
    sessions = [make_session() for _ in range(N_SEQUENCES)]
    injects = [
        _inject(base, sid, marker)
        for sid, marker in zip(sessions, ("A", "B", "C", "D", "E", "F"), strict=True)
    ]
    assert len({result["cache_size"] for result in injects}) == 1
    assert len({result["tokens_injected"] for result in injects}) == 1

    barrier = threading.Barrier(N_SEQUENCES)
    telemetry_before = _batching(base)

    def run(sid: str) -> dict:
        barrier.wait(timeout=30)
        return _generate(base, sid, max_tokens=1, grammar='root ::= "X"')

    with ThreadPoolExecutor(max_workers=N_SEQUENCES) as executor:
        results = list(executor.map(run, sessions))

    assert all(result["n_tokens"] == 1 for result in results)
    assert len({tuple(result["tokens"]) for result in results}) == 1
    telemetry_after = _batching(base)
    co_scheduled_widths = [
        width
        for width in range(2, N_SEQUENCES + 1)
        if _hist_delta(telemetry_before, telemetry_after, width) >= 2
    ]
    assert co_scheduled_widths, (telemetry_before, telemetry_after)
    decoded_delta = telemetry_after["decoded_tokens"] - telemetry_before["decoded_tokens"]
    expected_rows = N_SEQUENCES + sum(result["n_tokens"] for result in results)
    assert decoded_delta == expected_rows, (
        "unrelated sequences were merged into one decode row solely because "
        f"position/token matched: expected {expected_rows} rows, observed {decoded_delta}"
    )


@pytest.mark.requires("audio")
def test_failed_media_logits_initialization_does_not_truncate_valid_sessions(base, make_session):
    source = make_session()
    source_inject = _inject(base, source, "Transactional logits boundary sentinel. ")
    bad = make_session()
    _inject(base, bad, "Raw audio follows: ")
    pcm = struct.pack("<640f", *([0.0] * 640))
    audio = _post(
        base,
        f"/sessions/{bad}/inject",
        json={"audio": base64.b64encode(pcm).decode("ascii")},
    )
    assert audio.status_code == 200, audio.text

    exercised_shared_batch = False
    for _ in range(5):
        sessions = [_fork(base, source)["session_id"] for _ in range(5)]
        before = {sid: source_inject["cache_size"] for sid in sessions}
        barrier = threading.Barrier(6)

        def run_valid(sid: str, *, _barrier=barrier) -> requests.Response:
            _barrier.wait(timeout=30)
            return _post(
                base,
                f"/sessions/{sid}/generate",
                json={"max_tokens": 1, "temperature": 0.0, "ignore_eos": True},
            )

        def run_invalid(*, _barrier=barrier) -> requests.Response:
            _barrier.wait(timeout=30)
            # Keep the media-ending request inside the scheduler's batching
            # window but behind already-released valid request threads.
            time.sleep(0.0005)
            return _post(
                base,
                f"/sessions/{bad}/generate",
                json={"max_tokens": 1, "temperature": 0.0, "ignore_eos": True},
            )

        try:
            with ThreadPoolExecutor(max_workers=6) as executor:
                valid_futures = [executor.submit(run_valid, sid) for sid in sessions]
                invalid_future = executor.submit(run_invalid)
                valid_responses = [future.result(timeout=30) for future in valid_futures]
                invalid_response = invalid_future.result(timeout=30)

            shared_error = "sequence ends in media embeddings"
            failed_valids = [
                sid
                for sid, response in zip(sessions, valid_responses, strict=True)
                if shared_error in response.text
            ]
            exercised_shared_batch = shared_error in invalid_response.text and bool(failed_valids)
            if not exercised_shared_batch:
                continue

            after = {}
            for sid in failed_valids:
                status = requests.get(f"{base}/sessions/{sid}", timeout=30)
                assert status.status_code == 200, status.text
                after[sid] = status.json()["cache_size"]
            expected_sizes = {sid: before[sid] for sid in failed_valids}
            assert after == expected_sizes, (
                "failed initialization truncated unrelated sessions: "
                f"before={expected_sizes}, after={after}"
            )

            control = _fork(base, source)["session_id"]
            try:
                expected = _generate(base, control, max_tokens=8)
                observed = _generate(base, failed_valids[0], max_tokens=8)
                assert observed["tokens"] == expected["tokens"]
                assert observed["text"] == expected["text"]
                assert observed["cache_size"] == expected["cache_size"]
            finally:
                _delete(base, control)
            break
        finally:
            for sid in sessions:
                _delete(base, sid)

    assert exercised_shared_batch, "could not co-schedule valid and media-ending initialization"


def test_completed_coalesced_siblings_keep_current_boundary_when_one_resumes(base, make_session):
    source = make_session()
    _inject(
        base,
        source,
        "Continue this exact deterministic sequence: 1 2 3 4 5 6 7 8 9. ",
    )
    siblings = [_fork(base, source)["session_id"] for _ in range(3)]
    barrier = threading.Barrier(3)
    telemetry_before = _batching(base)

    def generate_initial(sid: str) -> dict:
        barrier.wait(timeout=30)
        return _generate(base, sid, max_tokens=8)

    control = None
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            initial = list(executor.map(generate_initial, siblings))
        assert initial[0]["tokens"] == initial[1]["tokens"] == initial[2]["tokens"]
        telemetry_after = _batching(base)
        assert _hist_delta(telemetry_before, telemetry_after, 3) >= 8

        control = _fork(base, siblings[0])["session_id"]
        invalidator = make_session()
        _inject(base, invalidator, "Invalidate unrelated pooled logits. ")

        idle_before = {
            sid: requests.get(f"{base}/sessions/{sid}", timeout=30).json()["cache_size"]
            for sid in (siblings[1], control)
        }
        _generate(base, siblings[0], max_tokens=1)
        idle_status = {
            sid: requests.get(f"{base}/sessions/{sid}", timeout=30).json()
            for sid in (siblings[1], control)
        }
        idle_after = {sid: status["cache_size"] for sid, status in idle_status.items()}
        assert idle_after == idle_before
        for status in idle_status.values():
            assert status["boundary_token"] == initial[0]["tokens"][-1]

        suffix = " Continue from the exact cached sequence. "
        _inject(base, siblings[1], suffix)
        _inject(base, control, suffix)
        comparison_barrier = threading.Barrier(2)

        def generate_comparison(sid: str) -> dict:
            comparison_barrier.wait(timeout=30)
            return _generate(base, sid, max_tokens=16)

        with ThreadPoolExecutor(max_workers=2) as executor:
            observed_future = executor.submit(generate_comparison, siblings[1])
            expected_future = executor.submit(generate_comparison, control)
            observed = observed_future.result(timeout=30)
            expected = expected_future.result(timeout=30)
        assert observed["tokens"] == expected["tokens"]
        assert observed["text"] == expected["text"]
        assert observed["cache_size"] == expected["cache_size"]
    finally:
        for sid in siblings:
            _delete(base, sid)
        if control is not None:
            _delete(base, control)


@pytest.mark.parametrize("_attempt", range(3))
def test_cancel_one_of_six_leaves_other_jobs_byte_identical(base, make_session, _attempt):
    source = make_session()
    _inject(
        base,
        source,
        "Continue this deterministic sequence with many short tokens: 1 2 3 4 5. ",
    )

    control = _fork(base, source)["session_id"]
    baseline = _generate(base, control)
    _delete(base, control)

    sessions = [_fork(base, source)["session_id"] for _ in range(N_SEQUENCES)]
    cancelled_sid = sessions[0]
    barrier = threading.Barrier(N_SEQUENCES)
    second_cancelled_token = threading.Event()
    cancelled_events: list[dict] = []
    telemetry_before = _batching(base)

    def run(sid: str) -> dict | None:
        barrier.wait(timeout=30)
        if sid != cancelled_sid:
            return _generate(base, sid)
        response = requests.post(
            f"{base}/sessions/{sid}/generate",
            json={"stream": True, "max_tokens": 128, "temperature": 0.0, "ignore_eos": True},
            stream=True,
            timeout=60,
        )
        assert response.status_code == 200, response.text
        token_count = 0
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            cancelled_events.append(event)
            if event.get("type") == "token":
                token_count += 1
                if token_count == 2:
                    second_cancelled_token.set()
        return None

    try:
        with ThreadPoolExecutor(max_workers=N_SEQUENCES) as executor:
            futures = [executor.submit(run, sid) for sid in sessions]
            assert second_cancelled_token.wait(30), "six-way stream did not produce two tokens"
            cancel = _post(base, f"/sessions/{cancelled_sid}/cancel")
            assert cancel.status_code == 200, cancel.text
            assert cancel.json()["cancelled"] is True
            results = [future.result(timeout=30) for future in futures]

        telemetry_after = _batching(base)
        assert _hist_delta(telemetry_before, telemetry_after, N_SEQUENCES) >= 1
        done = [event for event in cancelled_events if event.get("type") == "done"]
        assert len(done) == 1, cancelled_events

        for result in results[1:]:
            assert result is not None
            assert result["text"] == baseline["text"]
            assert result["tokens"] == baseline["tokens"]
            assert result["cache_size"] == baseline["cache_size"]

        reused = _generate(base, cancelled_sid)
        assert reused["text"] == baseline["text"]
        assert reused["tokens"] == baseline["tokens"]
        assert reused["cache_size"] == baseline["cache_size"]
    finally:
        for sid in sessions:
            _delete(base, sid)


def test_per_job_sampling_stop_and_grammar_are_isolated(base, make_session):
    source = make_session()
    _inject(base, source, "Generate lowercase letters forever. ")
    sessions = [_fork(base, source)["session_id"] for _ in range(3)]
    barrier = threading.Barrier(3)
    requests_by_job = [
        {"max_tokens": 5, "stop": ["z"]},
        {"max_tokens": 11, "temperature": 0.7, "seed": 123},
        {"max_tokens": 17, "grammar": "root ::= [a-z ]+"},
    ]

    def run(item: tuple[str, dict]) -> dict:
        sid, params = item
        barrier.wait(timeout=30)
        return _generate(base, sid, **params)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run, zip(sessions, requests_by_job, strict=True)))
    assert results[0]["n_tokens"] <= 5
    assert results[1]["n_tokens"] == 11
    assert results[2]["n_tokens"] == 17
    for sid in sessions:
        _delete(base, sid)


def test_capacity_exhaustion_and_slot_reuse(base):
    pool = _usage(base)["pool"]
    free = pool["free_sequences"]
    created: list[str] = []
    try:
        for _ in range(free):
            response = _post(base, "/sessions")
            assert response.status_code == 200, response.text
            created.append(response.json()["session_id"])
        exhausted = _post(base, "/sessions")
        assert exhausted.status_code == 503, exhausted.text
        assert "capacity" in exhausted.json()["error"].lower()

        released = created.pop()
        _delete(base, released)
        recovered = _post(base, "/sessions")
        assert recovered.status_code == 200, recovered.text
        replacement = recovered.json()["session_id"]
        assert replacement != released
        created.append(replacement)
        _inject(base, replacement, "no stale token leakage")
    finally:
        for sid in created:
            _delete(base, sid)


@pytest.mark.parametrize("_attempt", range(12))
def test_delete_vs_fork_returns_only_linearizable_responses(base, make_session, _attempt):
    sid = make_session()
    injected = _inject(base, sid, "Atomic fork reservation sentinel. ")
    barrier = threading.Barrier(2)

    def fork_now() -> requests.Response:
        barrier.wait(timeout=30)
        return _post(base, f"/sessions/{sid}/fork")

    def delete_now() -> requests.Response:
        barrier.wait(timeout=30)
        return requests.delete(f"{base}/sessions/{sid}", timeout=30)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fork_future = executor.submit(fork_now)
        delete_future = executor.submit(delete_now)
        forked = fork_future.result(timeout=30)
        deleted = delete_future.result(timeout=30)

    assert forked.status_code in {200, 404}, forked.text
    assert deleted.status_code in {200, 409}, deleted.text
    if forked.status_code == 200:
        body = forked.json()
        assert body["cache_size"] == injected["cache_size"]
        _delete(base, body["session_id"])
    else:
        assert forked.json()["error"]
    if deleted.status_code == 409:
        assert forked.status_code == 200


@pytest.mark.parametrize("operation", ["status", "load"])
@pytest.mark.parametrize("_attempt", range(12))
def test_delete_vs_live_snapshot_never_reads_released_sequence(
    base, make_session, operation, _attempt
):
    sid = make_session()
    injected = _inject(base, sid, "Atomic live snapshot sentinel. ")
    barrier = threading.Barrier(2)

    def inspect_now() -> requests.Response:
        barrier.wait(timeout=30)
        if operation == "status":
            return requests.get(f"{base}/sessions/{sid}", timeout=30)
        return _post(base, f"/sessions/{sid}/load")

    def delete_now() -> requests.Response:
        barrier.wait(timeout=30)
        return requests.delete(f"{base}/sessions/{sid}", timeout=30)

    with ThreadPoolExecutor(max_workers=2) as executor:
        inspect_future = executor.submit(inspect_now)
        delete_future = executor.submit(delete_now)
        inspected = inspect_future.result(timeout=30)
        deleted = delete_future.result(timeout=30)

    assert inspected.status_code in {200, 404}, inspected.text
    assert deleted.status_code in {200, 409}, deleted.text
    if inspected.status_code == 200:
        body = inspected.json()
        assert body["session_id"] == sid
        assert body["cache_size"] == injected["cache_size"]
    else:
        assert inspected.json()["error"]
    if deleted.status_code == 409:
        assert inspected.status_code == 200


def test_same_session_conflict_is_409(base, make_session):
    sid = make_session()
    _inject(base, sid, "Write a very long response. ")

    first_token = threading.Event()

    def generate() -> None:
        response = requests.post(
            f"{base}/sessions/{sid}/generate",
            json={"stream": True, "max_tokens": 256, "temperature": 0.0, "ignore_eos": True},
            stream=True,
            timeout=180,
        )
        assert response.status_code == 200, response.text
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if line and line.startswith("data: ") and '"type":"token"' in line:
                first_token.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(generate)
        assert first_token.wait(30), "generation did not begin streaming"
        conflict = _post(base, f"/sessions/{sid}/inject", json={"text": "conflict"})
        assert conflict.status_code == 409, conflict.text
        _post(base, f"/sessions/{sid}/cancel")
        future.result(timeout=60)


def test_shared_prefix_owner_release_in_both_orders(base, make_session):
    control = make_session()
    _inject(base, control, "Shared prefix lifecycle. ")
    expected = _generate(base, control, max_tokens=8)["text"]

    for release_source_first in (True, False):
        source = make_session()
        _inject(base, source, "Shared prefix lifecycle. ")
        forked = _fork(base, source)["session_id"]
        survivor = forked if release_source_first else source
        released = source if release_source_first else forked
        _delete(base, released)
        assert _generate(base, survivor, max_tokens=8)["text"] == expected
        if survivor != source:  # make_session owns sources; explicitly clean fork survivor
            _delete(base, survivor)
