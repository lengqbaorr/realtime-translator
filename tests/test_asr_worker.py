"""Unit tests for sherpa-onnx endpoint-driven AsrWorker.

Tests:
  - test_initial_stream_gets_leading_context: first live stream gets context only
  - test_partial_emitted_during_streaming: live chunks emit partial results
  - test_endpoint_emits_final_and_resets: sherpa endpoint emits final + reset()
  - test_endpoint_final_sequence_after_partials: final seq follows partial seqs
  - test_short_endpoint_is_suppressed: one-word short endpoint keeps streaming
  - test_soft_boundary_commits_delta_without_reset: pause commits text segment
  - test_soft_boundary_suppresses_unstable_tail: incomplete tails wait for later
  - test_soft_boundary_retries_until_tail_stabilizes: retry commits stable text
  - test_live_chunks_dropped_under_backpressure: bounded queue drops old chunks
  - test_slow_result_callback_does_not_block_decode: callback runs off decode thread
  - test_partial_skip_on_duplicate_text: duplicate partial text is suppressed
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.asr_worker import AsrConfig, AsrWorker


class MockOnlineStream:
    """Simulates a sherpa-onnx OnlineStream for testing."""

    def __init__(self):
        self._chunks = []
        self._finished = False

    def accept_waveform(self, sample_rate: int, audio: np.ndarray):
        if not self._finished:
            self._chunks.append(audio)

    def input_finished(self):
        self._finished = True


class MockOnlineRecognizer:
    """Simulates a sherpa-onnx OnlineRecognizer for tests."""

    def __init__(self, endpoint_after_chunks: int | None = None):
        self._streams = []
        self._result_calls = 0
        self.endpoint_after_chunks = endpoint_after_chunks
        self.reset_calls = 0

    def create_stream(self):
        stream = MockOnlineStream()
        self._streams.append(stream)
        return stream

    def is_ready(self, stream):
        return False

    def decode_stream(self, stream):
        pass

    def get_result(self, stream):
        if stream._finished:
            return "hello world final"
        self._result_calls += 1
        return f"hello world {self._result_calls}"

    def is_endpoint(self, stream):
        if self.endpoint_after_chunks is None:
            return False
        live_chunks = [chunk for chunk in stream._chunks if len(chunk) == 512]
        return len(live_chunks) >= self.endpoint_after_chunks

    def reset(self, stream):
        self.reset_calls += 1
        stream._chunks.clear()
        stream._finished = False
        return True


def make_chunk():
    return np.zeros(512, dtype=np.float32)


def make_worker(recognizer, results, queue_size=60, partial_ms=50):
    worker = AsrWorker(
        config=AsrConfig(
            asr_queue_maxsize=queue_size,
            partial_update_interval_ms=partial_ms,
        ),
        result_callback=lambda r: results.append(r),
    )
    worker._recognizer = recognizer
    worker.start()
    return worker


def test_initial_stream_gets_leading_context():
    """First live stream gets short context, not the disposable warmup."""
    results = []
    recognizer = MockOnlineRecognizer(endpoint_after_chunks=None)
    worker = AsrWorker(
        config=AsrConfig(
            asr_queue_maxsize=60,
            initial_stream_context_ms=500,
        ),
        result_callback=lambda r: results.append(r),
    )
    worker._recognizer = recognizer
    worker.start()

    live_stream = worker._stream
    worker.stop()

    assert len(recognizer._streams) >= 2, (
        "Expected one warmup stream and one live stream"
    )
    assert live_stream is recognizer._streams[-1], (
        "Live stream should be created after the warmup stream"
    )
    assert len(live_stream._chunks) == 1, (
        f"Expected one leading-context chunk, got {len(live_stream._chunks)}"
    )
    assert len(live_stream._chunks[0]) == 8000, (
        f"Expected 500ms context, got {len(live_stream._chunks[0])} samples"
    )
    print("[PASS] test_initial_stream_gets_leading_context")


def test_endpoint_reset_does_not_add_leading_context_again():
    """Leading context is only for the first stream, not after endpoint reset."""
    results = []
    recognizer = MockOnlineRecognizer(endpoint_after_chunks=2)
    worker = AsrWorker(
        config=AsrConfig(
            asr_queue_maxsize=60,
            initial_stream_context_ms=500,
            partial_update_interval_ms=50,
        ),
        result_callback=lambda r: results.append(r),
    )
    worker._recognizer = recognizer
    worker.start()

    stream = worker._stream
    worker.submit_live_chunk(make_chunk())
    worker.submit_live_chunk(make_chunk())
    time.sleep(0.3)
    worker.stop()

    assert recognizer.reset_calls == 1, (
        f"Expected one endpoint reset, got {recognizer.reset_calls}"
    )
    assert stream is worker._stream, "Endpoint reset should reuse the same stream"
    assert len(stream._chunks) == 0, (
        "Mock reset should clear chunks and no new leading context should be added"
    )
    print("[PASS] test_endpoint_reset_does_not_add_leading_context_again")


def test_partial_emitted_during_streaming():
    """Submit live chunks without endpoint -> partials appear, no final."""
    results = []
    recognizer = MockOnlineRecognizer(endpoint_after_chunks=None)
    worker = make_worker(recognizer, results)

    for _ in range(3):
        worker.submit_live_chunk(make_chunk())
        time.sleep(0.08)

    time.sleep(0.2)
    worker.stop()

    partials = [r for r in results if not r.is_final]
    finals = [r for r in results if r.is_final]

    assert len(partials) >= 1, (
        f"Expected at least 1 partial, got {len(partials)}"
    )
    assert len(finals) == 0, (
        f"Expected no final without sherpa endpoint, got {len(finals)}"
    )
    print(f"[PASS] test_partial_emitted_during_streaming: {len(partials)} partials")


def test_endpoint_emits_final_and_resets():
    """Sherpa endpoint -> final result and recognizer.reset(stream)."""
    results = []
    recognizer = MockOnlineRecognizer(endpoint_after_chunks=2)
    worker = make_worker(recognizer, results)
    first_stream = worker._stream

    worker.submit_live_chunk(make_chunk())
    worker.submit_live_chunk(make_chunk())
    time.sleep(0.3)
    worker.stop()

    finals = [r for r in results if r.is_final]

    assert len(finals) == 1, (
        f"Expected 1 final from endpoint, got {len(finals)}"
    )
    assert not first_stream._finished, (
        "Endpoint handling should not call input_finished for live streaming"
    )
    assert recognizer.reset_calls == 1, (
        f"Expected recognizer.reset once, got {recognizer.reset_calls}"
    )
    assert worker._stream is first_stream, (
        "Endpoint reset should keep the same OnlineStream object"
    )
    assert finals[0].speech_ms == 64.0
    assert finals[0].total_ms == 64.0
    print("[PASS] test_endpoint_emits_final_and_resets")


def test_endpoint_final_sequence_after_partials():
    """Final emitted by endpoint should have sequence after prior partials."""
    results = []
    recognizer = MockOnlineRecognizer(endpoint_after_chunks=4)
    worker = make_worker(recognizer, results)

    for _ in range(4):
        worker.submit_live_chunk(make_chunk())
        time.sleep(0.08)

    time.sleep(0.3)
    worker.stop()

    partials = [r for r in results if not r.is_final]
    finals = [r for r in results if r.is_final]

    assert len(finals) == 1, (
        f"Expected 1 endpoint final, got {len(finals)}"
    )
    for partial in partials:
        assert partial.sequence < finals[0].sequence, (
            f"Partial seq {partial.sequence} >= final seq {finals[0].sequence}"
        )
    print(f"[PASS] test_endpoint_final_sequence_after_partials: "
          f"{len(partials)} partials")


def test_short_endpoint_is_suppressed():
    """One-word short endpoint should not reset the stream."""
    results = []

    class ShortRecognizer(MockOnlineRecognizer):
        def get_result(self, stream):
            return "now"

    recognizer = ShortRecognizer(endpoint_after_chunks=2)
    worker = AsrWorker(
        config=AsrConfig(
            asr_queue_maxsize=60,
            partial_update_interval_ms=50,
            min_endpoint_duration_ms=2500,
            min_endpoint_words=3,
        ),
        result_callback=lambda r: results.append(r),
    )
    worker._recognizer = recognizer
    worker.start()

    worker.submit_live_chunk(make_chunk())
    worker.submit_live_chunk(make_chunk())
    time.sleep(0.3)
    worker.stop()

    finals = [r for r in results if r.is_final]

    assert recognizer.reset_calls == 0, (
        f"Expected no reset for short endpoint, got {recognizer.reset_calls}"
    )
    assert len(finals) == 0, (
        f"Expected no final for short endpoint, got {len(finals)}"
    )
    print("[PASS] test_short_endpoint_is_suppressed")


def test_soft_boundary_commits_delta_without_reset():
    """Soft pause boundary commits current text without recognizer reset."""
    results = []

    class TextRecognizer(MockOnlineRecognizer):
        def get_result(self, stream):
            return "HELLO WORLD THIS IS A TEST"

    recognizer = TextRecognizer(endpoint_after_chunks=None)
    worker = AsrWorker(
        config=AsrConfig(
            asr_queue_maxsize=60,
            partial_update_interval_ms=10000,
            soft_boundary_commit_delay_ms=10,
            soft_boundary_retry_interval_ms=10,
            soft_boundary_max_wait_ms=50,
        ),
        result_callback=lambda r: results.append(r),
    )
    worker._recognizer = recognizer
    worker.start()

    worker.submit_live_chunk(make_chunk())
    time.sleep(0.1)
    worker.submit_soft_boundary("comma", 420.0)
    time.sleep(0.2)
    worker.stop()

    finals = [r for r in results if r.is_final]

    assert recognizer.reset_calls == 0, (
        f"Expected no reset for soft boundary, got {recognizer.reset_calls}"
    )
    assert len(finals) == 1, (
        f"Expected one committed segment, got {len(finals)}"
    )
    assert finals[0].text == "HELLO WORLD THIS IS A TEST,"
    print("[PASS] test_soft_boundary_commits_delta_without_reset")


def test_soft_boundary_suppresses_unstable_tail():
    """Soft boundary should not commit text ending in a connector fragment."""
    results = []

    class TextRecognizer(MockOnlineRecognizer):
        def get_result(self, stream):
            return "CORROSION OF STEEL STRUCTURES SUCH AS BRIDGES AND"

    recognizer = TextRecognizer(endpoint_after_chunks=None)
    worker = AsrWorker(
        config=AsrConfig(
            asr_queue_maxsize=60,
            partial_update_interval_ms=10000,
            soft_boundary_commit_delay_ms=10,
        ),
        result_callback=lambda r: results.append(r),
    )
    worker._recognizer = recognizer
    worker.start()

    worker.submit_live_chunk(make_chunk())
    time.sleep(0.1)
    worker.submit_soft_boundary("comma", 700.0)
    time.sleep(0.2)
    worker.stop()

    finals = [r for r in results if r.is_final]
    assert len(finals) == 0, (
        f"Expected no committed segment for unstable tail, got {len(finals)}"
    )
    print("[PASS] test_soft_boundary_suppresses_unstable_tail")


def test_soft_boundary_retries_until_tail_stabilizes():
    """Pending boundary should commit after ASR tail becomes stable."""
    results = []

    class StabilizingRecognizer(MockOnlineRecognizer):
        def __init__(self):
            super().__init__(endpoint_after_chunks=None)
            self.calls = 0

        def get_result(self, stream):
            self.calls += 1
            if self.calls < 3:
                return "WELCOME CLASS DOCTOR T"
            return "WELCOME CLASS DOCTOR TURBAN DARL"

    recognizer = StabilizingRecognizer()
    worker = AsrWorker(
        config=AsrConfig(
            asr_queue_maxsize=60,
            partial_update_interval_ms=10000,
            soft_boundary_commit_delay_ms=10,
            soft_boundary_retry_interval_ms=10,
            soft_boundary_max_wait_ms=200,
        ),
        result_callback=lambda r: results.append(r),
    )
    worker._recognizer = recognizer
    worker.start()

    worker.submit_live_chunk(make_chunk())
    time.sleep(0.1)
    worker.submit_soft_boundary("sentence", 720.0)
    time.sleep(0.3)
    worker.stop()

    finals = [r for r in results if r.is_final]
    assert len(finals) == 1, (
        f"Expected one committed segment after stabilization, got {len(finals)}"
    )
    assert finals[0].text == "WELCOME CLASS DOCTOR TURBAN DARL."
    print("[PASS] test_soft_boundary_retries_until_tail_stabilizes")


def test_live_chunks_dropped_under_backpressure():
    """Queue full of chunks -> old live chunks are dropped."""
    results = []
    recognizer = MockOnlineRecognizer(endpoint_after_chunks=None)
    worker = make_worker(recognizer, results, queue_size=3)

    original_process_chunk = worker._process_chunk

    def slow_chunk(chunk):
        time.sleep(0.3)
        original_process_chunk(chunk)

    worker._process_chunk = slow_chunk

    for _ in range(8):
        worker.submit_live_chunk(make_chunk())
        time.sleep(0.01)

    time.sleep(1.2)
    worker.stop()

    assert worker.dropped_live_chunks > 0, (
        f"Expected dropped_live_chunks > 0, got {worker.dropped_live_chunks}"
    )
    print(f"[PASS] test_live_chunks_dropped_under_backpressure: "
          f"dropped={worker.dropped_live_chunks}")


def test_slow_result_callback_does_not_block_decode():
    """Slow result callback should not block endpoint/reset in decode thread."""
    results = []

    def slow_callback(result):
        time.sleep(0.5)
        results.append(result)

    recognizer = MockOnlineRecognizer(endpoint_after_chunks=2)
    worker = AsrWorker(
        config=AsrConfig(
            asr_queue_maxsize=60,
            result_queue_maxsize=10,
            partial_update_interval_ms=50,
        ),
        result_callback=slow_callback,
    )
    worker._recognizer = recognizer
    worker.start()

    t0 = time.time()
    worker.submit_live_chunk(make_chunk())
    worker.submit_live_chunk(make_chunk())
    time.sleep(0.2)
    elapsed = time.time() - t0

    worker.stop()

    assert recognizer.reset_calls == 1, (
        f"Expected endpoint reset despite slow callback, got {recognizer.reset_calls}"
    )
    assert elapsed < 0.45, (
        f"Decode path appears blocked by slow callback: elapsed={elapsed:.2f}s"
    )
    assert any(r.is_final for r in results), "Expected final result to be delivered"
    print("[PASS] test_slow_result_callback_does_not_block_decode")


def test_partial_skip_on_duplicate_text():
    """Same partial text should not be emitted twice."""
    results = []

    class ConstantRecognizer(MockOnlineRecognizer):
        def get_result(self, stream):
            return "hello partial now"

    recognizer = ConstantRecognizer(endpoint_after_chunks=4)
    worker = make_worker(recognizer, results)

    for _ in range(4):
        worker.submit_live_chunk(make_chunk())
        time.sleep(0.08)

    time.sleep(0.3)
    worker.stop()

    partials = [r for r in results if not r.is_final]
    finals = [r for r in results if r.is_final]

    assert len(partials) <= 1, (
        f"Expected at most 1 duplicate-suppressed partial, got {len(partials)}"
    )
    assert len(finals) == 1, (
        f"Expected 1 endpoint final, got {len(finals)}"
    )
    assert finals[0].text == "hello partial now."
    print("[PASS] test_partial_skip_on_duplicate_text")


if __name__ == "__main__":
    test_initial_stream_gets_leading_context()
    test_endpoint_reset_does_not_add_leading_context_again()
    test_partial_emitted_during_streaming()
    test_endpoint_emits_final_and_resets()
    test_endpoint_final_sequence_after_partials()
    test_short_endpoint_is_suppressed()
    test_soft_boundary_commits_delta_without_reset()
    test_soft_boundary_suppresses_unstable_tail()
    test_soft_boundary_retries_until_tail_stabilizes()
    test_live_chunks_dropped_under_backpressure()
    test_slow_result_callback_does_not_block_decode()
    test_partial_skip_on_duplicate_text()
    print("\nAll ASR worker tests PASSED")
