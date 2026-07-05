"""Unit tests for AsrWorker — uses MockAsrModel, no real model needed.

Tests:
  - test_submit_and_process: 1 segment → callback called once with sequence=0
  - test_sequence_ordering: multiple segments → ordered sequences
  - test_backpressure_drop_oldest: queue full → oldest dropped
  - test_low_confidence_flagged_not_dropped: low logprob → flagged, not dropped
"""

import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.asr_worker import AsrConfig, AsrResult, AsrWorker
from core.config import CaptureConfig

import numpy as np


class MockSegment:
    """Mimics faster_whisper.model.Segment for mock _transcribe."""

    def __init__(self, text, avg_logprob=0.0, no_speech_prob=0.0):
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class MockInfo:
    """Mimics faster_whisper.model.TranscriptionInfo."""

    def __init__(self):
        self.language = "en"
        self.duration = 0.0


class MockWhisperModel:
    """Mock for faster_whisper.WhisperModel.

    Returns a fixed list of segments on transcribe().
    """

    def __init__(self, size="base.en", device="cpu", compute_type="int8"):
        self.loaded = True
        self._segments = [
            MockSegment("hello world this is a test", -0.3, 0.02),
        ]
        self._info = MockInfo()

    def transcribe(self, audio, **kwargs):
        class SegmentsIterable:
            def __init__(self, segs, info):
                self._segs = segs
                self._info = info

            def __iter__(self):
                return iter(self._segs)

            @property
            def info(self):
                return self._info

        return SegmentsIterable(self._segments, self._info), self._info


def make_silent_segment(config=None):
    """Generate a silent float32 mono segment @ 16kHz (~1s)."""
    if config is None:
        config = CaptureConfig()
    samples = config.TARGET_SAMPLE_RATE
    return np.zeros(samples, dtype=np.float32)


def make_tone_segment(config=None):
    """Generate a tone segment (simulates speech)."""
    if config is None:
        config = CaptureConfig()
    sr = config.TARGET_SAMPLE_RATE
    t = np.arange(sr, dtype=np.float64)
    return (0.3 * np.sin(2 * np.pi * 440 * t / sr)).astype(np.float32)


def test_submit_and_process():
    """Submit 1 segment → result_callback called once with sequence=0."""
    results = []

    worker = AsrWorker(
        config=AsrConfig(),
        result_callback=lambda r: results.append(r),
    )
    worker._model = MockWhisperModel()
    worker.start()

    segment = make_silent_segment()
    worker.submit_segment(segment, 500.0, 1000.0)

    time.sleep(0.5)
    worker.stop()

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0].sequence == 0, (
        f"Expected sequence=0, got {results[0].sequence}"
    )
    assert results[0].text == "hello world this is a test"
    assert results[0].speech_ms == 500.0
    assert results[0].total_ms == 1000.0
    assert results[0].is_low_confidence is False
    print("[PASS] test_submit_and_process")


def test_sequence_ordering():
    """Submit segments rapidly → ordered sequences preserved."""
    results = []

    worker = AsrWorker(
        config=AsrConfig(),
        result_callback=lambda r: results.append(r),
    )
    # Simulate fast worker: set asr_queue maxsize high, process instantly
    worker.config.ASR_QUEUE_MAXSIZE = 100
    worker._model = MockWhisperModel()
    worker.start()

    segment = make_silent_segment()
    num_segments = 5

    for i in range(num_segments):
        worker.submit_segment(segment, 100.0, 200.0)

    time.sleep(1.0)
    worker.stop()

    assert len(results) == num_segments, (
        f"Expected {num_segments} results, got {len(results)}"
    )

    sequences = [r.sequence for r in results]
    assert sequences == sorted(sequences), (
        f"Sequences out of order: {sequences}"
    )
    assert sequences == list(range(num_segments)), (
        f"Expected [0..{num_segments - 1}], got {sequences}"
    )
    print(f"[PASS] test_sequence_ordering: {sequences}")


def test_backpressure_drop_oldest():
    """Fill queue → oldest dropped, newest kept, counter incremented."""
    results = []

    worker = AsrWorker(
        config=AsrConfig(ASR_QUEUE_MAXSIZE=3),
        result_callback=lambda r: results.append(r),
    )
    worker._model = MockWhisperModel()

    # Override _worker_loop to be slow (block on queue, take 0.5s per item)
    original_loop = worker._worker_loop

    def slow_loop():
        while not worker._stop_event.is_set():
            try:
                item = worker._asr_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            time.sleep(0.5)
            seq, segment, speech_ms, total_ms = item
            result = worker._transcribe(segment, seq, speech_ms, total_ms)
            if worker.result_callback:
                worker.result_callback(result)

    worker._worker_loop = slow_loop
    worker.start()

    segment = make_silent_segment()

    # Submit more than queue can hold (queue maxsize=3)
    for i in range(6):
        worker.submit_segment(segment, 100.0, 200.0)
        time.sleep(0.01)

    time.sleep(0.3)
    worker.stop()

    assert worker.dropped_asr_segments > 0, (
        f"Expected dropped_asr_segments > 0, got {worker.dropped_asr_segments}"
    )
    assert len(results) > 0, "Expected at least 1 result"

    # The oldest segments (smallest sequences) should be dropped,
    # the newest (largest sequences) should survive.
    sequences = [r.sequence for r in results]
    max_seq = max(sequences) if sequences else 0
    min_seq = min(sequences) if sequences else 0
    assert min_seq > 0 or max_seq < 5, (
        f"Expected oldest dropped: sequences {sequences}"
    )
    print(f"[PASS] test_backpressure_drop_oldest: "
          f"dropped={worker.dropped_asr_segments}, "
          f"results={len(results)}, sequences={sequences}")


def test_low_confidence_flagged_not_dropped():
    """Low avg_logprob → is_low_confidence=True, still called back."""
    results = []

    class LowConfidenceMockModel:
        def __init__(self, size="base.en", device="cpu", compute_type="int8"):
            self.loaded = True

        def transcribe(self, audio, **kwargs):
            low_seg = MockSegment("noisy audio", -2.5, 0.8)
            info = MockInfo()

            class Iter:
                def __init__(self):
                    self._info = info

                def __iter__(self):
                    return iter([low_seg])

                @property
                def info(self):
                    return self._info

            return Iter(), info

    worker = AsrWorker(
        config=AsrConfig(LOW_CONFIDENCE_AVG_LOGPROB=-1.0),
        result_callback=lambda r: results.append(r),
    )
    worker._model = LowConfidenceMockModel()
    worker.start()

    segment = make_tone_segment()
    worker.submit_segment(segment, 300.0, 800.0)

    time.sleep(0.5)
    worker.stop()

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0].is_low_confidence is True, (
        "Expected is_low_confidence=True for logprob < threshold"
    )
    assert results[0].avg_logprob == -2.5
    assert results[0].no_speech_prob == 0.8
    print(f"[PASS] test_low_confidence_flagged_not_dropped: "
          f"text='{results[0].text}', is_low={results[0].is_low_confidence}")


if __name__ == "__main__":
    test_submit_and_process()
    test_sequence_ordering()
    test_backpressure_drop_oldest()
    test_low_confidence_flagged_not_dropped()
    print("\nAll ASR worker tests PASSED")
