"""Unit tests for simplified DspVad (no segmentation, always-feed).

Tests:
  - test_always_forwards_chunks: speech_chunk_callback called for every chunk
  - test_vad_state_tracking: VAD state transitions work for UI meter
  - test_no_segments_returned: process_chunk always returns None
"""

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import CaptureConfig
from core.dsp_vad import DspVad


class MockVADIterator:
    def __init__(self, sequence: list[tuple[int, dict | None]]):
        self.sequence = list(sequence)
        self.idx = 0
        self.chunks_in_current = 0

    def __call__(self, chunk, return_seconds=True):
        if self.idx >= len(self.sequence):
            return None
        n_chunks, value = self.sequence[self.idx]
        self.chunks_in_current += 1
        if self.chunks_in_current >= n_chunks:
            self.chunks_in_current = 0
            self.idx += 1
            return value
        return None

    def reset_states(self):
        pass


def make_silent_chunk(config: CaptureConfig):
    base = int(config.CHUNK_SIZE * 48000 / config.TARGET_SAMPLE_RATE)
    return np.zeros(base * 2, dtype=np.int16).tobytes()


def test_always_forwards_chunks():
    """speech_chunk_callback called for chunks regardless of VAD state."""
    config = CaptureConfig()
    config.DEBUG_SAVE_WAV = False

    mock = MockVADIterator([
        (100, None),
    ])

    chunk_calls = []

    def on_chunk(chunk):
        chunk_calls.append(chunk)

    dsp = DspVad(raw_buffer=deque(), config=config,
                 device_rate=48000, device_channels=2)
    dsp.set_speech_chunk_callback(on_chunk)
    dsp.initialize()
    dsp.vad_iterator = mock

    silent = make_silent_chunk(config)
    for _ in range(20):
        dsp.process_chunk(silent)

    # Resampler accumulator may need multiple input chunks per output chunk,
    # but over 20 iterations at least some chunks should be forwarded
    assert len(chunk_calls) >= 5, (
        f"Expected >= 5 chunk calls forwarded, got {len(chunk_calls)}"
    )
    print(f"[PASS] test_always_forwards_chunks: {len(chunk_calls)} chunks forwarded in 20 iterations")


def test_vad_state_tracking():
    """VAD state transitions work correctly for UI meter."""
    config = CaptureConfig()
    config.DEBUG_SAVE_WAV = False

    mock = MockVADIterator([
        (3, None),           # No speech (IDLE)
        (1, {"start": 0.0}), # SPEECH start
        (5, None),           # SPEECH continues
        (1, {"end": 0.0}),   # PENDING_FINALIZE
        (10, None),          # Silence → should go back to IDLE after 3s
    ])

    dsp = DspVad(raw_buffer=deque(), config=config,
                 device_rate=48000, device_channels=2)
    dsp.initialize()
    dsp.vad_iterator = mock

    silent = make_silent_chunk(config)
    states = []

    for _ in range(50):
        dsp.process_chunk(silent)
        states.append(dsp.state)

    # Eventually SPEECH should appear
    assert "SPEECH" in states, (
        f"Expected SPEECH state to appear, states seen: {set(states)}"
    )
    # Eventually PENDING_FINALIZE should appear
    assert "PENDING_FINALIZE" in states, (
        f"Expected PENDING_FINALIZE state to appear, states seen: {set(states)}"
    )
    # After enough silence, should return to IDLE
    final_state = states[-1]
    assert final_state == "IDLE" or final_state == "PENDING_FINALIZE", (
        f"Expected final state IDLE or PENDING_FINALIZE, got {final_state}"
    )
    print(f"[PASS] test_vad_state_tracking: states={set(states)}")


def test_no_segments_returned():
    """process_chunk always returns None (no segmentation)."""
    config = CaptureConfig()
    config.DEBUG_SAVE_WAV = False

    mock = MockVADIterator([
        (3, None),
        (1, {"start": 0.0}),
        (3, None),
        (1, {"end": 0.0}),
        (10, None),
    ])

    dsp = DspVad(raw_buffer=deque(), config=config,
                 device_rate=48000, device_channels=2)
    dsp.initialize()
    dsp.vad_iterator = mock

    silent = make_silent_chunk(config)
    results = []

    total_chunks = 3 + 1 + 3 + 1 + 10 + 5
    for _ in range(total_chunks):
        result = dsp.process_chunk(silent)
        results.append(result)

    non_none = [r for r in results if r is not None]
    assert len(non_none) == 0, (
        f"Expected 0 non-None results (no segmentation), got {len(non_none)}"
    )
    print(f"[PASS] test_no_segments_returned: all {len(results)} results are None")


if __name__ == "__main__":
    test_always_forwards_chunks()
    test_vad_state_tracking()
    test_no_segments_returned()
    print("\nAll VAD tests PASSED")
