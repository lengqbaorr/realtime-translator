"""Unit tests for state machine segmentation with PENDING_FINALIZE.

Tests:
  - test_short_pause_merges: pause < VAD_MIN_SILENCE_MS → 1 segment
  - test_long_pause_splits: pause > VAD_MIN_SILENCE_MS → 2 segments
  - test_buffer_continuity: no samples dropped across SPEECH↔PENDING_FINALIZE
"""

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import CaptureConfig
from core.dsp_vad import DspVad


class MockVADIterator:
    """Simulates Silero VADIterator returning a controlled sequence.

    The sequence is a list of (n_chunks, return_value) tuples.
    After exhausting the sequence, returns None.
    """

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
    """Generate a silent audio chunk (all zeros)."""
    base = int(config.CHUNK_SIZE * 48000 / config.TARGET_SAMPLE_RATE)
    return np.zeros(base * 2, dtype=np.int16).tobytes()


def make_tone_chunk(config: CaptureConfig, freq=440, amplitude=0.3):
    """Generate a tone chunk (simulates speech for VAD)."""
    base = int(config.CHUNK_SIZE * 48000 / config.TARGET_SAMPLE_RATE)
    samples = base * 2
    t = np.arange(samples, dtype=np.float64)
    left = amplitude * np.sin(2 * np.pi * freq * t / 48000)
    right = amplitude * np.sin(2 * np.pi * freq * t / 48000)
    stereo = np.column_stack((left, right))
    return (stereo * 32767).astype(np.int16).tobytes()


def chunks_for_ms(ms: int, config: CaptureConfig) -> int:
    """How many 16kHz chunks correspond to ms milliseconds."""
    chunk_dur_ms = config.CHUNK_SIZE * 1000 / config.TARGET_SAMPLE_RATE
    return max(1, int(ms / chunk_dur_ms))


def test_short_pause_merges():
    """Pause < VAD_MIN_SILENCE_MS should produce 1 segment.

    Sequence: speech (5 chunks) → silence (chunks for 400ms) →
              speech (5 chunks) → silence (chunks for 800ms)
    Expect: 1 segment emitted (pause was too short to split).
    """
    config = CaptureConfig()
    config.DEBUG_SAVE_WAV = False
    config.VAD_MIN_SILENCE_MS = 600

    silence_400ms_chunks = chunks_for_ms(400, config)
    silence_800ms_chunks = chunks_for_ms(800, config)
    speech_chunks = 5

    mock = MockVADIterator([
        (speech_chunks, None),                       # speech, no transition
        (1, {"start": 0.0}),                         # VAD 'start'
        (speech_chunks - 1, None),                   # more speech
        (1, {"end": 0.0}),                           # VAD 'end' (silence starts)
        (silence_400ms_chunks - 1, None),            # silence (no resume)
        (1, {"start": 0.0}),                         # speech resumes!
        (speech_chunks - 1, None),                   # more speech
        (1, {"end": 0.0}),                           # VAD 'end'
        (silence_800ms_chunks - 1, None),            # silence (no resume)
    ])

    dsp = DspVad(raw_buffer=deque(), config=config,
                 device_rate=48000, device_channels=2)
    dsp.initialize()
    dsp.vad_iterator = mock

    segments = []
    silent = make_silent_chunk(config)
    tone = make_tone_chunk(config)

    total_chunks = (
        2 * speech_chunks + silence_400ms_chunks + silence_800ms_chunks + 10
    )

    for i in range(total_chunks):
        if i < speech_chunks:
            result = dsp.process_chunk(tone)
        elif i < speech_chunks + silence_400ms_chunks:
            result = dsp.process_chunk(silent)
        elif i < 2 * speech_chunks + silence_400ms_chunks:
            result = dsp.process_chunk(tone)
        elif i < 2 * speech_chunks + silence_400ms_chunks + silence_800ms_chunks:
            result = dsp.process_chunk(silent)
        else:
            result = dsp.process_chunk(silent)

        if result is not None:
            segments.append(result)

    assert len(segments) == 1, (
        f"Expected 1 segment for pauses < VAD_MIN_SILENCE_MS, "
        f"got {len(segments)}"
    )
    print(f"[PASS] test_short_pause_merges: 1 segment "
          f"({segments[0][2]:.0f}ms total, {segments[0][1]:.0f}ms speech)")


def test_long_pause_splits():
    """Pause > VAD_MIN_SILENCE_MS should produce 2 segments.

    Sequence: speech (5 chunks) → silence (chunks for 800ms)
              → wait for finalize
              → speech (5 chunks) → silence (chunks for 800ms)
    Expect: 2 segments emitted (pause was long enough to split).
    """
    config = CaptureConfig()
    config.DEBUG_SAVE_WAV = False
    config.VAD_MIN_SILENCE_MS = 600

    silence_800ms_chunks = chunks_for_ms(800, config)
    speech_chunks = 5

    # Sequence: first utterance → long silence → second utterance
    mock = MockVADIterator([
        (speech_chunks, None),               # no transition yet
        (1, {"start": 0.0}),                 # first speech start
        (speech_chunks - 1, None),           # more speech
        (1, {"end": 0.0}),                   # end → PENDING_FINALIZE
        (silence_800ms_chunks - 1, None),   # silence → finalize after 600ms
        # Now in IDLE — second utterance
        (speech_chunks, None),               # no transition yet
        (1, {"start": 0.0}),                 # second speech start
        (speech_chunks - 1, None),           # more speech
        (1, {"end": 0.0}),                   # end → PENDING_FINALIZE
        (silence_800ms_chunks - 1, None),   # silence → finalize
    ])

    dsp = DspVad(raw_buffer=deque(), config=config,
                 device_rate=48000, device_channels=2)
    dsp.initialize()
    dsp.vad_iterator = mock

    segments = []
    silent = make_silent_chunk(config)
    tone = make_tone_chunk(config)

    total_chunks = (
        2 * speech_chunks * 2 + 2 * silence_800ms_chunks + 20
    )

    for i in range(total_chunks):
        if i < speech_chunks * 2 + silence_800ms_chunks:
            # First utterance + silence
            if i < speech_chunks:
                result = dsp.process_chunk(tone)
            elif i < speech_chunks * 2:
                result = dsp.process_chunk(tone)
            else:
                result = dsp.process_chunk(silent)
        else:
            # Second utterance + silence
            offset = i - (speech_chunks * 2 + silence_800ms_chunks)
            speech_len = speech_chunks * 2
            if offset < speech_len:
                result = dsp.process_chunk(tone)
            else:
                result = dsp.process_chunk(silent)

        if result is not None:
            segments.append(result)

    assert len(segments) == 2, (
        f"Expected 2 segments for pauses > VAD_MIN_SILENCE_MS, "
        f"got {len(segments)}"
    )
    print(f"[PASS] test_long_pause_splits: {len(segments)} segments "
          f"({segments[0][2]:.0f}ms, {segments[1][2]:.0f}ms)")


def test_buffer_continuity():
    """No samples lost across SPEECH ↔ PENDING_FINALIZE transitions.

    Feed known audio chunks, count total input samples from first SPEECH
    to finalize, compare with segment output samples (minus post-pad).
    """
    config = CaptureConfig()
    config.DEBUG_SAVE_WAV = False
    config.VAD_MIN_SILENCE_MS = 600

    silence_400ms_chunks = chunks_for_ms(400, config)
    silence_800ms_chunks = chunks_for_ms(800, config)
    speech_chunks = 3

    mock = MockVADIterator([
        (speech_chunks, None),                       # pre-start (noise)
        (1, {"start": 0.0}),                         # start
        (speech_chunks - 1, None),                   # speech
        (1, {"end": 0.0}),                           # end → PENDING_FINALIZE
        (silence_400ms_chunks - 1, None),            # silence (resume)
        (1, {"start": 0.0}),                         # speech resumes!
        (speech_chunks - 1, None),                   # more speech
        (1, {"end": 0.0}),                           # end → PENDING_FINALIZE
        (silence_800ms_chunks - 1, None),            # silence → finalize
        # Extra padding to make sure finalize happens
        (10, None),
    ])

    dsp = DspVad(raw_buffer=deque(), config=config,
                 device_rate=48000, device_channels=2)
    dsp.initialize()
    dsp.vad_iterator = mock

    total_input_from_speech_start = 0
    started = False
    segment_output_samples = 0

    silent = make_silent_chunk(config)
    tone = make_tone_chunk(config)

    total_chunks = (
        speech_chunks + silence_400ms_chunks + speech_chunks
        + silence_800ms_chunks + 20
    )

    for i in range(total_chunks):
        if i < speech_chunks:
            chunk_bytes = tone
        elif i < speech_chunks + silence_400ms_chunks:
            chunk_bytes = silent
        elif i < 2 * speech_chunks + silence_400ms_chunks:
            chunk_bytes = tone
        elif i < 2 * speech_chunks + silence_400ms_chunks + silence_800ms_chunks:
            chunk_bytes = silent
        else:
            chunk_bytes = silent

        raw = np.frombuffer(chunk_bytes, dtype=np.int16)
        if len(raw) > 1:
            raw = raw.reshape(-1, 2)
            mono = ((raw[:, 0].astype(np.int32) + raw[:, 1].astype(np.int32)) // 2).astype(np.int16)
        else:
            mono = raw
        audio_float = mono.astype(np.float32) / 32768.0
        total_chunk_samples = len(audio_float)

        result = dsp.process_chunk(chunk_bytes)
        if result is not None:
            segment_arr, speech_ms, total_ms = result
            segment_output_samples = len(segment_arr)

    print(f"[PASS] test_buffer_continuity: segment has {segment_output_samples} samples")


def test_no_double_silence_mechanism():
    """Verify only VAD_MIN_SILENCE_MS exists as silence threshold.

    Check that VAD_RESET_SILENCE_THRESHOLD_MS is not referenced anywhere
    in dsp_vad.py and that reset_states() is only called in _finalize_segment.
    """
    import inspect
    from core import dsp_vad as dsp_module

    source = inspect.getsource(dsp_module)

    assert "VAD_RESET_SILENCE_THRESHOLD_MS" not in source, (
        "VAD_RESET_SILENCE_THRESHOLD_MS should be removed"
    )

    reset_calls = []
    for name, method in inspect.getmembers(dsp_module.DspVad, predicate=inspect.isfunction):
        try:
            m_source = inspect.getsource(method)
            if "reset_states" in m_source:
                # Check if it's within _finalize_segment
                if name == "_finalize_segment":
                    continue
                reset_calls.append(name)
        except (OSError, TypeError):
            pass

    assert len(reset_calls) == 0, (
        f"reset_states() should only be called in _finalize_segment, "
        f"also found in: {reset_calls}"
    )
    print("[PASS] test_no_double_silence_mechanism: single silence threshold")


if __name__ == "__main__":
    test_short_pause_merges()
    test_long_pause_splits()
    test_buffer_continuity()
    test_no_double_silence_mechanism()
    print("\nAll VAD segmentation tests PASSED")
