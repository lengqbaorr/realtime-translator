import time
from collections import deque

import numpy as np
import soxr
from silero_vad import VADIterator, load_silero_vad

from core.config import CaptureConfig


class DspVad:
    """DSP processing + VAD state machine.

    State machine: IDLE → SPEECH → PENDING_FINALIZE → IDLE

    PENDING_FINALIZE is a hangover state: when VAD signals 'end',
    we don't finalize immediately. Instead we keep appending audio
    to the segment buffer and wait VAD_MIN_SILENCE_MS of continuous
    silence. If speech resumes within that window, we return to SPEECH
    seamlessly (buffer continuity guaranteed). This prevents premature
    segmentation during natural pauses (breath, hesitation, 300–800ms).

    Single source of truth for silence threshold: VAD_MIN_SILENCE_MS.
    VADIterator's native min_silence_duration_ms=0 (per-frame detection)
    — all timing is handled by PENDING_FINALIZE's hangover timer.

    reset_states() is called only at the moment of finalizing a segment,
    never independently.
    """

    def __init__(
        self,
        raw_buffer: deque,
        config: CaptureConfig,
        device_rate: int,
        device_channels: int,
    ):
        self.raw_buffer = raw_buffer
        self.config = config
        self.device_rate = device_rate
        self.device_channels = device_channels

        self.segment_callback = None

        self.vad_model = None
        self.vad_iterator = None
        self.resampler = None

        self.ACCUMULATOR_SIZE = config.CHUNK_SIZE * 4
        self.accum_buf = np.zeros(self.ACCUMULATOR_SIZE, dtype=np.float32)
        self.accum_pos = 0

        self.state = "IDLE"
        self.pre_buffer = []
        self.pre_buffer_samples = 0
        self.pre_pad_samples = config.PRE_SPEECH_PAD_SAMPLES

        self.segment_buf = np.zeros(config.MAX_SEGMENT_SAMPLES, dtype=np.float32)
        self.write_pos = 0
        self.speech_only_start = 0

        self.silence_since_last = 0.0
        self.hangover_timer = 0.0

        chunk_dur_ms = config.CHUNK_SIZE * 1000 / config.TARGET_SAMPLE_RATE

        self.chunk_index = 0
        self.chunk_count = 0
        self.speech_duration = 0.0

    def set_segment_callback(self, callback):
        """Set callback for completed segments (Phase 2 hook)."""
        self.segment_callback = callback

    def initialize(self):
        """Load VAD model and create resampler.

        Called before capture starts to avoid delay on first audio.
        VADIterator's native min_silence_duration_ms=0: all silence
        timing is handled by the PENDING_FINALIZE state machine,
        not by VAD's internal counter.
        """
        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(
            self.vad_model,
            threshold=self.config.VAD_THRESHOLD,
            sampling_rate=self.config.TARGET_SAMPLE_RATE,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
        )
        self.resampler = soxr.ResampleStream(
            in_rate=self.device_rate,
            out_rate=self.config.TARGET_SAMPLE_RATE,
            num_channels=1,
            dtype="float32",
            quality=self.config.RESAMPLE_QUALITY,
        )

    def process_chunk(self, raw_bytes: bytes) -> tuple | None:
        """Process one raw audio chunk.

        Returns:
            (segment_array, speech_ms, total_ms) if a segment completes,
            None otherwise.
        """
        raw = np.frombuffer(raw_bytes, dtype=np.int16)

        if self.device_channels > 1:
            raw = raw.reshape(-1, self.device_channels)
            if self.config.MONO_STRATEGY == "average_safe":
                left = raw[:, 0].astype(np.int32)
                right = raw[:, 1].astype(np.int32)
                mono = ((left + right) // 2).astype(np.int16)
            else:
                mono = raw[:, 0].copy()
        else:
            mono = raw

        audio_float = mono.astype(np.float32) / 32768.0

        resampled = self.resampler.resample_chunk(audio_float, last=False)
        n_out = len(resampled)

        while self.accum_pos + n_out > self.ACCUMULATOR_SIZE:
            if self.accum_pos >= self.config.CHUNK_SIZE:
                remaining = self.accum_pos - self.config.CHUNK_SIZE
                if remaining > 0:
                    self.accum_buf[:remaining] = self.accum_buf[self.config.CHUNK_SIZE:self.accum_pos]
                self.accum_pos = remaining
            else:
                n_out = self.ACCUMULATOR_SIZE - self.accum_pos
                resampled = resampled[:n_out]

        self.accum_buf[self.accum_pos:self.accum_pos + n_out] = resampled[:n_out]
        self.accum_pos += n_out

        if self.accum_pos < self.config.CHUNK_SIZE:
            return None

        chunk_16k = self.accum_buf[:self.config.CHUNK_SIZE].copy()
        remaining = self.accum_pos - self.config.CHUNK_SIZE
        if remaining > 0:
            self.accum_buf[:remaining] = self.accum_buf[self.config.CHUNK_SIZE:self.accum_pos]
        self.accum_pos = remaining

        self._update_pre_buffer(chunk_16k)

        speech_dict = self.vad_iterator(chunk_16k, return_seconds=True)
        has_start = speech_dict is not None and "start" in speech_dict
        has_end = speech_dict is not None and "end" in speech_dict

        result = None
        chunk_dur_ms = len(chunk_16k) * 1000 / self.config.TARGET_SAMPLE_RATE

        if self.state == "IDLE":
            self.silence_since_last += chunk_dur_ms

            if has_start:
                self.silence_since_last = 0.0
                self.state = "SPEECH"
                self._start_segment()
                self._append_to_segment(chunk_16k)
                self.speech_duration = len(chunk_16k) / self.config.TARGET_SAMPLE_RATE
                print(f"[SEGMENT] SPEECH start")

        elif self.state == "SPEECH":
            self._append_to_segment(chunk_16k)
            self.speech_duration += len(chunk_16k) / self.config.TARGET_SAMPLE_RATE

            if has_end:
                self.state = "PENDING_FINALIZE"
                self.hangover_timer = 0.0
                print(f"[SEGMENT] PENDING_FINALIZE enter at {self.speech_duration:.2f}s")

        elif self.state == "PENDING_FINALIZE":
            self._append_to_segment(chunk_16k)
            self.hangover_timer += chunk_dur_ms

            if has_start:
                print(f"[SEGMENT] SPEECH resume after {self.hangover_timer:.0f}ms hangover")
                self.state = "SPEECH"
                self.speech_duration += len(chunk_16k) / self.config.TARGET_SAMPLE_RATE

            elif self.hangover_timer >= self.config.VAD_MIN_SILENCE_MS:
                result = self._finalize_segment()
                if result is not None:
                    print(f"[SEGMENT] finalize: {result[2]:.0f}ms total, "
                          f"{result[1]:.0f}ms speech, {self.chunk_index} chunks")
                self.state = "IDLE"

        return result

    def _update_pre_buffer(self, chunk: np.ndarray):
        """Maintain sample-accurate ring buffer for pre-speech padding.

        Preserves at most PRE_SPEECH_PAD_SAMPLES of recent audio.
        """
        self.pre_buffer.append(chunk)
        self.pre_buffer_samples += len(chunk)
        while self.pre_buffer_samples > self.pre_pad_samples:
            oldest = self.pre_buffer.pop(0)
            self.pre_buffer_samples -= len(oldest)

    def _start_segment(self):
        """Copy pre-buffer content into the pre-allocated segment buffer.

        Marks speech_only_start so that actual speech duration
        can be tracked independently of padding.
        """
        self.write_pos = 0
        for chunk in self.pre_buffer:
            n = len(chunk)
            end = self.write_pos + n
            if end > self.config.MAX_SEGMENT_SAMPLES:
                break
            self.segment_buf[self.write_pos:end] = chunk
            self.write_pos = end
        self.speech_only_start = self.write_pos

    def _append_to_segment(self, chunk: np.ndarray):
        """Append resampled chunk to the pre-allocated buffer.

        Called from SPEECH and PENDING_FINALIZE — no sample is ever
        dropped during state transitions.
        """
        n = len(chunk)
        end = self.write_pos + n
        if end > self.config.MAX_SEGMENT_SAMPLES:
            return
        self.segment_buf[self.write_pos:end] = chunk
        self.write_pos = end

    def _finalize_segment(self) -> tuple | None:
        """Finalize and emit the accumulated segment.

        Called when PENDING_FINALIZE hangover expires.
        Resets VAD states at the same time — single reset point.
        Emit only if speech_only_samples >= MIN_SPEECH_DURATION_SAMPLES.
        Returns (segment_array, speech_ms, total_ms) or None.
        """
        speech_only_samples = self.write_pos - self.speech_only_start

        if speech_only_samples < self.config.MIN_SPEECH_DURATION_SAMPLES:
            self.write_pos = 0
            self.vad_iterator.reset_states()
            return None

        segment = self.segment_buf[:self.write_pos].copy()
        total_ms = self.write_pos * 1000 / self.config.TARGET_SAMPLE_RATE
        speech_ms = speech_only_samples * 1000 / self.config.TARGET_SAMPLE_RATE
        self.write_pos = 0

        self.chunk_index += 1
        self.vad_iterator.reset_states()
        return (segment, speech_ms, total_ms)
