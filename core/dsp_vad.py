import time
from collections import deque

import numpy as np
import soxr
from silero_vad import VADIterator, load_silero_vad

from core.config import CaptureConfig


class DspVad:
    """Audio resampling + VAD state tracking for UI meter.

    All resampled chunks are forwarded to ASR continuously via
    speech_chunk_callback. VAD is used only for UI state display,
    NOT for segmentation. Sherpa-ONNX endpoint detection handles
    utterance finalization.
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

        self.speech_chunk_callback = None
        self.soft_boundary_callback = None

        self.vad_model = None
        self.vad_iterator = None
        self.resampler = None

        self.ACCUMULATOR_SIZE = config.CHUNK_SIZE * 4
        self.accum_buf = np.zeros(self.ACCUMULATOR_SIZE, dtype=np.float32)
        self.accum_pos = 0

        self.state = "IDLE"
        self.speech_duration = 0.0
        self._pending_silence_ms = 0.0
        self._pending_boundary_emitted = False
        self._forwarded_chunks = 0

    def set_speech_chunk_callback(self, callback):
        self.speech_chunk_callback = callback

    def set_soft_boundary_callback(self, callback):
        self.soft_boundary_callback = callback

    def initialize(self):
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

    def process_chunk(self, raw_bytes: bytes):
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

        self._update_vad_state(chunk_16k)

        if self.speech_chunk_callback:
            self.speech_chunk_callback(chunk_16k)
            self._forwarded_chunks += 1
            if self._forwarded_chunks == 1:
                print(
                    "[DSP] first 16k chunk forwarded to ASR: "
                    f"{len(chunk_16k)} samples",
                    flush=True,
                )

        return None

    def _update_vad_state(self, chunk_16k: np.ndarray):
        speech_dict = self.vad_iterator(chunk_16k, return_seconds=True)
        has_start = speech_dict is not None and "start" in speech_dict
        has_end = speech_dict is not None and "end" in speech_dict
        chunk_dur_ms = len(chunk_16k) * 1000 / self.config.TARGET_SAMPLE_RATE

        if self.state == "IDLE":
            if has_start:
                self.state = "SPEECH"
                self.speech_duration = chunk_dur_ms / 1000
                print("[VAD] state IDLE -> SPEECH", flush=True)
        elif self.state == "SPEECH":
            self.speech_duration += chunk_dur_ms / 1000
            if has_end:
                self.state = "PENDING_FINALIZE"
                self._pending_silence_ms = 0.0
                self._pending_boundary_emitted = False
                print("[VAD] state SPEECH -> PENDING_FINALIZE", flush=True)
        elif self.state == "PENDING_FINALIZE":
            if has_start:
                if (not self._pending_boundary_emitted
                        and self._pending_silence_ms >= self.config.SOFT_COMMA_PAUSE_MS):
                    boundary_type = (
                        "sentence"
                        if self._pending_silence_ms >= self.config.SOFT_SENTENCE_PAUSE_MS
                        else "comma"
                    )
                    self._emit_soft_boundary(boundary_type, self._pending_silence_ms)
                self.state = "SPEECH"
                self.speech_duration += chunk_dur_ms / 1000
                self._pending_silence_ms = 0.0
                self._pending_boundary_emitted = False
                print("[VAD] state PENDING_FINALIZE -> SPEECH", flush=True)
            else:
                self._pending_silence_ms += chunk_dur_ms
                if (not self._pending_boundary_emitted
                        and self._pending_silence_ms >= self.config.SOFT_SENTENCE_PAUSE_MS):
                    self._emit_soft_boundary("sentence", self._pending_silence_ms)
                    self._pending_boundary_emitted = True
                if self._pending_silence_ms > 3000:
                    self.state = "IDLE"
                    self.speech_duration = 0.0
                    print("[VAD] state PENDING_FINALIZE -> IDLE", flush=True)

    def _emit_soft_boundary(self, boundary_type: str, pause_ms: float):
        if not self.soft_boundary_callback:
            return
        print(
            f"[VAD] soft boundary {boundary_type}: pause={pause_ms:.0f}ms",
            flush=True,
        )
        self.soft_boundary_callback(boundary_type, pause_ms)
