"""ASR worker — transcribes speech segments in a background daemon thread.

Plugs into the pipeline via set_segment_callback(submit_segment).
submit_segment is non-blocking (queue.put_nowait + immediate return).
Worker thread runs faster-whisper model.transcribe() on queued segments.
"""

import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.config import AsrConfig


@dataclass
class AsrResult:
    text: str
    speech_ms: float
    total_ms: float
    avg_logprob: float
    no_speech_prob: float
    sequence: int
    is_low_confidence: bool


class AsrWorker:
    """Single-worker ASR processor.

    One daemon thread reads from an internal bounded queue,
    runs faster-whisper inference, and calls result_callback.

    submit_segment is designed to be passed as pipeline segment_callback.
    """

    def __init__(
        self,
        config: AsrConfig | None = None,
        result_callback: Callable[[AsrResult], None] | None = None,
    ):
        self.config = config or AsrConfig()
        self.result_callback = result_callback

        self._model = None
        self._stop_event = threading.Event()
        self._asr_queue: queue.Queue = queue.Queue(maxsize=self.config.ASR_QUEUE_MAXSIZE)
        self._worker_thread: threading.Thread | None = None
        self._sequence = 0
        self._lock = threading.Lock()
        self.dropped_asr_segments: int = 0

    def start(self):
        """Load faster-whisper model and start worker thread.

        Model loads once — verified by logging model load count.
        Thread runs until stop() is called.
        If _model is already set (e.g. mock for tests), skip loading.
        """
        if self._model is None:
            print(f"[ASR] Loading model {self.config.MODEL_SIZE} "
                  f"({self.config.DEVICE}, {self.config.COMPUTE_TYPE})...")
            self._load_model()
            print("[ASR] Model loaded")
        else:
            print("[ASR] Using pre-set model (test mode)")

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self._worker_thread.start()

    def _load_model(self):
        """Load faster-whisper model. Separated for test mocking."""
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self.config.MODEL_SIZE,
            device=self.config.DEVICE,
            compute_type=self.config.COMPUTE_TYPE,
        )

    def submit_segment(
        self, segment: np.ndarray, speech_ms: float, total_ms: float
    ):
        """Non-blocking callback for pipeline segment_callback.

        Queues the segment with an incrementing sequence number.
        If the queue is full, drops the oldest segment (FIFO drop-oldest).
        """
        with self._lock:
            seq = self._sequence
            self._sequence += 1

        item = (seq, segment, speech_ms, total_ms)

        try:
            self._asr_queue.put_nowait(item)
        except queue.Full:
            try:
                self._asr_queue.get_nowait()
                self._asr_queue.put_nowait(item)
                self.dropped_asr_segments += 1
            except queue.Empty:
                pass

    def stop(self):
        """Graceful shutdown: signal stop, unblock queue, join thread."""
        self._stop_event.set()
        try:
            self._asr_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

    def _worker_loop(self):
        """Worker thread loop — blocking get, transcribe, callback."""
        while not self._stop_event.is_set():
            try:
                item = self._asr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            seq, segment, speech_ms, total_ms = item

            try:
                result = self._transcribe(segment, seq, speech_ms, total_ms)
                if self.result_callback:
                    self.result_callback(result)
            except Exception as exc:
                print(f"[ASR] Error processing segment {seq}: {exc}")

    def _transcribe(
        self,
        segment: np.ndarray,
        seq: int,
        speech_ms: float,
        total_ms: float,
    ) -> AsrResult:
        """Run faster-whisper inference on a single segment.

        segment: float32 mono @ 16kHz
        Returns AsrResult with text and confidence metrics.
        """
        segments, info = self._model.transcribe(
            segment,
            language=self.config.LANGUAGE,
            beam_size=self.config.BEAM_SIZE,
            condition_on_previous_text=self.config.CONDITION_ON_PREVIOUS_TEXT,
            vad_filter=self.config.VAD_FILTER,
            word_timestamps=self.config.WORD_TIMESTAMPS,
        )

        text_parts = []
        best_logprob = 0.0
        no_speech_prob = 0.0

        for seg in segments:
            text_parts.append(seg.text)
            best_logprob = min(best_logprob, seg.avg_logprob)
            no_speech_prob = max(no_speech_prob, seg.no_speech_prob)

        text = " ".join(text_parts).strip()
        is_low_confidence = best_logprob < self.config.LOW_CONFIDENCE_AVG_LOGPROB

        return AsrResult(
            text=text,
            speech_ms=speech_ms,
            total_ms=total_ms,
            avg_logprob=best_logprob,
            no_speech_prob=no_speech_prob,
            sequence=seq,
            is_low_confidence=is_low_confidence,
        )
