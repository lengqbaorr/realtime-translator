import os
import queue
import threading
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from core.config import CaptureConfig
from core.capture_thread import CaptureThread
from core.dsp_vad import DspVad

THREAD_PRIORITY_ABOVE_NORMAL = 1
THREAD_PRIORITY_TIME_CRITICAL = 15
THREAD_SET_INFORMATION = 0x0020

try:
    import win32api
    import win32process
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False


@dataclass
class BenchmarkStats:
    dropped_segments: int = 0
    dropped_wav_jobs: int = 0


@dataclass
class PipelineEvent:
    type: str
    data: dict


class Pipeline:
    """Orchestrates the 3-tier audio capture pipeline.

    Thread model:
      - CaptureThread: runs PortAudio callback, pushes raw bytes to deque
      - DspVad: runs in daemon thread, polls deque, processes audio
      - Consumer: receives segments, logs events, calls Phase 2 hook

    Supports both blocking (run) and async (start_async/stop) modes.
    """

    def __init__(self, config: CaptureConfig | None = None,
                 log_queue: queue.Queue | None = None):
        self.config = config or CaptureConfig()
        self.log_queue = log_queue
        self.segment_callback = None
        self.stop_event = threading.Event()

        self.raw_buffer: queue.Queue = queue.Queue(maxsize=self.config.RAW_QUEUE_MAXSIZE)
        self.segment_queue: queue.Queue = queue.Queue(
            maxsize=self.config.SEGMENT_QUEUE_MAXSIZE
        )
        self.wav_queue: queue.Queue = queue.Queue(
            maxsize=self.config.WAV_QUEUE_MAXSIZE
        )

        self.capture = CaptureThread(self.raw_buffer, self.config)
        self.dsp_vad: DspVad | None = None

        self.dsp_thread: threading.Thread | None = None
        self.wav_thread: threading.Thread | None = None
        self.consumer_thread: threading.Thread | None = None

        self.stats = BenchmarkStats()
        self.chunk_index = 0
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def set_segment_callback(self, callback):
        self.segment_callback = callback

    def _emit(self, event_type: str, **data):
        if self.log_queue:
            self.log_queue.put(PipelineEvent(type=event_type, data=data))

    def run(self, stop_event: threading.Event):
        """Block until stop_event is set (console mode).

        Consumer runs in daemon thread from _start_threads().
        run() only waits — guaranteed exactly 1 consumer loop.
        """
        self.stop_event = stop_event
        self._initialize()
        self._start_threads()
        stop_event.wait()
        self.segment_queue.put(None)
        self._cleanup()

    def start_async(self):
        """Non-blocking start for UI mode.

        Returns immediately. Consumer runs in daemon thread.
        Call stop() to gracefully shut down.
        """
        self.stop_event.clear()
        self._initialize()
        self._start_threads()
        self._is_running = True
        self._emit("status", status="recording", device=self.capture.device_info.get("name", "?"))

    def stop(self):
        """Graceful shutdown of all threads."""
        self._is_running = False
        self.stop_event.set()
        self.segment_queue.put(None)

        for t in [self.dsp_thread, self.wav_thread, self.consumer_thread]:
            if t and t.is_alive():
                t.join(timeout=3)

        self.capture.stop()
        self._emit("status", status="stopped")

    def _initialize(self):
        """Initialize capture device, DSP+VAD, and create dirs."""
        os.makedirs(self.config.DEBUG_SAVE_DIR, exist_ok=True)
        self.capture.initialize()
        self.dsp_vad = DspVad(
            self.raw_buffer,
            self.config,
            self.capture.device_rate,
            self.capture.device_channels,
        )
        if self.segment_callback:
            self.dsp_vad.set_segment_callback(self.segment_callback)
        self.dsp_vad.initialize()

    def _start_threads(self):
        """Start capture stream and daemon threads."""
        self.capture.start()

        self.dsp_thread = threading.Thread(
            target=self._dsp_loop,
            args=(self.stop_event,),
            daemon=True,
        )
        self.dsp_thread.start()

        if self.config.DEBUG_SAVE_WAV:
            self.wav_thread = threading.Thread(
                target=self._wav_loop,
                args=(self.stop_event,),
                daemon=True,
            )
            self.wav_thread.start()

        self.consumer_thread = threading.Thread(
            target=self._consumer_loop,
            args=(self.stop_event,),
            daemon=True,
        )
        self.consumer_thread.start()

        self._set_thread_priority(THREAD_PRIORITY_ABOVE_NORMAL)

    def _dsp_loop(self, stop_event: threading.Event):
        self._set_thread_priority(THREAD_PRIORITY_ABOVE_NORMAL)
        silence_warned = False

        while not stop_event.is_set():
            try:
                raw_bytes = self.raw_buffer.get(timeout=1.0)
            except queue.Empty:
                if not silence_warned:
                    print("[WARN] No audio received for 1s — check device connection")
                    silence_warned = True
                continue
            silence_warned = False

            result = self.dsp_vad.process_chunk(raw_bytes)
            if result is not None:
                try:
                    self.segment_queue.put_nowait(result)
                except queue.Full:
                    try:
                        self.segment_queue.get_nowait()
                        self.segment_queue.put_nowait(result)
                        self.stats.dropped_segments += 1
                    except queue.Empty:
                        pass

    def _wav_loop(self, stop_event: threading.Event):
        while not stop_event.is_set():
            try:
                item = self.wav_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            filename, audio, sr = item
            try:
                sf.write(filename, audio, sr)
            except Exception as exc:
                self._emit("error", message=f"WAV write failed: {exc}")

    def _consumer_loop(self, stop_event: threading.Event):
        last_silence = 0

        while not stop_event.is_set():
            try:
                segment_data = self.segment_queue.get(timeout=0.05)
            except queue.Empty:
                self._emit_vad_status()
                continue

            if segment_data is None:
                break

            segment, speech_ms, total_ms = segment_data
            sr = self.config.TARGET_SAMPLE_RATE

            if self.config.DEBUG_SAVE_WAV:
                self.chunk_index += 1
                filename = os.path.join(
                    self.config.DEBUG_SAVE_DIR, f"chunk_{self.chunk_index:03d}.wav"
                )
                try:
                    self.wav_queue.put_nowait((filename, segment, sr))
                except queue.Full:
                    print(f"[WARN] WAV queue full — dropped {filename}")
                    self.stats.dropped_wav_jobs += 1

            if self.segment_callback:
                self.segment_callback(segment, speech_ms, total_ms)

            self._emit("chunk",
                       index=self.chunk_index,
                       filename=f"chunk_{self.chunk_index:03d}.wav",
                       total_ms=total_ms,
                       speech_ms=speech_ms,
                       dropped=self.stats.dropped_segments)

    def _emit_vad_status(self):
        """Emit VAD status for UI level meter."""
        if not self.dsp_vad:
            return
        state = self.dsp_vad.state
        duration = getattr(self.dsp_vad, "speech_duration", 0.0)
        self._emit("vad", state=state, duration=duration)

    def _set_thread_priority(self, priority: int):
        if not HAS_PYWIN32:
            return
        try:
            tid = win32api.GetCurrentThreadId()
            handle = win32api.OpenThread(
                THREAD_SET_INFORMATION, False, tid
            )
            win32process.SetThreadPriority(handle, priority)
            handle.Close()
        except Exception:
            pass

    def _cleanup(self):
        self.capture.stop()

    def get_stats(self) -> dict:
        return {
            "chunks": self.chunk_index,
            "state": self.dsp_vad.state if self.dsp_vad else "N/A",
            "dropped_raw_chunks": self.capture.stats_dropped_raw,
            "dropped_segments": self.stats.dropped_segments,
            "dropped_wav_jobs": self.stats.dropped_wav_jobs,
            "pa_input_overflows": self.capture.stats_overflow,
        }
