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
    """Orchestrates audio capture pipeline.

    Thread model:
      - CaptureThread: runs PortAudio callback, pushes raw bytes to deque
      - DspVad: runs in daemon thread, polls deque, resamples audio,
        forwards all chunks to ASR, tracks VAD state for UI
      - No segmentation — sherpa-onnx endpoint detection handles it

    Supports both blocking (run) and async (start_async/stop) modes.
    """

    def __init__(self, config: CaptureConfig | None = None,
                 log_queue: queue.Queue | None = None):
        self.config = config or CaptureConfig()
        self.log_queue = log_queue
        self.live_chunk_callback = None
        self.soft_boundary_callback = None
        self.stop_event = threading.Event()

        self.raw_buffer: queue.Queue = queue.Queue(maxsize=self.config.RAW_QUEUE_MAXSIZE)
        self.wav_queue: queue.Queue = queue.Queue(maxsize=self.config.WAV_QUEUE_MAXSIZE)

        self.capture = CaptureThread(self.raw_buffer, self.config)
        self.dsp_vad: DspVad | None = None

        self.dsp_thread: threading.Thread | None = None
        self.wav_thread: threading.Thread | None = None

        self.chunk_index = 0
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def set_live_chunk_callback(self, callback):
        self.live_chunk_callback = callback
        print("[PIPE] live chunk callback connected", flush=True)

    def set_soft_boundary_callback(self, callback):
        self.soft_boundary_callback = callback
        print("[PIPE] soft boundary callback connected", flush=True)

    def _emit(self, event_type: str, **data):
        if self.log_queue:
            self.log_queue.put(PipelineEvent(type=event_type, data=data))

    def run(self, stop_event: threading.Event):
        """Block until stop_event is set (console mode)."""
        self.stop_event = stop_event
        self._initialize()
        self._start_threads()
        stop_event.wait()
        self._cleanup()

    def start_async(self):
        """Non-blocking start for UI mode."""
        self.stop_event.clear()
        print("[PIPE] starting capture pipeline", flush=True)
        self._initialize()
        self._start_threads()
        self._is_running = True
        self._emit("status", status="recording", device=self.capture.device_info.get("name", "?"))
        print("[PIPE] capture pipeline started", flush=True)

    def stop(self):
        """Graceful shutdown of all threads."""
        print("[PIPE] stopping capture pipeline", flush=True)
        self._is_running = False
        self.stop_event.set()

        for t in [self.dsp_thread, self.wav_thread]:
            if t and t.is_alive():
                t.join(timeout=3)

        self.capture.stop()
        self._emit("status", status="stopped")
        stats = self.get_stats()
        print(
            "[PIPE] stopped: "
            f"state={stats['state']}, "
            f"dropped_raw={stats['dropped_raw_chunks']}, "
            f"pa_overflows={stats['pa_input_overflows']}",
            flush=True,
        )

    def _initialize(self):
        os.makedirs(self.config.DEBUG_SAVE_DIR, exist_ok=True)
        self.capture.initialize()
        self.dsp_vad = DspVad(
            self.raw_buffer,
            self.config,
            self.capture.device_rate,
            self.capture.device_channels,
        )
        if self.live_chunk_callback:
            self.dsp_vad.set_speech_chunk_callback(self.live_chunk_callback)
        if self.soft_boundary_callback:
            self.dsp_vad.set_soft_boundary_callback(self.soft_boundary_callback)
        self.dsp_vad.initialize()
        print(
            "[PIPE] initialized DSP: "
            f"device_rate={self.capture.device_rate}, "
            f"channels={self.capture.device_channels}, "
            f"target_rate={self.config.TARGET_SAMPLE_RATE}, "
            f"chunk_size={self.config.CHUNK_SIZE}",
            flush=True,
        )

    def _start_threads(self):
        self.capture.start()

        self.dsp_thread = threading.Thread(
            target=self._dsp_loop,
            args=(self.stop_event,),
            daemon=True,
        )
        self.dsp_thread.start()

        self._set_thread_priority(THREAD_PRIORITY_ABOVE_NORMAL)

    def _dsp_loop(self, stop_event: threading.Event):
        self._set_thread_priority(THREAD_PRIORITY_ABOVE_NORMAL)
        silence_warned = False
        print("[PIPE] DSP loop started", flush=True)

        while not stop_event.is_set():
            try:
                raw_bytes = self.raw_buffer.get(timeout=0.5)
            except queue.Empty:
                if self.dsp_vad:
                    self._emit_vad_status()
                if not silence_warned:
                    print("[WARN] No audio received for 1s — check device connection")
                    silence_warned = True
                continue
            silence_warned = False

            self.dsp_vad.process_chunk(raw_bytes)
            self._emit_vad_status()

        print("[PIPE] DSP loop stopped", flush=True)

    def _emit_vad_status(self):
        if not self.dsp_vad:
            return
        self._emit("vad",
                    state=self.dsp_vad.state,
                    duration=getattr(self.dsp_vad, "speech_duration", 0.0))

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
            "pa_input_overflows": self.capture.stats_overflow,
        }
