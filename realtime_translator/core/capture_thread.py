import queue

import pyaudiowpatch as pyaudio

from core.config import CaptureConfig


class CaptureThread:
    """WASAPI loopback capture using PyAudio callback mode.

    This thread owns the PortAudio stream and runs its callback.
    The callback MUST be kept lightweight: it only pushes raw bytes
    to a bounded queue.Queue and reads status_flags. No numpy,
    no resample, no VAD, no I/O.

    queue.Queue.put_nowait() uses a C-level mutex internally,
    but on an uncontended path (DSP thread is blocked on get())
    the acquire is ~100ns — acceptable for PortAudio's "no locks"
    guideline. The old deque approach was theoretically more pure
    but provided no blocking read for the DSP thread, forcing
    sleep-poll jitter.
    """

    def __init__(self, raw_buffer: queue.Queue, config: CaptureConfig):
        self.raw_buffer = raw_buffer
        self.config = config

        self.p: pyaudio.PyAudio | None = None
        self.stream: pyaudio.Stream | None = None
        self.device_info: dict = {}
        self.device_rate: int = 0
        self.device_channels: int = 0

        self.stats_overflow: int = 0
        self.stats_dropped_raw: int = 0

    def initialize(self):
        """Find loopback device and populate device info.

        Must be called before start() so that device_rate and
        device_channels are available for the DSP thread.
        """
        self.p = pyaudio.PyAudio()
        self.device_info = self._find_loopback_device()
        self.device_rate = int(self.device_info["defaultSampleRate"])
        self.device_channels = self.device_info["maxInputChannels"]

        print(f"[VAD] Device: {self.device_info['name']}")
        print(f"[VAD] Rate: {self.device_rate} Hz  |  "
              f"Channels: {self.device_channels}")

    def start(self):
        """Open stream in callback mode and start it.

        Non-blocking: returns immediately after stream.start_stream().
        The callback runs in PortAudio's internal thread.
        """
        frames_per_buffer = int(
            self.config.CHUNK_SIZE * self.device_rate / self.config.TARGET_SAMPLE_RATE
        )

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.device_channels,
            rate=self.device_rate,
            input=True,
            input_device_index=self.device_info["index"],
            frames_per_buffer=frames_per_buffer,
            stream_callback=self._callback,
        )
        self.stream.start_stream()

        print(f"[VAD] Read: {frames_per_buffer} samples/chunk "
              f"({frames_per_buffer * 1000 / self.device_rate:.0f}ms)")

    def stop(self):
        """Stop and close the stream, terminate PyAudio."""
        if self.stream:
            try:
                self.stream.stop_stream()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.p:
            self.p.terminate()
            self.p = None

    def _callback(self, in_data, frame_count, time_info, status_flags):
        """PortAudio callback — runs in PortAudio's audio thread.

        Pushes raw bytes to bounded queue. Reads status_flags and
        increments counters for paInputOverflow. If the queue is
        full (DSP thread is too slow), increments dropped_raw counter.
        """
        if status_flags & pyaudio.paInputOverflow:
            self.stats_overflow += 1

        try:
            self.raw_buffer.put_nowait(in_data)
        except queue.Full:
            self.stats_dropped_raw += 1

        return (None, pyaudio.paContinue)

    def _find_loopback_device(self):
        """Find WASAPI loopback device for system audio capture."""
        wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
        speakers = self.p.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )
        if not speakers["isLoopbackDevice"]:
            for loopback in self.p.get_loopback_device_info_generator():
                if speakers["name"] in loopback["name"]:
                    speakers = loopback
                    break
            else:
                raise RuntimeError(
                    "No loopback device found. Ensure speakers are enabled."
                )
        return speakers
