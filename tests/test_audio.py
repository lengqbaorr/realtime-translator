import os
import sys
import tempfile
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import CaptureConfig

SAMPLE_RATE = CaptureConfig.TARGET_SAMPLE_RATE


def record_loopback(duration_sec=5, output_wav=None):
    """Record loopback audio to a WAV file for testing.

    Returns the file path of the recorded WAV.
    """
    if output_wav is None:
        output_wav = os.path.join(tempfile.gettempdir(), "test_loopback.wav")

    p = pyaudio.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )
        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                raise RuntimeError("No loopback device found")

        device_rate = int(default_speakers["defaultSampleRate"])
        device_channels = default_speakers["maxInputChannels"]

        frames = []
        stop = threading.Event()

        def callback(in_data, frame_count, time_info, status):
            frames.append(in_data)
            return (in_data, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paInt16,
            channels=device_channels,
            rate=device_rate,
            input=True,
            input_device_index=default_speakers["index"],
            frames_per_buffer=1024,
            stream_callback=callback,
        )

        stream.start_stream()
        time.sleep(duration_sec)
        stream.stop_stream()
        stream.close()

        raw = np.frombuffer(b"".join(frames), dtype=np.int16)
        if device_channels > 1:
            raw = raw.reshape(-1, device_channels)
            raw = np.mean(raw, axis=1).astype(np.int16)

        sf.write(output_wav, raw.astype(np.float32) / 32768.0, device_rate)
    finally:
        p.terminate()

    return output_wav


def test_loopback_capture():
    """Verify loopback recording produces a non-empty audio file."""
    wav_path = record_loopback(duration_sec=3)

    assert os.path.exists(wav_path), "WAV file was not created"
    assert os.path.getsize(wav_path) > 44, "WAV file is too small (header only)"

    data, sr = sf.read(wav_path)
    assert len(data) > 0, "Audio data is empty"
    expected_samples = int(sr * 3)
    tolerance = sr * 0.5
    assert abs(len(data) - expected_samples) < tolerance, (
        f"Expected ~{expected_samples} samples, got {len(data)}"
    )

    rms = np.sqrt(np.mean(data ** 2))
    print(f"[TEST] Duration: {len(data) / sr:.1f}s  |  "
          f"Samples: {len(data)}  |  RMS: {rms:.6f}")

    os.remove(wav_path)
    print("[TEST] test_loopback_capture PASSED")


if __name__ == "__main__":
    test_loopback_capture()
