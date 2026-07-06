# Realtime Translator

Real-time system-audio capture and streaming ASR for Windows.

The current architecture uses sherpa-onnx as the owner of streaming
recognition and endpoint/finalization. The audio pipeline only captures,
downmixes, resamples, and forwards continuous 16 kHz chunks to ASR.

## Features

- Low-latency WASAPI loopback capture via PyAudioWPatch
- Streaming ASR with sherpa-onnx transducer models
- sherpa-onnx endpoint detection for final transcript boundaries
- Partial ASR updates while audio is still streaming
- Tkinter UI with global hotkeys
- Optional Silero VAD state tracking for the UI meter only

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python main.py               # console mode
python main.py --mode ui     # UI mode
```

## Requirements

- Windows 10+
- Python 3.10+
- sherpa-onnx model files under `models/zipformer-en/`

## Project Structure

```text
├── core/
│   ├── config.py          # CaptureConfig and AsrConfig
│   ├── audio_capture.py   # Backward-compatible pipeline wrapper
│   ├── capture_thread.py  # WASAPI loopback capture
│   ├── dsp_vad.py         # Downmix/resample + UI VAD meter
│   ├── asr_worker.py      # sherpa-onnx streaming ASR worker
│   ├── pipeline.py        # Capture -> DSP -> ASR chunk forwarding
│   ├── ui.py              # Tkinter UI + hotkeys
│   └── benchmark.py       # Latency benchmarks
├── models/                # sherpa-onnx model assets
├── tests/                 # Unit/integration tests
├── main.py                # Console/UI entry point
└── requirements.txt
```
