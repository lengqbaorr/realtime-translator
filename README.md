# Realtime Translator

Real-time speech capture, VAD segmentation, and translation pipeline for Windows.

## Features

- **Low-latency audio capture** via WASAPI loopback (PyAudioWPatch)
- **Voice Activity Detection** using Silero VAD with DSP pre-filtering
- **State-machine segmentation** — merges short pauses, splits on silence
- **Thread-safe pipeline** — async capture, DSP, and callback chain
- **Global hotkeys** — Ctrl+Shift+R toggle, Ctrl+Shift+Q quit
- **Debug mode** — saves raw audio chunks for analysis

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py          # CLI mode
python main_ui.py       # UI mode (tkinter + hotkeys)
```

## Requirements

- Windows 10+ (WASAPI exclusive)
- Python 3.10+

## Project Structure

```
├── core/
│   ├── config.py          # App configuration & settings
│   ├── audio_capture.py   # WASAPI loopback capture
│   ├── capture_thread.py  # Background capture thread
│   ├── dsp_vad.py         # DSP + Silero VAD pipeline
│   ├── pipeline.py        # Segmentation state machine
│   ├── ui.py              # Tkinter UI + global hotkeys
│   └── benchmark.py       # Latency benchmarks
├── tests/                 # Unit tests
├── main.py                # CLI entry point
├── main_ui.py             # UI entry point
└── requirements.txt
```
