"""Entry point — console or UI mode.

Usage:
    python main.py               # console mode (Enter to toggle)
    python main.py --mode ui     # UI with hotkeys
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.asr_worker import AsrConfig, AsrResult, AsrWorker
from core.audio_capture import AudioCapture
from core.pipeline import PipelineEvent
from core.ui import CaptureUI


_last_partial_len = 0


def print_asr_result(r: AsrResult):
    global _last_partial_len

    if r.is_final:
        if _last_partial_len:
            print("\r" + (" " * _last_partial_len) + "\r", end="", flush=True)
        print(f"[FINAL] {r.text}", flush=True)
        _last_partial_len = 0
    else:
        line = f"[PARTIAL] {r.text}"
        padding = " " * max(0, _last_partial_len - len(line))
        print("\r" + line + padding, end="", flush=True)
        _last_partial_len = len(line)


def make_asr_callback(ui=None, print_console=True):
    def _cb(r: AsrResult):
        if print_console:
            print_asr_result(r)
        if ui:
            ui.log_queue.put(PipelineEvent("asr_text", {
                "text": r.text,
                "is_final": r.is_final,
            }))
    return _cb


def run_console(asr_worker):
    capture = AudioCapture(
        live_callback=asr_worker.submit_live_chunk,
        soft_boundary_callback=asr_worker.submit_soft_boundary,
    )

    running = False
    print("Press Enter to start/stop capture, 'q' + Enter to quit")
    try:
        while True:
            cmd = input()
            if cmd.strip().lower() == "q":
                break
            if not running:
                capture.pipeline.start_async()
                running = True
                print("[CAPTURE STARTED]")
            else:
                capture.pipeline.stop()
                running = False
                print("[CAPTURE STOPPED]")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if running:
            capture.pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description="Real-time Audio Capture & ASR")
    parser.add_argument("--mode", choices=["console", "ui"], default="console",
                        help="console (default) or ui (tkinter + hotkeys)")
    args = parser.parse_args()

    if args.mode == "ui":
        ui = CaptureUI()

        asr_worker = AsrWorker(
            config=AsrConfig(),
            result_callback=make_asr_callback(ui, print_console=False),
        )
        asr_worker.start()

        ui.pipeline.set_live_chunk_callback(asr_worker.submit_live_chunk)
        ui.pipeline.set_soft_boundary_callback(asr_worker.submit_soft_boundary)

        try:
            ui.run()
        finally:
            asr_worker.stop()
    else:
        asr_worker = AsrWorker(
            config=AsrConfig(),
            result_callback=print_asr_result,
        )
        asr_worker.start()

        try:
            run_console(asr_worker)
        finally:
            asr_worker.stop()


if __name__ == "__main__":
    main()
