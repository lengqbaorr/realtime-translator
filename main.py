import os
import signal
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.audio_capture import AudioCapture


def main():
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    capture = AudioCapture()
    capture.start(stop_event)


if __name__ == "__main__":
    main()
