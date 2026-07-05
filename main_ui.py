"""UI entry point — starts the tkinter-based capture controller.

Usage:
    python -m realtime_translator.main_ui

Hotkeys (global):
    Ctrl+Shift+R — toggle recording start/stop
    Ctrl+Shift+Q — quit application
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ui import CaptureUI


def main():
    ui = CaptureUI()
    ui.run()


if __name__ == "__main__":
    main()
