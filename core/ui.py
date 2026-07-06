import queue
from datetime import datetime

import tkinter as tk
from tkinter import ttk

from core.config import CaptureConfig
from core.pipeline import Pipeline


try:
    from pynput import keyboard as pynput_kb
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False


HOTKEY_TOGGLE = "<ctrl>+<shift>+r"
HOTKEY_QUIT = "<ctrl>+<shift>+q"

LABEL_TOGGLE = "Ctrl+Shift+R"
LABEL_QUIT = "Ctrl+Shift+Q"


class CaptureUI:
    """Tkinter-based UI for controlling the audio capture pipeline.

    Hotkeys (global):
      Ctrl+Shift+R — toggle recording start/stop
      Ctrl+Shift+Q — quit application

    Pipeline runs in background daemon threads.
    UI polls log_queue via tkinter.after() for thread-safe updates.
    """

    WINDOW_TITLE = "Real-Time Audio Capture"
    WINDOW_W = 420
    WINDOW_H = 680

    def __init__(self, config: CaptureConfig | None = None):
        self.config = config or CaptureConfig()
        self.log_queue: queue.Queue = queue.Queue()
        self.pipeline = Pipeline(self.config, log_queue=self.log_queue)
        self._partial_row_active = False

        self.root = tk.Tk()
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(f"{self.WINDOW_W}x{self.WINDOW_H}")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._hotkey_listener: pynput_kb.GlobalHotKeys | None = None

        self._build_ui()
        self._start_hotkeys()

    def _build_ui(self):
        # ── Top frame: status + controls ──
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        self.status_canvas = tk.Canvas(top, width=16, height=16, highlightthickness=0)
        self.status_canvas.pack(side=tk.LEFT, padx=(0, 6))
        self._draw_status_dot("stopped")

        self.status_label = ttk.Label(top, text="Stopped", font=("Segoe UI", 10, "bold"))
        self.status_label.pack(side=tk.LEFT)

        self.toggle_btn = ttk.Button(top, text="Start", width=8, command=self._toggle)
        self.toggle_btn.pack(side=tk.RIGHT)

        # ── Device info ──
        info_frame = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        info_frame.pack(fill=tk.X)

        ttk.Label(info_frame, text="Device:", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.device_label = ttk.Label(info_frame, text="—", font=("Segoe UI", 8))
        self.device_label.pack(side=tk.LEFT, padx=(4, 0))

        # ── Level meter ──
        meter_frame = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        meter_frame.pack(fill=tk.X)

        self.meter_canvas = tk.Canvas(meter_frame, height=20, highlightthickness=0)
        self.meter_canvas.pack(fill=tk.X)

        self.state_label = ttk.Label(meter_frame, text="", font=("Segoe UI", 7))
        self.state_label.pack()

        # ── Log area ──
        log_frame = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        log_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(log_frame, text="Captured Chunks:", font=("Segoe UI", 8)).pack(anchor=tk.W)

        self.log_list = tk.Listbox(
            log_frame, height=10, font=("Consolas", 8),
            selectmode=tk.SINGLE, activestyle=tk.NONE,
        )
        self.log_list.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        # ── ASR text area ──
        asr_frame = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        asr_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(asr_frame, text="ASR Output:", font=("Segoe UI", 8)).pack(anchor=tk.W)

        self.asr_list = tk.Listbox(
            asr_frame, height=6, font=("Consolas", 9),
            selectmode=tk.SINGLE, activestyle=tk.NONE,
        )
        self.asr_list.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        # ── Bottom frame: stats + hotkeys ──
        bottom = ttk.Frame(self.root, padding=(8, 4, 8, 8))
        bottom.pack(fill=tk.X)

        self.stats_label = ttk.Label(
            bottom, text="Chunks: 0  |  Dropped: 0",
            font=("Segoe UI", 8),
        )
        self.stats_label.pack(side=tk.LEFT)

        hotkey_label = ttk.Label(
            bottom,
            text=f"{LABEL_TOGGLE}: Toggle  |  {LABEL_QUIT}: Quit",
            font=("Segoe UI", 7),
            foreground="gray",
        )
        hotkey_label.pack(side=tk.RIGHT)

        # ── Start polling ──
        self.root.after(100, self._poll_log_queue)
        self.root.after(200, self._update_meter)

    def _draw_status_dot(self, state: str):
        self.status_canvas.delete("all")
        color = {"recording": "#22c55e", "stopped": "#ef4444", "error": "#f59e0b"}
        fill = color.get(state, "#6b7280")
        self.status_canvas.create_oval(2, 2, 14, 14, fill=fill, outline="")

    def _toggle(self):
        if self.pipeline.is_running:
            self._stop()
        else:
            self._start()

    def _start(self):
        if self.pipeline.is_running:
            return
        try:
            self.pipeline.start_async()
            self._draw_status_dot("recording")
            self.status_label.config(text="Recording...", foreground="#22c55e")
            self.toggle_btn.config(text="Stop")
            self.device_label.config(
                text=self.pipeline.capture.device_info.get("name", "?")
            )
        except Exception as exc:
            self.status_label.config(text=f"Error: {exc}", foreground="#f59e0b")
            self._draw_status_dot("error")

    def _stop(self):
        if not self.pipeline.is_running:
            return
        self.pipeline.stop()
        self._draw_status_dot("stopped")
        self.status_label.config(text="Stopped", foreground="#ef4444")
        self.toggle_btn.config(text="Start")

    def _on_close(self):
        self._stop()
        if self._hotkey_listener:
            self._hotkey_listener.stop()
        self.root.destroy()

    def _poll_log_queue(self):
        """Poll log queue and update UI widgets.

        Called via tkinter.after() from the main thread,
        ensuring thread-safe widget updates.
        """
        while True:
            try:
                event = self.log_queue.get_nowait()
            except queue.Empty:
                break

            t = event.type
            d = event.data

            if t == "chunk":
                ts = datetime.now().strftime("%H:%M:%S")
                idx = d.get("index", 0)
                total_s = d.get("total_ms", 0) / 1000
                speech_s = d.get("speech_ms", 0) / 1000
                dropped = d.get("dropped", 0)
                line = f"{ts}  chunk_{idx:03d}.wav  {total_s:.1f}s  ({speech_s:.2f}s speech)"
                self.log_list.insert(0, line)

                while self.log_list.size() > 100:
                    self.log_list.delete(tk.END)

                self.stats_label.config(
                    text=f"Chunks: {self.pipeline.chunk_index}  |  Dropped: {dropped}"
                )

            elif t == "status":
                status = d.get("status", "")
                if status == "recording":
                    self.device_label.config(text=d.get("device", "?"))
                elif status == "stopped":
                    pass

            elif t == "asr_text":
                text = d.get("text", "").strip()
                is_final = d.get("is_final", False)
                if text:
                    if is_final:
                        if self._partial_row_active and self.asr_list.size() > 0:
                            self.asr_list.delete(tk.END)
                        self.asr_list.insert(tk.END, text)
                        self._partial_row_active = False
                    else:
                        if self._partial_row_active and self.asr_list.size() > 0:
                            self.asr_list.delete(tk.END)
                        self.asr_list.insert(tk.END, text + " ...")
                        self._partial_row_active = True
                    while self.asr_list.size() > 50:
                        self.asr_list.delete(0)
                    self.asr_list.see(tk.END)

            elif t == "error":
                self.log_list.insert(0, f"[ERR] {d.get('message', '')}")

        self.root.after(100, self._poll_log_queue)

    def _update_meter(self):
        """Update level meter and VAD state label."""
        if self.pipeline.is_running and self.pipeline.dsp_vad:
            state = self.pipeline.dsp_vad.state
            duration = getattr(self.pipeline.dsp_vad, "speech_duration", 0.0)

            self.meter_canvas.delete("all")
            w = self.meter_canvas.winfo_width() or (self.WINDOW_W - 20)

            if state == "SPEECH":
                filled = min(int(w * min(duration / 5.0, 1.0)), w)
                self.meter_canvas.create_rectangle(0, 0, filled, 20,
                                                    fill="#22c55e", outline="")
                self.meter_canvas.create_rectangle(filled, 0, w, 20,
                                                    fill="#374151", outline="")
                self.state_label.config(text=f"SPEECH  {duration:.1f}s")
            elif state == "PENDING_FINALIZE":
                self.meter_canvas.create_rectangle(0, 0, w, 20,
                                                    fill="#eab308", outline="")
                self.state_label.config(text="PENDING...")
            else:
                self.meter_canvas.create_rectangle(0, 0, w, 20,
                                                    fill="#1f2937", outline="")
                self.state_label.config(text="SILENCE")

        self.root.after(150, self._update_meter)

    def _start_hotkeys(self):
        """Start global hotkey listener using pynput."""
        if not HAS_PYNPUT:
            self.log_list.insert(0, "[WARN] pynput not installed — hotkeys disabled")
            return

        def on_toggle():
            self.root.after(0, self._toggle)

        def on_quit():
            self.root.after(0, self._on_close)

        hotkeys = {
            HOTKEY_TOGGLE: on_toggle,
            HOTKEY_QUIT: on_quit,
        }

        self._hotkey_listener = pynput_kb.GlobalHotKeys(hotkeys)
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()

    def run(self):
        """Start the tkinter main loop."""
        self.root.mainloop()
