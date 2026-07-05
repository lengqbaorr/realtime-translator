from core.config import CaptureConfig
from core.pipeline import Pipeline


class AudioCapture:
    """Backward-compatible wrapper around the refactored Pipeline.

    Preserves the same interface as Phase 1 v1:
      - AudioCapture(callback=...)
      - start(stop_event)
      - set_callback(...)
      - cleanup()

    Under the hood, delegates to Pipeline which runs the 3-tier
    threaded architecture (capture callback → DSP+VAD → consumer).
    """

    def __init__(self, callback=None, config=None):
        self.config = config or CaptureConfig()
        self.pipeline = Pipeline(self.config)
        if callback:
            self.pipeline.set_segment_callback(callback)

    def set_callback(self, callback):
        self.pipeline.set_segment_callback(callback)

    def start(self, stop_event):
        self.pipeline.run(stop_event)

    def cleanup(self):
        pass
