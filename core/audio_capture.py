from core.config import CaptureConfig
from core.pipeline import Pipeline


class AudioCapture:
    """Backward-compatible wrapper around the refactored Pipeline.

    Preserves the same interface as Phase 1 v1:
      - AudioCapture(callback=..., live_callback=...)
      - start(stop_event)
      - cleanup()

    callback is kept only for old callers and is not wired.
    live_callback feeds continuous 16 kHz chunks to sherpa-onnx ASR.
    soft_boundary_callback emits pause-based text boundaries.
    sherpa-onnx endpoint detection handles finalization internally.
    """

    def __init__(self, callback=None, live_callback=None,
                 soft_boundary_callback=None, config=None):
        self.config = config or CaptureConfig()
        self.pipeline = Pipeline(self.config)
        if live_callback:
            self.pipeline.set_live_chunk_callback(live_callback)
        if soft_boundary_callback:
            self.pipeline.set_soft_boundary_callback(soft_boundary_callback)

    def start(self, stop_event):
        self.pipeline.run(stop_event)

    def cleanup(self):
        pass
