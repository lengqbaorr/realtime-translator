from dataclasses import dataclass, field


@dataclass
class CaptureConfig:
    TARGET_SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 512

    VAD_THRESHOLD: float = 0.5
    PRE_SPEECH_PAD_MS: int = 300
    POST_SPEECH_PAD_MS: int = 200
    MIN_SPEECH_DURATION_MS: int = 300
    VAD_MIN_SILENCE_MS: int = 600

    RAW_QUEUE_MAXSIZE: int = 30
    SEGMENT_QUEUE_MAXSIZE: int = 10
    WAV_QUEUE_MAXSIZE: int = 20

    LOG_INTERVAL_CHUNKS: int = 5
    RMS_SILENCE_THRESHOLD: float = 0.005
    SILENCE_WARN_SECONDS: int = 5

    DEBUG_SAVE_WAV: bool = True
    DEBUG_SAVE_DIR: str = "captured_speech"

    RESAMPLE_QUALITY: str = "HQ"
    MONO_STRATEGY: str = "average_safe"

    MAX_SEGMENT_DURATION_S: int = 30

    @property
    def PRE_SPEECH_PAD_SAMPLES(self) -> int:
        return self.PRE_SPEECH_PAD_MS * self.TARGET_SAMPLE_RATE // 1000

    @property
    def POST_SPEECH_PAD_SAMPLES(self) -> int:
        return self.POST_SPEECH_PAD_MS * self.TARGET_SAMPLE_RATE // 1000

    @property
    def MIN_SPEECH_DURATION_SAMPLES(self) -> int:
        return self.MIN_SPEECH_DURATION_MS * self.TARGET_SAMPLE_RATE // 1000

    @property
    def VAD_MIN_SILENCE_SAMPLES(self) -> int:
        return self.VAD_MIN_SILENCE_MS * self.TARGET_SAMPLE_RATE // 1000

    @property
    def MAX_SEGMENT_SAMPLES(self) -> int:
        return self.MAX_SEGMENT_DURATION_S * self.TARGET_SAMPLE_RATE
