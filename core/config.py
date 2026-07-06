from dataclasses import dataclass, field


@dataclass
class CaptureConfig:
    TARGET_SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 512

    VAD_THRESHOLD: float = 0.5
    PRE_SPEECH_PAD_MS: int = 1200
    POST_SPEECH_PAD_MS: int = 300
    MIN_SPEECH_DURATION_MS: int = 300
    VAD_MIN_SILENCE_MS: int = 600
    SOFT_COMMA_PAUSE_MS: int = 500
    SOFT_SENTENCE_PAUSE_MS: int = 700

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

    MAX_SEGMENT_DURATION_S: int = 60

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


@dataclass
class AsrConfig:
    encoder: str = "models/zipformer-en/encoder-epoch-99-avg-1.int8.onnx"
    decoder: str = "models/zipformer-en/decoder-epoch-99-avg-1.int8.onnx"
    joiner: str = "models/zipformer-en/joiner-epoch-99-avg-1.int8.onnx"
    tokens: str = "models/zipformer-en/tokens.txt"
    num_threads: int = 2
    partial_update_interval_ms: int = 800
    asr_queue_maxsize: int = 60
    result_queue_maxsize: int = 100
    warmup_silence_ms: int = 1000
    initial_stream_context_ms: int = 500
    endpoint_rule1_min_trailing_silence: float = 60.0
    endpoint_rule2_min_trailing_silence: float = 1.6
    endpoint_rule3_min_utterance_length: float = 120.0
    min_endpoint_duration_ms: int = 2500
    min_endpoint_words: int = 3
    soft_boundary_commit_delay_ms: int = 600
    soft_boundary_retry_interval_ms: int = 150
    soft_boundary_max_wait_ms: int = 1800
    min_soft_boundary_words: int = 3
    debug_console_logs: bool = True
    debug_stream_log_interval_s: float = 2.0
