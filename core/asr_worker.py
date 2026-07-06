import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.config import AsrConfig


@dataclass
class AsrResult:
    text: str
    is_final: bool
    sequence: int
    speech_ms: float | None = None
    total_ms: float | None = None


class AsrWorker:
    """Single-worker ASR processor using sherpa-onnx OnlineRecognizer.

    One daemon thread reads from an internal bounded queue.
    Queue items are tagged tuples: ("chunk", np.ndarray).
    Utterance finalization is owned by sherpa-onnx endpoint detection.
    """

    def __init__(
        self,
        config: AsrConfig | None = None,
        result_callback: Callable[[AsrResult], None] | None = None,
    ):
        self.config = config or AsrConfig()
        self.result_callback = result_callback

        self._recognizer = None
        self._stream = None
        self._stop_event = threading.Event()
        self._asr_queue: queue.Queue = queue.Queue(maxsize=self.config.asr_queue_maxsize)
        self._result_queue: queue.Queue = queue.Queue(
            maxsize=self.config.result_queue_maxsize
        )
        self._worker_thread: threading.Thread | None = None
        self._result_thread: threading.Thread | None = None
        self._sequence = 0
        self._lock = threading.Lock()
        self.dropped_live_chunks: int = 0
        self._last_emitted_partial_text: str = ""
        self._committed_text: str = ""
        self._last_partial_time: float = 0.0
        self._endpoint_pending: bool = False
        self._total_samples: int = 0
        self._stream_id: int = 0
        self._stream_chunk_count: int = 0
        self._last_stream_log_time: float = 0.0
        self._pending_soft_boundary = None

    def start(self):
        if self._recognizer is None:
            self._load_model()
        self._warmup_recognizer()
        self._stream = self._create_stream()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self._result_thread = threading.Thread(
            target=self._result_loop,
            daemon=True,
        )
        self._result_thread.start()
        self._worker_thread.start()
        self._log("worker started")

    def _log(self, message: str):
        if self.config.debug_console_logs:
            print(f"\n[ASRDBG] {message}", flush=True)

    def _load_model(self):
        import sherpa_onnx

        print(f"[ASR] Loading sherpa-onnx transducer model...")
        print(f"[ASR] Encoder: {self.config.encoder}")
        print(f"[ASR] Decoder: {self.config.decoder}")
        print(f"[ASR] Joiner: {self.config.joiner}")
        print(f"[ASR] Tokens: {self.config.tokens}")
        print("[ASR] Endpoint rules: "
              f"rule1={self.config.endpoint_rule1_min_trailing_silence}s, "
              f"rule2={self.config.endpoint_rule2_min_trailing_silence}s, "
              f"rule3={self.config.endpoint_rule3_min_utterance_length}s")
        print("[ASR] Endpoint guard: "
              f"min_duration={self.config.min_endpoint_duration_ms}ms, "
              f"min_words={self.config.min_endpoint_words}")
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=self.config.tokens,
            encoder=self.config.encoder,
            decoder=self.config.decoder,
            joiner=self.config.joiner,
            num_threads=self.config.num_threads,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=self.config.endpoint_rule1_min_trailing_silence,
            rule2_min_trailing_silence=self.config.endpoint_rule2_min_trailing_silence,
            rule3_min_utterance_length=self.config.endpoint_rule3_min_utterance_length,
            decoding_method="greedy_search",
            provider="cpu",
        )
        print("[ASR] Model loaded")

    def _warmup_recognizer(self):
        """Run model warmup on a disposable stream.

        Do not feed warmup silence into the real streaming session. With
        endpoint detection enabled, leading artificial silence in every new
        stream can cause premature endpoints and clip the next utterance start.
        """
        silence_samples = int(self.config.warmup_silence_ms * 16000 / 1000)
        if silence_samples <= 0:
            return

        self._log(f"warmup disposable stream: {silence_samples} samples")
        stream = self._recognizer.create_stream()
        silence = np.zeros(silence_samples, dtype=np.float32)
        stream.accept_waveform(16000, silence)
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

    def _create_stream(self):
        stream = self._recognizer.create_stream()
        self._stream_id += 1
        self._stream_chunk_count = 0
        self._total_samples = 0
        self._endpoint_pending = False
        self._last_emitted_partial_text = ""
        self._last_partial_time = time.time()
        self._last_stream_log_time = time.time()
        self._prime_initial_stream(stream)
        self._log(f"stream #{self._stream_id} created; live stream ready")
        return stream

    def _prime_initial_stream(self, stream):
        """Add short leading context only to the first live stream."""
        if self._stream_id != 1:
            return

        silence_samples = int(self.config.initial_stream_context_ms * 16000 / 1000)
        if silence_samples <= 0:
            return

        silence = np.zeros(silence_samples, dtype=np.float32)
        stream.accept_waveform(16000, silence)
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        self._log(
            f"stream #1 primed with {self.config.initial_stream_context_ms}ms "
            "leading silence context"
        )

    def submit_live_chunk(self, chunk: np.ndarray):
        """Feed a 16kHz float32 audio chunk to the streaming ASR."""
        item = ("chunk", chunk)
        self._put_asr_item(item)

    def submit_soft_boundary(self, boundary_type: str, pause_ms: float):
        """Commit current ASR hypothesis at a pause without resetting stream."""
        due_time = time.time() + self.config.soft_boundary_commit_delay_ms / 1000
        item = ("boundary", boundary_type, pause_ms, due_time)
        self._put_asr_item(item)

    def _put_asr_item(self, item):
        try:
            self._asr_queue.put_nowait(item)
        except queue.Full:
            try:
                self._asr_queue.get_nowait()
                self.dropped_live_chunks += 1
                self._log(
                    "ASR queue full; dropped oldest live chunk "
                    f"(dropped={self.dropped_live_chunks}, "
                    f"queue_size={self._asr_queue.qsize()})"
                )
            except queue.Empty:
                pass
            try:
                self._asr_queue.put_nowait(item)
            except queue.Full:
                self.dropped_live_chunks += 1
                self._log(
                    "ASR queue still full; dropped incoming item "
                    f"(dropped={self.dropped_live_chunks})"
                )

    def stop(self):
        self._log("stop requested")
        self._stop_event.set()
        try:
            self._asr_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        self._enqueue_result_sentinel()
        if self._result_thread and self._result_thread.is_alive():
            self._result_thread.join(timeout=5)
        self._log(
            f"worker stopped; dropped_live_chunks={self.dropped_live_chunks}"
        )

    def _enqueue_result_sentinel(self):
        try:
            self._result_queue.put_nowait(None)
        except queue.Full:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._result_queue.put_nowait(None)
            except queue.Full:
                pass

    def _submit_result(self, result: AsrResult):
        try:
            self._result_queue.put_nowait(result)
        except queue.Full:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                self._log("result queue full; dropped ASR result")

    def _result_loop(self):
        while True:
            try:
                result = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    continue
                continue

            if result is None:
                break

            if not self.result_callback:
                continue

            try:
                self.result_callback(result)
            except Exception as exc:
                print(f"[ASR] Error in result callback: {exc}", flush=True)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._asr_queue.get(timeout=0.05)
            except queue.Empty:
                self._maybe_commit_pending_soft_boundary()
                continue

            if item is None:
                break

            try:
                if item[0] == "chunk":
                    self._process_chunk(item[1])
                elif item[0] == "boundary":
                    self._store_soft_boundary(item[1], item[2], item[3])
            except Exception as exc:
                print(f"[ASR] Error processing {item[0]}: {exc}")

    def _process_chunk(self, chunk: np.ndarray):
        self._total_samples += len(chunk)
        self._stream_chunk_count += 1
        if self._stream_chunk_count == 1:
            self._log(
                f"stream #{self._stream_id} first live chunk: "
                f"{len(chunk)} samples, "
                f"queue_size={self._asr_queue.qsize()}"
            )
        now = time.time()
        if now - self._last_stream_log_time >= self.config.debug_stream_log_interval_s:
            self._last_stream_log_time = now
            self._log(
                f"stream #{self._stream_id} progress: "
                f"{self._total_samples * 1000 / 16000:.0f}ms audio, "
                f"chunks={self._stream_chunk_count}, "
                f"queue_size={self._asr_queue.qsize()}, "
                f"dropped={self.dropped_live_chunks}"
            )
        self._stream.accept_waveform(16000, chunk)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        elapsed_ms = (now - self._last_partial_time) * 1000
        if elapsed_ms >= self.config.partial_update_interval_ms:
            text = self._recognizer.get_result(self._stream).strip()
            delta = self._get_uncommitted_text(text)
            if delta and delta != self._last_emitted_partial_text:
                self._last_emitted_partial_text = delta
                self._last_partial_time = now
                with self._lock:
                    seq = self._sequence
                    self._sequence += 1
                self._submit_result(AsrResult(
                        text=delta, is_final=False, sequence=seq,
                    ))

        if self._recognizer.is_endpoint(self._stream) and not self._endpoint_pending:
            endpoint_text = self._recognizer.get_result(self._stream).strip()
            if not self._should_accept_endpoint(endpoint_text):
                self._log(
                    f"endpoint suppressed on stream #{self._stream_id}: "
                    f"{self._total_samples * 1000 / 16000:.0f}ms audio, "
                    f"words={len(endpoint_text.split())}, "
                    f"text='{endpoint_text[:80]}'"
                )
                return

            self._log(
                f"endpoint detected on stream #{self._stream_id}: "
                f"{self._total_samples * 1000 / 16000:.0f}ms audio, "
                f"chunks={self._stream_chunk_count}, "
                f"queue_size={self._asr_queue.qsize()}"
            )
            self._signal_endpoint(endpoint_text)
            return

        self._maybe_commit_pending_soft_boundary(now)

    def _store_soft_boundary(self, boundary_type: str, pause_ms: float, due_time: float):
        current = self._pending_soft_boundary
        if current and current["type"] == "sentence" and boundary_type == "comma":
            return
        now = time.time()
        self._pending_soft_boundary = {
            "type": boundary_type,
            "pause_ms": pause_ms,
            "due_time": due_time,
            "expires_at": now + self.config.soft_boundary_max_wait_ms / 1000,
        }
        self._log(
            f"soft boundary {boundary_type} pending: "
            f"pause={pause_ms:.0f}ms, "
            f"delay={self.config.soft_boundary_commit_delay_ms}ms"
        )

    def _maybe_commit_pending_soft_boundary(self, now: float | None = None):
        if not self._pending_soft_boundary:
            return

        now = now or time.time()
        if now < self._pending_soft_boundary["due_time"]:
            return

        pending = self._pending_soft_boundary
        text = self._recognizer.get_result(self._stream).strip()
        delta = self._get_uncommitted_text(text)
        reason = self._soft_boundary_block_reason(delta)

        if reason:
            if now < pending["expires_at"]:
                pending["due_time"] = (
                    now + self.config.soft_boundary_retry_interval_ms / 1000
                )
                self._log(
                    f"soft boundary {pending['type']} waiting: {reason}, "
                    f"text='{delta[:100]}'"
                )
                return

            self._pending_soft_boundary = None
            self._log(
                f"soft boundary {pending['type']} suppressed: {reason}, "
                f"text='{delta[:100]}'"
            )
            return

        self._pending_soft_boundary = None
        self._commit_soft_boundary(pending["type"], pending["pause_ms"], delta, text)

    def _soft_boundary_block_reason(self, delta: str) -> str | None:
        if not delta:
            return "no new text"

        word_count = len(delta.split())
        if word_count < self.config.min_soft_boundary_words:
            return f"too few words ({word_count})"
        if self._has_unstable_tail(delta):
            return "unstable tail"

        return None

    def _commit_soft_boundary(
        self,
        boundary_type: str,
        pause_ms: float,
        delta: str,
        full_text: str,
    ):
        punct = "," if boundary_type == "comma" else "."
        display_text = self._with_punctuation(delta, punct)

        self._committed_text = full_text
        self._last_emitted_partial_text = ""
        self._last_partial_time = time.time()

        with self._lock:
            seq = self._sequence
            self._sequence += 1

        self._submit_result(AsrResult(
            text=display_text,
            is_final=True,
            sequence=seq,
            speech_ms=self._total_samples * 1000 / 16000,
            total_ms=self._total_samples * 1000 / 16000,
        ))
        self._log(
            f"soft boundary {boundary_type} committed: "
            f"pause={pause_ms:.0f}ms, text='{display_text[:120]}'"
        )

    def _get_uncommitted_text(self, text: str) -> str:
        if not text:
            return ""
        if self._committed_text and text.startswith(self._committed_text):
            return text[len(self._committed_text):].strip()
        if self._committed_text:
            common = 0
            max_common = min(len(text), len(self._committed_text))
            while common < max_common and text[common] == self._committed_text[common]:
                common += 1
            return text[common:].strip()
        return text.strip()

    def _with_punctuation(self, text: str, punct: str) -> str:
        text = text.strip()
        if not text:
            return text
        if text[-1] in ",.!?;:":
            return text
        return f"{text}{punct}"

    def _has_unstable_tail(self, text: str) -> bool:
        words = text.strip().split()
        if not words:
            return True

        tail = words[-1].strip(" ,.!?;:").upper()
        unstable_tails = {
            "A", "AN", "THE",
            "TO", "IN", "INTO", "OF", "ON", "AT", "BY", "FOR", "FROM",
            "WITH", "AS", "AND", "OR", "BUT",
            "THAT", "THIS", "THESE", "THOSE",
            "O", "T", "TRA", "LOW",
        }
        return len(tail) <= 1 or tail in unstable_tails

    def _should_accept_endpoint(self, text: str) -> bool:
        duration_ms = self._total_samples * 1000 / 16000
        word_count = len(text.split())
        return not (
            duration_ms < self.config.min_endpoint_duration_ms
            and word_count < self.config.min_endpoint_words
        )

    def _signal_endpoint(self, text: str | None = None):
        stream_id = self._stream_id
        self._endpoint_pending = True

        if text is None:
            text = self._recognizer.get_result(self._stream).strip()
        speech_ms = self._total_samples * 1000 / 16000

        with self._lock:
            seq = self._sequence
            self._sequence += 1

        if text:
            delta = self._get_uncommitted_text(text)
        else:
            delta = ""

        if delta:
            self._submit_result(AsrResult(
                text=self._with_punctuation(delta, "."),
                is_final=True,
                sequence=seq,
                speech_ms=speech_ms, total_ms=speech_ms,
            ))
        self._log(
            f"final emitted from stream #{stream_id}: "
            f"seq={seq}, speech_ms={speech_ms:.0f}, text='{delta[:120]}'"
        )

        self._recognizer.reset(self._stream)
        self._stream_id += 1
        self._stream_chunk_count = 0
        self._total_samples = 0
        self._endpoint_pending = False
        self._last_emitted_partial_text = ""
        self._committed_text = ""
        self._pending_soft_boundary = None
        self._last_partial_time = time.time()
        self._last_stream_log_time = time.time()
        self._log(
            f"stream #{self._stream_id} reset by sherpa recognizer; "
            "same OnlineStream continues"
        )
