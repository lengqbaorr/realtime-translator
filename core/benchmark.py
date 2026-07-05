"""Benchmark script — CPU time, overflow counters, pipeline measurement.

Measures:
  - Total CPU time per dsp.process_chunk() call (synthetic)
  - Real-device capture for N seconds with overflow/drop tracking

Usage:
    # Synthetic CPU benchmark (no real device)
    python -m realtime_translator.core.benchmark

    # Real-device overflow test (5 min, requires audio playing)
    python -m realtime_translator.core.benchmark --real
"""

import argparse
import os
import statistics
import sys
import threading
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import CaptureConfig
from core.dsp_vad import DspVad
from core.pipeline import Pipeline


BASELINE_MS = 1.860


def make_synthetic_chunk(config, device_rate=48000):
    """Generate a synthetic stereo audio chunk for benchmarking."""
    base = int(config.CHUNK_SIZE * device_rate / config.TARGET_SAMPLE_RATE)
    samples = base * 4
    t = np.arange(samples, dtype=np.float64)
    left = 0.3 * np.sin(2 * np.pi * 440 * t / device_rate)
    right = 0.3 * np.sin(2 * np.pi * 440 * t / device_rate)
    stereo = np.column_stack((left, right))
    stereo_int16 = (stereo * 32767).astype(np.int16)
    return stereo_int16.tobytes()


def run_cpu_benchmark(num_chunks=500):
    """Synthetic CPU benchmark — measures dsp.process_chunk() time."""
    config = CaptureConfig()
    config.DEBUG_SAVE_WAV = False

    dsp = DspVad(
        raw_buffer=deque(),
        config=config,
        device_rate=48000,
        device_channels=2,
    )
    dsp.initialize()

    chunk = make_synthetic_chunk(config)

    for i in range(50):
        dsp.process_chunk(chunk)

    dsp_times = []
    for i in range(num_chunks):
        t0 = time.perf_counter_ns()
        result = dsp.process_chunk(chunk)
        t1 = time.perf_counter_ns()

        elapsed_ms = (t1 - t0) / 1_000_000
        if elapsed_ms < 5.0:
            dsp_times.append(elapsed_ms)

    print("\n=== CPU Time (per chunk) ===")
    print(f"Benchmark: {num_chunks} synthetic chunks @ 48kHz stereo\n")

    avg_ms = statistics.mean(dsp_times) if dsp_times else 0
    med_ms = statistics.median(dsp_times) if dsp_times else 0
    p99_ms = sorted(dsp_times)[int(len(dsp_times) * 0.99)] if dsp_times else 0
    change = ((avg_ms - BASELINE_MS) / BASELINE_MS) * 100 if BASELINE_MS else 0

    print(f"{'Metric':<35} {'Old (ms)':<12} {'New (ms)':<12} {'Change':<10}")
    print("-" * 72)
    print(f"{'dsp.process_chunk() avg':<35} {BASELINE_MS:<12.3f} {avg_ms:<12.3f} {change:+.0f}%")
    print(f"{'dsp.process_chunk() median':<35} {'':<12} {med_ms:<12.3f} {'':<10}")
    print(f"{'dsp.process_chunk() p99':<35} {'':<12} {p99_ms:<12.3f} {'':<10}")
    print()
    if change <= -50:
        print(f"  [OK] Target achieved: {-change:.0f}% reduction (>=50% target)")
    else:
        print(f"  [WARN] Target missed: {-change:.0f}% reduction (<50% target)")

    return avg_ms


def run_overflow_test(duration_s=300):
    """Real-device capture test — tracks overflow/drop counters.

    Captures audio from the default loopback device for `duration_s`
    seconds. Audio must be playing during the test for realistic results.
    Prints all 3 counters at the end.

    Returns dict with final counter values.
    """
    config = CaptureConfig()
    config.DEBUG_SAVE_WAV = True

    pipeline = Pipeline(config)
    stop_event = threading.Event()

    print(f"\n=== Real-Device Overflow Test ({duration_s}s) ===")
    print("Play audio now (YouTube, music, etc.)...\n")

    def timeout():
        stop_event.set()

    timer = threading.Timer(duration_s, timeout)
    timer.daemon = True
    timer.start()
    t_start = time.time()

    try:
        pipeline.run(stop_event)
    except Exception as exc:
        print(f"Pipeline error: {exc}")
    finally:
        timer.cancel()
        elapsed = time.time() - t_start

    stats = pipeline.get_stats()

    print(f"\n=== Results ({elapsed:.0f}s real device) ===")
    print(f"{'Chunks written':<30} {stats['chunks']}")
    print(f"{'paInputOverflow':<30} {stats['pa_input_overflows']}")
    print(f"{'Dropped raw chunks (queue full)':<30} {stats['dropped_raw_chunks']}")
    print(f"{'Dropped segments':<30} {stats['dropped_segments']}")
    print(f"{'VAD state at exit':<30} {stats['state']}")

    if (stats['pa_input_overflows'] == 0
            and stats['dropped_raw_chunks'] == 0
            and stats['dropped_segments'] == 0):
        print("\n  [OK] All counters = 0 — no real-time violations.")
    else:
        print("\n  [WARN] Non-zero counters detected — see above.")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Pipeline benchmark")
    parser.add_argument("--real", action="store_true",
                        help="Run real-device overflow test (default: synthetic CPU)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Overflow test duration in seconds (default: 300)")
    args = parser.parse_args()

    if args.real:
        run_overflow_test(args.duration)
    else:
        run_cpu_benchmark()


if __name__ == "__main__":
    main()
