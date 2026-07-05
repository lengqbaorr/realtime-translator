# Kien truc codebase hien tai

Tai lieu nay mo ta kien truc hien tai cua repo `realtime-capture-translate` dua tren source code trong workspace. Ung dung hien tai tap trung vao thu audio he thong theo thoi gian thuc tren Windows, xu ly DSP, phat hien giong noi bang Silero VAD, tach cau noi thanh cac segment va tuy chon luu WAV debug. Phan translate/transcription chua co implementation rieng trong codebase; hien tai he thong chi cung cap callback hook de noi tiep cac buoc Phase 2.

## Tong quan

Codebase la mot ung dung Python cho Windows, gom hai che do chay:

- CLI: `main.py`
- UI: `main_ui.py`

Core runtime nam trong package `core/`. Luong xu ly chinh do `Pipeline` dieu phoi, voi ba tang thuc thi:

```text
WASAPI loopback
    |
    v
CaptureThread
    |
    v
raw_buffer Queue[bytes]
    |
    v
DspVad
    |
    v
segment_queue Queue[(audio, speech_ms, total_ms)]
    |
    v
Consumer loop
    |
    +--> segment_callback(segment, speech_ms, total_ms)
    |
    +--> wav_queue Queue[(filename, audio, sample_rate)]
             |
             v
        WAV writer thread
```

## Cau truc thu muc

```text
.
├── main.py
├── main_ui.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── core/
│   ├── __init__.py
│   ├── audio_capture.py
│   ├── benchmark.py
│   ├── capture_thread.py
│   ├── config.py
│   ├── dsp_vad.py
│   ├── pipeline.py
│   └── ui.py
├── tests/
│   ├── __init__.py
│   ├── test_audio.py
│   └── test_vad_segmentation.py
└── captured_speech/
    └── chunk_*.wav
```

`captured_speech/` la thu muc output debug do runtime tao ra khi `DEBUG_SAVE_WAV = True`. No dang la artifact sinh ra, khong phai module source.

## Entry points

### `main.py`

Day la entrypoint CLI. File nay:

- Tao `threading.Event` lam tin hieu dung.
- Dang ky handler cho `SIGINT` de bat `Ctrl+C`.
- Khoi tao `AudioCapture`.
- Goi `AudioCapture.start(stop_event)`.

`AudioCapture` khong tu minh xu ly audio; no chi la wrapper cho `Pipeline`.

### `main_ui.py`

Day la entrypoint UI. File nay:

- Khoi tao `CaptureUI`.
- Goi `ui.run()` de chay `tkinter` main loop.

UI co global hotkey:

- `Ctrl+Shift+R`: bat/tat recording.
- `Ctrl+Shift+Q`: thoat ung dung.

## Cau hinh

Tat ca cau hinh runtime nam trong `core/config.py`, dataclass `CaptureConfig`.

Nhom cau hinh chinh:

- Audio target:
  - `TARGET_SAMPLE_RATE = 16000`
  - `CHUNK_SIZE = 512`
- VAD va segmentation:
  - `VAD_THRESHOLD = 0.5`
  - `PRE_SPEECH_PAD_MS = 300`
  - `POST_SPEECH_PAD_MS = 200`
  - `MIN_SPEECH_DURATION_MS = 300`
  - `VAD_MIN_SILENCE_MS = 600`
  - `MAX_SEGMENT_DURATION_S = 30`
- Queue backpressure:
  - `RAW_QUEUE_MAXSIZE = 30`
  - `SEGMENT_QUEUE_MAXSIZE = 10`
  - `WAV_QUEUE_MAXSIZE = 20`
- Debug output:
  - `DEBUG_SAVE_WAV = True`
  - `DEBUG_SAVE_DIR = "captured_speech"`
- DSP:
  - `RESAMPLE_QUALITY = "HQ"`
  - `MONO_STRATEGY = "average_safe"`

`CaptureConfig` cung cap cac property quy doi millisecond sang sample count, vi du `PRE_SPEECH_PAD_SAMPLES`, `VAD_MIN_SILENCE_SAMPLES` va `MAX_SEGMENT_SAMPLES`.

## Cac module core

### `core/audio_capture.py`

`AudioCapture` la lop wrapper tuong thich nguoc quanh `Pipeline`.

Public API:

- `AudioCapture(callback=None, config=None)`
- `set_callback(callback)`
- `start(stop_event)`
- `cleanup()`

Neu callback duoc truyen vao, wrapper gan callback do vao pipeline bang `pipeline.set_segment_callback(callback)`.

### `core/pipeline.py`

`Pipeline` la trung tam dieu phoi runtime.

Trach nhiem:

- Tao cac queue noi bo:
  - `raw_buffer`: raw audio bytes tu PortAudio callback.
  - `segment_queue`: segment hoan thanh tu DSP/VAD.
  - `wav_queue`: job ghi WAV debug.
- Khoi tao `CaptureThread`.
- Khoi tao `DspVad` sau khi da biet sample rate va channel count cua thiet bi.
- Start/stop cac thread nen.
- Phat event UI qua `log_queue`.
- Goi callback phase sau khi co segment.
- Ghi file WAV debug neu bat cau hinh.
- Tong hop stats runtime.

Hai che do chay:

- `run(stop_event)`: blocking mode cho CLI.
- `start_async()` va `stop()`: non-blocking mode cho UI.

Thread model trong `Pipeline`:

```text
PortAudio callback thread
    -> CaptureThread._callback()
    -> raw_buffer.put_nowait(raw bytes)

DSP daemon thread
    -> Pipeline._dsp_loop()
    -> DspVad.process_chunk(raw bytes)
    -> segment_queue.put_nowait(segment)

Consumer daemon thread
    -> Pipeline._consumer_loop()
    -> enqueue WAV write
    -> call segment_callback
    -> emit UI/log event

WAV daemon thread
    -> Pipeline._wav_loop()
    -> soundfile.write(...)
```

Backpressure hien tai:

- Neu `raw_buffer` day, `CaptureThread` tang `stats_dropped_raw` va bo chunk moi.
- Neu `segment_queue` day, `Pipeline` bo segment cu nhat roi chen segment moi, dong thoi tang `dropped_segments`.
- Neu `wav_queue` day, consumer hien tai dung `put_nowait`; truong hop queue full co the gay exception chua duoc bat trong consumer loop.

Event UI:

- `status`: trang thai recording/stopped va device name.
- `chunk`: chunk index, filename, total duration, speech duration, dropped count.
- `vad`: state hien tai cua VAD va speech duration.
- `error`: loi ghi WAV.

Luu y ve implementation hien tai:

- `run(stop_event)` goi `_start_threads()`, trong khi `_start_threads()` da start `consumer_thread`, sau do `run()` lai goi `_consumer_loop(stop_event)` tren thread chinh. Nghia la CLI mode hien tai co kha nang co hai consumer cung doc `segment_queue`.
- `stop()` dat `stop_event`, day sentinel `None` vao `segment_queue`, join cac thread va dung capture stream. `wav_loop` thoat theo `stop_event`; sentinel `None` cho `wav_queue` chi duoc xu ly neu co item, nhung `stop()` hien tai khong day sentinel vao `wav_queue`.

### `core/capture_thread.py`

`CaptureThread` quan ly WASAPI loopback capture qua `pyaudiowpatch`.

Vong doi:

1. `initialize()`
   - Tao `pyaudio.PyAudio()`.
   - Tim loopback device tu default output device cua WASAPI.
   - Luu `device_rate`, `device_channels`, `device_info`.
2. `start()`
   - Mo PortAudio stream o callback mode.
   - Tinh `frames_per_buffer` dua tren device sample rate va target sample rate.
   - Bat dau stream.
3. `_callback(...)`
   - Chay trong audio thread cua PortAudio.
   - Giu callback nhe: chi day raw bytes vao queue va dem overflow/drop.
4. `stop()`
   - Dung stream.
   - Dong stream.
   - Terminate PyAudio.

Device discovery:

- Lay WASAPI host API.
- Lay default output device.
- Neu default output khong phai loopback device, tim loopback device co ten khop voi speakers.
- Neu khong tim thay, raise `RuntimeError`.

### `core/dsp_vad.py`

`DspVad` thuc hien xu ly DSP va state machine segmentation.

Pipeline xu ly mot raw chunk:

```text
raw bytes int16
    |
    v
numpy int16 array
    |
    v
stereo -> mono
    |
    v
float32 normalized [-1.0, 1.0]
    |
    v
soxr resample device_rate -> 16 kHz
    |
    v
accumulator -> fixed 512-sample VAD chunks
    |
    v
Silero VADIterator
    |
    v
state machine segmentation
```

Mono conversion:

- Neu `device_channels > 1` va `MONO_STRATEGY == "average_safe"`, code ep left/right sang `int32`, lay trung binh bang integer division, roi ep lai `int16`.
- Neu khong, lay channel dau tien.

Resampling:

- Dung `soxr.ResampleStream`.
- Input rate la sample rate cua device.
- Output rate la `TARGET_SAMPLE_RATE`, mac dinh 16 kHz.
- Output dtype `float32`.

Accumulator:

- `ACCUMULATOR_SIZE = CHUNK_SIZE * 4`.
- Resampled output duoc tich luy cho den khi du `CHUNK_SIZE`.
- Moi lan du 512 samples, cat ra mot `chunk_16k` cho VAD.

State machine:

```text
IDLE
  |
  | VAD start
  v
SPEECH
  |
  | VAD end
  v
PENDING_FINALIZE
  |
  | VAD start truoc khi het hangover
  v
SPEECH

PENDING_FINALIZE
  |
  | silence >= VAD_MIN_SILENCE_MS
  v
finalize segment -> IDLE
```

Y nghia cac state:

- `IDLE`: chua co speech dang active, chi duy tri pre-buffer.
- `SPEECH`: dang ghi chunk vao segment buffer.
- `PENDING_FINALIZE`: VAD da bao end, nhung he thong tiep tuc append audio trong khoang hangover de merge cac pause ngan.

Pre-buffer:

- `_update_pre_buffer()` luu cac chunk gan nhat.
- Gioi han boi `PRE_SPEECH_PAD_SAMPLES`.
- Khi speech bat dau, `_start_segment()` copy pre-buffer vao segment de khong cat mat dau cau.

Segment buffer:

- `segment_buf` la numpy array pre-allocated voi kich thuoc `MAX_SEGMENT_SAMPLES`.
- `write_pos` la vi tri ghi hien tai.
- `speech_only_start` danh dau noi bat dau speech thuc su sau pre-padding.

Finalize:

- `_finalize_segment()` chi emit segment neu speech duration dat `MIN_SPEECH_DURATION_SAMPLES`.
- Segment tra ve co dang `(segment_array, speech_ms, total_ms)`.
- `vad_iterator.reset_states()` chi duoc goi tai thoi diem finalize.

### `core/ui.py`

`CaptureUI` la giao dien Tkinter dieu khien pipeline.

Thanh phan UI:

- Status dot va label recording/stopped/error.
- Nut Start/Stop.
- Device label.
- Level meter theo state VAD.
- Danh sach chunk da capture.
- Stats chunk/dropped.
- Hotkey hint.

Thread-safety:

- `Pipeline` khong update Tkinter truc tiep.
- Pipeline day `PipelineEvent` vao `log_queue`.
- UI goi `_poll_log_queue()` bang `root.after(...)` tren main thread.
- Meter doc state cua `pipeline.dsp_vad` theo chu ky bang `_update_meter()`.

Hotkey:

- Dung `pynput.keyboard.GlobalHotKeys`.
- Callback hotkey dung `root.after(0, ...)` de chuyen thao tac ve Tkinter main thread.

### `core/benchmark.py`

Module benchmark co hai che do:

- Synthetic CPU benchmark:
  - Tao chunk stereo 48 kHz bang numpy.
  - Khoi tao `DspVad` voi device gia lap 48 kHz, 2 channels.
  - Do thoi gian `dsp.process_chunk()` qua nhieu chunk.
- Real-device overflow test:
  - Chay `Pipeline` trong mot thoi luong cau hinh.
  - Theo doi stats: chunk ghi, `paInputOverflow`, dropped raw chunks, dropped segments, state luc thoat.

## Tests

### `tests/test_vad_segmentation.py`

Test state machine VAD bang `MockVADIterator`, khong phu thuoc vao audio device thuc.

Test cases:

- `test_short_pause_merges`: pause ngan hon `VAD_MIN_SILENCE_MS` phai merge thanh mot segment.
- `test_long_pause_splits`: pause dai hon `VAD_MIN_SILENCE_MS` phai tach thanh hai segment.
- `test_buffer_continuity`: kiem tra segment output co du sample qua chuyen state.
- `test_no_double_silence_mechanism`: dam bao khong con silence threshold thu hai va `reset_states()` chi nam trong finalize.

### `tests/test_audio.py`

Test integration voi WASAPI loopback thuc:

- Ghi loopback audio trong vai giay.
- Luu WAV vao temp dir.
- Kiem tra file ton tai, co audio data, sample count gan voi duration ky vong.
- Tinh RMS de log muc audio.

Test nay phu thuoc Windows, WASAPI device va audio dang phat.

## Data flow chi tiet

### 1. Capture raw audio

`CaptureThread` mo WASAPI loopback stream bang `pyaudiowpatch`. Audio callback nhan `in_data` o dang bytes int16, co so channel va sample rate goc cua device. Callback khong lam numpy, khong resample, khong VAD va khong I/O. No chi day bytes vao `raw_buffer`.

### 2. DSP va VAD

`Pipeline._dsp_loop()` lay raw bytes tu `raw_buffer` va goi `DspVad.process_chunk(raw_bytes)`.

`DspVad`:

1. Convert bytes sang numpy int16.
2. Downmix stereo/multi-channel ve mono.
3. Normalize sang float32.
4. Resample ve 16 kHz.
5. Cat thanh chunk 512 samples.
6. Goi Silero `VADIterator`.
7. Cap nhat state machine.
8. Khi segment hoan thanh, tra ve `(segment, speech_ms, total_ms)`.

### 3. Segment consumption

Neu `DspVad` tra ve segment, `_dsp_loop()` day vao `segment_queue`.

`_consumer_loop()`:

1. Lay segment tu `segment_queue`.
2. Neu bat debug WAV, tang `chunk_index` va day job ghi WAV vao `wav_queue`.
3. Goi `segment_callback(segment, speech_ms, total_ms)` neu callback da duoc set.
4. Emit event `chunk` cho UI/log.

### 4. WAV debug writer

`_wav_loop()` lay item tu `wav_queue` va ghi file bang `soundfile.write(filename, audio, sr)`.

File duoc ghi vao `captured_speech/chunk_XXX.wav`.

## Extension points

### Noi transcription/translation

Diem noi chinh la:

```python
pipeline.set_segment_callback(callback)
```

Hoac qua wrapper:

```python
capture = AudioCapture(callback=callback)
```

Callback nhan:

```python
def callback(segment, speech_ms, total_ms):
    ...
```

Trong do:

- `segment`: numpy array float32 mono 16 kHz.
- `speech_ms`: thoi luong speech thuc, khong tinh pre-padding.
- `total_ms`: tong duration segment, co tinh pre-padding va hangover silence.

Vi callback duoc goi trong consumer thread, code transcription/translation nen tranh block qua lau. Neu tac vu nang, nen day segment sang queue rieng hoac worker pool.

### Thay doi segmentation

Nhung tham so nen dieu chinh dau tien:

- `VAD_THRESHOLD`: do nhay cua Silero VAD.
- `VAD_MIN_SILENCE_MS`: pause dai bao nhieu thi tach segment.
- `PRE_SPEECH_PAD_MS`: giu lai bao nhieu audio truoc speech start.
- `MIN_SPEECH_DURATION_MS`: bo cac speech qua ngan.
- `MAX_SEGMENT_DURATION_S`: tran bao ve bo nho segment.

### Thay doi device/capture

`CaptureThread._find_loopback_device()` hien mac dinh chon loopback device tu default output device. Neu can chon device thu cong, nen them config cho device index/name va sua logic discovery tai module nay.

## Phu thuoc chinh

Tu `requirements.txt`:

- `pyaudiowpatch`: capture WASAPI loopback.
- `silero-vad`: VAD model va iterator.
- `torch`: backend cho Silero VAD.
- `numpy`: xu ly buffer audio.
- `soxr`: resampling.
- `soundfile`: ghi WAV debug va doc/ghi test audio.
- `pynput`: global hotkeys cho UI.
- `pywin32`: thread priority tren Windows.

## Gioi han va rui ro hien tai

- Code hien phu thuoc Windows/WASAPI, khong portable sang macOS/Linux neu khong thay capture backend.
- Translate/transcription chua co trong source, chi co callback hook.
- CLI mode co kha nang start hai consumer loop cung doc `segment_queue`.
- `wav_queue.put_nowait(...)` trong consumer loop chua bat `queue.Full`.
- `POST_SPEECH_PAD_MS` co property trong config nhung state machine hien tai dung hangover `VAD_MIN_SILENCE_MS`; post-pad rieng chua duoc ap dung truc tiep trong finalize.
- Test `test_audio.py` can audio device thuc va co the fail tren CI/headless environment.
- Thu muc `captured_speech/` chua nam trong `.gitignore`, nen WAV debug co the bi hien la untracked.

## Cach chay

CLI:

```bash
python main.py
```

UI:

```bash
python main_ui.py
```

Synthetic benchmark:

```bash
python -m core.benchmark
```

Real-device benchmark:

```bash
python -m core.benchmark --real --duration 300
```

Tests:

```bash
python -m pytest tests
```

Luu y: repo hien khong khai bao `pytest` trong `requirements.txt`, va mot so test phu thuoc audio device thuc.
