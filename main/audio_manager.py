"""
Audio Manager - handles microphone recording and audio playback.

Key design:
  - Records from microphone while pre-recorded audio plays through speakers.
  - The microphone captures EVERYTHING: the played audio + audience / ambient sounds.
  - After playback ends, recording continues for a configurable buffer period
    to catch trailing speech or reactions.
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import wave
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("Please install sounddevice:  pip install sounddevice")

try:
    import soundfile as sf
except ImportError:
    raise ImportError("Please install soundfile:  pip install soundfile")

import config


def _effective_playback_backend() -> str:
    """Resolve PLAYBACK_BACKEND: prefer afplay on macOS when available."""
    raw = getattr(config, "PLAYBACK_BACKEND", "auto")
    mode = raw.lower().strip() if isinstance(raw, str) else "auto"

    def _afplay_ready() -> bool:
        return sys.platform == "darwin" and shutil.which("afplay") is not None

    if mode == "sounddevice":
        return "sounddevice"
    if mode == "afplay":
        return "afplay" if _afplay_ready() else "sounddevice"
    # auto or unknown
    return "afplay" if _afplay_ready() else "sounddevice"


def _sd_output_stream_kwargs() -> dict:
    """kwargs for sd.play / OutputStream (latency + optional explicit device)."""
    kwargs: dict = {}
    lat = getattr(config, "PLAYBACK_LATENCY", None)
    if lat is not None:
        kwargs["latency"] = lat
    dev = getattr(config, "PLAYBACK_DEVICE", None)
    if dev is not None:
        kwargs["device"] = dev
    return kwargs


def _is_riff_wav_header(data: bytes) -> bool:
    return len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WAVE"


def _afplay_subprocess(path: str) -> None:
    subprocess.run(["afplay", path], check=True)


def _afplay_from_wav_bytes(wav_bytes: bytes) -> None:
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes)
        _afplay_subprocess(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _afplay_from_numpy(data: np.ndarray, sample_rate: int) -> None:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, data, sample_rate)
        _afplay_subprocess(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _afplay_from_file(path: str) -> None:
    """Play with afplay; transcode to temp WAV if format is not directly supported."""
    ext = Path(path).suffix.lower()
    if ext in (".wav", ".aiff", ".aif", ".caf"):
        _afplay_subprocess(path)
        return
    data, sr = sf.read(path, dtype="float32")
    _afplay_from_numpy(data, sr)


def playback_numpy_sync(data: np.ndarray, sample_rate: int) -> None:
    """Play float audio to the default output (afplay on mac when selected, else sounddevice)."""
    if _effective_playback_backend() == "afplay":
        _afplay_from_numpy(data, sample_rate)
        return
    sd.play(data, samplerate=sample_rate, **_sd_output_stream_kwargs())
    sd.wait()


class AudioRecorder:
    """Records audio from the default microphone into a WAV buffer."""

    def __init__(
        self,
        sample_rate: int = config.RECORD_SAMPLE_RATE,
        channels: int = config.RECORD_CHANNELS,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._stream = None

    def start(self):
        """Begin recording (non-blocking). Call stop() when done."""
        self._frames = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> bytes:
        """
        Stop recording and return the captured audio as WAV bytes.
        """
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            return b""

        audio_data = np.concatenate(self._frames, axis=0)
        return self._numpy_to_wav_bytes(audio_data)

    def _callback(self, indata, frames, time_info, status):
        if self._recording:
            self._frames.append(indata.copy())

    def _numpy_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy int16 audio array to in-memory WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        buf.seek(0)
        return buf.read()


class AudioPlayer:
    """Plays audio files or raw audio data through the default speakers."""

    def __init__(self):
        self._playback_done = threading.Event()

    def play_file(self, filepath: str, blocking: bool = True):
        """
        Play a WAV/FLAC/OGG file.
        If blocking=True, this call returns only after playback finishes.
        """
        if _effective_playback_backend() == "afplay":
            self._playback_done.clear()

            def body():
                _afplay_from_file(filepath)

            if blocking:
                body()
                self._playback_done.set()
            else:
                threading.Thread(
                    target=lambda: (body(), self._playback_done.set()),
                    daemon=True,
                ).start()
            return

        data, sr = sf.read(filepath, dtype="float32")
        self._play_array(data, sr, blocking)

    def play_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = config.PLAYBACK_SAMPLE_RATE,
        blocking: bool = True,
    ):
        """
        Play raw audio bytes (WAV or MP3 format).
        Tries to read format from the byte header.
        """
        # Coqui WAV：直接交给 afplay，避免 float 往返与 PortAudio 缓冲问题
        if _effective_playback_backend() == "afplay" and _is_riff_wav_header(audio_bytes):
            self._playback_done.clear()

            def body():
                _afplay_from_wav_bytes(audio_bytes)

            if blocking:
                body()
                self._playback_done.set()
            else:
                threading.Thread(
                    target=lambda: (body(), self._playback_done.set()),
                    daemon=True,
                ).start()
            return

        buf = io.BytesIO(audio_bytes)
        try:
            data, sr = sf.read(buf, dtype="float32")
        except Exception:
            # If soundfile can't read it (e.g. MP3), fall back to pydub
            buf.seek(0)
            data, sr = self._decode_with_pydub(buf)

        self._play_array(data, sr, blocking)

    def _play_array(self, data: np.ndarray, sample_rate: int, blocking: bool):
        self._playback_done.clear()

        def on_finished():
            self._playback_done.set()

        if _effective_playback_backend() == "afplay":

            def body():
                _afplay_from_numpy(data, sample_rate)

            if blocking:
                body()
                on_finished()
            else:
                threading.Thread(
                    target=lambda: (body(), on_finished()),
                    daemon=True,
                ).start()
            return

        sd.play(data, samplerate=sample_rate, **_sd_output_stream_kwargs())
        if blocking:
            sd.wait()
            on_finished()
        else:
            # Non-blocking: set event when playback ends (poll in a thread)
            def _wait_thread():
                sd.wait()
                on_finished()

            t = threading.Thread(target=_wait_thread, daemon=True)
            t.start()

    def wait_until_done(self):
        """Block until current playback finishes."""
        self._playback_done.wait()

    @staticmethod
    def _decode_with_pydub(buf: io.BytesIO):
        """Fallback decoder using pydub (handles MP3, etc.)."""
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub is required for MP3 playback: pip install pydub\n"
                "Also ensure ffmpeg is installed on your system."
            )
        buf.seek(0)
        seg = AudioSegment.from_file(buf)
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= 32768.0  # normalize int16 to float32
        if seg.channels == 2:
            samples = samples.reshape(-1, 2)
        return samples, sr


def record_during_playback(
    audio_filepath: str,
    post_buffer: float = config.POST_PLAYBACK_BUFFER,
) -> bytes:
    """
    High-level helper:
      1. Start microphone recording
      2. Play the given audio file through speakers
      3. After playback ends, keep recording for `post_buffer` seconds
      4. Stop recording and return captured WAV bytes

    The returned audio contains everything the mic heard:
    the played-back audio, audience voices, ambient noise, etc.
    """
    recorder = AudioRecorder()
    player = AudioPlayer()

    # 1. Start recording
    recorder.start()
    print("    [mic] Recording started...")

    # 2. Play audio (blocking)
    player.play_file(audio_filepath, blocking=True)
    print(f"    [mic] Playback done. Listening for {post_buffer}s more...")

    # 3. Extra buffer to capture trailing sounds
    time.sleep(post_buffer)

    # 4. Stop and return WAV bytes
    wav_bytes = recorder.stop()
    print(f"    [mic] Recording stopped. Captured {len(wav_bytes)} bytes.")
    return wav_bytes


def record_during_playback_bytes(
    stimulus_wav_bytes: bytes,
    post_buffer: float = config.POST_PLAYBACK_BUFFER,
) -> bytes:
    """
    与 record_during_playback 相同，但「刺激音」来自内存中的 WAV bytes
    （例如由 TTS 合成的问句）。
    """
    buf = io.BytesIO(stimulus_wav_bytes)
    try:
        data, sr = sf.read(buf, dtype="float32")
    except Exception as e:
        raise ValueError(f"无法从内存解析 WAV（需为 soundfile 可读格式）: {e}") from e

    recorder = AudioRecorder()

    recorder.start()
    print("    [mic] Recording started...")
    playback_numpy_sync(data, sr)
    print(f"    [mic] Stimulus playback done. Listening for {post_buffer}s more...")

    time.sleep(post_buffer)

    wav_bytes = recorder.stop()
    print(f"    [mic] Recording stopped. Captured {len(wav_bytes)} bytes.")
    return wav_bytes
