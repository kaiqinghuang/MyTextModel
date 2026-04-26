"""
Audio Manager - handles microphone recording and audio playback.

Key design:
  - Records from microphone while pre-recorded audio plays through speakers.
  - The microphone captures EVERYTHING: the played audio + audience / ambient sounds.
  - After playback ends, recording continues for a configurable buffer period
    to catch trailing speech or reactions.
"""

import io
import time
import threading
import wave
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

        sd.play(data, samplerate=sample_rate)
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
