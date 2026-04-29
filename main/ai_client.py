"""
AI Client - OpenAI 纯文本问答 + Coqui XTTS-v2 本地克隆声。

Flow per turn:
  1. 将「问句文本」发往 OpenAI 聊天补全得到英文复述问句（不经过音频 API）。
  2. 将该英文文本送入 Coqui XTTS-v2 合成并扬声器播放。

Voice cloning approach:
  - On startup, the reference voice WAV is analyzed ONCE to extract speaker
    embeddings (gpt_cond_latent + speaker_embedding).
  - These cached embeddings are reused for every synthesis call.
  - No re-analysis of the reference audio on each sentence.
"""

import io
import os
import wave
import tempfile
import numpy as np
import torch

from openai import OpenAI

import config


def _patch_torch_load_for_coqui_tts() -> None:
    """PyTorch 2.6+ 将 torch.load 默认改为 weights_only=True；Coqui XTTS 的 checkpoint
    需反序列化含 XttsConfig 等类的 pickle，必须允许完整载入。仅对未显式传 weights_only 的调用补默认值。
    """
    _orig_load = torch.load

    def _load(*args, **kwargs):  # type: ignore[misc]
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load  # noqa: WPS442


_patch_torch_load_for_coqui_tts()


def _patch_torchaudio_load_via_soundfile() -> None:
    """
    torchaudio 2.9+ 默认走 TorchCodec → 依赖系统 FFmpeg；Homebrew/ffmpeg 不全时常报 libavutil 缺失。
    Coqui XTTS 对 reference 仅 torchaudio.load(本地.wav)，可用 soundfile 读 WAV，避免 FFmpeg/TorchCodec。
    """
    try:
        import soundfile as sf
    except ImportError:
        return

    try:
        import torchaudio
    except ImportError:
        return

    _orig_load = torchaudio.load

    def _load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None,
              buffer_size=4096, backend=None):
        path_str = os.fspath(uri) if isinstance(uri, (str, os.PathLike)) else None

        skip_codec = (
            path_str is not None
            and path_str.lower().endswith(".wav")
            and not path_str.startswith(("http://", "https://"))
        )

        if not skip_codec:
            return _orig_load(
                uri,
                frame_offset=frame_offset,
                num_frames=num_frames,
                normalize=normalize,
                channels_first=channels_first,
                format=format,
                buffer_size=buffer_size,
                backend=backend,
            )

        data, sr = sf.read(path_str, dtype="float32", always_2d=True)

        wav = torch.from_numpy(np.ascontiguousarray(data.T))

        if frame_offset != 0:
            wav = wav[:, frame_offset:]
        if num_frames is not None and num_frames >= 0:
            wav = wav[:, :num_frames]

        if not channels_first:
            wav = wav.transpose(0, 1)

        return wav, sr

    torchaudio.load = _load  # noqa: WPS442


_patch_torchaudio_load_via_soundfile()


# ============================================================
# OpenAI: 纯文本对话
# ============================================================

class ConversationAI:
    """
    使用普通 chat.completions（文本），不参与麦克风/音频 API。
    维护会话历史以防多轮需要上下文。
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.history: list[dict] = []  # conversation message history

    def send_question_text_get_reply(self, question_text: str) -> str:
        """把本轮问句正文发给 OpenAI，得到模型回复（通常为英文复述问句）。"""
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            *self._trimmed_history(),
            {"role": "user", "content": question_text},
        ]

        print("    [openai] Sending question text...")
        response = self.client.chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )

        assistant_text = response.choices[0].message.content or ""
        print(f"    [openai] Response: {assistant_text}")

        self.history.append({"role": "user", "content": question_text})
        self.history.append({"role": "assistant", "content": assistant_text})

        return assistant_text

    def _trimmed_history(self) -> list[dict]:
        """Return the last N turns of history to stay within context limits."""
        max_messages = config.MAX_HISTORY_TURNS * 2  # 2 messages per turn
        return self.history[-max_messages:]


# ============================================================
# Coqui XTTS-v2: Text -> Cloned Voice Audio (local)
# ============================================================

class VoiceCloneTTS:
    """
    Converts text to speech using Coqui XTTS-v2 with voice cloning.

    Voice cloning workflow:
      1. On __init__, load the XTTS-v2 model
      2. Analyze reference_voice.wav ONCE to extract speaker embeddings
      3. Cache the embeddings (gpt_cond_latent + speaker_embedding)
      4. Every synthesize() call reuses the cached embeddings directly
         -- no re-reading of reference audio, no re-computation

    First run will download the model (~1.8 GB). Subsequent runs use cache.
    """

    def __init__(self):
        self.model = None
        self._tts_model = None  # underlying XTTS model for direct inference
        self.reference_wav = self._resolve_reference_path()

        # Cached speaker embeddings (computed once, reused forever)
        self._gpt_cond_latent = None
        self._speaker_embedding = None

        self._load_model()
        self._cache_speaker_embeddings()

    def _resolve_reference_path(self) -> str:
        """Resolve the reference voice WAV path."""
        ref = config.COQUI_REFERENCE_WAV
        if os.path.isabs(ref):
            path = ref
        else:
            path = os.path.join(os.path.dirname(__file__), ref)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Reference voice file not found: {path}\n"
                f"Please record 6-15 seconds of clean speech and save as:\n"
                f"  {path}\n"
                f"Tip: Use 'python -c \"import sounddevice as sd; import soundfile as sf; "
                f"audio = sd.rec(int(10*16000), samplerate=16000, channels=1, dtype=\\\"int16\\\"); "
                f"sd.wait(); sf.write(\\\"reference_voice.wav\\\", audio, 16000)\"' "
                f"to record 10 seconds."
            )
        return path

    def _load_model(self):
        """Load the XTTS-v2 model. Downloads on first run."""
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError(
                "Coqui TTS not installed. Run:\n"
                "  pip install TTS\n"
                "Note: requires Python 3.9-3.11 and PyTorch."
            )

        device = config.COQUI_DEVICE

        # Validate device availability, fallback to CPU if needed
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                print("    [coqui] CUDA not available.")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                    print("    [coqui] Using Apple Metal (MPS) instead.")
                else:
                    device = "cpu"
                    print("    [coqui] Falling back to CPU.")
            elif device == "mps":
                if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    print("    [coqui] MPS not available, falling back to CPU.")
                    device = "cpu"
        except ImportError:
            device = "cpu"

        print(f"    [coqui] Loading XTTS-v2 model on {device}...")
        print(f"    [coqui] (First run downloads ~1.8 GB, please be patient)")
        self.model = TTS(model_name=config.COQUI_MODEL_NAME).to(device)

        # Get the underlying XTTS model for direct inference
        if self.model.synthesizer and self.model.synthesizer.tts_model:
            self._tts_model = self.model.synthesizer.tts_model

        print(f"    [coqui] Model loaded successfully.")

    def _cache_speaker_embeddings(self):
        """
        Analyze the reference voice ONCE and cache the speaker embeddings.
        This is the key optimization: these embeddings capture "how you sound"
        and are reused for every single synthesis call without re-computation.
        """
        if self._tts_model is None:
            print("    [coqui] Warning: cannot cache embeddings, will use high-level API.")
            return

        print(f"    [coqui] Analyzing reference voice: {os.path.basename(self.reference_wav)}")
        print(f"    [coqui] Extracting speaker embeddings (one-time operation)...")

        self._gpt_cond_latent, self._speaker_embedding = (
            self._tts_model.get_conditioning_latents(
                audio_path=[self.reference_wav]
            )
        )

        print(f"    [coqui] Speaker embeddings cached. Ready for synthesis.")

    def synthesize(self, text: str, language: str | None = None) -> bytes:
        """
        Synthesize text into speech using the cloned voice.
        Uses the high-level tts_to_file API which handles text splitting
        and inference internally for maximum stability.
        Returns WAV audio bytes.

        language: XTTS-v2 语言代码；默认使用 config.COQUI_LANGUAGE。
                  生成中文问句刺激音时可传入 config.QUESTION_TTS_LANGUAGE（如 zh-cn）。
        """
        print(f"    [coqui] Synthesizing: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        return self._synthesize_stable(text, language=language)

    def _synthesize_stable(self, text: str, language: str | None = None) -> bytes:
        """
        Stable path: uses high-level TTS API which handles text splitting
        and full_inference internally. Re-reads reference audio each call
        but is the most reliable method on CPU.
        """

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            lang = language if language is not None else config.COQUI_LANGUAGE
            self.model.tts_to_file(
                text=text,
                speaker_wav=self.reference_wav,
                language=lang,
                file_path=tmp_path,
            )

            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()

            print(f"    [coqui] Synthesized {len(wav_bytes)} bytes.")
            return wav_bytes

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _numpy_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
        """Convert a float32 numpy audio array to WAV bytes."""
        if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = (audio_array * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_array.tobytes())
        buf.seek(0)
        return buf.read()
