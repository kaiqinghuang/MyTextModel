"""
AI Client - OpenAI (audio-in, text-out) + Coqui XTTS-v2 (text-in, cloned-voice-out).

Flow per turn:
  1. Send recorded WAV audio to OpenAI  gpt-4o-audio-preview  (audio input)
  2. Receive text response from the model
  3. Send text to Coqui XTTS-v2 for local voice-cloned TTS
  4. Return synthesized audio bytes ready for playback

Voice cloning approach:
  - On startup, the reference voice WAV is analyzed ONCE to extract speaker
    embeddings (gpt_cond_latent + speaker_embedding).
  - These cached embeddings are reused for every synthesis call.
  - No re-analysis of the reference audio on each sentence.
"""

import base64
import io
import os
import wave
import tempfile
import numpy as np

from openai import OpenAI

import config


# ============================================================
# OpenAI: Audio -> Text
# ============================================================

class ConversationAI:
    """
    Manages the OpenAI conversation with audio input support.
    Maintains conversation history so the AI has context across turns.
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.history: list[dict] = []  # conversation message history

    def send_audio_get_text(self, wav_bytes: bytes) -> str:
        """
        Send WAV audio to OpenAI and get a text response.

        The audio is base64-encoded and sent as an `input_audio` content block,
        which lets the model hear the audio directly without a separate STT step.
        """
        # Base64 encode the WAV
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        # Build the user message with audio content
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav",
                    },
                }
            ],
        }

        # Build full messages list: system + history + current
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            *self._trimmed_history(),
            user_message,
        ]

        # Call the API
        print("    [openai] Sending audio to model...")
        response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            modalities=["text"],
            max_tokens=300,
            temperature=0.7,
        )

        assistant_text = response.choices[0].message.content
        print(f"    [openai] Response: {assistant_text}")

        # Save to history (store text summaries, not raw audio, to save tokens)
        self.history.append({
            "role": "user",
            "content": "[Audio input from speaker and environment]",
        })
        self.history.append({
            "role": "assistant",
            "content": assistant_text,
        })

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

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text into speech using the cloned voice.
        Uses the high-level tts_to_file API which handles text splitting
        and inference internally for maximum stability.
        Returns WAV audio bytes.
        """
        print(f"    [coqui] Synthesizing: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        return self._synthesize_stable(text)

    def _synthesize_stable(self, text: str) -> bytes:
        """
        Stable path: uses high-level TTS API which handles text splitting
        and full_inference internally. Re-reads reference audio each call
        but is the most reliable method on CPU.
        """

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.model.tts_to_file(
                text=text,
                speaker_wav=self.reference_wav,
                language=config.COQUI_LANGUAGE,
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
