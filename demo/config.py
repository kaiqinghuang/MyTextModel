"""
Configuration for the Voice Conversation System.
Fill in your API keys and adjust settings before running.
"""

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# API Keys  (replace with your own)
# ============================================================

# ============================================================
# Coqui TTS (XTTS-v2) - Local Voice Clone Settings
# ============================================================
# XTTS-v2 clones your voice from a short reference audio file.
# Record 6-15 seconds of clean speech and save as WAV.
# Place the file in the project root or provide an absolute path.
COQUI_REFERENCE_WAV = "reference_voice.wav"

# XTTS-v2 model name (downloaded automatically on first run)
COQUI_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Language for synthesis
COQUI_LANGUAGE = "en"

# Device for Coqui XTTS-v2
# NVIDIA GPU: "cuda"  |  Mac Apple Silicon: "mps"  |  CPU only: "cpu"
# Note: XTTS-v2 has known MPS compatibility issues on Mac, use "cpu" for stability.
# M2 Max CPU is fast enough for this workload.
COQUI_DEVICE = "cpu"

# Temperature for TTS generation (lower = more stable, higher = more expressive)
# XTTS-v2 default is 0.75. Adjust if voice sounds unstable or robotic.
COQUI_TEMPERATURE = 0.75

# Repetition penalty (XTTS-v2 default is 1.0, raise to 1.5-2.0 if audio loops)
COQUI_REPETITION_PENALTY = 1.0

# Top-k and top-p sampling for voice generation (XTTS-v2 defaults)
COQUI_TOP_K = 50
COQUI_TOP_P = 0.85

# ============================================================
# OpenAI Model Settings
# ============================================================
OPENAI_MODEL = "gpt-4o-audio-preview"

# System prompt: serene, calm, peaceful conversational style
SYSTEM_PROMPT = """You are a calm, serene presence in a conversation. Your responses embody:

- Tranquility: You speak as if sitting beside a still lake at dawn. Never rushed, never anxious.
- Gentleness: Every word is chosen with care and softness. You do not raise your voice in tone or urgency.
- Equanimity: Nothing disturbs your inner peace. Whether the topic is joyful or heavy, you respond with the same quiet composure.
- Warmth without excess: You are kind, but not effusive. Your warmth is like sunlight filtered through clouds.
- Brevity with depth: You prefer fewer words that carry meaning over long explanations. Silence is comfortable for you.

Style guidelines:
- Keep responses concise (1 to 3 sentences typically, unless the topic truly calls for more).
- Use a measured, unhurried pace. Imagine each sentence has a breath between them.
- Avoid exclamation marks, ALL CAPS, or any markers of excitement or alarm.
- When you hear ambient sounds or multiple voices, simply acknowledge what you perceive with calm curiosity.
- You may gently reflect on what you hear, offer a quiet observation, or ask a soft question.
- Respond in English.

You are having a live voice conversation. You will hear audio that may include a primary speaker and sometimes background voices or ambient sounds. Treat everything you hear as part of the natural environment of this conversation."""

# ============================================================
# Audio Settings
# ============================================================
# Microphone recording sample rate
RECORD_SAMPLE_RATE = 16000

# Channels for recording (1 = mono, sufficient for voice)
RECORD_CHANNELS = 1

# Seconds to keep recording AFTER the pre-recorded audio finishes playing.
# Set to 0 to stop recording immediately when playback ends.
POST_PLAYBACK_BUFFER = 0.0

# Playback sample rate for Coqui XTTS-v2 output (model outputs 24000 Hz)
PLAYBACK_SAMPLE_RATE = 24000

# ============================================================
# Audio Slots
# ============================================================
AUDIO_SLOTS_DIR = "audio_slots"
NUM_SLOTS = 5

# ============================================================
# Conversation Settings
# ============================================================
# Max conversation history turns to keep in context (each turn = user + assistant)
MAX_HISTORY_TURNS = 10
