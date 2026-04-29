"""
Configuration for the Voice Conversation System.
Fill in your API keys and adjust settings before running.
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # 未安装 python-dotenv 时跳过；请用 pip install python-dotenv 或自行 export OPENAI_API_KEY
    pass
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

# Language for synthesis (OpenAI 回复朗读，一般为英文）
COQUI_LANGUAGE = "en"

# 本地小模型生成问句的中文 TTS（XTTS-v2 multilingual 语言码）
QUESTION_TTS_LANGUAGE = "zh-cn"

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
# OpenAI Model Settings（纯文本对话，不使用音频预览 API）
# ============================================================
OPENAI_CHAT_MODEL = "gpt-4o"

# System prompt：把用户消息里的问句用你的理解再问一遍（给后续英文 TTS 朗读）
SYSTEM_PROMPT = """根据你的理解重新问出这个问题。

「用户」消息里会直接给出一句问句文本（常为中文）。请理解这句话真正在问什么，再用你自己的措辞把同一意思再问一遍；必须是问句。

只输出你重新问出的这一句，不要前缀或解释。

为与下游语音合成一致，请用英文写出这句问句；若难以理解则简短用英文说明。"""

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
# 会话轮数（不再使用预制 audio_slots：每轮由小模型生成 1 句问句并已 TTS 播放）
# ============================================================
NUM_CONVERSATION_TURNS = 5

# ============================================================
# Conversation Settings
# ============================================================
# Max conversation history turns to keep in context (each turn = user + assistant)
MAX_HISTORY_TURNS = 10
