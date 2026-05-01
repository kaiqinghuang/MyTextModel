"""
Configuration for the Voice Conversation System.
Fill in your API keys and adjust settings before running.
"""

import os
from pathlib import Path

# 无论从仓库根目录还是 main/ 目录运行，都能找到项目根目录的 .env
_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _ROOT / ".env"

try:
    from dotenv import load_dotenv

    load_dotenv(_ENV_FILE)
    load_dotenv()  # 若当前工作目录另有 .env，可覆盖补充
except ImportError:
    # 未安装 python-dotenv 时跳过；请用 pip install python-dotenv 或自行 export OPENAI_API_KEY
    pass
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# API Keys  (replace with your own)
# ============================================================

# ============================================================
# TTS backend（默认云端 ElevenLabs；本地 Coqui 仅在显式开启时加载）
# ============================================================
TTS_BACKEND = "elevenlabs"  # "elevenlabs" | "coqui"

# ElevenLabs：在控制台克隆音色后复制 Voice ID；API Key 可在账户页面生成。
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()

# 常用模型：eleven_multilingual_v2（中英等多语）；eleven_turbo_v2_5（更快）。
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
ELEVENLABS_TIMEOUT_SEC = 120
ELEVENLABS_STABILITY = 0.5
ELEVENLABS_SIMILARITY_BOOST = 0.75
ELEVENLABS_STYLE = 0.0
ELEVENLABS_USE_SPEAKER_BOOST = True

# ============================================================
# Coqui TTS (XTTS-v2) - Local Voice Clone Settings
# ============================================================
# XTTS-v2 clones your voice from a short reference audio file.
# Record 6-15 seconds of clean speech and save as WAV.
# Place the file in the project root or provide an absolute path.
COQUI_REFERENCE_WAV = "susie_reference_voice.wav"

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

# 将英文重问再翻回中文，作为下一轮小模型 seed（不直接播报）
SEED_TRANSLATE_PROMPT = """你是一个精确改写翻译器。把用户给出的一句英文问句翻译成自然、简洁、语义等价的中文问句。

只输出中文问句本身，不要解释，不要加引号。"""

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

# 回放后端（杂音仍严重时见下方说明）。
# — "auto"：macOS 且存在 afplay 时走系统自带的 afplay（与 Finder/Music 同一路径，一般最干净）；
#   否则用 sounddevice（PortAudio）。
# — "afplay"：强制用 afplay（仅 macOS 有效）。
# — "sounddevice"：仅用 sounddevice。
#
# 「两个音箱」：在 macOS「音频 MIDI 设置」里建「聚集设备」或多输出装置，再在
# 「系统设置 → 声音」里把默认输出设为该设备；本项目只走默认输出，不负责编组多个物理端点。
PLAYBACK_BACKEND = "auto"

# PLAYBACK_BACKEND 为 sounddevice 时生效：加大输出缓冲减轻爆音。「high」仍可试；
# afplay 路径下该项无效。
PLAYBACK_LATENCY = "high"

# PLAYBACK_BACKEND 为 sounddevice 时可选：固定输出设备序号（sounddevice.query_devices()）。
# None = 默认设备。多数双音箱请在系统里做聚集设备，无需填此项。
PLAYBACK_DEVICE = None

# Arduino pump trigger settings (optional).
# Set ARDUINO_SERIAL_PORT to your board port, e.g. "/dev/tty.usbmodem1101".
# Empty/None disables pump trigger without affecting conversation flow.
ARDUINO_SERIAL_PORT = "/dev/cu.usbmodem141301"
ARDUINO_BAUDRATE = 9600

# ============================================================
# 会话轮数（不再使用预制 audio_slots：每轮由小模型生成 1 句问句并已 TTS 播放）
# ============================================================
NUM_CONVERSATION_TURNS = 5

# ============================================================
# Conversation Settings
# ============================================================
# Max conversation history turns to keep in context (each turn = user + assistant)
MAX_HISTORY_TURNS = 10
