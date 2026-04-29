#!/usr/bin/env python3
"""
Voice Conversation System — Main Orchestrator

每轮：
  1. 本地小 GPT 生成 1 句中文问句 → XTTS 经扬声器播出
  2. **同一问句正文**直连 OpenAI 文本 API（不经麦克风、「听筒」音频模型）
  3. 将模型返回的文本再 XTTS → 扬声器播放

Usage:
  python main.py                 # 跑 config.NUM_CONVERSATION_TURNS 轮（默认 5）
  python main.py --loop          # 无限循环轮次
  python main.py --turns 10       # 共 10 轮
  python main.py --skip 3        # 跳过前 2 轮，从第 3 轮开始（与旧版 --slot 3 对齐）
  python main.py --test-mic      # 测麦克风
  python main.py --test-tts      # 测 Coqui TTS
"""

import argparse
import sys
import time

import config
from ai_client import ConversationAI, VoiceCloneTTS
from audio_manager import AudioRecorder, AudioPlayer


def run_conversation(
    *,
    loop: bool = False,
    num_turns: int | None = None,
    skip_first: int = 0,
) -> None:
    """
    skip_first: 跳过的完整轮次数（在进入循环前快进计数器）。
    """
    n_total = num_turns if num_turns is not None else config.NUM_CONVERSATION_TURNS

    print(f"\n{'='*60}")
    print(f"  Voice Conversation System")
    print(f"  每轮：小模型问句 → 扬声器播报 → OpenAI（文本）→ 扬声器播报回复")
    print(f"  本轮设置：{'无限循环' if loop else f'{n_total} 轮'}｜跳过前 {skip_first} 轮")
    print(f"{'='*60}\n")

    print("  [init] 加载问句生成器（GPT + patterns）...")
    from question_generator import QuestionStructureGenerator  # noqa: E402

    try:
        qgen = QuestionStructureGenerator()
    except Exception as e:
        print(f"  [error] 加载问句生成器失败: {e}")
        sys.exit(1)

    ai = ConversationAI()
    tts = VoiceCloneTTS()
    player = AudioPlayer()

    # 下一轮将要执行的序号（从 1 起计）；若 --skip 2 则从第 3 轮开始
    next_round = skip_first + 1

    while True:
        if not loop and next_round > n_total:
            print("\n[done] 已全部完成。会话结束。\n")
            break

        turn_num = next_round

        print(f"\n{'- '*30}")
        print(f"  TURN {turn_num}{' (∞)' if loop else f'/{n_total}' if n_total else ''}")
        print(f"{'- '*30}")

        # ---------- Step 1: 生成问句 → TTS → 扬声器（仅回放，不向 API 上传录音） ----------
        print(f"\n  [step 1] 生成本地问句并经扬声器播报...")
        try:
            question_cn = qgen.generate_sentence()
        except Exception as e:
            print(f"  [error] 问句生成失败: {e}")
            continue

        print(f"  [local model] 「{question_cn}」")

        try:
            stimulus_wav = tts.synthesize(
                question_cn,
                language=config.QUESTION_TTS_LANGUAGE,
            )
        except Exception as e:
            print(f"  [error] 问句 TTS 失败: {e}")
            continue

        try:
            player.play_bytes(stimulus_wav, blocking=True)
        except Exception as e:
            print(f"  [error] 问句播放失败: {e}")
            continue

        # ---------- Step 2: 同一问句文本 → OpenAI（纯文本，非音频预览模型） ----------
        print(f"\n  [step 2] Sending question text to OpenAI...")
        try:
            reply_text = ai.send_question_text_get_reply(question_cn)
        except Exception as e:
            print(f"  [error] OpenAI API failed: {e}")
            continue

        if not reply_text or not reply_text.strip():
            print("  [warn] AI returned empty response. Continuing.")
            continue

        # ---------- Step 3–4：英文回复克隆声 ----------
        print(f"\n  [step 3] Synthesizing cloned voice response (English)...")
        try:
            reply_audio = tts.synthesize(reply_text)
        except Exception as e:
            print(f"  [error] Coqui TTS failed: {e}")
            print(f"          AI said: \"{reply_text}\"")
            continue

        print(f"\n  [step 4] Playing AI response...")
        try:
            player.play_bytes(reply_audio, blocking=True)
        except Exception as e:
            print(f"  [error] Playback failed: {e}")

        time.sleep(0.5)

        next_round += 1

    print("\n[exit] Session ended. Goodbye.\n")


def test_microphone() -> None:
    """Quick test: record 3 seconds from mic and play it back."""
    print("\n[test] Microphone test - recording 3 seconds...")
    recorder = AudioRecorder()
    player = AudioPlayer()

    recorder.start()
    time.sleep(3)
    wav_bytes = recorder.stop()

    print(f"[test] Captured {len(wav_bytes)} bytes. Playing back...")
    player.play_bytes(wav_bytes, sample_rate=config.RECORD_SAMPLE_RATE, blocking=True)
    print("[test] Done.")


def test_tts() -> None:
    """Quick test: synthesize a short sentence with Coqui XTTS-v2."""
    print("\n[test] Coqui XTTS-v2 TTS test...")
    print(f"[test] Reference voice: {config.COQUI_REFERENCE_WAV}")
    tts = VoiceCloneTTS()
    player = AudioPlayer()

    text = "The morning mist settles gently over the quiet hills."
    print(f"[test] Synthesizing: \"{text}\"")

    audio = tts.synthesize(text)
    print(f"[test] Got {len(audio)} bytes. Playing...")
    player.play_bytes(audio, sample_rate=config.PLAYBACK_SAMPLE_RATE, blocking=True)
    print("[test] Sentence 1 done.")

    text2 = "Sometimes the quietest moments hold the deepest meaning."
    print(f"\n[test] Synthesizing second sentence (cached voice)...")
    audio2 = tts.synthesize(text2)
    player.play_bytes(audio2, sample_rate=config.PLAYBACK_SAMPLE_RATE, blocking=True)
    print("[test] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice conversation — local GPT + TTS playback + OpenAI text + TTS playback"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="无限循环会话轮次（不再受 --turns 限制）",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help=f"总轮次数（不与 --loop 并用；默认 config.NUM_CONVERSATION_TURNS={config.NUM_CONVERSATION_TURNS}）",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="在开始主循环前跳过前 N 轮（用于断点接续；例如 --skip 2 从第 3 轮等价于旧版 --slot 3）",
    )
    parser.add_argument(
        "--test-mic",
        action="store_true",
        help="Test microphone recording and playback",
    )
    parser.add_argument(
        "--test-tts",
        action="store_true",
        help="Test Coqui XTTS-v2 voice clone synthesis",
    )

    args = parser.parse_args()

    if args.test_mic:
        test_microphone()
        return

    if args.test_tts:
        test_tts()
        return

    turns = args.turns if args.turns is not None else config.NUM_CONVERSATION_TURNS
    if args.loop:
        turns = None
    skip = max(0, args.skip)

    run_conversation(loop=args.loop, num_turns=turns, skip_first=skip)


if __name__ == "__main__":
    main()
