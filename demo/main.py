#!/usr/bin/env python3
"""
Voice Conversation System - Main Orchestrator

Turn-based flow:
  1. Play pre-recorded audio slot N through speakers
  2. Microphone records everything (played audio + audience + ambient)
  3. Recorded audio is sent directly to OpenAI (no text transcription step)
  4. OpenAI responds with text
  5. Text is sent to Coqui XTTS-v2 for local voice-cloned TTS
  6. Cloned voice response is played through speakers
  7. Repeat with slot N+1

Usage:
  python main.py              # Run all 5 slots in sequence
  python main.py --slot 2     # Start from slot 2
  python main.py --loop       # Loop all slots continuously
  python main.py --test-mic   # Test microphone setup
  python main.py --test-tts   # Test Coqui XTTS-v2 TTS
"""

import argparse
import os
import sys
import time
import glob

import config
from audio_manager import AudioRecorder, AudioPlayer, record_during_playback
from ai_client import ConversationAI, VoiceCloneTTS


def get_audio_slots() -> list[str]:
    """
    Discover audio files in the slots directory, sorted by filename.
    Supports: .wav, .mp3, .flac, .ogg
    """
    slots_dir = os.path.join(os.path.dirname(__file__), config.AUDIO_SLOTS_DIR)

    if not os.path.isdir(slots_dir):
        print(f"[error] Audio slots directory not found: {slots_dir}")
        print(f"        Run 'python setup_slots.py' to create placeholder audio files.")
        sys.exit(1)

    extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(slots_dir, ext)))

    files.sort()

    if not files:
        print(f"[error] No audio files found in: {slots_dir}")
        print(f"        Place your pre-recorded audio files there (slot_01.wav, slot_02.wav, ...)")
        print(f"        Or run 'python setup_slots.py' to create test files.")
        sys.exit(1)

    return files


def run_conversation(start_slot: int = 0, loop: bool = False):
    """
    Main conversation loop.
    """
    # Discover audio slots
    slots = get_audio_slots()
    total_slots = len(slots)
    print(f"\n{'='*60}")
    print(f"  Voice Conversation System")
    print(f"  Audio slots found: {total_slots}")
    for i, s in enumerate(slots):
        print(f"    [{i+1}] {os.path.basename(s)}")
    print(f"{'='*60}\n")

    # Initialize AI clients
    ai = ConversationAI()
    tts = VoiceCloneTTS()
    player = AudioPlayer()

    current = start_slot

    while True:
        if current >= total_slots:
            if loop:
                print("\n[loop] Restarting from slot 1...\n")
                current = 0
            else:
                print("\n[done] All audio slots have been played. Conversation complete.")
                break

        slot_path = slots[current]
        slot_name = os.path.basename(slot_path)
        turn_num = current + 1

        print(f"\n{'- '*30}")
        print(f"  TURN {turn_num}/{total_slots} : {slot_name}")
        print(f"{'- '*30}")

        # ----------------------------------------------------------
        # Step 1: Play pre-recorded audio + record from microphone
        # ----------------------------------------------------------
        print(f"\n  [step 1] Playing '{slot_name}' and recording environment...")
        recorded_wav = record_during_playback(
            slot_path,
            post_buffer=config.POST_PLAYBACK_BUFFER,
        )

        if not recorded_wav:
            print("  [warn] No audio captured. Skipping this turn.")
            current += 1
            continue

        # ----------------------------------------------------------
        # Step 2: Send recorded audio to OpenAI (audio-in, text-out)
        # ----------------------------------------------------------
        print(f"\n  [step 2] Sending audio to AI model...")
        try:
            reply_text = ai.send_audio_get_text(recorded_wav)
        except Exception as e:
            print(f"  [error] OpenAI API failed: {e}")
            print(f"          Skipping this turn.")
            current += 1
            continue

        if not reply_text or not reply_text.strip():
            print("  [warn] AI returned empty response. Moving to next slot.")
            current += 1
            continue

        # ----------------------------------------------------------
        # Step 3: Synthesize AI response with cloned voice (Coqui XTTS-v2)
        # ----------------------------------------------------------
        print(f"\n  [step 3] Synthesizing cloned voice response...")
        try:
            reply_audio = tts.synthesize(reply_text)
        except Exception as e:
            print(f"  [error] Coqui TTS failed: {e}")
            print(f"          AI said: \"{reply_text}\"")
            print(f"          Skipping TTS for this turn.")
            current += 1
            continue

        # ----------------------------------------------------------
        # Step 4: Play the AI's voice-cloned response
        # ----------------------------------------------------------
        print(f"\n  [step 4] Playing AI response...")
        try:
            player.play_bytes(reply_audio, blocking=True)
        except Exception as e:
            print(f"  [error] Playback failed: {e}")

        # Small pause between turns for natural pacing
        time.sleep(0.5)

        current += 1

    print("\n[exit] Session ended. Goodbye.\n")


def test_microphone():
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


def test_tts():
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

    # Second sentence reuses cached embeddings (no re-analysis of reference)
    text2 = "Sometimes the quietest moments hold the deepest meaning."
    print(f"\n[test] Synthesizing second sentence (using cached voice)...")
    audio2 = tts.synthesize(text2)
    print(f"[test] Got {len(audio2)} bytes. Playing...")
    player.play_bytes(audio2, sample_rate=config.PLAYBACK_SAMPLE_RATE, blocking=True)
    print("[test] Sentence 2 done. Both used the same cached speaker embeddings.")


def main():
    parser = argparse.ArgumentParser(
        description="Voice Conversation System - Pre-recorded audio meets AI voice clone"
    )
    parser.add_argument(
        "--slot",
        type=int,
        default=1,
        help="Start from this slot number (1-based, default: 1)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the conversation continuously",
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

    # Start from the specified slot (convert 1-based to 0-based)
    start = max(0, args.slot - 1)
    run_conversation(start_slot=start, loop=args.loop)


if __name__ == "__main__":
    main()
