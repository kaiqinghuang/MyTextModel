#!/usr/bin/env python3
"""
Setup Slots - Generate 5 placeholder audio files for testing.

Uses gTTS (Google Text-to-Speech) to create simple test audio.
Replace these with your own pre-recorded audio files for production use.

Usage:
  python setup_slots.py              # Generate with gTTS
  python setup_slots.py --sine       # Generate simple sine wave beeps (no internet needed)
"""

import argparse
import os
import sys
import struct
import math
import wave

SLOTS_DIR = os.path.join(os.path.dirname(__file__), "audio_slots")

# 5 sample sentences for testing (English, calm tone content)
SAMPLE_SENTENCES = [
    "Good morning. I hope you slept well last night.",
    "The weather looks peaceful today, don't you think?",
    "I was thinking about that book we talked about before.",
    "Sometimes silence can be more meaningful than words.",
    "Let's take a moment to appreciate this quiet afternoon.",
]


def generate_with_gtts():
    """Generate test audio files using Google Text-to-Speech."""
    try:
        from gtts import gTTS
    except ImportError:
        print("[error] gTTS not installed. Run: pip install gTTS")
        print("        Or use --sine flag for offline generation.")
        sys.exit(1)

    os.makedirs(SLOTS_DIR, exist_ok=True)

    for i, sentence in enumerate(SAMPLE_SENTENCES, start=1):
        filename = f"slot_{i:02d}.mp3"
        filepath = os.path.join(SLOTS_DIR, filename)

        if os.path.exists(filepath):
            print(f"  [skip] {filename} already exists")
            continue

        print(f"  [generating] {filename}: \"{sentence}\"")
        tts = gTTS(text=sentence, lang="en", slow=False)
        tts.save(filepath)
        print(f"  [saved] {filepath}")

    print(f"\n[done] {len(SAMPLE_SENTENCES)} audio slots ready in: {SLOTS_DIR}")


def generate_sine_waves():
    """
    Generate simple sine-wave beep files (no internet needed).
    Each file has a different frequency to distinguish slots.
    A short text label is printed so you know which slot is playing.
    """
    os.makedirs(SLOTS_DIR, exist_ok=True)

    frequencies = [440, 523, 587, 659, 784]  # A4, C5, D5, E5, G5
    duration = 2.0  # seconds
    sample_rate = 16000
    amplitude = 16000

    for i, freq in enumerate(frequencies, start=1):
        filename = f"slot_{i:02d}.wav"
        filepath = os.path.join(SLOTS_DIR, filename)

        if os.path.exists(filepath):
            print(f"  [skip] {filename} already exists")
            continue

        print(f"  [generating] {filename}: {freq}Hz sine wave, {duration}s")

        num_samples = int(sample_rate * duration)
        with wave.open(filepath, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)

            for n in range(num_samples):
                # Apply fade in/out envelope to avoid clicks
                t = n / sample_rate
                envelope = 1.0
                fade = 0.1  # seconds
                if t < fade:
                    envelope = t / fade
                elif t > duration - fade:
                    envelope = (duration - t) / fade

                value = int(amplitude * envelope * math.sin(2 * math.pi * freq * t))
                wf.writeframes(struct.pack("<h", value))

        print(f"  [saved] {filepath}")

    print(f"\n[done] {len(frequencies)} sine-wave slots ready in: {SLOTS_DIR}")
    print("[note] These are simple beeps for testing. Replace with your recorded audio.")


def main():
    parser = argparse.ArgumentParser(description="Generate placeholder audio slots")
    parser.add_argument(
        "--sine",
        action="store_true",
        help="Generate sine wave beeps instead of TTS (offline, no API needed)",
    )

    args = parser.parse_args()

    print(f"\n[setup] Generating {len(SAMPLE_SENTENCES)} audio slots...\n")

    if args.sine:
        generate_sine_waves()
    else:
        generate_with_gtts()

    print("\nNext steps:")
    print("  1. Replace these files with your own pre-recorded audio")
    print("  2. Fill in API keys in config.py")
    print("  3. Run: python main.py")


if __name__ == "__main__":
    main()
