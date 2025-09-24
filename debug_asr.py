#!/usr/bin/env python3
"""
Debug ASR transcription to see what's happening
"""

from riva_proto_simple import SimpleRivaASR, SimpleRivaTTS
import os

def test_tts_asr_loop():
    print("=== TTS → ASR Debug Test ===")

    # Test text
    test_text = "Hello this is a test"
    print(f"Input text: '{test_text}'")

    # Step 1: Generate TTS
    print("\n1. Testing TTS...")
    tts = SimpleRivaTTS()
    wav_path = tts.synthesize(test_text)

    if wav_path and os.path.exists(wav_path):
        file_size = os.path.getsize(wav_path)
        print(f"✓ TTS generated: {wav_path} ({file_size} bytes)")
    else:
        print("✗ TTS failed")
        return

    # Step 2: Test ASR
    print("\n2. Testing ASR...")
    asr = SimpleRivaASR()

    # Test with automatic punctuation OFF (like in tests)
    transcript1 = asr.transcribe_file(wav_path, automatic_punctuation=False)
    print(f"ASR (no punct): '{transcript1}' (length: {len(transcript1)})")

    # Test with automatic punctuation ON
    transcript2 = asr.transcribe_file(wav_path, automatic_punctuation=True)
    print(f"ASR (with punct): '{transcript2}' (length: {len(transcript2)})")

    # Step 3: Test similarity
    print("\n3. Testing similarity...")
    from audio_processing import calculate_text_similarity

    if transcript1:
        sim1 = calculate_text_similarity(test_text, transcript1)
        print(f"Similarity (no punct): {sim1:.2%}")

    if transcript2:
        sim2 = calculate_text_similarity(test_text, transcript2)
        print(f"Similarity (with punct): {sim2:.2%}")

    # Cleanup
    try:
        os.unlink(wav_path)
        print(f"\n✓ Cleaned up {wav_path}")
    except:
        pass

if __name__ == "__main__":
    test_tts_asr_loop()
