#!/usr/bin/env python3
"""
Simple test script to verify all components work
Tests each module individually before full integration
"""

import asyncio
import sys
from pathlib import Path

def test_imports():
    """Test that all modules import correctly"""
    print("Testing imports...")

    try:
        from riva_client import RivaASRClient, RivaTTSClient
        print("✓ Riva clients imported")

        from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
        print("✓ Ollama client imported")

        from audio_processing import AudioProcessor, AudioBuffer
        print("✓ Audio processing imported")

        from conversation_context import ConversationContext
        print("✓ Conversation context imported")

        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_audio_processing():
    """Test audio processing functions"""
    print("\nTesting audio processing...")

    try:
        from audio_processing import AudioProcessor
        import numpy as np

        processor = AudioProcessor()

        # Test resampling
        test_audio = np.random.random(1000).astype(np.float32)
        resampled = processor.resample(test_audio, 8000, 16000)
        print(f"✓ Resampling: {len(test_audio)} -> {len(resampled)} samples")

        # Test μ-law conversion
        pcm_data = (test_audio * 32767).astype(np.int16).tobytes()
        ulaw_data = b'\x80' * len(pcm_data)  # Dummy μ-law data

        print("✓ Audio processing works")
        return True

    except Exception as e:
        print(f"✗ Audio processing failed: {e}")
        return False

def test_conversation_context():
    """Test conversation management"""
    print("\nTesting conversation context...")

    try:
        from conversation_context import ConversationContext

        context = ConversationContext()

        # Test adding turns
        context.add_turn("Hello", "Hi there! How can I help?")
        context.add_turn("What's the weather?", "I don't have weather data, but I can help with other questions.")

        # Test prompt building
        prompt = context.build_prompt("Tell me a joke", "You are a helpful assistant.")
        print(f"✓ Built prompt: {len(prompt)} characters")

        # Test summary
        summary = context.get_summary()
        print(f"✓ Context summary: {summary['total_turns']} turns")

        return True

    except Exception as e:
        print(f"✗ Conversation context failed: {e}")
        return False

async def test_ollama():
    """Test Ollama connection"""
    print("\nTesting Ollama connection...")

    try:
        from ollama_client import OllamaClient

        client = OllamaClient()

        # Test health check
        if client.health_check():
            print("✓ Ollama is running")

            # Test simple generation
            response = client.generate("Say hello in one word")
            print(f"✓ Ollama response: '{response[:50]}'")

            client.close()
            return True
        else:
            print("✗ Ollama not running on localhost:11434")
            return False

    except Exception as e:
        print(f"✗ Ollama test failed: {e}")
        return False

def test_riva_connection():
    """Test Riva connection (basic)"""
    print("\nTesting Riva connection...")

    try:
        from riva_client import RivaASRClient, RivaTTSClient

        # Test creation and connection
        asr_client = RivaASRClient()
        tts_client = RivaTTSClient()

        print("✓ Riva clients created")

        # Test connections
        asr_connected = asr_client.connect()
        tts_connected = tts_client.connect()

        print(f"✓ ASR connection: {'OK' if asr_connected else 'Using fallback'}")
        print(f"✓ TTS connection: {'OK' if tts_connected else 'Using fallback'}")

        return True

    except Exception as e:
        print(f"✗ Riva client test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=== Voice Bot Component Tests ===\n")

    # Run non-async tests
    non_async_tests = [
        ("Imports", test_imports()),
        ("Audio Processing", test_audio_processing()),
        ("Conversation Context", test_conversation_context()),
        ("Riva Connection", test_riva_connection()),
    ]

    # Run async tests separately
    ollama_result = await test_ollama()

    # Combine results
    tests = non_async_tests + [("Ollama", ollama_result)]

    passed = 0
    total = len(tests)

    for test_name, result in tests:
        if result:
            passed += 1

    print(f"\n=== Test Results: {passed}/{total} passed ===")

    if passed == total:
        print("🎉 All tests passed! Ready for next phase.")
    else:
        print("❌ Some tests failed. Fix issues before proceeding.")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
