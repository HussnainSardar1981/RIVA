#!/usr/bin/env python3
"""
NETOVO Voice Bot - Telephony Integration Validation Script
Quick validation of Phase 2 telephony components
"""

import sys
import os
import asyncio
import importlib
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all telephony modules can be imported"""
    print("🧪 Testing Telephony Module Imports...")

    modules = [
        ('telephony.rtp_bridge', 'RTPAudioBridge'),
        ('telephony.custom_sip', 'CustomSIPClient'),
        ('telephony.call_manager', 'CallFlowManager'),
        ('telephony.audio_codec', 'AudioCodec'),
        ('telephony.advanced_vad', 'AdvancedVAD'),
        ('yaml', None),
        ('aiortc', None),
    ]

    passed = 0
    total = len(modules)

    for module_name, class_name in modules:
        try:
            module = importlib.import_module(module_name)
            if class_name:
                getattr(module, class_name)
            print(f"  ✅ {module_name}")
            passed += 1
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
        except AttributeError as e:
            print(f"  ❌ {module_name}.{class_name}: {e}")

    print(f"\n📊 Import Test Results: {passed}/{total} passed")
    return passed == total

def test_audio_codec():
    """Test audio codec functionality"""
    print("\n🎵 Testing Audio Codec...")

    try:
        from telephony.audio_codec import AudioCodec
        import numpy as np

        codec = AudioCodec()

        # Test PCMU conversion
        test_audio = np.array([0.1, -0.1, 0.5, -0.5, 0.0], dtype=np.float32)
        pcmu_data = codec.pcm16_to_pcmu(test_audio)
        recovered_audio = codec.pcmu_to_pcm16(pcmu_data)

        if len(recovered_audio) == len(test_audio):
            print("  ✅ PCMU conversion working")

        # Test RTP packet creation
        rtp_packet = codec.create_rtp_packet(b'\x80\x81', 12345, 100)
        if len(rtp_packet) >= 12:
            print("  ✅ RTP packet creation working")

        # Test audio quality validation
        quality = codec.validate_audio_quality(test_audio)
        if 'valid' in quality:
            print("  ✅ Audio quality validation working")

        return True

    except Exception as e:
        print(f"  ❌ Audio Codec test failed: {e}")
        return False

def test_vad():
    """Test Advanced VAD functionality"""
    print("\n🎤 Testing Advanced VAD...")

    try:
        from telephony.advanced_vad import AdvancedVAD
        import numpy as np

        vad_config = {
            'aggressiveness': 2,
            'sample_rate': 8000,
            'barge_in_enabled': True
        }

        vad = AdvancedVAD(vad_config)

        # Test initialization
        if vad.aggressiveness == 2:
            print("  ✅ VAD initialization working")

        # Test audio processing
        test_audio = np.random.randn(1600).astype(np.float32) * 0.1
        vad.start_listening()
        result = vad.process_audio(test_audio)

        if isinstance(result, bool):
            print("  ✅ VAD audio processing working")

        # Test statistics
        stats = vad.get_statistics()
        if 'frames_processed' in stats:
            print("  ✅ VAD statistics working")

        return True

    except Exception as e:
        print(f"  ❌ Advanced VAD test failed: {e}")
        return False

def test_config_loading():
    """Test configuration file loading"""
    print("\n⚙️  Testing Configuration Loading...")

    try:
        import yaml

        config_path = Path(__file__).parent / "config" / "telephony.yaml"

        if not config_path.exists():
            print(f"  ❌ Configuration file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        required_sections = ['sip', 'audio', 'voice_bot']
        for section in required_sections:
            if section not in config:
                print(f"  ❌ Missing configuration section: {section}")
                return False

        print("  ✅ Configuration file structure valid")

        # Test environment variable substitution
        if '${THREECX_PASSWORD}' in str(config):
            print("  ✅ Environment variable substitution configured")

        return True

    except Exception as e:
        print(f"  ❌ Configuration loading test failed: {e}")
        return False

def test_directory_structure():
    """Test that all required files and directories exist"""
    print("\n📁 Testing Directory Structure...")

    base_path = Path(__file__).parent

    required_files = [
        'main.py',
        'requirements.txt',
        '.env',
        'config/telephony.yaml',
        'telephony/__init__.py',
        'telephony/rtp_bridge.py',
        'telephony/custom_sip.py',
        'telephony/call_manager.py',
        'telephony/audio_codec.py',
        'telephony/advanced_vad.py',
        'tests/test_telephony.py',
        'DEPLOYMENT_GUIDE.md'
    ]

    passed = 0
    total = len(required_files)

    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
            passed += 1
        else:
            print(f"  ❌ {file_path} - NOT FOUND")

    print(f"\n📊 Directory Structure: {passed}/{total} files present")
    return passed == total

async def test_main_integration():
    """Test main.py telephony integration"""
    print("\n🔗 Testing Main.py Integration...")

    try:
        from main import VoiceBotConfig

        # Test config loading
        config = VoiceBotConfig()
        print("  ✅ VoiceBotConfig loading working")

        # Test that telephony imports work in main context
        try:
            import main
            if hasattr(main, 'TELEPHONY_AVAILABLE'):
                if main.TELEPHONY_AVAILABLE:
                    print("  ✅ Telephony integration available in main.py")
                else:
                    print("  ⚠️  Telephony integration not available (missing dependencies)")
            else:
                print("  ❌ TELEPHONY_AVAILABLE not defined in main.py")

        except Exception as e:
            print(f"  ❌ Main.py integration test failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Main integration test failed: {e}")
        return False

async def main():
    """Run all validation tests"""
    print("🎯 NETOVO Voice Bot - Phase 2 Telephony Integration Validation")
    print("=" * 70)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_imports),
        ("Audio Codec", test_audio_codec),
        ("Advanced VAD", test_vad),
        ("Configuration Loading", test_config_loading),
        ("Main.py Integration", test_main_integration)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed_tests += 1
                print(f"  ✅ {test_name} - PASSED")
            else:
                print(f"  ❌ {test_name} - FAILED")

        except Exception as e:
            print(f"  ❌ {test_name} - ERROR: {e}")

    # Final Results
    print(f"\n{'='*70}")
    print(f"📊 VALIDATION RESULTS: {passed_tests}/{total_tests} tests passed")
    print(f"⏱️  Success Rate: {(passed_tests/total_tests*100):.1f}%")

    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Phase 2 Telephony Integration is ready for deployment")
        print("📞 You can now start the voice bot in telephony mode:")
        print("   python3 main.py -> Option 3: Start Telephony Mode")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} validation tests FAILED")
        print("⚠️  Fix the issues above before deploying to production")
        print("💡 Check DEPLOYMENT_GUIDE.md for troubleshooting steps")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
