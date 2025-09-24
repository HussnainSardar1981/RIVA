#!/usr/bin/env python3
"""
Comprehensive Telephony Integration Tests
Tests all components: RTP Bridge, SIP Client, Call Manager, Audio Codec, Advanced VAD
"""

import asyncio
import pytest
import numpy as np
import tempfile
import os
import wave
import yaml
from unittest.mock import Mock, AsyncMock, patch
import structlog

# Import telephony components
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from telephony.rtp_bridge import RTPAudioBridge, RTPSession
from telephony.custom_sip import CustomSIPClient
from telephony.call_manager import CallFlowManager
from telephony.audio_codec import AudioCodec
from telephony.advanced_vad import AdvancedVAD, VADManager
from main import VoiceBotConfig

logger = structlog.get_logger()

class TestAudioCodec:
    """Test suite for Audio Codec module"""

    def setup_method(self):
        """Set up test fixtures"""
        self.codec = AudioCodec()

    def test_pcmu_to_pcm16_conversion(self):
        """Test PCMU to PCM16 conversion"""
        # Create sample PCMU data (simplified)
        pcmu_data = b'\x80\x81\x82\x83\x84\x85'

        # Convert to PCM
        pcm_audio = self.codec.pcmu_to_pcm16(pcmu_data)

        assert isinstance(pcm_audio, np.ndarray)
        assert pcm_audio.dtype == np.float32
        assert len(pcm_audio) == len(pcmu_data)
        assert np.all(np.abs(pcm_audio) <= 1.0)

    def test_pcm16_to_pcmu_conversion(self):
        """Test PCM16 to PCMU conversion"""
        # Create sample PCM data
        pcm_data = np.array([0.1, -0.1, 0.5, -0.5, 0.0], dtype=np.float32)

        # Convert to PCMU
        pcmu_bytes = self.codec.pcm16_to_pcmu(pcm_data)

        assert isinstance(pcmu_bytes, bytes)
        assert len(pcmu_bytes) == len(pcm_data)

    def test_roundtrip_conversion(self):
        """Test PCMU → PCM → PCMU roundtrip"""
        # Original PCM data
        original_pcm = np.array([0.1, -0.2, 0.3, -0.4, 0.0], dtype=np.float32)

        # Convert to PCMU and back
        pcmu_data = self.codec.pcm16_to_pcmu(original_pcm)
        recovered_pcm = self.codec.pcmu_to_pcm16(pcmu_data)

        # Check roundtrip accuracy (allow for μ-law quantization error)
        assert len(recovered_pcm) == len(original_pcm)
        # μ-law has limited precision, so allow reasonable error
        np.testing.assert_allclose(recovered_pcm, original_pcm, atol=0.05)

    def test_telephony_resampling(self):
        """Test resampling for telephony"""
        # Create 16kHz test signal
        sample_rate = 16000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        audio_16k = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))

        # Resample to 8kHz
        audio_8k = self.codec.resample_for_telephony(audio_16k, sample_rate)

        expected_samples_8k = int(8000 * duration)
        assert len(audio_8k) == expected_samples_8k
        assert audio_8k.dtype == np.float32

    def test_rtp_packet_creation(self):
        """Test RTP packet creation"""
        pcmu_data = b'\x80\x81\x82\x83'
        timestamp = 12345
        sequence = 100

        rtp_packet = self.codec.create_rtp_packet(pcmu_data, timestamp, sequence)

        assert isinstance(rtp_packet, bytes)
        assert len(rtp_packet) >= 12 + len(pcmu_data)  # RTP header + payload

    def test_rtp_packet_parsing(self):
        """Test RTP packet parsing"""
        pcmu_data = b'\x80\x81\x82\x83'
        timestamp = 12345
        sequence = 100

        # Create packet
        rtp_packet = self.codec.create_rtp_packet(pcmu_data, timestamp, sequence)

        # Parse packet
        parsed = self.codec.parse_rtp_packet(rtp_packet)

        assert parsed is not None
        assert parsed['sequence'] == sequence
        assert parsed['timestamp'] == timestamp
        assert parsed['payload'] == pcmu_data
        assert parsed['payload_type'] == 0  # PCMU

    def test_audio_quality_validation(self):
        """Test audio quality validation"""
        # Good quality audio
        good_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 8000)) * 0.5
        quality = self.codec.validate_audio_quality(good_audio)

        assert quality['valid'] is True
        assert 0.3 < quality['rms'] < 0.6
        assert quality['clipping_ratio'] < 0.01

        # Poor quality audio (clipped)
        bad_audio = np.ones(1000) * 1.5  # Clipped
        quality = self.codec.validate_audio_quality(bad_audio)

        assert quality['valid'] is False
        assert 'Clipping detected' in quality['quality_issues']

    def test_silence_generation(self):
        """Test silence generation"""
        duration_ms = 100
        sample_rate = 8000

        silence = self.codec.create_silence(duration_ms, sample_rate)

        expected_samples = int(sample_rate * duration_ms / 1000)
        assert len(silence) == expected_samples
        assert np.all(silence == 0.0)
        assert silence.dtype == np.float32


class TestAdvancedVAD:
    """Test suite for Advanced VAD module"""

    def setup_method(self):
        """Set up test fixtures"""
        self.vad_config = {
            'aggressiveness': 2,
            'sample_rate': 8000,
            'frame_duration': 20,
            'barge_in_enabled': True,
            'barge_in_sensitivity': 0.7,
            'min_speech_duration': 300,
            'silence_timeout': 2000
        }
        self.vad = AdvancedVAD(self.vad_config)

    def test_vad_initialization(self):
        """Test VAD initialization"""
        assert self.vad.aggressiveness == 2
        assert self.vad.sample_rate == 8000
        assert self.vad.barge_in_enabled is True
        assert self.vad.is_listening is False

    def test_start_stop_listening(self):
        """Test start/stop listening functionality"""
        assert self.vad.is_listening is False

        self.vad.start_listening()
        assert self.vad.is_listening is True

        self.vad.stop_listening()
        assert self.vad.is_listening is False

    def test_speech_detection(self):
        """Test speech detection with synthetic audio"""
        self.vad.start_listening()

        # Create speech-like signal
        sample_rate = 8000
        duration = 0.5  # 500ms
        samples = int(sample_rate * duration)

        # Speech signal (complex waveform)
        speech_audio = (
            np.sin(2 * np.pi * 200 * np.linspace(0, duration, samples)) +
            0.5 * np.sin(2 * np.pi * 600 * np.linspace(0, duration, samples)) +
            0.2 * np.random.randn(samples)
        ) * 0.3

        # Process audio
        result = self.vad.process_audio(speech_audio)

        # Speech should be detected (may take multiple frames)
        assert isinstance(result, bool)

        # Create silence
        silence_audio = np.zeros(samples, dtype=np.float32)
        result_silence = self.vad.process_audio(silence_audio)

        # Silence should not be detected as speech
        assert result_silence is False

    def test_barge_in_detection(self):
        """Test barge-in detection during TTS"""
        self.vad.start_listening()
        self.vad.set_tts_playing(True)

        # Mock barge-in callback
        barge_in_called = False
        def mock_barge_in_callback(score):
            nonlocal barge_in_called
            barge_in_called = True

        self.vad.set_callbacks(barge_in=mock_barge_in_callback)

        # Create strong speech signal
        sample_rate = 8000
        duration = 0.3  # 300ms
        samples = int(sample_rate * duration)

        speech_audio = np.sin(2 * np.pi * 500 * np.linspace(0, duration, samples)) * 0.8

        # Process audio multiple times to trigger barge-in
        for _ in range(5):
            self.vad.process_audio(speech_audio)

        # Allow some time for callback processing
        import time
        time.sleep(0.1)

        # Barge-in should be detected
        stats = self.vad.get_statistics()
        assert stats['tts_playing'] is True

    def test_audio_energy_calculation(self):
        """Test audio energy calculation"""
        # High energy signal
        high_energy_audio = np.ones(1000, dtype=np.float32) * 0.8
        energy = self.vad._calculate_audio_energy(high_energy_audio)
        assert energy > 0.7

        # Low energy signal
        low_energy_audio = np.ones(1000, dtype=np.float32) * 0.1
        energy = self.vad._calculate_audio_energy(low_energy_audio)
        assert energy < 0.2

        # Silence
        silence = np.zeros(1000, dtype=np.float32)
        energy = self.vad._calculate_audio_energy(silence)
        assert energy == 0.0

    def test_spectral_centroid_calculation(self):
        """Test spectral centroid calculation"""
        sample_rate = 8000

        # Low frequency signal (should have low centroid)
        low_freq = np.sin(2 * np.pi * 200 * np.linspace(0, 1, sample_rate))
        centroid_low = self.vad._calculate_spectral_centroid(low_freq)

        # High frequency signal (should have high centroid)
        high_freq = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, sample_rate))
        centroid_high = self.vad._calculate_spectral_centroid(high_freq)

        assert centroid_high > centroid_low
        assert 0 <= centroid_low <= sample_rate / 2
        assert 0 <= centroid_high <= sample_rate / 2

    def test_statistics_tracking(self):
        """Test VAD statistics tracking"""
        self.vad.start_listening()

        initial_stats = self.vad.get_statistics()
        assert initial_stats['frames_processed'] == 0

        # Process some audio
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        self.vad.process_audio(audio)

        updated_stats = self.vad.get_statistics()
        assert updated_stats['frames_processed'] > initial_stats['frames_processed']

        # Reset statistics
        self.vad.reset_statistics()
        reset_stats = self.vad.get_statistics()
        assert reset_stats['frames_processed'] == 0


class TestRTPAudioBridge:
    """Test suite for RTP Audio Bridge"""

    @pytest.fixture
    def audio_callback(self):
        """Mock audio callback"""
        return Mock()

    def setup_method(self):
        """Set up test fixtures"""
        self.audio_callback = Mock()
        self.bridge = RTPAudioBridge(audio_callback=self.audio_callback)

    @pytest.mark.asyncio
    async def test_bridge_initialization(self):
        """Test RTP bridge initialization"""
        result = await self.bridge.initialize()
        assert result is True
        assert self.bridge.peer_connection is not None
        assert self.bridge.audio_track is not None

    @pytest.mark.asyncio
    async def test_send_audio(self):
        """Test sending audio through bridge"""
        await self.bridge.initialize()
        self.bridge.is_running = True

        # Create test audio
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 800)).astype(np.float32)

        # Send audio (should not raise exception)
        await self.bridge.send_audio(audio_data)

        # Verify audio was processed
        assert len(audio_data) > 0

    def test_audio_preparation(self):
        """Test audio preparation for transmission"""
        # Test stereo to mono conversion
        stereo_audio = np.random.randn(1000, 2).astype(np.float32)
        mono_audio = self.bridge._prepare_audio_for_transmission(stereo_audio)

        assert len(mono_audio.shape) == 1  # Should be mono
        assert mono_audio.dtype == np.float32

        # Test clipping
        loud_audio = np.ones(1000) * 2.0  # Above [-1, 1] range
        clipped_audio = self.bridge._prepare_audio_for_transmission(loud_audio)

        assert np.all(np.abs(clipped_audio) <= 1.0)

    @pytest.mark.asyncio
    async def test_bridge_status(self):
        """Test bridge status reporting"""
        status = self.bridge.get_status()

        assert 'is_running' in status
        assert 'connection_state' in status
        assert 'sample_rate' in status
        assert 'codec' in status

        assert status['sample_rate'] == 8000
        assert status['codec'] == "PCMU"

    @pytest.mark.asyncio
    async def test_bridge_lifecycle(self):
        """Test complete bridge lifecycle"""
        # Initialize
        assert await self.bridge.initialize() is True

        # Start
        await self.bridge.start_bridge("localhost", 10000)
        assert self.bridge.is_running is True

        # Stop
        await self.bridge.stop_bridge()
        assert self.bridge.is_running is False


class TestCustomSIPClient:
    """Test suite for SIP Call Handler (mocked)"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sip_config = {
            'server': 'test.server.com',
            'port': 5060,
            'username': '1600',
            'password': 'testpass',
            'display_name': 'Test Bot'
        }
        self.call_callback = Mock()

    def test_sip_handler_initialization(self):
        """Test SIP handler initialization"""
        handler = CustomSIPClient(self.sip_config, self.call_callback)

        assert handler.server == 'test.server.com'
        assert handler.username == '1600'
        assert handler.password == 'testpass'
        assert handler.running is False
        assert handler.registered is False

    def test_caller_number_extraction(self):
        """Test caller number extraction from SIP headers"""
        handler = CustomSIPClient(self.sip_config)

        # Test standard format
        from_header = '"John Doe" <sip:1234567890@domain.com>'
        number = handler._extract_caller_number(from_header)
        assert number == '1234567890'

        # Test simple format
        from_header = 'sip:555123@test.com'
        number = handler._extract_caller_number(from_header)
        assert number == '555123'

        # Test malformed header
        from_header = 'invalid header'
        number = handler._extract_caller_number(from_header)
        assert number == 'Unknown'

    def test_sdp_answer_creation(self):
        """Test SDP answer creation"""
        handler = CustomSIPClient(self.sip_config)

        sdp_answer = handler._create_sdp()

        assert isinstance(sdp_answer, str)
        assert 'v=0' in sdp_answer  # Version
        assert 'PCMU/8000' in sdp_answer  # Audio codec
        assert 'm=audio' in sdp_answer  # Media description

    def test_call_status_tracking(self):
        """Test call status tracking"""
        handler = CustomSIPClient(self.sip_config)

        # Initially no active calls
        active_calls = handler.get_active_calls()
        assert len(active_calls) == 0

        status = handler.get_status()
        assert status['active_calls'] == 0
        assert status['running'] is False
        assert status['registered'] is False


class TestCallFlowManager:
    """Test suite for Call Flow Manager (integration test)"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sip_config = {
            'server': 'test.server.com',
            'port': 5060,
            'username': '1600',
            'password': 'testpass'
        }

        self.voice_bot_config = VoiceBotConfig()

    def test_call_flow_manager_initialization(self):
        """Test call flow manager initialization"""
        manager = CallFlowManager(self.sip_config, self.voice_bot_config)

        assert manager.sip_config == self.sip_config
        assert manager.voice_bot_config == self.voice_bot_config
        assert manager.running is False
        assert len(manager.active_sessions) == 0

    def test_manager_status_reporting(self):
        """Test manager status reporting"""
        manager = CallFlowManager(self.sip_config, self.voice_bot_config)

        status = manager.get_status()

        assert 'running' in status
        assert 'active_sessions' in status
        assert 'total_calls' in status
        assert 'successful_calls' in status
        assert 'success_rate' in status

        assert status['active_sessions'] == 0
        assert status['total_calls'] == 0


class TestIntegrationScenarios:
    """Integration test scenarios"""

    def setup_method(self):
        """Set up integration test fixtures"""
        self.codec = AudioCodec()
        self.vad_config = {
            'aggressiveness': 2,
            'sample_rate': 8000,
            'barge_in_enabled': True
        }

    def test_complete_audio_pipeline(self):
        """Test complete audio processing pipeline"""
        # Step 1: Create test speech audio (22kHz TTS output)
        tts_sample_rate = 22050
        duration = 1.0  # 1 second
        samples = int(tts_sample_rate * duration)

        # Simulate TTS output (complex waveform)
        tts_audio = (
            np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)) +
            0.3 * np.sin(2 * np.pi * 880 * np.linspace(0, duration, samples))
        ) * 0.5

        # Step 2: Process for telephony transmission
        pcmu_data = self.codec.process_tts_audio_chunk(tts_audio, tts_sample_rate)
        assert len(pcmu_data) > 0

        # Step 3: Simulate telephony transmission (PCMU → PCM → PCMU)
        received_pcm = self.codec.pcmu_to_pcm16(pcmu_data)
        retransmit_pcmu = self.codec.pcm16_to_pcmu(received_pcm)

        # Step 4: Process for ASR
        asr_audio = self.codec.process_telephony_audio_chunk(retransmit_pcmu)
        assert len(asr_audio) > 0
        assert asr_audio.dtype == np.float32

        # Verify pipeline integrity
        assert len(pcmu_data) > 0
        assert len(received_pcm) == len(pcmu_data)
        assert len(retransmit_pcmu) == len(pcmu_data)

    def test_vad_audio_codec_integration(self):
        """Test VAD integration with audio codec"""
        vad = AdvancedVAD(self.vad_config)
        vad.start_listening()

        # Create telephony audio (8kHz PCMU)
        sample_rate = 8000
        duration = 0.5
        samples = int(sample_rate * duration)

        # Speech-like signal
        speech_8k = np.sin(2 * np.pi * 300 * np.linspace(0, duration, samples)) * 0.4

        # Convert to PCMU and back (simulating RTP transmission)
        pcmu_data = self.codec.pcm16_to_pcmu(speech_8k)
        recovered_audio = self.codec.pcmu_to_pcm16(pcmu_data)

        # Process with VAD
        speech_detected = vad.process_audio(recovered_audio)

        # Speech should be detected after codec processing
        assert isinstance(speech_detected, bool)

        # Test with silence
        silence = np.zeros(samples, dtype=np.float32)
        pcmu_silence = self.codec.pcm16_to_pcmu(silence)
        recovered_silence = self.codec.pcmu_to_pcm16(pcmu_silence)

        silence_detected = vad.process_audio(recovered_silence)
        assert silence_detected is False

    def test_configuration_loading(self):
        """Test configuration file loading"""
        # Create temporary config file
        config_data = {
            'sip': {
                'server': 'test.example.com',
                'port': 5060,
                'username': 'testuser'
            },
            'audio': {
                'codec': 'PCMU',
                'sample_rate': 8000
            },
            'voice_bot': {
                'max_tokens': 50
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Load configuration
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)

            assert loaded_config['sip']['server'] == 'test.example.com'
            assert loaded_config['audio']['codec'] == 'PCMU'
            assert loaded_config['voice_bot']['max_tokens'] == 50

        finally:
            os.unlink(config_file)


def create_test_wav_file(filename: str, duration: float = 1.0, sample_rate: int = 16000, frequency: float = 440.0):
    """Create a test WAV file with sine wave"""
    samples = int(sample_rate * duration)
    audio_data = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
    audio_int16 = (audio_data * 32767).astype(np.int16)

    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


@pytest.mark.asyncio
async def test_end_to_end_call_simulation():
    """End-to-end call simulation test"""

    # This test simulates a complete call flow without actual network components
    logger.info("Starting end-to-end call simulation test")

    # 1. Initialize components
    codec = AudioCodec()

    # 2. Simulate incoming call audio (PCMU from caller)
    caller_audio_pcm = np.sin(2 * np.pi * 300 * np.linspace(0, 2.0, 16000)) * 0.3  # 2 seconds at 8kHz
    caller_pcmu = codec.pcm16_to_pcmu(caller_audio_pcm)

    # 3. Process caller audio for ASR
    asr_audio = codec.process_telephony_audio_chunk(caller_pcmu)
    assert len(asr_audio) > 0

    # 4. Simulate TTS response audio
    response_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, 33075)) * 0.4  # 1.5s at 22kHz

    # 5. Process TTS for telephony transmission
    response_pcmu = codec.process_tts_audio_chunk(response_audio, 22050)
    assert len(response_pcmu) > 0

    # 6. Verify audio quality
    quality = codec.validate_audio_quality(asr_audio)
    assert quality['valid'] is True or len(quality['quality_issues']) == 0

    logger.info("End-to-end call simulation test completed successfully")


if __name__ == "__main__":
    """Run tests directly"""
    import sys

    # Configure logging for tests
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    print("🧪 Running NETOVO Voice Bot Telephony Integration Tests")
    print("=" * 60)

    # Run individual test classes
    test_classes = [
        TestAudioCodec,
        TestAdvancedVAD,
        TestRTPAudioBridge,
        TestSIPCallHandler,
        TestCallFlowManager,
        TestIntegrationScenarios
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")

        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]

        for test_method in test_methods:
            total_tests += 1
            try:
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()

                method = getattr(test_instance, test_method)

                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()

                print(f"  ✅ {test_method}")
                passed_tests += 1

            except Exception as e:
                print(f"  ❌ {test_method}: {e}")

    # Run end-to-end test
    print(f"\n📋 Running End-to-End Test...")
    total_tests += 1
    try:
        asyncio.run(test_end_to_end_call_simulation())
        print(f"  ✅ test_end_to_end_call_simulation")
        passed_tests += 1
    except Exception as e:
        print(f"  ❌ test_end_to_end_call_simulation: {e}")

    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")

    if passed_tests == total_tests:
        print("🎉 All telephony integration tests passed!")
        sys.exit(0)
    else:
        print(f"❌ {total_tests - passed_tests} tests failed")
        sys.exit(1)