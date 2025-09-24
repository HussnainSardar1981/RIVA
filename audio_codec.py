"""
Audio Codec Module for Telephony Integration
Handles PCMU/G.711 μ-law codec conversions and telephony audio processing
"""

import numpy as np
import audioop
import struct
import structlog
from typing import Optional

logger = structlog.get_logger()

class AudioCodec:
    """Audio codec for telephony integration with G.711 PCMU support"""

    def __init__(self):
        """Initialize audio codec"""
        self.sample_rate_8k = 8000   # Standard telephony
        self.sample_rate_16k = 16000  # ASR optimal
        self.sample_rate_22k = 22050  # TTS output

        # G.711 PCMU parameters
        self.frame_size_8k = 160      # 20ms at 8kHz
        self.frame_size_16k = 320     # 20ms at 16kHz

        logger.info("Audio codec initialized with G.711 PCMU support")

    def pcmu_to_pcm16(self, pcmu_data: bytes) -> np.ndarray:
        """
        Convert G.711 PCMU (μ-law) to 16-bit PCM

        Args:
            pcmu_data: Raw PCMU bytes from RTP

        Returns:
            numpy array of float32 audio data in range [-1, 1]
        """
        try:
            if not pcmu_data:
                return np.array([], dtype=np.float32)

            # Convert μ-law to 16-bit linear PCM using stdlib audioop
            pcm_bytes = audioop.ulaw2lin(pcmu_data, 2)  # 2 bytes = 16-bit

            # Convert to numpy array
            pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

            # Convert to float32 in range [-1, 1]
            pcm_float = pcm_int16.astype(np.float32) / 32768.0

            logger.debug("PCMU to PCM16 conversion",
                        input_bytes=len(pcmu_data),
                        output_samples=len(pcm_float))

            return pcm_float

        except Exception as e:
            logger.error("PCMU to PCM16 conversion failed", error=str(e))
            return np.array([], dtype=np.float32)

    def pcm16_to_pcmu(self, pcm_data: np.ndarray) -> bytes:
        """
        Convert 16-bit PCM to G.711 PCMU (μ-law)

        Args:
            pcm_data: numpy array of float32 audio data in range [-1, 1]

        Returns:
            PCMU bytes for RTP transmission
        """
        try:
            if len(pcm_data) == 0:
                return b''

            # Ensure data is in correct range and type
            pcm_data = np.clip(pcm_data, -1.0, 1.0)

            # Convert float32 [-1, 1] to int16
            pcm_int16 = (pcm_data * 32767).astype(np.int16)

            # Convert to bytes
            pcm_bytes = pcm_int16.tobytes()

            # Convert linear PCM to μ-law using stdlib audioop
            pcmu_bytes = audioop.lin2ulaw(pcm_bytes, 2)  # 2 bytes = 16-bit

            logger.debug("PCM16 to PCMU conversion",
                        input_samples=len(pcm_data),
                        output_bytes=len(pcmu_bytes))

            return pcmu_bytes

        except Exception as e:
            logger.error("PCM16 to PCMU conversion failed", error=str(e))
            return b''

    def resample_for_telephony(self, audio: np.ndarray, input_rate: int) -> np.ndarray:
        """
        Resample audio to 8kHz for telephony
        Uses simple linear interpolation - for production, use soxr from audio_processing

        Args:
            audio: Input audio data
            input_rate: Input sample rate

        Returns:
            Audio resampled to 8kHz
        """
        try:
            if input_rate == self.sample_rate_8k:
                return audio

            # Calculate resampling ratio
            ratio = self.sample_rate_8k / input_rate
            target_length = int(len(audio) * ratio)

            if target_length == 0:
                return np.array([], dtype=np.float32)

            # Simple linear interpolation resampling
            # For production quality, use soxr from audio_processing module
            indices = np.linspace(0, len(audio) - 1, target_length)
            resampled = np.interp(indices, np.arange(len(audio)), audio)

            logger.debug("Audio resampled for telephony",
                        input_rate=input_rate,
                        output_rate=self.sample_rate_8k,
                        input_length=len(audio),
                        output_length=len(resampled))

            return resampled.astype(np.float32)

        except Exception as e:
            logger.error("Telephony resampling failed", error=str(e))
            return audio

    def resample_from_telephony(self, audio: np.ndarray, target_rate: int) -> np.ndarray:
        """
        Resample audio from 8kHz telephony to target rate

        Args:
            audio: 8kHz telephony audio
            target_rate: Target sample rate (typically 16kHz for ASR)

        Returns:
            Audio resampled to target rate
        """
        try:
            if target_rate == self.sample_rate_8k:
                return audio

            # Calculate resampling ratio
            ratio = target_rate / self.sample_rate_8k
            target_length = int(len(audio) * ratio)

            if target_length == 0:
                return np.array([], dtype=np.float32)

            # Simple linear interpolation resampling
            indices = np.linspace(0, len(audio) - 1, target_length)
            resampled = np.interp(indices, np.arange(len(audio)), audio)

            logger.debug("Audio resampled from telephony",
                        input_rate=self.sample_rate_8k,
                        output_rate=target_rate,
                        input_length=len(audio),
                        output_length=len(resampled))

            return resampled.astype(np.float32)

        except Exception as e:
            logger.error("From-telephony resampling failed", error=str(e))
            return audio

    def create_rtp_packet(self, pcmu_data: bytes, timestamp: int, sequence: int) -> bytes:
        """
        Create RTP packet with PCMU payload

        Args:
            pcmu_data: PCMU audio data
            timestamp: RTP timestamp
            sequence: RTP sequence number

        Returns:
            Complete RTP packet bytes
        """
        try:
            if not pcmu_data:
                return b''

            # RTP Header format (12 bytes)
            # V(2) | P(1) | X(1) | CC(4) | M(1) | PT(7) | Sequence(16) | Timestamp(32) | SSRC(32)

            version = 2  # RTP version 2
            padding = 0
            extension = 0
            csrc_count = 0
            marker = 0
            payload_type = 0  # PCMU payload type
            ssrc = 0x12345678  # Synchronization source identifier

            # Pack RTP header
            rtp_header = struct.pack('!BBHII',
                (version << 6) | (padding << 5) | (extension << 4) | csrc_count,
                (marker << 7) | payload_type,
                sequence & 0xFFFF,
                timestamp & 0xFFFFFFFF,
                ssrc
            )

            # Combine header and payload
            rtp_packet = rtp_header + pcmu_data

            logger.debug("RTP packet created",
                        header_size=len(rtp_header),
                        payload_size=len(pcmu_data),
                        total_size=len(rtp_packet),
                        sequence=sequence,
                        timestamp=timestamp)

            return rtp_packet

        except Exception as e:
            logger.error("RTP packet creation failed", error=str(e))
            return b''

    def parse_rtp_packet(self, rtp_packet: bytes) -> Optional[dict]:
        """
        Parse RTP packet to extract PCMU payload

        Args:
            rtp_packet: Complete RTP packet bytes

        Returns:
            Dictionary with RTP info and PCMU payload, or None if parsing fails
        """
        try:
            if len(rtp_packet) < 12:  # Minimum RTP header size
                return None

            # Unpack RTP header
            header_data = struct.unpack('!BBHII', rtp_packet[:12])

            vpxcc = header_data[0]
            mpt = header_data[1]
            sequence = header_data[2]
            timestamp = header_data[3]
            ssrc = header_data[4]

            # Extract fields
            version = (vpxcc >> 6) & 0x3
            padding = (vpxcc >> 5) & 0x1
            extension = (vpxcc >> 4) & 0x1
            csrc_count = vpxcc & 0xF
            marker = (mpt >> 7) & 0x1
            payload_type = mpt & 0x7F

            # Calculate header size
            header_size = 12 + (csrc_count * 4)

            if extension:
                # Handle extension header if present
                if len(rtp_packet) < header_size + 4:
                    return None
                ext_length = struct.unpack('!HH', rtp_packet[header_size:header_size+4])[1]
                header_size += 4 + (ext_length * 4)

            # Extract payload
            if len(rtp_packet) <= header_size:
                payload = b''
            else:
                payload = rtp_packet[header_size:]

                # Handle padding if present
                if padding and len(payload) > 0:
                    padding_length = payload[-1]
                    if padding_length <= len(payload):
                        payload = payload[:-padding_length]

            result = {
                'version': version,
                'padding': padding,
                'extension': extension,
                'csrc_count': csrc_count,
                'marker': marker,
                'payload_type': payload_type,
                'sequence': sequence,
                'timestamp': timestamp,
                'ssrc': ssrc,
                'payload': payload
            }

            logger.debug("RTP packet parsed",
                        payload_type=payload_type,
                        sequence=sequence,
                        timestamp=timestamp,
                        payload_size=len(payload))

            return result

        except Exception as e:
            logger.error("RTP packet parsing failed", error=str(e))
            return None

    def process_telephony_audio_chunk(self, pcmu_chunk: bytes) -> np.ndarray:
        """
        Process a complete telephony audio processing pipeline
        PCMU → PCM → Resample to 16kHz for ASR

        Args:
            pcmu_chunk: Raw PCMU bytes from RTP

        Returns:
            16kHz PCM audio ready for ASR processing
        """
        try:
            # Step 1: Convert PCMU to PCM
            pcm_audio = self.pcmu_to_pcm16(pcmu_chunk)

            if len(pcm_audio) == 0:
                return pcm_audio

            # Step 2: Resample from 8kHz to 16kHz for ASR
            asr_audio = self.resample_from_telephony(pcm_audio, self.sample_rate_16k)

            return asr_audio

        except Exception as e:
            logger.error("Telephony audio chunk processing failed", error=str(e))
            return np.array([], dtype=np.float32)

    def process_tts_audio_chunk(self, tts_audio: np.ndarray, input_rate: int = 22050) -> bytes:
        """
        Process TTS audio for telephony transmission
        TTS Audio → Resample to 8kHz → PCM → PCMU

        Args:
            tts_audio: TTS output audio (typically 22kHz)
            input_rate: Input sample rate from TTS

        Returns:
            PCMU bytes ready for RTP transmission
        """
        try:
            # Step 1: Resample to 8kHz telephony rate
            telephony_audio = self.resample_for_telephony(tts_audio, input_rate)

            if len(telephony_audio) == 0:
                return b''

            # Step 2: Convert to PCMU for transmission
            pcmu_bytes = self.pcm16_to_pcmu(telephony_audio)

            return pcmu_bytes

        except Exception as e:
            logger.error("TTS audio chunk processing failed", error=str(e))
            return b''

    def validate_audio_quality(self, audio: np.ndarray) -> dict:
        """
        Validate audio quality metrics

        Args:
            audio: Audio data to validate

        Returns:
            Dictionary with quality metrics
        """
        try:
            if len(audio) == 0:
                return {'valid': False, 'reason': 'Empty audio'}

            # Calculate RMS level
            rms = np.sqrt(np.mean(audio ** 2))

            # Check for clipping
            clipping_ratio = np.sum(np.abs(audio) >= 0.99) / len(audio)

            # Check for silence
            silence_ratio = np.sum(np.abs(audio) < 0.01) / len(audio)

            # Dynamic range
            dynamic_range = np.max(audio) - np.min(audio)

            # Quality assessment
            quality_issues = []
            if rms < 0.01:
                quality_issues.append('Too quiet')
            if rms > 0.8:
                quality_issues.append('Too loud')
            if clipping_ratio > 0.01:
                quality_issues.append('Clipping detected')
            if silence_ratio > 0.9:
                quality_issues.append('Mostly silent')
            if dynamic_range < 0.1:
                quality_issues.append('Low dynamic range')

            return {
                'valid': len(quality_issues) == 0,
                'rms': float(rms),
                'clipping_ratio': float(clipping_ratio),
                'silence_ratio': float(silence_ratio),
                'dynamic_range': float(dynamic_range),
                'quality_issues': quality_issues,
                'samples': len(audio)
            }

        except Exception as e:
            logger.error("Audio quality validation failed", error=str(e))
            return {'valid': False, 'reason': f'Validation error: {e}'}

    def create_silence(self, duration_ms: int, sample_rate: int = 8000) -> np.ndarray:
        """
        Create silence audio for padding or testing

        Args:
            duration_ms: Duration in milliseconds
            sample_rate: Sample rate

        Returns:
            Silent audio array
        """
        try:
            samples = int(sample_rate * duration_ms / 1000)
            return np.zeros(samples, dtype=np.float32)

        except Exception as e:
            logger.error("Silence creation failed", error=str(e))
            return np.array([], dtype=np.float32)