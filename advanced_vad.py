"""
Advanced Voice Activity Detection (VAD) for Telephony Integration
Provides barge-in capability and real-time speech detection during TTS playback
"""

import numpy as np
import webrtcvad
import threading
import asyncio
from typing import Optional, Callable, List, Dict, Any
from collections import deque
import time
import structlog

logger = structlog.get_logger()

class AdvancedVAD:
    """Advanced Voice Activity Detection with barge-in support"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Advanced VAD

        Args:
            config: VAD configuration dictionary
        """
        self.config = config

        # WebRTC VAD settings
        self.aggressiveness = config.get('aggressiveness', 2)  # 0-3
        self.sample_rate = config.get('sample_rate', 8000)
        self.frame_duration = config.get('frame_duration', 20)  # ms

        # Barge-in settings
        self.barge_in_enabled = config.get('barge_in_enabled', True)
        self.barge_in_sensitivity = config.get('barge_in_sensitivity', 0.7)
        self.interrupt_delay = config.get('interrupt_delay', 0.5)

        # Speech detection settings
        self.min_speech_duration = config.get('min_speech_duration', 300)  # ms
        self.silence_timeout = config.get('silence_timeout', 2000)  # ms

        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(self.aggressiveness)

        # Frame buffer settings
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.audio_buffer = deque(maxlen=50)  # Keep last 1 second of audio

        # State management
        self.is_listening = False
        self.is_speaking_detected = False
        self.tts_playing = False
        self.last_speech_time = 0
        self.speech_start_time = 0

        # Callbacks
        self.speech_start_callback: Optional[Callable] = None
        self.speech_end_callback: Optional[Callable] = None
        self.barge_in_callback: Optional[Callable] = None

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'speech_events': 0,
            'barge_in_events': 0,
            'false_positives': 0
        }

        logger.info("Advanced VAD initialized",
                   aggressiveness=self.aggressiveness,
                   barge_in_enabled=self.barge_in_enabled,
                   sample_rate=self.sample_rate)

    def set_callbacks(self,
                     speech_start: Optional[Callable] = None,
                     speech_end: Optional[Callable] = None,
                     barge_in: Optional[Callable] = None):
        """Set event callbacks"""
        self.speech_start_callback = speech_start
        self.speech_end_callback = speech_end
        self.barge_in_callback = barge_in

    def start_listening(self):
        """Start voice activity detection"""
        self.is_listening = True
        logger.info("VAD listening started")

    def stop_listening(self):
        """Stop voice activity detection"""
        self.is_listening = False
        logger.info("VAD listening stopped")

    def set_tts_playing(self, playing: bool):
        """Set TTS playback state for barge-in detection"""
        self.tts_playing = playing
        if playing:
            logger.debug("TTS playback started - monitoring for barge-in")
        else:
            logger.debug("TTS playback stopped")

    def process_audio(self, audio_data: np.ndarray) -> bool:
        """
        Process audio data and detect voice activity

        Args:
            audio_data: Audio samples (float32, -1 to 1)

        Returns:
            True if speech is detected, False otherwise
        """
        if not self.is_listening:
            return False

        try:
            # Add to audio buffer for analysis
            self.audio_buffer.append(audio_data)

            # Convert to format expected by WebRTC VAD
            audio_int16 = self._prepare_audio_for_vad(audio_data)

            if len(audio_int16) == 0:
                return False

            # Process in frames
            speech_detected = self._process_audio_frames(audio_int16)

            # Handle barge-in detection during TTS
            if speech_detected and self.tts_playing and self.barge_in_enabled:
                self._handle_barge_in_detection(audio_data)

            # Update statistics
            self.stats['frames_processed'] += 1
            if speech_detected:
                self.stats['speech_events'] += 1

            return speech_detected

        except Exception as e:
            logger.error("Error processing audio for VAD", error=str(e))
            return False

    def _prepare_audio_for_vad(self, audio_data: np.ndarray) -> np.ndarray:
        """Prepare audio data for WebRTC VAD processing"""
        try:
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample to supported rate if needed
            if self.sample_rate not in [8000, 16000, 32000, 48000]:
                target_rate = min([8000, 16000, 32000, 48000],
                                key=lambda x: abs(x - self.sample_rate))
                # Simple resampling (use soxr for production quality)
                ratio = target_rate / self.sample_rate
                target_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, target_length)
                audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
                self.sample_rate = target_rate
                self.frame_size = int(self.sample_rate * self.frame_duration / 1000)

            # Convert to int16 and ensure proper frame size
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)

            # Pad or truncate to frame boundaries
            remainder = len(audio_int16) % self.frame_size
            if remainder != 0:
                padding = self.frame_size - remainder
                audio_int16 = np.pad(audio_int16, (0, padding), mode='constant')

            return audio_int16

        except Exception as e:
            logger.error("Error preparing audio for VAD", error=str(e))
            return np.array([], dtype=np.int16)

    def _process_audio_frames(self, audio_int16: np.ndarray) -> bool:
        """Process audio in frames through WebRTC VAD"""
        try:
            frames = len(audio_int16) // self.frame_size
            speech_frames = 0

            for i in range(frames):
                start = i * self.frame_size
                end = start + self.frame_size
                frame = audio_int16[start:end]

                # Convert to bytes for WebRTC VAD
                frame_bytes = frame.tobytes()

                # Check if frame contains speech
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)

                if is_speech:
                    speech_frames += 1

            # Determine if overall chunk contains speech
            if frames == 0:
                return False

            speech_ratio = speech_frames / frames
            current_time = time.time() * 1000  # ms

            # Apply temporal smoothing and minimum duration
            if speech_ratio > 0.3:  # At least 30% of frames contain speech
                if not self.is_speaking_detected:
                    self.speech_start_time = current_time
                    self.is_speaking_detected = True

                    # Check minimum speech duration before triggering
                    if hasattr(self, '_pending_speech_start'):
                        if current_time - self._pending_speech_start > self.min_speech_duration:
                            self._trigger_speech_start()
                    else:
                        self._pending_speech_start = current_time

                self.last_speech_time = current_time
                return True

            else:
                # Check for end of speech
                if self.is_speaking_detected:
                    silence_duration = current_time - self.last_speech_time
                    if silence_duration > self.silence_timeout:
                        self._trigger_speech_end()
                        self.is_speaking_detected = False
                        if hasattr(self, '_pending_speech_start'):
                            delattr(self, '_pending_speech_start')

                return False

        except Exception as e:
            logger.error("Error processing VAD frames", error=str(e))
            return False

    def _handle_barge_in_detection(self, audio_data: np.ndarray):
        """Handle barge-in detection during TTS playback"""
        try:
            # Additional analysis for barge-in reliability
            energy_level = self._calculate_audio_energy(audio_data)
            spectral_centroid = self._calculate_spectral_centroid(audio_data)

            # Enhanced barge-in criteria
            barge_in_score = self._calculate_barge_in_score(energy_level, spectral_centroid)

            if barge_in_score > self.barge_in_sensitivity:
                logger.info("Barge-in detected during TTS", score=barge_in_score)

                self.stats['barge_in_events'] += 1

                # Trigger barge-in callback with delay
                if self.barge_in_callback:
                    # Use thread to avoid blocking audio processing
                    threading.Thread(
                        target=self._delayed_barge_in_callback,
                        args=(barge_in_score,)
                    ).start()

        except Exception as e:
            logger.error("Error handling barge-in detection", error=str(e))

    def _delayed_barge_in_callback(self, score: float):
        """Execute barge-in callback with delay"""
        try:
            time.sleep(self.interrupt_delay)

            # Double-check that speech is still detected
            if self.is_speaking_detected and self.barge_in_callback:
                self.barge_in_callback(score)

        except Exception as e:
            logger.error("Error in delayed barge-in callback", error=str(e))

    def _calculate_audio_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy of audio signal"""
        try:
            if len(audio_data) == 0:
                return 0.0

            return float(np.sqrt(np.mean(audio_data ** 2)))

        except Exception as e:
            logger.error("Error calculating audio energy", error=str(e))
            return 0.0

    def _calculate_spectral_centroid(self, audio_data: np.ndarray) -> float:
        """Calculate spectral centroid for speech characteristics"""
        try:
            if len(audio_data) < 64:  # Minimum length for FFT
                return 0.0

            # Simple spectral centroid calculation
            fft = np.abs(np.fft.fft(audio_data))
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)

            # Only consider positive frequencies
            half_len = len(fft) // 2
            fft_positive = fft[:half_len]
            freqs_positive = freqs[:half_len]

            # Calculate weighted centroid
            if np.sum(fft_positive) > 0:
                centroid = np.sum(freqs_positive * fft_positive) / np.sum(fft_positive)
                return float(centroid)
            else:
                return 0.0

        except Exception as e:
            logger.error("Error calculating spectral centroid", error=str(e))
            return 0.0

    def _calculate_barge_in_score(self, energy: float, spectral_centroid: float) -> float:
        """Calculate composite barge-in detection score"""
        try:
            # Normalize energy (typical speech energy range)
            energy_score = min(energy / 0.1, 1.0)  # 0.1 is typical speech RMS

            # Normalize spectral centroid (typical speech range 500-2000 Hz)
            centroid_score = 0.0
            if 500 <= spectral_centroid <= 2000:
                centroid_score = 1.0 - abs(spectral_centroid - 1250) / 750

            # Weighted combination
            composite_score = (energy_score * 0.7) + (centroid_score * 0.3)

            logger.debug("Barge-in score calculation",
                        energy=energy,
                        energy_score=energy_score,
                        spectral_centroid=spectral_centroid,
                        centroid_score=centroid_score,
                        composite_score=composite_score)

            return float(composite_score)

        except Exception as e:
            logger.error("Error calculating barge-in score", error=str(e))
            return 0.0

    def _trigger_speech_start(self):
        """Trigger speech start callback"""
        try:
            logger.debug("Speech start detected")

            if self.speech_start_callback:
                # Use thread to avoid blocking audio processing
                threading.Thread(target=self.speech_start_callback).start()

        except Exception as e:
            logger.error("Error triggering speech start", error=str(e))

    def _trigger_speech_end(self):
        """Trigger speech end callback"""
        try:
            current_time = time.time() * 1000
            speech_duration = current_time - self.speech_start_time

            logger.debug("Speech end detected", duration_ms=speech_duration)

            if self.speech_end_callback:
                # Use thread to avoid blocking audio processing
                threading.Thread(
                    target=self.speech_end_callback,
                    args=(speech_duration,)
                ).start()

        except Exception as e:
            logger.error("Error triggering speech end", error=str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get VAD statistics"""
        return {
            'frames_processed': self.stats['frames_processed'],
            'speech_events': self.stats['speech_events'],
            'barge_in_events': self.stats['barge_in_events'],
            'false_positives': self.stats['false_positives'],
            'is_listening': self.is_listening,
            'is_speaking_detected': self.is_speaking_detected,
            'tts_playing': self.tts_playing,
            'aggressiveness': self.aggressiveness,
            'barge_in_enabled': self.barge_in_enabled
        }

    def reset_statistics(self):
        """Reset VAD statistics"""
        self.stats = {
            'frames_processed': 0,
            'speech_events': 0,
            'barge_in_events': 0,
            'false_positives': 0
        }
        logger.info("VAD statistics reset")

    def calibrate(self, background_audio: List[np.ndarray], speech_audio: List[np.ndarray]):
        """
        Calibrate VAD thresholds based on background and speech samples

        Args:
            background_audio: List of background/silence audio samples
            speech_audio: List of speech audio samples
        """
        try:
            logger.info("Starting VAD calibration",
                       background_samples=len(background_audio),
                       speech_samples=len(speech_audio))

            # Analyze background noise characteristics
            background_energies = [self._calculate_audio_energy(audio) for audio in background_audio]
            background_centroids = [self._calculate_spectral_centroid(audio) for audio in background_audio]

            # Analyze speech characteristics
            speech_energies = [self._calculate_audio_energy(audio) for audio in speech_audio]
            speech_centroids = [self._calculate_spectral_centroid(audio) for audio in speech_audio]

            # Calculate optimal thresholds
            bg_energy_mean = np.mean(background_energies) if background_energies else 0.01
            speech_energy_mean = np.mean(speech_energies) if speech_energies else 0.1

            # Adjust barge-in sensitivity based on signal-to-noise ratio
            snr = speech_energy_mean / max(bg_energy_mean, 0.001)

            if snr > 10:  # High SNR - can be more sensitive
                self.barge_in_sensitivity = max(0.5, self.barge_in_sensitivity - 0.1)
            elif snr < 3:  # Low SNR - need to be less sensitive
                self.barge_in_sensitivity = min(0.9, self.barge_in_sensitivity + 0.1)

            logger.info("VAD calibration completed",
                       snr=snr,
                       new_sensitivity=self.barge_in_sensitivity,
                       background_energy_mean=bg_energy_mean,
                       speech_energy_mean=speech_energy_mean)

        except Exception as e:
            logger.error("Error during VAD calibration", error=str(e))


class VADManager:
    """Manages multiple VAD instances for different audio streams"""

    def __init__(self, default_config: Dict[str, Any]):
        self.default_config = default_config
        self.vad_instances: Dict[str, AdvancedVAD] = {}
        self.global_stats = {
            'total_streams': 0,
            'active_streams': 0
        }

    def create_vad_instance(self, stream_id: str, config: Optional[Dict[str, Any]] = None) -> AdvancedVAD:
        """Create a new VAD instance for a stream"""
        try:
            vad_config = config or self.default_config.copy()
            vad_instance = AdvancedVAD(vad_config)

            self.vad_instances[stream_id] = vad_instance
            self.global_stats['total_streams'] += 1
            self.global_stats['active_streams'] += 1

            logger.info("Created VAD instance", stream_id=stream_id)
            return vad_instance

        except Exception as e:
            logger.error("Error creating VAD instance", stream_id=stream_id, error=str(e))
            raise

    def remove_vad_instance(self, stream_id: str):
        """Remove VAD instance for a stream"""
        try:
            if stream_id in self.vad_instances:
                self.vad_instances[stream_id].stop_listening()
                del self.vad_instances[stream_id]
                self.global_stats['active_streams'] -= 1
                logger.info("Removed VAD instance", stream_id=stream_id)

        except Exception as e:
            logger.error("Error removing VAD instance", stream_id=stream_id, error=str(e))

    def get_vad_instance(self, stream_id: str) -> Optional[AdvancedVAD]:
        """Get VAD instance for a stream"""
        return self.vad_instances.get(stream_id)

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global VAD statistics"""
        return {
            **self.global_stats,
            'instance_stats': {
                stream_id: vad.get_statistics()
                for stream_id, vad in self.vad_instances.items()
            }
        }