"""
Professional audio processing module
Handles resampling, VAD, and format conversions
"""

import numpy as np
import soxr
import webrtcvad
import audioop
import wave
import tempfile
import os
import structlog
from difflib import SequenceMatcher
import string
import jiwer

logger = structlog.get_logger()

class AudioProcessor:
    """Handles all audio processing tasks"""

    def __init__(self):
        # Initialize VAD for barge-in detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
        self.vad_frame_duration = 20  # ms

    def resample(self, audio: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
        """High-quality resampling using soxr"""
        if input_rate == output_rate:
            return audio

        try:
            resampled = soxr.resample(audio, input_rate, output_rate)
            logger.debug("Audio resampled",
                        input_rate=input_rate,
                        output_rate=output_rate,
                        input_len=len(audio),
                        output_len=len(resampled))
            return resampled
        except Exception as e:
            logger.error("Resampling failed", error=str(e))
            raise

    def ulaw_to_pcm16(self, ulaw_data: bytes) -> np.ndarray:
        """Convert μ-law to 16-bit PCM using stdlib"""
        pcm_bytes = audioop.ulaw2lin(ulaw_data, 2)  # 2 bytes = 16-bit
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        # Convert to float32 in range [-1, 1]
        return pcm_array.astype(np.float32) / 32768.0

    def pcm16_to_ulaw(self, pcm_array: np.ndarray) -> bytes:
        """Convert 16-bit PCM to μ-law using stdlib"""
        # Convert float32 [-1, 1] to int16
        pcm_int16 = (pcm_array * 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()
        return audioop.lin2ulaw(pcm_bytes, 2)

    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Detect speech using WebRTC VAD"""
        # VAD expects 16-bit PCM at specific rates (8000, 16000, 32000, 48000)
        if sample_rate not in [8000, 16000, 32000, 48000]:
            # Resample to nearest supported rate
            target_rate = min([8000, 16000, 32000, 48000],
                            key=lambda x: abs(x - sample_rate))
            audio = self.resample(audio, sample_rate, target_rate)
            sample_rate = target_rate

        # Convert to 16-bit PCM bytes
        pcm_int16 = (audio * 32767).astype(np.int16)

        # VAD frame size calculation
        frame_length = int(sample_rate * self.vad_frame_duration / 1000)

        # Process in frames
        speech_frames = 0
        total_frames = 0

        for i in range(0, len(pcm_int16) - frame_length, frame_length):
            frame = pcm_int16[i:i + frame_length].tobytes()
            try:
                if self.vad.is_speech(frame, sample_rate):
                    speech_frames += 1
                total_frames += 1
            except:
                # Skip invalid frames
                continue

        # Return True if >30% of frames contain speech
        if total_frames == 0:
            return False

        speech_ratio = speech_frames / total_frames
        return speech_ratio > 0.3

    def resample_wav(self, input_wav_path: str, target_rate: int) -> str:
        """Resample WAV file to target sample rate, return new file path"""
        try:
            # Read input WAV
            with wave.open(input_wav_path, 'rb') as wav_in:
                frames = wav_in.readframes(wav_in.getnframes())
                sample_rate = wav_in.getframerate()
                channels = wav_in.getnchannels()

            # Convert to numpy array
            if channels == 1:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                # Convert stereo to mono
                audio = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels).astype(np.float32) / 32768.0
                audio = audio.mean(axis=1)

            # Resample if needed
            if sample_rate != target_rate:
                audio = self.resample(audio, sample_rate, target_rate)

            # Create output file
            output_fd, output_path = tempfile.mkstemp(suffix='.wav')
            os.close(output_fd)

            # Write resampled audio
            with wave.open(output_path, 'wb') as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2)  # 16-bit
                wav_out.setframerate(target_rate)
                audio_int16 = (audio * 32767).astype(np.int16)
                wav_out.writeframes(audio_int16.tobytes())

            logger.debug("WAV resampled",
                        input_rate=sample_rate,
                        output_rate=target_rate,
                        output_file=output_path)

            return output_path

        except Exception as e:
            logger.error("WAV resampling failed", error=str(e))
            return input_wav_path  # Return original on failure

class AudioBuffer:
    """Circular buffer for audio processing"""

    def __init__(self, max_duration_seconds: float = 10.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.samples_written = 0

    def add_audio(self, audio: np.ndarray):
        """Add audio to circular buffer"""
        audio_len = len(audio)

        if audio_len > self.max_samples:
            # Audio longer than buffer, take the last part
            audio = audio[-self.max_samples:]
            audio_len = self.max_samples

        # Handle wraparound
        end_pos = self.write_pos + audio_len
        if end_pos <= self.max_samples:
            self.buffer[self.write_pos:end_pos] = audio
        else:
            # Split across wraparound
            first_part = self.max_samples - self.write_pos
            self.buffer[self.write_pos:] = audio[:first_part]
            self.buffer[:audio_len - first_part] = audio[first_part:]

        self.write_pos = end_pos % self.max_samples
        self.samples_written = min(self.samples_written + audio_len, self.max_samples)

    def get_latest(self, duration_seconds: float) -> np.ndarray:
        """Get the latest N seconds of audio"""
        samples_needed = int(duration_seconds * self.sample_rate)
        samples_needed = min(samples_needed, self.samples_written)

        if samples_needed == 0:
            return np.array([], dtype=np.float32)

        start_pos = (self.write_pos - samples_needed) % self.max_samples

        if start_pos + samples_needed <= self.max_samples:
            return self.buffer[start_pos:start_pos + samples_needed].copy()
        else:
            # Handle wraparound
            first_part = self.max_samples - start_pos
            result = np.zeros(samples_needed, dtype=np.float32)
            result[:first_part] = self.buffer[start_pos:]
            result[first_part:] = self.buffer[:samples_needed - first_part]
            return result


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation"""
    if not text:
        return ""

    # Convert to lowercase
    normalized = text.lower()

    # Remove punctuation
    normalized = normalized.translate(str.maketrans('', '', string.punctuation))

    # Normalize whitespace
    normalized = ' '.join(normalized.split())

    return normalized


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using WER (Word Error Rate) - more appropriate for ASR testing"""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    if not norm1 or not norm2:
        return 0.0

    try:
        # Calculate WER (Word Error Rate)
        wer = jiwer.wer(norm1, norm2)
        # Convert WER to similarity score (1.0 - WER)
        # WER of 0 = perfect match (similarity = 1.0)
        # WER of 1 = completely different (similarity = 0.0)
        similarity = max(0.0, 1.0 - wer)

        logger.debug("WER calculation",
                    reference=norm1[:50],
                    hypothesis=norm2[:50],
                    wer=wer,
                    similarity=similarity)

        return similarity

    except Exception as e:
        logger.warning("WER calculation failed, falling back to SequenceMatcher", error=str(e))
        # Fallback to original method if WER fails
        matcher = SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()
