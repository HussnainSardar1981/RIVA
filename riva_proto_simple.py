"""
Simple Riva gRPC implementation without proto files
Uses the existing riva_tts_client and riva_streaming_asr_client approach
"""

import subprocess
import tempfile
import os
import wave
import numpy as np
import structlog

logger = structlog.get_logger()

class SimpleRivaASR:
    """Simple ASR using command line client"""

    def __init__(self, server_url="localhost:50051"):
        self.server_url = server_url
        self.use_docker = True  # Use Docker containers

    def transcribe_file(self, audio_file: str) -> str:
        """Transcribe audio file using Riva CLI client"""
        try:
            cmd = [
                "sudo", "docker", "exec", "riva-speech",
                "/opt/riva/bin/riva_streaming_asr_client",
                f"--riva_uri={self.server_url}",
                f"--audio_file={audio_file}",
                "--simulate_realtime=false"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                # Parse output to extract transcript
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'transcript:' in line.lower():
                        return line.split(':', 1)[1].strip().strip('"')

                # If no explicit transcript line, return last non-empty line
                for line in reversed(lines):
                    if line.strip() and not line.startswith('['):
                        return line.strip()

            logger.error("ASR failed", stdout=result.stdout, stderr=result.stderr)
            return ""

        except Exception as e:
            logger.error("ASR transcription failed", error=str(e))
            return ""

class SimpleRivaTTS:
    """Simple TTS using command line client"""

    def __init__(self, server_url="localhost:50051"):
        self.server_url = server_url
        self.use_docker = True  # Use Docker containers

    def synthesize(self, text: str, voice="English-US.Female-1", sample_rate=22050) -> str:
        """Synthesize text and return path to WAV file"""
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name

            cmd = [
                "sudo", "docker", "exec", "riva-speech",
                "/opt/riva/bin/riva_tts_client",
                f"--riva_uri={self.server_url}",
                f"--text={text}",
                f"--voice_name={voice}",
                f"--audio_file={output_path}",
                f"--rate={sample_rate}"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and os.path.exists(output_path):
                logger.info("TTS synthesis successful",
                          text=text[:50],
                          voice=voice,
                          output=output_path)
                return output_path
            else:
                logger.error("TTS failed", stdout=result.stdout, stderr=result.stderr)
                return ""

        except Exception as e:
            logger.error("TTS synthesis failed", error=str(e))
            return ""

def save_audio_as_wav(audio_data: np.ndarray, sample_rate: int, filename: str):
    """Save numpy audio array as WAV file"""
    try:
        # Ensure audio is in the right format
        if audio_data.dtype != np.int16:
            # Convert float32 [-1, 1] to int16
            audio_data = (audio_data * 32767).astype(np.int16)

        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        logger.info("Audio saved", file=filename, samples=len(audio_data), rate=sample_rate)
        return True

    except Exception as e:
        logger.error("Failed to save audio", error=str(e))
        return False

def load_wav_file(filename: str) -> tuple:
    """Load WAV file and return (audio_data, sample_rate)"""
    try:
        with wave.open(filename, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(-1)
            audio_data = np.frombuffer(frames, dtype=np.int16)

        logger.info("Audio loaded", file=filename, samples=len(audio_data), rate=sample_rate)
        return audio_data, sample_rate

    except Exception as e:
        logger.error("Failed to load audio", error=str(e))
        return None, None
    
