"""
Fixed Riva Docker implementation with proper file handling
Handles container-to-host file copying correctly
"""

import subprocess
import tempfile
import os
import wave
import numpy as np
import structlog
import uuid

logger = structlog.get_logger()

class SimpleRivaASR:
    """Simple ASR using Docker container client"""

    def __init__(self, server_url="localhost:50051", container="riva-speech"):
        self.server_url = server_url
        self.container = container

    def transcribe_file(self, audio_file: str, automatic_punctuation: bool = True) -> str:
        """Transcribe audio file using Riva CLI client in Docker"""
        try:
            # Copy audio file into container
            container_path = f"/tmp/riva_asr_input_{uuid.uuid4().hex}.wav"

            # Copy host file to container
            copy_cmd = ["sudo", "docker", "cp", audio_file, f"{self.container}:{container_path}"]
            copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=10)

            if copy_result.returncode != 0:
                logger.error("Failed to copy audio to container", error=copy_result.stderr)
                return "File copy error"

            # Run ASR client inside container
            cmd = [
                "sudo", "docker", "exec", self.container,
                "/opt/riva/clients/riva_streaming_asr_client",
                f"--riva_uri={self.server_url}",
                f"--audio_file={container_path}",
                "--simulate_realtime=false",
                "--model_name=conformer-en-US-asr-streaming-asr-bls-ensemble",
                "--language_code=en-US"
            ]

            # Add automatic punctuation control
            if not automatic_punctuation:
                cmd.append("--automatic_punctuation=false")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Cleanup container file
            subprocess.run(["sudo", "docker", "exec", self.container, "rm", "-f", container_path],
                          capture_output=True)

            if result.returncode == 0:
                # Parse output to extract transcript
                lines = result.stdout.strip().split('\n')

                # Look for transcript patterns - be more specific
                for line in lines:
                    if 'transcript:' in line.lower():
                        if ':' in line:
                            text = line.split(':', 1)[1].strip().strip('"')
                            if text and len(text) > 1 and not text.startswith('Throughput'):
                                logger.info("ASR successful", transcript=text)
                                return text

                # Look for final transcript patterns
                for line in lines:
                    if 'final' in line.lower() and 'transcript' in line.lower():
                        # Extract text after "final transcript:" or similar
                        parts = line.lower().split('transcript')
                        if len(parts) > 1:
                            text = parts[1].split(':', 1)[-1].strip().strip('"')
                            if text and len(text) > 1 and not text.startswith('Throughput'):
                                logger.info("ASR final transcript", transcript=text)
                                return text

                # Look for quoted text that looks like speech
                for line in lines:
                    if '"' in line and not 'Throughput' in line and not 'RTFX' in line:
                        # Extract quoted text
                        import re
                        quotes = re.findall(r'"([^"]*)"', line)
                        for quote in quotes:
                            if quote.strip() and len(quote.strip()) > 2:
                                logger.info("ASR quoted text", transcript=quote.strip())
                                return quote.strip()

                # Log the full output for debugging
                logger.error("No valid transcript found in ASR output",
                            stdout_lines=lines[:5])  # First 5 lines

            logger.error("ASR failed", stdout=result.stdout, stderr=result.stderr)
            return "Speech not understood"

        except Exception as e:
            logger.error("ASR transcription failed", error=str(e))
            return "Audio processing error"

class SimpleRivaTTS:
    """Simple TTS using Docker container client"""

    def __init__(self, server_url="localhost:50051", container="riva-speech"):
        self.server_url = server_url
        self.container = container

    def synthesize(self, text: str, voice="English-US.Female-1", sample_rate=22050) -> str:
        """Synthesize text and return path to WAV file"""
        try:
            # Create host temp file (final destination)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                host_output = tmp_file.name

            # Container temp file
            container_output = f"/tmp/riva_tts_{uuid.uuid4().hex}.wav"

            # Run TTS client inside container
            cmd = [
                "sudo", "docker", "exec", self.container,
                "/opt/riva/clients/riva_tts_client",
                f"--riva_uri={self.server_url}",
                f"--text={text}",
                f"--voice_name={voice}",
                f"--audio_file={container_output}",
                f"--rate={sample_rate}"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error("TTS failed", stdout=result.stdout, stderr=result.stderr)
                return ""

            # Copy the file from container to host
            copy_cmd = ["sudo", "docker", "cp", f"{self.container}:{container_output}", host_output]
            copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=10)

            # Cleanup container file (best effort)
            subprocess.run(["sudo", "docker", "exec", self.container, "rm", "-f", container_output],
                          capture_output=True)

            if copy_result.returncode == 0 and os.path.exists(host_output):
                logger.info("TTS synthesis successful",
                          text=text[:50],
                          voice=voice,
                          output=host_output)
                return host_output
            else:
                logger.error("TTS copy failed", error=copy_result.stderr)
                return ""

        except Exception as e:
            logger.error("TTS synthesis failed", error=str(e))
            return ""

def test_riva_services():
    """Test both ASR and TTS services"""
    print("Testing Riva services...")

    # Test TTS
    print("Testing TTS...")
    tts = SimpleRivaTTS()
    tts_result = tts.synthesize("Hello from the voice bot integration test")
    print(f"TTS result: {tts_result}")

    if tts_result and os.path.exists(tts_result):
        print(f"✓ TTS WAV created: {os.path.getsize(tts_result)} bytes")

        # Test ASR with the generated audio
        print("Testing ASR with generated audio...")
        asr = SimpleRivaASR()
        transcript = asr.transcribe_file(tts_result)
        print(f"ASR transcript: '{transcript}'")

        # Cleanup test file (use sudo for proper permissions)
        subprocess.run(["sudo", "rm", "-f", tts_result], capture_output=True)

        if transcript and "voice bot" in transcript.lower():
            print("✓ Full TTS → ASR pipeline works!")
            return True

    print("✗ Pipeline test failed")
    return False

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

if __name__ == "__main__":
    test_riva_services()
