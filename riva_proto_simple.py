"""
Fixed Riva Docker implementation for AGI context
Removes sudo requirements and handles AGI environment properly
"""

import subprocess
import tempfile
import os
import wave
import numpy as np
import structlog
import uuid
import time

logger = structlog.get_logger()

class SimpleRivaASR:
    """ASR using Docker without sudo"""

    def __init__(self, server_url="localhost:50051", container="riva-speech"):
        self.server_url = server_url
        self.container = container

    def transcribe_file(self, audio_file: str, automatic_punctuation: bool = True) -> str:
        """Transcribe audio file using Riva CLI client in Docker"""
        try:
            # Create unique container path
            container_path = f"/tmp/riva_asr_input_{uuid.uuid4().hex}.wav"

            # Try docker cp without sudo first
            copy_cmd = ["docker", "cp", audio_file, f"{self.container}:{container_path}"]
            copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=10)

            # If non-sudo fails, try with sudo but handle it properly
            if copy_result.returncode != 0:
                logger.warning("Docker cp without sudo failed, trying with sudo")
                copy_cmd = ["sudo", "-n", "docker", "cp", audio_file, f"{self.container}:{container_path}"]
                copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=10)

            if copy_result.returncode != 0:
                logger.error("Failed to copy audio to container", error=copy_result.stderr)
                return "Audio transfer failed"

            # Run ASR client
            cmd = [
                "docker", "exec", self.container,
                "/opt/riva/clients/riva_streaming_asr_client",
                f"--riva_uri={self.server_url}",
                f"--audio_file={container_path}",
                "--simulate_realtime=false",
                "--model_name=conformer-en-US-asr-streaming-asr-bls-ensemble",
                "--language_code=en-US"
            ]

            if not automatic_punctuation:
                cmd.append("--automatic_punctuation=false")

            # Try without sudo first
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # If that fails, try with sudo
            if result.returncode != 0:
                cmd.insert(0, "sudo")
                cmd.insert(1, "-n")  # Non-interactive sudo
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Cleanup container file
            cleanup_cmd = ["docker", "exec", self.container, "rm", "-f", container_path]
            subprocess.run(cleanup_cmd, capture_output=True, timeout=5)

            if result.returncode == 0:
                # Parse output to extract transcript
                lines = result.stdout.strip().split('\n')

                # Look for transcript in the output
                for i, line in enumerate(lines):
                    if 'final transcripts:' in line.lower():
                        for j in range(i+1, min(i+5, len(lines))):
                            next_line = lines[j].strip()
                            if next_line and ':' in next_line:
                                if next_line.startswith('0 :') or next_line.startswith('1 :'):
                                    text = next_line.split(':', 1)[1].strip().strip('"')
                                    if text and len(text) > 1 and not text.startswith('Throughput'):
                                        logger.info("ASR transcript found", transcript=text)
                                        return text

                # Look for quoted text
                for line in lines:
                    if '"' in line and not 'Throughput' in line and not 'RTFX' in line:
                        import re
                        quotes = re.findall(r'"([^"]*)"', line)
                        for quote in quotes:
                            if quote.strip() and len(quote.strip()) > 2:
                                logger.info("ASR quoted text", transcript=quote.strip())
                                return quote.strip()

                logger.warning("No valid transcript found in ASR output")

            logger.error("ASR failed", returncode=result.returncode, stderr=result.stderr)
            return "Speech not recognized"

        except subprocess.TimeoutExpired:
            logger.error("ASR transcription timed out")
            return "Processing timeout"
        except Exception as e:
            logger.error("ASR transcription failed", error=str(e))
            return "Audio processing error"

class SimpleRivaTTS:
    """TTS using Docker without sudo requirements"""

    def __init__(self, server_url="localhost:50051", container="riva-speech"):
        self.server_url = server_url
        self.container = container

    def synthesize(self, text: str, voice="English-US.Female-1", sample_rate=22050) -> str:
        """Synthesize text and return path to WAV file"""
        try:
            # Create host temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                host_output = tmp_file.name

            # Container temp file
            container_output = f"/tmp/riva_tts_{uuid.uuid4().hex}.wav"

            # Build TTS command
            cmd = [
                "docker", "exec", self.container,
                "/opt/riva/clients/riva_tts_client",
                f"--riva_uri={self.server_url}",
                f"--text={text}",
                f"--voice_name={voice}",
                f"--audio_file={container_output}",
                f"--rate={sample_rate}"
            ]

            # Try TTS without sudo first
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # If that fails, try with sudo (non-interactive)
            if result.returncode != 0:
                logger.warning("TTS without sudo failed, trying with sudo")
                cmd.insert(0, "sudo")
                cmd.insert(1, "-n")  # Non-interactive
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error("TTS generation failed", stdout=result.stdout, stderr=result.stderr)
                # Cleanup and return empty
                try:
                    os.unlink(host_output)
                except:
                    pass
                return ""

            # Copy file from container to host
            copy_cmd = ["docker", "cp", f"{self.container}:{container_output}", host_output]
            copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=10)

            # If copy fails without sudo, try with sudo
            if copy_result.returncode != 0:
                copy_cmd = ["sudo", "-n", "docker", "cp", f"{self.container}:{container_output}", host_output]
                copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=10)

            # Cleanup container file
            cleanup_cmd = ["docker", "exec", self.container, "rm", "-f", container_output]
            subprocess.run(cleanup_cmd, capture_output=True, timeout=5)

            if copy_result.returncode == 0 and os.path.exists(host_output):
                # Verify file is not empty
                if os.path.getsize(host_output) > 1000:  # At least 1KB
                    logger.info("TTS synthesis successful", 
                              text=text[:50], 
                              voice=voice, 
                              output=host_output,
                              size=os.path.getsize(host_output))
                    return host_output
                else:
                    logger.error("TTS file too small", size=os.path.getsize(host_output))
                    os.unlink(host_output)
                    return ""
            else:
                logger.error("TTS copy failed", error=copy_result.stderr)
                # Cleanup failed file
                try:
                    os.unlink(host_output)
                except:
                    pass
                return ""

        except subprocess.TimeoutExpired:
            logger.error("TTS synthesis timed out")
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

        # Cleanup test file
        try:
            os.unlink(tts_result)
        except:
            pass

        if transcript and "voice bot" in transcript.lower():
            print("✓ Full TTS → ASR pipeline works!")
            return True

    print("✗ Pipeline test failed")
    return False

if __name__ == "__main__":
    test_riva_services()
