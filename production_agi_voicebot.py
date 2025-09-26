#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
Simplified AGI VoiceBot - Direct Docker calls with proven commands
Uses the exact same TTS commands that work in your test
"""

import sys
import os
import time
import subprocess
import tempfile
import uuid
from datetime import datetime

# Set up project paths
project_dir = "/home/aiadmin/netovo_voicebot"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Simple logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SimpleBot - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/voicebot.log', mode='a'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class SimpleAGI:
    """Minimal AGI with correct command syntax"""

    def __init__(self):
        self.env = {}
        self.connected = True
        self.call_answered = False
        self._parse_env()

    def _parse_env(self):
        """Parse AGI environment"""
        env_count = 0
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            if ':' in line:
                key, value = line.split(':', 1)
                self.env[key.strip()] = value.strip()
                env_count += 1
        logger.info(f"AGI env parsed: {env_count} vars")

    def command(self, cmd):
        """Send AGI command"""
        try:
            logger.debug(f"AGI: {cmd}")
            print(cmd)
            sys.stdout.flush()

            result = sys.stdin.readline().strip()
            logger.debug(f"Response: {result}")

            return result
        except:
            self.connected = False
            return "ERROR"

    def answer(self):
        """Answer call"""
        result = self.command("ANSWER")
        success = result and result.startswith('200')
        if success:
            self.call_answered = True
            logger.info("Call answered")
        return success

    def hangup(self):
        """Hangup call"""
        self.command("HANGUP")
        self.connected = False

    def verbose(self, msg):
        """Verbose message"""
        return self.command(f'VERBOSE "{msg}"')

    def stream_file(self, filename):
        """Play audio file - NO QUOTES on filename"""
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]

        # Log what we're trying to play
        full_path = f"/var/lib/asterisk/sounds/{filename}.wav"
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            logger.info(f"Attempting to play: {filename} (file exists: {file_size} bytes)")
        else:
            logger.error(f"Audio file not found: {full_path}")

        result = self.command(f'STREAM FILE {filename} ""')
        success = result and result.startswith('200')
        logger.info(f"Stream file result: {result} (success: {success})")
        return success

    def record_file(self, filename):
        """Record audio - SIMPLE syntax"""
        result = self.command(f'RECORD FILE {filename} wav "#" 8000')
        return result and result.startswith('200')

    def sleep(self, seconds):
        """Sleep"""
        time.sleep(seconds)

class DirectTTSClient:
    """Direct Docker TTS using your proven commands"""

    def __init__(self, container="riva-speech"):
        self.container = container

    def synthesize(self, text, voice="English-US.Female-1", sample_rate=22050):
        """Direct TTS synthesis using Docker"""
        try:
            # Create host temp file
            host_output = f"/tmp/tts_agi_{uuid.uuid4().hex}.wav"
            container_output = f"/tmp/riva_tts_{uuid.uuid4().hex}.wav"

            # Run TTS without sudo (AGI doesn't have terminal access)
            cmd = [
                "docker", "exec", self.container,
                "/opt/riva/clients/riva_tts_client",
                f"--riva_uri=localhost:50051",
                f"--text={text}",
                f"--voice_name={voice}",
                f"--audio_file={container_output}",
                f"--rate={sample_rate}"
            ]

            logger.info(f"Running TTS: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"TTS failed: {result.stderr}")
                return None

            # Copy from container to host
            copy_cmd = ["docker", "cp", f"{self.container}:{container_output}", host_output]
            copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=10)

            # Cleanup container file
            subprocess.run(["docker", "exec", self.container, "rm", "-f", container_output],
                          capture_output=True)

            if copy_result.returncode == 0 and os.path.exists(host_output):
                file_size = os.path.getsize(host_output)
                logger.info(f"TTS success: {host_output} ({file_size} bytes)")
                return host_output
            else:
                logger.error(f"TTS copy failed: {copy_result.stderr}")
                return None

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

class DirectASRClient:
    """Direct Docker ASR using your proven commands"""

    def __init__(self, container="riva-speech"):
        self.container = container

    def transcribe_file(self, audio_file):
        """Direct ASR transcription using Docker"""
        try:
            container_path = f"/tmp/riva_asr_{uuid.uuid4().hex}.wav"

            # Copy to container
            copy_cmd = ["docker", "cp", audio_file, f"{self.container}:{container_path}"]
            copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=10)

            if copy_result.returncode != 0:
                logger.error(f"ASR copy failed: {copy_result.stderr}")
                return ""

            # Run ASR
            cmd = [
                "docker", "exec", self.container,
                "/opt/riva/clients/riva_streaming_asr_client",
                f"--riva_uri=localhost:50051",
                f"--audio_file={container_path}",
                "--simulate_realtime=false",
                "--language_code=en-US"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Cleanup
            subprocess.run(["docker", "exec", self.container, "rm", "-f", container_path],
                          capture_output=True)

            if result.returncode == 0:
                # Parse transcript
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if '"' in line and not 'Throughput' in line:
                        import re
                        quotes = re.findall(r'"([^"]*)"', line)
                        for quote in quotes:
                            if quote.strip() and len(quote.strip()) > 2:
                                logger.info(f"ASR result: {quote.strip()}")
                                return quote.strip()

            logger.warning("No transcript found")
            return ""

        except Exception as e:
            logger.error(f"ASR error: {e}")
            return ""

class SimpleOllamaClient:
    """Simple Ollama client"""

    def __init__(self):
        pass

    def generate(self, prompt, max_tokens=50):
        """Generate response"""
        try:
            import httpx

            payload = {
                "model": "orca2:7b",
                "prompt": f"You are Alexis, NETOVO customer support AI. Keep responses under 30 words.\n\nHuman: {prompt}\n\nAssistant:",
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1}
            }

            with httpx.Client(timeout=15.0) as client:
                response = client.post("http://localhost:11434/api/generate", json=payload)
                response.raise_for_status()

                result = response.json()
                text = result.get("response", "").strip()

                logger.info(f"Ollama response: {text[:50]}")
                return text

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "I'm having technical difficulties. How else can I help?"

def convert_audio_for_asterisk(input_wav):
    """Convert to 8kHz mono for Asterisk"""
    try:
        timestamp = int(time.time())
        output_path = f"/var/lib/asterisk/sounds/custom/tts_{timestamp}.wav"

        # Ensure directory exists
        os.makedirs("/var/lib/asterisk/sounds/custom", exist_ok=True)
        logger.info(f"Converting {input_wav} to {output_path}")

        # Convert with sox
        sox_cmd = ['sox', input_wav, '-r', '8000', '-c', '1', '-b', '16', output_path]
        logger.info(f"Sox command: {' '.join(sox_cmd)}")

        result = subprocess.run(sox_cmd, capture_output=True, text=True, timeout=10)

        logger.info(f"Sox result: returncode={result.returncode}")
        if result.stdout:
            logger.info(f"Sox stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"Sox stderr: {result.stderr}")

        if result.returncode == 0 and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            os.chmod(output_path, 0o644)
            logger.info(f"Audio converted successfully: {output_path} ({file_size} bytes)")
            return f"custom/tts_{timestamp}"  # Return without .wav extension
        else:
            logger.error(f"Sox conversion failed: returncode={result.returncode}, stderr={result.stderr}")
            return None

    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        import traceback
        logger.error(f"Conversion traceback: {traceback.format_exc()}")
        return None

def main():
    """Main AGI handler"""
    try:
        logger.info("=== Simple AGI VoiceBot Starting ===")

        # Initialize AGI
        agi = SimpleAGI()
        caller_id = agi.env.get('agi_callerid', 'Unknown')
        logger.info(f"Call from: {caller_id}")

        # Answer call
        if not agi.answer():
            logger.error("Failed to answer")
            return

        agi.sleep(1)
        agi.verbose("Simple VoiceBot Active")

        # Initialize components
        tts = DirectTTSClient()
        asr = DirectASRClient()
        ollama = SimpleOllamaClient()

        # Send greeting
        logger.info("Generating greeting...")
        greeting_text = "Hello, thank you for calling NETOVO. I'm Alexis. How can I help you?"

        tts_file = tts.synthesize(greeting_text)

        if tts_file and os.path.exists(tts_file):
            asterisk_file = convert_audio_for_asterisk(tts_file)

            # Cleanup TTS file
            try:
                os.unlink(tts_file)
            except:
                pass

            if asterisk_file:
                success = agi.stream_file(asterisk_file)
                logger.info(f"Greeting played: {success}")
            else:
                logger.error("Audio conversion failed")
                agi.stream_file("hello")  # Fallback
        else:
            logger.error("TTS greeting failed")
            agi.stream_file("hello")  # Fallback

        # Simple conversation loop
        for turn in range(3):
            logger.info(f"Conversation turn {turn + 1}")

            # Record user input
            record_file = f"/var/spool/asterisk/monitor/user_{int(time.time())}"

            logger.info("Recording user...")
            if agi.record_file(record_file):
                wav_file = f"{record_file}.wav"

                if os.path.exists(wav_file):
                    file_size = os.path.getsize(wav_file)
                    logger.info(f"Recording: {file_size} bytes")

                    if file_size > 1000:
                        # Transcribe
                        transcript = asr.transcribe_file(wav_file)

                        if transcript:
                            logger.info(f"User said: {transcript}")

                            # Check for exit
                            if any(word in transcript.lower() for word in ['bye', 'goodbye', 'thank you']):
                                response = "Thank you for calling NETOVO. Have a great day!"
                            else:
                                # Get AI response
                                response = ollama.generate(transcript)
                        else:
                            response = "I didn't catch that. Could you repeat?"
                    else:
                        response = "I didn't hear anything. Could you speak up?"

                    # Cleanup recording
                    try:
                        os.unlink(wav_file)
                    except:
                        pass
                else:
                    response = "Recording failed. Let me try again."
            else:
                response = "I'm having trouble hearing you."

            # Speak response
            logger.info(f"Responding: {response[:30]}...")

            tts_file = tts.synthesize(response)
            if tts_file and os.path.exists(tts_file):
                asterisk_file = convert_audio_for_asterisk(tts_file)

                try:
                    os.unlink(tts_file)
                except:
                    pass

                if asterisk_file:
                    agi.stream_file(asterisk_file)
                else:
                    agi.stream_file("demo-thanks")  # Fallback
            else:
                agi.stream_file("demo-thanks")  # Fallback

            # Check for exit
            if 'thank you' in response.lower() or 'great day' in response.lower():
                break

            agi.sleep(1)

        # End call
        logger.info("Ending call")
        agi.sleep(1)
        agi.hangup()

        logger.info("=== Simple VoiceBot completed ===")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        try:
            agi = SimpleAGI()
            agi.answer()
            agi.verbose("VoiceBot error")
            agi.sleep(1)
            agi.hangup()
        except:
            pass

if __name__ == "__main__":
    main()
