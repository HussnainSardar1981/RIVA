#!/usr/bin/env python3
"""
NETOVO RIVA Voice Bot - Raw AGI Version
No pyst2 dependency - direct Asterisk communication
"""

import sys
import os
import tempfile
from pathlib import Path
from shutil import copyfile

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from riva_client import RivaASRClient, RivaTTSClient
    from ollama_client import OllamaClient
    from audio_processing import AudioProcessor
    from conversation_context import ConversationContext
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

class RawAGI:
    """Raw AGI interface without pyst2"""

    def __init__(self):
        self.env = self._parse_environment()

    def _parse_environment(self):
        """Parse AGI environment from stdin"""
        env = {}
        while True:
            try:
                line = sys.stdin.readline().strip()
                if not line:
                    break
                if ':' in line:
                    key, value = line.split(':', 1)
                    env[key.strip()] = value.strip()
            except:
                break
        return env

    def command(self, cmd):
        """Send AGI command"""
        try:
            sys.stdout.write(f"{cmd}\n")
            sys.stdout.flush()
            result = sys.stdin.readline().strip()
            return result
        except:
            return "ERROR"

    def verbose(self, message, level=1):
        """Send verbose message"""
        return self.command(f'VERBOSE "{message}" {level}')

    def answer(self):
        """Answer call"""
        return self.command("ANSWER")

    def hangup(self):
        """Hangup call"""
        return self.command("HANGUP")

    def stream_file(self, filename):
        """Stream audio file"""
        return self.command(f'STREAM FILE "{filename}" ""')

    def record_file(self, filename, format="wav", escape_digits="#", timeout=10000):
        """Record audio"""
        return self.command(f'RECORD FILE "{filename}" "{format}" "{escape_digits}" {timeout}')

class NetovoRivaVoiceBot:
    """NETOVO Voice Bot with Raw AGI"""

    def __init__(self):
        self.agi = RawAGI()
        self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')

        # Configuration
        self.config = {
            'riva_server': 'localhost:50051',
            'ollama_url': 'http://127.0.0.1:11434',
            'ollama_model': 'orca2:7b',
            'telephony_rate': 8000,
            'tts_rate': 22050
        }

        # Components
        self.riva_tts = None
        self.riva_asr = None
        self.ollama_client = None

        # Asterisk sounds directory
        self.sounds_dir = "/var/lib/asterisk/sounds/custom"

    def _play_wav_file(self, wav_path):
        """Copy WAV file to Asterisk sounds directory and play it"""
        try:
            # Create sounds directory if it doesn't exist
            os.makedirs(self.sounds_dir, exist_ok=True)

            # Get basename without extension
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            dst_path = os.path.join(self.sounds_dir, f"{base_name}.wav")

            # Copy file to sounds directory
            copyfile(wav_path, dst_path)

            # Set proper permissions
            os.chmod(dst_path, 0o644)

            # Play file (Asterisk expects path relative to sounds dir, no extension)
            return self.agi.stream_file(f"custom/{base_name}")

        except Exception as e:
            self.agi.verbose(f"Error playing WAV file: {str(e)}")
            return "ERROR"

    def initialize_components(self):
        """Initialize RIVA and Ollama"""
        try:
            self.agi.verbose("Initializing RIVA TTS...")

            # Initialize RIVA TTS
            self.riva_tts = RivaTTSClient(self.config['riva_server'])
            if not self.riva_tts.connect():
                self.agi.verbose("RIVA TTS connection failed")
                return False

            # Initialize other components
            self.riva_asr = RivaASRClient(self.config['riva_server'])
            self.riva_asr.connect()

            self.ollama_client = OllamaClient(
                base_url=self.config['ollama_url'],
                model=self.config['ollama_model']
            )

            self.agi.verbose("RIVA components initialized successfully")
            return True

        except Exception as e:
            self.agi.verbose(f"Component initialization failed: {str(e)}")
            return False

    def send_tts_greeting(self):
        """Send RIVA TTS greeting"""
        try:
            greeting = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"

            self.agi.verbose("Generating RIVA TTS greeting...")

            # Generate TTS
            tts_file = self.riva_tts.synthesize_speech_to_file(
                greeting,
                sample_rate=self.config['telephony_rate']
            )

            if tts_file and os.path.exists(tts_file):
                # Play greeting using proper audio handler
                result = self._play_wav_file(tts_file)
                if result != "ERROR":
                    self.agi.verbose("RIVA TTS greeting played successfully")

                # Cleanup original temp file
                try:
                    os.unlink(tts_file)
                except:
                    pass

                return result != "ERROR"
            else:
                self.agi.verbose("TTS generation failed")
                return False

        except Exception as e:
            self.agi.verbose(f"TTS greeting error: {str(e)}")
            return False

    def handle_call(self):
        """Main call handling"""
        try:
            self.agi.verbose(f"NETOVO Voice Bot handling call from {self.caller_id}")

            # Answer call immediately
            self.agi.answer()
            self.agi.verbose("Call answered successfully")

            # Initialize components
            if not self.initialize_components():
                self.agi.verbose("Using fallback mode - playing demo message")
                self.agi.stream_file("demo-thanks")
                # Keep call alive for a bit so caller can hear the message
                import time
                time.sleep(3)
                self.agi.hangup()
                return

            # Send RIVA TTS greeting
            if not self.send_tts_greeting():
                self.agi.verbose("TTS failed, using fallback greeting")
                self.agi.stream_file("hello")

            # TODO: Add conversation loop here
            # For now, keep call alive for 10 seconds to demonstrate working audio
            self.agi.verbose("NETOVO Voice Bot greeting completed - keeping call alive")
            import time
            time.sleep(10)

            # Graceful hangup after demo period
            self.agi.verbose("Demo period complete - hanging up call")
            self.agi.hangup()

        except Exception as e:
            self.agi.verbose(f"Call handling error: {str(e)}")
            # Only hangup on error
            self.agi.hangup()

def main():
    """Main entry point"""
    try:
        print("NETOVO RIVA Voice Bot starting", file=sys.stderr)

        # Create and run voice bot
        voice_bot = NetovoRivaVoiceBot()
        voice_bot.handle_call()

        print("NETOVO RIVA Voice Bot completed", file=sys.stderr)

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()



    
