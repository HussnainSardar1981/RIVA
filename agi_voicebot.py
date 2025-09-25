#!/usr/bin/env python3
"""
NETOVO Voice Bot AGI Script - Clean Synchronous Version
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

# CRITICAL: Check if we're in AGI environment FIRST
if sys.stdin.isatty():
    print("ERROR: This script must be called from Asterisk AGI", file=sys.stderr)
    sys.exit(0)

# Use pyst2 AGI library
try:
    from asterisk.agi import AGI
except ImportError:
    print("ERROR: pyst2 module not found", file=sys.stderr)
    sys.exit(0)

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from riva_client import RivaASRClient, RivaTTSClient
    from ollama_client import OllamaClient
    from audio_processing import AudioProcessor
    from conversation_context import ConversationContext
except ImportError as e:
    print(f"ERROR: Failed to import voice bot modules: {e}", file=sys.stderr)
    sys.exit(0)

class NetovoVoiceBot:
    """NETOVO Voice Bot AGI Handler - Clean Synchronous Version"""

    def __init__(self):
        # Use pyst2 AGI library
        self.agi = AGI()

        # ANSWER IMMEDIATELY
        self.agi.answer()
        self.agi.verbose("NETOVO Voice Bot answered call", 1)

        self.config = {
            'riva_server': os.getenv("RIVA_SERVER", "localhost:50051"),
            'ollama_url': os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            'ollama_model': os.getenv("OLLAMA_MODEL", "orca2:7b"),
            'telephony_rate': int(os.getenv("TELEPHONY_SAMPLE_RATE", "8000")),
            'asr_rate': int(os.getenv("ASR_SAMPLE_RATE", "16000")),
            'tts_rate': int(os.getenv("TTS_SAMPLE_RATE", "22050")),
        }

        # Initialize components
        self.riva_asr = None
        self.riva_tts = None
        self.ollama_client = None
        self.audio_processor = None
        self.conversation_context = None

        # Call info
        self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
        self.extension = self.agi.env.get('agi_extension', 'Unknown')

    def initialize_components(self) -> bool:
        """Initialize all voice bot components"""
        try:
            self.agi.verbose("Initializing RIVA and Ollama components...", 1)

            # Initialize RIVA TTS (most important for greeting)
            self.riva_tts = RivaTTSClient(self.config['riva_server'])
            if not self.riva_tts.connect():
                self.agi.verbose("RIVA TTS failed to connect", 1)
                return False

            # Initialize other components
            self.riva_asr = RivaASRClient(self.config['riva_server'])
            self.riva_asr.connect()  # Don't fail if ASR doesn't work

            self.ollama_client = OllamaClient(
                base_url=self.config['ollama_url'],
                model=self.config['ollama_model']
            )

            self.audio_processor = AudioProcessor()
            self.conversation_context = ConversationContext()

            self.agi.verbose("Voice bot components initialized successfully", 1)
            return True

        except Exception as e:
            self.agi.verbose(f"Component initialization error: {str(e)}", 1)
            return False

    def handle_call(self):
        """Main call handling logic"""
        try:
            self.agi.verbose(f"Handling call from {self.caller_id}", 1)

            # Initialize components after answering
            if not self.initialize_components():
                self.agi.verbose("Using fallback mode - components failed", 1)
                self.send_fallback_greeting()
                self.agi.hangup()
                return

            # Send RIVA TTS greeting
            self.send_riva_greeting()

            # Simple completion for now
            self.agi.verbose("Call completed successfully", 1)

        except Exception as e:
            self.agi.verbose(f"Call handling error: {str(e)}", 1)
        finally:
            self.agi.hangup()

    def send_riva_greeting(self):
        """Send RIVA TTS greeting"""
        try:
            greeting_text = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"

            self.agi.verbose("Generating RIVA TTS greeting...", 1)

            # Generate TTS
            tts_file = self.riva_tts.synthesize_speech_to_file(
                greeting_text,
                sample_rate=self.config['telephony_rate']
            )

            if tts_file and os.path.exists(tts_file):
                # Play the greeting (Asterisk needs filename without .wav)
                base_filename = tts_file.replace('.wav', '')
                self.agi.stream_file(base_filename)
                self.agi.verbose("RIVA greeting played successfully", 1)

                # Cleanup
                try:
                    os.unlink(tts_file)
                except:
                    pass
            else:
                self.agi.verbose("TTS file generation failed, using fallback", 1)
                self.send_fallback_greeting()

        except Exception as e:
            self.agi.verbose(f"RIVA greeting error: {str(e)}", 1)
            self.send_fallback_greeting()

    def send_fallback_greeting(self):
        """Fallback greeting using Asterisk sounds"""
        try:
            self.agi.stream_file("hello")
            self.agi.stream_file("thank-you-for-calling")
        except:
            self.agi.stream_file("beep")

def main():
    """Main entry point"""
    try:
        # Create and run voice bot
        voice_bot = NetovoVoiceBot()
        voice_bot.handle_call()

    except Exception as e:
        # Write to stderr for debugging
        print(f"AGI Error: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    main()
