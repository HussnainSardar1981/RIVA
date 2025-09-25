#!/usr/bin/env python3
"""
NETOVO Voice Bot AGI Script
Integrates with Asterisk using AGI for seamless call handling
"""

import asyncio
import os
import sys
import tempfile
import wave
import struct
from pathlib import Path
from typing import Optional, Dict, Any
import structlog

# CRITICAL: Check if we're in AGI environment FIRST
if sys.stdin.isatty():
    print("WARNING: Running in test mode - setting default environment", file=sys.stderr)
    # Set default environment variables for testing
    os.environ.setdefault("TELEPHONY_SAMPLE_RATE", "8000")
    os.environ.setdefault("ASR_SAMPLE_RATE", "16000")
    os.environ.setdefault("TTS_SAMPLE_RATE", "22050")
    os.environ.setdefault("MAX_TOKENS", "50")
    os.environ.setdefault("RESPONSE_TIMEOUT", "30.0")
    os.environ.setdefault("RIVA_SERVER", "localhost:50051")
    os.environ.setdefault("OLLAMA_URL", "http://127.0.0.1:11434")
    os.environ.setdefault("OLLAMA_MODEL", "orca2:7b")

# Use pyst2 AGI library (like working.py)
try:
    from asterisk.agi import AGI
except ImportError:
    print("ERROR: pyst2 module not found", file=sys.stderr)
    sys.exit(0)

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from riva_client import RivaASRClient, RivaTTSClient
from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
from audio_processing import AudioProcessor
from conversation_context import ConversationContext

logger = structlog.get_logger()

# Using pyst2 AGI library instead of custom AGI class

class NetovoAGIVoiceBot:
    """NETOVO Voice Bot AGI Handler"""

    def __init__(self):
        # Use pyst2 AGI library (like working.py)
        self.agi = AGI()

        # ANSWER IMMEDIATELY (your working pattern)
        self.agi.answer()
        self.agi.verbose("NETOVO Voice Bot answered call", 1)

        self.config = self._load_config()

        # Initialize components
        self.riva_asr = None
        self.riva_tts = None
        self.ollama_client = None
        self.audio_processor = None
        self.conversation_context = None

        # Call info
        self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
        self.extension = self.agi.env.get('agi_extension', 'Unknown')

    def _safe_int(self, key: str, default: int) -> int:
        """Safely convert environment variable to int"""
        val = os.getenv(key)
        try:
            return int(val) if val else default
        except (ValueError, TypeError):
            return default

    def _safe_float(self, key: str, default: float) -> float:
        """Safely convert environment variable to float"""
        val = os.getenv(key)
        try:
            return float(val) if val else default
        except (ValueError, TypeError):
            return default

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment with safe parsing"""
        return {
            'riva_server': os.getenv("RIVA_SERVER", "localhost:50051"),
            'ollama_url': os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            'ollama_model': os.getenv("OLLAMA_MODEL", "orca2:7b"),
            'telephony_rate': self._safe_int("TELEPHONY_SAMPLE_RATE", 8000),
            'asr_rate': self._safe_int("ASR_SAMPLE_RATE", 16000),
            'tts_rate': self._safe_int("TTS_SAMPLE_RATE", 22050),
            'max_tokens': self._safe_int("MAX_TOKENS", 50),
            'response_timeout': self._safe_float("RESPONSE_TIMEOUT", 30.0)
        }

    async def initialize(self) -> bool:
        """Initialize all voice bot components"""
        try:
            logger.info("Initializing NETOVO Voice Bot components...")

            # Initialize RIVA ASR
            self.riva_asr = RivaASRClient(self.config['riva_server'])
            if not await self.riva_asr.initialize():
                logger.error("Failed to initialize RIVA ASR")
                return False

            # Initialize RIVA TTS
            self.riva_tts = RivaTTSClient(self.config['riva_server'])
            if not await self.riva_tts.initialize():
                logger.error("Failed to initialize RIVA TTS")
                return False

            # Initialize Ollama
            self.ollama_client = OllamaClient(
                base_url=self.config['ollama_url'],
                model=self.config['ollama_model']
            )
            if not await self.ollama_client.initialize():
                logger.error("Failed to initialize Ollama client")
                return False

            # Initialize audio processor
            self.audio_processor = AudioProcessor()

            # Initialize conversation context
            self.conversation_context = ConversationContext()

            logger.info("All NETOVO Voice Bot components initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize voice bot", error=str(e))
            return False

    async def handle_call(self):
        """Main call handling logic - call already answered in main()"""
        try:
            logger.info("Handling incoming call",
                       caller_id=self.caller_id,
                       extension=self.extension)

            # Call is already answered in main() - skip duplicate answer
            self.agi.verbose(f"NETOVO Voice Bot ready for call from {self.caller_id}")

            # Send greeting
            await self.send_greeting()

            # Main conversation loop
            await self.conversation_loop()

        except Exception as e:
            logger.error("Error handling call", error=str(e))
            self.agi.verbose(f"Call handling error: {str(e)}")
        finally:
            self.agi.hangup()

    async def send_greeting(self):
        """Send initial greeting to caller"""
        try:
            greeting_text = f"Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"

            logger.info("Generating greeting TTS")
            tts_file = self.riva_tts.synthesize_speech_to_file(
                greeting_text,
                sample_rate=self.config['telephony_rate']  # 8kHz for telephony
            )

            if tts_file and os.path.exists(tts_file):
                # Convert to Asterisk-compatible format
                asterisk_file = await self._convert_audio_for_asterisk(tts_file)

                if asterisk_file:
                    # Play greeting
                    self.agi.verbose("Playing greeting message")
                    result = self.agi.stream_file(asterisk_file.replace('.wav', ''))
                    logger.info("Greeting played", result=result)

                    # Cleanup
                    try:
                        os.unlink(asterisk_file)
                        os.unlink(tts_file)
                    except:
                        pass

        except Exception as e:
            logger.error("Error sending greeting", error=str(e))

    async def conversation_loop(self):
        """Main conversation processing loop"""
        conversation_active = True
        silence_count = 0

        while conversation_active and silence_count < 3:
            try:
                # Record user input
                logger.info("Recording user input")
                self.agi.verbose("Please speak after the tone")

                # Create temporary file for recording
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    record_file = tmp_file.name

                # Record audio (10 second timeout, end on #)
                result = self.agi.record_file(record_file.replace('.wav', ''), 'wav', '#', 10000)

                if not os.path.exists(record_file) or os.path.getsize(record_file) < 1000:
                    logger.warning("No audio recorded or file too small")
                    silence_count += 1
                    continue

                # Process the recorded audio
                await self.process_user_input(record_file)
                silence_count = 0  # Reset silence counter on successful input

                # Check for conversation end keywords
                if self.conversation_context.should_end_conversation():
                    logger.info("Conversation end detected")
                    await self.send_goodbye()
                    conversation_active = False

            except Exception as e:
                logger.error("Error in conversation loop", error=str(e))
                silence_count += 1

        if silence_count >= 3:
            logger.info("Ending call due to extended silence")
            await self.send_timeout_message()

    async def process_user_input(self, audio_file: str):
        """Process user audio input and generate response"""
        try:
            # Convert audio for ASR (8kHz -> 16kHz)
            asr_audio_file = await self._convert_audio_for_asr(audio_file)

            if not asr_audio_file:
                logger.error("Failed to convert audio for ASR")
                return

            # Transcribe speech
            logger.info("Transcribing user speech")
            transcript = self.riva_asr.transcribe_file(asr_audio_file)

            if not transcript or len(transcript.strip()) < 2:
                logger.warning("No meaningful transcript generated")
                await self.send_clarification_request()
                return

            logger.info("User transcript", transcript=transcript)
            self.agi.verbose(f"User said: {transcript}")

            # Add to conversation context
            self.conversation_context.add_user_message(transcript)

            # Generate AI response
            logger.info("Generating AI response")
            response = self.ollama_client.generate(
                transcript,
                system_prompt=self.conversation_context.get_context_string(),
                max_tokens=self.config['max_tokens']
            )

            if not response:
                logger.error("Failed to generate AI response")
                await self.send_error_response()
                return

            logger.info("AI response generated", response=response)
            self.agi.verbose(f"AI response: {response}")

            # Add AI response to context
            self.conversation_context.add_assistant_message(response)

            # Convert response to speech
            await self.send_ai_response(response)

            # Cleanup temp files
            try:
                os.unlink(audio_file)
                os.unlink(asr_audio_file)
            except:
                pass

        except Exception as e:
            logger.error("Error processing user input", error=str(e))
            await self.send_error_response()

    async def send_ai_response(self, response_text: str):
        """Convert AI response to speech and play to caller"""
        try:
            logger.info("Generating AI response TTS")
            tts_file = self.riva_tts.synthesize_speech_to_file(
                response_text,
                sample_rate=self.config['telephony_rate']
            )

            if tts_file and os.path.exists(tts_file):
                # Convert to Asterisk format
                asterisk_file = await self._convert_audio_for_asterisk(tts_file)

                if asterisk_file:
                    # Play response
                    result = self.agi.stream_file(asterisk_file.replace('.wav', ''))
                    logger.info("AI response played", result=result)

                    # Cleanup
                    try:
                        os.unlink(asterisk_file)
                        os.unlink(tts_file)
                    except:
                        pass

        except Exception as e:
            logger.error("Error sending AI response", error=str(e))

    async def send_clarification_request(self):
        """Ask user to repeat or clarify"""
        message = "I didn't catch that. Could you please repeat what you said?"
        await self.send_ai_response(message)

    async def send_error_response(self):
        """Send error message to user"""
        message = "I'm sorry, I'm having technical difficulties. Please try again or call back later."
        await self.send_ai_response(message)

    async def send_goodbye(self):
        """Send goodbye message"""
        message = "Thank you for calling NETOVO. Have a great day! Goodbye."
        await self.send_ai_response(message)

    async def send_timeout_message(self):
        """Send timeout message"""
        message = "Thank you for calling NETOVO. Goodbye!"
        await self.send_ai_response(message)

    async def _convert_audio_for_asr(self, input_file: str) -> Optional[str]:
        """Convert telephony audio (8kHz) to ASR format (16kHz)"""
        try:
            with tempfile.NamedTemporaryFile(suffix='_asr.wav', delete=False) as tmp_file:
                output_file = tmp_file.name

            # Use audio processor to resample
            success = self.audio_processor.convert_wav_file(
                input_file, output_file,
                target_rate=self.config['asr_rate']
            )

            return output_file if success else None

        except Exception as e:
            logger.error("Error converting audio for ASR", error=str(e))
            return None

    async def _convert_audio_for_asterisk(self, input_file: str) -> Optional[str]:
        """Convert TTS audio to Asterisk-compatible format (8kHz, 16-bit PCM)"""
        try:
            with tempfile.NamedTemporaryFile(suffix='_asterisk.wav', delete=False) as tmp_file:
                output_file = tmp_file.name

            # Use audio processor to convert to telephony format
            success = self.audio_processor.convert_wav_file(
                input_file, output_file,
                target_rate=self.config['telephony_rate']
            )

            return output_file if success else None

        except Exception as e:
            logger.error("Error converting audio for Asterisk", error=str(e))
            return None

    def initialize_sync(self) -> bool:
        """Initialize all voice bot components - synchronous version"""
        try:
            logger.info("Initializing NETOVO Voice Bot components...")

            # Initialize RIVA ASR
            self.riva_asr = RivaASRClient(self.config['riva_server'])
            if not self.riva_asr.connect():
                logger.error("Failed to initialize RIVA ASR")
                return False

            # Initialize RIVA TTS
            self.riva_tts = RivaTTSClient(self.config['riva_server'])
            if not self.riva_tts.connect():
                logger.error("Failed to initialize RIVA TTS")
                return False

            # Initialize Ollama
            self.ollama_client = OllamaClient(
                base_url=self.config['ollama_url'],
                model=self.config['ollama_model']
            )
            if not self.ollama_client.test_connection():
                logger.error("Failed to initialize Ollama client")
                return False

            # Initialize audio processor
            self.audio_processor = AudioProcessor()

            # Initialize conversation context
            self.conversation_context = ConversationContext()

            logger.info("All NETOVO Voice Bot components initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize voice bot", error=str(e))
            return False

    def handle_call_sync(self):
        """Main call handling logic - synchronous version"""
        try:
            logger.info("Handling incoming call",
                       caller_id=self.caller_id,
                       extension=self.extension)

            # Call is already answered in main() - skip duplicate answer
            self.agi.verbose(f"NETOVO Voice Bot ready for call from {self.caller_id}")

            # Send greeting
            self.send_greeting_sync()

            # Simple response for now
            self.agi.verbose("NETOVO Voice Bot conversation completed")

        except Exception as e:
            logger.error("Error handling call", error=str(e))
            self.agi.verbose(f"Call handling error: {str(e)}")
        finally:
            self.agi.hangup()

    def send_greeting_sync(self):
        """Send initial greeting to caller - synchronous version"""
        try:
            greeting_text = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant."

            logger.info("Generating greeting TTS")
            tts_file = self.riva_tts.synthesize_speech_to_file(
                greeting_text,
                sample_rate=self.config['telephony_rate']
            )

            if tts_file and os.path.exists(tts_file):
                # Play greeting (remove .wav extension for Asterisk)
                base_file = tts_file.replace('.wav', '')
                self.agi.verbose("Playing greeting message")
                self.agi.stream_file(base_file)
                logger.info("Greeting played successfully")

                # Cleanup
                try:
                    os.unlink(tts_file)
                except:
                    pass

        except Exception as e:
            logger.error("Error sending greeting", error=str(e))
            # Fallback to simple message
            self.agi.stream_file("demo-thanks")

async def main():
    """Main AGI script entry point"""
    try:
        # Configure logging for AGI
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        logger.info("Starting NETOVO Voice Bot AGI Script")

        # Create voice bot instance (already answers call in __init__)
        voice_bot = NetovoAGIVoiceBot()

        voice_bot.agi.verbose("Call answered, initializing voice bot components...", 1)

        # Now initialize components after answering
        if not await voice_bot.initialize():
            voice_bot.agi.verbose("Voice bot initialization failed")
            # Play error message before hanging up
            voice_bot.agi.stream_file("demo-unavail")
            logger.error("Voice bot initialization failed")
            voice_bot.agi.hangup()
            sys.exit(1)

        # Handle the call
        await voice_bot.handle_call()

        logger.info("NETOVO Voice Bot AGI Script completed successfully")

    except Exception as e:
        logger.error("AGI script fatal error", error=str(e))
        sys.exit(1)

def main_sync():
    """Synchronous main for pyst2 AGI"""
    try:
        # Configure logging for AGI
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        logger.info("Starting NETOVO Voice Bot AGI Script")

        # Create voice bot instance (already answers call in __init__)
        voice_bot = NetovoAGIVoiceBot()
        voice_bot.agi.verbose("Call answered, initializing voice bot components...", 1)

        # Initialize synchronously (remove async)
        if not voice_bot.initialize_sync():
            voice_bot.agi.verbose("Voice bot initialization failed")
            voice_bot.agi.stream_file("demo-unavail")
            logger.error("Voice bot initialization failed")
            voice_bot.agi.hangup()
            return

        # Handle the call synchronously
        voice_bot.handle_call_sync()

        logger.info("NETOVO Voice Bot AGI Script completed successfully")

    except Exception as e:
        logger.error("AGI script fatal error", error=str(e))

if __name__ == "__main__":
    main_sync()


