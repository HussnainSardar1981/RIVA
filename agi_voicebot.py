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

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from riva_client import RivaASRClient, RivaTTSClient
from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
from audio_processing import AudioProcessor
from conversation_context import ConversationContext

logger = structlog.get_logger()

class AGIInterface:
    """Simple AGI interface for Asterisk communication"""

    def __init__(self):
        self.env = {}
        self._parse_agi_environment()

    def _parse_agi_environment(self):
        """Parse AGI environment variables from stdin"""
        while True:
            line = sys.stdin.readline().strip()
            if not line:
                break
            if ':' in line:
                key, value = line.split(':', 1)
                self.env[key.strip()] = value.strip()

        logger.info("AGI Environment parsed", env=self.env)

    def execute(self, command: str) -> str:
        """Execute AGI command and return result"""
        sys.stdout.write(f"{command}\n")
        sys.stdout.flush()

        result = sys.stdin.readline().strip()
        logger.debug("AGI Command executed", command=command, result=result)
        return result

    def answer(self):
        """Answer the call"""
        return self.execute("ANSWER")

    def hangup(self):
        """Hang up the call"""
        return self.execute("HANGUP")

    def stream_file(self, filename: str, escape_digits: str = ""):
        """Stream audio file to caller"""
        return self.execute(f'STREAM FILE "{filename}" "{escape_digits}"')

    def record_file(self, filename: str, format: str = "wav", escape_digits: str = "#", timeout: int = 10000):
        """Record audio from caller"""
        return self.execute(f'RECORD FILE "{filename}" "{format}" "{escape_digits}" {timeout}')

    def get_variable(self, variable: str) -> str:
        """Get channel variable"""
        result = self.execute(f'GET VARIABLE "{variable}"')
        if result.startswith("200 result=1"):
            return result.split('(')[1].split(')')[0]
        return ""

    def set_variable(self, variable: str, value: str):
        """Set channel variable"""
        return self.execute(f'SET VARIABLE "{variable}" "{value}"')

    def verbose(self, message: str, level: int = 1):
        """Send verbose message to Asterisk"""
        return self.execute(f'VERBOSE "{message}" {level}')

class NetovoAGIVoiceBot:
    """NETOVO Voice Bot AGI Handler"""

    def __init__(self):
        self.agi = AGIInterface()
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

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment"""
        return {
            'riva_server': os.getenv("RIVA_SERVER", "localhost:50051"),
            'ollama_url': os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            'ollama_model': os.getenv("OLLAMA_MODEL", "orca2:7b"),
            'telephony_rate': int(os.getenv("TELEPHONY_SAMPLE_RATE", "8000")),
            'asr_rate': int(os.getenv("ASR_SAMPLE_RATE", "16000")),
            'tts_rate': int(os.getenv("TTS_SAMPLE_RATE", "22050")),
            'max_tokens': int(os.getenv("MAX_TOKENS", "50")),
            'response_timeout': float(os.getenv("RESPONSE_TIMEOUT", "30.0"))
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
        """Main call handling logic"""
        try:
            logger.info("Handling incoming call",
                       caller_id=self.caller_id,
                       extension=self.extension)

            # Answer the call
            self.agi.verbose(f"NETOVO Voice Bot answering call from {self.caller_id}")
            self.agi.answer()

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

        # Create and initialize voice bot
        voice_bot = NetovoAGIVoiceBot()

        if not await voice_bot.initialize():
            logger.error("Voice bot initialization failed")
            sys.exit(1)

        # Handle the call
        await voice_bot.handle_call()

        logger.info("NETOVO Voice Bot AGI Script completed successfully")

    except Exception as e:
        logger.error("AGI script fatal error", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
