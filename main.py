"""
NETOVO Professional Voice Bot Application
Production-ready version with proper error handling and configuration
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import structlog
from dotenv import load_dotenv

# Import working implementations
from riva_proto_simple import SimpleRivaASR, SimpleRivaTTS
from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
from audio_processing import AudioProcessor, AudioBuffer
from conversation_context import ConversationContext

# Load environment variables from .env file
ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(ENV_FILE)

# Configure structured logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
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

logger = structlog.get_logger()

class VoiceBotConfig:
    """Configuration class for Voice Bot"""

    def __init__(self):
        # Ollama settings
        self.ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "orca2:7b")

        # Riva settings
        self.riva_server = os.getenv("RIVA_SERVER", "localhost:50051")

        # Audio settings
        self.telephony_rate = int(os.getenv("TELEPHONY_SAMPLE_RATE", "8000"))
        self.asr_rate = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
        self.tts_rate = int(os.getenv("TTS_SAMPLE_RATE", "22050"))

        # Generation settings
        self.max_tokens = int(os.getenv("MAX_TOKENS", "50"))
        self.response_timeout = float(os.getenv("RESPONSE_TIMEOUT", "30.0"))

        # Validate required settings
        self._validate()

    def _validate(self):
        """Validate configuration"""
        if not self.ollama_url:
            raise ValueError("OLLAMA_URL is required")
        if not self.ollama_model:
            raise ValueError("OLLAMA_MODEL is required")
        if not self.riva_server:
            raise ValueError("RIVA_SERVER is required")

class VoiceBot:
    """Main Voice Bot orchestrator with production-ready implementation"""

    def __init__(self, config: VoiceBotConfig):
        """Initialize components with configuration"""
        self.config = config
        self.running = False

        # Initialize components to None - we'll create them in initialize()
        self.ollama: Optional[OllamaClient] = None
        self.riva_asr: Optional[SimpleRivaASR] = None
        self.riva_tts: Optional[SimpleRivaTTS] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.conversation: Optional[ConversationContext] = None

        logger.info("VoiceBot constructor completed", config=self._config_summary())

    def _config_summary(self) -> dict:
        """Return safe config summary for logging"""
        return {
            "ollama_url": self.config.ollama_url,
            "ollama_model": self.config.ollama_model,
            "riva_server": self.config.riva_server,
            "sample_rates": {
                "telephony": self.config.telephony_rate,
                "asr": self.config.asr_rate,
                "tts": self.config.tts_rate
            }
        }

    async def initialize(self) -> bool:
        """Initialize all components with proper error handling"""
        logger.info("Starting VoiceBot initialization...")

        try:
            # Step 1: Initialize Ollama client
            await self._initialize_ollama()

            # Step 2: Initialize Riva components
            await self._initialize_riva()

            # Step 3: Initialize other components
            await self._initialize_audio_and_conversation()

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error("Initialization failed", error=str(e), error_type=type(e).__name__)
            await self.cleanup()
            return False

    async def _initialize_ollama(self):
        """Initialize Ollama client with health checks"""
        logger.info("Initializing Ollama client...")

        self.ollama = OllamaClient(
            base_url=self.config.ollama_url,
            model=self.config.ollama_model
        )

        # Test health check
        if not self.ollama.health_check():
            logger.error("Ollama health check failed")
            await self._debug_ollama_connection()
            raise ConnectionError("Ollama health check failed")

        logger.info("Ollama health check passed")

        # Test generation
        try:
            test_response = self.ollama.generate("Hello", max_tokens=5)
            logger.info("Ollama generation test successful", response=test_response[:50])
        except Exception as e:
            logger.error("Ollama generation test failed", error=str(e))
            raise ConnectionError(f"Ollama generation failed: {e}")

    async def _debug_ollama_connection(self):
        """Debug Ollama connection issues"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.config.ollama_url}/api/tags")
                logger.error("Direct API test", status=response.status_code, url=self.config.ollama_url)
        except Exception as e:
            logger.error("Direct API test failed", error=str(e))

    async def _initialize_riva(self):
        """Initialize Riva ASR and TTS components"""
        logger.info("Initializing Riva components...")

        try:
            self.riva_asr = SimpleRivaASR(server_url=self.config.riva_server)
            self.riva_tts = SimpleRivaTTS(server_url=self.config.riva_server)
            logger.info("Riva components initialized")
        except Exception as e:
            logger.error("Riva initialization failed", error=str(e))
            raise ConnectionError(f"Riva initialization failed: {e}")

    async def _initialize_audio_and_conversation(self):
        """Initialize audio processor and conversation context"""
        logger.info("Initializing audio processor and conversation context...")

        try:
            self.audio_processor = AudioProcessor()
            self.conversation = ConversationContext()
            logger.info("Audio and conversation components initialized")
        except Exception as e:
            logger.error("Audio/conversation initialization failed", error=str(e))
            raise RuntimeError(f"Audio/conversation initialization failed: {e}")

    async def process_text_to_speech(self, text: str) -> str:
        """Convert text to speech and return WAV file path"""
        if not text.strip():
            logger.warning("Empty text provided for TTS")
            return ""

        try:
            logger.info("Processing TTS", text=text[:100])
            wav_path = self.riva_tts.synthesize(text)

            if wav_path and os.path.exists(wav_path):
                logger.info("TTS completed successfully", output=wav_path)
                return wav_path
            else:
                logger.error("TTS failed - no output file generated")
                return ""

        except Exception as e:
            logger.error("TTS processing failed", error=str(e))
            return ""

    async def process_speech_to_text(self, audio_file: str) -> str:
        """Convert speech file to text"""
        if not audio_file or not os.path.exists(audio_file):
            logger.error("Audio file does not exist", file=audio_file)
            return "Audio file not found"

        try:
            logger.info("Processing ASR", file=audio_file)
            transcript = self.riva_asr.transcribe_file(audio_file)

            if transcript:
                logger.info("ASR completed successfully", transcript=transcript[:100])
                return transcript
            else:
                logger.warning("ASR returned empty transcript")
                return "No speech detected"

        except Exception as e:
            logger.error("ASR processing failed", error=str(e))
            return "Speech processing error"

    async def get_llm_response(self, transcript: str) -> str:
        """Get response from Ollama LLM"""
        if not transcript.strip():
            return "I didn't hear anything. Could you please repeat?"

        try:
            logger.info("Generating LLM response", input=transcript[:100])

            response = self.ollama.generate(
                prompt=transcript,
                system_prompt=VOICE_BOT_SYSTEM_PROMPT,
                max_tokens=self.config.max_tokens
            )

            if response:
                # Add to conversation history
                self.conversation.add_turn(transcript, response)
                logger.info("LLM response generated successfully",
                           input_length=len(transcript),
                           output_length=len(response))
                return response
            else:
                logger.warning("LLM returned empty response")
                return "I'm processing your request. Please wait a moment."

        except Exception as e:
            logger.error("LLM processing failed", error=str(e))
            return "I'm sorry, I'm having trouble processing your request. Please try again."

    async def process_complete_pipeline(self, input_text: str) -> str:
        """Complete pipeline: Text -> TTS -> ASR -> LLM -> TTS"""
        if not input_text.strip():
            return "No input provided"

        logger.info("Starting complete pipeline", input=input_text[:100])
        temp_files = []

        try:
            # Step 1: Convert input text to speech
            logger.info("Pipeline Step 1: Text to Speech")
            tts_file = await self.process_text_to_speech(input_text)
            if not tts_file:
                return "TTS generation failed"
            temp_files.append(tts_file)

            # Step 2: Convert speech back to text (simulating phone input)
            logger.info("Pipeline Step 2: Speech to Text")
            transcript = await self.process_speech_to_text(tts_file)

            # Step 3: Get LLM response
            logger.info("Pipeline Step 3: LLM Processing")
            response = await self.get_llm_response(transcript)

            # Step 4: Convert response to speech
            logger.info("Pipeline Step 4: Response to Speech")
            response_file = await self.process_text_to_speech(response)
            if response_file:
                temp_files.append(response_file)

            logger.info("Complete pipeline successful",
                       original=input_text[:50],
                       transcript=transcript[:50],
                       response=response[:50])

            return response_file if response_file else response

        except Exception as e:
            logger.error("Pipeline processing failed", error=str(e))
            return "Pipeline error occurred"

        finally:
            # Cleanup temp files
            self._cleanup_temp_files(temp_files)

    def _cleanup_temp_files(self, temp_files: list):
        """Clean up temporary files"""
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug("Cleaned up temp file", file=temp_file)
            except Exception as e:
                logger.warning("Failed to cleanup temp file", file=temp_file, error=str(e))

    async def start_interactive_mode(self):
        """Start interactive voice bot mode"""
        logger.info("Starting Interactive Voice Bot Mode")
        print("\n🎙️  Voice Bot - Interactive Mode")
        print("Type 'quit', 'exit', or 'bye' to exit")
        print("-" * 50)

        while self.running:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    print("Please enter some text.")
                    continue

                print("🔄 Processing...")
                response = await self.process_complete_pipeline(user_input)
                print(f"🤖 Bot: {response}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error("Interactive mode error", error=str(e))
                print(f"❌ Error: {e}")

    async def run_tests(self):
        """Run comprehensive tests"""
        logger.info("Running Voice Bot Tests")

        test_cases = [
            "Hello, this is a test of the voice bot system",
            "How are you today?",
            "What services does NETOVO provide?",
            "Thank you for your help"
        ]

        print(f"\n🧪 Running {len(test_cases)} test cases...\n")

        passed = 0
        failed = 0

        for i, test_text in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: {test_text}")

            try:
                result = await self.process_complete_pipeline(test_text)
                if result and "error" not in result.lower():
                    print(f"✅ PASSED: {result}")
                    passed += 1
                else:
                    print(f"❌ FAILED: {result}")
                    failed += 1
            except Exception as e:
                print(f"❌ ERROR: {e}")
                failed += 1

            print()

        print(f"📊 Test Results: {passed} passed, {failed} failed")
        logger.info("Test run completed", passed=passed, failed=failed)

    async def start_server(self):
        """Start the voice bot server"""
        logger.info("Starting Voice Bot server...")
        self.running = True

        print(f"\n🎯 NETOVO Voice Bot v1.0")
        print("=" * 50)
        print("1. Interactive Mode")
        print("2. Run Tests")
        print("3. Exit")
        print("=" * 50)

        try:
            choice = input("Choose option (1-3): ").strip()

            if choice == "1":
                await self.start_interactive_mode()
            elif choice == "2":
                await self.run_tests()
            elif choice == "3":
                print("👋 Goodbye!")
            else:
                print("❌ Invalid choice. Exiting.")

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
        except Exception as e:
            logger.error("Server error", error=str(e))
            print(f"❌ Server error: {e}")

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")

        if self.ollama:
            try:
                self.ollama.close()
                logger.info("Ollama client closed")
            except Exception as e:
                logger.warning("Error closing Ollama client", error=str(e))

        # Additional cleanup for other components could go here
        logger.info("Resource cleanup completed")

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Voice Bot...")
        self.running = False
        await self.cleanup()
        logger.info("Voice Bot shutdown complete")

def setup_signal_handlers(voice_bot: VoiceBot):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        voice_bot.running = False

        # Schedule graceful shutdown
        asyncio.create_task(voice_bot.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    logger.info("🚀 Starting NETOVO Professional Voice Bot")

    try:
        # Load and validate configuration
        config = VoiceBotConfig()
        logger.info("Configuration loaded successfully")

        # Create and initialize voice bot
        voice_bot = VoiceBot(config)
        setup_signal_handlers(voice_bot)

        # Initialize components
        logger.info("Initializing Voice Bot components...")
        if not await voice_bot.initialize():
            logger.error("❌ Failed to initialize. Exiting.")
            sys.exit(1)

        logger.info("✅ Voice Bot initialization successful")

        # Start server
        await voice_bot.start_server()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("❌ Application failed", error=str(e), error_type=type(e).__name__)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Application terminated by user")
        sys.exit(0)
