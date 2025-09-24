"""
NETOVO Professional Voice Bot Application
Production-ready version with proper error handling and realistic testing
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import time

import structlog
from dotenv import load_dotenv

# Import working implementations
from riva_proto_simple import SimpleRivaASR, SimpleRivaTTS
from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
from audio_processing import AudioProcessor, AudioBuffer, calculate_text_similarity
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

class TestResult:
    """Detailed test result with validation"""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = time.time()
        self.end_time = None
        self.steps = {}
        self.passed = False
        self.error_message = ""
        self.details = {}

    def add_step(self, step_name: str, success: bool, details: str = ""):
        """Add a test step result"""
        self.steps[step_name] = {
            "success": success,
            "details": details,
            "timestamp": time.time()
        }

    def finish(self, passed: bool, error_message: str = ""):
        """Mark test as complete"""
        self.end_time = time.time()
        self.passed = passed
        self.error_message = error_message

    def duration(self) -> float:
        """Get test duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def summary(self) -> str:
        """Get test summary"""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        duration = f"{self.duration():.2f}s"

        if not self.passed and self.error_message:
            return f"{status} ({duration}) - {self.error_message}"

        step_summary = []
        for step, result in self.steps.items():
            step_status = "✓" if result["success"] else "✗"
            step_summary.append(f"{step_status} {step}")

        steps_str = " | ".join(step_summary) if step_summary else "No steps"
        return f"{status} ({duration}) - {steps_str}"

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

        # Track temp files for cleanup
        self.temp_files = set()

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
                # Track temp file for cleanup
                self.temp_files.add(wav_path)
                logger.info("TTS completed successfully", output=wav_path)
                return wav_path
            else:
                logger.error("TTS failed - no output file generated")
                return ""

        except Exception as e:
            logger.error("TTS processing failed", error=str(e))
            return ""

    async def process_speech_to_text(self, audio_file: str, for_testing: bool = False) -> str:
        """Convert speech file to text"""
        if not audio_file or not os.path.exists(audio_file):
            logger.error("Audio file does not exist", file=audio_file)
            return ""

        try:
            logger.info("Processing ASR", file=audio_file)

            # Resample to 16kHz before ASR processing
            resampled_file = self.audio_processor.resample_wav(audio_file, 16000)

            # Use appropriate punctuation settings for testing vs production
            automatic_punctuation = not for_testing  # False for tests, True for production

            transcript = self.riva_asr.transcribe_file(resampled_file, automatic_punctuation=automatic_punctuation)

            # Cleanup resampled file if different from original
            if resampled_file != audio_file:
                self.temp_files.add(resampled_file)

            if transcript and transcript.strip():
                logger.info("ASR completed successfully", transcript=transcript[:100])
                return transcript.strip()
            else:
                logger.warning("ASR returned empty transcript")
                return ""

        except Exception as e:
            logger.error("ASR processing failed", error=str(e))
            return ""

    async def get_llm_response(self, transcript: str) -> str:
        """Get response from Ollama LLM"""
        if not transcript.strip():
            return ""

        try:
            logger.info("Generating LLM response", input=transcript[:100])

            response = self.ollama.generate(
                prompt=transcript,
                system_prompt=VOICE_BOT_SYSTEM_PROMPT,
                max_tokens=self.config.max_tokens
            )

            if response and response.strip():
                # Add to conversation history
                self.conversation.add_turn(transcript, response)
                logger.info("LLM response generated successfully",
                           input_length=len(transcript),
                           output_length=len(response))
                return response.strip()
            else:
                logger.warning("LLM returned empty response")
                return ""

        except Exception as e:
            logger.error("LLM processing failed", error=str(e))
            return ""

    async def validate_complete_pipeline(self, input_text: str) -> TestResult:
        """Run complete pipeline with detailed validation"""
        test_result = TestResult(f"Pipeline Test: {input_text[:50]}...")

        if not input_text.strip():
            test_result.finish(False, "No input provided")
            return test_result

        logger.info("Starting validated pipeline test", input=input_text[:100])

        try:
            # Step 1: Text to Speech
            logger.info("Pipeline Step 1: Text to Speech")
            tts_file = await self.process_text_to_speech(input_text)

            if not tts_file or not os.path.exists(tts_file):
                test_result.add_step("TTS", False, "No output file generated")
                test_result.finish(False, "TTS generation failed")
                return test_result

            # Validate TTS file
            try:
                file_size = os.path.getsize(tts_file)
                if file_size < 1000:  # Less than 1KB indicates failure
                    test_result.add_step("TTS", False, f"File too small: {file_size} bytes")
                    test_result.finish(False, "TTS file too small")
                    return test_result
                test_result.add_step("TTS", True, f"Generated {file_size} bytes")
            except Exception as e:
                test_result.add_step("TTS", False, f"File validation error: {e}")
                test_result.finish(False, "TTS file validation failed")
                return test_result

            # Step 2: Speech to Text
            logger.info("Pipeline Step 2: Speech to Text")
            transcript = await self.process_speech_to_text(tts_file, for_testing=True)

            if not transcript:
                test_result.add_step("ASR", False, "Empty transcript")
                test_result.finish(False, "ASR returned empty transcript")
                return test_result

            # Use improved text similarity validation
            similarity_score = calculate_text_similarity(input_text, transcript)

            if similarity_score < 0.70:  # 70% similarity threshold
                test_result.add_step("ASR", False, f"Similarity too low: {similarity_score:.2%}")
                test_result.finish(False, f"ASR transcript similarity below threshold: {similarity_score:.2%}")
                return test_result

            test_result.add_step("ASR", True, f"Transcript: '{transcript[:50]}...' (similarity: {similarity_score:.2%})")

            # Step 3: LLM Processing
            logger.info("Pipeline Step 3: LLM Processing")
            llm_response = await self.get_llm_response(transcript)

            if not llm_response:
                test_result.add_step("LLM", False, "Empty response")
                test_result.finish(False, "LLM returned empty response")
                return test_result

            # Validate LLM response quality
            if len(llm_response) < 5:
                test_result.add_step("LLM", False, "Response too short")
                test_result.finish(False, "LLM response too short")
                return test_result

            test_result.add_step("LLM", True, f"Response: '{llm_response[:50]}...'")

            # Step 4: Response to Speech
            logger.info("Pipeline Step 4: Response to Speech")
            response_file = await self.process_text_to_speech(llm_response)

            if not response_file or not os.path.exists(response_file):
                test_result.add_step("Response TTS", False, "No output file generated")
                test_result.finish(False, "Response TTS failed")
                return test_result

            # Validate response TTS file
            try:
                response_file_size = os.path.getsize(response_file)
                if response_file_size < 1000:
                    test_result.add_step("Response TTS", False, f"File too small: {response_file_size} bytes")
                    test_result.finish(False, "Response TTS file too small")
                    return test_result
                test_result.add_step("Response TTS", True, f"Generated {response_file_size} bytes")
            except Exception as e:
                test_result.add_step("Response TTS", False, f"File validation error: {e}")
                test_result.finish(False, "Response TTS file validation failed")
                return test_result

            # All steps passed
            test_result.details = {
                "input": input_text,
                "transcript": transcript,
                "llm_response": llm_response,
                "similarity": similarity_score,
                "tts_file_size": file_size,
                "response_file_size": response_file_size
            }

            test_result.finish(True)
            logger.info("Complete pipeline validation successful",
                       input=input_text[:50],
                       transcript=transcript[:50],
                       response=llm_response[:50])

        except Exception as e:
            logger.error("Pipeline validation failed", error=str(e))
            test_result.finish(False, f"Exception: {str(e)}")

        return test_result

    async def process_complete_pipeline(self, input_text: str) -> str:
        """Simple pipeline for interactive use"""
        if not input_text.strip():
            return "No input provided"

        try:
            # Step 1: Text to Speech
            tts_file = await self.process_text_to_speech(input_text)
            if not tts_file:
                return "TTS generation failed"

            # Step 2: Speech to Text
            transcript = await self.process_speech_to_text(tts_file)
            if not transcript:
                return "Speech recognition failed"

            # Step 3: LLM Response
            response = await self.get_llm_response(transcript)
            if not response:
                return "Language model failed to respond"

            # Step 4: Response to Speech
            response_file = await self.process_text_to_speech(response)

            return response

        except Exception as e:
            logger.error("Pipeline processing failed", error=str(e))
            return f"Pipeline error: {str(e)}"

    def _cleanup_temp_files(self):
        """Clean up temporary files with better error handling"""
        cleaned = 0
        failed = 0

        for temp_file in list(self.temp_files):
            try:
                if temp_file and os.path.exists(temp_file):
                    os.chmod(temp_file, 0o666)  # Try to make it writable
                    os.unlink(temp_file)
                    self.temp_files.discard(temp_file)
                    cleaned += 1
                    logger.debug("Cleaned up temp file", file=temp_file)
            except Exception as e:
                failed += 1
                logger.debug("Failed to cleanup temp file", file=temp_file, error=str(e))

        if cleaned > 0 or failed > 0:
            logger.info("Temp file cleanup", cleaned=cleaned, failed=failed)

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
        """Run comprehensive tests with detailed validation"""
        logger.info("Running Voice Bot Tests with detailed validation")

        test_cases = [
            "Hello, this is a test of the voice bot system",
            "How are you today?",
            "What services does NETOVO provide?",
            "Thank you for your help"
        ]

        print(f"\n🧪 Running {len(test_cases)} comprehensive test cases...\n")

        passed = 0
        failed = 0
        results = []

        for i, test_text in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: {test_text}")
            print("🔄 Running detailed validation...")

            try:
                test_result = await self.validate_complete_pipeline(test_text)
                results.append(test_result)

                print(f"   {test_result.summary()}")

                if test_result.passed:
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                failed += 1
                print(f"   ❌ EXCEPTION: {e}")
                logger.error("Test case failed with exception", test=test_text, error=str(e))

            print()

        # Clean up temp files after tests
        self._cleanup_temp_files()

        print(f"📊 Final Test Results: {passed} passed, {failed} failed")
        print(f"⏱️  Total test duration: {sum(r.duration() for r in results):.2f}s")

        # Show detailed breakdown
        if failed > 0:
            print("\n❌ Failed Tests Details:")
            for result in results:
                if not result.passed:
                    print(f"   • {result.test_name}: {result.error_message}")

        logger.info("Test run completed", passed=passed, failed=failed, total_duration=sum(r.duration() for r in results))

    async def start_server(self):
        """Start the voice bot server"""
        logger.info("Starting Voice Bot server...")
        self.running = True

        print(f"\n🎯 NETOVO Voice Bot v1.0")
        print("=" * 50)
        print("1. Interactive Mode")
        print("2. Run Comprehensive Tests")
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
        finally:
            # Always clean up temp files on exit
            self._cleanup_temp_files()

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")

        # Clean up temp files
        self._cleanup_temp_files()

        if self.ollama:
            try:
                self.ollama.close()
                logger.info("Ollama client closed")
            except Exception as e:
                logger.warning("Error closing Ollama client", error=str(e))

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
