"""
Rewritten Main Voice Bot Application
Fixes initialization issues with proper error handling
"""

import asyncio
import os
import signal
import sys
import tempfile
from pathlib import Path

import structlog
from dotenv import load_dotenv

# Import working implementations
from riva_proto_simple import SimpleRivaASR, SimpleRivaTTS
from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
from audio_processing import AudioProcessor, AudioBuffer
from conversation_context import ConversationContext

# Load environment variables
load_dotenv()

# Configure structured logging
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

class VoiceBot:
    """Main Voice Bot orchestrator with working implementations"""

    def __init__(self):
        """Initialize components without testing connections yet"""
        # Audio settings
        self.telephony_rate = 8000  # 3CX/telephony
        self.asr_rate = 16000       # Riva ASR
        self.tts_rate = 22050       # Riva TTS
        self.running = False
        
        # Initialize components to None - we'll create them in initialize()
        self.ollama = None
        self.riva_asr = None
        self.riva_tts = None
        self.audio_processor = None
        self.conversation = None
        
        logger.info("VoiceBot constructor completed")

    async def initialize(self):
        """Initialize all components with proper error handling"""
        logger.info("Starting VoiceBot initialization...")

        try:
            # Step 1: Initialize Ollama client
            logger.info("Creating Ollama client...")
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "orca2:7b")
            
            self.ollama = OllamaClient(base_url=ollama_url, model=ollama_model)
            logger.info("Ollama client created", url=ollama_url, model=ollama_model)
            
            # Step 2: Test Ollama health immediately after creation
            logger.info("Testing Ollama health...")
            health_check = self.ollama.health_check()
            
            if not health_check:
                logger.error("Ollama health check failed immediately after creation")
                logger.error("Debugging info:")
                logger.error("- URL: %s", ollama_url)
                logger.error("- Model: %s", ollama_model)
                
                # Try a direct test
                try:
                    import httpx
                    with httpx.Client() as test_client:
                        response = test_client.get(f"{ollama_url}/api/tags", timeout=5.0)
                        logger.error("Direct API test: status=%s", response.status_code)
                except Exception as direct_test_error:
                    logger.error("Direct API test failed: %s", str(direct_test_error))
                
                raise ConnectionError("Ollama health check failed")
            
            logger.info("Ollama health check passed")
            
            # Step 3: Test Ollama generation
            logger.info("Testing Ollama generation...")
            try:
                test_response = await self.ollama.generate("Hello", max_tokens=5)
                logger.info("Ollama generation test successful", response=test_response[:50])
            except Exception as gen_error:
                logger.error("Ollama generation test failed", error=str(gen_error))
                raise ConnectionError("Ollama generation failed")
            
            # Step 4: Initialize other components
            logger.info("Initializing other components...")
            
            self.riva_asr = SimpleRivaASR(
                server_url=os.getenv("RIVA_SERVER", "localhost:50051")
            )
            
            self.riva_tts = SimpleRivaTTS(
                server_url=os.getenv("RIVA_SERVER", "localhost:50051")
            )
            
            self.audio_processor = AudioProcessor()
            self.conversation = ConversationContext()
            
            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error("Initialization failed", error=str(e), error_type=type(e).__name__)
            return False

    async def process_text_to_speech(self, text: str) -> str:
        """Convert text to speech and return WAV file path"""
        try:
            logger.info("Processing TTS", text=text[:50])
            wav_path = self.riva_tts.synthesize(text)
            
            if wav_path:
                logger.info("TTS completed", output=wav_path)
                return wav_path
            else:
                logger.error("TTS failed")
                return ""

        except Exception as e:
            logger.error("TTS processing failed", error=str(e))
            return ""

    async def process_speech_to_text(self, audio_file: str) -> str:
        """Convert speech file to text"""
        try:
            logger.info("Processing ASR", file=audio_file)
            transcript = self.riva_asr.transcribe_file(audio_file)
            logger.info("ASR completed", transcript=transcript)
            return transcript

        except Exception as e:
            logger.error("ASR processing failed", error=str(e))
            return "Speech processing error"

    async def get_llm_response(self, transcript: str) -> str:
        """Get response from Ollama LLM"""
        try:
            # Simple approach: just use transcript with system prompt
            response = await self.ollama.generate(
                prompt=transcript,
                system_prompt=VOICE_BOT_SYSTEM_PROMPT,
                max_tokens=50
            )

            # Add to conversation history
            self.conversation.add_turn(transcript, response)

            logger.info("LLM response generated",
                       input=transcript[:50],
                       output=response[:50])

            return response

        except Exception as e:
            logger.error("LLM processing failed", error=str(e))
            return "I'm sorry, I'm having trouble processing your request. Please try again."

    async def process_complete_pipeline(self, input_text: str) -> str:
        """Complete pipeline: Text -> TTS -> ASR -> LLM -> TTS"""
        logger.info("Starting complete pipeline test", input=input_text)

        try:
            # Step 1: Convert input text to speech
            logger.info("Step 1: Text to Speech")
            tts_file = await self.process_text_to_speech(input_text)

            if not tts_file:
                return "TTS generation failed"

            # Step 2: Convert speech back to text (simulating phone input)
            logger.info("Step 2: Speech to Text")
            transcript = await self.process_speech_to_text(tts_file)

            # Step 3: Get LLM response
            logger.info("Step 3: LLM Processing")
            response = await self.get_llm_response(transcript)

            # Step 4: Convert response to speech
            logger.info("Step 4: Response to Speech")
            response_file = await self.process_text_to_speech(response)

            # Cleanup temp files
            try:
                os.unlink(tts_file)
                if response_file:
                    os.unlink(response_file)
            except Exception:
                pass

            logger.info("Complete pipeline successful",
                       original=input_text,
                       transcript=transcript,
                       response=response)

            return response_file if response_file else response

        except Exception as e:
            logger.error("Pipeline processing failed", error=str(e))
            return "Pipeline error"

    async def start_interactive_mode(self):
        """Start interactive voice bot mode"""
        logger.info("Starting Interactive Voice Bot Mode")
        print("\nVoice Bot - Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                print("Processing...")

                # Process through complete pipeline
                response = await self.process_complete_pipeline(user_input)
                print(f"Bot: {response}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error("Interactive mode error", error=str(e))
                print(f"Error: {e}")

    async def run_tests(self):
        """Run comprehensive tests"""
        logger.info("Running Voice Bot Tests")

        tests = [
            "Hello, this is a test of the voice bot system",
            "How are you today?",
            "What services does NETOVO provide?",
            "Thank you for your help"
        ]

        for i, test_text in enumerate(tests, 1):
            print(f"\nTest {i}/{len(tests)}: {test_text}")

            try:
                result = await self.process_complete_pipeline(test_text)
                print(f"Result: {result}")
            except Exception as e:
                print(f"Failed: {e}")

        print("\nAll tests completed!")

    async def start_server(self):
        """Start the voice bot server"""
        logger.info("Starting Voice Bot server...")

        # Show menu
        print("\nNETOVO Voice Bot")
        print("=" * 50)
        print("1. Interactive Mode")
        print("2. Run Tests")
        print("3. Exit")
        print("=" * 50)

        choice = input("Choose option (1-3): ").strip()

        if choice == "1":
            await self.start_interactive_mode()
        elif choice == "2":
            await self.run_tests()
        elif choice == "3":
            print("Goodbye!")
        else:
            print("Invalid choice. Exiting.")

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Voice Bot...")
        self.running = False

        if self.ollama:
            self.ollama.close()

        logger.info("Voice Bot shutdown complete")

def setup_signal_handlers(voice_bot):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info("Received signal", signal=signum)
        voice_bot.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    logger.info("Starting NETOVO Professional Voice Bot")

    try:
        voice_bot = VoiceBot()
        setup_signal_handlers(voice_bot)

        # Initialize components
        logger.info("Initializing Voice Bot...")
        if not await voice_bot.initialize():
            logger.error("Failed to initialize. Exiting.")
            sys.exit(1)

        logger.info("Voice Bot initialization successful")
        
        # Start server
        await voice_bot.start_server()
        
    except Exception as e:
        logger.error("Main execution failed", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
