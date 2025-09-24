"""
Fixed Main Voice Bot Application
Uses working Riva integration and actual implementations
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
        # Initialize working components with explicit configuration
        self.riva_asr = SimpleRivaASR(
            server_url=os.getenv("RIVA_SERVER", "localhost:50051")
        )
        self.riva_tts = SimpleRivaTTS(
            server_url=os.getenv("RIVA_SERVER", "localhost:50051")
        )
        
        # Create Ollama client with same settings as working test
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "orca2:7b")
        
        logger.info("Creating OllamaClient", url=ollama_url, model=ollama_model)
        self.ollama = OllamaClient(
            base_url=ollama_url,
            model=ollama_model
        )
        
        self.audio_processor = AudioProcessor()
        self.conversation = ConversationContext()

        # Audio settings
        self.telephony_rate = 8000  # 3CX/telephony
        self.asr_rate = 16000       # Riva ASR
        self.tts_rate = 22050       # Riva TTS

        self.running = False
        
        logger.info("VoiceBot components created successfully")

    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Voice Bot components...")

        try:
            # Test Ollama connection with better error handling
            logger.info("Testing Ollama connection...")
            
            # Make sure we're using the right client
            if not self.ollama.health_check():
                logger.error("Ollama health check failed")
                logger.error("Please check:")
                logger.error("1. Ollama service is running: systemctl status ollama")
                logger.error("2. Model is available: ollama list | grep orca2")
                logger.error("3. API endpoint accessible: curl http://localhost:11434/api/tags")
                raise ConnectionError("Ollama not available")

            logger.info("Ollama connection successful")

            # Test a simple generation to make sure it works
            try:
                test_response = await self.ollama.generate("Hello", max_tokens=5)
                logger.info("Ollama generation test successful", response=test_response[:30])
            except Exception as e:
                logger.error("Ollama generation test failed", error=str(e))
                raise ConnectionError("Ollama generation not working")

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error("Initialization failed", error=str(e))
            return False

    async def process_text_to_speech(self, text: str) -> str:
        """Convert text to speech and return WAV file path"""
        try:
            logger.info("Processing TTS", text=text[:50])

            # Use working TTS implementation
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

            # Use working ASR implementation
            transcript = self.riva_asr.transcribe_file(audio_file)

            logger.info("ASR completed", transcript=transcript)
            return transcript

        except Exception as e:
            logger.error("ASR processing failed", error=str(e))
            return "Speech processing error"

    async def get_llm_response(self, transcript: str) -> str:
        """Get response from Ollama LLM"""
        try:
            # Build context-aware prompt
            prompt = self.conversation.build_prompt(transcript, VOICE_BOT_SYSTEM_PROMPT)

            # Generate response
            response = await self.ollama.generate(
                prompt=transcript,
                system_prompt=VOICE_BOT_SYSTEM_PROMPT,
                max_tokens=50  # Keep responses short for telephony
            )

            # Add to conversation history
            self.conversation.add_turn(transcript, response)

            logger.info("LLM response generated",
                       input=transcript[:50],
                       output=response[:50])

            return response

        except Exception as e:
            logger.error("LLM processing failed", error=str(e))
            # Fallback response
            return "I'm sorry, I'm having trouble processing your request. Please try again."

    async def process_complete_pipeline(self, input_text: str) -> str:
        """
        Complete pipeline: Text → TTS → ASR → LLM → TTS
        This tests the full voice bot pipeline
        """
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
            except:
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
        print("\n🎤 NETOVO Voice Bot - Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 50)

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                print("🤖 Processing...")

                # Process through complete pipeline
                response = await self.process_complete_pipeline(user_input)

                print(f"Bot: {response}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
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
            print(f"\n🧪 Test {i}/{len(tests)}: {test_text}")

            try:
                result = await self.process_complete_pipeline(test_text)
                print(f"✅ Result: {result}")
            except Exception as e:
                print(f"❌ Failed: {e}")

        print("\n🎉 All tests completed!")

    async def start_server(self):
        """Start the voice bot server"""
        logger.info("Starting Voice Bot server...")

        # Show menu
        print("\n🎤 NETOVO Voice Bot")
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
            print("👋 Goodbye!")
        else:
            print("Invalid choice. Exiting.")

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Voice Bot...")
        self.running = False

        # Close connections
        if hasattr(self.ollama, 'close'):
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

    voice_bot = VoiceBot()
    setup_signal_handlers(voice_bot)

    # Initialize components
    if not await voice_bot.initialize():
        logger.error("Failed to initialize. Exiting.")
        sys.exit(1)

    # Start server
    await voice_bot.start_server()

if __name__ == "__main__":
    asyncio.run(main())
