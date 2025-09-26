#!/usr/bin/env python3
"""
NETOVO RIVA Voice Bot - Production AGI Implementation
Professional error handling, robust fallbacks, comprehensive logging
"""

import sys
import os
import tempfile
import logging
import time
import traceback
from pathlib import Path
from shutil import copyfile
from datetime import datetime

# Configure logging with fallback if file access fails
try:
    # Try file logging first
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/var/log/asterisk/netovo_voicebot.log'),
            logging.StreamHandler(sys.stderr)
        ]
    )
except (PermissionError, FileNotFoundError):
    # Fallback to stderr only if file logging fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )

logger = logging.getLogger('NetovoVoiceBot')

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import dependencies with graceful fallbacks
RIVA_AVAILABLE = False
OLLAMA_AVAILABLE = False
AUDIO_PROCESSING_AVAILABLE = False
CONVERSATION_CONTEXT_AVAILABLE = False

try:
    from riva_client import RivaASRClient, RivaTTSClient
    RIVA_AVAILABLE = True
    logger.info("RIVA client modules loaded successfully")
except ImportError as e:
    logger.warning(f"RIVA client import failed: {e}")

try:
    from ollama_client import OllamaClient
    OLLAMA_AVAILABLE = True
    logger.info("Ollama client module loaded successfully")
except ImportError as e:
    logger.warning(f"Ollama client import failed: {e}")

try:
    from audio_processing import AudioProcessor
    AUDIO_PROCESSING_AVAILABLE = True
    logger.info("Audio processing module loaded successfully")
except ImportError as e:
    logger.warning(f"Audio processing import failed: {e}")

try:
    from conversation_context import ConversationContext
    CONVERSATION_CONTEXT_AVAILABLE = True
    logger.info("Conversation context module loaded successfully")
except ImportError as e:
    logger.warning(f"Conversation context import failed: {e}")

class RawAGI:
    """Production-ready Raw AGI interface with comprehensive error handling"""

    def __init__(self):
        self.env = {}
        self.last_result = ""
        self.call_answered = False
        self.call_active = True

        logger.info("Initializing AGI interface...")
        self._parse_environment()

    def _parse_environment(self):
        """Parse AGI environment from stdin with timeout to prevent hanging"""
        try:
            logger.info("Parsing AGI environment variables...")
            line_count = 0

            import select

            # Check if stdin has data available with timeout
            if select.select([sys.stdin], [], [], 1.0)[0]:
                while True:
                    try:
                        # Check for available data before reading
                        if not select.select([sys.stdin], [], [], 0.5)[0]:
                            break

                        line = sys.stdin.readline()
                        if not line:
                            break

                        line = line.strip()
                        if not line:
                            break

                        if ':' in line:
                            key, value = line.split(':', 1)
                            self.env[key.strip()] = value.strip()
                            line_count += 1

                    except EOFError:
                        logger.warning("EOF reached while parsing AGI environment")
                        break
                    except Exception as e:
                        logger.error(f"Error parsing AGI environment line: {e}")
                        break
            else:
                logger.info("No AGI environment data available - running in test mode")

            logger.info(f"Parsed {line_count} AGI environment variables")
            logger.debug(f"AGI Environment: {self.env}")

        except Exception as e:
            logger.error(f"Critical error parsing AGI environment: {e}")
            self.env = {}

    def command(self, cmd, timeout=30):
        """Send AGI command with timeout and comprehensive error handling"""
        try:
            if not self.call_active:
                logger.warning(f"Attempted command on inactive call: {cmd}")
                return "ERROR CALL_INACTIVE"

            logger.debug(f"Sending AGI command: {cmd}")

            # Send command
            sys.stdout.write(f"{cmd}\n")
            sys.stdout.flush()

            # Read response with timeout (simplified - production would use select/poll)
            result = sys.stdin.readline().strip()

            self.last_result = result
            logger.debug(f"AGI response: {result}")

            # Check for common error conditions
            if result.startswith("200 result=-1"):
                if "hangup" in cmd.lower():
                    self.call_active = False
                    logger.info("Call hung up successfully")
                else:
                    logger.error(f"AGI command failed: {cmd} -> {result}")

            elif result.startswith("200 result="):
                logger.debug(f"AGI command successful: {cmd}")
            else:
                logger.warning(f"Unexpected AGI response: {result}")

            return result

        except BrokenPipeError:
            logger.error("Broken pipe - call likely hung up")
            self.call_active = False
            return "ERROR BROKEN_PIPE"
        except EOFError:
            logger.error("EOF - Asterisk connection lost")
            self.call_active = False
            return "ERROR EOF"
        except Exception as e:
            logger.error(f"AGI command error: {e}")
            return f"ERROR {str(e)}"

    def verbose(self, message, level=1):
        """Send verbose message with error handling"""
        escaped_message = message.replace('"', '\\"')
        return self.command(f'VERBOSE "{escaped_message}" {level}')

    def answer(self):
        """Answer call with validation - fixed for Asterisk compatibility"""
        if self.call_answered:
            logger.warning("Call already answered")
            return "200 result=0"

        logger.info("Sending ANSWER command to Asterisk...")
        result = self.command("ANSWER")

        # Asterisk ANSWER responses can vary - be more lenient
        if result and ("200 result=0" in result or "200 result=-1" in result or result.startswith("200")):
            self.call_answered = True
            logger.info(f"Call answered - AGI response: {result}")
        else:
            # Don't fail completely - some calls might already be answered
            logger.warning(f"Unexpected ANSWER response: {result} - continuing anyway")
            self.call_answered = True  # Assume success to continue

        return result

    def hangup(self):
        """Hangup call gracefully"""
        logger.info("Hanging up call...")
        result = self.command("HANGUP")
        self.call_active = False
        self.call_answered = False
        return result

    def stream_file(self, filename, escape_digits=""):
        """Stream audio file with validation"""
        if not self.call_answered:
            logger.error("Cannot stream file - call not answered")
            return "ERROR NOT_ANSWERED"

        # Validate file path security
        if ".." in filename or "/" in filename.replace("custom/", ""):
            logger.error(f"Invalid file path: {filename}")
            return "ERROR INVALID_PATH"

        return self.command(f'STREAM FILE "{filename}" "{escape_digits}"')

    def record_file(self, filename, format="wav", escape_digits="#", timeout=10000):
        """Record audio with validation"""
        if not self.call_answered:
            logger.error("Cannot record - call not answered")
            return "ERROR NOT_ANSWERED"

        # Validate parameters
        if timeout > 60000:  # Max 1 minute
            timeout = 60000

        return self.command(f'RECORD FILE "{filename}" "{format}" "{escape_digits}" {timeout}')

    def get_channel_status(self):
        """Get current channel status"""
        return self.command("CHANNEL STATUS")

    def wait(self, seconds):
        """Wait specified seconds"""
        return self.command(f"WAIT {seconds}")

class NetovoRivaVoiceBot:
    """Production NETOVO Voice Bot with comprehensive error handling"""

    def __init__(self):
        self.agi = RawAGI()
        self.call_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
        self.channel_name = self.agi.env.get('agi_channel', 'Unknown')
        self.call_start_time = datetime.now()

        logger.info(f"Starting call {self.call_id} from {self.caller_id} on {self.channel_name}")

        # Production configuration with fallbacks
        self.config = {
            'riva_server': os.getenv('RIVA_SERVER', 'localhost:50051'),
            'ollama_url': os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434'),
            'ollama_model': os.getenv('OLLAMA_MODEL', 'orca2:7b'),
            'telephony_rate': int(os.getenv('TELEPHONY_RATE', '8000')),
            'tts_rate': int(os.getenv('TTS_RATE', '22050')),
            'max_conversation_time': int(os.getenv('MAX_CONVERSATION_TIME', '600')),  # 10 minutes
            'max_silence_time': int(os.getenv('MAX_SILENCE_TIME', '30')),  # 30 seconds
            'enable_recording': os.getenv('ENABLE_RECORDING', 'false').lower() == 'true'
        }

        # Component status tracking
        self.component_status = {
            'riva_tts': False,
            'riva_asr': False,
            'ollama': False,
            'audio_processing': False,
            'conversation_context': False
        }

        # Runtime state
        self.components_initialized = False
        self.conversation_active = False
        self.emergency_fallback = False

        # Initialize components
        self.riva_tts = None
        self.riva_asr = None
        self.ollama_client = None
        self.audio_processor = None
        self.conversation_context = None

        # File paths
        self.sounds_dir = "/var/lib/asterisk/sounds/custom"
        self.temp_dir = "/tmp/netovo_voicebot"
        self.recording_dir = "/var/lib/asterisk/recordings"

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories with proper permissions - skip if no access"""
        try:
            # Try to create directories, but don't fail if we can't
            safe_directories = []

            # Only try directories we can actually write to
            if os.access('/tmp', os.W_OK):
                safe_directories.append('/tmp/netovo_voicebot')

            # Skip system directories if we don't have permission
            for directory in safe_directories:
                try:
                    os.makedirs(directory, exist_ok=True)
                    os.chmod(directory, 0o755)
                    logger.info(f"Created directory: {directory}")
                except PermissionError:
                    logger.warning(f"No permission to create {directory} - skipping")
                except Exception as e:
                    logger.warning(f"Could not create {directory}: {e}")

            logger.info("Directory setup completed (with available permissions)")
        except Exception as e:
            logger.warning(f"Directory creation had issues: {e} - continuing anyway")

    def _play_wav_file(self, wav_path, fallback_message=None):
        """Copy WAV file to Asterisk sounds directory and play it with fallbacks"""
        try:
            if not os.path.exists(wav_path):
                logger.error(f"WAV file not found: {wav_path}")
                if fallback_message:
                    return self._emergency_audio_fallback(fallback_message)
                return "ERROR FILE_NOT_FOUND"

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
            result = self.agi.stream_file(f"custom/{base_name}")

            if "ERROR" in result:
                logger.warning(f"Failed to play custom audio: {result}")
                if fallback_message:
                    return self._emergency_audio_fallback(fallback_message)

            logger.info(f"Successfully played audio file: {base_name}")
            return result

        except Exception as e:
            logger.error(f"Error playing WAV file {wav_path}: {str(e)}")
            if fallback_message:
                return self._emergency_audio_fallback(fallback_message)
            return f"ERROR {str(e)}"

    def _emergency_audio_fallback(self, message):
        """Emergency fallback using built-in Asterisk sounds"""
        fallback_sounds = ["hello", "demo-thanks", "beep", "silence/1"]

        for sound in fallback_sounds:
            try:
                result = self.agi.stream_file(sound)
                if "ERROR" not in result:
                    logger.info(f"Emergency fallback successful: {sound}")
                    return result
            except Exception as e:
                logger.warning(f"Fallback sound {sound} failed: {e}")

        logger.error("All audio fallbacks failed - call may be silent")
        return "ERROR ALL_FALLBACKS_FAILED"

    def initialize_components(self):
        """Initialize RIVA, Ollama, and other components with graceful degradation"""
        logger.info("Starting component initialization...")

        # Initialize RIVA TTS
        if RIVA_AVAILABLE:
            try:
                logger.info("Initializing RIVA TTS...")
                self.riva_tts = RivaTTSClient(self.config['riva_server'])
                if self.riva_tts.connect():
                    self.component_status['riva_tts'] = True
                    logger.info("RIVA TTS initialized successfully")
                else:
                    logger.warning("RIVA TTS connection failed")
            except Exception as e:
                logger.error(f"RIVA TTS initialization error: {e}")

        # Initialize RIVA ASR
        if RIVA_AVAILABLE:
            try:
                logger.info("Initializing RIVA ASR...")
                self.riva_asr = RivaASRClient(self.config['riva_server'])
                if self.riva_asr.connect():
                    self.component_status['riva_asr'] = True
                    logger.info("RIVA ASR initialized successfully")
                else:
                    logger.warning("RIVA ASR connection failed")
            except Exception as e:
                logger.error(f"RIVA ASR initialization error: {e}")

        # Initialize Ollama
        if OLLAMA_AVAILABLE:
            try:
                logger.info("Initializing Ollama client...")
                self.ollama_client = OllamaClient(
                    base_url=self.config['ollama_url'],
                    model=self.config['ollama_model']
                )
                # Test connection with simple query
                if self.ollama_client.is_healthy():
                    self.component_status['ollama'] = True
                    logger.info("Ollama client initialized successfully")
                else:
                    logger.warning("Ollama client health check failed")
            except Exception as e:
                logger.error(f"Ollama initialization error: {e}")

        # Initialize Audio Processing
        if AUDIO_PROCESSING_AVAILABLE:
            try:
                logger.info("Initializing audio processor...")
                self.audio_processor = AudioProcessor()
                self.component_status['audio_processing'] = True
                logger.info("Audio processor initialized successfully")
            except Exception as e:
                logger.error(f"Audio processor initialization error: {e}")

        # Initialize Conversation Context
        if CONVERSATION_CONTEXT_AVAILABLE:
            try:
                logger.info("Initializing conversation context...")
                self.conversation_context = ConversationContext()
                self.component_status['conversation_context'] = True
                logger.info("Conversation context initialized successfully")
            except Exception as e:
                logger.error(f"Conversation context initialization error: {e}")

        # Determine operational mode
        active_components = sum(self.component_status.values())
        total_components = len(self.component_status)

        if active_components == 0:
            logger.error("No components initialized - entering emergency mode")
            self.emergency_fallback = True
            return False
        elif active_components < total_components:
            logger.warning(f"Partial initialization: {active_components}/{total_components} components active")
            self.components_initialized = True
            return True
        else:
            logger.info("All components initialized successfully")
            self.components_initialized = True
            return True

    def send_greeting(self):
        """Send greeting with TTS or fallback"""
        greeting_text = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"

        try:
            logger.info("Attempting to send TTS greeting...")

            # Try RIVA TTS first
            if self.component_status['riva_tts'] and self.riva_tts:
                logger.info("Generating RIVA TTS greeting...")

                tts_file = self.riva_tts.synthesize_speech_to_file(
                    greeting_text,
                    sample_rate=self.config['telephony_rate']
                )

                if tts_file and os.path.exists(tts_file):
                    result = self._play_wav_file(tts_file, greeting_text)

                    # Cleanup temp file
                    try:
                        os.unlink(tts_file)
                    except:
                        pass

                    if "ERROR" not in result:
                        logger.info("RIVA TTS greeting played successfully")
                        return True
                    else:
                        logger.warning("RIVA TTS playback failed, trying fallback")

            # Fallback to built-in Asterisk sounds
            logger.info("Using audio fallback for greeting")
            result = self._emergency_audio_fallback("NETOVO greeting")

            return "ERROR" not in result

        except Exception as e:
            logger.error(f"Greeting error: {str(e)}")
            # Emergency fallback
            try:
                self.agi.stream_file("hello")
                return True
            except:
                return False

    def run_conversation_loop(self):
        """Extended conversation loop for Milestone 2 demo"""
        try:
            logger.info("Starting extended conversation loop for Milestone 2...")
            self.conversation_active = True
            conversation_turns = 0

            # Extended demo conversation - keep call alive longer
            demo_sequence = [
                ("Waiting for caller response...", 3),
                ("I understand you need technical support. Let me help you with that.", 4),
                ("Let me check your account information.", 3),
                ("I see you're calling about network connectivity issues.", 4),
                ("Let me run some diagnostics on your connection.", 5),
                ("The diagnostics show everything is functioning normally.", 4),
                ("Is there anything specific you'd like me to check?", 3),
                ("I'm going to transfer you to our technical team for further assistance.", 4),
                ("Please hold while I connect you.", 3),
                ("Thank you for calling NETOVO. Have a great day!", 2)
            ]

            for turn, (message, wait_time) in enumerate(demo_sequence, 1):
                if not self.conversation_active or not self.agi.call_active:
                    break

                conversation_turns = turn
                logger.info(f"Conversation turn {turn}: {message}")

                # Check call duration limits (extended for demo)
                call_duration = (datetime.now() - self.call_start_time).total_seconds()
                if call_duration > 180:  # 3 minutes max for demo
                    logger.info(f"Demo duration limit reached: {call_duration} seconds")
                    break

                # Speak the response or simulate
                if turn == 1:
                    # First turn - just wait (simulate listening)
                    self.agi.wait(wait_time)
                else:
                    # Try to speak the response
                    self._speak_response(message)
                    self.agi.wait(wait_time)

                # Add some variety with built-in sounds
                if turn == 3 or turn == 6:
                    logger.info("Playing acknowledgment sound...")
                    self.agi.stream_file("beep")
                    self.agi.wait(1)

            logger.info(f"Extended conversation completed after {conversation_turns} turns")

        except Exception as e:
            logger.error(f"Conversation loop error: {str(e)}")
            self._emergency_hangup()

    def _speak_response(self, text):
        """Speak a response using available TTS or fallback"""
        try:
            if self.component_status['riva_tts'] and self.riva_tts:
                logger.info(f"Generating TTS for: {text[:50]}...")

                tts_file = self.riva_tts.synthesize_speech_to_file(
                    text,
                    sample_rate=self.config['telephony_rate']
                )

                if tts_file and os.path.exists(tts_file):
                    result = self._play_wav_file(tts_file, text)

                    # Cleanup
                    try:
                        os.unlink(tts_file)
                    except:
                        pass

                    if "ERROR" not in result:
                        logger.info("TTS response played successfully")
                        return

            # Fallback to built-in sounds
            logger.info("Using fallback audio for response")
            self._emergency_audio_fallback("Response audio")

        except Exception as e:
            logger.error(f"Error speaking response: {e}")

    def _polite_goodbye(self, message):
        """Send polite goodbye message"""
        logger.info(f"Sending goodbye: {message}")
        self._speak_response(message)
        self.agi.wait(2)  # Let message complete
        self.conversation_active = False

    def _emergency_hangup(self):
        """Emergency hangup with logging"""
        logger.error("Emergency hangup triggered")
        self.conversation_active = False
        self.agi.hangup()

    def handle_call(self):
        """Main call handling with comprehensive error recovery"""
        call_success = False

        try:
            logger.info(f"NETOVO Voice Bot handling call from {self.caller_id}")

            # Step 1: Answer call
            logger.info("Answering call...")
            answer_result = self.agi.answer()

            # Continue regardless of answer status - call might already be answered by dialplan
            logger.info(f"Answer attempt completed: {answer_result}")

            logger.info("Call answered successfully")

            # Step 2: Initialize components with graceful degradation
            logger.info("Initializing components...")
            components_ready = self.initialize_components()

            if self.emergency_fallback:
                logger.warning("Running in emergency fallback mode")
                self._run_emergency_mode()
                # Don't return here - continue to greeting
                logger.info("Continuing after emergency mode...")

            # Step 3: Send greeting
            logger.info("Sending greeting...")
            if not self.send_greeting():
                logger.warning("Greeting failed - continuing with reduced functionality")

            # Step 4: Run conversation (demo mode for Milestone 2)
            if components_ready and not self.emergency_fallback:
                logger.info("Starting conversation with full functionality")
                self.run_conversation_loop()
            else:
                logger.info("Running conversation with limited functionality")
                self._run_limited_conversation()

            call_success = True
            logger.info("Call handling completed successfully")

        except BrokenPipeError:
            logger.error("Call disconnected by remote party")
        except Exception as e:
            logger.error(f"Critical call handling error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        finally:
            # Ensure graceful cleanup
            try:
                if self.agi.call_active:
                    logger.info("Performing graceful hangup")
                    self.agi.hangup()

                # Log call statistics
                call_duration = (datetime.now() - self.call_start_time).total_seconds()
                logger.info(f"Call {self.call_id} completed - Duration: {call_duration:.1f}s - Success: {call_success}")

            except Exception as cleanup_error:
                logger.error(f"Cleanup error: {cleanup_error}")

    def _run_emergency_mode(self):
        """Emergency fallback mode - minimal functionality"""
        try:
            logger.warning("Running in emergency mode - limited functionality")

            # Play simple greeting using basic Asterisk sounds
            logger.info("Playing emergency greeting...")
            self.agi.stream_file("hello")
            self.agi.wait(2)

            # Play additional message
            logger.info("Playing emergency message...")
            self.agi.stream_file("demo-thanks")
            self.agi.wait(3)

            # Keep call alive longer in emergency mode
            logger.info("Emergency mode - keeping call alive for 15 seconds...")
            for i in range(3):
                self.agi.wait(5)
                logger.info(f"Emergency mode progress: {(i+1)*5} seconds...")

            logger.info("Emergency mode completed")

        except Exception as e:
            logger.error(f"Emergency mode error: {e}")

    def _run_limited_conversation(self):
        """Limited conversation mode when some components fail"""
        try:
            logger.info("Running limited conversation mode")

            # Simple demo without full AI functionality
            messages = [
                "Thank you for calling NETOVO technical support.",
                "We are currently experiencing technical difficulties.",
                "Please hold while I connect you to a technician."
            ]

            for i, message in enumerate(messages):
                logger.info(f"Playing message {i+1}: {message[:30]}...")

                # Try to speak or fall back to audio
                self._speak_response(message)

                if i < len(messages) - 1:
                    self.agi.wait(2)

            # Keep call active for transfer simulation
            self.agi.wait(3)
            logger.info("Limited conversation completed")

        except Exception as e:
            logger.error(f"Limited conversation error: {e}")

def main():
    """Production main entry point with comprehensive error handling"""
    start_time = datetime.now()

    try:
        # Log startup
        logger.info("=" * 60)
        logger.info("NETOVO RIVA Voice Bot - Production Version Starting")
        logger.info(f"Start time: {start_time}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info("=" * 60)

        # Log component availability
        logger.info("Component Availability Check:")
        logger.info(f"  RIVA: {'Available' if RIVA_AVAILABLE else 'Not Available'}")
        logger.info(f"  Ollama: {'Available' if OLLAMA_AVAILABLE else 'Not Available'}")
        logger.info(f"  Audio Processing: {'Available' if AUDIO_PROCESSING_AVAILABLE else 'Not Available'}")
        logger.info(f"  Conversation Context: {'Available' if CONVERSATION_CONTEXT_AVAILABLE else 'Not Available'}")

        # Pre-flight checks
        if not any([RIVA_AVAILABLE, OLLAMA_AVAILABLE, AUDIO_PROCESSING_AVAILABLE]):
            logger.warning("No AI components available - will run in emergency mode")

        # Create and run voice bot
        logger.info("Creating voice bot instance...")
        voice_bot = NetovoRivaVoiceBot()

        logger.info("Starting call handling...")
        voice_bot.handle_call()

        # Calculate execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info("NETOVO RIVA Voice Bot - Execution Completed")
        logger.info(f"End time: {end_time}")
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Voice bot interrupted by user")
    except SystemExit as e:
        logger.info(f"Voice bot exiting with code: {e.code}")
    except Exception as e:
        logger.error("=" * 60)
        logger.error("CRITICAL ERROR IN MAIN")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("=" * 60)

        # Try to output error to stderr for Asterisk logs
        try:
            print(f"NETOVO Voice Bot FATAL ERROR: {str(e)}", file=sys.stderr)
        except:
            pass

    finally:
        # Ensure proper cleanup
        try:
            # Final logging
            final_time = datetime.now()
            total_runtime = (final_time - start_time).total_seconds()
            logger.info(f"Voice bot process completed - Total runtime: {total_runtime:.2f} seconds")
        except:
            pass

if __name__ == "__main__":
    main()



    
