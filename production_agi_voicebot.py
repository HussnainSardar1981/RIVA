#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
NETOVO Production AGI VoiceBot - Professional 3CX Customer Support
Integrates RIVA ASR/TTS, Ollama LLM, and conversation context management
Uses correct AGI command syntax and robust error handling
"""

import sys
import os
import time
import tempfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Set up project paths BEFORE any other imports
project_dir = "/home/aiadmin/netovo_voicebot"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Configure structured logging to voicebot.log
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - VoiceBot - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/voicebot.log', mode='a'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class ProductionAGI:
    """Production-grade AGI interface with correct command syntax"""

    def __init__(self):
        self.env = {}
        self.connected = True
        self.call_answered = False

        logger.info("=== NETOVO AGI VoiceBot Initializing ===")
        self._parse_environment()

    def _parse_environment(self):
        """Parse AGI environment variables from stdin"""
        try:
            env_count = 0
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    break
                if ':' in line:
                    key, value = line.split(':', 1)
                    self.env[key.strip()] = value.strip()
                    env_count += 1

            logger.info(f"AGI environment parsed: {env_count} variables")
            logger.info(f"Caller ID: {self.env.get('agi_callerid', 'Unknown')}")
            logger.info(f"Channel: {self.env.get('agi_channel', 'Unknown')}")

        except Exception as e:
            logger.error(f"Environment parsing error: {e}")
            self.env = {}

    def command(self, cmd):
        """Send AGI command with proper error handling"""
        if not self.connected:
            logger.warning(f"Attempted command on disconnected channel: {cmd}")
            return "ERROR"

        try:
            logger.debug(f"AGI Command: {cmd}")

            # Send command to Asterisk
            print(cmd)
            sys.stdout.flush()

            # Read response
            result = sys.stdin.readline()
            if not result:
                logger.error("No response from Asterisk - connection lost")
                self.connected = False
                return "ERROR"

            result = result.strip()
            logger.debug(f"AGI Response: {result}")

            # Parse result
            if result.startswith('200'):
                return result
            elif result.startswith('510'):
                logger.error(f"Invalid AGI command: {cmd} -> {result}")
                return "ERROR"
            elif result.startswith('511'):
                logger.error(f"Command not permitted: {cmd} -> {result}")
                return "ERROR"
            else:
                logger.warning(f"Unexpected AGI response: {result}")
                return result

        except BrokenPipeError:
            logger.error("Broken pipe - call likely disconnected")
            self.connected = False
            return "ERROR"
        except Exception as e:
            logger.error(f"AGI command error: {e}")
            return "ERROR"

    def verbose(self, message, level=1):
        """Send verbose message to Asterisk logs"""
        escaped_msg = message.replace('"', '\\"')
        return self.command(f'VERBOSE "{escaped_msg}" {level}')

    def answer(self):
        """Answer the call"""
        if self.call_answered:
            logger.warning("Call already answered")
            return True

        result = self.command("ANSWER")
        if result and result.startswith('200'):
            self.call_answered = True
            logger.info("Call answered successfully")
            return True
        else:
            logger.error(f"Failed to answer call: {result}")
            return False

    def hangup(self):
        """Hangup the call"""
        logger.info("Hanging up call...")
        self.command("HANGUP")
        self.connected = False
        self.call_answered = False

    def stream_file(self, filename, escape_digits=""):
        """Stream audio file - CORRECT AGI SYNTAX (no quotes on filename)"""
        if not self.call_answered:
            logger.error("Cannot stream file - call not answered")
            return False

        # Remove file extension if present
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]

        # Correct AGI syntax: STREAM FILE filename "escape_digits"
        result = self.command(f'STREAM FILE {filename} "{escape_digits}"')

        success = result and result.startswith('200')
        if success:
            logger.info(f"Successfully streamed file: {filename}")
        else:
            logger.error(f"Failed to stream file {filename}: {result}")

        return success

    def record_file(self, filename, format="wav", escape_digits="#", timeout=10000, silence=3000):
        """Record audio file - CORRECT AGI SYNTAX"""
        if not self.call_answered:
            logger.error("Cannot record - call not answered")
            return False

        # Correct AGI syntax: RECORD FILE filename format escape_digits timeout [offset] [BEEP] [silence]
        result = self.command(f'RECORD FILE {filename} {format} {escape_digits} {timeout} 0 0 {silence}')

        success = result and result.startswith('200')
        if success:
            logger.info(f"Recording completed: {filename}.{format}")
        else:
            logger.error(f"Recording failed: {result}")

        return success

    def get_channel_status(self):
        """Get channel status"""
        return self.command("CHANNEL STATUS")

    def sleep(self, seconds):
        """Sleep using Python time.sleep (not AGI WAIT which has issues)"""
        time.sleep(seconds)

class NetovoProductionVoiceBot:
    """Production NETOVO VoiceBot with full AI integration"""

    def __init__(self):
        self.agi = ProductionAGI()
        self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
        self.channel = self.agi.env.get('agi_channel', 'Unknown')
        self.call_start_time = datetime.now()

        # Configuration
        self.config = {
            'max_conversation_turns': 8,
            'max_call_duration': 300,  # 5 minutes
            'recording_timeout': 8000,  # 8 seconds
            'silence_timeout': 3000,    # 3 seconds
            'riva_server': 'localhost:50051',
            'ollama_model': 'orca2:7b'
        }

        # Component references
        self.riva_tts = None
        self.riva_asr = None
        self.ollama_client = None
        self.conversation_context = None

        # Directories
        self.sounds_dir = "/var/lib/asterisk/sounds/custom"
        self.monitor_dir = "/var/spool/asterisk/monitor"
        self.temp_dir = "/tmp/netovo_voicebot"

        # Call state
        self.conversation_active = True
        self.components_initialized = False

        # Ensure directories exist
        self._ensure_directories()

        logger.info(f"VoiceBot initialized for call from {self.caller_id}")

    def _ensure_directories(self):
        """Create necessary directories with proper permissions"""
        try:
            directories = [self.sounds_dir, self.monitor_dir, self.temp_dir]
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                os.chmod(directory, 0o755)
            logger.info("Directory structure verified")
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")

    def initialize_components(self):
        """Initialize RIVA, Ollama, and conversation components"""
        try:
            logger.info("Initializing VoiceBot components...")

            # Import your working modules
            from riva_client import RivaASRClient, RivaTTSClient
            from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
            from conversation_context import ConversationContext

            logger.info("Modules imported successfully")

            # Initialize RIVA TTS
            logger.info("Connecting to RIVA TTS...")
            self.riva_tts = RivaTTSClient(self.config['riva_server'])
            tts_connected = self.riva_tts.connect()
            logger.info(f"RIVA TTS connected: {tts_connected}")

            # Initialize RIVA ASR
            logger.info("Connecting to RIVA ASR...")
            self.riva_asr = RivaASRClient(self.config['riva_server'])
            asr_connected = self.riva_asr.connect()
            logger.info(f"RIVA ASR connected: {asr_connected}")

            # Initialize Ollama
            logger.info("Connecting to Ollama...")
            self.ollama_client = OllamaClient(model=self.config['ollama_model'])
            ollama_connected = self.ollama_client.health_check()
            logger.info(f"Ollama connected: {ollama_connected}")

            # Initialize conversation context
            self.conversation_context = ConversationContext()

            # Check if minimum components are available
            if tts_connected and asr_connected and ollama_connected:
                self.components_initialized = True
                logger.info("All components initialized successfully")
                return True
            else:
                logger.warning("Some components failed to initialize")
                return False

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False

    def convert_audio_for_asterisk(self, input_wav_path):
        """Convert audio to Asterisk-compatible format using sox"""
        try:
            timestamp = int(time.time())
            output_filename = f"tts_{timestamp}.wav"
            output_path = os.path.join(self.sounds_dir, output_filename)

            # Use sox to convert to 8kHz mono wav (compatible with Asterisk)
            sox_cmd = [
                'sox', input_wav_path,
                '-r', '8000',      # 8kHz sample rate
                '-c', '1',         # Mono
                '-b', '16',        # 16-bit
                output_path
            ]

            result = subprocess.run(sox_cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and os.path.exists(output_path):
                os.chmod(output_path, 0o644)
                logger.info(f"Audio converted to Asterisk format: {output_path}")
                return f"custom/{output_filename.replace('.wav', '')}"  # Return without extension
            else:
                logger.error(f"Audio conversion failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None

    def play_tts_message(self, text):
        """Generate TTS and play through Asterisk"""
        try:
            if not self.riva_tts:
                logger.error("TTS not available")
                return False

            logger.info(f"Generating TTS: {text[:50]}...")

            # Generate TTS using your RIVA client
            tts_file_path = self.riva_tts.synthesize(text, sample_rate=22050)

            if not tts_file_path or not os.path.exists(tts_file_path):
                logger.error("TTS generation failed")
                return False

            # Convert for Asterisk
            asterisk_file = self.convert_audio_for_asterisk(tts_file_path)

            # Cleanup original TTS file
            try:
                os.unlink(tts_file_path)
            except:
                pass

            if not asterisk_file:
                logger.error("Audio conversion failed")
                return False

            # Play through Asterisk
            success = self.agi.stream_file(asterisk_file)

            if success:
                logger.info("TTS message played successfully")
            else:
                logger.error("Failed to play TTS message")

            return success

        except Exception as e:
            logger.error(f"TTS playback error: {e}")
            return False

    def record_and_transcribe_user_input(self):
        """Record user speech and transcribe using RIVA ASR"""
        try:
            if not self.riva_asr:
                logger.error("ASR not available")
                return ""

            timestamp = int(time.time())
            record_filename = f"user_input_{timestamp}"
            record_path = os.path.join(self.monitor_dir, record_filename)

            logger.info("Recording user input...")

            # Record audio using correct AGI syntax
            success = self.agi.record_file(
                record_filename,
                format="wav",
                escape_digits="#*",
                timeout=self.config['recording_timeout'],
                silence=self.config['silence_timeout']
            )

            if not success:
                logger.error("Recording command failed")
                return ""

            # Check if recording file exists
            wav_file = f"{record_path}.wav"
            if not os.path.exists(wav_file):
                logger.error(f"Recording file not found: {wav_file}")
                return ""

            # Check file size
            file_size = os.path.getsize(wav_file)
            if file_size < 1000:  # Less than 1KB indicates no speech
                logger.info("No speech detected (file too small)")
                try:
                    os.unlink(wav_file)
                except:
                    pass
                return ""

            logger.info(f"Speech recorded: {file_size} bytes")

            # Transcribe using your RIVA ASR client
            transcript = self.riva_asr.transcribe_file(wav_file)

            # Cleanup recording file
            try:
                os.unlink(wav_file)
            except:
                pass

            if transcript and transcript.strip():
                logger.info(f"Transcription: {transcript}")
                return transcript.strip()
            else:
                logger.info("No speech recognized")
                return ""

        except Exception as e:
            logger.error(f"Speech recording/transcription error: {e}")
            return ""

    def get_ai_response(self, user_input):
        """Get AI response using Ollama with conversation context"""
        try:
            if not self.ollama_client:
                return "I apologize, I'm having technical difficulties."

            if not user_input.strip():
                return "I didn't hear you clearly. Could you please repeat that?"

            logger.info(f"Getting AI response for: {user_input}")

            # Use conversation context to build prompt
            from ollama_client import VOICE_BOT_SYSTEM_PROMPT

            if self.conversation_context:
                full_prompt = self.conversation_context.build_prompt(user_input, VOICE_BOT_SYSTEM_PROMPT)
            else:
                full_prompt = f"{VOICE_BOT_SYSTEM_PROMPT}\n\nHuman: {user_input}\n\nAssistant:"

            # Generate response
            response = self.ollama_client.generate(
                prompt=user_input,
                system_prompt=VOICE_BOT_SYSTEM_PROMPT,
                max_tokens=75
            )

            if response and response.strip():
                # Add to conversation context
                if self.conversation_context:
                    self.conversation_context.add_turn(user_input, response)

                logger.info(f"AI Response: {response[:50]}...")
                return response.strip()
            else:
                logger.error("Empty AI response")
                return "I apologize, I'm having trouble processing that right now."

        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "I'm sorry, I'm experiencing technical difficulties. How else can I help you?"

    def run_conversation_loop(self):
        """Main conversation loop with full AI integration"""
        try:
            logger.info("Starting conversation loop...")
            turn_count = 0
            consecutive_failures = 0
            max_failures = 3

            while (self.conversation_active and
                   turn_count < self.config['max_conversation_turns'] and
                   consecutive_failures < max_failures):

                # Check call duration
                call_duration = (datetime.now() - self.call_start_time).total_seconds()
                if call_duration > self.config['max_call_duration']:
                    logger.info(f"Maximum call duration reached: {call_duration}s")
                    self.play_tts_message("Thank you for calling NETOVO. Have a great day!")
                    break

                turn_count += 1
                logger.info(f"Conversation turn {turn_count}")

                # Record and transcribe user input
                user_input = self.record_and_transcribe_user_input()

                if not user_input:
                    consecutive_failures += 1
                    logger.warning(f"No user input detected (failure {consecutive_failures}/{max_failures})")

                    if consecutive_failures == 1:
                        self.play_tts_message("I didn't catch that. Could you please speak clearly?")
                    elif consecutive_failures == 2:
                        self.play_tts_message("I'm having trouble hearing you. Please try again.")
                    else:
                        self.play_tts_message("I apologize for the difficulty. Let me transfer you to someone who can help.")
                        break
                    continue

                # Reset failure count on successful input
                consecutive_failures = 0

                # Check for conversation exit keywords
                user_lower = user_input.lower()
                exit_keywords = ['goodbye', 'bye', 'hang up', 'end call', 'thanks goodbye', 'that\'s all']
                if any(keyword in user_lower for keyword in exit_keywords):
                    self.play_tts_message("Thank you for calling NETOVO. Have a wonderful day!")
                    break

                # Check for escalation keywords
                escalation_keywords = ['human', 'agent', 'person', 'representative', 'manager', 'transfer']
                if any(keyword in user_lower for keyword in escalation_keywords):
                    self.play_tts_message("I'll connect you with one of our team members right away. Please hold.")
                    break

                # Check conversation context for escalation
                if self.conversation_context and self.conversation_context.should_escalate():
                    logger.info("Conversation context suggests escalation")
                    self.play_tts_message("Let me connect you with a specialist who can better assist you.")
                    break

                # Get AI response
                ai_response = self.get_ai_response(user_input)

                # Play AI response
                if not self.play_tts_message(ai_response):
                    logger.error("Failed to play AI response")
                    self.play_tts_message("I apologize, I'm having technical difficulties.")
                    break

                # Brief pause between turns
                self.agi.sleep(0.5)

            logger.info(f"Conversation completed after {turn_count} turns")

        except Exception as e:
            logger.error(f"Conversation loop error: {e}")
            try:
                self.play_tts_message("I apologize, we're experiencing technical difficulties.")
            except:
                pass

    def send_greeting(self):
        """Send initial greeting"""
        try:
            if self.conversation_context:
                greeting = self.conversation_context.get_greeting_prompt()
            else:
                greeting = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"

            if greeting:
                return self.play_tts_message(greeting)
            return True

        except Exception as e:
            logger.error(f"Greeting error: {e}")
            return False

    def handle_call(self):
        """Main call handling method"""
        call_success = False

        try:
            logger.info(f"=== Handling call from {self.caller_id} ===")

            # Step 1: Answer call
            if not self.agi.answer():
                logger.error("Failed to answer call")
                return

            # Brief pause for media establishment
            self.agi.sleep(1)

            # Step 2: Initialize components
            if not self.initialize_components():
                logger.error("Component initialization failed - using basic fallback")
                self.agi.verbose("VoiceBot component initialization failed")

                # Basic fallback
                try:
                    self.agi.stream_file("hello")
                    self.agi.sleep(2)
                    self.agi.stream_file("goodbye")
                except:
                    pass
                return

            # Step 3: Send greeting
            logger.info("Sending greeting...")
            if not self.send_greeting():
                logger.warning("Greeting failed - continuing anyway")

            # Step 4: Run conversation
            logger.info("Starting conversation...")
            self.run_conversation_loop()

            call_success = True
            logger.info("Call handling completed successfully")

        except Exception as e:
            logger.error(f"Critical call handling error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Emergency fallback
            try:
                self.play_tts_message("I apologize for the technical difficulty.")
            except:
                try:
                    self.agi.stream_file("pbx-invalid")
                except:
                    pass

        finally:
            # Graceful cleanup
            try:
                call_duration = (datetime.now() - self.call_start_time).total_seconds()
                logger.info(f"Call completed - Duration: {call_duration:.1f}s - Success: {call_success}")

                self.agi.sleep(1)
                if self.agi.connected:
                    self.agi.hangup()

            except Exception as cleanup_error:
                logger.error(f"Cleanup error: {cleanup_error}")

def main():
    """Main entry point for AGI script"""
    start_time = datetime.now()

    try:
        logger.info("=" * 60)
        logger.info("NETOVO Production VoiceBot Starting")
        logger.info(f"Start time: {start_time}")
        logger.info(f"Python version: {sys.version}")
        logger.info("=" * 60)

        # Create and run voicebot
        voicebot = NetovoProductionVoiceBot()
        voicebot.handle_call()

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info("NETOVO Production VoiceBot Completed")
        logger.info(f"End time: {end_time}")
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("VoiceBot interrupted")
    except SystemExit as e:
        logger.info(f"VoiceBot exiting with code: {e.code}")
    except Exception as e:
        logger.error("=" * 60)
        logger.error("FATAL ERROR IN MAIN")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Type: {type(e).__name__}")

        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("=" * 60)

        # Emergency AGI fallback
        try:
            emergency_agi = ProductionAGI()
            emergency_agi.answer()
            emergency_agi.verbose("VoiceBot fatal error")
            emergency_agi.stream_file("pbx-invalid")
            emergency_agi.sleep(2)
            emergency_agi.hangup()
        except:
            pass

if __name__ == "__main__":
    main()
