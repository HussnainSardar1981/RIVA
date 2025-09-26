#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
NETOVO Production AGI VoiceBot - Fixed Version
Handles complete conversation with proper AGI error handling
"""

import sys
import os
import time
import tempfile
import subprocess
import wave
from pathlib import Path

# Set up paths BEFORE any other imports
project_dir = "/home/aiadmin/netovo_voicebot"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Set up logging to voicebot.log as requested
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VoiceBot - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/voicebot.log', mode='a'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class ProductionAGI:
    """Production AGI interface with robust error handling"""

    def __init__(self):
        self.env = {}
        self.connected = True
        logger.info("AGI interface initializing")
        self._parse_environment()

    def _parse_environment(self):
        """Parse AGI environment from stdin"""
        try:
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
            logger.info(f"AGI environment parsed: {len(self.env)} variables")
        except Exception as e:
            logger.error(f"Environment parsing error: {e}")

    def command(self, cmd):
        """Send AGI command with error handling"""
        if not self.connected:
            return "ERROR"
        
        try:
            logger.debug(f"AGI Command: {cmd}")
            sys.stdout.write(f"{cmd}\n")
            sys.stdout.flush()
            
            result = sys.stdin.readline()
            if not result:
                self.connected = False
                return "ERROR"
                
            result = result.strip()
            logger.debug(f"AGI Response: {result}")
            
            if result.startswith('200'):
                return result
            elif result.startswith('510'):
                logger.error(f"Invalid AGI command: {cmd}")
                return "ERROR"
            else:
                return result
                
        except Exception as e:
            logger.error(f"AGI command error: {e}")
            self.connected = False
            return "ERROR"

    def verbose(self, message, level=1):
        """Send verbose message"""
        return self.command(f'VERBOSE "{message}" {level}')

    def answer(self):
        """Answer call"""
        result = self.command("ANSWER")
        success = result != "ERROR" and not result.startswith('510')
        logger.info(f"Call answer result: {success}")
        return success

    def hangup(self):
        """Hangup call"""
        self.command("HANGUP")
        self.connected = False

    def stream_file(self, filename, escape_digits=""):
        """Stream audio file"""
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        result = self.command(f'STREAM FILE "{filename}" "{escape_digits}"')
        success = result != "ERROR" and not result.startswith('510')
        logger.info(f"Stream file '{filename}' result: {success}")
        return success

    def record_file(self, filename, format="wav", escape_digits="#", timeout=10000, max_silence=3000):
        """Record audio with timeout and silence detection"""
        result = self.command(f'RECORD FILE "{filename}" "{format}" "{escape_digits}" {timeout} 0 0 {max_silence}')
        success = result != "ERROR" and not result.startswith('510')
        logger.info(f"Record file result: {success}")
        return success

class NetovoProductionVoiceBot:
    """Production VoiceBot with minimal dependencies for stability"""

    def __init__(self):
        self.agi = ProductionAGI()
        self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
        self.channel = self.agi.env.get('agi_channel', 'Unknown')
        
        # Configuration
        self.config = {
            'max_conversation_turns': 5,
            'response_timeout': 8,
            'silence_timeout': 3000,
            'max_call_duration': 180,
            'riva_server': 'localhost:50051',
            'ollama_url': 'http://127.0.0.1:11434',
            'ollama_model': 'orca2:7b'
        }

        # Component references (will be initialized later)
        self.riva_tts = None
        self.riva_asr = None  
        self.ollama_client = None
        self.audio_processor = None
        self.conversation = None
        
        # Directories
        self.sounds_dir = "/var/lib/asterisk/sounds/custom"
        self.monitor_dir = "/var/spool/asterisk/monitor"
        
        # Call state
        self.call_start_time = time.time()
        self.conversation_active = True
        
        # Create directories
        try:
            os.makedirs(self.sounds_dir, exist_ok=True)
            os.makedirs(self.monitor_dir, exist_ok=True)
        except:
            pass

    def safe_import_modules(self):
        """Safely import required modules with fallbacks"""
        try:
            logger.info("Attempting to import VoiceBot modules...")
            
            # Import modules one by one
            from riva_client import RivaASRClient, RivaTTSClient
            from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
            from audio_processing import AudioProcessor
            from conversation_context import ConversationContext
            
            logger.info("All modules imported successfully")
            return True, (RivaASRClient, RivaTTSClient, OllamaClient, VOICE_BOT_SYSTEM_PROMPT, AudioProcessor, ConversationContext)
            
        except Exception as e:
            logger.error(f"Module import failed: {e}")
            return False, None

    def initialize_components(self):
        """Initialize components with error handling"""
        try:
            logger.info("Initializing VoiceBot components...")
            
            # Import modules
            success, modules = self.safe_import_modules()
            if not success:
                logger.error("Cannot import required modules")
                return False
            
            RivaASRClient, RivaTTSClient, OllamaClient, VOICE_BOT_SYSTEM_PROMPT, AudioProcessor, ConversationContext = modules
            
            # Initialize components
            self.riva_tts = RivaTTSClient(self.config['riva_server'])
            self.riva_asr = RivaASRClient(self.config['riva_server'])
            
            if not self.riva_tts.connect():
                logger.error("RIVA TTS connection failed")
                return False
                
            if not self.riva_asr.connect():
                logger.error("RIVA ASR connection failed")
                return False

            # Initialize Ollama
            self.ollama_client = OllamaClient(
                base_url=self.config['ollama_url'],
                model=self.config['ollama_model']
            )
            
            if not self.ollama_client.health_check():
                logger.error("Ollama health check failed")
                return False

            # Initialize other components
            self.audio_processor = AudioProcessor()
            self.conversation = ConversationContext()

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False

    def convert_audio_for_asterisk(self, input_wav):
        """Convert audio to Asterisk-compatible format"""
        try:
            timestamp = int(time.time())
            base_name = f"voicebot_{timestamp}"
            output_path = os.path.join(self.sounds_dir, f"{base_name}.wav")
            
            # Use sox to convert to 8kHz µ-law
            cmd = [
                'sox', input_wav,
                '-r', '8000',
                '-c', '1', 
                '-e', 'mu-law',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(output_path):
                os.chmod(output_path, 0o644)
                logger.info(f"Audio converted: {output_path}")
                return f"custom/{base_name}"
            else:
                logger.error(f"Audio conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None

    def play_tts_message(self, text):
        """Generate and play TTS message"""
        try:
            if not self.riva_tts:
                logger.error("TTS not initialized")
                return False

            logger.info(f"Generating TTS: {text[:50]}...")
            
            tts_file = self.riva_tts.synthesize(text, sample_rate=22050)
            
            if not tts_file or not os.path.exists(tts_file):
                logger.error("TTS generation failed")
                return False

            asterisk_file = self.convert_audio_for_asterisk(tts_file)
            
            # Cleanup temp file
            try:
                os.unlink(tts_file)
            except:
                pass

            if not asterisk_file:
                logger.error("Audio conversion failed")
                return False

            result = self.agi.stream_file(asterisk_file)
            return result

        except Exception as e:
            logger.error(f"TTS playback error: {e}")
            return False

    def record_and_transcribe(self):
        """Record user speech and transcribe"""
        try:
            if not self.riva_asr:
                logger.error("ASR not initialized")
                return ""

            timestamp = int(time.time())
            record_file = os.path.join(self.monitor_dir, f"user_speech_{timestamp}")
            
            logger.info("Recording user speech...")
            
            success = self.agi.record_file(
                record_file,
                format="wav",
                escape_digits="#",
                timeout=self.config['response_timeout'] * 1000,
                max_silence=self.config['silence_timeout']
            )
            
            if not success:
                logger.error("Recording failed")
                return ""

            wav_file = f"{record_file}.wav"
            if not os.path.exists(wav_file):
                logger.error(f"Recording file not found: {wav_file}")
                return ""

            file_size = os.path.getsize(wav_file)
            if file_size < 1000:
                logger.info("No speech detected (file too small)")
                os.unlink(wav_file)
                return ""

            logger.info(f"Speech recorded: {file_size} bytes")

            transcript = self.riva_asr.transcribe_file(wav_file)
            
            try:
                os.unlink(wav_file)
            except:
                pass

            if transcript and transcript.strip():
                logger.info(f"Transcription: {transcript}")
                return transcript.strip()
            else:
                logger.info("No speech understood")
                return ""

        except Exception as e:
            logger.error(f"Speech recording error: {e}")
            return ""

    def get_ai_response(self, user_input):
        """Get AI response"""
        try:
            if not self.ollama_client:
                return "I apologize, I'm having technical difficulties."

            if not user_input.strip():
                return "I didn't hear you clearly. Could you please repeat that?"

            logger.info(f"Getting AI response for: {user_input}")

            response = self.ollama_client.generate(
                prompt=user_input,
                system_prompt="You are Alexis, a helpful customer service AI for NETOVO. Keep responses under 50 words for voice calls. Be professional and direct.",
                max_tokens=75
            )

            if response and response.strip():
                if self.conversation:
                    self.conversation.add_turn(user_input, response)
                
                logger.info(f"AI Response: {response[:50]}...")
                return response.strip()
            else:
                logger.error("Empty AI response")
                return "I apologize, I'm having trouble processing that right now."

        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "I apologize for the technical difficulty. How can I assist you today?"

    def run_conversation_loop(self):
        """Main conversation loop"""
        try:
            turn_count = 0
            consecutive_failures = 0
            
            while (self.conversation_active and 
                   turn_count < self.config['max_conversation_turns'] and
                   consecutive_failures < 3):
                
                if time.time() - self.call_start_time > self.config['max_call_duration']:
                    self.play_tts_message("Thank you for calling NETOVO. Have a great day!")
                    break

                turn_count += 1
                logger.info(f"Conversation turn {turn_count}")

                user_input = self.record_and_transcribe()

                if not user_input:
                    consecutive_failures += 1
                    
                    if consecutive_failures == 1:
                        self.play_tts_message("I didn't catch that. Could you please speak clearly?")
                    elif consecutive_failures == 2:
                        self.play_tts_message("I'm having trouble hearing you.")
                    else:
                        self.play_tts_message("I apologize for the difficulty. Let me transfer you.")
                        break
                    continue

                consecutive_failures = 0

                # Check for exit keywords
                user_lower = user_input.lower()
                if any(word in user_lower for word in ['goodbye', 'bye', 'hang up', 'end call', 'thank you']):
                    self.play_tts_message("Thank you for calling NETOVO. Have a wonderful day!")
                    break

                # Check for escalation
                if any(word in user_lower for word in ['human', 'agent', 'person', 'representative']):
                    self.play_tts_message("I'll connect you with a human agent right away.")
                    break

                ai_response = self.get_ai_response(user_input)
                
                if not self.play_tts_message(ai_response):
                    logger.error("Failed to play AI response")
                    break

                time.sleep(0.5)

            logger.info(f"Conversation ended after {turn_count} turns")

        except Exception as e:
            logger.error(f"Conversation loop error: {e}")
            self.play_tts_message("I apologize, but we're experiencing technical difficulties.")

    def handle_call(self):
        """Main call handler"""
        call_answered = False
        
        try:
            logger.info(f"Handling call from {self.caller_id} on channel {self.channel}")
            
            # Answer call
            if not self.agi.answer():
                logger.error("Failed to answer call")
                return

            call_answered = True
            logger.info("Call answered successfully")
            
            # Brief pause for media setup
            time.sleep(1)

            # Try to initialize components
            if not self.initialize_components():
                logger.error("Component initialization failed - using fallback")
                self.agi.verbose("VoiceBot initialization failed")
                self.agi.stream_file("hello")
                time.sleep(3)
                self.agi.stream_file("goodbye")
                return

            # Send greeting
            greeting = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"
            
            if not self.play_tts_message(greeting):
                logger.error("Failed to play greeting - using fallback")
                self.agi.stream_file("hello")
                time.sleep(2)

            # Start conversation
            self.run_conversation_loop()

        except Exception as e:
            logger.error(f"Call handling error: {e}")
            if call_answered:
                try:
                    self.play_tts_message("I apologize for the technical difficulty.")
                except:
                    self.agi.stream_file("pbx-invalid")
        
        finally:
            try:
                time.sleep(1)
                if self.agi.connected:
                    self.agi.hangup()
                logger.info("Call ended")
            except:
                pass

def main():
    """Main entry point"""
    try:
        logger.info("=== NETOVO Production VoiceBot Starting ===")
        
        voice_bot = NetovoProductionVoiceBot()
        voice_bot.handle_call()
        
        logger.info("=== NETOVO Production VoiceBot Completed ===")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        try:
            # Emergency fallback
            agi = ProductionAGI()
            agi.answer()
            agi.verbose("VoiceBot fatal error")
            agi.stream_file("pbx-invalid")
            time.sleep(2)
            agi.hangup()
        except:
            pass

if __name__ == "__main__":
    main()
