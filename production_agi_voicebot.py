#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
NETOVO Production AGI VoiceBot - No Sudo Required
Fixed version that doesn't require sudo and handles TTS failures gracefully
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

# Set up logging
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
            elif result.startswith('510') or result.startswith('511'):
                logger.error(f"AGI command failed: {cmd} -> {result}")
                return "ERROR"
            else:
                return result
                
        except Exception as e:
            logger.error(f"AGI command error: {e}")
            self.connected = False
            return "ERROR"

    def verbose(self, message, level=1):
        """Send verbose message"""
        # Escape quotes in message
        escaped_message = message.replace('"', '\\"')
        return self.command(f'VERBOSE "{escaped_message}" {level}')

    def answer(self):
        """Answer call"""
        result = self.command("ANSWER")
        success = result != "ERROR"
        logger.info(f"Call answer result: {success}")
        return success

    def hangup(self):
        """Hangup call"""
        if self.connected:
            self.command("HANGUP")
            self.connected = False

    def stream_file(self, filename, escape_digits=""):
        """Stream audio file - remove extension for Asterisk"""
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        
        result = self.command(f'STREAM FILE "{filename}" "{escape_digits}"')
        success = result != "ERROR"
        logger.info(f"Stream file '{filename}' result: {success}")
        return success

    def record_file(self, filename, format="wav", escape_digits="#", timeout=10000, max_silence=3000):
        """Record audio with timeout and silence detection"""
        result = self.command(f'RECORD FILE "{filename}" "{format}" "{escape_digits}" {timeout} 0 0 {max_silence}')
        success = result != "ERROR"
        logger.info(f"Record file result: {success}")
        return success

    def wait(self, seconds):
        """Wait for specified seconds"""
        return self.command(f"WAIT {seconds}")

class NetovoProductionVoiceBot:
    """Production VoiceBot with no sudo requirements"""

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

        # Component references
        self.riva_tts = None
        self.riva_asr = None  
        self.ollama_client = None
        self.conversation = None
        
        # Directories - use /tmp to avoid permission issues
        self.sounds_dir = "/tmp/asterisk_sounds"
        self.monitor_dir = "/tmp/asterisk_monitor"
        
        # Call state
        self.call_start_time = time.time()
        self.conversation_active = True
        
        # Create directories with proper permissions
        try:
            os.makedirs(self.sounds_dir, mode=0o755, exist_ok=True)
            os.makedirs(self.monitor_dir, mode=0o755, exist_ok=True)
        except Exception as e:
            logger.error(f"Directory creation failed: {e}")

    def safe_import_modules(self):
        """Safely import required modules"""
        try:
            logger.info("Importing VoiceBot modules...")
            
            from riva_client import RivaASRClient, RivaTTSClient
            from ollama_client import OllamaClient
            from conversation_context import ConversationContext
            
            logger.info("All modules imported successfully")
            return True, (RivaASRClient, RivaTTSClient, OllamaClient, ConversationContext)
            
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
            
            RivaASRClient, RivaTTSClient, OllamaClient, ConversationContext = modules
            
            # Initialize RIVA components
            self.riva_tts = RivaTTSClient(self.config['riva_server'])
            self.riva_asr = RivaASRClient(self.config['riva_server'])
            
            riva_tts_ok = self.riva_tts.connect()
            riva_asr_ok = self.riva_asr.connect()
            
            logger.info(f"RIVA TTS connected: {riva_tts_ok}")
            logger.info(f"RIVA ASR connected: {riva_asr_ok}")

            # Initialize Ollama
            self.ollama_client = OllamaClient(
                base_url=self.config['ollama_url'],
                model=self.config['ollama_model']
            )
            
            ollama_ok = self.ollama_client.health_check()
            logger.info(f"Ollama connected: {ollama_ok}")

            # Initialize conversation context
            self.conversation = ConversationContext()

            # Return True if at least some components work
            if riva_tts_ok or ollama_ok:
                logger.info("Partial component initialization successful")
                return True
            else:
                logger.error("All component initialization failed")
                return False

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False

    def convert_audio_simple(self, input_wav):
        """Convert audio without requiring sudo"""
        try:
            timestamp = int(time.time())
            base_name = f"voicebot_{timestamp}"
            
            # Try to copy to Asterisk sounds directory
            try:
                asterisk_sounds = "/var/lib/asterisk/sounds/custom"
                os.makedirs(asterisk_sounds, exist_ok=True)
                
                output_path = os.path.join(asterisk_sounds, f"{base_name}.wav")
                
                # Simple copy without format conversion first
                import shutil
                shutil.copy2(input_wav, output_path)
                os.chmod(output_path, 0o644)
                
                logger.info(f"Audio copied to Asterisk sounds: {output_path}")
                return f"custom/{base_name}"
                
            except Exception as e:
                logger.warning(f"Cannot copy to Asterisk sounds dir: {e}")
                
                # Fallback: try to use sox without sudo
                try:
                    output_path = os.path.join(self.sounds_dir, f"{base_name}.wav")
                    
                    # Try sox without sudo
                    cmd = ['sox', input_wav, '-r', '8000', '-c', '1', output_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0 and os.path.exists(output_path):
                        # Create symlink to Asterisk directory if possible
                        try:
                            asterisk_link = f"/var/lib/asterisk/sounds/custom/{base_name}.wav"
                            os.symlink(output_path, asterisk_link)
                            return f"custom/{base_name}"
                        except:
                            # Use absolute path as fallback
                            return output_path.replace('.wav', '')
                    else:
                        logger.error(f"Sox conversion failed: {result.stderr}")
                        return None
                        
                except Exception as e:
                    logger.error(f"Sox fallback failed: {e}")
                    return None
                
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None

    def play_tts_message(self, text):
        """Generate and play TTS message with fallbacks"""
        try:
            if not self.riva_tts:
                logger.error("TTS not initialized")
                return self.play_fallback_message()

            logger.info(f"Generating TTS: {text[:50]}...")
            
            # Generate TTS
            tts_file = self.riva_tts.synthesize(text, sample_rate=8000)  # Use 8kHz directly
            
            if not tts_file or not os.path.exists(tts_file):
                logger.error("TTS generation failed")
                return self.play_fallback_message()

            # Convert audio for Asterisk
            asterisk_file = self.convert_audio_simple(tts_file)
            
            # Cleanup temp file
            try:
                os.unlink(tts_file)
            except:
                pass

            if not asterisk_file:
                logger.error("Audio conversion failed")
                return self.play_fallback_message()

            # Play the file
            result = self.agi.stream_file(asterisk_file)
            
            if result:
                logger.info("TTS message played successfully")
            else:
                logger.error("Failed to stream TTS file")
                return self.play_fallback_message()
                
            return result

        except Exception as e:
            logger.error(f"TTS playback error: {e}")
            return self.play_fallback_message()

    def play_fallback_message(self):
        """Play fallback message when TTS fails"""
        try:
            # Try common Asterisk sound files
            fallback_sounds = ["hello", "beep", "demo-thanks", "digits/1"]
            
            for sound in fallback_sounds:
                if self.agi.stream_file(sound):
                    logger.info(f"Played fallback sound: {sound}")
                    return True
                    
            logger.error("All fallback sounds failed")
            return False
            
        except Exception as e:
            logger.error(f"Fallback message error: {e}")
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
                logger.info("No speech detected")
                os.unlink(wav_file)
                return ""

            logger.info(f"Speech recorded: {file_size} bytes")

            # Transcribe
            transcript = self.riva_asr.transcribe_file(wav_file)
            
            # Cleanup
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

            system_prompt = "You are Alexis, a helpful customer service AI for NETOVO. Keep responses under 40 words for voice calls. Be professional and direct."

            response = self.ollama_client.generate(
                prompt=user_input,
                system_prompt=system_prompt,
                max_tokens=60
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
            return "I apologize for the technical difficulty."

    def run_simple_interaction(self):
        """Run a simple interaction without complex conversation loop"""
        try:
            logger.info("Starting simple interaction")
            
            # Send greeting
            greeting = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"
            
            if not self.play_tts_message(greeting):
                logger.error("Greeting failed")
                return False

            # Wait a moment
            self.agi.wait(2)

            # Try one interaction
            user_input = self.record_and_transcribe()
            
            if user_input:
                logger.info(f"User said: {user_input}")
                
                ai_response = self.get_ai_response(user_input)
                
                if not self.play_tts_message(ai_response):
                    logger.error("Response playback failed")
                    return False
            else:
                self.play_tts_message("I didn't hear anything. Thank you for calling.")

            # Goodbye
            self.play_tts_message("Thank you for calling NETOVO. Have a great day!")
            
            return True

        except Exception as e:
            logger.error(f"Simple interaction error: {e}")
            return False

    def handle_call(self):
        """Main call handler with extensive fallbacks"""
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
            self.agi.wait(1)

            # Send verbose message to Asterisk
            self.agi.verbose("NETOVO VoiceBot call handler active")

            # Try to initialize components
            if self.initialize_components():
                logger.info("Components initialized - starting interaction")
                
                # Run simple interaction
                if not self.run_simple_interaction():
                    logger.error("Interaction failed - using basic fallback")
                    self.play_fallback_message()
                    self.agi.wait(3)
            else:
                logger.error("Component initialization failed - using fallback")
                self.agi.verbose("VoiceBot components failed to initialize")
                
                # Basic fallback interaction
                self.play_fallback_message()  # Play hello/beep
                self.agi.wait(2)
                self.play_fallback_message()  # Another sound
                self.agi.wait(3)

        except Exception as e:
            logger.error(f"Call handling error: {e}")
            if call_answered:
                try:
                    self.play_fallback_message()
                    self.agi.wait(2)
                except:
                    pass
        
        finally:
            try:
                logger.info("Ending call")
                self.agi.wait(1)  # Brief pause before hangup
                if self.agi.connected:
                    self.agi.hangup()
                logger.info("Call ended gracefully")
            except Exception as e:
                logger.error(f"Hangup error: {e}")

def main():
    """Main entry point"""
    try:
        logger.info("=== NETOVO Production VoiceBot Starting ===")
        
        voice_bot = NetovoProductionVoiceBot()
        voice_bot.handle_call()
        
        logger.info("=== NETOVO Production VoiceBot Completed ===")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        # Emergency fallback
        try:
            agi = ProductionAGI()
            agi.answer()
            agi.verbose("VoiceBot fatal error occurred")
            agi.stream_file("pbx-invalid")
            agi.wait(2)
            agi.hangup()
        except:
            pass

if __name__ == "__main__":
    main()
