#!/usr/bin/env python3
"""
NETOVO Production AGI VoiceBot
Handles complete conversation flow with proper error handling
"""

import sys
import os
import tempfile
import time
import subprocess
import wave
from pathlib import Path
import logging

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from riva_client import RivaASRClient, RivaTTSClient
    from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
    from audio_processing import AudioProcessor
    from conversation_context import ConversationContext
except ImportError as e:
    print(f"IMPORT_ERROR: {e}", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VoiceBot - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/voicebot.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class ProductionAGI:
    """Production-ready AGI interface with proper error handling"""

    def __init__(self):
        self.env = {}
        self.connected = True
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
            
            # Check for success (200 result code)
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
        return result != "ERROR"

    def hangup(self):
        """Hangup call"""
        self.command("HANGUP")
        self.connected = False

    def stream_file(self, filename, escape_digits=""):
        """Stream audio file"""
        # Remove any file extension for Asterisk
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        result = self.command(f'STREAM FILE "{filename}" "{escape_digits}"')
        return result != "ERROR"

    def record_file(self, filename, format="wav", escape_digits="#", timeout=10000, max_silence=3000):
        """Record audio with timeout and silence detection"""
        result = self.command(f'RECORD FILE "{filename}" "{format}" "{escape_digits}" {timeout} 0 0 {max_silence}')
        return result != "ERROR"

    def get_data(self, filename, timeout=5000, max_digits=1):
        """Play file and wait for DTMF input"""
        result = self.command(f'GET DATA "{filename}" {timeout} {max_digits}')
        if result.startswith('200') and 'result=' in result:
            # Extract the digit from result
            try:
                digit = result.split('result=')[1].split()[0]
                return digit if digit != '0' else ""
            except:
                return ""
        return ""

class NetovoProductionVoiceBot:
    """Production NETOVO Voice Bot with complete conversation handling"""

    def __init__(self):
        self.agi = ProductionAGI()
        self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
        self.channel = self.agi.env.get('agi_channel', 'Unknown')
        
        # Configuration
        self.config = {
            'riva_server': 'localhost:50051',
            'ollama_url': 'http://127.0.0.1:11434',
            'ollama_model': 'orca2:7b',
            'max_conversation_turns': 10,
            'response_timeout': 10,
            'silence_timeout': 3000,
            'max_call_duration': 300  # 5 minutes max
        }

        # Components
        self.riva_tts = None
        self.riva_asr = None
        self.ollama_client = None
        self.audio_processor = None
        self.conversation = None
        
        # Asterisk directories
        self.sounds_dir = "/var/lib/asterisk/sounds/custom"
        self.monitor_dir = "/var/spool/asterisk/monitor"
        
        # Call state
        self.call_start_time = time.time()
        self.conversation_active = True
        
        # Create directories
        os.makedirs(self.sounds_dir, exist_ok=True)
        os.makedirs(self.monitor_dir, exist_ok=True)

    def initialize_components(self):
        """Initialize all voice bot components"""
        try:
            logger.info(f"Initializing VoiceBot for caller {self.caller_id}")
            
            # Initialize RIVA components
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

            # Initialize audio processor and conversation context
            self.audio_processor = AudioProcessor()
            self.conversation = ConversationContext()

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False

    def convert_wav_for_asterisk(self, input_wav):
        """Convert WAV to 8kHz µ-law format for Asterisk"""
        try:
            # Output file in sounds directory
            base_name = f"voicebot_{int(time.time())}"
            output_wav = os.path.join(self.sounds_dir, f"{base_name}.wav")
            
            # Use sox to convert to 8kHz µ-law
            cmd = [
                'sox', input_wav, 
                '-r', '8000',  # 8kHz sample rate
                '-c', '1',     # Mono
                '-e', 'mu-law', # µ-law encoding
                output_wav
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(output_wav):
                # Set proper permissions
                os.chmod(output_wav, 0o644)
                logger.debug(f"Audio converted for Asterisk: {output_wav}")
                return f"custom/{base_name}"
            else:
                logger.error(f"Audio conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None

    def play_tts_message(self, text):
        """Generate TTS and play through Asterisk"""
        try:
            logger.info(f"Generating TTS: {text[:50]}...")
            
            # Generate TTS at 22kHz (RIVA's native rate)
            tts_file = self.riva_tts.synthesize(text, sample_rate=22050)
            
            if not tts_file or not os.path.exists(tts_file):
                logger.error("TTS generation failed")
                return False

            # Convert for Asterisk telephony
            asterisk_file = self.convert_wav_for_asterisk(tts_file)
            
            # Cleanup original TTS file
            try:
                os.unlink(tts_file)
            except:
                pass

            if not asterisk_file:
                logger.error("Audio conversion failed")
                return False

            # Play through Asterisk
            result = self.agi.stream_file(asterisk_file)
            
            if result:
                logger.info("TTS message played successfully")
            else:
                logger.error("Failed to play TTS message")
                
            return result

        except Exception as e:
            logger.error(f"TTS playback error: {e}")
            return False

    def record_user_speech(self):
        """Record user speech and transcribe"""
        try:
            # Generate unique filename
            timestamp = int(time.time())
            record_file = os.path.join(self.monitor_dir, f"user_speech_{timestamp}")
            
            logger.info("Recording user speech...")
            
            # Record with silence detection (3 seconds of silence ends recording)
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

            # Check if file exists
            wav_file = f"{record_file}.wav"
            if not os.path.exists(wav_file):
                logger.error(f"Recording file not found: {wav_file}")
                return ""

            # Check file size
            file_size = os.path.getsize(wav_file)
            if file_size < 1000:  # Less than 1KB indicates no speech
                logger.info("No speech detected (file too small)")
                os.unlink(wav_file)
                return ""

            logger.info(f"Speech recorded: {file_size} bytes")

            # Transcribe with RIVA ASR
            transcript = self.riva_asr.transcribe_file(wav_file)
            
            # Cleanup recording
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
        """Get AI response from Ollama"""
        try:
            if not user_input.strip():
                return "I didn't hear you clearly. Could you please repeat that?"

            logger.info(f"Getting AI response for: {user_input}")

            # Build context-aware prompt
            if self.conversation:
                prompt = self.conversation.build_prompt(user_input, VOICE_BOT_SYSTEM_PROMPT)
            else:
                prompt = f"{VOICE_BOT_SYSTEM_PROMPT}\n\nHuman: {user_input}\n\nAssistant:"

            # Generate response
            response = self.ollama_client.generate(
                prompt=user_input,
                system_prompt=VOICE_BOT_SYSTEM_PROMPT,
                max_tokens=75  # Keep responses concise for voice
            )

            if response and response.strip():
                # Add to conversation history
                if self.conversation:
                    self.conversation.add_turn(user_input, response)
                
                logger.info(f"AI Response: {response[:50]}...")
                return response.strip()
            else:
                logger.error("Empty AI response")
                return "I apologize, I'm having trouble processing that right now. How else can I help you?"

        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "I apologize for the technical difficulty. How can I assist you today?"

    def handle_conversation_loop(self):
        """Main conversation loop"""
        try:
            turn_count = 0
            consecutive_failures = 0
            
            while (self.conversation_active and 
                   turn_count < self.config['max_conversation_turns'] and
                   consecutive_failures < 3):
                
                # Check call duration
                if time.time() - self.call_start_time > self.config['max_call_duration']:
                    self.play_tts_message("Thank you for calling NETOVO. Have a great day!")
                    break

                turn_count += 1
                logger.info(f"Conversation turn {turn_count}")

                # Record user input
                user_input = self.record_user_speech()

                if not user_input:
                    consecutive_failures += 1
                    
                    if consecutive_failures == 1:
                        self.play_tts_message("I didn't catch that. Could you please speak clearly?")
                    elif consecutive_failures == 2:
                        self.play_tts_message("I'm having trouble hearing you. Please speak louder or check your connection.")
                    else:
                        self.play_tts_message("I apologize for the technical difficulty. Let me transfer you to a human agent.")
                        break
                    continue

                # Reset failure count on successful input
                consecutive_failures = 0

                # Check for call termination keywords
                user_lower = user_input.lower()
                if any(word in user_lower for word in ['goodbye', 'bye', 'hang up', 'end call', 'thank you']):
                    self.play_tts_message("Thank you for calling NETOVO. Have a wonderful day!")
                    break

                # Check for escalation requests
                if any(word in user_lower for word in ['human', 'agent', 'person', 'representative', 'manager']):
                    self.play_tts_message("I'll connect you with a human agent right away. Please hold.")
                    # Here you would implement transfer logic
                    break

                # Get AI response
                ai_response = self.get_ai_response(user_input)
                
                # Play AI response
                if not self.play_tts_message(ai_response):
                    logger.error("Failed to play AI response")
                    break

                # Brief pause between turns
                time.sleep(0.5)

            logger.info(f"Conversation ended after {turn_count} turns")

        except Exception as e:
            logger.error(f"Conversation loop error: {e}")
            self.play_tts_message("I apologize, but we're experiencing technical difficulties. Please call back later.")

    def handle_call(self):
        """Main call handling workflow"""
        call_handled = False
        
        try:
            logger.info(f"Handling call from {self.caller_id} on channel {self.channel}")
            
            # Answer the call first
            if not self.agi.answer():
                logger.error("Failed to answer call")
                return

            logger.info("Call answered successfully")
            call_handled = True

            # Brief pause to ensure media is established
            time.sleep(1)

            # Initialize components
            if not self.initialize_components():
                logger.error("Component initialization failed - using fallback")
                self.agi.stream_file("pbx-invalid")
                time.sleep(2)
                return

            # Send greeting
            greeting = self.conversation.get_greeting_prompt()
            if not greeting:
                greeting = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"

            if not self.play_tts_message(greeting):
                logger.error("Failed to play greeting - using fallback")
                self.agi.stream_file("hello")
                time.sleep(2)
                return

            # Start conversation loop
            self.handle_conversation_loop()

        except Exception as e:
            logger.error(f"Call handling error: {e}")
            if call_handled:
                try:
                    self.play_tts_message("I apologize for the technical difficulty. Please call back later.")
                except:
                    pass
        
        finally:
            # Ensure proper cleanup
            try:
                # Brief pause before hangup to ensure last message is heard
                time.sleep(2)
                self.agi.hangup()
                logger.info("Call ended gracefully")
            except:
                pass

            # Cleanup temp files
            self.cleanup_temp_files()

    def cleanup_temp_files(self):
        """Clean up temporary audio files"""
        try:
            # Clean files older than 1 hour from sounds directory
            current_time = time.time()
            for file_path in Path(self.sounds_dir).glob("voicebot_*.wav"):
                if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                    file_path.unlink()
            
            # Clean old recordings
            for file_path in Path(self.monitor_dir).glob("user_speech_*.wav"):
                if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                    file_path.unlink()
                    
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main entry point"""
    try:
        logger.info("NETOVO Production VoiceBot starting")
        
        # Create and run voice bot
        voice_bot = NetovoProductionVoiceBot()
        voice_bot.handle_call()
        
        logger.info("NETOVO Production VoiceBot completed")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        # Ensure we don't leave the call hanging
        try:
            agi = ProductionAGI()
            agi.hangup()
        except:
            pass

if __name__ == "__main__":
    main()
