#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
Working AGI VoiceBot using your existing files with correct AGI syntax
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path

# Add your project directory
project_dir = "/home/aiadmin/netovo_voicebot"
sys.path.insert(0, project_dir)

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VoiceBot - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/voicebot.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class WorkingAGI:
    """AGI with correct Asterisk command syntax"""
    
    def __init__(self):
        self.env = {}
        self.connected = True
        self._parse_environment()
    
    def _parse_environment(self):
        """Parse AGI environment"""
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            if ':' in line:
                key, value = line.split(':', 1)
                self.env[key.strip()] = value.strip()
        logger.info(f"AGI environment parsed: {len(self.env)} variables")
    
    def command(self, cmd):
        """Send AGI command"""
        try:
            logger.debug(f"AGI Command: {cmd}")
            print(cmd)
            sys.stdout.flush()
            
            result = sys.stdin.readline().strip()
            logger.debug(f"AGI Response: {result}")
            return result
        except Exception as e:
            logger.error(f"AGI command error: {e}")
            self.connected = False
            return "ERROR"
    
    def answer(self):
        """Answer call"""
        result = self.command("ANSWER")
        success = result.startswith('200')
        logger.info(f"Answer result: {success}")
        return success
    
    def hangup(self):
        """Hangup call"""
        if self.connected:
            self.command("HANGUP")
            self.connected = False
    
    def verbose(self, message):
        """Send verbose message"""
        # Escape quotes properly
        escaped = message.replace('"', '\\"')
        return self.command(f'VERBOSE "{escaped}"')
    
    def stream_file(self, filename, escape_digits=""):
        """Play audio file - correct syntax"""
        # Remove extension if present
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        
        result = self.command(f'STREAM FILE {filename} "{escape_digits}"')
        success = result.startswith('200')
        logger.info(f"Stream file '{filename}' result: {success}")
        return success
    
    def record_file(self, filename, format="wav", escape_digits="#", timeout=10000, max_silence=3000):
        """Record audio file - CORRECT AGI syntax"""
        # AGI RECORD FILE syntax: RECORD FILE filename format escape_digits timeout [offset] [BEEP] [silence]
        result = self.command(f'RECORD FILE {filename} {format} {escape_digits} {timeout} BEEP {max_silence}')
        success = result.startswith('200')
        logger.info(f"Record file result: {success}")
        return success

def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        import glob
        # Clean old TTS files
        for pattern in ['/tmp/*.wav', '/tmp/tmp*.wav']:
            for old_file in glob.glob(pattern):
                try:
                    if os.path.getctime(old_file) < time.time() - 300:  # 5 minutes old
                        os.unlink(old_file)
                except:
                    pass
    except:
        pass

def main():
    """Main AGI handler"""
    try:
        logger.info("=== NETOVO Production VoiceBot Starting ===")
        
        # Clean up old files first
        cleanup_temp_files()
        
        # Initialize AGI
        agi = WorkingAGI()
        caller_id = agi.env.get('agi_callerid', 'Unknown')
        logger.info(f"Handling call from: {caller_id}")
        
        # Answer call
        if not agi.answer():
            logger.error("Failed to answer call")
            return
        
        logger.info("Call answered successfully")
        agi.verbose("NETOVO VoiceBot Active")
        
        # Import your working modules
        try:
            logger.info("Importing your existing modules...")
            from riva_client import RivaASRClient, RivaTTSClient
            from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
            from conversation_context import ConversationContext
            logger.info("All modules imported successfully")
        except Exception as e:
            logger.error(f"Module import failed: {e}")
            agi.stream_file("hello")
            time.sleep(3)
            agi.hangup()
            return
        
        # Initialize components
        try:
            logger.info("Initializing components...")
            
            # TTS
            tts = RivaTTSClient()
            tts_connected = tts.connect()
            logger.info(f"TTS connected: {tts_connected}")
            
            # ASR
            asr = RivaASRClient()
            asr_connected = asr.connect()
            logger.info(f"ASR connected: {asr_connected}")
            
            # Ollama
            ollama = OllamaClient()
            ollama_connected = ollama.health_check()
            logger.info(f"Ollama connected: {ollama_connected}")
            
            # Conversation context
            conversation = ConversationContext()
            
            # Check if we have minimum required components
            if not tts_connected:
                raise Exception("TTS is required but not connected")
                
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            agi.stream_file("hello")
            time.sleep(3)
            agi.hangup()
            return
        
        # Main conversation
        try:
            logger.info("Starting conversation...")
            
            # Generate and play greeting
            greeting = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"
            logger.info("Generating greeting TTS...")
            
            greeting_file = tts.synthesize(greeting, sample_rate=8000)  # Use 8kHz for telephony
            
            if greeting_file and os.path.exists(greeting_file):
                logger.info(f"Greeting TTS generated: {greeting_file}")
                
                # Copy to Asterisk sounds directory
                try:
                    sounds_dir = "/var/lib/asterisk/sounds/custom"
                    os.makedirs(sounds_dir, exist_ok=True)
                    
                    greeting_dest = os.path.join(sounds_dir, "greeting.wav")
                    shutil.copy2(greeting_file, greeting_dest)
                    os.chmod(greeting_dest, 0o644)
                    
                    logger.info("Greeting copied to Asterisk sounds")
                    
                    # Play greeting
                    if agi.stream_file("custom/greeting"):
                        logger.info("Greeting played successfully")
                    else:
                        logger.error("Failed to play greeting")
                        agi.stream_file("hello")
                    
                    # Clean up temp file
                    os.unlink(greeting_file)
                    
                except Exception as e:
                    logger.error(f"Greeting playback error: {e}")
                    agi.stream_file("hello")
            else:
                logger.error("Greeting TTS generation failed")
                agi.stream_file("hello")
            
            # Simple conversation loop
            max_turns = 3
            for turn in range(1, max_turns + 1):
                logger.info(f"Conversation turn {turn}")
                
                # Record user input with CORRECT syntax
                record_filename = f"/tmp/user_input_{int(time.time())}"
                logger.info(f"Recording user input to: {record_filename}")
                
                # Use correct AGI RECORD FILE syntax
                recording_success = agi.record_file(
                    record_filename,
                    format="wav", 
                    escape_digits="#",
                    timeout=8000,  # 8 seconds
                    max_silence=3000  # 3 seconds silence
                )
                
                if not recording_success:
                    logger.error("Recording command failed")
                    agi.stream_file("beep")
                    continue
                
                # Check if recording file exists
                wav_file = f"{record_filename}.wav"
                if not os.path.exists(wav_file):
                    logger.error(f"Recording file not found: {wav_file}")
                    if turn == 1:
                        agi.verbose("I didn't hear anything. Could you please speak?")
                        agi.stream_file("beep")
                    continue
                
                # Check file size
                file_size = os.path.getsize(wav_file)
                logger.info(f"Recording file size: {file_size} bytes")
                
                if file_size < 1000:  # Less than 1KB
                    logger.info("Recording too small, no speech detected")
                    os.unlink(wav_file)
                    if turn == 1:
                        agi.verbose("Please speak after the beep")
                        agi.stream_file("beep")
                    continue
                
                # Transcribe with ASR
                if asr_connected:
                    logger.info("Transcribing speech...")
                    transcript = asr.transcribe_file(wav_file)
                    logger.info(f"Transcript: {transcript}")
                    
                    if transcript and transcript.strip():
                        # Get AI response
                        if ollama_connected:
                            logger.info("Getting AI response...")
                            ai_response = ollama.generate(
                                transcript,
                                system_prompt=VOICE_BOT_SYSTEM_PROMPT,
                                max_tokens=50
                            )
                            logger.info(f"AI response: {ai_response}")
                            
                            if ai_response and ai_response.strip():
                                # Generate response TTS
                                logger.info("Generating response TTS...")
                                response_file = tts.synthesize(ai_response, sample_rate=8000)
                                
                                if response_file and os.path.exists(response_file):
                                    try:
                                        # Copy to Asterisk sounds
                                        response_dest = os.path.join(sounds_dir, f"response_{turn}.wav")
                                        shutil.copy2(response_file, response_dest)
                                        os.chmod(response_dest, 0o644)
                                        
                                        # Play response
                                        if agi.stream_file(f"custom/response_{turn}"):
                                            logger.info("AI response played successfully")
                                            
                                            # Add to conversation context
                                            conversation.add_turn(transcript, ai_response)
                                        else:
                                            logger.error("Failed to play AI response")
                                            agi.stream_file("demo-thanks")
                                        
                                        # Cleanup
                                        os.unlink(response_file)
                                        
                                    except Exception as e:
                                        logger.error(f"Response playback error: {e}")
                                        agi.stream_file("demo-thanks")
                                else:
                                    logger.error("Response TTS generation failed")
                                    agi.stream_file("demo-thanks")
                            else:
                                logger.error("Empty AI response")
                                agi.stream_file("demo-thanks")
                        else:
                            logger.error("Ollama not connected")
                            agi.stream_file("demo-thanks")
                    else:
                        logger.info("No speech understood")
                        agi.verbose("I didn't understand that")
                        agi.stream_file("beep")
                else:
                    logger.error("ASR not connected")
                    agi.stream_file("demo-thanks")
                
                # Cleanup recording
                try:
                    os.unlink(wav_file)
                except:
                    pass
                
                # Brief pause between turns
                time.sleep(1)
            
            # End conversation
            logger.info("Conversation completed")
            goodbye_msg = "Thank you for calling NETOVO. Have a great day!"
            
            # Generate goodbye TTS
            goodbye_file = tts.synthesize(goodbye_msg, sample_rate=8000)
            if goodbye_file and os.path.exists(goodbye_file):
                try:
                    goodbye_dest = os.path.join(sounds_dir, "goodbye.wav")
                    shutil.copy2(goodbye_file, goodbye_dest)
                    os.chmod(goodbye_dest, 0o644)
                    agi.stream_file("custom/goodbye")
                    os.unlink(goodbye_file)
                except:
                    agi.stream_file("goodbye")
            else:
                agi.stream_file("goodbye")
            
        except Exception as e:
            logger.error(f"Conversation error: {e}")
            agi.stream_file("demo-thanks")
        
        # End call
        logger.info("Ending call")
        time.sleep(1)
        agi.hangup()
        
        logger.info("=== VoiceBot completed successfully ===")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        try:
            agi = WorkingAGI()
            agi.answer()
            agi.stream_file("hello")
            time.sleep(2)
            agi.hangup()
        except:
            pass

if __name__ == "__main__":
    main()
