#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
Final working AGI using your fixed files with correct RECORD syntax
"""

import sys
import os
import time
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

class FixedAGI:
    """AGI with correct RECORD FILE syntax"""
    
    def __init__(self):
        self.env = {}
        self.connected = True
        self._parse_environment()
    
    def _parse_environment(self):
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            if ':' in line:
                key, value = line.split(':', 1)
                self.env[key.strip()] = value.strip()
        logger.info(f"AGI environment parsed: {len(self.env)} variables")
    
    def command(self, cmd):
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
        result = self.command("ANSWER")
        success = result.startswith('200')
        logger.info(f"Answer result: {success}")
        return success
    
    def hangup(self):
        if self.connected:
            self.command("HANGUP")
            self.connected = False
    
    def verbose(self, message):
        escaped = message.replace('"', '\\"')
        return self.command(f'VERBOSE "{escaped}"')
    
    def stream_file(self, filename, escape_digits=""):
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        result = self.command(f'STREAM FILE {filename} "{escape_digits}"')
        success = result.startswith('200')
        logger.info(f"Stream file '{filename}' result: {success}")
        return success
    
    def record_file(self, filename, timeout=8000, silence=3000):
        """CORRECT AGI RECORD FILE syntax"""
        # Correct syntax: RECORD FILE filename format escape_digits timeout [offset] [BEEP] [silence]
        result = self.command(f'RECORD FILE {filename} wav # {timeout} BEEP {silence}')
        success = result.startswith('200')
        logger.info(f"Record file '{filename}' result: {success}")
        return success

def main():
    try:
        logger.info("=== NETOVO Production VoiceBot Starting ===")
        
        # Initialize AGI
        agi = FixedAGI()
        caller_id = agi.env.get('agi_callerid', 'Unknown')
        logger.info(f"Handling call from: {caller_id}")
        
        # Answer call
        if not agi.answer():
            logger.error("Failed to answer call")
            return
        
        logger.info("Call answered successfully")
        agi.verbose("NETOVO VoiceBot Active")
        
        # Import your existing modules
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
            tts = RivaTTSClient()
            tts_connected = tts.connect()
            logger.info(f"TTS connected: {tts_connected}")
            
            asr = RivaASRClient()
            asr_connected = asr.connect()
            logger.info(f"ASR connected: {asr_connected}")
            
            ollama = OllamaClient()
            ollama_connected = ollama.health_check()
            logger.info(f"Ollama connected: {ollama_connected}")
            
            conversation = ConversationContext()
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            agi.stream_file("hello")
            time.sleep(3)
            agi.hangup()
            return
        
        # Start conversation
        try:
            logger.info("Starting conversation...")
            
            # Generate greeting
            if tts_connected:
                greeting = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"
                logger.info("Generating greeting TTS...")
                
                greeting_file = tts.synthesize(greeting, sample_rate=8000)
                
                if greeting_file and os.path.exists(greeting_file):
                    logger.info(f"Greeting TTS generated: {greeting_file}, size: {os.path.getsize(greeting_file)}")
                    
                    # Copy to Asterisk sounds
                    try:
                        sounds_dir = "/var/lib/asterisk/sounds/custom"
                        os.makedirs(sounds_dir, exist_ok=True)
                        greeting_dest = os.path.join(sounds_dir, "greeting.wav")
                        shutil.copy2(greeting_file, greeting_dest)
                        os.chmod(greeting_dest, 0o644)
                        
                        # Play greeting
                        if agi.stream_file("custom/greeting"):
                            logger.info("Greeting played successfully")
                        else:
                            logger.error("Failed to play greeting, using fallback")
                            agi.stream_file("hello")
                        
                        os.unlink(greeting_file)
                        
                    except Exception as e:
                        logger.error(f"Greeting playback error: {e}")
                        agi.stream_file("hello")
                else:
                    logger.error("Greeting TTS generation failed")
                    agi.stream_file("hello")
            else:
                logger.error("TTS not connected")
                agi.stream_file("hello")
            
            # Simple conversation
            for turn in range(1, 4):  # Max 3 turns
                logger.info(f"Conversation turn {turn}")
                
                # Record user input with CORRECT syntax
                record_filename = f"/tmp/user_input_{int(time.time())}"
                logger.info(f"Recording user input to: {record_filename}")
                
                # Use FIXED record syntax
                if agi.record_file(record_filename, timeout=8000, silence=3000):
                    wav_file = f"{record_filename}.wav"
                    
                    if os.path.exists(wav_file):
                        file_size = os.path.getsize(wav_file)
                        logger.info(f"Recording file size: {file_size} bytes")
                        
                        if file_size > 1000:  # Valid recording
                            # Transcribe
                            if asr_connected:
                                logger.info("Transcribing speech...")
                                transcript = asr.transcribe_file(wav_file)
                                logger.info(f"Transcript: {transcript}")
                                
                                if transcript and transcript.strip() and not "error" in transcript.lower():
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
                                            if tts_connected:
                                                logger.info("Generating response TTS...")
                                                response_file = tts.synthesize(ai_response, sample_rate=8000)
                                                
                                                if response_file and os.path.exists(response_file):
                                                    try:
                                                        response_dest = os.path.join(sounds_dir, f"response_{turn}.wav")
                                                        shutil.copy2(response_file, response_dest)
                                                        os.chmod(response_dest, 0o644)
                                                        
                                                        if agi.stream_file(f"custom/response_{turn}"):
                                                            logger.info("AI response played successfully")
                                                            conversation.add_turn(transcript, ai_response)
                                                        else:
                                                            logger.error("Failed to play AI response")
                                                            agi.stream_file("demo-thanks")
                                                        
                                                        os.unlink(response_file)
                                                        
                                                    except Exception as e:
                                                        logger.error(f"Response playback error: {e}")
                                                        agi.stream_file("demo-thanks")
                                                else:
                                                    logger.error("Response TTS generation failed")
                                                    agi.stream_file("demo-thanks")
                                            else:
                                                agi.stream_file("demo-thanks")
                                        else:
                                            agi.stream_file("demo-thanks")
                                    else:
                                        agi.stream_file("demo-thanks")
                                else:
                                    logger.info("No valid speech understood")
                                    agi.stream_file("beep")
                            else:
                                agi.stream_file("demo-thanks")
                        else:
                            logger.info("Recording too small")
                            agi.stream_file("beep")
                        
                        # Cleanup
                        try:
                            os.unlink(wav_file)
                        except:
                            pass
                    else:
                        logger.error("Recording file not found")
                        agi.stream_file("beep")
                else:
                    logger.error("Recording command failed")
                    agi.stream_file("beep")
                
                time.sleep(1)
            
            # Goodbye
            logger.info("Conversation completed")
            if tts_connected:
                goodbye_file = tts.synthesize("Thank you for calling NETOVO. Have a great day!", sample_rate=8000)
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

if __name__ == "__main__":
    main()
