#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
Minimal AGI VoiceBot using your existing working files
"""

import sys
import os
import time
from pathlib import Path

# Add your project directory
project_dir = "/home/aiadmin/netovo_voicebot"
sys.path.insert(0, project_dir)

# Simple logging
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

class SimpleAGI:
    """Minimal AGI with correct command syntax"""
    
    def __init__(self):
        self.env = {}
        self.connected = True
        self._parse_env()
    
    def _parse_env(self):
        """Parse AGI environment"""
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            if ':' in line:
                key, value = line.split(':', 1)
                self.env[key.strip()] = value.strip()
        logger.info(f"AGI env parsed: {len(self.env)} vars")
    
    def command(self, cmd):
        """Send AGI command"""
        try:
            logger.debug(f"AGI: {cmd}")
            print(cmd)
            sys.stdout.flush()
            
            result = sys.stdin.readline().strip()
            logger.debug(f"Response: {result}")
            
            return result
        except:
            self.connected = False
            return "ERROR"
    
    def answer(self):
        """Answer call"""
        result = self.command("ANSWER")
        return "200" in result
    
    def hangup(self):
        """Hangup call"""
        self.command("HANGUP")
        self.connected = False
    
    def verbose(self, msg):
        """Verbose message"""
        return self.command(f'VERBOSE "{msg}"')
    
    def stream_file(self, filename):
        """Play audio file"""
        result = self.command(f'STREAM FILE {filename} ""')
        return "200" in result
    
    def record_file(self, filename, timeout=10000, silence=3000):
        """Record audio"""
        result = self.command(f'RECORD FILE {filename} wav # {timeout} 0 0 {silence}')
        return "200" in result
    
    def sleep(self, seconds):
        """Sleep (using Python sleep, not AGI WAIT)"""
        time.sleep(seconds)

def main():
    """Main AGI handler"""
    try:
        logger.info("=== Minimal AGI VoiceBot Starting ===")
        
        # Initialize AGI
        agi = SimpleAGI()
        caller_id = agi.env.get('agi_callerid', 'Unknown')
        logger.info(f"Call from: {caller_id}")
        
        # Answer call
        if not agi.answer():
            logger.error("Failed to answer")
            return
        
        logger.info("Call answered")
        agi.verbose("NETOVO VoiceBot Active")
        
        # Import your working modules
        try:
            logger.info("Importing your modules...")
            from riva_client import RivaASRClient, RivaTTSClient
            from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
            from conversation_context import ConversationContext
            logger.info("Modules imported successfully")
        except Exception as e:
            logger.error(f"Import failed: {e}")
            # Basic fallback
            agi.stream_file("hello")
            agi.sleep(2)
            agi.stream_file("goodbye")
            agi.hangup()
            return
        
        # Initialize your components
        try:
            logger.info("Initializing components...")
            
            # RIVA TTS
            tts = RivaTTSClient()
            tts_ok = tts.connect()
            logger.info(f"TTS connected: {tts_ok}")
            
            # RIVA ASR  
            asr = RivaASRClient()
            asr_ok = asr.connect()
            logger.info(f"ASR connected: {asr_ok}")
            
            # Ollama
            ollama = OllamaClient()
            ollama_ok = ollama.health_check()
            logger.info(f"Ollama connected: {ollama_ok}")
            
            # Conversation context
            conversation = ConversationContext()
            
        except Exception as e:
            logger.error(f"Component init failed: {e}")
            agi.stream_file("hello")
            agi.sleep(3)
            agi.hangup()
            return
        
        # Simple interaction
        try:
            logger.info("Starting interaction...")
            
            # Generate greeting with TTS
            if tts_ok:
                greeting = "Hello, thank you for calling NETOVO. I'm Alexis. How can I help you?"
                logger.info("Generating TTS greeting...")
                
                tts_file = tts.synthesize(greeting)
                
                if tts_file and os.path.exists(tts_file):
                    logger.info(f"TTS file created: {tts_file}")
                    
                    # Copy to Asterisk sounds directory
                    try:
                        import shutil
                        dest_dir = "/var/lib/asterisk/sounds/custom"
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        dest_file = os.path.join(dest_dir, "greeting.wav")
                        shutil.copy2(tts_file, dest_file)
                        os.chmod(dest_file, 0o644)
                        
                        logger.info(f"Copied to: {dest_file}")
                        
                        # Play the greeting
                        if agi.stream_file("custom/greeting"):
                            logger.info("Greeting played successfully")
                        else:
                            logger.error("Failed to play greeting")
                            agi.stream_file("hello")
                        
                        # Cleanup
                        try:
                            os.unlink(tts_file)
                        except:
                            pass
                            
                    except Exception as e:
                        logger.error(f"File copy failed: {e}")
                        agi.stream_file("hello")
                else:
                    logger.error("TTS generation failed")
                    agi.stream_file("hello")
            else:
                logger.info("TTS not available, using fallback")
                agi.stream_file("hello")
            
            # Brief pause
            agi.sleep(1)
            
            # Record user input
            if asr_ok:
                logger.info("Recording user input...")
                record_file = "/tmp/user_input"
                
                if agi.record_file(record_file, timeout=8000, silence=3000):
                    wav_file = f"{record_file}.wav"
                    
                    if os.path.exists(wav_file):
                        logger.info(f"Recording saved: {wav_file}")
                        
                        # Transcribe
                        transcript = asr.transcribe_file(wav_file)
                        logger.info(f"Transcript: {transcript}")
                        
                        if transcript and transcript.strip():
                            # Get AI response
                            if ollama_ok:
                                response = ollama.generate(
                                    transcript, 
                                    system_prompt=VOICE_BOT_SYSTEM_PROMPT,
                                    max_tokens=50
                                )
                                logger.info(f"AI response: {response}")
                                
                                # Generate response TTS
                                if response and tts_ok:
                                    response_file = tts.synthesize(response)
                                    
                                    if response_file:
                                        try:
                                            dest_file = "/var/lib/asterisk/sounds/custom/response.wav"
                                            shutil.copy2(response_file, dest_file)
                                            os.chmod(dest_file, 0o644)
                                            
                                            agi.stream_file("custom/response")
                                            logger.info("Response played")
                                            
                                            os.unlink(response_file)
                                        except Exception as e:
                                            logger.error(f"Response playback failed: {e}")
                                            agi.stream_file("demo-thanks")
                                else:
                                    agi.stream_file("demo-thanks")
                            else:
                                agi.stream_file("demo-thanks")
                        else:
                            logger.info("No speech detected")
                            agi.stream_file("demo-thanks")
                        
                        # Cleanup recording
                        try:
                            os.unlink(wav_file)
                        except:
                            pass
                    else:
                        logger.error("Recording file not found")
                        agi.stream_file("demo-thanks")
                else:
                    logger.error("Recording failed")
                    agi.stream_file("demo-thanks")
            else:
                logger.info("ASR not available")
                agi.stream_file("demo-thanks")
            
            # Goodbye
            agi.sleep(1)
            agi.stream_file("goodbye")
            
        except Exception as e:
            logger.error(f"Interaction error: {e}")
            agi.stream_file("demo-thanks")
        
        # End call
        logger.info("Ending call")
        agi.sleep(1)
        agi.hangup()
        
        logger.info("=== VoiceBot completed successfully ===")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        try:
            agi = SimpleAGI()
            agi.answer()
            agi.stream_file("hello")
            agi.sleep(2)
            agi.hangup()
        except:
            pass

if __name__ == "__main__":
    main()
