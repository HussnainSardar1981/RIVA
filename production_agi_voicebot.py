#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
Debug version using your existing files to find TTS failure
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Add your project directory
project_dir = "/home/aiadmin/netovo_voicebot"
sys.path.insert(0, project_dir)

# Enhanced logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - VoiceBot - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/voicebot.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

class SimpleAGI:
    """Simple AGI"""
    
    def __init__(self):
        self.env = {}
        self.connected = True
        self._parse_env()
    
    def _parse_env(self):
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            if ':' in line:
                key, value = line.split(':', 1)
                self.env[key.strip()] = value.strip()
        logger.info(f"AGI env parsed: {len(self.env)} vars")
    
    def command(self, cmd):
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
        result = self.command("ANSWER")
        return "200" in result
    
    def hangup(self):
        self.command("HANGUP")
        self.connected = False
    
    def verbose(self, msg):
        return self.command(f'VERBOSE "{msg}"')
    
    def stream_file(self, filename):
        result = self.command(f'STREAM FILE {filename} ""')
        return "200" in result

def debug_tts_failure():
    """Debug why TTS is failing"""
    try:
        logger.info("=== DEBUGGING TTS FAILURE ===")
        
        # Import your TTS client
        from riva_client import RivaTTSClient
        
        # Create TTS instance
        tts = RivaTTSClient()
        
        # Test connection
        connected = tts.connect()
        logger.info(f"TTS Connection: {connected}")
        
        if not connected:
            logger.error("TTS connection failed")
            return None
        
        # Test the synthesize method step by step
        test_text = "Hello, this is a test"
        logger.info(f"Testing TTS synthesis with: {test_text}")
        
        try:
            # Call synthesize and debug each step
            logger.info("Calling tts.synthesize()...")
            tts_file = tts.synthesize(test_text, sample_rate=22050)
            logger.info(f"TTS synthesize returned: {tts_file}")
            
            if tts_file:
                logger.info(f"TTS file path: {tts_file}")
                
                # Check if file exists
                if os.path.exists(tts_file):
                    file_size = os.path.getsize(tts_file)
                    logger.info(f"TTS file exists, size: {file_size} bytes")
                    
                    if file_size > 0:
                        logger.info("TTS file generation SUCCESS")
                        return tts_file
                    else:
                        logger.error("TTS file is empty")
                        return None
                else:
                    logger.error(f"TTS file does not exist: {tts_file}")
                    return None
            else:
                logger.error("TTS synthesize returned None/empty")
                return None
                
        except Exception as e:
            logger.error(f"TTS synthesize exception: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    except Exception as e:
        logger.error(f"Debug TTS failure exception: {e}")
        return None

def debug_docker_access():
    """Debug if Docker commands work in AGI context"""
    try:
        logger.info("=== DEBUGGING DOCKER ACCESS ===")
        
        # Test basic docker command
        logger.info("Testing 'docker ps' command...")
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=10)
        logger.info(f"Docker ps return code: {result.returncode}")
        logger.info(f"Docker ps stdout: {result.stdout[:200]}")
        logger.info(f"Docker ps stderr: {result.stderr[:200]}")
        
        # Test sudo docker command
        logger.info("Testing 'sudo docker ps' command...")
        result = subprocess.run(['sudo', 'docker', 'ps'], capture_output=True, text=True, timeout=10)
        logger.info(f"Sudo docker ps return code: {result.returncode}")
        logger.info(f"Sudo docker ps stdout: {result.stdout[:200]}")
        logger.info(f"Sudo docker ps stderr: {result.stderr[:200]}")
        
        # Test if riva container is running
        logger.info("Testing riva container access...")
        result = subprocess.run(['sudo', 'docker', 'exec', 'riva-speech', 'ls', '/opt/riva'], 
                              capture_output=True, text=True, timeout=10)
        logger.info(f"Riva container test return code: {result.returncode}")
        logger.info(f"Riva container test stdout: {result.stdout[:200]}")
        logger.info(f"Riva container test stderr: {result.stderr[:200]}")
        
    except Exception as e:
        logger.error(f"Docker access debug error: {e}")

def debug_file_permissions():
    """Debug file system permissions"""
    try:
        logger.info("=== DEBUGGING FILE PERMISSIONS ===")
        
        # Check temp directories
        temp_dirs = ['/tmp', '/var/tmp', '/home/aiadmin/netovo_voicebot']
        
        for dir_path in temp_dirs:
            try:
                logger.info(f"Checking directory: {dir_path}")
                if os.path.exists(dir_path):
                    stat = os.stat(dir_path)
                    logger.info(f"  Exists: True, Mode: {oct(stat.st_mode)}, Owner: {stat.st_uid}:{stat.st_gid}")
                    
                    # Test write access
                    test_file = os.path.join(dir_path, 'agi_test_write.tmp')
                    try:
                        with open(test_file, 'w') as f:
                            f.write('test')
                        logger.info(f"  Write access: OK")
                        os.unlink(test_file)
                    except Exception as e:
                        logger.info(f"  Write access: FAILED - {e}")
                else:
                    logger.info(f"  Exists: False")
            except Exception as e:
                logger.error(f"Error checking {dir_path}: {e}")
                
        # Check current user context
        logger.info(f"Current UID: {os.getuid()}")
        logger.info(f"Current GID: {os.getgid()}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
    except Exception as e:
        logger.error(f"File permissions debug error: {e}")

def main():
    try:
        logger.info("=== DEBUG AGI VoiceBot Starting ===")
        
        # Initialize AGI
        agi = SimpleAGI()
        caller_id = agi.env.get('agi_callerid', 'Unknown')
        logger.info(f"Call from: {caller_id}")
        
        # Answer call
        if not agi.answer():
            logger.error("Failed to answer")
            return
        
        logger.info("Call answered")
        agi.verbose("DEBUG VoiceBot Active")
        
        # Run all debug tests
        debug_docker_access()
        debug_file_permissions()
        
        # Import your modules
        try:
            logger.info("Importing your modules...")
            from riva_client import RivaASRClient, RivaTTSClient
            from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
            logger.info("Modules imported successfully")
        except Exception as e:
            logger.error(f"Import failed: {e}")
            agi.stream_file("hello")
            time.sleep(3)
            agi.hangup()
            return
        
        # Test TTS in detail
        tts_file = debug_tts_failure()
        
        if tts_file:
            logger.info("TTS DEBUG: SUCCESS - File generated")
            
            # Try to copy to Asterisk directory
            try:
                import shutil
                dest_dir = "/var/lib/asterisk/sounds/custom"
                os.makedirs(dest_dir, exist_ok=True)
                dest_file = os.path.join(dest_dir, "debug_greeting.wav")
                shutil.copy2(tts_file, dest_file)
                os.chmod(dest_file, 0o644)
                
                logger.info(f"File copied to: {dest_file}")
                
                # Try to play it
                if agi.stream_file("custom/debug_greeting"):
                    logger.info("DEBUG TTS PLAYBACK: SUCCESS")
                else:
                    logger.error("DEBUG TTS PLAYBACK: FAILED")
                    agi.stream_file("hello")
                
                # Cleanup
                try:
                    os.unlink(tts_file)
                    os.unlink(dest_file)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"File copy/playback error: {e}")
                agi.stream_file("hello")
        else:
            logger.error("TTS DEBUG: FAILED - No file generated")
            agi.stream_file("hello")
        
        # Keep call alive for a moment to hear the result
        logger.info("Keeping call alive for 10 seconds for testing...")
        time.sleep(10)
        
        # End call
        logger.info("Debug complete, ending call")
        agi.hangup()
        
        logger.info("=== DEBUG completed ===")
        
    except Exception as e:
        logger.error(f"Debug fatal error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        try:
            agi = SimpleAGI()
            agi.answer()
            agi.stream_file("hello")
            time.sleep(3)
            agi.hangup()
        except:
            pass

if __name__ == "__main__":
    main()
