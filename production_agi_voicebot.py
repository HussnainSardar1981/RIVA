#!/usr/bin/env python3
"""
DIAGNOSTIC AGI VoiceBot - Debug Version
This version will help identify exactly what's failing
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Force logging to work in AGI context
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - DIAGNOSTIC - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/voicebot_debug.log'),  # Use /tmp for guaranteed write access
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

def debug_log(message):
    """Ensure debug messages get logged"""
    logger.info(message)
    print(f"DEBUG: {message}", file=sys.stderr)
    sys.stderr.flush()

class DiagnosticAGI:
    """Simplified AGI for debugging"""

    def __init__(self):
        debug_log("=== DIAGNOSTIC AGI STARTING ===")
        self.env = {}
        self.connected = True
        self._parse_environment()

    def _parse_environment(self):
        """Parse AGI environment"""
        try:
            debug_log("Parsing AGI environment...")
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
            debug_log(f"AGI Environment parsed: {len(self.env)} variables")
        except Exception as e:
            debug_log(f"Environment parsing error: {e}")

    def command(self, cmd):
        """Send AGI command"""
        try:
            debug_log(f"AGI Command: {cmd}")
            sys.stdout.write(f"{cmd}\n")
            sys.stdout.flush()
            
            result = sys.stdin.readline()
            if result:
                result = result.strip()
                debug_log(f"AGI Response: {result}")
                return result
            return "ERROR"
        except Exception as e:
            debug_log(f"AGI command error: {e}")
            return "ERROR"

    def answer(self):
        """Answer call"""
        result = self.command("ANSWER")
        success = result != "ERROR" and not result.startswith('510')
        debug_log(f"Answer result: {success} ({result})")
        return success

    def verbose(self, message):
        """Send verbose message"""
        return self.command(f'VERBOSE "{message}" 1')

    def stream_file(self, filename):
        """Stream audio file"""
        result = self.command(f'STREAM FILE "{filename}" ""')
        success = result != "ERROR" and not result.startswith('510')
        debug_log(f"Stream file result: {success} ({result})")
        return success

    def hangup(self):
        """Hangup call"""
        self.command("HANGUP")
        self.connected = False

def test_python_environment():
    """Test if we can import required modules"""
    debug_log("=== TESTING PYTHON ENVIRONMENT ===")
    
    # Test sys.path
    debug_log(f"Python executable: {sys.executable}")
    debug_log(f"Python version: {sys.version}")
    debug_log(f"Current working directory: {os.getcwd()}")
    debug_log(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Test if we're in virtual environment
    venv_path = os.environ.get('VIRTUAL_ENV')
    debug_log(f"Virtual environment: {venv_path}")
    
    # Add the project directory to path
    project_dir = "/home/aiadmin/netovo_voicebot"
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
        debug_log(f"Added {project_dir} to Python path")
    
    # Test imports one by one
    import_results = {}
    
    modules_to_test = [
        'numpy', 'httpx', 'grpc', 'soxr', 'webrtcvad', 
        'structlog', 'tenacity', 'wave', 'tempfile'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            import_results[module_name] = "OK"
            debug_log(f"Import {module_name}: OK")
        except Exception as e:
            import_results[module_name] = str(e)
            debug_log(f"Import {module_name}: FAILED - {e}")
    
    # Test our custom modules
    custom_modules = [
        'riva_client', 'ollama_client', 'audio_processing', 'conversation_context'
    ]
    
    for module_name in custom_modules:
        try:
            module = __import__(module_name)
            import_results[module_name] = "OK"
            debug_log(f"Import {module_name}: OK")
        except Exception as e:
            import_results[module_name] = str(e)
            debug_log(f"Import {module_name}: FAILED - {e}")
    
    return import_results

def test_services():
    """Test external services"""
    debug_log("=== TESTING SERVICES ===")
    
    service_results = {}
    
    # Test Ollama
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://127.0.0.1:11434/api/tags")
            if response.status_code == 200:
                service_results['ollama'] = "OK"
                debug_log("Ollama service: OK")
            else:
                service_results['ollama'] = f"HTTP {response.status_code}"
                debug_log(f"Ollama service: HTTP {response.status_code}")
    except Exception as e:
        service_results['ollama'] = str(e)
        debug_log(f"Ollama service: FAILED - {e}")
    
    # Test RIVA (simple connection test)
    try:
        import grpc
        channel = grpc.insecure_channel('localhost:50051')
        future = grpc.channel_ready_future(channel)
        future.result(timeout=5)
        service_results['riva'] = "OK"
        debug_log("RIVA service: OK")
        channel.close()
    except Exception as e:
        service_results['riva'] = str(e)
        debug_log(f"RIVA service: FAILED - {e}")
    
    return service_results

def test_file_permissions():
    """Test file and directory permissions"""
    debug_log("=== TESTING FILE PERMISSIONS ===")
    
    paths_to_test = [
        "/var/lib/asterisk/sounds/custom",
        "/var/spool/asterisk/monitor", 
        "/var/log/asterisk",
        "/tmp"
    ]
    
    for path in paths_to_test:
        try:
            if os.path.exists(path):
                if os.access(path, os.W_OK):
                    debug_log(f"Path {path}: WRITABLE")
                else:
                    debug_log(f"Path {path}: NOT WRITABLE")
            else:
                debug_log(f"Path {path}: DOES NOT EXIST")
        except Exception as e:
            debug_log(f"Path {path}: ERROR - {e}")

def run_minimal_conversation():
    """Run a minimal conversation test"""
    debug_log("=== STARTING MINIMAL CONVERSATION TEST ===")
    
    try:
        agi = DiagnosticAGI()
        caller_id = agi.env.get('agi_callerid', 'Unknown')
        debug_log(f"Handling call from: {caller_id}")
        
        # Answer call
        if not agi.answer():
            debug_log("FAILED to answer call")
            return False
        
        debug_log("Call answered successfully")
        
        # Send verbose message
        agi.verbose("NETOVO Diagnostic Voice Bot Active")
        
        # Play a simple sound file
        debug_log("Playing greeting sound...")
        if agi.stream_file("hello"):
            debug_log("Greeting played successfully")
        else:
            debug_log("FAILED to play greeting")
        
        # Keep call alive for testing
        debug_log("Keeping call alive for 10 seconds...")
        for i in range(10):
            debug_log(f"Call alive: {i+1}/10 seconds")
            time.sleep(1)
            if not agi.connected:
                debug_log("Call disconnected early")
                break
        
        # Play goodbye
        debug_log("Playing goodbye...")
        agi.stream_file("goodbye")
        
        debug_log("Test conversation completed successfully")
        return True
        
    except Exception as e:
        debug_log(f"Conversation test failed: {e}")
        debug_log(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main diagnostic entry point"""
    try:
        debug_log("=== NETOVO DIAGNOSTIC VOICEBOT STARTING ===")
        debug_log(f"Script called with args: {sys.argv}")
        debug_log(f"Environment USER: {os.environ.get('USER', 'unknown')}")
        debug_log(f"Environment HOME: {os.environ.get('HOME', 'unknown')}")
        debug_log(f"Process UID/GID: {os.getuid()}/{os.getgid()}")
        
        # Run all diagnostic tests
        import_results = test_python_environment()
        service_results = test_services()
        test_file_permissions()
        
        # Count failures
        import_failures = sum(1 for result in import_results.values() if result != "OK")
        service_failures = sum(1 for result in service_results.values() if result != "OK")
        
        debug_log(f"=== DIAGNOSTIC SUMMARY ===")
        debug_log(f"Import failures: {import_failures}/{len(import_results)}")
        debug_log(f"Service failures: {service_failures}/{len(service_results)}")
        
        if import_failures > 0:
            debug_log("CRITICAL: Import failures detected - cannot proceed")
            debug_log("Failed imports:")
            for module, result in import_results.items():
                if result != "OK":
                    debug_log(f"  {module}: {result}")
            
            # Try basic AGI anyway
            try:
                agi = DiagnosticAGI()
                agi.answer()
                agi.verbose("Diagnostic failed - check logs")
                agi.stream_file("pbx-invalid")
                time.sleep(3)
                agi.hangup()
            except:
                pass
            return
        
        # If imports work, try minimal conversation
        if run_minimal_conversation():
            debug_log("=== DIAGNOSTIC COMPLETED SUCCESSFULLY ===")
        else:
            debug_log("=== DIAGNOSTIC COMPLETED WITH ERRORS ===")
            
    except Exception as e:
        debug_log(f"=== DIAGNOSTIC FATAL ERROR ===")
        debug_log(f"Fatal error: {e}")
        debug_log(f"Traceback: {traceback.format_exc()}")
        
        # Try to at least answer and hangup
        try:
            agi = DiagnosticAGI()
            agi.answer()
            agi.verbose("Fatal diagnostic error")
            time.sleep(2)
            agi.hangup()
        except:
            pass

if __name__ == "__main__":
    main()
