#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
FIXED AGI VoiceBot - Proper file paths and WAV format
"""

import sys
import os
import time
import subprocess
import tempfile
import uuid
from datetime import datetime

# Set up project paths
project_dir = "/home/aiadmin/netovo_voicebot"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Simple logging with fallback
import logging

# Try to log to asterisk log, fallback to stderr only
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - FixedBot - %(message)s',
        handlers=[
            logging.FileHandler('/var/log/asterisk/voicebot.log', mode='a'),
            logging.StreamHandler(sys.stderr)
        ]
    )
except PermissionError:
    # Fallback to stderr only if can't write to log file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - FixedBot - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )

logger = logging.getLogger(__name__)

class SimpleAGI:
    """Minimal AGI with correct command syntax"""

    def __init__(self):
        self.env = {}
        self.connected = True
        self.call_answered = False
        self._parse_env()

    def _parse_env(self):
        """Parse AGI environment"""
        env_count = 0
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            if ':' in line:
                key, value = line.split(':', 1)
                self.env[key.strip()] = value.strip()
                env_count += 1
        logger.info(f"AGI env parsed: {env_count} vars")

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
        success = result and result.startswith('200')
        if success:
            self.call_answered = True
            logger.info("Call answered")
        return success

    def hangup(self):
        """Hangup call"""
        self.command("HANGUP")
        self.connected = False

    def verbose(self, msg):
        """Verbose message"""
        return self.command(f'VERBOSE "{msg}"')

    def stream_file(self, filename):
        """Play audio file - NO QUOTES on filename"""
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]

        # Check for both WAV and SLIN16 files in root sounds directory
        wav_path = f"/usr/share/asterisk/sounds/{filename}.wav"
        sln16_path = f"/usr/share/asterisk/sounds/{filename}.sln16"

        if os.path.exists(wav_path):
            file_size = os.path.getsize(wav_path)
            logger.info(f"Playing WAV: {filename} (file exists: {file_size} bytes)")
        elif os.path.exists(sln16_path):
            file_size = os.path.getsize(sln16_path)
            logger.info(f"Playing SLIN16: {filename} (file exists: {file_size} bytes)")
        else:
            logger.error(f"Audio file not found: {wav_path} or {sln16_path}")

        result = self.command(f'STREAM FILE {filename} ""')
        success = result and result.startswith('200')
        logger.info(f"Stream file result: {result} (success: {success})")
        return success

    def record_file(self, filename):
        """Record audio - SIMPLE syntax"""
        result = self.command(f'RECORD FILE {filename} wav "#" 8000 10 0 BEEP')
        return result and result.startswith('200')

    def sleep(self, seconds):
        """Sleep"""
        time.sleep(seconds)

class DirectTTSClient:
    """Direct Docker TTS using your proven commands"""

    def __init__(self, container="riva-speech"):
        self.container = container

    def synthesize(self, text, voice="English-US.Female-1", sample_rate=22050):
        """Direct TTS synthesis using Docker"""
        try:
            # Create host temp file
            host_output = f"/tmp/tts_agi_{uuid.uuid4().hex}.wav"
            container_output = f"/tmp/riva_tts_{uuid.uuid4().hex}.wav"

            # Run TTS without sudo (AGI doesn't have terminal access)
            cmd = [
                "docker", "exec", self.container,
                "/opt/riva/clients/riva_tts_client",
                f"--riva_uri=localhost:50051",
                f"--text={text}",
                f"--voice_name={voice}",
                f"--audio_file={container_output}",
                f"--rate={sample_rate}"
            ]

            logger.info(f"Running TTS: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"TTS failed: {result.stderr}")
                return None

            # Copy from container to host
            copy_cmd = ["docker", "cp", f"{self.container}:{container_output}", host_output]
            copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=10)

            # Cleanup container file
            subprocess.run(["docker", "exec", self.container, "rm", "-f", container_output],
                          capture_output=True)

            if copy_result.returncode == 0 and os.path.exists(host_output):
                file_size = os.path.getsize(host_output)
                logger.info(f"TTS success: {host_output} ({file_size} bytes)")
                return host_output
            else:
                logger.error(f"TTS copy failed: {copy_result.stderr}")
                return None

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

class DirectASRClient:
    """Direct Docker ASR using your proven commands"""

    def __init__(self, container="riva-speech"):
        self.container = container

    def transcribe_file(self, audio_file):
        """Direct ASR transcription using Docker with format conversion"""
        try:
            container_path = f"/tmp/riva_asr_{uuid.uuid4().hex}.wav"
            converted_path = f"/tmp/converted_{uuid.uuid4().hex}.wav"

            # Convert audio to RIVA-compatible format (16kHz, mono, 16-bit)
            sox_cmd = [
                'sox', audio_file,
                '-r', '16000',    # 16kHz (RIVA preferred)
                '-c', '1',        # Mono
                '-b', '16',       # 16-bit
                '-e', 'signed-integer',  # PCM
                converted_path
            ]

            logger.info(f"Converting audio for ASR: {' '.join(sox_cmd)}")
            convert_result = subprocess.run(sox_cmd, capture_output=True, text=True, timeout=10)

            if convert_result.returncode != 0:
                logger.error(f"Audio conversion failed: {convert_result.stderr}")
                return ""

            # Check converted file size
            if os.path.exists(converted_path):
                file_size = os.path.getsize(converted_path)
                logger.info(f"Converted audio file: {file_size} bytes")
                if file_size < 1000:
                    logger.error("Converted file too small")
                    return ""
            else:
                logger.error("Converted file not created")
                return ""

            # Copy converted file to container
            copy_cmd = ["docker", "cp", converted_path, f"{self.container}:{container_path}"]
            copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=10)

            if copy_result.returncode != 0:
                logger.error(f"ASR copy failed: {copy_result.stderr}")
                return ""

            # Run ASR
            cmd = [
                "docker", "exec", self.container,
                "/opt/riva/clients/riva_streaming_asr_client",
                f"--riva_uri=localhost:50051",
                f"--audio_file={container_path}",
                "--simulate_realtime=false",
                "--language_code=en-US"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # DEBUG: Log everything RIVA returns
            logger.info(f"RIVA ASR returncode: {result.returncode}")
            logger.info(f"RIVA ASR stdout: {repr(result.stdout)}")
            logger.info(f"RIVA ASR stderr: {repr(result.stderr)}")

            # Cleanup
            subprocess.run(["docker", "exec", self.container, "rm", "-f", container_path],
                          capture_output=True)
            try:
                os.unlink(converted_path)
            except:
                pass

            if result.returncode == 0:
                # Parse transcript - ENHANCED
                full_output = result.stdout.strip()
                logger.info(f"Full RIVA output: {full_output}")

                if not full_output:
                    logger.warning("RIVA returned empty output")
                    return ""

                lines = full_output.split('\n')
                logger.info(f"RIVA output lines: {len(lines)}")

                # Look for RIVA's final transcript format: "0 : [transcript]"
                for i, line in enumerate(lines):
                    logger.info(f"Line {i}: {repr(line)}")

                    # RIVA format: "0 : Hello." or "0 : Can you please resolve my email verification issue? "
                    if line.strip().startswith("0 : "):
                        transcript = line.strip()[4:].strip()  # Remove "0 : " prefix
                        if transcript and len(transcript) > 1:
                            # Clean up transcript
                            transcript = transcript.strip('"').strip("'").strip()
                            if transcript:
                                logger.info(f"RIVA transcript found: {transcript}")
                                return transcript

                # Fallback: Look for any line with meaningful speech content
                for line in lines:
                    # Skip metadata lines
                    if any(x in line.lower() for x in ['loading', 'file:', 'done loading', 'audio processed', 'run time', 'total audio', 'throughput']):
                        continue

                    # Look for lines that look like speech
                    if line.strip() and not line.strip().startswith('-') and len(line.strip()) > 3:
                        # Remove timestamps and confidence scores
                        import re
                        cleaned = re.sub(r'\d+\.\d+e[+-]\d+', '', line)  # Remove scientific notation
                        cleaned = re.sub(r'\d{4,}', '', cleaned)          # Remove timestamps
                        cleaned = cleaned.strip()

                        # Check if it looks like speech
                        if cleaned and len(cleaned) > 3 and any(c.isalpha() for c in cleaned):
                            logger.info(f"Fallback transcript: {cleaned}")
                            return cleaned

            else:
                logger.error(f"RIVA ASR failed with returncode: {result.returncode}")
                logger.error(f"RIVA stderr: {result.stderr}")

            logger.warning("No transcript found after all parsing attempts")
            return ""

        except Exception as e:
            logger.error(f"ASR error: {e}")
            return ""

class NetovoAIClient:
    """Professional NETOVO AI client with full conversation management"""

    def __init__(self):
        # Import the NETOVO conversation manager
        import sys
        sys.path.append('/home/aiadmin/netovo_voicebot')

        from netovo_conversation_manager import NetovoConversationManager
        from netovo_ollama_client import NetovoOllamaClient

        self.conversation_manager = NetovoConversationManager()
        self.ollama_client = NetovoOllamaClient()
        self.call_start_time = time.time()

    def get_greeting(self):
        """Get professional NETOVO greeting"""
        return self.ollama_client.generate_greeting()

    def generate_response(self, user_input: str, phone_number: str = None) -> tuple:
        """
        Generate professional NETOVO response
        Returns: (response_text, should_transfer, should_end)
        """
        try:
            # Use conversation manager for intelligent routing
            response, action = self.conversation_manager.generate_response(
                user_input, phone_number
            )

            # For complex issues, enhance with Ollama
            if action == "continue" and "troubleshooting" in self.conversation_manager.conversation_state:
                # Add AI enhancement for better user experience
                context = f"Issue category: {self.conversation_manager.current_issue}, Step: {self.conversation_manager.troubleshooting_step}"
                enhanced_response = self.ollama_client.generate_response(user_input, context)

                # Combine structured response with AI enhancement if needed
                if len(response.split()) < 5:  # If response is too short, enhance it
                    response = enhanced_response
                else:
                    # Use the structured response (it's already comprehensive)
                    pass

            elif action == "continue":
                # For general conversation, use AI
                response = self.ollama_client.generate_response(user_input)

            # Determine actions
            should_transfer = action == "transfer"
            should_end = action == "end_call"

            # Check for user wanting to end conversation
            if self.conversation_manager.should_end_conversation(user_input):
                response = self.ollama_client.generate_closing_response()
                should_end = True

            logger.info(f"NETOVO AI response: {response[:100]}")
            logger.info(f"Action: {action}, Transfer: {should_transfer}, End: {should_end}")

            return response, should_transfer, should_end

        except Exception as e:
            logger.error(f"NETOVO AI error: {e}")
            return "I'm experiencing technical difficulties. Let me transfer you to our support team for immediate assistance.", True, False

    def get_conversation_summary(self):
        """Get conversation summary for logging"""
        duration = int(time.time() - self.call_start_time)
        ai_summary = self.ollama_client.get_conversation_summary()

        return f"Call duration: {duration}s, State: {self.conversation_manager.conversation_state}, Summary: {ai_summary}"

    def reset_for_new_call(self):
        """Reset for new call"""
        self.conversation_manager = NetovoConversationManager()
        self.ollama_client.reset_conversation()
        self.call_start_time = time.time()

def convert_audio_for_asterisk(input_wav):
    """Convert to exact Asterisk-compatible format"""
    try:
        timestamp = int(time.time())
        
        # Try multiple format approaches
        formats_to_try = [
            {
                'ext': 'wav',
                'path': f"/usr/share/asterisk/sounds/tts_{timestamp}.wav",
                'sox_args': [
                    'sox', input_wav,
                    '-r', '8000',      # 8kHz sample rate
                    '-c', '1',         # Mono
                    '-b', '16',        # 16-bit
                    '-e', 'signed-integer',  # PCM
                    '-t', 'wav'        # Explicitly specify WAV format
                ]
            },
            {
                'ext': 'sln16', 
                'path': f"/usr/share/asterisk/sounds/tts_{timestamp}.sln16",
                'sox_args': [
                    'sox', input_wav,
                    '-r', '8000',      # 8kHz 
                    '-c', '1',         # Mono
                    '-b', '16',        # 16-bit
                    '-e', 'signed-integer',  # PCM
                    '-t', 'raw'        # Raw format (what .sln16 is)
                ]
            },
            {
                'ext': 'gsm',
                'path': f"/usr/share/asterisk/sounds/tts_{timestamp}.gsm", 
                'sox_args': [
                    'sox', input_wav,
                    '-r', '8000',      # 8kHz
                    '-c', '1',         # Mono  
                    '-t', 'gsm'        # GSM format (very compatible)
                ]
            }
        ]
        
        for fmt in formats_to_try:
            try:
                logger.info(f"Trying {fmt['ext']} format...")
                
                # Add output path to sox command
                sox_cmd = fmt['sox_args'] + [fmt['path']]
                logger.info(f"Sox command: {' '.join(sox_cmd)}")
                
                result = subprocess.run(sox_cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and os.path.exists(fmt['path']):
                    file_size = os.path.getsize(fmt['path'])
                    if file_size > 100:  # Valid file
                        os.chmod(fmt['path'], 0o644)
                        filename = f"tts_{timestamp}"
                        logger.info(f"SUCCESS: {fmt['ext']} format created: {filename} ({file_size} bytes)")
                        return filename
                    else:
                        logger.warning(f"{fmt['ext']} file too small: {file_size} bytes")
                        os.unlink(fmt['path'])
                else:
                    logger.warning(f"{fmt['ext']} conversion failed: {result.stderr}")
                    
            except Exception as e:
                logger.warning(f"{fmt['ext']} format failed: {e}")
                
        # If all formats fail, try copying a working built-in file and replacing content
        try:
            logger.info("Trying built-in file replacement method...")
            
            # Find a working built-in file to use as template
            template_files = [
                "/var/lib/asterisk/sounds/demo-thanks.wav",
                "/var/lib/asterisk/sounds/demo-congrats.wav", 
                "/var/lib/asterisk/sounds/hello.wav"
            ]
            
            template_file = None
            for tf in template_files:
                if os.path.exists(tf):
                    template_file = tf
                    break
                    
            if template_file:
                output_path = f"/usr/share/asterisk/sounds/tts_{timestamp}.wav"
                
                # Use sox to match the exact format of the working template
                sox_cmd = [
                    'sox', input_wav,
                    output_path,
                    'rate', '8000',    # Alternative syntax
                    'channels', '1',   # Alternative syntax  
                    'bits', '16'       # Alternative syntax
                ]
                
                result = subprocess.run(sox_cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    if file_size > 100:
                        os.chmod(output_path, 0o644)
                        logger.info(f"Template-based conversion success: {file_size} bytes")
                        return f"tts_{timestamp}"
                        
        except Exception as e:
            logger.error(f"Template method failed: {e}")
            
        logger.error("All audio conversion methods failed")
        return None
        
    except Exception as e:
        logger.error(f"Audio conversion fatal error: {e}")
        return None

def main():
    """Main AGI handler"""
    try:
        logger.info("=== FIXED AGI VoiceBot Starting ===")

        # Initialize AGI
        agi = SimpleAGI()
        caller_id = agi.env.get('agi_callerid', 'Unknown')
        logger.info(f"Call from: {caller_id}")

        # Answer call
        if not agi.answer():
            logger.error("Failed to answer")
            return

        agi.sleep(1)
        agi.verbose("Fixed VoiceBot Active")

        # Initialize NETOVO professional components
        tts = DirectTTSClient()
        asr = DirectASRClient()
        netovo_ai = NetovoAIClient()

        # Send professional NETOVO greeting
        logger.info("Generating NETOVO professional greeting...")
        greeting_text = netovo_ai.get_greeting()

        # Extract caller phone number for context
        caller_phone = agi.env.get('agi_callerid', 'Unknown')

        tts_file = tts.synthesize(greeting_text)

        if tts_file and os.path.exists(tts_file):
            asterisk_file = convert_audio_for_asterisk(tts_file)

            # Cleanup TTS file
            try:
                os.unlink(tts_file)
            except:
                pass

            if asterisk_file:
                success = agi.stream_file(asterisk_file)
                logger.info(f"Greeting played: {success}")
            else:
                logger.error("Audio conversion failed")
                # Find any working built-in sound as fallback
                for fallback in ['demo-thanks', 'demo-congrats', 'beep']:
                    if agi.stream_file(fallback):
                        break
        else:
            logger.error("TTS greeting failed")
            # Fallback to built-in sound
            for fallback in ['demo-thanks', 'demo-congrats', 'beep']:
                if agi.stream_file(fallback):
                    break

        # Conversation loop - continue until user hangs up or explicit exit
        max_turns = 50  # Safety limit - about 20-25 minutes
        failed_interactions = 0
        start_time = time.time()

        for turn in range(max_turns):
            logger.info(f"Conversation turn {turn + 1}")

            # Record user input
            record_file = f"/var/spool/asterisk/monitor/user_{int(time.time())}"

            logger.info("Recording user...")
            if agi.record_file(record_file):
                wav_file = f"{record_file}.wav"

                if os.path.exists(wav_file):
                    file_size = os.path.getsize(wav_file)
                    logger.info(f"Recording: {file_size} bytes")

                    if file_size > 1000:
                        # Transcribe
                        transcript = asr.transcribe_file(wav_file)

                        if transcript:
                            logger.info(f"User said: {transcript}")
                            failed_interactions = 0  # Reset counter on successful interaction

                            # Check for USER exit intents (not AI responses)
                            user_exit_phrases = [
                                'goodbye', 'good bye', 'bye', 'bye bye',
                                'that\'s all', 'that is all', 'nothing else',
                                'you\'ve helped me', 'problem solved', 'all set',
                                'transfer me', 'human agent', 'speak to someone',
                                'i\'m done', 'we\'re done', 'finished'
                            ]

                            # Check for emergency/urgent
                            urgent_phrases = ['emergency', 'urgent', 'critical']

                            # Generate professional NETOVO response
                            response, should_transfer, should_end = netovo_ai.generate_response(
                                transcript, caller_phone
                            )

                            # Override with exit checks
                            if any(phrase in transcript.lower() for phrase in user_exit_phrases):
                                response = netovo_ai.generate_closing_response()
                                should_end = True
                            elif any(phrase in transcript.lower() for phrase in urgent_phrases):
                                response = "I understand this is urgent. Let me transfer you to our priority support team immediately."
                                should_transfer = True
                        else:
                            failed_interactions += 1
                            if failed_interactions >= 3:
                                response = "I'm having trouble hearing you clearly. Let me transfer you to a human agent who can better assist you."
                                should_transfer = True
                                should_end = False
                            else:
                                response = "I didn't catch that. Could you repeat?"
                                should_transfer = False
                                should_end = False
                    else:
                        response = "I didn't hear anything. Could you speak up?"
                        should_transfer = False
                        should_end = False

                    # Cleanup recording
                    try:
                        os.unlink(wav_file)
                    except:
                        pass
                else:
                    response = "Recording failed. Let me try again."
                    should_transfer = False
                    should_end = False
            else:
                response = "I'm having trouble hearing you."
                should_transfer = False
                should_end = False

            # Speak response
            logger.info(f"Responding: {response[:30]}...")

            tts_file = tts.synthesize(response)
            if tts_file and os.path.exists(tts_file):
                asterisk_file = convert_audio_for_asterisk(tts_file)

                try:
                    os.unlink(tts_file)
                except:
                    pass

                if asterisk_file:
                    agi.stream_file(asterisk_file)
                else:
                    # Fallback to built-in sound
                    for fallback in ['demo-thanks', 'demo-congrats']:
                        if agi.stream_file(fallback):
                            break
            else:
                # Fallback to built-in sound
                for fallback in ['demo-thanks', 'demo-congrats']:
                    if agi.stream_file(fallback):
                        break

            # Check for exit conditions based on NETOVO AI response
            # 1. AI determined call should end
            if should_end:
                logger.info("NETOVO AI determined call should end")
                break

            # 2. AI determined transfer is needed
            if should_transfer:
                logger.info("NETOVO AI requested transfer - ending conversation")
                break

            # 3. Too many failed interactions
            if failed_interactions >= 3:
                logger.info("Too many failed interactions - ending conversation")
                break

            # 4. Maximum conversation time (15 minutes)
            if time.time() - start_time > 900:  # 15 minutes
                logger.info("Maximum conversation time reached - ending conversation")
                agi.stream_file("demo-thanks")  # Quick goodbye
                break

            # 5. Check if call is still connected
            if not agi.connected:
                logger.info("Call disconnected - ending conversation")
                break

            agi.sleep(1)

        # Log conversation summary
        summary = netovo_ai.get_conversation_summary()
        logger.info(f"Conversation Summary: {summary}")

        # End call
        logger.info("Ending call")
        agi.sleep(1)
        agi.hangup()

        logger.info("=== Fixed VoiceBot completed ===")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        try:
            agi = SimpleAGI()
            agi.answer()
            agi.verbose("VoiceBot error")
            agi.sleep(1)
            agi.hangup()
        except:
            pass

if __name__ == "__main__":
    main()
