"""
Call Flow Manager for NETOVO AI Voice Bot Telephony Integration
Orchestrates complete call handling: SIP → RTP → Voice Bot → Response
"""

import asyncio
import structlog
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import uuid
import numpy as np

# Import existing voice bot components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import VoiceBotConfig directly to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import VoiceBot, VoiceBotConfig
from conversation_context import ConversationContext
from telephony.custom_sip import CustomSIPClient
from telephony.rtp_bridge import RTPAudioBridge, RTPSession
from telephony.audio_codec import AudioCodec

logger = structlog.get_logger()

def _import_voice_bot_classes():
    """Dynamically import VoiceBot classes to avoid circular imports"""
    try:
        import main
        return main.VoiceBot, main.VoiceBotConfig
    except ImportError as e:
        logger.error("Failed to import VoiceBot classes", error=str(e))
        return None, None

class CallFlowManager:
    """Manages complete call flow from 3CX to AI Voice Bot"""

    def __init__(self, sip_config: Dict[str, Any], voice_bot_config: Dict[str, Any]):
        """
        Initialize Call Flow Manager

        Args:
            sip_config: 3CX SIP configuration
            voice_bot_config: Voice bot configuration
        """
        self.sip_config = sip_config
        self.voice_bot_config = voice_bot_config

        # Initialize components
        self.sip_handler: Optional[CustomSIPClient] = None
        self.voice_bot: Optional[Any] = None
        self.audio_codec = AudioCodec()

        # Active sessions management
        self.active_sessions: Dict[str, CallSession] = {}
        self.running = False

        # Performance metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.call_stats = []

        logger.info("Call Flow Manager initialized")

    async def initialize(self) -> bool:
        """Initialize all telephony components"""
        try:
            logger.info("Initializing Call Flow Manager components...")

            # Step 1: Initialize Voice Bot
            VoiceBot, VoiceBotConfig = _import_voice_bot_classes()
            if not VoiceBot:
                logger.error("Failed to import VoiceBot class")
                return False

            self.voice_bot = VoiceBot(self.voice_bot_config)
            if not await self.voice_bot.initialize():
                logger.error("Voice Bot initialization failed")
                return False

            # Step 2: Initialize Custom SIP Handler
            self.sip_handler = CustomSIPClient(
                config=self.sip_config,
                call_callback=self._handle_incoming_call
            )

            logger.info("All Call Flow Manager components initialized successfully")
            return True

        except Exception as e:
            logger.error("Call Flow Manager initialization failed", error=str(e))
            return False

    async def start(self):
        """Start the call flow manager"""
        try:
            logger.info("Starting Call Flow Manager...")

            if not self.sip_handler:
                raise RuntimeError("Call Flow Manager not initialized")

            # Start SIP handler
            await self.sip_handler.start()

            self.running = True
            logger.info("Call Flow Manager started successfully")

            # Start background tasks
            asyncio.create_task(self._session_cleanup_task())
            asyncio.create_task(self._metrics_logging_task())

        except Exception as e:
            logger.error("Failed to start Call Flow Manager", error=str(e))
            raise

    async def _handle_incoming_call(self, call_info: Dict[str, Any]):
        """Handle incoming call from 3CX"""
        try:
            call_id = call_info['call_id']
            caller_number = call_info['caller_number']

            logger.info("Processing incoming call",
                       call_id=call_id,
                       caller=caller_number)

            # Create call session
            session = CallSession(
                call_id=call_id,
                caller_number=caller_number,
                manager=self,
                dialog=call_info['dialog']
            )

            # Store active session
            self.active_sessions[call_id] = session

            # Start call processing
            await session.start_call_processing()

            self.total_calls += 1

        except Exception as e:
            logger.error("Error handling incoming call", error=str(e))

    async def transfer_to_human(self, call_id: str, target_extension: str = "100") -> bool:
        """Transfer call to human agent"""
        try:
            logger.info("Transferring call to human agent",
                       call_id=call_id,
                       target=target_extension)

            if not self.sip_handler:
                return False

            # Use SIP handler to transfer call
            success = await self.sip_handler.transfer_call(call_id, target_extension)

            if success:
                # Clean up session
                if call_id in self.active_sessions:
                    session = self.active_sessions[call_id]
                    await session.end_call("transferred")
                    del self.active_sessions[call_id]

                logger.info("Call transferred successfully", call_id=call_id)
            else:
                logger.error("Call transfer failed", call_id=call_id)

            return success

        except Exception as e:
            logger.error("Error transferring call", call_id=call_id, error=str(e))
            return False

    async def end_call(self, call_id: str) -> bool:
        """End specific call"""
        try:
            logger.info("Ending call", call_id=call_id)

            # End session
            if call_id in self.active_sessions:
                session = self.active_sessions[call_id]
                await session.end_call("user_hangup")
                del self.active_sessions[call_id]

            # Hangup SIP call
            if self.sip_handler:
                success = await self.sip_handler.hangup_call(call_id)
                return success

            return True

        except Exception as e:
            logger.error("Error ending call", call_id=call_id, error=str(e))
            return False

    async def _session_cleanup_task(self):
        """Background task to clean up stale sessions"""
        while self.running:
            try:
                current_time = datetime.now()
                stale_sessions = []

                for call_id, session in self.active_sessions.items():
                    # Check if session is older than 30 minutes
                    if (current_time - session.start_time).total_seconds() > 1800:
                        stale_sessions.append(call_id)

                # Clean up stale sessions
                for call_id in stale_sessions:
                    logger.warning("Cleaning up stale session", call_id=call_id)
                    await self.end_call(call_id)

                # Wait 60 seconds before next cleanup
                await asyncio.sleep(60)

            except Exception as e:
                logger.error("Error in session cleanup task", error=str(e))
                await asyncio.sleep(60)

    async def _metrics_logging_task(self):
        """Background task to log performance metrics"""
        while self.running:
            try:
                logger.info("Call Flow Manager metrics",
                           active_sessions=len(self.active_sessions),
                           total_calls=self.total_calls,
                           successful_calls=self.successful_calls,
                           success_rate=f"{(self.successful_calls/max(1,self.total_calls)*100):.1f}%")

                # Wait 5 minutes before next metrics log
                await asyncio.sleep(300)

            except Exception as e:
                logger.error("Error in metrics logging task", error=str(e))
                await asyncio.sleep(300)

    async def stop(self):
        """Stop the call flow manager"""
        try:
            logger.info("Stopping Call Flow Manager...")

            self.running = False

            # End all active sessions
            for call_id in list(self.active_sessions.keys()):
                await self.end_call(call_id)

            # Stop SIP handler
            if self.sip_handler:
                await self.sip_handler.stop()

            # Cleanup voice bot
            if self.voice_bot:
                await self.voice_bot.cleanup()

            logger.info("Call Flow Manager stopped")

        except Exception as e:
            logger.error("Error stopping Call Flow Manager", error=str(e))

    def get_status(self) -> Dict[str, Any]:
        """Get call flow manager status"""
        return {
            'running': self.running,
            'active_sessions': len(self.active_sessions),
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'success_rate': self.successful_calls / max(1, self.total_calls),
            'sip_status': self.sip_handler.get_status() if self.sip_handler else None,
            'session_details': [session.get_summary() for session in self.active_sessions.values()]
        }

class CallSession:
    """Represents a single active call session"""

    def __init__(self, call_id: str, caller_number: str, manager: CallFlowManager, dialog):
        self.call_id = call_id
        self.caller_number = caller_number
        self.manager = manager
        self.dialog = dialog

        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.state = "initializing"

        # Audio and conversation components
        self.rtp_bridge: Optional[RTPAudioBridge] = None
        self.rtp_session: Optional[RTPSession] = None
        self.conversation_context = ConversationContext()

        # Session statistics
        self.audio_chunks_processed = 0
        self.voice_bot_interactions = 0
        self.escalation_requested = False

    async def start_call_processing(self):
        """Start processing the call"""
        try:
            logger.info("Starting call processing", call_id=self.call_id)

            self.state = "connecting"

            # Step 1: Initialize RTP bridge
            await self._initialize_rtp_bridge()

            # Step 2: Send greeting
            await self._send_greeting()

            # Step 3: Start audio processing loop
            self.state = "active"
            asyncio.create_task(self._audio_processing_loop())

            logger.info("Call processing started successfully", call_id=self.call_id)

        except Exception as e:
            logger.error("Error starting call processing", call_id=self.call_id, error=str(e))
            await self.end_call("initialization_error")

    async def _initialize_rtp_bridge(self):
        """Initialize RTP audio bridge for this call"""
        try:
            # Create RTP bridge with callback for incoming audio
            self.rtp_bridge = RTPAudioBridge(audio_callback=self._process_incoming_audio)

            # Initialize bridge
            await self.rtp_bridge.initialize()

            # Create RTP session
            self.rtp_session = RTPSession(self.call_id, self.rtp_bridge)

            # Start bridge (in production, would use actual RTP parameters from SDP)
            await self.rtp_bridge.start_bridge("localhost", 10000)

            logger.info("RTP bridge initialized", call_id=self.call_id)

        except Exception as e:
            logger.error("RTP bridge initialization failed", call_id=self.call_id, error=str(e))
            raise

    async def _send_greeting(self):
        """Send initial greeting to caller"""
        try:
            greeting_text = self.conversation_context.get_greeting_prompt()
            if not greeting_text:
                greeting_text = "Hello, thank you for calling NETOVO. I'm Alexis, your AI assistant. How can I help you today?"

            await self._send_voice_response(greeting_text)

        except Exception as e:
            logger.error("Error sending greeting", call_id=self.call_id, error=str(e))

    def _process_incoming_audio(self, audio_data: np.ndarray):
        """Process incoming audio from caller"""
        try:
            if len(audio_data) == 0:
                return

            # Convert telephony audio to ASR format
            asr_audio = self.manager.audio_codec.process_telephony_audio_chunk(
                self.manager.audio_codec.pcm16_to_pcmu(audio_data)
            )

            if len(asr_audio) > 0:
                # Queue for ASR processing
                asyncio.create_task(self._process_speech_async(asr_audio))
                self.audio_chunks_processed += 1

        except Exception as e:
            logger.error("Error processing incoming audio", call_id=self.call_id, error=str(e))

    async def _process_speech_async(self, audio_data: np.ndarray):
        """Process speech asynchronously"""
        try:
            # Simple VAD check
            if not self.manager.voice_bot.audio_processor.detect_speech(audio_data, 16000):
                return

            # Save audio to temporary file for ASR
            import tempfile
            import wave

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name

            # Save audio as WAV
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

            # Process through voice bot
            transcript = await self.manager.voice_bot.process_speech_to_text(temp_path)

            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass

            if transcript:
                logger.info("Speech recognized", call_id=self.call_id, transcript=transcript[:50])

                # Check for escalation keywords
                if self.conversation_context.should_escalate():
                    await self._handle_escalation()
                    return

                # Get AI response
                response = await self.manager.voice_bot.get_llm_response(transcript)

                if response:
                    # Send voice response
                    await self._send_voice_response(response)
                    self.voice_bot_interactions += 1

        except Exception as e:
            logger.error("Error processing speech", call_id=self.call_id, error=str(e))

    async def _send_voice_response(self, text: str):
        """Convert text to speech and send to caller"""
        try:
            logger.info("Sending voice response", call_id=self.call_id, text=text[:50])

            # Generate TTS
            tts_file = await self.manager.voice_bot.process_text_to_speech(text)

            if tts_file and os.path.exists(tts_file):
                # Load TTS audio
                import wave
                with wave.open(tts_file, 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    sample_rate = wav_file.getframerate()
                    tts_audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

                # Process for telephony
                pcmu_data = self.manager.audio_codec.process_tts_audio_chunk(tts_audio, sample_rate)

                # Send through RTP bridge
                if self.rtp_bridge and len(pcmu_data) > 0:
                    # Convert PCMU back to PCM for RTP bridge
                    pcm_audio = self.manager.audio_codec.pcmu_to_pcm16(pcmu_data)
                    await self.rtp_bridge.send_audio(pcm_audio)

                # Cleanup TTS file
                try:
                    os.unlink(tts_file)
                except:
                    pass

        except Exception as e:
            logger.error("Error sending voice response", call_id=self.call_id, error=str(e))

    async def _handle_escalation(self):
        """Handle escalation to human agent"""
        try:
            logger.info("Handling escalation request", call_id=self.call_id)

            self.escalation_requested = True

            # Send transfer message
            await self._send_voice_response(
                "I'll transfer you to one of our human agents who can better assist you. Please hold while I connect you."
            )

            # Wait a moment for TTS to complete
            await asyncio.sleep(3)

            # Transfer to human agent
            await self.manager.transfer_to_human(self.call_id)

        except Exception as e:
            logger.error("Error handling escalation", call_id=self.call_id, error=str(e))

    async def _audio_processing_loop(self):
        """Main audio processing loop for the call"""
        try:
            logger.info("Starting audio processing loop", call_id=self.call_id)

            while self.state == "active":
                # Audio processing is handled by callbacks
                # This loop just maintains the session
                await asyncio.sleep(1)

                # Check for session timeout
                if (datetime.now() - self.start_time).total_seconds() > 1800:  # 30 minutes
                    logger.warning("Session timeout", call_id=self.call_id)
                    await self.end_call("timeout")
                    break

        except Exception as e:
            logger.error("Error in audio processing loop", call_id=self.call_id, error=str(e))
            await self.end_call("processing_error")

    async def end_call(self, reason: str):
        """End the call session"""
        try:
            logger.info("Ending call session", call_id=self.call_id, reason=reason)

            self.state = "ending"
            self.end_time = datetime.now()

            # Stop RTP bridge
            if self.rtp_bridge:
                await self.rtp_bridge.stop_bridge()

            # Update manager statistics
            if reason not in ["initialization_error", "processing_error"]:
                self.manager.successful_calls += 1

            # Log call statistics
            call_duration = (self.end_time - self.start_time).total_seconds()

            logger.info("Call session ended",
                       call_id=self.call_id,
                       duration=f"{call_duration:.1f}s",
                       audio_chunks=self.audio_chunks_processed,
                       interactions=self.voice_bot_interactions,
                       escalated=self.escalation_requested,
                       reason=reason)

            self.state = "ended"

        except Exception as e:
            logger.error("Error ending call session", call_id=self.call_id, error=str(e))

    def get_summary(self) -> Dict[str, Any]:
        """Get call session summary"""
        current_time = datetime.now()
        duration = (self.end_time or current_time) - self.start_time

        return {
            'call_id': self.call_id,
            'caller_number': self.caller_number,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': duration.total_seconds(),
            'state': self.state,
            'audio_chunks_processed': self.audio_chunks_processed,
            'voice_bot_interactions': self.voice_bot_interactions,
            'escalation_requested': self.escalation_requested,
            'conversation_turns': len(self.conversation_context.turns)
        }
