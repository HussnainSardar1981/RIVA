"""
RTP Audio Bridge for 3CX Telephony Integration
Handles real-time audio streaming to/from 3CX calls using aiortc
"""

import asyncio
import logging
import struct
from typing import Optional, Callable
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import structlog

logger = structlog.get_logger()

class AudioStreamTrack(MediaStreamTrack):
    """Custom audio track for RTP streaming"""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self.audio_queue = asyncio.Queue()
        self.sample_rate = 8000  # 3CX standard
        self.channels = 1

    async def recv(self):
        """Receive audio frame for transmission"""
        try:
            # Get audio data from queue
            audio_data = await self.audio_queue.get()

            # Convert numpy array to audio frame
            frame = self._create_audio_frame(audio_data)
            return frame

        except Exception as e:
            logger.error("Error receiving audio frame", error=str(e))
            raise

    def _create_audio_frame(self, audio_data: np.ndarray):
        """Create audio frame from numpy data"""
        # Convert float32 to int16 for RTP transmission
        if audio_data.dtype == np.float32:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)

        # Create audio frame (simplified - would use proper aiortc AudioFrame)
        return audio_int16

    async def add_audio(self, audio_data: np.ndarray):
        """Add audio data to transmission queue"""
        await self.audio_queue.put(audio_data)

class RTPAudioBridge:
    """RTP Audio Bridge for 3CX Integration"""

    def __init__(self, audio_callback: Optional[Callable] = None):
        """
        Initialize RTP Bridge

        Args:
            audio_callback: Function to call with received audio data
        """
        self.audio_callback = audio_callback
        self.peer_connection: Optional[RTCPeerConnection] = None
        self.audio_track: Optional[AudioStreamTrack] = None
        self.is_running = False

        # Audio parameters for 3CX
        self.sample_rate = 8000  # G.711 PCMU standard
        self.frame_size = 160    # 20ms at 8kHz
        self.codec = "PCMU"      # G.711 μ-law

        logger.info("RTP Audio Bridge initialized",
                   sample_rate=self.sample_rate,
                   codec=self.codec)

    async def initialize(self):
        """Initialize RTP connection components"""
        try:
            # Create RTCPeerConnection
            self.peer_connection = RTCPeerConnection()

            # Create audio track for transmission
            self.audio_track = AudioStreamTrack()

            # Add track to peer connection
            self.peer_connection.addTrack(self.audio_track)

            # Set up event handlers
            self._setup_event_handlers()

            logger.info("RTP Bridge components initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize RTP Bridge", error=str(e))
            return False

    def _setup_event_handlers(self):
        """Set up peer connection event handlers"""

        @self.peer_connection.on("track")
        async def on_track(track):
            """Handle incoming audio track"""
            logger.info("Received track", kind=track.kind)

            if track.kind == "audio":
                # Start processing incoming audio
                asyncio.create_task(self._process_incoming_audio(track))

        @self.peer_connection.on("connectionstatechange")
        async def on_connection_state_change():
            """Handle connection state changes"""
            state = self.peer_connection.connectionState if self.peer_connection else "not_initialized"
            logger.info("RTP connection state changed", state=state)

            if state == "connected":
                self.is_running = True
            elif state in ["disconnected", "failed", "closed"]:
                self.is_running = False

    async def _process_incoming_audio(self, track):
        """Process incoming audio from 3CX"""
        logger.info("Starting incoming audio processing")

        try:
            while self.is_running:
                # Receive audio frame
                frame = await track.recv()

                # Convert frame to numpy array
                audio_data = self._frame_to_numpy(frame)

                # Process audio through callback
                if self.audio_callback and len(audio_data) > 0:
                    # Run callback in thread pool to avoid blocking
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.audio_callback, audio_data
                    )

        except Exception as e:
            logger.error("Error processing incoming audio", error=str(e))

    def _frame_to_numpy(self, frame) -> np.ndarray:
        """Convert audio frame to numpy array"""
        try:
            # Convert audio frame to numpy (simplified implementation)
            # In real implementation, would extract PCM data from frame
            if hasattr(frame, 'to_ndarray'):
                return frame.to_ndarray()
            else:
                # Fallback for basic frame types
                return np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0

        except Exception as e:
            logger.error("Error converting frame to numpy", error=str(e))
            return np.array([], dtype=np.float32)

    async def send_audio(self, audio_data: np.ndarray):
        """Send audio data to 3CX"""
        try:
            if self.audio_track and self.is_running:
                # Resample audio to 8kHz if needed
                if len(audio_data) > 0:
                    # Ensure audio is at 8kHz sample rate
                    processed_audio = self._prepare_audio_for_transmission(audio_data)

                    # Send to audio track
                    await self.audio_track.add_audio(processed_audio)

                    logger.debug("Audio sent to RTP stream", samples=len(processed_audio))
            else:
                logger.warning("Cannot send audio - RTP bridge not ready")

        except Exception as e:
            logger.error("Error sending audio", error=str(e))

    def _prepare_audio_for_transmission(self, audio_data: np.ndarray) -> np.ndarray:
        """Prepare audio for RTP transmission"""
        try:
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Convert to appropriate sample rate (8kHz for G.711)
            # Note: In production, use proper resampling from audio_processing module
            target_length = int(len(audio_data) * self.sample_rate / 22050)  # Assume input is 22kHz from TTS
            if target_length != len(audio_data):
                # Simple resampling (use soxr in production)
                indices = np.linspace(0, len(audio_data) - 1, target_length)
                audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)

            # Ensure audio is in float32 range [-1, 1]
            audio_data = np.clip(audio_data, -1.0, 1.0)

            return audio_data.astype(np.float32)

        except Exception as e:
            logger.error("Error preparing audio for transmission", error=str(e))
            return audio_data

    async def handle_sdp_offer(self, offer_sdp: str) -> str:
        """Handle SDP offer from 3CX and create answer"""
        try:
            # Create offer object
            offer = RTCSessionDescription(sdp=offer_sdp, type="offer")

            # Set remote description
            await self.peer_connection.setRemoteDescription(offer)

            # Create answer
            answer = await self.peer_connection.createAnswer()

            # Set local description
            await self.peer_connection.setLocalDescription(answer)

            logger.info("SDP offer processed, answer created")
            return answer.sdp

        except Exception as e:
            logger.error("Error handling SDP offer", error=str(e))
            raise

    async def start_bridge(self, remote_host: str, remote_port: int):
        """Start RTP bridge connection"""
        try:
            if not self.peer_connection:
                await self.initialize()

            logger.info("Starting RTP bridge", remote_host=remote_host, remote_port=remote_port)

            # In production, would establish RTP connection to remote_host:remote_port
            # This is simplified - actual implementation would use ICE candidates, STUN, etc.

            self.is_running = True
            logger.info("RTP bridge started successfully")

        except Exception as e:
            logger.error("Failed to start RTP bridge", error=str(e))
            raise

    async def stop_bridge(self):
        """Stop RTP bridge"""
        try:
            logger.info("Stopping RTP bridge")

            self.is_running = False

            if self.peer_connection:
                await self.peer_connection.close()
                self.peer_connection = None

            self.audio_track = None

            logger.info("RTP bridge stopped")

        except Exception as e:
            logger.error("Error stopping RTP bridge", error=str(e))

    def get_status(self) -> dict:
        """Get current bridge status"""
        return {
            "is_running": self.is_running,
            "connection_state": self.peer_connection.connectionState if self.peer_connection else "not_initialized",
            "sample_rate": self.sample_rate,
            "codec": self.codec
        }

class RTPSession:
    """Manages a single RTP session for a call"""

    def __init__(self, call_id: str, bridge: RTPAudioBridge):
        self.call_id = call_id
        self.bridge = bridge
        self.start_time = asyncio.get_event_loop().time()
        self.audio_packets_sent = 0
        self.audio_packets_received = 0

    async def process_audio(self, audio_data: np.ndarray):
        """Process audio for this session"""
        await self.bridge.send_audio(audio_data)
        self.audio_packets_sent += 1

    def get_session_stats(self) -> dict:
        """Get session statistics"""
        current_time = asyncio.get_event_loop().time()
        duration = current_time - self.start_time

        return {
            "call_id": self.call_id,
            "duration": duration,
            "packets_sent": self.audio_packets_sent,
            "packets_received": self.audio_packets_received,
            "bridge_status": self.bridge.get_status()
        }
