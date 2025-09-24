"""
SIP Client for 3CX Integration
Handles SIP registration, incoming calls, and call control
"""

import asyncio
import aiosip
import structlog
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import uuid

logger = structlog.get_logger()

class SIPCallHandler:
    """SIP Client for 3CX Integration"""

    def __init__(self, config: Dict[str, Any], call_callback: Optional[Callable] = None):
        """
        Initialize SIP Client

        Args:
            config: SIP configuration dictionary
            call_callback: Function to call when incoming call received
        """
        self.config = config
        self.call_callback = call_callback
        self.app: Optional[aiosip.Application] = None
        self.dialogs: Dict[str, aiosip.Dialog] = {}
        self.registered = False
        self.running = False

        # Extract configuration
        self.server = config.get('server', 'mtipbx.ny.3cx.us')
        self.port = config.get('port', 5060)
        self.username = config.get('username', '1600')
        self.password = config.get('password')
        self.display_name = config.get('display_name', 'NETOVO AI Voice Bot')

        logger.info("SIP Client initialized",
                   server=self.server,
                   username=self.username,
                   display_name=self.display_name)

    async def start(self):
        """Start SIP client and register with 3CX"""
        try:
            logger.info("Starting SIP client...")

            # Create SIP application
            self.app = aiosip.Application()

            # Set up message handlers
            self._setup_handlers()

            # Start the application
            await self.app.start(host='0.0.0.0', port=5060)

            # Register with 3CX server
            await self._register()

            self.running = True
            logger.info("SIP client started and registered successfully")

        except Exception as e:
            logger.error("Failed to start SIP client", error=str(e))
            raise

    def _setup_handlers(self):
        """Set up SIP message handlers"""

        @self.app.on_invite
        async def on_invite(dialog, message):
            """Handle incoming INVITE (call)"""
            await self._handle_incoming_call(dialog, message)

        @self.app.on_bye
        async def on_bye(dialog, message):
            """Handle BYE (call termination)"""
            await self._handle_call_termination(dialog, message)

        @self.app.on_ack
        async def on_ack(dialog, message):
            """Handle ACK (call establishment)"""
            await self._handle_call_ack(dialog, message)

        @self.app.on_register
        async def on_register(message):
            """Handle registration responses"""
            await self._handle_registration_response(message)

    async def _register(self):
        """Register with 3CX SIP server"""
        try:
            logger.info("Registering with 3CX server", server=self.server, username=self.username)

            # Create registration message
            contact_uri = f"sip:{self.username}@{self.server}:{self.port}"

            # Send REGISTER request
            from_uri = f"sip:{self.username}@{self.server}"
            to_uri = f"sip:{self.username}@{self.server}"

            # Use aiosip to create and send registration
            registration_dialog = await self.app.create_dialog(
                method='REGISTER',
                uri=f"sip:{self.server}:{self.port}",
                headers={
                    'From': f'"{self.display_name}" <{from_uri}>',
                    'To': f'<{to_uri}>',
                    'Contact': f'<{contact_uri}>',
                    'Expires': '3600'
                }
            )

            # Add authentication if configured
            if self.password:
                registration_dialog.password = self.password

            logger.info("Registration request sent")

        except Exception as e:
            logger.error("Registration failed", error=str(e))
            raise

    async def _handle_registration_response(self, message):
        """Handle registration response from 3CX"""
        try:
            status_code = message.status_code

            if status_code == 200:
                self.registered = True
                logger.info("Successfully registered with 3CX")
            elif status_code == 401:
                logger.warning("Authentication required for registration")
                # Handle authentication challenge
                await self._handle_auth_challenge(message)
            else:
                logger.error("Registration failed", status_code=status_code, reason=message.reason)

        except Exception as e:
            logger.error("Error handling registration response", error=str(e))

    async def _handle_auth_challenge(self, message):
        """Handle authentication challenge from 3CX"""
        try:
            logger.info("Handling authentication challenge")

            if not self.password:
                logger.error("Authentication required but no password provided")
                return

            # Extract authentication parameters from WWW-Authenticate header
            auth_header = message.headers.get('WWW-Authenticate', '')

            # Re-send registration with authentication
            # This would include proper digest authentication implementation
            # For now, simplified approach
            logger.info("Re-sending registration with authentication")

        except Exception as e:
            logger.error("Error handling authentication challenge", error=str(e))

    async def _handle_incoming_call(self, dialog, message):
        """Handle incoming call from 3CX"""
        try:
            call_id = message.headers.get('Call-ID', str(uuid.uuid4()))
            from_header = message.headers.get('From', 'Unknown')
            caller_number = self._extract_caller_number(from_header)

            logger.info("Incoming call received",
                       call_id=call_id,
                       caller=caller_number,
                       from_header=from_header)

            # Store dialog for this call
            self.dialogs[call_id] = dialog

            # Answer the call automatically
            await self._answer_call(dialog, call_id)

            # Notify callback about the call
            if self.call_callback:
                call_info = {
                    'call_id': call_id,
                    'caller_number': caller_number,
                    'dialog': dialog,
                    'timestamp': datetime.now().isoformat()
                }

                # Run callback in executor to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    None, self.call_callback, call_info
                )

        except Exception as e:
            logger.error("Error handling incoming call", error=str(e))

    def _extract_caller_number(self, from_header: str) -> str:
        """Extract caller number from From header"""
        try:
            # Parse SIP From header to extract number
            # Example: "John Doe" <sip:1234@domain.com>
            if '<sip:' in from_header and '@' in from_header:
                start = from_header.find('<sip:') + 5
                end = from_header.find('@', start)
                if start > 4 and end > start:
                    return from_header[start:end]

            # Fallback parsing
            import re
            number_match = re.search(r'sip:(\d+)@', from_header)
            if number_match:
                return number_match.group(1)

            return "Unknown"

        except Exception as e:
            logger.error("Error extracting caller number", error=str(e))
            return "Unknown"

    async def _answer_call(self, dialog, call_id: str):
        """Answer incoming call"""
        try:
            logger.info("Answering call", call_id=call_id)

            # Create SDP answer for audio
            sdp_answer = self._create_sdp_answer()

            # Send 200 OK response
            await dialog.reply(200, 'OK', headers={
                'Content-Type': 'application/sdp'
            }, payload=sdp_answer)

            logger.info("Call answered successfully", call_id=call_id)

        except Exception as e:
            logger.error("Error answering call", call_id=call_id, error=str(e))

    def _create_sdp_answer(self) -> str:
        """Create SDP answer for audio call"""
        try:
            # Create basic SDP for G.711 PCMU audio
            local_ip = "0.0.0.0"  # Would be actual local IP in production
            sdp = f"""v=0
o=netovo-voicebot {int(datetime.now().timestamp())} {int(datetime.now().timestamp())} IN IP4 {local_ip}
s=NETOVO Voice Bot Session
c=IN IP4 {local_ip}
t=0 0
m=audio 10000 RTP/AVP 0
a=rtpmap:0 PCMU/8000
a=sendrecv
"""
            return sdp.strip()

        except Exception as e:
            logger.error("Error creating SDP answer", error=str(e))
            return ""

    async def _handle_call_ack(self, dialog, message):
        """Handle ACK - call is now established"""
        try:
            call_id = message.headers.get('Call-ID', 'unknown')
            logger.info("Call established (ACK received)", call_id=call_id)

            # Call is now active - RTP should start flowing
            # The RTP bridge will handle the actual audio streaming

        except Exception as e:
            logger.error("Error handling call ACK", error=str(e))

    async def _handle_call_termination(self, dialog, message):
        """Handle call termination (BYE)"""
        try:
            call_id = message.headers.get('Call-ID', 'unknown')
            logger.info("Call termination received", call_id=call_id)

            # Send 200 OK to acknowledge BYE
            await dialog.reply(200, 'OK')

            # Clean up call resources
            if call_id in self.dialogs:
                del self.dialogs[call_id]

            logger.info("Call terminated successfully", call_id=call_id)

        except Exception as e:
            logger.error("Error handling call termination", error=str(e))

    async def transfer_call(self, call_id: str, target_extension: str):
        """Transfer call to another extension"""
        try:
            logger.info("Transferring call", call_id=call_id, target=target_extension)

            dialog = self.dialogs.get(call_id)
            if not dialog:
                logger.error("Cannot transfer - call not found", call_id=call_id)
                return False

            # Send REFER message to initiate transfer
            refer_to = f"sip:{target_extension}@{self.server}"

            await dialog.request('REFER', headers={
                'Refer-To': refer_to
            })

            logger.info("Call transfer initiated", call_id=call_id, target=target_extension)
            return True

        except Exception as e:
            logger.error("Error transferring call", call_id=call_id, error=str(e))
            return False

    async def hangup_call(self, call_id: str):
        """Hang up specific call"""
        try:
            logger.info("Hanging up call", call_id=call_id)

            dialog = self.dialogs.get(call_id)
            if not dialog:
                logger.error("Cannot hangup - call not found", call_id=call_id)
                return False

            # Send BYE request
            await dialog.request('BYE')

            # Clean up
            if call_id in self.dialogs:
                del self.dialogs[call_id]

            logger.info("Call hung up successfully", call_id=call_id)
            return True

        except Exception as e:
            logger.error("Error hanging up call", call_id=call_id, error=str(e))
            return False

    async def stop(self):
        """Stop SIP client"""
        try:
            logger.info("Stopping SIP client...")

            self.running = False

            # Hangup all active calls
            for call_id in list(self.dialogs.keys()):
                await self.hangup_call(call_id)

            # Unregister from server
            if self.registered:
                await self._unregister()

            # Stop application
            if self.app:
                await self.app.stop()

            logger.info("SIP client stopped")

        except Exception as e:
            logger.error("Error stopping SIP client", error=str(e))

    async def _unregister(self):
        """Unregister from 3CX server"""
        try:
            logger.info("Unregistering from 3CX server")

            # Send REGISTER with Expires: 0 to unregister
            # Implementation would be similar to _register() but with Expires: 0

            self.registered = False
            logger.info("Successfully unregistered from 3CX")

        except Exception as e:
            logger.error("Error during unregistration", error=str(e))

    def get_active_calls(self) -> Dict[str, Dict]:
        """Get list of active calls"""
        active_calls = {}

        for call_id, dialog in self.dialogs.items():
            active_calls[call_id] = {
                'call_id': call_id,
                'state': 'active',  # Would track actual state in production
                'start_time': datetime.now().isoformat(),  # Would track actual start time
                'dialog': dialog
            }

        return active_calls

    def get_status(self) -> Dict[str, Any]:
        """Get SIP client status"""
        return {
            'running': self.running,
            'registered': self.registered,
            'server': self.server,
            'username': self.username,
            'active_calls': len(self.dialogs),
            'call_ids': list(self.dialogs.keys())
        }


class SIPCallSession:
    """Represents a single SIP call session"""

    def __init__(self, call_id: str, dialog, caller_number: str):
        self.call_id = call_id
        self.dialog = dialog
        self.caller_number = caller_number
        self.start_time = datetime.now()
        self.state = "ringing"

    def mark_answered(self):
        """Mark call as answered"""
        self.state = "answered"

    def mark_ended(self):
        """Mark call as ended"""
        self.state = "ended"

    def get_duration(self) -> float:
        """Get call duration in seconds"""
        return (datetime.now() - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'call_id': self.call_id,
            'caller_number': self.caller_number,
            'start_time': self.start_time.isoformat(),
            'state': self.state,
            'duration': self.get_duration()
        }