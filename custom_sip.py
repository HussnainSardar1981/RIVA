"""
Custom SIP Client for 3CX Integration
Uses raw UDP sockets - no external SIP library dependencies
Implements basic SIP protocol for NETOVO Voice Bot
"""

import socket
import threading
import time
import uuid
import hashlib
import structlog
from typing import Optional, Callable, Dict, Any
from datetime import datetime

logger = structlog.get_logger()

class CustomSIPClient:
    """Custom SIP client using raw UDP sockets for 3CX integration"""

    def __init__(self, config: Dict[str, Any], call_callback: Optional[Callable] = None):
        """
        Initialize Custom SIP Client

        Args:
            config: SIP configuration dictionary
            call_callback: Function to call when incoming call received
        """
        self.config = config
        self.call_callback = call_callback

        # SIP configuration
        self.server = config.get('server', 'mtipbx.ny.3cx.us')
        self.port = config.get('port', 5060)
        self.username = config.get('username', '1600')
        self.password = config.get('password')
        self.auth_username = config.get('auth_username', self.username)
        self.display_name = config.get('display_name', 'NETOVO AI Voice Bot')

        # Local settings
        self.local_ip = self._get_local_ip()
        self.local_port = config.get('local_port', 5061)
        self.socket = None

        # State management
        self.registered = False
        self.running = False
        self.active_calls = {}
        self.sequence_number = 1
        self.call_id_counter = 0

        # Authentication
        self.auth_nonce = None
        self.auth_realm = None

        logger.info("Custom SIP Client initialized",
                   server=self.server,
                   username=self.username,
                   local_ip=self.local_ip)

    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Create a socket to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((self.server, self.port))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    async def start(self):
        """Start SIP client"""
        try:
            logger.info("Starting Custom SIP client...")

            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.local_port))
            self.socket.settimeout(1.0)  # Non-blocking with 1s timeout

            self.running = True

            # Start message handling thread
            self.message_thread = threading.Thread(target=self._message_handler)
            self.message_thread.daemon = True
            self.message_thread.start()

            # Register with server
            await self._register()

            logger.info("Custom SIP client started successfully")

        except Exception as e:
            logger.error("Failed to start Custom SIP client", error=str(e))
            raise

    def _message_handler(self):
        """Handle incoming SIP messages"""
        while self.running:
            try:
                if self.socket:
                    data, addr = self.socket.recvfrom(4096)
                    message = data.decode('utf-8')
                    self._process_message(message, addr)

            except socket.timeout:
                continue
            except Exception as e:
                logger.error("Error in message handler", error=str(e))

    def _process_message(self, message: str, addr: tuple):
        """Process incoming SIP message"""
        try:
            lines = message.strip().split('\r\n')
            if not lines:
                return

            request_line = lines[0]
            headers = {}

            # Parse headers
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()

            logger.debug("SIP message received", request=request_line[:50])

            # Handle different SIP methods
            if request_line.startswith('SIP/2.0'):
                # Response message
                self._handle_response(request_line, headers, addr)
            elif 'INVITE' in request_line:
                # Incoming call
                self._handle_invite(request_line, headers, addr)
            elif 'BYE' in request_line:
                # Call termination
                self._handle_bye(request_line, headers, addr)
            elif 'ACK' in request_line:
                # Call establishment
                self._handle_ack(request_line, headers, addr)

        except Exception as e:
            logger.error("Error processing SIP message", error=str(e))

    def _handle_response(self, response_line: str, headers: Dict, addr: tuple):
        """Handle SIP response"""
        try:
            parts = response_line.split(' ', 2)
            if len(parts) >= 2:
                status_code = int(parts[1])

                if status_code == 200:
                    # Success response
                    if 'contact' in headers:
                        self.registered = True
                        logger.info("SIP registration successful")
                elif status_code == 401:
                    # Authentication required
                    self._handle_auth_challenge(headers)
                else:
                    logger.warning("SIP response", code=status_code)

        except Exception as e:
            logger.error("Error handling SIP response", error=str(e))

    def _handle_invite(self, request_line: str, headers: Dict, addr: tuple):
        """Handle incoming INVITE (call)"""
        try:
            call_id = headers.get('call-id', str(uuid.uuid4()))
            from_header = headers.get('from', 'Unknown')
            to_header = headers.get('to', '')

            logger.info("Incoming INVITE", call_id=call_id, from_header=from_header)

            # Extract caller information
            caller_number = self._extract_number(from_header)

            # Store call information
            call_info = {
                'call_id': call_id,
                'caller_number': caller_number,
                'from_header': from_header,
                'to_header': to_header,
                'addr': addr,
                'timestamp': datetime.now().isoformat()
            }

            self.active_calls[call_id] = call_info

            # Send 200 OK response
            self._send_200_ok(call_id, headers, addr)

            # Notify callback about incoming call
            if self.call_callback:
                threading.Thread(
                    target=self.call_callback,
                    args=(call_info,)
                ).start()

        except Exception as e:
            logger.error("Error handling INVITE", error=str(e))

    def _handle_bye(self, request_line: str, headers: Dict, addr: tuple):
        """Handle BYE (call termination)"""
        try:
            call_id = headers.get('call-id')
            logger.info("Call termination received", call_id=call_id)

            # Send 200 OK response
            self._send_simple_response("200 OK", headers, addr)

            # Remove from active calls
            if call_id and call_id in self.active_calls:
                del self.active_calls[call_id]

        except Exception as e:
            logger.error("Error handling BYE", error=str(e))

    def _handle_ack(self, request_line: str, headers: Dict, addr: tuple):
        """Handle ACK (call establishment)"""
        try:
            call_id = headers.get('call-id')
            logger.info("Call established (ACK received)", call_id=call_id)

        except Exception as e:
            logger.error("Error handling ACK", error=str(e))

    def _extract_caller_number(self, from_header: str) -> str:
        """Extract caller number from SIP From header"""
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

    def _handle_auth_challenge(self, headers: Dict):
        """Handle authentication challenge"""
        try:
            www_auth = headers.get('www-authenticate', '')

            # Extract nonce and realm
            if 'nonce=' in www_auth:
                nonce_start = www_auth.find('nonce="') + 7
                nonce_end = www_auth.find('"', nonce_start)
                self.auth_nonce = www_auth[nonce_start:nonce_end]

            if 'realm=' in www_auth:
                realm_start = www_auth.find('realm="') + 7
                realm_end = www_auth.find('"', realm_start)
                self.auth_realm = www_auth[realm_start:realm_end]

            logger.info("Authentication challenge received", realm=self.auth_realm)

            # Re-send REGISTER with authentication
            self._send_register_with_auth()

        except Exception as e:
            logger.error("Error handling auth challenge", error=str(e))

    async def _register(self):
        """Send REGISTER request"""
        try:
            call_id = f"netovo-{uuid.uuid4().hex}"

            register_msg = self._create_register_message(call_id)
            self._send_message(register_msg)

            logger.info("REGISTER sent", server=self.server)

        except Exception as e:
            logger.error("Registration failed", error=str(e))

    def _send_register_with_auth(self):
        """Send REGISTER with authentication"""
        try:
            if not self.auth_nonce or not self.password:
                logger.error("Cannot authenticate - missing nonce or password")
                return

            call_id = f"netovo-auth-{uuid.uuid4().hex}"
            register_msg = self._create_register_message(call_id, with_auth=True)
            self._send_message(register_msg)

        except Exception as e:
            logger.error("Authenticated registration failed", error=str(e))

    def _create_register_message(self, call_id: str, with_auth: bool = False) -> str:
        """Create REGISTER message"""
        contact_uri = f"sip:{self.username}@{self.local_ip}:{self.local_port}"
        from_uri = f"sip:{self.username}@{self.server}"
        to_uri = from_uri

        msg_lines = [
            f"REGISTER sip:{self.server} SIP/2.0",
            f"Via: SIP/2.0/UDP {self.local_ip}:{self.local_port};branch=z9hG4bK{uuid.uuid4().hex[:8]}",
            f"From: <{from_uri}>;tag={uuid.uuid4().hex[:8]}",
            f"To: <{to_uri}>",
            f"Contact: <{contact_uri}>",
            f"Call-ID: {call_id}",
            f"CSeq: {self.sequence_number} REGISTER",
            "Max-Forwards: 70",
            "Expires: 3600",
            f"User-Agent: NETOVO-VoiceBot/2.0",
            "Content-Length: 0"
        ]

        if with_auth and self.auth_nonce and self.password:
            # Create digest authentication
            auth_response = self._create_digest_response("REGISTER", f"sip:{self.server}")
            auth_header = f'Digest username="{self.username}", realm="{self.auth_realm}", nonce="{self.auth_nonce}", uri="sip:{self.server}", response="{auth_response}"'
            msg_lines.insert(-1, f"Authorization: {auth_header}")

        self.sequence_number += 1

        return "\r\n".join(msg_lines) + "\r\n\r\n"

    def _create_digest_response(self, method: str, uri: str) -> str:
        """Create digest authentication response"""
        try:
            ha1 = hashlib.md5(f"{self.auth_username}:{self.auth_realm}:{self.password}".encode()).hexdigest()
            ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
            response = hashlib.md5(f"{ha1}:{self.auth_nonce}:{ha2}".encode()).hexdigest()
            return response
        except Exception as e:
            logger.error("Error creating digest response", error=str(e))
            return ""

    def _send_200_ok(self, call_id: str, request_headers: Dict, addr: tuple):
        """Send 200 OK response to INVITE"""
        try:
            # Create SDP for audio
            sdp = self._create_sdp()

            response_lines = [
                "SIP/2.0 200 OK",
                f"Via: {request_headers.get('via', '')}",
                f"From: {request_headers.get('from', '')}",
                f"To: {request_headers.get('to', '')};tag={uuid.uuid4().hex[:8]}",
                f"Call-ID: {call_id}",
                f"CSeq: {request_headers.get('cseq', '1 INVITE')}",
                f"Contact: <sip:{self.username}@{self.local_ip}:{self.local_port}>",
                "Content-Type: application/sdp",
                f"Content-Length: {len(sdp)}"
            ]

            response = "\r\n".join(response_lines) + "\r\n\r\n" + sdp
            self._send_message_to_addr(response, addr)

        except Exception as e:
            logger.error("Error sending 200 OK", error=str(e))

    def _send_simple_response(self, status: str, request_headers: Dict, addr: tuple):
        """Send simple response"""
        try:
            response_lines = [
                f"SIP/2.0 {status}",
                f"Via: {request_headers.get('via', '')}",
                f"From: {request_headers.get('from', '')}",
                f"To: {request_headers.get('to', '')}",
                f"Call-ID: {request_headers.get('call-id', '')}",
                f"CSeq: {request_headers.get('cseq', '')}",
                "Content-Length: 0"
            ]

            response = "\r\n".join(response_lines) + "\r\n\r\n"
            self._send_message_to_addr(response, addr)

        except Exception as e:
            logger.error("Error sending simple response", error=str(e))

    def _create_sdp(self) -> str:
        """Create SDP for audio session"""
        session_id = int(time.time())

        sdp = f"""v=0
o=netovo {session_id} {session_id} IN IP4 {self.local_ip}
s=NETOVO Voice Bot Session
c=IN IP4 {self.local_ip}
t=0 0
m=audio 10000 RTP/AVP 0
a=rtpmap:0 PCMU/8000
a=sendrecv
"""
        return sdp

    def _extract_number(self, from_header: str) -> str:
        """Extract phone number from From header"""
        try:
            if '<sip:' in from_header and '@' in from_header:
                start = from_header.find('<sip:') + 5
                end = from_header.find('@', start)
                if start > 4 and end > start:
                    return from_header[start:end]

            # Fallback extraction
            import re
            number_match = re.search(r'sip:(\d+)@', from_header)
            if number_match:
                return number_match.group(1)

            return "Unknown"

        except Exception as e:
            logger.error("Error extracting number", error=str(e))
            return "Unknown"

    def _send_message(self, message: str):
        """Send SIP message to server"""
        self._send_message_to_addr(message, (self.server, self.port))

    def _send_message_to_addr(self, message: str, addr: tuple):
        """Send message to specific address"""
        try:
            if self.socket:
                self.socket.sendto(message.encode('utf-8'), addr)
                logger.debug("SIP message sent", addr=addr, lines=len(message.split('\n')))
        except Exception as e:
            logger.error("Error sending message", error=str(e))

    async def transfer_call(self, call_id: str, target_extension: str) -> bool:
        """Transfer call to another extension"""
        try:
            logger.info("Transferring call", call_id=call_id, target=target_extension)

            call_info = self.active_calls.get(call_id)
            if not call_info:
                logger.error("Call not found for transfer", call_id=call_id)
                return False

            # Send REFER message for call transfer
            refer_msg = self._create_refer_message(call_id, target_extension)
            self._send_message_to_addr(refer_msg, call_info['addr'])

            logger.info("Call transfer initiated", call_id=call_id)
            return True

        except Exception as e:
            logger.error("Error transferring call", call_id=call_id, error=str(e))
            return False

    def _create_refer_message(self, call_id: str, target_extension: str) -> str:
        """Create REFER message for call transfer"""
        call_info = self.active_calls[call_id]
        refer_to = f"sip:{target_extension}@{self.server}"

        msg_lines = [
            f"REFER sip:{target_extension}@{self.server} SIP/2.0",
            f"Via: SIP/2.0/UDP {self.local_ip}:{self.local_port};branch=z9hG4bK{uuid.uuid4().hex[:8]}",
            f"From: {call_info['from_header']}",
            f"To: {call_info['to_header']}",
            f"Call-ID: {call_id}",
            f"CSeq: {self.sequence_number} REFER",
            f"Refer-To: {refer_to}",
            "Content-Length: 0"
        ]

        self.sequence_number += 1
        return "\r\n".join(msg_lines) + "\r\n\r\n"

    async def hangup_call(self, call_id: str) -> bool:
        """Hang up specific call"""
        try:
            logger.info("Hanging up call", call_id=call_id)

            call_info = self.active_calls.get(call_id)
            if not call_info:
                logger.error("Call not found for hangup", call_id=call_id)
                return False

            # Send BYE message
            bye_msg = self._create_bye_message(call_id)
            self._send_message_to_addr(bye_msg, call_info['addr'])

            # Remove from active calls
            del self.active_calls[call_id]

            logger.info("Call hung up successfully", call_id=call_id)
            return True

        except Exception as e:
            logger.error("Error hanging up call", call_id=call_id, error=str(e))
            return False

    def _create_bye_message(self, call_id: str) -> str:
        """Create BYE message"""
        call_info = self.active_calls[call_id]

        msg_lines = [
            f"BYE sip:{call_info['caller_number']}@{self.server} SIP/2.0",
            f"Via: SIP/2.0/UDP {self.local_ip}:{self.local_port};branch=z9hG4bK{uuid.uuid4().hex[:8]}",
            f"From: {call_info['to_header']}",
            f"To: {call_info['from_header']}",
            f"Call-ID: {call_id}",
            f"CSeq: {self.sequence_number} BYE",
            "Content-Length: 0"
        ]

        self.sequence_number += 1
        return "\r\n".join(msg_lines) + "\r\n\r\n"

    def get_active_calls(self) -> Dict[str, Dict]:
        """Get list of active calls"""
        return self.active_calls.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get SIP client status"""
        return {
            'running': self.running,
            'registered': self.registered,
            'server': self.server,
            'username': self.username,
            'local_ip': self.local_ip,
            'active_calls': len(self.active_calls),
            'call_ids': list(self.active_calls.keys())
        }

    async def stop(self):
        """Stop SIP client"""
        try:
            logger.info("Stopping Custom SIP client...")

            self.running = False

            # Hangup all active calls
            for call_id in list(self.active_calls.keys()):
                await self.hangup_call(call_id)

            # Close socket
            if self.socket:
                self.socket.close()
                self.socket = None

            self.registered = False
            logger.info("Custom SIP client stopped")

        except Exception as e:
            logger.error("Error stopping SIP client", error=str(e))

