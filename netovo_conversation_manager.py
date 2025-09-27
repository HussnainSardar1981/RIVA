#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
NETOVO Professional Conversation Manager
Implements the full NETOVO.json conversation flow with Ollama
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class NetovoConversationManager:
    """Professional NETOVO IT Support conversation manager"""

    def __init__(self):
        self.conversation_state = "greeting"
        self.current_issue = None
        self.troubleshooting_step = 0
        self.failed_attempts = 0
        self.user_phone = None
        self.user_name = None

        # Knowledge base scripts
        self.knowledge_base = {
            "wifi": self._wifi_troubleshooting,
            "internet": self._wifi_troubleshooting,
            "wireless": self._wifi_troubleshooting,
            "connection": self._wifi_troubleshooting,
            "slow": self._performance_troubleshooting,
            "performance": self._performance_troubleshooting,
            "computer": self._performance_troubleshooting,
            "keyboard": self._keyboard_mouse_troubleshooting,
            "mouse": self._keyboard_mouse_troubleshooting,
            "vpn": self._vpn_troubleshooting,
            "printer": self._printer_troubleshooting,
            "printing": self._printer_troubleshooting,
            "password": self._password_reset,
            "login": self._password_reset,
            "forgot": self._password_reset,
            "mfa": self._mfa_troubleshooting,
            "authentication": self._mfa_troubleshooting,
            "email": self._email_troubleshooting,
            "outlook": self._email_troubleshooting,
            "remote": self._remote_desktop_troubleshooting,
            "desktop": self._remote_desktop_troubleshooting
        }

        # High priority keywords requiring immediate escalation
        self.high_priority_keywords = [
            "system down", "security breach", "hacked", "virus", "malware",
            "data loss", "server down", "network down", "emergency",
            "critical", "urgent", "production down", "can't access anything"
        ]

        # Issues requiring immediate escalation
        self.escalation_issues = [
            "hardware", "monitor", "screen", "display", "new installation",
            "software installation", "licensing", "billing", "account setup",
            "network configuration", "firewall", "server", "domain"
        ]

    def generate_response(self, user_input: str, phone_number: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate professional NETOVO response
        Returns: (response_text, action_type)
        action_type: 'continue', 'transfer', 'end_call', 'send_sms'
        """
        user_input = user_input.lower().strip()

        if phone_number:
            self.user_phone = phone_number

        # Check for high priority issues first
        if self._is_high_priority(user_input):
            return self._handle_high_priority()

        # State machine handling
        if self.conversation_state == "greeting":
            return self._handle_greeting(user_input)
        elif self.conversation_state == "triage":
            return self._handle_triage(user_input)
        elif self.conversation_state == "troubleshooting":
            return self._handle_troubleshooting(user_input)
        elif self.conversation_state == "escalation_confirm":
            return self._handle_escalation_confirm(user_input)
        elif self.conversation_state == "name_collection":
            return self._handle_name_collection(user_input)
        else:
            return self._handle_fallback(user_input)

    def _is_high_priority(self, user_input: str) -> bool:
        """Check if user input contains high priority keywords"""
        return any(keyword in user_input for keyword in self.high_priority_keywords)

    def _needs_escalation(self, user_input: str) -> bool:
        """Check if issue requires immediate escalation"""
        return any(keyword in user_input for keyword in self.escalation_issues)

    def _identify_issue_category(self, user_input: str) -> Optional[str]:
        """Identify which knowledge base category applies"""
        for keyword, handler in self.knowledge_base.items():
            if keyword in user_input:
                return keyword
        return None

    def _handle_greeting(self, user_input: str) -> Tuple[str, str]:
        """Handle initial greeting and issue capture"""
        self.conversation_state = "triage"

        # Extract issue category
        self.current_issue = self._identify_issue_category(user_input)

        if self.current_issue:
            issue_name = self._get_issue_display_name(user_input)
            response = f"Okay, I understand you're having an issue with {issue_name}."

            # Check if needs immediate escalation
            if self._needs_escalation(user_input):
                self.conversation_state = "escalation_confirm"
                response += " For an issue like that, a technician will need to take a look. Would you like me to transfer you to our live support team now?"
                return response, "continue"
            else:
                # Start troubleshooting
                self.conversation_state = "troubleshooting"
                self.troubleshooting_step = 0
                kb_response = self.knowledge_base[self.current_issue]()
                return response + " " + kb_response, "continue"
        else:
            # Couldn't identify issue - ask for clarification
            return "I'm sorry, I didn't catch the specific issue. Could you please state the technical problem you're experiencing?", "continue"

    def _handle_triage(self, user_input: str) -> Tuple[str, str]:
        """Handle issue triage after greeting"""
        # This handles cases where greeting didn't capture the issue
        self.current_issue = self._identify_issue_category(user_input)

        if self.current_issue:
            if self._needs_escalation(user_input):
                self.conversation_state = "escalation_confirm"
                return "For an issue like that, a technician will need to take a look. Would you like me to transfer you to our live support team now?", "continue"
            else:
                self.conversation_state = "troubleshooting"
                self.troubleshooting_step = 0
                return self.knowledge_base[self.current_issue](), "continue"
        else:
            # Still couldn't identify - escalate
            self.conversation_state = "escalation_confirm"
            return "I'm going to connect you with our support team so they can better assist. Would you like me to transfer you now?", "continue"

    def _handle_troubleshooting(self, user_input: str) -> Tuple[str, str]:
        """Handle ongoing troubleshooting steps"""
        # Check for user indicating they want to give up or need help
        if any(phrase in user_input for phrase in ["doesn't work", "didn't work", "still not working", "same problem", "no change"]):
            self.troubleshooting_step += 1

            # Try next step if available
            if hasattr(self, f'_get_{self.current_issue}_step_{self.troubleshooting_step}'):
                method = getattr(self, f'_get_{self.current_issue}_step_{self.troubleshooting_step}')
                return method(), "continue"
            else:
                # No more steps - escalate
                self.conversation_state = "escalation_confirm"
                return "It seems we've tried the initial steps and the issue persists. This will require a technician. I'm transferring you to our live support team now. I've logged the steps we took so you don't have to repeat them.", "transfer"

        elif any(phrase in user_input for phrase in ["yes", "okay", "sure", "done", "tried", "did that"]):
            # User completed step - ask for result
            return "Did that resolve the issue?", "continue"

        else:
            # Continue with current troubleshooting
            return "Please let me know when you've completed that step.", "continue"

    def _handle_escalation_confirm(self, user_input: str) -> Tuple[str, str]:
        """Handle escalation confirmation"""
        if any(word in user_input for word in ["yes", "okay", "sure", "transfer"]):
            return "Certainly. Connecting you now, please hold.", "transfer"
        else:
            self.conversation_state = "name_collection"
            return f"Okay, understood. A ticket will be created for a technician to call you back at this number. Can I get your first name to add to the ticket?", "continue"

    def _handle_name_collection(self, user_input: str) -> Tuple[str, str]:
        """Handle name collection for callback"""
        # Extract likely name from input
        words = user_input.split()
        potential_names = [word.capitalize() for word in words if word.isalpha() and len(word) > 1]

        if potential_names:
            self.user_name = potential_names[0]
            return f"Thank you, {self.user_name}. A technician will call you back at {self.user_phone or 'this number'} within the next business day. Is there anything else I can help you with today?", "continue"
        else:
            return "I didn't catch your name. Could you spell it for me?", "continue"

    def _handle_high_priority(self) -> Tuple[str, str]:
        """Handle high priority issues requiring immediate escalation"""
        return "Understood. This requires immediate attention. I am transferring you to a live technician right now. Please hold.", "transfer"

    def _handle_fallback(self, user_input: str) -> Tuple[str, str]:
        """Handle unrecognized input"""
        return "I'm sorry, I didn't catch that. Could you please state the technical issue again?", "continue"

    def _get_issue_display_name(self, user_input: str) -> str:
        """Get user-friendly display name for the issue"""
        if any(word in user_input for word in ["wifi", "wireless", "internet", "connection"]):
            return "your Wi-Fi connection"
        elif any(word in user_input for word in ["slow", "performance"]):
            return "slow computer performance"
        elif any(word in user_input for word in ["keyboard", "mouse"]):
            return "your keyboard or mouse"
        elif "vpn" in user_input:
            return "VPN connection"
        elif any(word in user_input for word in ["printer", "printing"]):
            return "your printer"
        elif any(word in user_input for word in ["password", "login", "forgot"]):
            return "password issues"
        elif "mfa" in user_input or "authentication" in user_input:
            return "multi-factor authentication"
        elif any(word in user_input for word in ["email", "outlook"]):
            return "email synchronization"
        elif any(word in user_input for word in ["remote", "desktop"]):
            return "remote desktop access"
        else:
            return "that issue"

    # Knowledge Base Troubleshooting Scripts
    def _wifi_troubleshooting(self) -> str:
        steps = [
            "Let's start with a full restart of both your computer and your Wi-Fi router. Have you tried that yet?",
            "Alright. Next, let's try forgetting the Wi-Fi network in your computer's settings and reconnecting. Can you do that now?",
            "Now let's check if other devices can connect to the same Wi-Fi. Can you test with your phone?",
            "Let's check if airplane mode is turned off on your computer. Can you verify that?",
            "We'll need to update your Wi-Fi drivers. Can you check Windows Update or your device manager?"
        ]
        if self.troubleshooting_step < len(steps):
            return steps[self.troubleshooting_step]
        return "Let's try connecting to a different network to test if the issue is with your device or the network."

    def _performance_troubleshooting(self) -> str:
        steps = [
            "Let's start by restarting your computer completely. Have you done that recently?",
            "Now let's close any unnecessary programs running in the background. Can you check your task manager?",
            "Let's check for Windows updates. Can you run Windows Update now?",
            "Let's run a quick antivirus scan to check for any malware. Do you have antivirus software installed?",
            "We should clear your temporary files. Can you run Disk Cleanup utility?"
        ]
        if self.troubleshooting_step < len(steps):
            return steps[self.troubleshooting_step]
        return "Let's check how much free space you have on your hard drive. Can you check that for me?"

    def _keyboard_mouse_troubleshooting(self) -> str:
        steps = [
            "Let's start by checking if your keyboard and mouse have fresh batteries if they're wireless.",
            "Now let's try plugging them into a different USB port on your computer.",
            "Let's test them on a different computer if you have one available.",
            "We should update the device drivers. Can you check Device Manager for any error indicators?"
        ]
        if self.troubleshooting_step < len(steps):
            return steps[self.troubleshooting_step]
        return "Let's try using the built-in on-screen keyboard to see if it's a hardware issue."

    def _vpn_troubleshooting(self) -> str:
        steps = [
            "Let's restart your VPN client application completely. Can you close it and reopen it?",
            "Now let's check your internet connection without VPN. Can you browse the web normally?",
            "Let's try re-entering your VPN credentials. Do you have your username and password handy?",
            "Let's restart your computer and try connecting again.",
            "We should update your VPN client software. Can you check for updates?"
        ]
        if self.troubleshooting_step < len(steps):
            return steps[self.troubleshooting_step]
        return "Let's try connecting to a different VPN server location if that option is available."

    def _printer_troubleshooting(self) -> str:
        steps = [
            "Let's start by checking if your printer is powered on and showing any error messages.",
            "Now let's restart both your printer and your computer.",
            "Let's check if your printer is connected to the network or USB properly.",
            "Let's verify you have paper loaded and sufficient ink or toner.",
            "We should update your printer drivers. Can you check the manufacturer's website?"
        ]
        if self.troubleshooting_step < len(steps):
            return steps[self.troubleshooting_step]
        return "Let's try printing from a different device to test if it's a computer-specific issue."

    def _password_reset(self) -> str:
        """Password reset handling"""
        return "I can help with a password reset. What type of account password do you need to reset?"

    def _mfa_troubleshooting(self) -> str:
        steps = [
            "Let's make sure your device's clock is synchronized. Can you check your date and time settings?",
            "Now let's try re-registering your device with the MFA system.",
            "Let's update or reinstall your authenticator app if you're using one."
        ]
        if self.troubleshooting_step < len(steps):
            return steps[self.troubleshooting_step]
        return "We may need to reset your MFA settings completely."

    def _email_troubleshooting(self) -> str:
        steps = [
            "Let's restart Outlook completely. Can you close it and reopen it?",
            "Now let's confirm you're connected to the internet and can browse websites.",
            "Let's try removing and re-adding your email account in Outlook.",
            "We should check for Outlook updates. Can you run Office updates?"
        ]
        if self.troubleshooting_step < len(steps):
            return steps[self.troubleshooting_step]
        return "Let's check your email account settings and server configurations."

    def _remote_desktop_troubleshooting(self) -> str:
        steps = [
            "Let's make sure your VPN is connected first. Can you verify your VPN status?",
            "Now let's verify the destination computer is powered on and connected.",
            "Let's double-check your remote desktop credentials and server address.",
            "Let's reboot your local computer and try connecting again."
        ]
        if self.troubleshooting_step < len(steps):
            return steps[self.troubleshooting_step]
        return "We may need to check the remote desktop service settings on the target computer."

    def get_greeting(self) -> str:
        """Get the initial greeting"""
        return "Thank you for calling Netovo Support. This is Alexis. What's the issue you're experiencing today?"

    def should_end_conversation(self, user_input: str) -> bool:
        """Check if conversation should end based on user input"""
        end_phrases = [
            "that's all", "nothing else", "you've helped", "problem solved",
            "all set", "thank you", "thanks", "goodbye", "bye"
        ]
        return any(phrase in user_input.lower() for phrase in end_phrases)
