#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
NETOVO Professional Ollama Client
Integrates the NETOVO conversation system with Ollama LLM using comprehensive prompt-based logic
"""

import logging
import httpx
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class NetovoOllamaClient:
    """Professional NETOVO-enhanced Ollama client with comprehensive prompt-based conversation logic"""

    def __init__(self, model="orca2:7b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.conversation_history = []

        # Import the conversation manager to get the comprehensive prompt
        from netovo_conversation_manager import NetovoConversationManager
        self.conversation_manager = NetovoConversationManager()
        self.system_prompt = self.conversation_manager.get_netovo_prompt()

    def generate_response(self, user_input: str, context: str = "") -> Tuple[str, bool, bool]:
        """
        Generate professional NETOVO response using comprehensive prompt-based logic
        Returns: (response_text, should_transfer, should_end)
        """
        try:
            # Build conversation context
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]

            # Add conversation history (last 6 messages to maintain context)
            if self.conversation_history:
                for msg in self.conversation_history[-6:]:
                    messages.append(msg)

            # Add current context if provided
            if context:
                context_msg = f"Context: {context}\n\nUser: {user_input}"
            else:
                context_msg = user_input

            messages.append({"role": "user", "content": context_msg})

            # Generate response with Ollama
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": 150,  # Allow for ACTION/RESPONSE format
                    "temperature": 0.1,  # Keep responses consistent
                    "top_p": 0.9,
                    "stop": ["\n\n", "User:", "Assistant:"]  # Stop at natural breaks
                }
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()

                result = response.json()
                ai_response = result["message"]["content"].strip()

                # Parse ACTION/RESPONSE format
                response_text, should_transfer, should_end = self._parse_ai_response(ai_response)

                # Clean up response
                response_text = self._clean_response(response_text)

                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": response_text})

                # Keep history manageable
                if len(self.conversation_history) > 12:
                    self.conversation_history = self.conversation_history[-8:]

                logger.info(f"NETOVO AI response: {response_text[:100]}")
                logger.info(f"Action parsed - Transfer: {should_transfer}, End: {should_end}")

                return response_text, should_transfer, should_end

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "I'm experiencing technical difficulties. Let me transfer you to our support team for immediate assistance.", True, False

    def _parse_ai_response(self, ai_response: str) -> Tuple[str, bool, bool]:
        """
        Parse the AI response in ACTION/RESPONSE format
        Returns: (response_text, should_transfer, should_end)
        """
        should_transfer = False
        should_end = False
        response_text = ai_response

        try:
            # Look for ACTION: and RESPONSE: format
            if "ACTION:" in ai_response and "RESPONSE:" in ai_response:
                lines = ai_response.split('\n')
                action_line = ""
                response_line = ""

                for line in lines:
                    if line.startswith("ACTION:"):
                        action_line = line.replace("ACTION:", "").strip().lower()
                    elif line.startswith("RESPONSE:"):
                        response_line = line.replace("RESPONSE:", "").strip()

                if action_line and response_line:
                    response_text = response_line

                    # Parse actions
                    if "transfer" in action_line:
                        should_transfer = True
                    elif "end_call" in action_line:
                        should_end = True
                    elif "send_sms" in action_line:
                        # For now, treat SMS as continue (we'll add SMS functionality later)
                        pass
                    # "continue" is default - no flags set

            logger.info(f"Parsed action - Transfer: {should_transfer}, End: {should_end}")

        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            # Fallback to original response
            response_text = ai_response

        return response_text, should_transfer, should_end

    def _clean_response(self, response: str) -> str:
        """Clean and optimize the AI response for phone conversation"""
        # Remove any markdown or formatting
        response = response.replace("**", "").replace("*", "")

        # Remove any system-like responses or action indicators
        response = response.replace("Assistant:", "").replace("Alexis:", "")
        response = response.replace("ACTION:", "").replace("RESPONSE:", "")

        # Ensure it ends properly
        response = response.strip()
        if not response.endswith(('.', '?', '!')):
            response += "."

        # Limit length for phone conversation
        sentences = response.split('. ')
        if len(sentences) > 2:
            response = '. '.join(sentences[:2]) + '.'

        return response

    def generate_greeting(self) -> str:
        """Generate the standard NETOVO greeting"""
        return self.conversation_manager.get_greeting()

    def generate_escalation_response(self, reason: str = "technical") -> str:
        """Generate appropriate escalation response"""
        if reason == "urgent":
            return "Understood. This requires immediate attention. I am transferring you to a live technician right now. Please hold."
        elif reason == "complex":
            return "It seems we've tried the initial steps and the issue persists. This will require a technician. I'm transferring you to our live support team now. I've logged the steps we took so you don't have to repeat them."
        elif reason == "hardware":
            return "For an issue like that, a technician will need to take a look. Would you like me to transfer you to our live support team now?"
        else:
            return "I'm going to connect you with our support team so they can better assist. Would you like me to transfer you now?"

    def generate_callback_response(self, phone_number: str = None, name: str = None) -> str:
        """Generate callback arrangement response"""
        phone_text = f"at {phone_number}" if phone_number else "at this number"
        name_text = f", {name}" if name else ""

        return f"Okay, understood{name_text}. A ticket will be created for a technician to call you back {phone_text} within the next business day. Is there anything else I can help you with today?"

    def generate_closing_response(self) -> str:
        """Generate professional closing response"""
        return "Thank you for calling Netovo Support. Have a great day!"

    def reset_conversation(self):
        """Reset conversation history for new call"""
        self.conversation_history = []

    def get_conversation_summary(self) -> str:
        """Get summary of current conversation for logging"""
        if not self.conversation_history:
            return "No conversation history"

        user_messages = [msg["content"] for msg in self.conversation_history if msg["role"] == "user"]
        return f"User issues discussed: {' | '.join(user_messages[-3:])}"
