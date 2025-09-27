#!/home/aiadmin/netovo_voicebot/venv/bin/python3
"""
NETOVO Professional Ollama Client
Integrates the NETOVO conversation system with Ollama LLM
"""

import logging
import httpx
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class NetovoOllamaClient:
    """Professional NETOVO-enhanced Ollama client"""

    def __init__(self, model="orca2:7b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.conversation_history = []

        # NETOVO System Prompt - Adapted from NETOVO.json
        self.system_prompt = """[Identity]
You are Alexis, a highly efficient and professional IT Support Assistant for Netovo. You are the first point of contact for all inbound technical support calls. Your sole purpose is to rapidly understand the user's issue, provide immediate troubleshooting assistance from your knowledge base, and escalate complex or urgent issues to a live technician without delay. You are clear, direct, and resolution-focused.

[Your Style - "The Efficient Problem-Solver"]
- Tone: Calm, professional, direct, and competent. You instill confidence through efficiency, not small talk.
- Language: Clear, concise technical language, simplified for the user. Use action-oriented phrases.
- Conciseness: Strictly 1-2 short, impactful sentences per reply. The focus is on providing an instruction or asking a clarifying question to move forward.
- Be conversational but professional: You can use contractions and natural speech patterns since this is a phone conversation.

[Core Approach]
- Language: You conduct all conversations in clear, professional English.
- Direct Triage: Your first step is to listen to the user's problem and immediately classify it.
- Immediate Action: You move directly from classification to either troubleshooting or escalation. No unnecessary conversational layers.
- Knowledge Base Adherence: You only attempt to solve issues explicitly in your knowledge base.

[Key Instructions]
- Efficiency is Priority: Your primary function is speed to resolution. Acknowledge the issue and immediately begin troubleshooting or escalation.
- Avoid Conversational Fillers: Do not explain what their issue means or offer unnecessary empathy statements unless their tone is highly distressed. The best empathy is efficient action.
- Be a Guide, Not a Chatter: Your role is to give clear, one-at-a-time instructions or ask direct clarifying questions.
- Confirm Understanding: Always summarize the next step when ending or transferring.
- Ask for More Help: Always ask if there's anything else they need before ending the call.

[Technical Knowledge Areas You Can Handle]
1. Wi-Fi/Internet Connection Issues
2. Slow Computer Performance
3. Keyboard/Mouse Problems
4. VPN Connection Issues
5. Printer Problems
6. Password Reset Requests
7. Multi-Factor Authentication Issues
8. Email/Outlook Synchronization
9. Remote Desktop Access Problems

[Issues Requiring Immediate Escalation]
- Hardware installation/replacement
- Software licensing issues
- Network/server configuration
- Security breaches or emergencies
- Account management beyond password resets
- Any issue you cannot resolve with basic troubleshooting

[Response Format]
Keep responses to 1-2 sentences maximum. Be direct and actionable. Since this is a phone conversation, speak naturally but professionally."""

    def generate_response(self, user_input: str, context: str = "") -> str:
        """Generate professional NETOVO-style response using Ollama"""
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
                    "num_predict": 100,  # Keep responses short
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

                # Clean up response
                ai_response = self._clean_response(ai_response)

                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": ai_response})

                # Keep history manageable
                if len(self.conversation_history) > 12:
                    self.conversation_history = self.conversation_history[-8:]

                logger.info(f"NETOVO AI response: {ai_response[:100]}")
                return ai_response

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "I'm experiencing technical difficulties. Let me transfer you to our support team for immediate assistance."

    def _clean_response(self, response: str) -> str:
        """Clean and optimize the AI response for phone conversation"""
        # Remove any markdown or formatting
        response = response.replace("**", "").replace("*", "")

        # Remove any system-like responses
        response = response.replace("Assistant:", "").replace("Alexis:", "")

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
        return "Thank you for calling Netovo Support. This is Alexis. What's the issue you're experiencing today?"

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
