"""
Resilient Ollama client for ORCA2:7b LLM
Connects to local Ollama on localhost:11434
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
from typing import Optional

logger = structlog.get_logger()

class OllamaClient:
    """Client for local Ollama ORCA2:7b"""

    def __init__(self, base_url="http://localhost:11434", model="orca2:7b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=30.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 100) -> str:
        """Generate response from Ollama"""

        # Build the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,
                "stop": ["Human:", "\n\n"]
            }
        }

        try:
            logger.info("Sending request to Ollama", prompt=prompt[:50], model=self.model)
            response = self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "").strip()

            logger.info("Received response from Ollama", response=generated_text[:50])
            return generated_text

        except Exception as e:
            logger.error("Ollama request failed", error=str(e))
            raise

    def health_check(self) -> bool:
        """Check if Ollama is running using /api/tags endpoint"""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False

    def close(self):
        """Close the HTTP client"""
        self.client.close()

# Default system prompt for voice bot
VOICE_BOT_SYSTEM_PROMPT = """You are Alexis, a professional customer support AI voice assistant for NETOVO.
Keep responses under concise and meaningful for telephony quality.
Be helpful, concise, and professional.
Never re-introduce yourself after the first greeting."""
