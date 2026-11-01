"""
Groq LLM Client - Drop-in replacement for Gemini

Uses Groq API for fast, reliable inference.
Free tier: 30 RPM, 14,400 requests/day (for llama-3.1-8b-instant)

Setup:
1. Get free API key: https://console.groq.com
2. Set env: export GROQ_API_KEY=gsk_xxxxx
3. pip install groq
"""

import json
import logging
import os
import time
from typing import Type, TypeVar
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

_client_instance = None


class GeminiAPIError(Exception):
    """Custom exception for API errors (named for backward compatibility)"""
    def __init__(self, message: str, status_code: int = None, retryable: bool = False):
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(self.message)


class GeminiClient:
    """
    Groq API client with Gemini-compatible interface.
    
    Models (Free Tier Limits):
    - llama-3.1-8b-instant: 30 RPM, 14,400 RPD (RECOMMENDED - highest quota)
    - llama-3.3-70b-versatile: 30 RPM, 1,000 RPD (better quality, lower quota)
    
    Usage:
        client = GeminiClient()
        result = client.generate_structured(
            prompt="Parse this resume...",
            output_schema=ParsedResume
        )
    """
    
    # Model configurations with rate limits
    MODELS = {
        "llama-3.1-8b-instant": {
            "rpm": 30,
            "rpd": 14400,
            "tpm": 6000,
            "description": "Fast, high quota - best for testing"
        },
        "llama-3.3-70b-versatile": {
            "rpm": 30,
            "rpd": 1000,
            "tpm": 12000,
            "description": "Best quality, lower quota"
        },
    }
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key. Reads from GROQ_API_KEY or GEMINI_API_KEY env var.
            model: Model ID. Default: llama-3.1-8b-instant (highest free quota)
        """
        # Support both GROQ_API_KEY and GEMINI_API_KEY for compatibility
        self.api_key = (
            api_key or 
            os.getenv("GROQ_API_KEY") or 
            os.getenv("GEMINI_API_KEY")
        )
        
        if not self.api_key:
            raise ValueError(
                "API key not found.\n"
                "Get a FREE key from: https://console.groq.com\n"
                "Then set: export GROQ_API_KEY=gsk_your_key_here"
            )
        
        # Default to highest quota model
        self.model = model or os.getenv("GROQ_MODEL") or os.getenv("GEMINI_MODEL") or "llama-3.1-8b-instant"
        
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Groq client"""
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
            logger.info(f"✓ Groq client initialized with model: {self.model}")
        except ImportError:
            raise ImportError(
                "groq package not installed.\n"
                "Run: pip install groq"
            )
        except Exception as e:
            raise GeminiAPIError(f"Failed to initialize Groq client: {e}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        system_instruction: str = None,
        max_retries: int = 5,
        initial_delay: float = 2.0,
        **kwargs
    ) -> str:
        """
        Generate text completion with retry logic.
        
        Args:
            prompt: User prompt
            temperature: 0.0-2.0, lower = more deterministic
            max_tokens: Maximum output tokens
            system_instruction: Optional system prompt
            max_retries: Number of retries for rate limits
            initial_delay: Initial backoff delay in seconds
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        messages.append({"role": "user", "content": prompt})
        
        last_exception = None
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check for rate limit errors
                is_rate_limit = any(kw in error_str for kw in [
                    "429", "rate", "limit", "quota", "too many", "exceeded"
                ])
                
                # Check for auth errors (don't retry)
                is_auth_error = any(kw in error_str for kw in [
                    "401", "403", "invalid", "unauthorized", "api_key", "authentication"
                ])
                
                if is_auth_error:
                    raise GeminiAPIError(
                        f"Authentication error: {e}\n"
                        f"Check your GROQ_API_KEY at https://console.groq.com/keys",
                        status_code=401,
                        retryable=False
                    )
                
                if is_rate_limit and attempt < max_retries:
                    logger.warning(
                        f"⏳ Rate limited (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60)  # Max 60s backoff
                    continue
                
                # Out of retries or non-retryable error
                if is_rate_limit:
                    raise GeminiAPIError(
                        f"Rate limit exceeded after {max_retries + 1} attempts.\n"
                        f"Free tier limits for {self.model}:\n"
                        f"  - 30 requests/minute\n"
                        f"  - Check usage: https://console.groq.com/settings/usage",
                        status_code=429,
                        retryable=True
                    )
                else:
                    raise GeminiAPIError(f"Groq API error: {e}", retryable=False)
        
        raise GeminiAPIError(
            f"Failed after {max_retries + 1} attempts: {last_exception}",
            retryable=False
        )
    
    def generate_structured(
        self, 
        prompt: str, 
        output_schema: Type[T],
        temperature: float = 0.1,
        system_instruction: str = None,
        max_retries: int = 3
    ) -> T:
        """
        Generate structured JSON output validated against Pydantic schema.
        
        Args:
            prompt: User prompt
            output_schema: Pydantic model class for validation
            temperature: 0.0-2.0, lower = more deterministic
            system_instruction: Optional system prompt
            max_retries: Retries for JSON parsing failures
            
        Returns:
            Validated Pydantic model instance
        """
        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
        
        full_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. No markdown code blocks (no ```)
3. No explanations before or after
4. Start directly with {{ and end with }}
5. Follow this exact schema:

{schema_json}

Your JSON response:"""
        
        json_system = (
            "You are a JSON-only assistant. You MUST output ONLY valid JSON. "
            "NEVER include markdown code blocks (```). "
            "NEVER include explanations or text outside the JSON. "
            "Start your response with { and end with }. "
            "Ensure all strings are properly escaped."
        )
        
        if system_instruction:
            full_system = f"{system_instruction}\n\n{json_system}"
        else:
            full_system = json_system
        
        last_error = None
        temp = temperature
        
        for attempt in range(max_retries + 1):
            try:
                response_text = self.generate(
                    prompt=full_prompt,
                    temperature=temp,
                    system_instruction=full_system
                )
                
                # Clean and parse response
                cleaned = self._clean_json_response(response_text)
                return output_schema.model_validate_json(cleaned)
                
            except GeminiAPIError:
                raise  # Re-raise API errors
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1}: {last_error}")
                logger.debug(f"Raw response: {response_text[:500]}...")
                temp = min(temp + 0.1, 0.5)  # Increase temperature slightly
            except Exception as e:
                last_error = f"Validation error: {e}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1}: {last_error}")
                temp = min(temp + 0.1, 0.5)
        
        raise GeminiAPIError(
            f"Failed to generate valid JSON after {max_retries + 1} attempts: {last_error}",
            retryable=False
        )
    
    def _clean_json_response(self, text: str) -> str:
        """Clean LLM response to extract valid JSON."""
        if not text:
            return "{}"
        
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Find JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        
        return text
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4


def get_gemini_client(api_key: str = None, model: str = None) -> GeminiClient:
    """
    Get or create singleton client instance.
    
    Args:
        api_key: Optional API key
        model: Optional model override
        
    Returns:
        GeminiClient instance
    """
    global _client_instance
    
    if _client_instance is None:
        _client_instance = GeminiClient(api_key=api_key, model=model)
    
    return _client_instance


def reset_client():
    """Reset singleton client (useful for testing)."""
    global _client_instance
    _client_instance = None
