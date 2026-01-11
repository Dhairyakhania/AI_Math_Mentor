"""
Google Gemini client wrapper for the Math Mentor application.
Provides a unified interface for interacting with Gemini models.
"""

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Optional, Union, Generator
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config
import json


class GeminiClient:
    """
    Wrapper for Google Gemini API with rate limiting and error handling.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.GOOGLE_API_KEY
        self.model_name = model_name or Config.GEMINI_MODEL
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 60.0 / Config.REQUESTS_PER_MINUTE
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from Gemini.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
        
        Returns:
            Generated text response
        """
        self._rate_limit()
        
        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"
        
        # Override generation config if needed
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        try:
            if generation_config:
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            else:
                response = self.model.generate_content(full_prompt)
            
            return response.text
        
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                # Rate limit hit, wait and retry
                time.sleep(60)
                raise
            raise
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Generate a JSON response from Gemini.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
        
        Returns:
            Parsed JSON response
        """
        # Add JSON instruction to prompt
        json_instruction = "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation, just the JSON object."
        
        full_system = (system_prompt or "") + json_instruction
        
        response = self.generate(prompt, system_prompt=full_system, temperature=0.3)
        
        # Clean response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response[:500]}") from e
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream a response from Gemini.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
        
        Yields:
            Text chunks as they're generated
        """
        self._rate_limit()
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"
        
        response = self.model.generate_content(full_prompt, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return self.model.count_tokens(text).total_tokens
    
    def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Multi-turn chat with Gemini.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            system_prompt: Optional system instructions
        
        Returns:
            Assistant's response
        """
        self._rate_limit()
        
        # Start chat
        chat = self.model.start_chat(history=[])
        
        # Add system prompt as first user message if provided
        if system_prompt:
            chat.send_message(f"System Instructions: {system_prompt}")
        
        # Send all previous messages
        for msg in messages[:-1]:
            if msg["role"] == "user":
                chat.send_message(msg["content"])
        
        # Get response for last message
        response = chat.send_message(messages[-1]["content"])
        
        return response.text


# Singleton instance
_gemini_client = None

def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client singleton"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client