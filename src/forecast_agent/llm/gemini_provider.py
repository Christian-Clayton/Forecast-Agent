"""Google Gemini provider using the native generateContent endpoint.

The OpenAI-compatible endpoint (/v1beta/openai/chat/completions) is unreliable
for free-tier models, so this provider uses the native Gemini endpoint directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from pydantic import SecretStr

from forecast_agent.llm.provider import ChatMessage, Completion
from forecast_agent.llm.openai_compat_provider import ProviderError

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)


class GeminiProvider:
    """Gemini provider using the native generateContent endpoint.

    Implements the same complete() interface as LangChainProvider so it can
    be used as a drop-in replacement in the forecasting pipeline.
    """

    def __init__(
        self,
        *,
        api_key: str | SecretStr,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_seconds: float = 180.0,
        max_retries: int = 3,
        initial_backoff_seconds: float = 0.25,
        request_delay_seconds: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not key.strip():
            raise ValueError("api_key must not be empty")

        self._api_key = key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff_seconds
        self._request_delay = request_delay_seconds
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._sleep = asyncio.sleep  # For FallbackProvider compatibility

    @classmethod
    def from_settings(cls, settings) -> "GeminiProvider":
        model = settings.openai_compat_model or (
            settings.openai_compat_model_list[0] if settings.openai_compat_model_list else "gemini-3.5-flash-lite"
        )
        return cls(
            api_key=settings.openai_compat_token,
            model=model,
            temperature=0.0,
            max_tokens=512,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            initial_backoff_seconds=settings.llm_initial_backoff_seconds,
            request_delay_seconds=1.0,
        )

    async def complete(
        self,
        messages,
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Completion:
        """Complete a request using the native Gemini endpoint."""
        import requests

        # Use the specified model or fall back to the configured one
        use_model = model or self._model

        # Convert messages to Gemini format
        # Gemini uses "contents" with "parts" instead of "messages"
        # Handle both ChatMessage (role attribute) and LangChain messages (type attribute)
        contents = []
        system_instruction = None

        for msg in messages:
            # Check for ChatMessage format (role attribute)
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", str(msg))

            # If no role, check for LangChain message format (type attribute)
            if role is None:
                msg_type = getattr(msg, "type", None)
                if msg_type == "system":
                    role = "system"
                elif msg_type == "human":
                    role = "user"
                elif msg_type == "ai":
                    role = "assistant"
                else:
                    role = "user"  # Default to user

            if role == "system":
                # System messages go into systemInstruction
                system_instruction = {"parts": [{"text": content}]}
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                # Unknown role - treat as user
                contents.append({"role": "user", "parts": [{"text": content}]})

        # Build the request payload
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = f"{self._base_url}/models/{use_model}:generateContent"
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                # Use anyio.to_thread.run_sync for compatibility with anyio task groups
                import anyio
                response = await anyio.to_thread.run_sync(
                    lambda: requests.post(url, headers=headers, json=payload, timeout=self._timeout)
                )

                if response.status_code == 200:
                    data = response.json()
                    # Extract the text from the response
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts)
                    else:
                        text = ""

                    # Extract usage metadata
                    usage_metadata = data.get("usageMetadata", {})

                    # Buffer between requests
                    if self._request_delay > 0:
                        await anyio.sleep(self._request_delay)

                    return Completion(
                        content=text,
                        model=use_model,
                        input_tokens=usage_metadata.get("promptTokenCount", 0),
                        output_tokens=usage_metadata.get("candidatesTokenCount", 0),
                        cost_usd=0.0,  # Free tier
                    )
                elif response.status_code == 429:
                    # Rate limited - don't retry, let FallbackProvider switch models
                    raise ProviderError(f"Gemini rate limited: {response.text[:200]}")
                elif response.status_code == 503:
                    # Service unavailable - retry with backoff
                    last_error = ProviderError(f"Gemini service unavailable: {response.text[:200]}")
                    if attempt < self._max_retries:
                        await anyio.sleep(self._initial_backoff * (2 ** attempt))
                        continue
                    raise last_error
                else:
                    raise ProviderError(f"Gemini API error {response.status_code}: {response.text[:200]}")

            except ProviderError:
                raise
            except Exception as error:
                last_error = error
                if attempt < self._max_retries:
                    await anyio.sleep(self._initial_backoff * (2 ** attempt))
                    continue
                raise ProviderError(f"Gemini provider failed after retries: {error}") from error

        raise ProviderError(f"Gemini provider failed: {last_error}")

    def _is_rate_limit_error(self, error) -> bool:
        """Check if the error is a rate limit error."""
        error_str = str(error).lower()
        return "429" in error_str or "rate limit" in error_str or "too many requests" in error_str
