import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr

from forecast_agent.config import Settings
from forecast_agent.llm.provider import ChatMessage, Completion


class ProviderError(RuntimeError):
    """Raised when an OpenAI-compatible provider cannot complete a request."""


class OpenAICompatProvider:
    def __init__(
        self,
        api_key: str | SecretStr | None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        initial_backoff_seconds: float = 0.25,
        circuit_failure_threshold: int = 5,
        circuit_reset_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if api_key is None:
            raise ValueError("OpenAI-compatible provider API key is required")
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not key.strip():
            raise ValueError("OpenAI-compatible provider API key must not be empty")
        parsed_url = urlparse(base_url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError("provider base URL must be a credential-free HTTPS URL")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between zero and five")
        if (
            not math.isfinite(initial_backoff_seconds)
            or initial_backoff_seconds < 0
        ):
            raise ValueError("initial_backoff_seconds must be finite and nonnegative")
        if circuit_failure_threshold < 1:
            raise ValueError("circuit_failure_threshold must be positive")
        if not math.isfinite(circuit_reset_seconds) or circuit_reset_seconds <= 0:
            raise ValueError("circuit_reset_seconds must be finite and positive")
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._initial_backoff_seconds = initial_backoff_seconds
        self._circuit_failure_threshold = circuit_failure_threshold
        self._circuit_reset_seconds = circuit_reset_seconds
        self._sleep = sleep
        self._clock = clock
        self._consecutive_failures = 0
        self._circuit_opened_at: float | None = None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> "OpenAICompatProvider":
        return cls(
            api_key=settings.openai_compat_api_key,
            base_url=settings.openai_compat_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            initial_backoff_seconds=settings.llm_initial_backoff_seconds,
            circuit_failure_threshold=settings.llm_circuit_failure_threshold,
            circuit_reset_seconds=settings.llm_circuit_reset_seconds,
            client=client,
            sleep=sleep,
            clock=clock,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Completion:
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        self._ensure_circuit_closed()

        payload = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
            except httpx.TransportError as error:
                if attempt < self._max_retries:
                    await self._wait(attempt)
                    continue
                self._record_transient_failure()
                raise ProviderError(
                    "OpenAI-compatible provider request failed after retries"
                ) from error

            if response.status_code in (408, 429) or response.status_code >= 500:
                if attempt < self._max_retries:
                    await self._wait(attempt)
                    continue
                self._record_transient_failure()
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                raise ProviderError(
                    f"OpenAI-compatible provider returned HTTP {response.status_code}"
                ) from error
            completion = self._parse_completion(response.json(), model)
            self._consecutive_failures = 0
            self._circuit_opened_at = None
            return completion

        raise ProviderError("OpenAI-compatible provider retries were exhausted")

    async def _wait(self, attempt: int) -> None:
        delay = min(self._initial_backoff_seconds * (2**attempt), 2.0)
        if delay:
            await self._sleep(delay)

    def _ensure_circuit_closed(self) -> None:
        if self._circuit_opened_at is None:
            return
        if self._clock() - self._circuit_opened_at < self._circuit_reset_seconds:
            raise ProviderError("OpenAI-compatible provider circuit breaker is open")
        self._circuit_opened_at = None
        self._consecutive_failures = 0

    def _record_transient_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._circuit_failure_threshold:
            self._circuit_opened_at = self._clock()

    @staticmethod
    def _parse_completion(payload: Any, requested_model: str) -> Completion:
        if not isinstance(payload, dict):
            raise ProviderError("provider response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("provider response contains no completion choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ProviderError("provider response has no text completion")

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return Completion(
            content=content,
            model=str(payload.get("model") or requested_model),
            request_id=str(payload["id"]) if payload.get("id") is not None else None,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_usd=usage.get("cost"),
        )
