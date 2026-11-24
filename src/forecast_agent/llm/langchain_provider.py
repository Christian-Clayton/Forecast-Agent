"""LangChain / LangGraph provider wrapper.

Implements the existing ``LLMProvider`` protocol using LangChain's
``ChatOpenAI`` so the existing forecasting pipeline can be driven by the
free-tier OpenRouter models (or any compatible routing model) without any
change to the stage logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import SecretStr

from forecast_agent.llm.provider import ChatMessage, Completion
from forecast_agent.llm.openai_compat_provider import ProviderError

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI  # pragma: no cover - typing only

_T = TypeVar("_T")


class LangChainProvider:
    """OpenAI-compatible provider implemented on top of LangChain's
    ``ChatOpenAI``.  It presents the same ``complete()`` async interface that
    ``ForecastAgent`` expects, so the stage logic and graph code can be
    replaced wholesale without touching downstream consumers.

    The wrapper deliberately preserves the existing retry / backoff / circuit
    behaviour so the free-model pipeline continues to tolerate 429s and
    transient 5xx errors exactly the way the httpx-based provider did.
    """

    def __init__(
        self,
        *,
        api_key: str | SecretStr,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_seconds: float = 180.0,
        max_retries: int = 3,
        initial_backoff_seconds: float = 0.25,
        circuit_failure_threshold: int = 5,
        circuit_reset_seconds: float = 30.0,
        request_delay_seconds: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not key.strip():
            raise ValueError("api_key must not be empty")

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff_seconds
        self._request_delay = request_delay_seconds
        self._circuit_threshold = circuit_failure_threshold
        self._circuit_reset = circuit_reset_seconds
        self._sleep = sleep
        self._clock = clock
        self._consecutive_failures = 0
        self._circuit_opened_at: float | None = None

        # Instantiate the LangChain chat model.  ``base_url`` points at
        # OpenRouter so ``ChatOpenAI`` talks to it exactly like any OpenAI
        # endpoint, and the ``api_key`` is forwarded as-is.
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - the package is a hard dep
            raise RuntimeError(
                "langchain_openai is required. Install with: pip install langchain_openai"
            ) from exc

        self._chat: ChatOpenAI = ChatOpenAI(
            openai_api_key=key,
            openai_api_base=base_url,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=None,  # default httpx client; we retry at this layer
            default_headers={"HTTP-Referer": "forecast-agent"},
        )

    # -- public API -----------------------------------------------------
    @classmethod
    def from_settings(cls, settings) -> "LangChainProvider":  # type: ignore[override]
        # Use the first model as the primary, but keep the full list for rotation
        model = settings.openai_compat_model or settings.openai_compat_model_list[0] if settings.openai_compat_model_list else "openrouter/free"
        return cls(
            api_key=settings.openai_compat_token,
            base_url=settings.openai_compat_base_url,
            model=model,
            temperature=0.0,
            max_tokens=512,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            initial_backoff_seconds=settings.llm_initial_backoff_seconds,
            circuit_failure_threshold=settings.llm_circuit_failure_threshold,
            circuit_reset_seconds=settings.llm_circuit_reset_seconds,
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
        if model and model != self._model:
            # Respawn with the requested model if it differs from the one we
            # were built with (used by the free router which cycles models).
            self._chat = self._build_chat_model(model, temperature)
        self._ensure_circuit_closed()

        for attempt in range(self._max_retries + 1):
            try:
                # Import all required message types
                from langchain_core.messages import (
                    SystemMessage,
                    HumanMessage,
                    AIMessage,
                )

                # Convert custom ChatMessage to LangChain message types
                # SystemMessage/LangChainMessage don't have a 'role' attribute
                langchain_messages = []
                for msg in messages:
                    if hasattr(msg, "role"):
                        role = msg.role
                    else:
                        # Already a LangChain message type
                        role = None

                    if role == "system" or (role is None and isinstance(msg, SystemMessage)):
                        langchain_messages.append(SystemMessage(content=msg.content))
                    elif role == "user":
                        langchain_messages.append(HumanMessage(content=msg.content))
                    elif role == "assistant":
                        langchain_messages.append(AIMessage(content=msg.content))
                    else:
                        # Unknown message type - preserve as-is
                        langchain_messages.append(msg)

                # Select the model to use for this request
                # If model is specified, create a new ChatOpenAI instance with that model
                # This allows the FallbackProvider to rotate across different models
                if model:
                    from langchain_openai import ChatOpenAI

                    chat_model = ChatOpenAI(
                        openai_api_key=self._chat.openai_api_key,
                        openai_api_base=self._base_url,
                        model_name=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=self._timeout,
                        max_retries=self._max_retries,
                    )
                else:
                    chat_model = self._chat

                response = await chat_model.ainvoke(
                    langchain_messages,
                    config={
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "request_timeout": self._timeout,
                    },
                )
            except Exception as error:  # transient: transport / timeout / model unavailable
                # Don't retry on rate limit (429) - let the FallbackProvider switch models
                # Also don't record transient failure on 429 to avoid opening circuit breaker
                if self._is_rate_limit_error(error):
                    raise ProviderError(
                        f"OpenAI-compatible provider rate limited: {error}"
                    ) from error
                if attempt < self._max_retries:
                    await self._wait(attempt)
                    continue
                self._record_transient_failure()
                raise ProviderError(
                    f"OpenAI-compatible provider failed after retries: {error}"
                ) from error

            # LangChain 1.x returns an AIMessage; extract the text and usage.
            content = response.content if hasattr(response, "content") else None
            if not content:
                # Some models return reasoning text in a different field.
                content = getattr(response, "reasoning", None) or getattr(
                    response, "auxiliary_text", None
                )
            if not content:
                raise ProviderError("provider returned empty completion")

            # Extract usage from the response
            usage = self._extract_usage(response)
            # Ensure usage has all required keys
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
                "cost": usage.get("cost", 0.0) or 0.0,
            }
            completion = Completion(
                content=str(content),
                model=getattr(response, "model_name", self._model),
                request_id=getattr(response, "id", None),
                input_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"],
                cost_usd=usage["cost"],
            )
            self._consecutive_failures = 0
            self._circuit_opened_at = None
            # Buffer between requests to avoid rate limiting.
            if self._request_delay > 0:
                await self._sleep(self._request_delay)
            return completion

        raise ProviderError("OpenAI-compatible provider retries were exhausted")

    # -- helpers ---------------------------------------------------------
    def _build_chat_model(self, model: str, temperature: float) -> "ChatOpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "langchain_openai is required. Install with: pip install langchain_openai"
            ) from exc
        return ChatOpenAI(
            openai_api_key=self._chat.openai_api_key,
            openai_api_base=self._base_url,
            model_name=model,
            temperature=temperature,
            max_tokens=self._max_tokens,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    def _extract_usage(self, response) -> dict:
        """Normalise LangChain usage metadata into {prompt_tokens, completion_tokens, cost}."""
        usage = {}
        # LangChain 1.x AIMessage has usage_metadata
        metadata = getattr(response, "usage_metadata", None)
        if isinstance(metadata, dict):
            usage = {
                "prompt_tokens": metadata.get("input_tokens") or 0,
                "completion_tokens": metadata.get("output_tokens") or 0,
                "cost": metadata.get("total_cost") or metadata.get("cost") or 0.0,
            }
        # Fallback: check message kwargs for usage info
        if not usage:
            for attr in ("usage_metadata", "usage", "response_metadata", "llm_output"):
                val = getattr(response, attr, None)
                if isinstance(val, dict):
                    usage.update({k: v for k, v in val.items() if k in ("input_tokens", "output_tokens", "total_cost", "cost")})
        # LangChain message has response_metadata with usage
        if not usage:
            response_metadata = getattr(response, "response_metadata", None)
            if isinstance(response_metadata, dict):
                for key in ("prompt_token_count", "completion_token_count", "total_cost", "cost"):
                    if key in response_metadata:
                        usage_key = "prompt_tokens" if key == "prompt_token_count" else ("completion_tokens" if key == "completion_token_count" else key)
                        usage[usage_key] = response_metadata[key] or 0
        # Ensure all required keys exist
        usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "cost": usage.get("cost", 0.0) or 0.0,
        }
        return usage

    def _is_rate_limit_error(self, error) -> bool:
        """Check if the error is a rate limit error (429)."""
        error_str = str(error).lower()
        return "429" in error_str or "rate limit" in error_str or "too many requests" in error_str

    def _ensure_circuit_closed(self) -> None:
        if self._circuit_opened_at is None:
            return
        if self._clock() - self._circuit_opened_at < self._circuit_reset:
            raise ProviderError("OpenAI-compatible provider circuit breaker is open")
        self._circuit_opened_at = None
        self._consecutive_failures = 0

    def _record_transient_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._circuit_threshold:
            self._circuit_opened_at = self._clock()

    async def _wait(self, attempt: int) -> None:
        delay = min(self._initial_backoff * (2**attempt), 2.0)
        if delay:
            await self._sleep(delay)


class ModelRotator:
    """Manages multiple OpenAI-compatible models with round-robin selection
    and fallback on failure. Avoids rate limiting by spreading requests
    across multiple free models.

    Example:
        rotator = ModelRotator(["model1:free", "model2:free", "model3:free"])
        provider = rotator.get_provider(token, base_url="https://openrouter.ai/api/v1")
        await provider.complete(messages, model=rotator.get_model(), ...)
    """

    def __init__(
        self,
        model_list: list[str],
        base_url: str = "https://openrouter.ai/api/v1",
        request_delay_seconds: float = 1.0,
    ) -> None:
        self.model_list = model_list
        self.base_url = base_url.rstrip("/")
        self.request_delay = request_delay_seconds
        self._index = 0
        self._usage_counts: dict[str, int] = {m: 0 for m in model_list}
        self._failures: dict[str, int] = {m: 0 for m in model_list}

    def _select_next(self) -> str:
        """Round-robin selection with failure avoiding."""
        if len(self.model_list) == 1:
            return self.model_list[0]

        # Sort models by failure count (ascending), then by usage (ascending)
        sorted_models = sorted(
            self.model_list,
            key=lambda m: (self._failures.get(m, 0), self._usage_counts.get(m, 0))
        )
        # Pick from the least-failed, least-used model
        best = sorted_models[0]
        # Rotate index to the best model
        self._index = self.model_list.index(best) if best in self.model_list else 0
        self._usage_counts[best] = self._usage_counts.get(best, 0) + 1
        return best

    def get_model(self) -> str:
        """Get the next model to use (round-robin)."""
        model = self.model_list[self._index % len(self.model_list)]
        self._index += 1
        self._usage_counts[model] = self._usage_counts.get(model, 0) + 1
        return model

    def increment_failures(self, model: str) -> None:
        """Record a failure for a model."""
        self._failures[model] = self._failures.get(model, 0) + 1

    def reset_failures(self, model: str) -> None:
        """Reset failure count for a model."""
        self._failures[model] = 0

    def reset(self) -> None:
        """Reset all counters (e.g., when switching to a fresh set of models)."""
        self._index = 0
        self._usage_counts = {m: 0 for m in self.model_list}
        self._failures = {m: 0 for m in self.model_list}

    def get_status(self) -> dict:
        """Get current status for debugging."""
        return {
            "model_list": self.model_list,
            "current_index": self._index % len(self.model_list),
            "usage_counts": self._usage_counts,
            "failure_counts": self._failures,
        }


class FallbackProvider:
    """Wraps a LangChainProvider and falls back to the next model on failure.

    This implements the fallback mechanism: if a request fails (401, 429, 5xx,
    timeout, etc.), it automatically tries the next model in the list.

    Usage:
        rotator = ModelRotator(["model1:free", "model2:free", "model3:free"])
        base_provider = LangChainProvider(api_key=token, model=rotator.get_model())
        provider = FallbackProvider(base_provider, rotator)
        await provider.complete(messages, model=rotator.get_model(), ...)
    """

    def __init__(
        self,
        primary_provider,
        rotator: "ModelRotator",
        max_fallback_iterations: int = 10,
    ) -> None:
        self._primary = primary_provider
        self._rotator = rotator
        self._max_fallback = max_fallback_iterations
        self._current_index = 0

    async def complete(
        self,
        messages,
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Completion:
        """Complete a request, trying multiple models if needed.

        Falls back to the next model in the list if the current one fails.
        """
        # Update model selection: use requested model if specified, otherwise rotate
        if model:
            current_model = model
            self._rotator.increment_failures(model)
        else:
            current_model = self._rotator.get_model()

        # Convert custom ChatMessage to LangChain message types
        langchain_messages = []
        for msg in messages:
            if msg.role == "system":
                from langchain_core.messages import SystemMessage
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                from langchain_core.messages import HumanMessage
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                from langchain_core.messages import AIMessage
                langchain_messages.append(AIMessage(content=msg.content))

        # Retry on 429 (rate limit) with backoff before trying next model
        last_error = None
        for attempt in range(self._max_fallback + 1):
            try:
                # Call the primary provider's complete() method, which handles rate limits
                response = await self._primary.complete(
                    langchain_messages,
                    model=current_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # Success - reset failure counts for this model
                self._rotator.reset_failures(current_model)
                self._rotator.increment_failures(current_model)

                # Buffer between requests to avoid rate limiting
                if self._primary._request_delay > 0:
                    await self._primary._sleep(self._primary._request_delay)

                return response

            except Exception as error:
                last_error = error
                self._rotator.increment_failures(current_model)
                # Check if the error is a rate limit (429), switch to next model
                if self._is_rate_limit_error(error) and attempt < self._max_fallback:
                    # Switch to next model on rate limit
                    next_model = self._rotator.get_model()
                    if next_model != current_model:
                        current_model = next_model
                        continue
                # Try the next model (rotate)
                if attempt < self._max_fallback:
                    next_model = self._rotator.get_model()
                    if next_model != current_model:
                        current_model = next_model
                        continue
                # All fallback attempts exhausted
                raise ProviderError(
                    f"OpenAI-compatible provider failed after {self._max_fallback + 1} attempts: {error}"
                ) from error

    def _is_rate_limit_error(self, error) -> bool:
        """Check if the error is a rate limit error (429)."""
        error_str = str(error).lower()
        return "429" in error_str or "rate limit" in error_str or "too many requests" in error_str

    def _extract_usage(self, response) -> dict:
        """Normalise LangChain usage metadata into {prompt_tokens, completion_tokens, cost}."""
        usage = {}
        try:
            metadata = getattr(response, "usage_metadata", None)
            if isinstance(metadata, dict):
                usage = {
                    "prompt_tokens": metadata.get("input_tokens", metadata.get("prompt_tokens", 0)),
                    "completion_tokens": metadata.get("output_tokens", metadata.get("completion_tokens", 0)),
                    "cost": metadata.get("total_cost", metadata.get("cost")),
                }
        except Exception:
            usage = {}
        if not usage:
            for attr in ("usage_metadata", "usage", "response_metadata", "llm_output"):
                val = getattr(response, attr, None)
                if isinstance(val, dict):
                    usage.update({k: v for k, v in val.items() if k in ("input_tokens", "output_tokens", "total_cost", "cost")})
        return usage

    async def close(self):
        """Close the primary provider if it has a close method."""
        if hasattr(self._primary, "aclose"):
            await self._primary.aclose()
