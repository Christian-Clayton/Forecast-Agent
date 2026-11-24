"""Multi-provider fallback: rotate across every model on every configured API.

The forecasting pipeline asks a single ``complete()`` from one LLMProvider.  When
that provider (or a specific model) is rate-limited (429), timeout-gone (503) or
otherwise unavailable, the pipeline dies.  ``MultiProvider`` fixes that by holding
a list of providers, each with its own model list and base URL, and advancing to
the next model--first within a provider's list, then to the next provider's list.

Request flow:  the pipeline passes an explicit ``model`` to ``complete()``; that
model is attempted first (if it belongs to one of our providers).  On any
failure (rate limit, 5xx, transport, 401) we immediately drop to the next model
and keep going until every configured model on every configured API has been
tried.  The first successful completion is returned; otherwise the last error is
re-raised.

Every attempt (provider, model, status, timing, tokens, cost, error) is recorded
on ``self.attempts`` so the pipeline can emit a detailed per-question log.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable
from pydantic import BaseModel

from forecast_agent.llm.provider import ChatMessage, Completion


class MultiProviderAttempt(BaseModel):
    """One completed/failed attempt at a model."""

    attempt: int
    provider: str
    model: str
    status: str  # "start" | "success" | "error"
    error: str | None = None
    response_preview: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    timing_ms: float = 0.0


class MultiProvider:
    """Rotates across multiple LLM providers and every model on each API."""

    def __init__(self, providers):
        """
        Args:
            providers: list of (name, provider, model_list), where ``provider`` is
                an object implementing ``complete(messages, model=..., max_tokens,
                temperature=...) -> Completion`` and ``model_list`` is the ordered
                list of model names to try on that provider (e.g. a Gemini free
                tier with six free models, or an OpenRouter free tier with six
                free models).
        """
        self.providers = list(providers)  # (name, provider, model_list)
        self._provider_idx = 0
        self._model_idx = {i: 0 for i in range(len(self.providers))}
        self.attempts = 0
        self.log: list[MultiProviderAttempt] = []

    # -- introspection ------------------------------------------------------
    @property
    def provider_names(self):
        return [p[0] for p in self.providers]

    @property
    def model_lists(self):
        return [p[2] for p in self.providers]

    @property
    def base_urls(self):
        # providers are assumed to carry the URL they talk to
        return [getattr(p[1], "_base_url", "") for p in self.providers]

    # -- helpers -------------------------------------------------------------
    def _candidates(self, requested_model: str) -> list[tuple[int, str]]:
        """Ordered (provider_idx, model) pairs to try.

        The explicitly requested model is attempted first (when it maps to one of
        our providers), then every model of every provider.
        """
        requested = (requested_model or "").strip()
        if not requested:
            requested = self.providers[0][2][0] if self.providers else ""
        ordered: list[tuple[int, str]] = []
        if requested:
            for i, (name, prov, mlist) in enumerate(self.providers):
                if requested in mlist:
                    ordered.append((i, requested))
                    break
        for i, (name, prov, mlist) in enumerate(self.providers):
            for m in mlist:
                if (i, m) not in candidates_set(ordered):
                    ordered.append((i, m))
        if not ordered:
            for i, (name, prov, mlist) in enumerate(self.providers):
                if mlist:
                    ordered.append((i, mlist[0]))
        return ordered

    def record_attempt(self, attempt: MultiProviderAttempt) -> None:
        self.log.append(attempt)

    def reset(self) -> None:
        self.attempts = 0
        self.log = []
        self._provider_idx = 0
        self._model_idx = {i: 0 for i in range(len(self.providers))}

    # -- primary API ---------------------------------------------------------
    async def complete(self, messages, *, model="", max_tokens, temperature=0.0):
        """Try every configured model on every configured API until one succeeds."""
        self.reset()
        requested = (model or "").strip()
        ordered = self._candidates(requested)
        last_error: Exception | None = None

        for provider_idx, model_name in ordered:
            provider_name, provider, _ = self.providers[provider_idx]
            self._provider_idx = provider_idx
            self.attempts += 1
            started = time.perf_counter()
            try:
                completion = await provider.complete(
                    messages, model=model_name, max_tokens=max_tokens, temperature=temperature
                )
                timing_ms = (time.perf_counter() - started) * 1000.0
                record = MultiProviderAttempt(
                    attempt=self.attempts,
                    provider=provider_name,
                    model=model_name,
                    status="success",
                    response_preview=completion.content[:500],
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    cost_usd=completion.cost_usd or 0.0,
                    timing_ms=timing_ms,
                )
                self.record_attempt(record)
                return completion
            except Exception as error:  # rate limit, 5xx, transport, 401...
                last_error = error
                timing_ms = (time.perf_counter() - started) * 1000.0
                record = MultiProviderAttempt(
                    attempt=self.attempts,
                    provider=provider_name,
                    model=model_name,
                    status="error",
                    error=str(error),
                    timing_ms=timing_ms,
                )
                self.record_attempt(record)
        raise last_error if last_error is not None else RuntimeError("no models configured")


def candidates_set(ordered):
    return {(i, m) for i, m in ordered}
