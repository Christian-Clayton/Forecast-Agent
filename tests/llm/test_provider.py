import asyncio

import httpx
import pytest

from forecast_agent.llm.openai_compat_provider import (
    OpenAICompatProvider,
    ProviderError,
)
from forecast_agent.llm.provider import ChatMessage


def test_openai_compatible_provider_retries_transient_errors_and_parses_usage() -> None:
    attempts = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "id": "request-1",
                "model": "provider/model-a",
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "cost": 0.0002,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = OpenAICompatProvider(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        client=client,
        max_retries=1,
        initial_backoff_seconds=0,
    )
    completion = asyncio.run(
        provider.complete(
            [ChatMessage(role="user", content="Return JSON.")],
            model="provider/model-a",
            max_tokens=50,
        )
    )

    assert attempts == 2
    assert completion.content == '{"ok": true}'
    assert completion.input_tokens == 12
    assert completion.output_tokens == 4
    assert completion.cost_usd == pytest.approx(0.0002)
    assert completion.request_id == "request-1"
    asyncio.run(client.aclose())


def test_openai_compatible_provider_requires_a_configured_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatProvider(api_key=None)


def test_openai_compatible_provider_opens_circuit_after_transient_failures() -> None:
    attempts = 0
    clock = [0.0]

    def handle_request(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = OpenAICompatProvider(
        api_key="test-key",
        client=client,
        max_retries=0,
        circuit_failure_threshold=1,
        circuit_reset_seconds=30,
        clock=lambda: clock[0],
    )
    messages = [ChatMessage(role="user", content="Return JSON.")]

    async def verify() -> None:
        with pytest.raises(ProviderError, match="HTTP 503"):
            await provider.complete(messages, model="provider/model-a", max_tokens=50)
        with pytest.raises(ProviderError, match="circuit breaker is open"):
            await provider.complete(messages, model="provider/model-a", max_tokens=50)
        clock[0] = 31
        completion = await provider.complete(
            messages, model="provider/model-a", max_tokens=50
        )
        assert completion.content == '{"ok": true}'

    asyncio.run(verify())
    assert attempts == 2
    asyncio.run(client.aclose())
