"""Self-growing agent integration.

This module provides the entry point for the self-growing forecasting agent.
The agent can dynamically create and register tools based on incoming queries,
so the user doesn't need to manually add new capabilities.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, UTC
import time
from typing import Any, Optional
from uuid import uuid4

from forecast_agent.agent.pipeline import ForecastAgent as PipelineForecastAgent
from forecast_agent.config import Settings
from forecast_agent.llm.multi_provider import MultiProvider
from forecast_agent.agent.question_logger import QuestionLogger
from forecast_agent.llm.langchain_provider import LangChainProvider, ModelRotator, FallbackProvider
from forecast_agent.llm.gemini_provider import GeminiProvider
from forecast_agent.agent.stages.schemas import FinalizeOutput
from forecast_agent.self_growth import DynamicToolGenerator, ToolRegistry, SelfGrowingAgent


class DynamicForecastAgent:
    """The self-growing forecasting agent.

    This agent extends the standard forecasting agent with dynamic tool
    creation capabilities. When a query requires a capability that
    doesn't exist, the agent automatically creates and registers new tools.

    Usage:
        from forecast_agent.self_growth import DynamicForecastAgent, ToolRegistry

        settings = Settings()
        registry = ToolRegistry()
        agent = DynamicForecastAgent(settings, registry)

        result = await agent.run("Will AAPL be higher in 2 weeks?")
        print(result.answer.probability)
    """

    def __init__(
        self,
        settings: Settings,
        registry: Optional[ToolRegistry] = None,
        provider: Optional[LangChainProvider] = None,
    ) -> None:
        self.settings = settings
        self.registry = registry or ToolRegistry()
        # Build a MultiProvider that rotates across ALL configured models on
        # every configured API. On a rate limit (429) / 5xx / transport error the
        # pipeline automatically tries the next model, then the next provider.
        self.provider = self._create_multi_provider()
        self._graph = None
        self._agent = None
        self._self_growing_agent = None

    def _create_multi_provider(self) -> "MultiProvider":
        """Build a MultiProvider covering every configured model on every
        configured API, so a rate limit on one model/API falls through to the
        next.

        Providers are (name, provider, model_list).  For the Gemini free tier we
        instantiate one GeminiProvider per model (each accepts ``model=`` in its
        ``complete()`` so MultiProvider can dial a specific model in on every
        attempt).
        """
        from forecast_agent.llm.gemini_provider import GeminiProvider
        from forecast_agent.llm.langchain_provider import LangChainProvider

        token = self.settings.openai_compat_token
        if token is None or not str(token.get_secret_value()).strip():
            raise ValueError("OPENAI_COMPAT_TOKEN must be configured before use")

        api_key = token.get_secret_value()
        base_url = self.settings.openai_compat_base_url
        model_list = list(self.settings.openai_compat_model_list) or [
            self.settings.openai_compat_model or "gemini-3.5-flash-lite"
        ]

        providers: list[tuple[str, LLMProvider, list[str]]] = []

        # Gemini free tier: native endpoint, one provider per model so we can
        # rotate across all of them when one hits a 429.
        if "generativelanguage.googleapis.com" in base_url:
            for model in model_list:
                providers.append(
                    (
                        f"gemini-{model.split('/')[-1]}",
                        GeminiProvider(
                            api_key=api_key,
                            model=model,
                        ),
                        [model],
                    )
                )
        else:
            # OpenRouter / generic OpenAI-compatible tier: one provider, all models.
            providers.append(
                (
                    "openai-compat",
                    LangChainProvider(
                        api_key=api_key,
                        base_url=base_url,
                        model=model_list[0],
                    ),
                    model_list,
                )
            )
        return MultiProvider(providers)

    def _get_or_create_graph(self):
        """Get or create the LangGraph."""
        if self._graph is None:
            self._graph = build_graph(self.provider, self.settings)
        return self._graph

    def _get_or_create_agent(self, question_logger=None):
        """Get or create the pipeline agent."""
        if self._agent is None:
            self._agent = PipelineForecastAgent(self.provider, self.settings, question_logger=question_logger)
        return self._agent

    def _get_self_growing_agent(self):
        """Get or create the self-growing agent."""
        if self._self_growing_agent is None:
            self._self_growing_agent = SelfGrowingAgent(
                registry=self.registry,
                generator=DynamicToolGenerator(registry=self.registry),
            )
        return self._self_growing_agent

    async def run(self, raw_question: str, **kwargs: Any) -> "ForecastRun":
        """Run a query with dynamic tool creation.

        The agent analyzes the question, determines what tools are needed,
        creates any missing tools, and registers them before running the
        pipeline.

        Args:
            raw_question: The question to answer
            **kwargs: Additional arguments (e.g., as_of)

        Returns:
            The forecast result
        """
        # Step 1: Analyze the query and determine needed tools
        self_growing_agent = self._get_self_growing_agent()
        needed_tools = self_growing_agent._analyze_query(raw_question)

        # One detailed log per question (tool calls, provider responses, retries)
        question_id = uuid4().hex
        question_logger = QuestionLogger(
            question_id=question_id,
            question_text=raw_question,
        )
        self._question_logger = question_logger  # keep for use in _enrich_question

        print(f"  [Self-Growth] Analyzing query: {raw_question}")
        print(f"  [Self-Growth] Needed tools: {[t['name'] for t in needed_tools]}")

        if needed_tools:
            print(f"  [Self-Growth] Registering {len(needed_tools)} new tool(s)...")
            for tool_def in needed_tools:
                tool_name = tool_def["name"]
                tool_desc = tool_def["description"]
                print(f"      + Registered: {tool_name} - {tool_desc}")

                # Register with the appropriate MCP server
                server = self_growing_agent._get_or_create_server(
                    tool_def["module"]
                )
                self_growing_agent.registrar.register_tool_with_server(
                    tool_def, server
                )

                # Register with the central registry
                self.registry.register_tool(
                    name=tool_name,
                    description=tool_desc,
                    parameters=tool_def["parameters"],
                    module=tool_def["module"],
                    identifier=tool_def["identifier"],
                )

        # Step 2: Run the pipeline using ForecastAgent
        print(f"  [Pipeline] Running 8-stage forecasting pipeline...")
        pipeline_start = time.monotonic()
        forecast_agent = self._get_or_create_agent(question_logger=question_logger)
        result = await forecast_agent.run(raw_question, **kwargs)
        question_latency = time.monotonic() - pipeline_start

        # Step 3: If refused, enrich the question with context and retry
        max_retries = 2
        current_question = raw_question
        for retry in range(max_retries):
            if result.answer is not None or result.refusal_reason is None:
                break  # Got an answer or no refusal reason, stop retrying

            print(f"  [Self-Growth] Refused: {result.refusal_reason}")
            question_logger.model_call(
                provider="self-growth",
                model="enrichment-llm",
                status="refused",
                response=result.refusal_reason,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                timing_ms=0.0,
            )
            print(
                f"  [Self-Growth] Enriching question with context (retry {retry + 1}/{max_retries})..."
            )

            # Use the LLM to generate a more specific, falsifiable question
            enriched_question = await self._enrich_question(current_question, result.refusal_reason)
            if enriched_question and enriched_question != current_question:
                print(f"  [Self-Growth] Enriched question: {enriched_question}")
                question_logger.stage(
                    name="self-growth-retry",
                    status="enriched",
                    model_response=enriched_question,
                )
                current_question = enriched_question
                result = await forecast_agent.run(current_question, **kwargs)
            else:
                question_logger.stage(
                    name="self-growth-retry",
                    status="failed",
                    error="enrichment returned no usable question",
                )
                break

        prob = result.forecast.probability if result.forecast else None
        print(f"  [Self-Growth] Done. Result: {prob if prob is not None else 'N/A'}")
        question_logger.summary(
            forecast=result.forecast.model_dump(mode="json") if result.forecast else None,
            refusal_reason=result.refusal_reason,
            token_usage={
                "steps_used": result.steps_used,
                "tokens_used": result.tokens_used,
                "cost_used_usd": result.cost_used_usd,
            },
            latency_s=question_latency,
        )
        return result

    async def _enrich_question(self, question: str, refusal_reason: str) -> str:
            """Use the LLM to add context and make a vague question falsifiable.

            Args:
                question: The original vague question
                refusal_reason: Why the pipeline refused it

            Returns:
                A more specific, falsifiable version of the question
            """
            from forecast_agent.llm.provider import ChatMessage
            from datetime import datetime, UTC

            today = datetime.now(UTC).strftime("%B %d, %Y")

            # Detect domain from keywords in question/refusal
            question_lower = question.lower()
            refusal_lower = refusal_reason.lower()
            combined = question_lower + " " + refusal_lower
            domain = "general"
            climate_kw = ["temperature", "climate", "global", "warming", "carbon", "emissions", "greenhouse", "sea level", "arctic", "ice", "precipitation", "drought", "flood"]
            finance_kw = ["stock", "price", "market", "trading", "share", "equity", "dividend", "earnings", "valuation", "forecast", "future price", "higher", "lower", "rise", "fall"]
            for kw in climate_kw:
                if kw in combined:
                    domain = "climate"
                    break
            if domain == "general":
                for kw in finance_kw:
                    if kw in combined:
                        domain = "finance"
                        break

            # Domain-specific guidance
            if domain == "climate":
                tips = [
                    "For climate/environment questions, specify a named region (country, continent, or global average) and a specific temperature metric (e.g., 'NASA GISS global average temperature anomaly', 'NOAA global mean surface temperature').",
                    "Use a clear baseline: 'pre-industrial (1850-1900)', '1951-1980 average', or '20th century average'. These are the standard scientific baselines.",
                    "Reference authoritative sources: NASA GISS, NOAA National Climate Report, Berkeley Earth, or IPCC.",
                    "Resolution dates MUST be after {today} and within 1-3 years so climate data is available.",
                ]
                example = "\"Will NASA GISS global average surface temperature increase by at least 0.3C relative to the 1951-1980 baseline by the end of 2027?\""
            elif domain == "finance":
                tips = [
                    "For stock/predict questions, specify a named stock/ticker (e.g., 'AAPL', 'MSFT', 'NVDA') and a specific measurement (e.g., 'closing price', 'adjusted close', 'market cap').",
                    "Use a clear FUTURE time horizon - resolution date MUST be after {today} and within 1-30 days so market data is available (e.g., 'by December 31, 2027' or 'in 7 days'). NEVER use a past date.",
                    "If you need historical price data, specify a series_id from the available list (cell-00 through cell-05), and give start and end dates in ISO format (e.g., '2026-11-01T00:00:00Z' to '2026-11-20T00:00:00Z').",
                    "Reference authoritative sources: company earnings reports, SEC filings, financial news, or official stock exchange data.",
                    "The series_ids available in the data store are: cell-00, cell-01, cell-02, cell-03, cell-04, cell-05. Use these when requesting historical price data.",
                ]
                example = '"Will AAPL closing price be higher than $250.00 on December 31, 2027? Reason about the data you have and pick the correct series and horizon."'
            else:
                tips = [
                    "Add a specific country, company, region, or measurable metric",
                    "Add a clear FUTURE time horizon - resolution date MUST be after {today} (e.g., 'by December 2027' or 'in the next 12 months'). NEVER use a past date.",
                    "Add a measurable outcome with a clear resolution criterion (e.g., 'above XX%' or 'higher than XX').",
                    "Reference authoritative sources: official government reports, central bank data, scientific organizations, or public company filings.",
                ]
                example = '"Will the US CDC-reported childhood vaccination rate for MMR increase above 92% by December 2027?"'

            prompt = (
                "The forecasting pipeline refused this question because it's too vague:\n"
                f"Question: \"{question}\"\n"
                f"Refusal reason: \"{refusal_reason}\"\n"
                f"\n"
                f"Today's date is {today}. Rewrite this question to be specific and falsifiable. Add reasonable assumptions for:\n"
                "- A specific entity/subject (e.g., a country, company, or measurable metric)\n"
                "- A clear FUTURE time horizon - the resolution date MUST be after {today} (e.g., \"by December 2027\" or \"in the next 12 months\"). NEVER use a past date.\n"
                "- A measurable outcome with a clear resolution criterion\n"
                f"\n"
                f"Domain: {domain}\n"
                "Tips:\n"
                + "\n".join(f"  - {t.format(today=today)}" for t in tips)
                + "\n"
                "Keep the same general topic but make it concrete. Return ONLY the rewritten question, nothing else.\n"
                "Examples:\n"
                "- \"Will vaccination rates increase next year?\" -> \"Will the US CDC-reported childhood vaccination rate for MMR increase above 92% by December 2027?\"\n"
                + (f"- \"Will global temperatures rise further?\" -> {example}\n" if domain == "climate" else "")
            )
            messages = [ChatMessage(role="user", content=prompt)]
            try:
                completion = await self.provider.complete(
                    messages, model="", max_tokens=300, temperature=0.3
                )
                enriched = completion.content.strip().strip('"')
                # Basic validation - should be a question and longer than original
                if enriched and "?" in enriched and len(enriched) > len(question):
                    return enriched
            except Exception as e:
                print(f"  [Self-Growth] Enrichment failed: {e}")
            return question

    def run_sync(self, raw_question: str, **kwargs: Any) -> "ForecastRun":
        """Synchronous wrapper for run()."""
        return _async_run(self.run(raw_question, **kwargs))

    def get_available_tools(self) -> list[str]:
        """Get list of all available tool names."""
        return [t.name for t in self.registry.list_tools()]

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all registered tools with full details."""
        return [{
            "name": t.name,
            "description": t.description,
            "module": t.module,
        } for t in self.registry.list_tools()]


def create_dynamic_agent(settings: Settings, registry: Optional[ToolRegistry] = None) -> DynamicForecastAgent:
    """Create a self-growing forecasting agent.

    Args:
        settings: Application settings
        registry: Optional tool registry (creates new one if not provided)

    Returns:
        A DynamicForecastAgent instance
    """
    return DynamicForecastAgent(settings, registry=registry)


# Shared event loop for keeping async code alive across multiple calls
# The pipeline's AsyncExitStack requires a live event loop, and asyncio.run()
# closes the loop when done. We keep a single loop alive for the entire test.
_shared_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Get or create a shared event loop that stays alive for the test.

    Returns:
        The shared event loop
    """
    global _shared_loop
    if _shared_loop is None or _shared_loop.is_closed():
        _shared_loop = asyncio.new_event_loop()
    return _shared_loop


def _async_run(coro):
    """Run a coroutine using anyio.from_thread.run() which properly manages the async context.

    Uses anyio.from_thread.run() instead of asyncio.run() or anyio.run() so that
    the anyio task group (used by MCP servers) works correctly with the
    existing asyncio event loop.
    """
    import anyio
    from anyio import from_thread

    # anyio.from_thread.run() runs the coroutine in the current thread's
    # event loop, properly setting up the anyio async context.
    return from_thread.run(coro)


def create_dynamic_agent(settings: Settings, registry: Optional[ToolRegistry] = None) -> DynamicForecastAgent:
    """Create a self-growing forecasting agent.

    Args:
        settings: Application settings
        registry: Optional tool registry (creates new one if not provided)

    Returns:
        A DynamicForecastAgent instance
    """
    return DynamicForecastAgent(settings, registry=registry)


def run_self_growing_agent(
    question: str,
    settings: Optional[Settings] = None,
    registry: Optional[ToolRegistry] = None,
) -> "ForecastRun":
    """Quick function to run a query with a self-growing agent.

    Args:
        question: The question to answer
        settings: Application settings (creates new one if not provided)
        registry: Optional tool registry

    Returns:
        The forecast result
    """
    settings = settings or Settings()
    agent = create_dynamic_agent(settings, registry=registry)
    return _async_run(agent.run(question))
