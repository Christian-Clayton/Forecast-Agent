"""LangChain / LangGraph forecasting pipeline.

Converts the 8-stage forecasting agent into a structured LangGraph graph.
Each stage is a node; stage routing is a conditional transition on the state
schema (no ad-hoc if/else chains in run code).
"""

from __future__ import annotations

import logging
import json
import asyncio
from datetime import datetime, UTC
from typing import Any
from uuid import uuid4

from langchain_core.messages import ChatMessage
from pydantic import BaseModel, ConfigDict, Field

from forecast_agent.agent.stages import get_stage_stub
from forecast_agent.agent.stages.schemas import (
    DecomposeOutput,
    EnsembleOutput,
    FinalizeOutput,
    ModelOutput,
)
from forecast_agent.config import Settings
from forecast_agent.llm.langchain_provider import LangChainProvider
from forecast_agent.models import Forecast, Question
from forecast_agent.skills.loader import SkillLoader

from forecast_agent.agent.graph_state import ForecastAgentState


# Stage name -> (state field names to read from state)
_STAGE_STATE_KEYS = {
    "intake": ["question"],
    "classify": ["falsifiable_text", "question_type"],
    "base_rate": ["falsifiable_text", "question_type"],
    "decompose": ["falsifiable_text", "domain_class"],
    "model": ["falsifiable_text", "domain_class", "base_rate", "base_rate_source", "decompose_output"],
    "ensemble": ["question", "domain_class", "base_rate"],
    "critique": ["domain_class"],
    "finalize": ["falsifiable_text", "domain_class", "base_rate_source"],
}

# Stage name -> key to write to state
# Lang key -> state field for each stage
_STAGE_LANGC_TO_STATE = {
    "intake": {"raw_question": "question", "as_of": "__AS_OF_LITERAL__"},
    "classify": {"falsifiable_text": "falsifiable_text", "question_type": "question_type"},
    "base_rate": {"falsifiable_text": "falsifiable_text", "question_type": "question_type"},
    "decompose": {"falsifiable_text": "falsifiable_text", "domain_class": "domain_class"},
    "model": {"falsifiable_text": "falsifiable_text", "domain_class": "domain_class", "base_rate": "base_rate", "base_rate_source": "base_rate_source", "decomposition": "decompose_output", "baseline_result": "__BASELINE_RESULT__", "candidate_score": "__CANDIDATE_SCORE__"},
    "ensemble": {"question": "question", "domain_class": "domain_class", "base_rate": "base_rate", "tool_call_ids": "__TOOL_CALL_IDS__"},
    "critique": {"bottom_line": "__BOTTOM_LINE__", "working": "__WORKING__", "tool_call_ids": "__TOOL_CALL_IDS__", "domain_class": "domain_class"},
    "finalize": {"falsifiable_text": "falsifiable_text", "domain_class": "domain_class", "toolkit": "__TOOLKIT__", "probability": "model_output.probability", "quantiles": "model_output.quantiles", "tool_call_ids": "__TOOL_CALL_IDS__", "base_rate_source": "base_rate_source", "reasoning": "model_output.reasoning"},
}

_STAGE_OUTPUT_KEY = {
    "intake": None,  # intake writes multiple fields
    "classify": "domain_class",
    "base_rate": "base_rate",
    "decompose": "decompose_output",
    "model": "model_output",
    "ensemble": "ensemble_output",
    "critique": "critique_output",
    "finalize": "finalize_output",
}

# Fields that intake writes to state
_INTEST_FIELDS = ["falsifiable_text", "question_type", "refusal_reason", "budget_used", "steps_used", "tokens_used"]


_LOGGER = logging.getLogger(__name__)


def _build_messages(build_prompt_fn, system_kwargs: dict, user_payload: dict) -> list[ChatMessage]:
    system_content = build_prompt_fn(**system_kwargs)
    user_content = json.dumps(user_payload, default=str)
    return [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ]


def _get_state_value(state: ForecastAgentState, field_path: str) -> Any:
    parts = field_path.split(".")
    value = state
    for part in parts:
        value = getattr(value, part)
    return value


def build_graph(provider: LangChainProvider, settings: Settings):
    from langgraph.graph import StateGraph

    graph = StateGraph(ForecastAgentState)
    graph.set_entry_point("intake")

    def make_node(stage: str):
        async def node(state: ForecastAgentState) -> dict:
            build_prompt_fn, validate_output = get_stage_stub(stage)
            # Build input from state
            stage_input = {}
            # Process all keys from the lang-ch mapping
            for lang_key, mapping_state_key in _STAGE_LANGC_TO_STATE[stage].items():
                if mapping_state_key.startswith("__") and mapping_state_key.endswith("__"):
                    # Literal value (empty string for as_of; None for baseline/candidate)
                    stage_input[lang_key] = "" if mapping_state_key == "__AS_OF_LITERAL__" else None
                else:
                    stage_input[lang_key] = _get_state_value(state, mapping_state_key)
            # Build prompt
            system_content = build_prompt_fn(**stage_input)
            user_payload = dict(stage_input)
            messages = _build_messages(build_prompt_fn, stage_input, user_payload)
            response = await provider.complete(messages, model=settings.openai_compat_model, max_tokens=settings.max_tokens)
            parsed = None
            # Try direct JSON parse
            try:
                parsed = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback: extract JSON object from conversational text (same as pipeline)
                _start = response.content.find('{')
                _end = response.content.rfind('}', _start)
                if _start != -1 and _end > _start:
                    try:
                        parsed = json.loads(response.content[_start:_end + 1])
                    except json.JSONDecodeError:
                        parsed = None
            output = validate_output(parsed)

            # Map output model fields to state fields
            if stage == "intake":
                return {
                    "falsifiable_text": output.falsifiable_text,
                    "question_type": output.type,
                    "refusal_reason": output.refusal_reason,
                    "resolution_criteria": output.resolution_criteria,
                    "resolution_source": output.resolution_source,
                    "resolves_at": output.resolves_at,
                }
            elif stage == "classify":
                        return {
                            "domain_class": output.domain_class,
                            "falsifiable_text": state.falsifiable_text,
                            "question_type": state.question_type,
                        }
            elif stage == "base_rate":
                return {
                    "base_rate": output.unverified_prior,
                    "falsifiable_text": state.falsifiable_text,
                    "question_type": state.question_type,
                }
            elif stage == "decompose":
                return {"decompose_output": output}
            elif stage == "model":
                return {
                    "model_output": output,
                    "falsifiable_text": state.falsifiable_text,
                    "question_type": state.question_type,
                    "domain_class": state.domain_class,
                    "base_rate": state.base_rate,
                    "base_rate_source": state.base_rate_source,
                }
            elif stage == "ensemble":
                return {
                    "ensemble_output": output,
                    "falsifiable_text": state.falsifiable_text,
                    "question_type": state.question_type,
                    "domain_class": state.domain_class,
                    "base_rate": state.base_rate,
                    "tool_call_ids": state.get("tool_call_ids", []),
                }
            elif stage == "critique":
                return {
                    "critique_output": output,
                    "domain_class": state.domain_class,
                    "falsifiable_text": state.falsifiable_text,
                }
            elif stage == "finalize":
                # Return the FinalizeOutput model directly - LangGraph will
                # merge it into the state and Pydantic will handle conversion.
                return {
                    "finalize_output": output,
                    "falsifiable_text": state.falsifiable_text,
                    "question_type": state.question_type,
                    "domain_class": state.domain_class,
                    "base_rate_source": state.base_rate_source,
                }
        return node

    for stage in ["intake", "classify", "base_rate", "decompose", "model", "ensemble", "critique", "finalize"]:
        graph.add_node(stage, make_node(stage))
    graph.add_edge("intake", "classify")
    graph.add_edge("classify", "base_rate")
    graph.add_edge("base_rate", "decompose")
    graph.add_edge("decompose", "model")
    graph.add_edge("model", "ensemble")
    graph.add_edge("ensemble", "critique")
    graph.add_edge("critique", "finalize")
    graph.add_edge("finalize", "__end__")
    return graph.compile()


class ForecastAgent:
    """LangChain/LangGraph wrapper around the 8-stage forecasting pipeline."""

    def __init__(self, provider: LangChainProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings
        self._graph = build_graph(provider, settings)
        self._skills = SkillLoader(settings.skills_dir)

    async def run(self, raw_question: str, *, as_of: datetime | None = None) -> "ForecastRun":
        from forecast_agent.agent.pipeline import ForecastRun, Forecast

        if not raw_question.strip():
            raise ValueError("question must not be empty")

        state = ForecastAgentState(
            question=raw_question,
            falsifiable_text="",
            question_type="binary",
            domain_class="thin",
            base_rate=None,
            base_rate_source="unverified_prior",
            budget_used=0.0,
            steps_used=0,
            tokens_used=0,
            trace_id=str(uuid4()),
            started_at=datetime.now(UTC),
            mcp_sessions={},
        )

        # Debug: print state before ainvoke
        import logging
        _LOGGER.debug(f"Graph state before ainvoke: {state.model_dump()}")

        final_state = await self._graph.ainvoke(state)

        # Debug: print final state
        _LOGGER.debug(f"Graph final state: {final_state.model_dump() if hasattr(final_state, 'model_dump') else final_state}")

        # Handle the case where the graph returns a dict instead of a model
        if not isinstance(final_state, ForecastAgentState):
            final_state = ForecastAgentState(**final_state) if isinstance(final_state, dict) else final_state

        answer = getattr(final_state, "finalize_output", None)
        if answer is None:
            answer = FinalizeOutput()
        return ForecastRun(
            question=None,
            forecast=None,
            answer=answer,
            refusal_reason=getattr(final_state, "refusal_reason", None),
            trace={},
            steps_used=getattr(final_state, "steps_used", 0),
            tokens_used=getattr(final_state, "tokens_used", 0),
            cost_used_usd=0.0,
        )

    async def aclose(self) -> None:
        pass
