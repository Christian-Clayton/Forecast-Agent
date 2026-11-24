"""State schema for the LangGraph forecasting agent."""

from __future__ import annotations

from datetime import datetime, UTC
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from forecast_agent.agent.stages.schemas import (
    DecomposeOutput,
    EnsembleOutput,
    FinalizeOutput,
    ModelOutput,
)


class ForecastAgentState(BaseModel):
    question: str
    falsifiable_text: str = ""
    question_type: str = "binary"
    domain_class: str = "thin"
    base_rate: float | None = None
    base_rate_source: str = "unverified_prior"
    decompose_output: DecomposeOutput = Field(default_factory=DecomposeOutput)
    model_output: ModelOutput = Field(
        default_factory=lambda: ModelOutput(
            probability=0.5, reasoning="",
        )
    )
    ensemble_output: EnsembleOutput = Field(
        default_factory=lambda: EnsembleOutput(
            members=[],
            median=0.0,
            trimmed_mean=0.0,
            logit_mean=0.0,
            spread=0.0,
            spread_ratio=2.0,
            switch_to_toolkit_b=False,
        )
    )
    critique_output: FinalizeOutput = Field(
        default_factory=lambda: FinalizeOutput(bottom_line="", working="")
    )
    finalize_output: FinalizeOutput = Field(
        default_factory=lambda: FinalizeOutput(bottom_line="", working="")
    )
    refusal_reason: str | None = None
    budget_used: float = 0.0
    steps_used: int = 0
    tokens_used: int = 0
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mcp_sessions: dict = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
