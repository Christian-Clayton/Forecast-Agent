from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundaryModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class Question(BoundaryModel):
    id: str
    raw_text: str
    falsifiable_text: str
    type: Literal["binary", "quantile", "count_range"]
    resolution_criteria: str
    resolution_source: str
    resolves_at: datetime
    created_at: datetime


class Quantiles(BoundaryModel):
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float

    @model_validator(mode="after")
    def validate_order(self) -> "Quantiles":
        values = [self.q05, self.q25, self.q50, self.q75, self.q95]
        if values != sorted(values):
            raise ValueError("quantiles must be in nondecreasing order")
        return self


class EnsembleMember(BoundaryModel):
    model_id: str
    prompt_id: str
    probability: float = Field(ge=0, le=1)


class Forecast(BoundaryModel):
    id: str
    question_id: str
    created_at: datetime
    as_of: datetime
    domain_class: Literal["thin", "fat", "mixed"]
    classification_reasoning: str
    toolkit: Literal["A", "B"]
    probability: float | None = Field(default=None, ge=0, le=1)
    quantiles: Quantiles | None = None
    base_rate: float | None = Field(default=None, ge=0, le=1)
    base_rate_source: str | None = None
    ensemble_members: list[EnsembleMember] = Field(default_factory=list)
    ensemble_spread: float | None = Field(default=None, ge=0)
    tool_call_ids: list[str] = Field(default_factory=list)
    skill_ids_loaded: list[str] = Field(default_factory=list)
    cost_usd: float | None = Field(default=None, ge=0)
    latency_s: float | None = Field(default=None, ge=0)
    trace_id: str | None = None

    @model_validator(mode="after")
    def validate_estimate_shape(self) -> "Forecast":
        if (self.probability is None) == (self.quantiles is None):
            raise ValueError("provide exactly one of probability or quantiles")
        return self


class Resolution(BoundaryModel):
    question_id: str
    resolved_at: datetime
    outcome: bool | float | str
    resolver_note: str


class Score(BoundaryModel):
    forecast_id: str
    brier: float | None = Field(default=None, ge=0, le=1)
    log_score: float | None = Field(default=None, ge=0)
    pinball: float | None = Field(default=None, ge=0)
    baseline_brier: float | None = Field(default=None, ge=0, le=1)
    skill_score: float | None = None
