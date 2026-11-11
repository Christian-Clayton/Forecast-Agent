from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forecast_agent.models import EnsembleMember, Quantiles


class StageModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class IntakeInput(StageModel):
    raw_question: str
    as_of: datetime


class IntakeOutput(StageModel):
    falsifiable_text: str | None = None
    type: Literal["binary", "quantile", "count_range"] | None = None
    resolution_criteria: str | None = None
    resolution_source: str | None = None
    resolves_at: datetime | None = None
    refusal_reason: str | None = None

    @model_validator(mode="after")
    def validate_intake(self) -> "IntakeOutput":
        if self.refusal_reason is None and any(
            value is None
            for value in (
                self.falsifiable_text,
                self.type,
                self.resolution_criteria,
                self.resolution_source,
                self.resolves_at,
            )
        ):
            raise ValueError("complete intake fields or a refusal reason are required")
        return self


class ClassifyInput(StageModel):
    falsifiable_text: str
    question_type: Literal["binary", "quantile", "count_range"]


class ClassifyOutput(StageModel):
    domain_class: Literal["thin", "fat", "mixed"]
    reasoning: str


class BaseRateInput(StageModel):
    falsifiable_text: str
    question_type: Literal["binary", "quantile", "count_range"]


class BaseRateOutput(StageModel):
    reference_class_description: str | None = None
    unverified_prior: float | None = Field(default=None, ge=0, le=1)
    reasoning: str


class DecomposeInput(StageModel):
    falsifiable_text: str
    domain_class: Literal["thin", "fat", "mixed"]


class DecomposeOutput(StageModel):
    subquestions: list[str] = Field(default_factory=list)
    series_id: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    horizon: int = Field(default=1, ge=1)
    candidate_model: Literal["ets", "arima"] | None = None

    @model_validator(mode="after")
    def validate_series_window(self) -> "DecomposeOutput":
        if self.series_id is not None and (self.start is None or self.end is None):
            raise ValueError("series requests require both start and end")
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("series end must not precede start")
        return self


class ModelInput(StageModel):
    falsifiable_text: str
    domain_class: Literal["thin", "fat", "mixed"]
    base_rate: float | None = Field(default=None, ge=0, le=1)
    base_rate_source: str
    decomposition: DecomposeOutput
    baseline_result: dict[str, object] | None = None
    candidate_score: dict[str, object] | None = None


class ModelOutput(StageModel):
    probability: float | None = Field(default=None, ge=0, le=1)
    quantiles: Quantiles | None = None
    reasoning: str

    @model_validator(mode="after")
    def validate_estimate_shape(self) -> "ModelOutput":
        if (self.probability is None) == (self.quantiles is None):
            raise ValueError("provide exactly one of probability or quantiles")
        return self


class FinalizeInput(StageModel):
    falsifiable_text: str
    domain_class: Literal["thin", "fat", "mixed"]
    toolkit: Literal["A", "B"]
    probability: float | None = None
    quantiles: Quantiles | None = None
    tool_call_ids: list[str]
    base_rate_source: str
    reasoning: str


class EnsembleInput(StageModel):
    question: str
    domain_class: Literal["thin", "fat", "mixed"]
    base_rate: float | None = None
    tool_call_ids: list[str] = Field(default_factory=list)
    spread_ratio: float = Field(default=2.0, ge=0)
    member_count: int = Field(default=3, ge=1, le=10)


class EnsembleOutput(StageModel):
    members: list[EnsembleMember]
    median: float
    trimmed_mean: float
    logit_mean: float
    spread: float
    spread_ratio: float
    switch_to_toolkit_b: bool


class FinalizeOutput(StageModel):
    bottom_line: str
    working: str
