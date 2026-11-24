"""Tool selection stage: the model decides which tools to call before forecasting.

The model receives the question, its domain classification, and a list of
available tools. It returns a list of tool calls with arguments, which the
pipeline executes. This makes the agent autonomous about choosing the right
tool for the job.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forecast_agent.models import Quantiles


class ToolSelectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    falsifiable_text: str
    domain_class: Literal["thin", "fat", "mixed"]
    question_type: Literal["binary", "quantile", "count_range"]
    available_tools: list[dict]
    dataset_series: list[str] | None = None
    as_of: datetime | None = None


class ToolSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: list[dict[str, object]] = Field(default_factory=list)
    reasoning: str

    @model_validator(mode="after")
    def validate_tool_calls(self) -> "ToolSelectionOutput":
        # Validate each tool call's arguments against its schema
        for call in self.tool_calls:
            name = call.get("name")
            arguments = call.get("arguments", {})
            if not isinstance(name, str):
                raise ValueError("tool_calls must have a string 'name' field")
            if not isinstance(arguments, dict):
                raise ValueError("tool_calls must have a dict 'arguments' field")
        return self


def build_tool_selection_prompt(
    falsifiable_text: str,
    domain_class: Literal["thin", "fat", "mixed"],
    question_type: Literal["binary", "quantile", "count_range"],
    available_tools: list[dict],
    dataset_series: list[str] | None = None,
    as_of: datetime | None = None,
) -> str:
    """Build the system prompt for the tool selection stage.

    The model decides which tools (if any) to call based on the question
    and what information it needs. It should only call tools that add
    real value (e.g., live web data, historical time series) and not
    call trivial tools like base_rate or ensemble_aggregate.
    """
    tool_list = "\n".join(
        f"  - {t['name']}: {t['description']}\n    Args: {t['input_schema']}"
        for t in available_tools
    )
    return (
        "You are a tool selection agent. Given a question, decide which tools "
        "should be called to gather the information needed to make a good "
        "forecast. Do not call tools that add no value.\n\n"
        "AVAILABLE TOOLS:\n"
        f"{tool_list}\n\n"
        "RULES:\n"
        "- Only call a tool if it genuinely adds information you don't already have.\n"
        "- Do NOT call get_reference_class (base_rate) - that is handled by a "
        "separate stage.\n"
        "- Do NOT call ensemble_aggregate - that is handled by the model stage.\n"
        "- Do NOT call validate_quantiles - that is handled by the model stage.\n"
        "- Do NOT call fit_ets, fit_arima, walk_forward_score, or baseline_forecast "
        "- those are handled by the decompose stage.\n"
        "- Do call browser_fetch when you need live web data the data store does "
        "not have (e.g., recent news, current market data, recent statistics).\n"
        "- Do call get_series when you need historical data from the data store.\n"
        "- Return an empty tool_calls list if no tools are needed.\n"
        "- each tool_call must have a 'name' (string) and 'arguments' (object).\n"
        "- all arguments must match the tool's input_schema.\n\n"
        "QUESTION:\n"
        f"{falsifiable_text}\n\n"
        f"Domain class: {domain_class}\n"
        f"Question type: {question_type}\n"
        f"Time of analysis: {as_of}\n"
        f"Dataset series available: {dataset_series}\n"
        "Respond with a JSON object containing 'tool_calls' and 'reasoning'."
    )


def validate_tool_calls(output: dict) -> ToolSelectionOutput:
    """Validate the tool selection output."""
    return ToolSelectionOutput.model_validate(output)
