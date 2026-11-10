import math
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """Raised when a per-question step, token, or cost cap is reached."""


@dataclass
class QuestionBudget:
    max_steps: int
    max_tokens: int
    max_cost_usd: float
    steps_used: int = 0
    tokens_used: int = 0
    cost_used_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.max_tokens < 1:
            raise ValueError("step and token caps must be positive")
        if not math.isfinite(self.max_cost_usd) or self.max_cost_usd < 0:
            raise ValueError("cost cap must be finite and nonnegative")

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    def consume_step(self) -> None:
        if self.steps_used >= self.max_steps:
            raise BudgetExceeded("question step budget exceeded")
        self.steps_used += 1

    def record_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float | None,
    ) -> None:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token usage must be nonnegative")
        if cost_usd is not None and (not math.isfinite(cost_usd) or cost_usd < 0):
            raise ValueError("reported cost must be finite and nonnegative")

        self.tokens_used += input_tokens + output_tokens
        if cost_usd is not None:
            self.cost_used_usd += cost_usd

        if self.tokens_used > self.max_tokens:
            raise BudgetExceeded("question token budget exceeded")
        if cost_usd is None:
            raise BudgetExceeded(
                "provider cost was not reported; cost cap is unenforceable"
            )
        if self.cost_used_usd > self.max_cost_usd:
            raise BudgetExceeded("question cost budget exceeded")
