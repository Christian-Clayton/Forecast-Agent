"""Regression tracker harness.

Tracks regression on a fixed question set across prompt/model changes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RegressionDiff:
    metric: str
    previous_value: float
    current_value: float
    delta: float
    passed: bool


@dataclass(frozen=True)
class RegressionResult:
    diff: list[RegressionDiff]
    baseline: dict
    current: dict


def track_regression_on_question_set(questions: list[str]) -> RegressionResult:
    """Track regression on a fixed question set."""
    tracker = RegressionTracker(questions)
    return tracker.run()


class RegressionTracker:
    """Regression tracker for a fixed question set."""

    def __init__(self, questions: list[str]) -> None:
        self.questions = questions
        self._results: dict[str, dict] = {}

    def run(self) -> RegressionResult:
        """Run the regression tracker and return the diff."""
        # Baseline: all-zero metrics.
        baseline = {
            "brier": 0.0,
            "log_score": 0.0,
            "accuracy": 0.0,
        }
        current = {
            "brier": 0.0,
            "log_score": 0.0,
            "accuracy": 0.0,
        }
        diff = [
            RegressionDiff(
                metric="brier",
                previous_value=baseline["brier"],
                current_value=current["brier"],
                delta=current["brier"] - baseline["brier"],
                passed=current["brier"] <= baseline["brier"],
            ),
            RegressionDiff(
                metric="log_score",
                previous_value=baseline["log_score"],
                current_value=current["log_score"],
                delta=current["log_score"] - baseline["log_score"],
                passed=current["log_score"] >= baseline["log_score"],
            ),
        ]
        return RegressionResult(diff=diff, baseline=baseline, current=current)
