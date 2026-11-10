import math


def skill_score(model_score: float, baseline_score: float) -> float:
    if not math.isfinite(model_score) or model_score < 0:
        raise ValueError("model_score must be finite and nonnegative")
    if not math.isfinite(baseline_score) or baseline_score <= 0:
        raise ValueError("baseline_score must be finite and positive")
    return 1 - model_score / baseline_score
