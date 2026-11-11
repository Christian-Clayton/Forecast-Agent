"""Stage 2: classify.

Classifies a falsifiable question as thin-tailed, fat-tailed, or mixed.
Fat-tailed questions route to Toolkit B (scenarios and bounds), not a point
probability.
"""

from forecast_agent.agent.stages.schemas import ClassifyOutput

_FAT_KEYWORDS = frozenset(
    {
        "market",
        "war",
        "pandemic",
        "regulation",
        "breakthrough",
        "disruption",
        "crash",
        "boom",
        "bubble",
        "black swan",
        "rare event",
        "tail risk",
        "extreme event",
        "volatility",
    }
)


def build_classify_prompt(falsifiable_text: str, question_type: str) -> str:
    """Return the system prompt and user message for the classify stage."""
    return (
        "Classify the question as thin-tailed, fat-tailed, or mixed.\n"
        "Output ONLY JSON matching the ClassifyOutput schema.\n"
        "Reasoning rules:\n"
        "  - Fat-tailed: a single extreme event could dominate the outcome "
        "(markets, war, pandemics, breakthroughs, rare events).\n"
        "  - Thin-tailed: outcomes cluster around a base rate (counts, "
        "measurements, stable processes).\n"
        "  - Mixed: decompose and classify each sub-part separately.\n"
        "  - Always state the reasoning explicitly.\n"
        f"  - Question: {falsifiable_text}\n"
        f"  - Type: {question_type}"
    )


def classify_domain(falsifiable_text: str) -> str:
    """Heuristic domain classification before the LLM pass."""
    text = falsifiable_text.lower()
    if any(k in text for k in _FAT_KEYWORDS):
        return "fat"
    return "thin"


def validate_classify(output: dict) -> ClassifyOutput:
    """Validate classify output."""
    domain = output.get("domain_class")
    if domain not in {"thin", "fat", "mixed"}:
        raise ValueError("domain_class must be thin, fat, or mixed")
    return ClassifyOutput(**output)
