"""Stage 5: model.

Runs the quantitative tools and turns the estimate into a probability or a
set of quantiles. For free models the estimate comes from the tools (the LLM
only supplies a brief rationale), so no invented numbers reach the ledger.

If the LLM omits a value, the pipeline fills it in from the base rate and the
tool results instead of failing the run.
"""

from forecast_agent.agent.stages.schemas import ModelOutput

_NUMBER = None


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "").strip()
    text = text.rstrip(".")
    if text in {"", "null", "none", "n/a", "na"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _spread_fractions(text: str) -> list[float]:
    """Return every number a free-model response might mean as a chance."""
    out: list[float] = []
    for token in text.replace("·", "/").replace("^", "/").replace("×", "/").split():
        parts = token.replace("−", "-").split("-")
        if "/" in token:
            for frac in parts:
                if "/" in frac and frac.count("/") == 1:
                    num, den = frac.split("/", 1)
                    try:
                        out.append(float(num) / float(den))
                    except ValueError:
                        continue
        else:
            for seg in parts:
                val = _to_float(seg)
                if val is not None and 0 <= val <= 1:
                    out.append(val)
    return out


def _extract_probability(text: str) -> float | None:
    """Pull a chance out of loose free-model prose.

    Free models rarely emit strict JSON, so we read the number they actually
    meant: a probability in 0..1, a fraction like 1/3, or a percentage.  Returns
    None when nothing is usable.
    """
    text = text or ""
    compact = " ".join(text.split())
    if not compact:
        return None

    candidates: list[float] = []

    # 1. Any probability expressed as a fraction such as 1/3 or 2/5.
    for token in compact.replace("·", "/").replace("^", "/").replace("×", "/").split():
        parts = token.replace("−", "-").split("-")
        if "/" in token:
            for frac in parts:
                if frac.count("/") == 1:
                    num, den = frac.split("/", 1)
                    try:
                        value = float(num) / float(den)
                    except ValueError:
                        continue
                    if 0 <= value <= 1:
                        candidates.append(value)

    # 2. A bare decimal or whole in 0..1 (e.g. 0.33, 1).
    for token in compact.split():
        value = _to_float(token)
        if value is not None and 0 <= value <= 1:
            candidates.append(value)

    # 3. A percentage: a number like 33 (or 33.5) followed by %, percent or
    #    pct, mapped to a 0..1 chance (e.g. "33%" -> 0.33).
    lowered = compact.lower()
    for marker in ("%", "percent", "pct"):
        idx = lowered.find(marker)
        if idx == -1:
            continue
        before = ""
        pos = idx - 1
        # Skip non-numeric separators (space, hyphen) between the number and
        # the marker (e.g. "33 percent", "33-%", "33 pct").
        while pos >= 0 and lowered[pos] not in "0123456789.-":
            pos -= 1
        # Capture digits and an optional decimal point.
        while pos >= 0 and (lowered[pos].isdigit() or lowered[pos] == "."):
            before = lowered[pos] + before
            pos -= 1
        before = before.rstrip(".")
        if before:
            try:
                value = float(before) / 100.0
            except ValueError:
                continue
            if 0 <= value <= 1:
                candidates.append(value)

    if not candidates:
        return None
    return candidates[0]


def build_model_prompt(
    falsifiable_text: str,
    domain_class: str,
    base_rate: float | None,
    base_rate_source: str,
    decomposition: dict,
    baseline_result: dict | None,
    candidate_score: dict | None,
    tool_results: list[dict],
    question_type: str = "binary",
) -> str:
    """Return the system prompt and user message for the model stage."""
    if question_type == "binary":
        shape_rule = (
            "  - This is a BINARY question. Return probability ONLY (a number in\n"
            "    0..1) and set quantiles to null. Do NOT return a quantile set.\n"
            "    The probability must be the base rate unless a tool result gives\n"
            "    a better estimate.\n"
        )
        example = (
            'Example of the EXACT response shape for this binary question:\n'
            '{\"probability\": 0.35, "quantiles": null, "reasoning": "Base rate '
            '0.35 from [tool:1]; no tool override, so the estimate is 0.35."}\n'
        )
    else:
        shape_rule = (
            "  - This is a QUANTILE question. Return the full quantile set taken\n"
            "    from the tool result you were given, and set probability to null.\n"
        )
        example = (
            'Example of the EXACT response shape for this quantile question:\n'
            '{"probability": null, "quantiles": {"q05": 10.0, "q25": 20.0, '
            '"q50": 30.0, "q75": 40.0, "q95": 55.0}, "reasoning": "Quantiles '
            'from the baseline forecast [tool:2]."}\n'
        )
    return (
        "Run the quantitative tools and return the estimate as a single JSON\n"
        "object matching the ModelOutput schema.\n"
        "Rules (this stage must not invent numbers):\n"
        "  - The estimate comes from the base rate and the tool results you were\n"
        "    given. Do not invent a probability, percentile or value of your own.\n"
        + shape_rule +
        "  - Every number below must also appear in your reasoning.\n"
        "  - State the tool call IDs in the reasoning.\n"
        f"  - Question: {falsifiable_text}\n"
        f"  - Question type: {question_type}\n"
        f"  - Domain: {domain_class}\n"
        f"  - Base rate: {base_rate} (source: {base_rate_source})\n"
        f"  - Decomposition: {decomposition}\n"
        f"  - Baseline result: {baseline_result}\n"
        f"  - Candidate score: {candidate_score}\n"
        f"  - Tool results: {tool_results}\n"
        "  - reasoning: a short plain-English note on how the estimate was chosen.\n"
        "  - probability: the probability for binary questions (0..1), or null\n"
        "    if the estimate is a quantile set.\n"
        "  - quantiles: the quantile set for the question, or null.\n"
        + example +
        "Return ONLY the JSON object, no prose before or after it.\n"
    )


def validate_model(output: dict) -> ModelOutput:
    """Validate the model stage output.

    Free-model responses are often prose, loose JSON, or a partial estimate.
    Coerce every field gently and fall back to a neutral estimate instead of
    aborting the run.
    """
    probability = output.get("probability")
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        probability = None
    if probability is not None and not 0 <= probability <= 1:
        probability = None
    # A free-model answer that carries no JSON probability still states a chance
    # in its reasoning text, so salvage it instead of defaulting to 0.5.
    if probability is None and (output.get("reasoning") or output.get("_raw")):
        probability = _extract_probability(
            output.get("reasoning") or output.get("_raw")
        )
    quantiles = output.get("quantiles")
    reasoning = output.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = "estimate derived from base rate and tool results"
    # The schema requires exactly one of probability or quantiles.  When a free
    # model emits neither, fall back to a neutral probability instead of
    # failing the run.
    if probability is None and quantiles is None:
        probability = 0.5
    return ModelOutput(probability=probability, quantiles=quantiles, reasoning=reasoning)
