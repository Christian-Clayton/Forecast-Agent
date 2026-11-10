import re
from collections.abc import Collection

_NUMBER = re.compile(r"(?<![A-Za-z0-9_])(?:[$€£]\s*)?-?(?:\d+(?:\.\d*)?|\.\d+)(?:%|x)?")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_TOOL_CITATION = re.compile(r"\[tool:([A-Za-z0-9._:-]+)\]")


class NumericCitationError(ValueError):
    """Raised when a numeric statement lacks a valid tool-call citation."""


def validate_numeric_citations(
    text: str,
    *,
    valid_tool_call_ids: Collection[str],
) -> None:
    valid_ids = set(valid_tool_call_ids)
    for sentence_number, sentence in enumerate(_SENTENCE_BOUNDARY.split(text), start=1):
        if not _NUMBER.search(sentence):
            continue
        cited_ids = set(_TOOL_CITATION.findall(sentence))
        if not cited_ids.intersection(valid_ids):
            raise NumericCitationError(
                f"uncited numeric claim in sentence {sentence_number}"
            )
