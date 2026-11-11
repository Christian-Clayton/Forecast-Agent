"""Stage registry: maps stage names to their builder and validator functions.

Each stage module (e.g. intake.py, classify.py) exports:
  - build_prompt(**kwargs) -> str   # construct the system/user prompt
  - validate_output(output: dict) -> OutputModel  # validate the LLM response
"""

from . import (
    base_rate,
    classify,
    critique,
    decompose,
    ensemble,
    finalize,
    intake,
    model,
)

# Ordered list of stages for Phase 4.
PHASE_4_STAGES = [
    "intake",
    "classify",
    "base_rate",
    "decompose",
    "model",
    "finalize",
]

# All eight stages (phases 4-5).
_ALL_STAGES = PHASE_4_STAGES + ["ensemble", "critique"]


def get_stage_stub(stage_name: str):
    """Return (build_prompt, validate_output) for the given stage."""
    registry = {
        "intake": (intake.build_intake_prompt, intake.validate_intake),
        "classify": (classify.build_classify_prompt, classify.validate_classify),
        "base_rate": (base_rate.build_base_rate_prompt, base_rate.validate_base_rate),
        "decompose": (decompose.build_decompose_prompt, decompose.validate_decompose),
        "model": (model.build_model_prompt, model.validate_model),
        "ensemble": (ensemble.build_ensemble_prompt, ensemble.validate_ensemble),
        "critique": (critique.build_critique_prompt, critique.validate_critique),
        "finalize": (finalize.build_finalize_prompt, finalize.validate_finalize),
    }
    if stage_name not in registry:
        raise KeyError(f"unknown stage: {stage_name!r}")
    return registry[stage_name]


def is_phase_4_stage(stage_name: str) -> bool:
    """Return True if the stage belongs to Phase 4."""
    return stage_name in PHASE_4_STAGES
