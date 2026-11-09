"""Explicit evaluator request settings, independent of answer-generation limits."""
from __future__ import annotations

EVALUATOR_OUTPUT_CAP = 4096
EVALUATOR_INPUT_CAP = 65536
EVALUATOR_TIMEOUT_SECONDS = 240
REASONING_CHOICES = ("default", "none", "low", "medium", "high", "xhigh", "max")


def reasoning_parameter(value="default"):
    """Omit the parameter for the API default; preserve explicit reasoning choices."""
    if value not in REASONING_CHOICES:
        raise ValueError("unknown evaluator reasoning preference")
    return None if value == "default" else value


def evaluator_settings(*, model, reasoning="default",
                       output_cap=EVALUATOR_OUTPUT_CAP):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("evaluator model must be explicitly selected")
    if not isinstance(output_cap, int) or isinstance(output_cap, bool) or output_cap < 1:
        raise ValueError("evaluator output cap must be a positive integer")
    return {"model": model, "reasoning_effort": reasoning_parameter(reasoning),
            "reasoning_preference": reasoning, "max_output_tokens": output_cap,
            "max_input_tokens": EVALUATOR_INPUT_CAP,
            "timeout_seconds": EVALUATOR_TIMEOUT_SECONDS}


def identified_evidence(rows):
    """Preserve supplied source labels without inventing passage numbering."""
    return [{key: row[key] for key in ("id", "evidence_id", "title", "label", "source_label", "passage_number", "text")
             if key in row} if isinstance(row, dict) else row for row in rows]
