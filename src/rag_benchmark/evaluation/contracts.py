"""Versioned evaluation sidecars; operational failures are not semantic zeros."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

EVALUATION_VERSION = "evaluation-v1"


def aggregate_metric_results(results):
    keys = sorted({key for r in results for key in r["metrics"]})
    output = {}
    for key in keys:
        cells = [r["metrics"].get(key, {"status": "missing_input", "value": None}) for r in results]
        values = [c["value"] for c in cells if c["status"] == "computed"]
        complete = len(values) == len(cells)
        statuses = {c["status"] for c in cells}
        output[key] = {"value": sum(values) / len(values) if values and complete else None,
            "status": "computed" if complete and values else next(iter(statuses)) if len(statuses) == 1 else "partial",
            "protocol_version": EVALUATION_VERSION, "n_expected": len(cells), "n_evaluated": len(values),
            "n_missing": len(cells) - len(values),
            "status_counts": {s: sum(c["status"] == s for c in cells) for s in sorted(statuses)},
            "reason": None if complete else "incomplete_population_no_silent_complete_case_mean"}
    return output


@dataclass(frozen=True)
class MetricResult:
    value: float | None
    status: str = "computed"
    protocol_version: str = EVALUATION_VERSION
    numerator: float | None = None
    denominator: float | None = None
    n_expected: int = 0
    n_evaluated: int = 0
    reason: str | None = None

    def __post_init__(self):
        if self.value is not None and not isfinite(self.value):
            raise ValueError("metric values must be finite")
        if self.status == "computed" and self.value is None:
            raise ValueError("computed metrics require a value")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "n_missing": self.n_expected - self.n_evaluated}


def ratio_result(numerator: int, denominator: int, *, missing: int = 0) -> MetricResult:
    return MetricResult(
        value=numerator / denominator if denominator and not missing else None,
        status="partial" if missing else "computed" if denominator else "not_applicable",
        numerator=numerator, denominator=denominator,
        n_expected=denominator, n_evaluated=denominator - missing,
        reason="incomplete_evaluation" if missing else "empty_denominator" if not denominator else None,
    )
