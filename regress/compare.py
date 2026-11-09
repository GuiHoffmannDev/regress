"""Diffing, thresholds and drift detection. Pure functions: no I/O, no LLM calls."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .models import CaseResult, Category, RunMeta

Severity = Literal["ok", "warning", "critical"]
Status = Literal["pass", "warn", "fail"]


class Thresholds(BaseModel):
    """All deltas are absolute pass-rate changes (0.03 = 3 percentage points)."""

    warn_delta: float = Field(default=0.03, ge=0, le=1)
    critical_delta: float = Field(default=0.08, ge=0, le=1)
    drift_floor: float = Field(default=0.90, ge=0, le=1)
    drift_window: int = Field(default=7, ge=2)
    min_summary_score: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def _ordered(self) -> Thresholds:
        if self.warn_delta > self.critical_delta:
            raise ValueError("warn_delta must not exceed critical_delta")
        return self

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Thresholds:
        names = {
            "warn_delta": "REGRESS_WARN_DELTA",
            "critical_delta": "REGRESS_CRITICAL_DELTA",
            "drift_floor": "REGRESS_DRIFT_FLOOR",
            "drift_window": "REGRESS_DRIFT_WINDOW",
            "min_summary_score": "REGRESS_MIN_SUMMARY_SCORE",
        }
        return cls.model_validate({field: env[var] for field, var in names.items() if env.get(var)})


class CategoryDelta(BaseModel):
    cases: int
    baseline_accuracy: float
    current_accuracy: float
    delta: float


class Diff(BaseModel):
    baseline_run_id: str
    common_cases: int
    added_cases: list[str]
    removed_cases: list[str]
    baseline_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float
    per_category: dict[str, CategoryDelta]
    regressions: list[str]
    improvements: list[str]
    flip_p_value: float
    severity: Severity


class Drift(BaseModel):
    window: int
    floor: float
    runs_available: int
    moving_average: float | None
    triggered: bool


class RunSummary(BaseModel):
    """Everything downstream consumers (report, PR comment, Slack) need. Written to summary.json."""

    status: Status
    run: RunMeta
    baseline: RunMeta | None
    diff: Diff | None
    drift: Drift
    thresholds: Thresholds


def rate(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def mcnemar_p(regressions: int, improvements: int) -> float:
    """Exact two-sided McNemar test on flipped cases.

    Answers "could this many flips in this direction be noise?". Informational only:
    gating uses the configurable thresholds.
    """
    n = regressions + improvements
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(regressions, improvements) + 1)) / 2**n
    return min(1.0, 2 * tail)


def severity_for(delta: float, thresholds: Thresholds) -> Severity:
    drop = round(-delta, 9)  # Avoid float noise turning 0.08 into 0.0800000001.
    if drop > thresholds.critical_delta:
        return "critical"
    if drop > thresholds.warn_delta:
        return "warning"
    return "ok"


def compare_runs(
    baseline: Mapping[str, CaseResult],
    current: Mapping[str, CaseResult],
    baseline_run_id: str,
    thresholds: Thresholds,
) -> Diff | None:
    """Diff two runs on the cases they share. Returns None when they share no cases."""
    common = sorted(baseline.keys() & current.keys())
    if not common:
        return None

    before_rate = rate(baseline[c].passed for c in common)
    after_rate = rate(current[c].passed for c in common)
    delta = after_rate - before_rate

    per_category: dict[str, CategoryDelta] = {}
    for category in Category:
        ids = [c for c in common if current[c].expected_category == category]
        if not ids:
            continue
        before = rate(baseline[c].category_correct for c in ids)
        after = rate(current[c].category_correct for c in ids)
        per_category[category.value] = CategoryDelta(
            cases=len(ids), baseline_accuracy=before, current_accuracy=after, delta=after - before
        )

    regressions = [c for c in common if baseline[c].passed and not current[c].passed]
    improvements = [c for c in common if not baseline[c].passed and current[c].passed]
    return Diff(
        baseline_run_id=baseline_run_id,
        common_cases=len(common),
        added_cases=sorted(current.keys() - baseline.keys()),
        removed_cases=sorted(baseline.keys() - current.keys()),
        baseline_pass_rate=before_rate,
        current_pass_rate=after_rate,
        pass_rate_delta=delta,
        per_category=per_category,
        regressions=regressions,
        improvements=improvements,
        flip_p_value=mcnemar_p(len(regressions), len(improvements)),
        severity=severity_for(delta, thresholds),
    )


def detect_drift(pass_rates: list[float], thresholds: Thresholds) -> Drift:
    """Level check on the moving average, independent of per-run deltas.

    Per-run checks compare against the previous run, so a slow slide re-baselines itself
    every time and never alerts. An absolute floor on the moving average does not.
    """
    window = thresholds.drift_window
    average = mean(pass_rates[-window:]) if len(pass_rates) >= window else None
    return Drift(
        window=window,
        floor=thresholds.drift_floor,
        runs_available=len(pass_rates),
        moving_average=average,
        triggered=average is not None and average < thresholds.drift_floor,
    )


def overall_status(diff: Diff | None, drift: Drift) -> Status:
    if diff and diff.severity == "critical":
        return "fail"
    if (diff and diff.severity == "warning") or drift.triggered:
        return "warn"
    return "pass"
