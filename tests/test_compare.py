import pytest
from pydantic import ValidationError

from regress.compare import (
    Thresholds,
    compare_runs,
    detect_drift,
    mcnemar_p,
    overall_status,
    severity_for,
)
from regress.models import CaseResult


def result(case_id: str, passed: bool, category: str = "billing") -> CaseResult:
    return CaseResult(
        case_id=case_id,
        expected_category=category,
        predicted_category=category if passed else "general",
        passed=passed,
        latency_ms=1.0,
    )


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(0.05, "ok"), (-0.03, "ok"), (-0.0334, "warning"), (-0.08, "warning"), (-0.0834, "critical")],
)
def test_severity_boundaries_are_exclusive(delta, expected):
    assert severity_for(delta, Thresholds()) == expected


def test_mcnemar_p():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(2, 0) == 0.5  # 2 flips one way: easily chance
    assert mcnemar_p(6, 0) == pytest.approx(0.03125)  # 6 flips one way: unlikely chance
    assert mcnemar_p(3, 3) == 1.0


def test_compare_uses_only_shared_cases():
    baseline = {"a": result("a", True), "b": result("b", True), "gone": result("gone", False)}
    current = {"a": result("a", True), "b": result("b", False), "new": result("new", False)}

    diff = compare_runs(baseline, current, "base", Thresholds())

    assert diff.common_cases == 2
    assert diff.added_cases == ["new"]
    assert diff.removed_cases == ["gone"]
    assert diff.regressions == ["b"]
    assert diff.improvements == []
    assert diff.pass_rate_delta == pytest.approx(-0.5)
    assert diff.per_category["billing"].current_accuracy == pytest.approx(0.5)
    assert diff.severity == "critical"


def test_compare_with_no_shared_cases_returns_none():
    assert compare_runs({"a": result("a", True)}, {"b": result("b", True)}, "x", Thresholds()) is None


def test_slow_drift_is_caught_when_no_single_run_alerts():
    thresholds = Thresholds()
    rates = [0.97 - 0.015 * i for i in range(10)]  # loses 1.5 pp per run

    per_run = [severity_for(after - before, thresholds) for before, after in zip(rates, rates[1:])]
    assert set(per_run) == {"ok"}

    drift = detect_drift(rates, thresholds)
    assert drift.triggered
    assert drift.moving_average < thresholds.drift_floor
    assert overall_status(None, drift) == "warn"


def test_drift_needs_a_full_window():
    drift = detect_drift([0.5] * 6, Thresholds())
    assert drift.moving_average is None
    assert not drift.triggered


def test_thresholds_from_env():
    t = Thresholds.from_env({"REGRESS_WARN_DELTA": "0.05", "REGRESS_CRITICAL_DELTA": ""})
    assert t.warn_delta == 0.05
    assert t.critical_delta == 0.08

    with pytest.raises(ValidationError):
        Thresholds.from_env({"REGRESS_WARN_DELTA": "0.2", "REGRESS_CRITICAL_DELTA": "0.1"})
    with pytest.raises(ValidationError):
        Thresholds.from_env({"REGRESS_DRIFT_FLOOR": "1.5"})
