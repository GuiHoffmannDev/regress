import json

import httpx
import openai
import pytest

from regress import cli, slack
from regress.models import GoldenDataset
from regress.store import Store

from .conftest import ROOT, FakeClient

DATASET = GoldenDataset.load(ROOT / "golden" / "dataset.json")


def perfect_answers() -> dict:
    return {c.input: (c.expected.category.value, c.expected.summary) for c in DATASET.cases}


def wrong_category(case) -> str:
    return "general" if case.expected.category.value != "general" else "billing"


@pytest.fixture
def run(monkeypatch, tmp_path):
    def _run(answers: dict, *extra: str) -> tuple[int, FakeClient]:
        client = FakeClient(answers)
        monkeypatch.setattr(cli, "AsyncOpenAI", lambda **kwargs: client)
        code = cli.main(
            [
                "run",
                "--prompts-dir", str(ROOT / "prompts"),
                "--dataset", str(ROOT / "golden" / "dataset.json"),
                "--db", str(tmp_path / "history.db"),
                "--out", str(tmp_path / "out"),
                *extra,
            ]
        )
        return code, client

    return _run


def summary(tmp_path) -> dict:
    return json.loads((tmp_path / "out" / "summary.json").read_text())


def test_first_run_passes_without_baseline(run, tmp_path):
    code, client = run(perfect_answers())

    assert code == 0
    assert client.calls == 2 * len(DATASET.cases)  # classifier + judge per case
    data = summary(tmp_path)
    assert data["status"] == "pass"
    assert data["diff"] is None
    assert data["run"]["pass_rate"] == 1.0
    assert (tmp_path / "out" / "report.html").exists()


def test_critical_regression_fails_and_alerts(run, tmp_path, monkeypatch):
    run(perfect_answers())

    broken = perfect_answers()
    flipped = DATASET.cases[:6]  # 6/60 = 10 pp drop, above the 8 pp critical threshold
    for case in flipped:
        broken[case.input] = (wrong_category(case), case.expected.summary)

    sent = []
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T/B/X")
    monkeypatch.setenv("REPORT_URL", "https://example.com/report")
    monkeypatch.setattr(slack, "send", lambda url, payload: sent.append(payload))

    code, _ = run(broken)

    assert code == 1
    data = summary(tmp_path)
    assert data["status"] == "fail"
    assert data["diff"]["severity"] == "critical"
    assert data["diff"]["regressions"] == [c.id for c in flipped]
    assert data["diff"]["pass_rate_delta"] == pytest.approx(-0.1)

    report = (tmp_path / "out" / "report.html").read_text()
    comment = (tmp_path / "out" / "comment.md").read_text()
    for case in flipped:
        assert case.id in report
        assert case.id in comment
    assert "6 regressions detected, pass rate dropped from 100.0% to 90.0%" in comment

    assert len(sent) == 1
    assert "6 regressions detected" in sent[0]["text"]
    assert sent[0]["blocks"][-1]["elements"][0]["url"] == "https://example.com/report"


def test_small_regression_warns_without_blocking(run, tmp_path):
    run(perfect_answers())
    broken = perfect_answers()
    for case in DATASET.cases[:2]:  # 3.3 pp: above warn (3), below critical (8)
        broken[case.input] = (wrong_category(case), case.expected.summary)

    code, _ = run(broken)

    assert code == 0
    assert summary(tmp_path)["status"] == "warn"


def test_malformed_output_fails_only_that_case(run, tmp_path):
    answers = perfect_answers()
    bad_case = DATASET.cases[10]
    answers[bad_case.input] = "this is not json"

    code, _ = run(answers)

    assert code == 0
    run_id = summary(tmp_path)["run"]["id"]
    results = Store(tmp_path / "history.db").results(run_id)
    assert results[bad_case.id].passed is False
    assert "ValidationError" in results[bad_case.id].error
    assert sum(r.passed for r in results.values()) == len(DATASET.cases) - 1


def test_auth_error_aborts_the_run(run, tmp_path):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = openai.AuthenticationError(
        "bad key", response=httpx.Response(401, request=request), body=None
    )
    answers = {c.input: error for c in DATASET.cases}

    with pytest.raises(openai.AuthenticationError):
        run(answers)
    assert not (tmp_path / "out" / "summary.json").exists()


def test_unknown_baseline_exits_before_any_llm_call(run):
    code, client = run(perfect_answers(), "--baseline", "does-not-exist")

    assert code == 2
    assert client.calls == 0


def test_notify_command_reads_summary(run, tmp_path, monkeypatch):
    run(perfect_answers())
    broken = perfect_answers()
    for case in DATASET.cases[:6]:
        broken[case.input] = (wrong_category(case), case.expected.summary)
    run(broken, "--no-notify")

    sent = []
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T/B/X")
    monkeypatch.setattr(slack, "send", lambda url, payload: sent.append(payload))

    code = cli.main(["notify", "--summary", str(tmp_path / "out" / "summary.json")])

    assert code == 0
    assert len(sent) == 1
