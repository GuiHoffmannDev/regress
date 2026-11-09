import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from regress.runner import JUDGE_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent


class FakeClient:
    """Offline stand-in for AsyncOpenAI.

    `answers` maps email text to (category, summary), to a raw response string,
    or to an exception instance to raise.
    """

    def __init__(self, answers: dict, judge_score: int = 5) -> None:
        self.answers = answers
        self.judge_score = judge_score
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def _create(self, *, model, messages, **kwargs):
        self.calls += 1
        if messages[0]["content"] == JUDGE_SYSTEM_PROMPT:
            content = json.dumps({"score": self.judge_score, "reason": "fake judge"})
        else:
            answer = self.answers[messages[-1]["content"]]
            if isinstance(answer, Exception):
                raise answer
            content = (
                answer
                if isinstance(answer, str)
                else json.dumps({"category": answer[0], "summary": answer[1]})
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "SLACK_WEBHOOK_URL",
        "REPORT_URL",
        "JUDGE_MODEL",
        "JUDGE_REASONING_EFFORT",
        "OPENAI_BASE_URL",
        "GITHUB_SHA",
        "REGRESS_WARN_DELTA",
        "REGRESS_CRITICAL_DELTA",
        "REGRESS_DRIFT_FLOOR",
        "REGRESS_DRIFT_WINDOW",
        "REGRESS_MIN_SUMMARY_SCORE",
        "REGRESS_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
