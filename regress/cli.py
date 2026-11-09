"""Command line entry point: `regress run` and `regress notify`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from openai import AsyncOpenAI

from . import slack
from .compare import RunSummary, Thresholds, compare_runs, detect_drift, overall_status, rate
from .models import CaseResult, GoldenDataset, PromptConfig, RunMeta, latest_prompt
from .report import headline, render_comment, render_html
from .runner import JudgeConfig, run_eval
from .store import Store

TREND_RUNS = 20


def _int_env(name: str, default: int, low: int, high: int) -> int:
    value = int(os.environ.get(name) or default)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
    return value


async def _evaluate(prompt, dataset, judge, min_score, concurrency) -> list[CaseResult]:
    # Any OpenAI-compatible server works (Ollama: http://localhost:11434/v1).
    # `or None` so an empty env var (common in CI) falls back to the SDK default.
    async with AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None) as client:
        return await run_eval(client, prompt, dataset, judge, min_score, concurrency)


def notify(summary: RunSummary, report_url: str | None) -> bool:
    """Send a Slack alert for warn/fail runs. Returns False only when sending failed."""
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print("SLACK_WEBHOOK_URL not set; skipping Slack alert.")
        return True
    if summary.status == "pass":
        print("Status is pass; no Slack alert sent.")
        return True
    try:
        slack.send(webhook, slack.build_payload(summary, report_url))
    except Exception as exc:  # An alerting outage must not hide the eval result.
        print(f"Slack alert failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False
    print("Slack alert sent.")
    return True


def cmd_run(args: argparse.Namespace) -> int:
    thresholds = Thresholds.from_env()
    judge = JudgeConfig(
        model=os.environ.get("JUDGE_MODEL") or "gpt-4o",
        reasoning_effort=os.environ.get("JUDGE_REASONING_EFFORT") or None,
    )
    concurrency = _int_env("REGRESS_CONCURRENCY", 8, 1, 64)
    prompt_path = args.prompt or latest_prompt(args.prompts_dir)
    prompt = PromptConfig.load(prompt_path)
    dataset = GoldenDataset.load(args.dataset)

    store = Store(args.db)
    if args.baseline and store.get_run(args.baseline) is None:
        print(f"Baseline run {args.baseline!r} not found in {args.db}", file=sys.stderr)
        return 2

    print(
        f"Evaluating {len(dataset.cases)} cases: prompt {prompt.version} ({prompt_path}), "
        f"model {prompt.model}, judge {judge.label}, dataset {dataset.version}"
    )
    results = asyncio.run(
        _evaluate(prompt, dataset, judge, thresholds.min_summary_score, concurrency)
    )

    now = datetime.now(timezone.utc)
    scores = [r.judge_score for r in results if r.judge_score is not None]
    meta = RunMeta(
        id=f"{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
        created_at=now,
        prompt_version=prompt.version,
        prompt_hash=prompt.content_hash,
        model=prompt.model,
        judge_model=judge.label,
        dataset_version=dataset.version,
        git_sha=os.environ.get("GITHUB_SHA"),
        pass_rate=rate(r.passed for r in results),
        accuracy=rate(r.category_correct for r in results),
        mean_summary_score=mean(scores) if scores else None,
    )
    store.save_run(meta, results)

    baseline = store.get_run(args.baseline) if args.baseline else store.previous_run(meta.id)
    current = {r.case_id: r for r in results}
    baseline_results = store.results(baseline.id) if baseline else None
    diff = (
        compare_runs(baseline_results, current, baseline.id, thresholds)
        if baseline and baseline_results
        else None
    )
    history = store.history(dataset.version, max(TREND_RUNS, thresholds.drift_window))
    drift = detect_drift([r.pass_rate for r in history], thresholds)
    summary = RunSummary(
        status=overall_status(diff, drift),
        run=meta,
        baseline=baseline if diff else None,
        diff=diff,
        drift=drift,
        thresholds=thresholds,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    (args.out / "comment.md").write_text(render_comment(summary, current), encoding="utf-8")
    (args.out / "report.html").write_text(
        render_html(summary, current, baseline_results if diff else None, dataset, history[-TREND_RUNS:]),
        encoding="utf-8",
    )
    print(f"[{summary.status.upper()}] {headline(summary)}")
    print(f"Report: {args.out / 'report.html'}")

    if not args.no_notify:
        notify(summary, os.environ.get("REPORT_URL"))
    return 1 if summary.status == "fail" else 0


def cmd_notify(args: argparse.Namespace) -> int:
    summary = RunSummary.model_validate_json(args.summary.read_text(encoding="utf-8"))
    return 0 if notify(summary, args.report_url) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regress", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Evaluate a prompt against the golden dataset and diff it.")
    run.add_argument("--prompt", type=Path, help="Prompt YAML (default: highest version in --prompts-dir)")
    run.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    run.add_argument("--dataset", type=Path, default=Path("golden/dataset.json"))
    run.add_argument("--db", type=Path, default=Path(".regress/history.db"))
    run.add_argument("--out", type=Path, default=Path("out"))
    run.add_argument("--baseline", help="Run id to compare against (default: the previous run)")
    run.add_argument("--no-notify", action="store_true", help="Skip the Slack alert")
    run.set_defaults(handler=cmd_run)

    notify_cmd = sub.add_parser("notify", help="Send the Slack alert for a finished run.")
    notify_cmd.add_argument("--summary", type=Path, default=Path("out/summary.json"))
    notify_cmd.add_argument("--report-url", default=os.environ.get("REPORT_URL"))
    notify_cmd.set_defaults(handler=cmd_notify)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
