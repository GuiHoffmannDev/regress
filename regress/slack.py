"""Slack incoming-webhook alerts."""

from __future__ import annotations

import json
import urllib.request

from .compare import RunSummary
from .report import headline

ICON = {"pass": ":white_check_mark:", "warn": ":warning:", "fail": ":rotating_light:"}


def _escape(text: str) -> str:
    """Slack mrkdwn treats &, < and > as control characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_payload(summary: RunSummary, report_url: str | None) -> dict:
    run = summary.run
    title = f"{ICON[summary.status]} Prompt eval {summary.status.upper()}: {run.prompt_version} on {run.model}"
    blocks: list[dict] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{_escape(title)}*\n{_escape(headline(summary))}"},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": _escape(
                        f"run {run.id} · dataset {run.dataset_version} · "
                        f"baseline {summary.baseline.id if summary.baseline else 'none'}"
                        + (f" · commit {run.git_sha[:8]}" if run.git_sha else "")
                    ),
                }
            ],
        },
    ]
    if report_url and report_url.startswith(("https://", "http://")):
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Open diff report"},
                        "url": report_url,
                    }
                ],
            }
        )
    return {"text": f"{title}: {headline(summary)}", "blocks": blocks}


def send(webhook_url: str, payload: dict, timeout: float = 10) -> None:
    # urllib also opens file:// and ftp:// URLs; only allow HTTPS webhooks.
    if not webhook_url.startswith("https://"):
        raise ValueError("SLACK_WEBHOOK_URL must be an https:// URL")
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout):  # Raises HTTPError on non-2xx.
        pass
