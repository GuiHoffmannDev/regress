"""Self-contained HTML diff report and the markdown PR comment."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from statistics import mean

from .compare import RunSummary, rate
from .models import CaseResult, GoldenDataset, RunMeta

STATUS_ICON = {"pass": "🟢", "warn": "🟡", "fail": "🔴"}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.1f} pp"


def headline(summary: RunSummary) -> str:
    """One sentence with the numbers that matter. Shared by Slack, the PR comment and the report."""
    parts: list[str] = []
    diff = summary.diff
    if diff is None:
        parts.append(f"No baseline to compare against; pass rate {pct(summary.run.pass_rate)}.")
    else:
        n = len(diff.regressions)
        before, after = pct(diff.baseline_pass_rate), pct(diff.current_pass_rate)
        if diff.pass_rate_delta < 0:
            movement = f"dropped from {before} to {after}"
        elif diff.pass_rate_delta > 0:
            movement = f"rose from {before} to {after}"
        else:
            movement = f"held at {after}"
        sentence = f"{n} regression{'' if n == 1 else 's'} detected, pass rate {movement}"
        if diff.improvements:
            sentence += f" ({len(diff.improvements)} improved)"
        parts.append(sentence + ".")
    drift = summary.drift
    if drift.triggered and drift.moving_average is not None:
        parts.append(
            f"Slow drift: the {drift.window}-run average ({pct(drift.moving_average)}) "
            f"is below the {pct(drift.floor)} floor."
        )
    return " ".join(parts)


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; good enough for a few dozen latency samples."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))]


def stats(results: list[CaseResult]) -> dict[str, float | None]:
    scores = [r.judge_score for r in results if r.judge_score is not None]
    latencies = [r.latency_ms for r in results if r.error is None]
    return {
        "Pass rate": rate(r.passed for r in results),
        "Category accuracy": rate(r.category_correct for r in results),
        "Mean summary score": mean(scores) if scores else None,
        "p50 latency (ms)": _percentile(latencies, 0.5),
        "p95 latency (ms)": _percentile(latencies, 0.95),
        "Total tokens": float(sum(r.prompt_tokens + r.completion_tokens for r in results)),
        "Errors": float(sum(r.error is not None for r in results)),
    }


# Metric name -> (formatter, higher is better)
METRIC_FORMAT = {
    "Pass rate": (pct, True),
    "Category accuracy": (pct, True),
    "Mean summary score": (lambda v: f"{v:.2f} / 5", True),
    "p50 latency (ms)": (lambda v: f"{v:.0f}", False),
    "p95 latency (ms)": (lambda v: f"{v:.0f}", False),
    "Total tokens": (lambda v: f"{v:,.0f}", False),
    "Errors": (lambda v: f"{v:.0f}", False),
}


def _fmt(metric: str, value: float | None) -> str:
    return "—" if value is None else METRIC_FORMAT[metric][0](value)


def _delta(metric: str, before: float | None, after: float | None) -> tuple[str, str]:
    if before is None or after is None:
        return "—", ""
    change = after - before
    if metric in ("Pass rate", "Category accuracy"):
        text = pp(change)
    elif metric == "Mean summary score":
        text = f"{change:+.2f}"
    else:
        text = f"{change:+,.0f}"
    if abs(change) < 1e-9:
        return text, ""
    better = (change > 0) == METRIC_FORMAT[metric][1]
    return text, "good" if better else "bad"


def _scorecard(
    current: Mapping[str, CaseResult], baseline: Mapping[str, CaseResult] | None
) -> tuple[dict, dict | None]:
    """Compare on shared cases only, so dataset changes do not masquerade as regressions."""
    if baseline is None:
        return stats(list(current.values())), None
    common = sorted(current.keys() & baseline.keys())
    return stats([current[c] for c in common]), stats([baseline[c] for c in common])


def _output_cell(result: CaseResult | None) -> str:
    if result is None:
        return "<td class='muted'>not run</td>"
    if result.predicted_category is None:
        return f"<td><span class='tag bad'>error</span><div class='err'>{escape(result.error or '')}</div></td>"
    score = f"{result.judge_score}/5" if result.judge_score is not None else "no score"
    status = "good" if result.passed else "bad"
    reason = f"<div class='muted'>{escape(result.judge_reason or result.error or '')}</div>"
    return (
        f"<td><span class='tag {status}'>{escape(result.predicted_category.value)} · {score}</span>"
        f"<div>{escape(result.summary or '')}</div>{reason}</td>"
    )


def _cases_table(
    ids: list[str],
    dataset: GoldenDataset,
    baseline: Mapping[str, CaseResult],
    current: Mapping[str, CaseResult],
) -> str:
    if not ids:
        return "<p class='muted'>None.</p>"
    cases = {c.id: c for c in dataset.cases}
    rows = []
    for case_id in ids:
        case = cases.get(case_id)
        email = escape(case.input) if case else "<span class='muted'>removed from dataset</span>"
        expected = (
            f"<span class='tag'>{escape(case.expected.category.value)}</span>"
            f"<div>{escape(case.expected.summary)}</div>"
            f"<div class='muted'>{escape(case.expected_difficulty.value)}"
            f"{' · ' + escape(case.notes) if case.notes else ''}</div>"
            if case
            else "—"
        )
        rows.append(
            f"<tr><td><code>{escape(case_id)}</code></td><td class='email'>{email}</td>"
            f"<td>{expected}</td>{_output_cell(baseline.get(case_id))}{_output_cell(current.get(case_id))}</tr>"
        )
    return (
        "<table class='cases'><thead><tr><th>Case</th><th>Email</th><th>Expected</th>"
        "<th>Baseline output</th><th>This run</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def trend_svg(history: list[RunMeta], window: int, floor: float) -> str:
    if len(history) < 2:
        return "<p class='muted'>Not enough runs on this dataset version for a trend yet.</p>"
    width, height, pad = 760, 240, 40
    rates = [r.pass_rate for r in history]
    low = max(0.0, min(min(rates), floor) - 0.05)
    span = 1.0 - low or 1.0

    def x(i: int) -> float:
        return pad + i * (width - 2 * pad) / (len(history) - 1)

    def y(value: float) -> float:
        return height - pad - (value - low) / span * (height - 2 * pad)

    line = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(rates))
    moving = [
        (i, mean(rates[i - window + 1 : i + 1])) for i in range(window - 1, len(rates))
    ]
    moving_line = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in moving)
    points = "".join(
        f"<circle cx='{x(i):.1f}' cy='{y(r.pass_rate):.1f}' r='4' class='pt'>"
        f"<title>{escape(r.id)} · {escape(r.prompt_version)} · {escape(r.model)} · {pct(r.pass_rate)}</title></circle>"
        f"<text x='{x(i):.1f}' y='{height - pad + 16}' class='axis' text-anchor='middle'>{escape(r.prompt_version)}</text>"
        for i, r in enumerate(history)
    )
    return f"""<svg viewBox='0 0 {width} {height}' role='img' aria-label='Pass rate over the last {len(history)} runs'>
<line x1='{pad}' x2='{width - pad}' y1='{y(floor):.1f}' y2='{y(floor):.1f}' class='floor'/>
<text x='{width - pad}' y='{y(floor) - 6:.1f}' class='axis' text-anchor='end'>drift floor {pct(floor)}</text>
<text x='{pad - 6}' y='{y(1.0) + 4:.1f}' class='axis' text-anchor='end'>100%</text>
<text x='{pad - 6}' y='{y(low) + 4:.1f}' class='axis' text-anchor='end'>{pct(low)}</text>
{f"<polyline points='{moving_line}' class='ma'/>" if moving else ""}
<polyline points='{line}' class='line'/>
{points}
</svg>
<p class='muted'>Solid: pass rate per run. Dashed: {window}-run moving average.</p>"""


CSS = """
:root { --bg:#fff; --fg:#1d2026; --muted:#6b7280; --line:#e5e7eb; --card:#f8f9fb;
        --good:#137a3f; --good-bg:#e3f5ea; --bad:#b42318; --bad-bg:#fdeceb; --warn:#9a6700; --accent:#2f5bd3; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14161a; --fg:#e6e8eb; --muted:#9aa1ab; --line:#2a2e35; --card:#1b1e23;
          --good:#5cc98a; --good-bg:#16301f; --bad:#f2837a; --bad-bg:#3a1c1a; --warn:#e3b341; --accent:#7ea2ff; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,sans-serif; }
main { max-width:1200px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:32px 0 8px; }
.muted { color:var(--muted); font-size:12px; }
.status { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:12px; }
.status.pass { background:var(--good-bg); color:var(--good); }
.status.warn { background:#fff4d6; color:var(--warn); }
.status.fail { background:var(--bad-bg); color:var(--bad); }
.headline { font-size:16px; margin:12px 0 0; }
table { border-collapse:collapse; width:100%; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-size:12px; color:var(--muted); font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.good { color:var(--good); } .bad { color:var(--bad); }
.tag { display:inline-block; font-size:12px; padding:0 6px; border-radius:4px; background:var(--card); border:1px solid var(--line); margin-bottom:4px; }
.tag.good { background:var(--good-bg); border-color:transparent; } .tag.bad { background:var(--bad-bg); border-color:transparent; }
.err { color:var(--bad); font-family:ui-monospace,monospace; font-size:12px; white-space:pre-wrap; overflow-wrap:anywhere; }
.cases td { width:22%; } .cases td:first-child { width:6%; white-space:nowrap; } .email { white-space:pre-wrap; }
.meta td:first-child { color:var(--muted); width:180px; }
.scroll { overflow-x:auto; }
svg { width:100%; height:auto; }
svg .line { fill:none; stroke:var(--accent); stroke-width:2; }
svg .ma { fill:none; stroke:var(--muted); stroke-width:2; stroke-dasharray:6 4; }
svg .floor { stroke:var(--bad); stroke-dasharray:3 3; }
svg .pt { fill:var(--accent); } svg .axis { fill:var(--muted); font-size:11px; }
"""


def render_html(
    summary: RunSummary,
    current: Mapping[str, CaseResult],
    baseline: Mapping[str, CaseResult] | None,
    dataset: GoldenDataset,
    history: list[RunMeta],
) -> str:
    run, diff = summary.run, summary.diff
    now, before = _scorecard(current, baseline)

    meta_rows = [
        ("Run", run.id),
        ("Created", run.created_at.isoformat(timespec="seconds")),
        ("Prompt", f"{run.prompt_version} (hash {run.prompt_hash})"),
        ("Model", run.model),
        ("Judge model", run.judge_model),
        ("Dataset", f"{run.dataset_version} · {len(current)} cases"),
        ("Git SHA", run.git_sha or "—"),
        (
            "Baseline",
            f"{summary.baseline.id} · {summary.baseline.prompt_version} · {summary.baseline.model}"
            if summary.baseline
            else "none (first run)",
        ),
    ]
    if diff and (diff.added_cases or diff.removed_cases):
        meta_rows.append(
            (
                "Dataset changes",
                f"compared on {diff.common_cases} shared cases; "
                f"{len(diff.added_cases)} added, {len(diff.removed_cases)} removed",
            )
        )
    meta = "".join(f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in meta_rows)

    score_rows = []
    for metric, value in now.items():
        old = before[metric] if before else None
        delta_text, delta_class = _delta(metric, old, value)
        score_rows.append(
            f"<tr><td>{metric}</td><td class='num'>{_fmt(metric, old)}</td>"
            f"<td class='num'>{_fmt(metric, value)}</td><td class='num {delta_class}'>{delta_text}</td></tr>"
        )

    category_section = ""
    if diff:
        category_rows = "".join(
            f"<tr><td>{escape(name)}</td><td class='num'>{d.cases}</td>"
            f"<td class='num'>{pct(d.baseline_accuracy)}</td><td class='num'>{pct(d.current_accuracy)}</td>"
            f"<td class='num {'bad' if d.delta < 0 else 'good' if d.delta > 0 else ''}'>{pp(d.delta)}</td></tr>"
            for name, d in diff.per_category.items()
        )
        category_section = (
            "<h2>Per-category accuracy</h2><div class='scroll'><table><thead><tr><th>Category</th>"
            "<th class='num'>Cases</th><th class='num'>Baseline</th><th class='num'>This run</th>"
            f"<th class='num'>Δ</th></tr></thead><tbody>{category_rows}</tbody></table></div>"
            "<p class='muted'>Informational. With ~15 cases per category one flip moves accuracy "
            "by ~7 pp, so only the overall pass rate gates the build.</p>"
        )

    t = summary.thresholds
    significance = (
        f"Exact McNemar p-value on flips: {diff.flip_p_value:.3f} "
        f"({'unlikely to be noise' if diff.flip_p_value < 0.05 else 'could be noise'}). "
        if diff
        else ""
    )
    gating = (
        f"{significance}Warning when pass rate drops more than {pp(t.warn_delta)[1:]}, "
        f"critical above {pp(t.critical_delta)[1:]}. A case passes when the category matches "
        f"and the summary scores at least {t.min_summary_score}/5."
    )

    empty: dict[str, CaseResult] = {}
    failing_now = [c for c in sorted(current) if not current[c].passed]
    cases_html = ""
    if diff:
        cases_html += f"<h2>Regressions ({len(diff.regressions)})</h2><div class='scroll'>"
        cases_html += _cases_table(diff.regressions, dataset, baseline or empty, current) + "</div>"
        cases_html += f"<h2>Improvements ({len(diff.improvements)})</h2><div class='scroll'>"
        cases_html += _cases_table(diff.improvements, dataset, baseline or empty, current) + "</div>"
    cases_html += f"<h2>All failing cases in this run ({len(failing_now)})</h2><div class='scroll'>"
    cases_html += _cases_table(failing_now, dataset, baseline or empty, current) + "</div>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval report {escape(run.prompt_version)}</title><style>{CSS}</style></head>
<body><main>
<h1>Eval report · {escape(run.prompt_version)} · {escape(run.model)}</h1>
<span class="status {summary.status}">{summary.status.upper()}</span>
<p class="headline">{escape(headline(summary))}</p>
<h2>Run metadata</h2><table class="meta">{meta}</table>
<h2>Scorecard</h2><div class="scroll"><table><thead><tr><th>Metric</th><th class="num">Baseline</th>
<th class="num">This run</th><th class="num">Δ</th></tr></thead><tbody>{''.join(score_rows)}</tbody></table></div>
<p class="muted">{escape(gating)}</p>
{category_section}
<h2>Trend (last {len(history)} runs on dataset {escape(run.dataset_version)})</h2>
{trend_svg(history, t.drift_window, t.drift_floor)}
{cases_html}
</main></body></html>
"""


def render_comment(summary: RunSummary, current: Mapping[str, CaseResult]) -> str:
    run, diff = summary.run, summary.diff
    lines = [
        f"### {STATUS_ICON[summary.status]} Prompt eval: {summary.status.upper()}",
        "",
        f"**{headline(summary)}**",
        "",
        f"Prompt `{run.prompt_version}` · model `{run.model}` · dataset `{run.dataset_version}` · run `{run.id}`",
        "",
    ]
    if diff:
        lines += [
            "| | Baseline | This run | Δ |",
            "|---|---:|---:|---:|",
            f"| Pass rate | {pct(diff.baseline_pass_rate)} | {pct(diff.current_pass_rate)} | {pp(diff.pass_rate_delta)} |",
        ]
        for name, d in diff.per_category.items():
            lines.append(
                f"| {name} accuracy | {pct(d.baseline_accuracy)} | {pct(d.current_accuracy)} | {pp(d.delta)} |"
            )
        lines.append("")
        if diff.regressions:
            lines.append("<details><summary>Regressed cases</summary>\n")
            for case_id in diff.regressions:
                r = current[case_id]
                got = r.predicted_category.value if r.predicted_category else "error"
                lines.append(
                    f"- `{case_id}`: expected `{r.expected_category.value}`, got `{got}`, "
                    f"summary score {r.judge_score if r.judge_score is not None else '—'}"
                )
            lines.append("\n</details>\n")
        lines.append(f"Flip p-value (McNemar): {diff.flip_p_value:.3f}.")
    t = summary.thresholds
    lines.append(
        f"Gates: warn > {t.warn_delta * 100:.1f} pp drop, critical > {t.critical_delta * 100:.1f} pp drop "
        f"(critical blocks merge); drift floor {pct(t.drift_floor)} over {t.drift_window} runs."
    )
    return "\n".join(lines) + "\n"
