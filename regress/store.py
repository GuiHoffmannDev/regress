"""SQLite run history. One row per run, one row per case per run."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import CaseResult, RunMeta

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                  TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    prompt_hash         TEXT NOT NULL,
    model               TEXT NOT NULL,
    judge_model         TEXT NOT NULL,
    dataset_version     TEXT NOT NULL,
    git_sha             TEXT,
    pass_rate           REAL NOT NULL,
    accuracy            REAL NOT NULL,
    mean_summary_score  REAL
);
CREATE TABLE IF NOT EXISTS case_results (
    run_id              TEXT NOT NULL REFERENCES runs(id),
    case_id             TEXT NOT NULL,
    expected_category   TEXT NOT NULL,
    predicted_category  TEXT,
    summary             TEXT,
    judge_score         INTEGER,
    judge_reason        TEXT,
    passed              INTEGER NOT NULL,
    latency_ms          REAL NOT NULL,
    prompt_tokens       INTEGER NOT NULL,
    completion_tokens   INTEGER NOT NULL,
    error               TEXT,
    PRIMARY KEY (run_id, case_id)
);
"""

RUN_COLUMNS = list(RunMeta.model_fields)
CASE_COLUMNS = list(CaseResult.model_fields)


def _insert(table: str, columns: list[str]) -> str:
    names = ", ".join(columns)
    params = ", ".join(f":{c}" for c in columns)
    return f"INSERT INTO {table} ({names}) VALUES ({params})"


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def save_run(self, meta: RunMeta, results: list[CaseResult]) -> None:
        with self.conn:  # One transaction: never a run without its cases.
            self.conn.execute(_insert("runs", RUN_COLUMNS), meta.model_dump(mode="json"))
            self.conn.executemany(
                _insert("case_results", ["run_id", *CASE_COLUMNS]),
                [{"run_id": meta.id, **r.model_dump(mode="json")} for r in results],
            )

    def get_run(self, run_id: str) -> RunMeta | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return RunMeta.model_validate(dict(row)) if row else None

    def previous_run(self, run_id: str) -> RunMeta | None:
        row = self.conn.execute(
            "SELECT * FROM runs WHERE rowid < (SELECT rowid FROM runs WHERE id = ?) "
            "ORDER BY rowid DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return RunMeta.model_validate(dict(row)) if row else None

    def results(self, run_id: str) -> dict[str, CaseResult]:
        rows = self.conn.execute(
            "SELECT * FROM case_results WHERE run_id = ? ORDER BY case_id", (run_id,)
        ).fetchall()
        return {
            row["case_id"]: CaseResult.model_validate({c: row[c] for c in CASE_COLUMNS})
            for row in rows
        }

    def history(self, dataset_version: str, limit: int) -> list[RunMeta]:
        """Most recent runs on one dataset version, oldest first.

        Pass rates on different dataset versions measure different bars, so they never mix.
        """
        rows = self.conn.execute(
            "SELECT * FROM runs WHERE dataset_version = ? ORDER BY rowid DESC LIMIT ?",
            (dataset_version, limit),
        ).fetchall()
        return [RunMeta.model_validate(dict(row)) for row in reversed(rows)]
