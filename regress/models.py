"""Typed contracts shared by every stage of the pipeline."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ClassifierOutput(BaseModel):
    """What the feature under test must return for every email."""

    category: Category
    summary: str = Field(min_length=1)


class FewShotExample(BaseModel):
    email: str
    output: ClassifierOutput


class PromptConfig(BaseModel):
    """One versioned file from /prompts. This is the "code" the pipeline runs CI against.

    The model lives here too, so swapping models is a prompt change and goes through the same gate.
    """

    version: str = Field(pattern=r"^v\d+$")
    created_at: datetime
    model: str
    temperature: float = Field(default=0.0, ge=0, le=2)
    # Sent as-is when set: "none" turns off thinking on Qwen3/Ollama; "low".."high" for OpenAI reasoning models.
    reasoning_effort: str | None = None
    system_prompt: str = Field(min_length=1)
    few_shot: list[FewShotExample] = []

    @property
    def number(self) -> int:
        return int(self.version[1:])

    @property
    def content_hash(self) -> str:
        """Detects in-place edits to a prompt file that kept the same version id."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    @classmethod
    def load(cls, path: Path) -> PromptConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def latest_prompt(directory: Path) -> Path:
    """Return the prompt file with the highest version number. Versions are append-only."""
    versions: dict[int, Path] = {}
    for path in sorted(directory.glob("*.yaml")):
        number = PromptConfig.load(path).number
        if number in versions:
            raise ValueError(f"Duplicate prompt version v{number}: {versions[number]} and {path}")
        versions[number] = path
    if not versions:
        raise FileNotFoundError(f"No prompt files (*.yaml) found in {directory}")
    return versions[max(versions)]


class ExpectedOutput(BaseModel):
    category: Category
    summary: str = Field(min_length=1)


class GoldenCase(BaseModel):
    id: str = Field(pattern=r"^case-\d{3}$")
    input: str = Field(min_length=1)
    expected: ExpectedOutput
    expected_difficulty: Difficulty
    notes: str = ""


class GoldenDataset(BaseModel):
    version: str
    updated_at: date
    review_status: str
    cases: list[GoldenCase]

    @model_validator(mode="after")
    def _unique_ids(self) -> GoldenDataset:
        ids = [case.id for case in self.cases]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate case ids: {duplicates}")
        return self

    @classmethod
    def load(cls, path: Path) -> GoldenDataset:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class CaseResult(BaseModel):
    """Every scoring dimension for one golden case in one run."""

    case_id: str
    expected_category: Category
    predicted_category: Category | None = None
    summary: str | None = None
    judge_score: int | None = None
    judge_reason: str | None = None
    passed: bool
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    @property
    def category_correct(self) -> bool:
        return self.predicted_category == self.expected_category


class RunMeta(BaseModel):
    id: str
    created_at: datetime
    prompt_version: str
    prompt_hash: str
    model: str
    judge_model: str
    dataset_version: str
    git_sha: str | None = None
    pass_rate: float
    accuracy: float
    mean_summary_score: float | None = None
