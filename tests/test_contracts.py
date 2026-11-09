from collections import Counter

from regress.models import Category, Difficulty, GoldenDataset, PromptConfig, latest_prompt

from .conftest import ROOT

DATASET = GoldenDataset.load(ROOT / "golden" / "dataset.json")


def test_golden_dataset_shape():
    assert 50 <= len(DATASET.cases) <= 100
    per_category = Counter(c.expected.category for c in DATASET.cases)
    assert all(per_category[category] >= 10 for category in Category)
    assert {c.expected_difficulty for c in DATASET.cases} == set(Difficulty)


def test_few_shot_examples_do_not_leak_into_the_dataset():
    inputs = {c.input.strip().lower() for c in DATASET.cases}
    for path in (ROOT / "prompts").glob("*.yaml"):
        for example in PromptConfig.load(path).few_shot:
            assert example.email.strip().lower() not in inputs, f"{path.name} leaks a golden case"


def test_latest_prompt_picks_highest_version(tmp_path):
    source = (ROOT / "prompts" / "v1.yaml").read_text()
    (tmp_path / "v1.yaml").write_text(source)
    (tmp_path / "v10.yaml").write_text(source.replace("version: v1", "version: v10"))
    (tmp_path / "v2.yaml").write_text(source.replace("version: v1", "version: v2"))

    assert latest_prompt(tmp_path).name == "v10.yaml"
