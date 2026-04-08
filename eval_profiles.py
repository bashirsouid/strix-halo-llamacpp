from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Curated fast HumanEval+ subset.
#
# Rationale:
# - 48 tasks is ~29% of the full HumanEval+ task count (164), which keeps runtime
#   short while still spanning a good mix of difficulty/discrimination buckets.
# - The IDs were selected from the repository's checked-in full-run task results so
#   the quick profile preserves the observed ordering between strong, okay, and weak
#   coding models while avoiding non-informative "everybody passes" / "nobody passes"
#   tasks.
HUMANEVAL_QUICK_V1_TASK_IDS: tuple[str, ...] = (
    # qwen-only / very hard
    "HumanEval/10",
    "HumanEval/20",
    "HumanEval/38",
    "HumanEval/75",
    "HumanEval/109",
    # qwen + qwen3.5
    "HumanEval/41",
    "HumanEval/50",
    "HumanEval/100",
    "HumanEval/125",
    "HumanEval/140",
    # nemotron-q8 + qwen
    "HumanEval/21",
    "HumanEval/57",
    "HumanEval/67",
    "HumanEval/94",
    "HumanEval/120",
    "HumanEval/146",
    # nemotron-q4 + qwen
    "HumanEval/46",
    "HumanEval/89",
    "HumanEval/105",
    "HumanEval/142",
    "HumanEval/154",
    "HumanEval/158",
    # nemotron-only edge cases
    "HumanEval/55",
    "HumanEval/160",
    # all but weaker models
    "HumanEval/8",
    "HumanEval/36",
    "HumanEval/49",
    "HumanEval/101",
    "HumanEval/149",
    "HumanEval/19",
    "HumanEval/90",
    "HumanEval/62",
    # solid mid-tier discriminators
    "HumanEval/0",
    "HumanEval/6",
    "HumanEval/17",
    "HumanEval/25",
    "HumanEval/31",
    "HumanEval/43",
    "HumanEval/63",
    "HumanEval/68",
    # nearly-easy sanity checks (still informative)
    "HumanEval/48",
    "HumanEval/1",
    "HumanEval/4",
    "HumanEval/9",
    "HumanEval/22",
    "HumanEval/44",
    "HumanEval/56",
    "HumanEval/69",
)


@dataclass(frozen=True)
class EvalProfile:
    requested: str
    name: str
    suite: str
    description: str
    use_mini: bool = False
    task_ids: tuple[str, ...] | None = None

    @property
    def task_count(self) -> int | None:
        return None if self.task_ids is None else len(self.task_ids)

    @property
    def is_custom_subset(self) -> bool:
        return bool(self.task_ids)


def resolve_eval_profile(profile: str, suite: str) -> EvalProfile:
    normalized = (profile or "full").strip().lower()
    if normalized == "full":
        return EvalProfile(
            requested=normalized,
            name="full",
            suite=suite,
            description="Full EvalPlus dataset",
        )
    if normalized == "mini":
        return EvalProfile(
            requested=normalized,
            name="mini",
            suite=suite,
            description="EvalPlus mini split",
            use_mini=True,
        )
    if normalized == "quick":
        if suite != "humaneval":
            raise ValueError(
                "The quick eval profile is only supported for humaneval. "
                "Use --profile mini or --profile full for mbpp."
            )
        return EvalProfile(
            requested=normalized,
            name="quick-v1",
            suite=suite,
            description="48-task curated HumanEval+ subset for fast code-authoring comparisons",
            task_ids=HUMANEVAL_QUICK_V1_TASK_IDS,
        )
    raise ValueError(f"Unknown eval profile: {profile}")


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_override_dataset(profile: EvalProfile, output_dir: Path) -> Path | None:
    """Materialize a stable EvalPlus override dataset for custom quick profiles.

    Returns the generated dataset path, or None when the profile uses the stock
    EvalPlus dataset.
    """
    if not profile.is_custom_subset:
        return None

    if profile.suite != "humaneval":
        raise ValueError(f"Custom subset generation is not implemented for {profile.suite}")

    try:
        from evalplus.data import get_human_eval_plus
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is required to build the quick evaluation dataset. "
            "Activate the project venv or install evalplus first."
        ) from exc

    dataset = get_human_eval_plus(version="default")
    missing = [task_id for task_id in profile.task_ids or () if task_id not in dataset]
    if missing:
        raise ValueError(
            f"EvalPlus dataset is missing {len(missing)} quick-profile tasks: {missing[:3]}"
        )

    output_path = output_dir / f"{profile.suite}-{profile.name}.jsonl.gz"

    if output_path.exists():
        try:
            rows = _load_jsonl_gz(output_path)
        except Exception:
            rows = []
        row_ids = tuple(row.get("task_id") for row in rows)
        if row_ids == profile.task_ids:
            return output_path

    rows = [dataset[task_id] for task_id in profile.task_ids or ()]
    _write_jsonl_gz(output_path, rows)
    return output_path
