from __future__ import annotations

import csv
from pathlib import Path

from side_model3_adapter_v2.scripts.aggregate_libero_state_failure_analysis import (
    COMBINED_COLUMNS,
    aggregate_analysis_csvs,
)
from side_model3_adapter_v2.state_failure_analysis import ANALYSIS_COLUMNS


def _row(*, checkpoint_step: int, task: int, trial: int) -> dict[str, str]:
    row = {column: "" for column in ANALYSIS_COLUMNS}
    row.update(
        {
            "checkpoint_step": str(checkpoint_step),
            "task": str(task),
            "trial": str(trial),
            "success": "False",
            "primary_failure": "搬运/放置失败",
            "needs_manual_review": "False",
        }
    )
    return row


def _write_analysis(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANALYSIS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_preserves_provenance_and_sorts_failures(tmp_path: Path) -> None:
    step_80 = tmp_path / "step_080000.csv"
    step_70 = tmp_path / "step_070000.csv"
    _write_analysis(step_80, [_row(checkpoint_step=80000, task=4, trial=1)])
    _write_analysis(
        step_70,
        [
            _row(checkpoint_step=70000, task=2, trial=3),
            _row(checkpoint_step=70000, task=2, trial=1),
        ],
    )

    output = tmp_path / "all_checkpoints_state_failure_analysis.csv"
    result = aggregate_analysis_csvs([step_80, step_70], output)

    assert result["input_count"] == 2
    assert result["failure_episode_count"] == 3
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMBINED_COLUMNS)
        rows = list(reader)
    assert [(row["checkpoint_step"], row["task"], row["trial"]) for row in rows] == [
        ("70000", "2", "1"),
        ("70000", "2", "3"),
        ("80000", "4", "1"),
    ]
    assert rows[0]["source_analysis_path"] == str(step_70.resolve())
