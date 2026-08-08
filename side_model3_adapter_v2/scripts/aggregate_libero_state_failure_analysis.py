"""Combine per-checkpoint LIBERO simulator-state failure analyses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from side_model3_adapter_v2.state_failure_analysis import ANALYSIS_COLUMNS


COMBINED_COLUMNS = (*ANALYSIS_COLUMNS, "source_analysis_path")


def aggregate_analysis_csvs(
    input_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Write a checkpoint-sorted, per-failure CSV from analyzer CSV files."""
    rows: list[dict[str, str]] = []
    seen_inputs: set[Path] = set()
    for input_path in input_paths:
        resolved_path = input_path.resolve()
        if resolved_path in seen_inputs:
            raise ValueError(f"duplicate analyzer input: {resolved_path}")
        seen_inputs.add(resolved_path)
        with resolved_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(ANALYSIS_COLUMNS):
                raise ValueError(f"unexpected analyzer CSV schema: {resolved_path}")
            for row in reader:
                rows.append(
                    {
                        **{column: row[column] for column in ANALYSIS_COLUMNS},
                        "source_analysis_path": str(resolved_path),
                    }
                )

    rows.sort(
        key=lambda row: (
            int(row["checkpoint_step"]),
            int(row["task"]),
            int(row["trial"]),
            row["source_analysis_path"],
        )
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMBINED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "input_count": len(seen_inputs),
        "failure_episode_count": len(rows),
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Per-checkpoint state_failure_analysis.csv; repeat for every checkpoint.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate_analysis_csvs(args.input, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
