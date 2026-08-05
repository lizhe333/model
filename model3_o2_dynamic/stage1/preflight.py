"""File-level integrity gate for all registered Dynamic Stage-1 demonstrations.

The gate intentionally opens only HDF5 metadata and group names.  It runs
before source-state selection, so a truncated task file fails without reading
an action/state payload or producing partial simulator artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py

from model3_o2_dynamic.config import Model3O2DynamicConfig, load_config

from .common import libero_root
from .contracts import (
    Stage1ContractError,
    Stage1DataConfig,
    demonstration_directory_for_suite,
    task_filenames_for_suite,
    task_names_for_suite,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_demonstration_file(
    path: str | Path,
    *,
    task: str,
    config: Stage1DataConfig,
) -> dict[str, Any]:
    """Open one source file without loading states/actions or teacher inputs."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise Stage1ContractError(f"missing Dynamic Stage-1 demonstration for {task}: {source}")
    try:
        with h5py.File(source, "r") as handle:
            if "data" not in handle or not isinstance(handle["data"], h5py.Group):
                raise Stage1ContractError(f"{task} HDF5 lacks a root data group: {source}")
            data = handle["data"]
            expected_demo_names = {f"demo_{index}" for index in range(config.demos_per_task)}
            observed_demo_names = {name for name in data.keys() if name.startswith("demo_")}
            if observed_demo_names != expected_demo_names:
                missing = sorted(expected_demo_names - observed_demo_names)
                extra = sorted(observed_demo_names - expected_demo_names)
                raise Stage1ContractError(
                    f"{task} HDF5 demo groups are incomplete: missing={missing}, extra={extra}"
                )
            bddl_file_name = data.attrs.get("bddl_file_name")
            if not isinstance(bddl_file_name, (str, bytes)) or not str(bddl_file_name):
                raise Stage1ContractError(f"{task} HDF5 lacks a non-empty BDDL identity")
    except Stage1ContractError:
        raise
    except Exception as error:
        raise Stage1ContractError(
            f"cannot open Dynamic Stage-1 HDF5 for {task}: {source}: "
            f"{type(error).__name__}: {error}"
        ) from error
    return {
        "task": task,
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "demo_count": config.demos_per_task,
        "payload_read": False,
    }


def preflight_demonstrations(
    *,
    dynamic_config: Model3O2DynamicConfig,
    output_root: str | Path,
    libero_root_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate every suite file before selection accesses any demo payload."""

    config = Stage1DataConfig()
    config.validate()
    suite = dynamic_config.evaluation.suite
    root = libero_root(libero_root_path)
    demonstration_root = root / "datasets" / demonstration_directory_for_suite(suite)
    filenames = task_filenames_for_suite(suite)
    records = [
        validate_demonstration_file(
            demonstration_root / filenames[task],
            task=task,
            config=config,
        )
        for task in task_names_for_suite(suite)
    ]
    result = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_demonstration_file_preflight",
        "suite": suite,
        "source_root": str(demonstration_root),
        "task_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "payload_read": False,
        "records": records,
    }
    destination = Path(output_root).expanduser().resolve() / "preflight" / "demonstration_integrity.json"
    _write_json(destination, result)
    result["manifest_path"] = str(destination)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            preflight_demonstrations(
                dynamic_config=load_config(args.config),
                output_root=args.output_root,
                libero_root_path=args.libero_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
