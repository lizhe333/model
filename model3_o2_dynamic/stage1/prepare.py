"""Seal the train/validation Dynamic Stage-1 response cache without test reads."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from model3_o2_dynamic.config import Model3O2DynamicConfig, load_config

from .action_normalization import load_official_o2_normalizer
from .cache import sha256_file
from .carrier import parent_config_path, validate_carrier_shard
from .collect import validate_collection_shard
from .contracts import (
    BRANCH_NAMES,
    LAYERS,
    Stage1DataConfig,
    Stage1ContractError,
    task_names_for_suite,
    validate_response_cache,
)
from .teacher import validate_teacher_shard


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _equal_identity(name: str, observed: dict[str, Any] | None, expected: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(observed, dict):
        raise Stage1ContractError(f"{name} identity is missing")
    if expected is not None and observed != expected:
        raise Stage1ContractError(f"{name} identity differs across Stage-1 shards")
    return observed


def _same_record(left: dict[str, Any], right: dict[str, Any], *, source: str) -> None:
    for key in ("sample_id", "task", "demo_id", "split", "source_index", "progress_bin"):
        if left.get(key) != right.get(key):
            raise Stage1ContractError(f"Stage-1 {source} record alignment failed at {key}")


def _load_collection_teacher_carrier(
    root: Path,
    *,
    split: str,
    task: str,
    demo_id: int,
    config: Stage1DataConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path, Path]:
    collection_path = root / "collection" / split / task / f"demo_{demo_id:03d}.pt"
    teacher_path = root / "teacher" / split / task / f"demo_{demo_id:03d}.pt"
    carrier_path = root / "carrier" / split / task / f"demo_{demo_id:03d}.pt"
    for path, label in ((collection_path, "collection"), (teacher_path, "teacher"), (carrier_path, "carrier")):
        if not path.is_file():
            raise FileNotFoundError(f"missing Stage-1 {label} shard: {path}")
    collection = torch.load(collection_path, map_location="cpu", weights_only=False)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    carrier = torch.load(carrier_path, map_location="cpu", weights_only=False)
    validate_collection_shard(collection, config=config)
    validate_teacher_shard(teacher, split=split, task=task, config=config)
    validate_carrier_shard(carrier, split=split, task=task, config=config)
    if int(collection["demo_id"]) != demo_id or int(teacher["demo_id"]) != demo_id or int(carrier["demo_id"]) != demo_id:
        raise Stage1ContractError("Stage-1 shard demo identity mismatch")
    return collection, teacher, carrier, collection_path, teacher_path, carrier_path


def prepare_train_validation_cache(
    *,
    dynamic_config: Model3O2DynamicConfig,
    output_root: str | Path,
    overwrite: bool = False,
) -> Path:
    """Materialize only train/validation data; test tensors are never opened here."""

    config = Stage1DataConfig()
    config.validate()
    root = Path(output_root).expanduser().resolve()
    output_path = root / "prepared" / "train_validation_response_cache.pt"
    if output_path.is_file() and not overwrite:
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        validate_response_cache(existing, require_trainable_splits_only=True)
        return output_path
    summary_path = root / "collection" / "train_validation" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing formal collection summary: {summary_path}")
    collection_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        collection_summary.get("splits") != ["train", "validation"]
        or collection_summary.get("state_count") != 4_500
        or collection_summary.get("branch_trajectory_count") != 18_000
    ):
        raise Stage1ContractError("formal train/validation collection summary is incomplete")

    normalizer = load_official_o2_normalizer(parent_config_path=parent_config_path(dynamic_config))
    action_identity = normalizer.identity()
    response_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    current_shards: list[dict[str, Any]] = []
    current_index: list[list[int]] = []
    source_identity: dict[str, Any] | None = None
    teacher_identity: dict[str, Any] | None = None
    shard_sources: list[dict[str, Any]] = []
    started = time.time()

    for split in ("train", "validation"):
        for task in task_names_for_suite(dynamic_config.evaluation.suite):
            for demo_id in range(50):
                if config.split_for_demo(demo_id) != split:
                    continue
                collection, teacher, carrier, collection_path, teacher_path, carrier_path = _load_collection_teacher_carrier(
                    root,
                    split=split,
                    task=task,
                    demo_id=demo_id,
                    config=config,
                )
                source_identity = _equal_identity(
                    "current carrier source",
                    carrier.get("source_identity"),
                    source_identity,
                )
                _equal_identity(
                    "carrier action normalizer",
                    carrier.get("action_normalization_identity"),
                    action_identity,
                )
                teacher_identity = _equal_identity(
                    "original Wan teacher",
                    teacher.get("teacher_identity"),
                    teacher_identity,
                )
                carrier_shard_index = len(current_shards)
                current_shards.append(
                    {
                        "path": str(carrier_path),
                        "sha256": sha256_file(carrier_path),
                        "shape": list(carrier["current_hidden"].shape),
                        "dtype": str(carrier["current_hidden"].dtype),
                        "task": task,
                        "split": split,
                        "demo_id": demo_id,
                    }
                )
                # Response is first averaged over the four common fit-noise
                # draws, then formed against the same-state zero branch.
                e0 = teacher["e0_global"].float()
                fit_mean = e0.mean(dim=2)
                response = fit_mean[:, :, :3] - fit_mean[:, :, 3:4]
                if tuple(response.shape) != (10, 3, 3, 4, 256):
                    raise Stage1ContractError(f"unexpected response target shape: {tuple(response.shape)}")
                response_parts.append(response)
                raw_actions = torch.stack([row["actions"] for row in collection["records"]], dim=0)
                action_parts.append(normalizer.normalize_action(raw_actions))
                for local_index, (collection_record, teacher_record, carrier_record) in enumerate(
                    zip(collection["records"], teacher["records"], carrier["records"])
                ):
                    _same_record(collection_record, teacher_record, source="teacher")
                    _same_record(collection_record, carrier_record, source="carrier")
                    records.append(
                        {
                            key: collection_record[key]
                            for key in (
                                "sample_id",
                                "task",
                                "task_position",
                                "demo_id",
                                "split",
                                "source_index",
                                "episode_progress",
                                "progress_bin",
                                "motion_labels",
                            )
                        }
                    )
                    current_index.append([carrier_shard_index, local_index])
                shard_sources.append(
                    {
                        "collection_path": str(collection_path),
                        "collection_sha256": sha256_file(collection_path),
                        "teacher_path": str(teacher_path),
                        "teacher_sha256": sha256_file(teacher_path),
                        "carrier_path": str(carrier_path),
                        "carrier_sha256": sha256_file(carrier_path),
                    }
                )
    if source_identity is None or teacher_identity is None:
        raise Stage1ContractError("no train/validation Stage-1 shards were prepared")
    response_targets = torch.cat(response_parts, dim=0).contiguous()
    actions = torch.cat(action_parts, dim=0).contiguous()
    index_tensor = torch.tensor(current_index, dtype=torch.int64)
    if len(records) != 4_500 or tuple(response_targets.shape) != (4_500, 3, 3, 4, 256):
        raise Stage1ContractError("formal train/validation cache cardinality changed")
    if tuple(actions.shape) != (4_500, 4, 8, 7) or tuple(index_tensor.shape) != (4_500, 2):
        raise Stage1ContractError("formal action/carrier index cache geometry changed")
    train_mask = torch.tensor([record["split"] == "train" for record in records], dtype=torch.bool)
    if int(train_mask.sum()) != 4_000:
        raise Stage1ContractError("train normalization split does not contain exactly 4000 states")
    train_response = response_targets[train_mask]
    normalization_mean = train_response.mean(dim=(0, 2))
    normalization_std = train_response.std(dim=(0, 2), unbiased=False).clamp_min(1.0e-6)
    if not torch.isfinite(response_targets).all() or not torch.isfinite(normalization_mean).all() or not torch.isfinite(normalization_std).all():
        raise Stage1ContractError("non-finite Dynamic Stage-1 response target / normalization")
    if float(response_targets.abs().sum()) == 0.0:
        raise Stage1ContractError("all Dynamic Stage-1 response targets are zero")
    payload = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_response_cache",
        "layers": list(LAYERS),
        "branch_names": list(BRANCH_NAMES),
        "response_branch_names": list(BRANCH_NAMES[:3]),
        "teacher_timestep": config.teacher_timestep,
        "target_space": "e0_global_projected_standardized",
        "normalization_fit_split": "train",
        "common_branch_noise": True,
        "source_identity": source_identity,
        "teacher_identity": teacher_identity,
        "action_normalization_identity": action_identity,
        "current_hidden": None,
        "current_hidden_shards": current_shards,
        "current_hidden_index": index_tensor,
        "actions": actions,
        "response_targets": response_targets,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
        "records": records,
        "source_shards": shard_sources,
        "collection_summary_path": str(summary_path),
        "collection_summary_sha256": sha256_file(summary_path),
        "test_state_count": 0,
        "heldout_noise_target_count": 0,
        "test_read": False,
        "started_unix": started,
        "completed_unix": time.time(),
    }
    validation = validate_response_cache(payload, require_trainable_splits_only=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    manifest = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_response_cache_manifest",
        "cache_path": str(output_path),
        "cache_sha256": sha256_file(output_path),
        "validation": validation,
        "test_read": False,
        "tensor_shapes": {
            "actions": list(actions.shape),
            "response_targets": list(response_targets.shape),
            "normalization_mean": list(normalization_mean.shape),
            "normalization_std": list(normalization_std.shape),
            "current_hidden_index": list(index_tensor.shape),
        },
        "carrier_shard_count": len(current_shards),
        "source_identity": source_identity,
        "teacher_identity": teacher_identity,
        "action_normalization_identity": action_identity,
    }
    _write_json(root / "prepared" / "train_validation_response_cache.manifest.json", manifest)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = prepare_train_validation_cache(
        dynamic_config=load_config(args.config),
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
