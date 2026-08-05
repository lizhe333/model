"""Original-Wan E0 teacher extraction for Dynamic Stage-1 responses."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from model3_o2_dynamic.config import Model3O2DynamicConfig, load_config

from analysis.action_response_local.gate0b_teacher import _encode_noise_batch, _projection
from analysis.action_response_local.teacher import _load_context, _load_original_wan, _preprocess_dual_view

from .collect import validate_collection_shard
from .common import sha256_file
from .contracts import (
    BRANCH_NAMES,
    LAYERS,
    Stage1DataConfig,
    Stage1ContractError,
    task_names_for_suite,
    task_subset_for_suite,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parent_config_path(dynamic: Model3O2DynamicConfig) -> Path:
    # .../<run>/checkpoints/weights/step_020000.pt -> .../<run>/config.yaml
    candidate = dynamic.initialization.model3_checkpoint.parent.parent.parent / "config.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(f"cannot locate pinned Model3 parent config: {candidate}")
    return candidate


def _original_teacher_config(dynamic: Model3O2DynamicConfig) -> dict[str, Any]:
    parent_path = _parent_config_path(dynamic)
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    if not isinstance(parent, dict) or not isinstance(parent.get("model"), dict):
        raise ValueError("pinned Model3 parent config lacks a model mapping")
    model = parent["model"]
    return {
        "resolved_model3_config": str(parent_path),
        "checkpoint_root": str(dynamic.base.backend.checkpoint_root),
        "base_model": {
            "model_id": str(model["model_id"]),
            "video_backbone_type": str(model["video_backbone_type"]),
            "video_backbone_name": str(model["video_backbone_name"]),
        },
    }


def _teacher_identity(dynamic: Model3O2DynamicConfig, pretrained: dict[str, Any]) -> dict[str, Any]:
    parent_config = _parent_config_path(dynamic)
    return {
        "teacher_kind": "original_pretrained_wan_e0_global",
        "original_wan": pretrained,
        "parent_config_path": str(parent_config),
        "parent_config_sha256": sha256_file(parent_config),
        "teacher_timestep": 250,
        "precision": "torch.bfloat16",
        "layers": list(LAYERS),
        "projection": {
            "kind": "fixed_gaussian_1536_to_256",
            "seed": 83021,
            "spatial_reduction": "future_tokens_global_mean",
        },
        "scheduler": {
            "kind": "WanContinuousFlowMatchScheduler",
            "num_train_timesteps": 1000,
            "shift": 5.0,
        },
        "conditioning": "cached_task_text_context_only",
    }


@dataclass
class OriginalWanE0Teacher:
    dynamic: Model3O2DynamicConfig
    device: torch.device
    components: Any
    pretrained_identity: dict[str, Any]
    projection: torch.Tensor
    identity: dict[str, Any]

    @classmethod
    def create(cls, dynamic: Model3O2DynamicConfig, *, device: str | torch.device) -> "OriginalWanE0Teacher":
        target_device = torch.device(device)
        if target_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Dynamic Stage-1 teacher extraction requires a CUDA device")
        torch.cuda.set_device(target_device)
        data_config = Stage1DataConfig()
        components, pretrained = _load_original_wan(_original_teacher_config(dynamic), target_device)
        projection = _projection(
            1536,
            data_config.projection_dim,
            data_config.projection_seed,
            target_device,
        )
        return cls(
            dynamic=dynamic,
            device=target_device,
            components=components,
            pretrained_identity=pretrained,
            projection=projection,
            identity=_teacher_identity(dynamic, pretrained),
        )

    def close(self) -> None:
        del self.components
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def validate_teacher_shard(payload: dict[str, Any], *, split: str, task: str, config: Stage1DataConfig) -> None:
    if payload.get("artifact_kind") != "stage1_e0_teacher_shard":
        raise Stage1ContractError("wrong Dynamic Stage-1 teacher artifact kind")
    if payload.get("track_id") != "model3_o2_dynamic" or payload.get("task") != task or payload.get("split") != split:
        raise Stage1ContractError("teacher shard identity mismatch")
    if tuple(payload.get("layers", ())) != LAYERS or tuple(payload.get("branch_names", ())) != BRANCH_NAMES:
        raise Stage1ContractError("teacher layer/branch identity changed")
    expected_seeds = config.heldout_noise_seeds if split == "test" else config.fit_noise_seeds
    if tuple(payload.get("noise_seeds", ())) != expected_seeds:
        raise Stage1ContractError("teacher noise identity changed")
    if payload.get("teacher_timestep") != config.teacher_timestep or payload.get("target_space") != "e0_global_projected":
        raise Stage1ContractError("teacher target space changed")
    records = payload.get("records")
    e0 = payload.get("e0_global")
    if not isinstance(records, list) or len(records) != config.states_per_demo:
        raise Stage1ContractError("teacher shard must contain ten selected states")
    if not isinstance(e0, torch.Tensor) or tuple(e0.shape) != (
        config.states_per_demo,
        len(LAYERS),
        len(expected_seeds),
        len(BRANCH_NAMES),
        len(config.stage_render_ticks),
        config.projection_dim,
    ):
        raise Stage1ContractError(f"teacher E0 shape mismatch: {getattr(e0, 'shape', None)}")
    if not torch.isfinite(e0.float()).all():
        raise Stage1ContractError("teacher E0 contains non-finite values")
    identity = payload.get("teacher_identity")
    if not isinstance(identity, dict) or not identity.get("original_wan", {}).get("pass"):
        raise Stage1ContractError("teacher did not identify a valid frozen original Wan")
    if payload.get("common_branch_noise") is not True:
        raise Stage1ContractError("teacher must use identical noise for all same-state branches")


def _matches_current_collection_provenance(
    payload: object,
    *,
    collection_path: Path,
    collection_sha256: str,
) -> bool:
    """Return whether an existing teacher shard came from this exact collection shard."""

    if not isinstance(payload, dict):
        return False
    return (
        payload.get("collection_path") == str(collection_path.expanduser().resolve())
        and payload.get("collection_sha256") == collection_sha256
    )


def _can_reuse_existing_teacher(
    *,
    output_path: Path,
    split: str,
    task: str,
    config: Stage1DataConfig,
    collection_path: Path,
    collection_sha256: str,
) -> bool:
    """Validate an existing teacher shard before deciding whether resume may reuse it."""

    try:
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        validate_teacher_shard(payload, split=split, task=task, config=config)
    except Exception:
        # Preserve the prior file until a complete replacement has been saved
        # to a temporary path and atomically promoted below.
        return False
    return _matches_current_collection_provenance(
        payload,
        collection_path=collection_path,
        collection_sha256=collection_sha256,
    )


@torch.inference_mode()
def _extract_record(
    teacher: OriginalWanE0Teacher,
    record: dict[str, Any],
    *,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    noise_seeds: tuple[int, ...],
    config: Stage1DataConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    current = _preprocess_dual_view(record["current_images"], config.model_view_resolution)
    branch_features: list[torch.Tensor] = []
    runtime: dict[str, Any] | None = None
    for branch_position, branch_name in enumerate(BRANCH_NAMES):
        stage_features: list[torch.Tensor] = []
        for stage_position, _render_tick in enumerate(config.stage_render_ticks):
            stage = _preprocess_dual_view(
                record["stage_images"][branch_position, stage_position],
                config.model_view_resolution,
            )
            # There is deliberately no branch term in this formula.  Every
            # branch at one state/stage observes the same Monte-Carlo teacher
            # draw; only its future image differs.
            seeds = [
                int(base_seed) + int(record["sample_id"]) * 1009 + stage_position * 17
                for base_seed in noise_seeds
            ]
            e0, _local_unused, meta = _encode_noise_batch(
                teacher.components,
                current_image=current,
                stage_image=stage,
                context=context,
                context_mask=context_mask,
                layers=list(LAYERS),
                global_projection=teacher.projection,
                local_projection=teacher.projection,
                scalar_timestep=config.teacher_timestep,
                noise_seeds=seeds,
                local_grid_size=(4, 7),
                device=teacher.device,
            )
            # [noise, layer, 256]
            if tuple(e0.shape) != (len(seeds), len(LAYERS), config.projection_dim):
                raise RuntimeError(f"unexpected E0 batch output: {tuple(e0.shape)}")
            stage_features.append(e0)
            if runtime is None:
                runtime = {
                    "stage_grid_size": list(meta["grid_size"]),
                    "tokens_per_frame": int(meta["tokens_per_frame"]),
                    "noise_seed_formula": "base_seed + sample_id * 1009 + stage_position * 17",
                }
        # [noise, layer, stage, 256]
        branch_features.append(torch.stack(stage_features, dim=2))
    # [layer, noise, branch, stage, 256]
    features = torch.stack(branch_features, dim=2).permute(1, 0, 2, 3, 4).contiguous()
    return features.to(dtype=torch.float32), runtime or {}


def extract_demo_shard(
    *,
    teacher: OriginalWanE0Teacher,
    collection_path: Path,
    output_path: Path,
    split: str,
    task: str,
    overwrite: bool,
) -> Path:
    config = Stage1DataConfig()
    current_collection_path = collection_path.expanduser().resolve()
    current_collection_sha256 = sha256_file(current_collection_path)
    if output_path.is_file() and not overwrite and _can_reuse_existing_teacher(
        output_path=output_path,
        split=split,
        task=task,
        config=config,
        collection_path=current_collection_path,
        collection_sha256=current_collection_sha256,
    ):
        return output_path
    collection = torch.load(current_collection_path, map_location="cpu", weights_only=False)
    validate_collection_shard(collection, config=config)
    if collection.get("task") != task or collection.get("split") != split:
        raise Stage1ContractError("collection / teacher split identity mismatch")
    records = collection["records"]
    noise_seeds = config.heldout_noise_seeds if split == "test" else config.fit_noise_seeds
    context_cpu, mask_cpu, context_path = _load_context(
        records[0]["instruction"], Path(teacher.dynamic.base.data.text_embedding_cache_dir)
    )
    context = context_cpu.unsqueeze(0).to(device=teacher.device, dtype=torch.bfloat16)
    context_mask = mask_cpu.unsqueeze(0).to(device=teacher.device, dtype=torch.bool)
    values: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    for record in records:
        e0, record_runtime = _extract_record(
            teacher,
            record,
            context=context,
            context_mask=context_mask,
            noise_seeds=noise_seeds,
            config=config,
        )
        values.append(e0)
        metadata.append(
            {
                key: record[key]
                for key in (
                    "sample_id",
                    "task",
                    "demo_id",
                    "split",
                    "source_index",
                    "progress_bin",
                    "motion_labels",
                )
            }
        )
        runtime = runtime or record_runtime
    payload = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_e0_teacher_shard",
        "task": task,
        "split": split,
        "demo_id": int(collection["demo_id"]),
        "layers": list(LAYERS),
        "branch_names": list(BRANCH_NAMES),
        "noise_seeds": list(noise_seeds),
        "noise_split": "heldout" if split == "test" else "fit",
        "teacher_timestep": config.teacher_timestep,
        "target_space": "e0_global_projected",
        "common_branch_noise": True,
        "records": metadata,
        "e0_global": torch.stack(values, dim=0),
        "teacher_identity": teacher.identity,
        "teacher_runtime": runtime,
        "context_cache_path": str(context_path),
        "context_cache_sha256": sha256_file(context_path),
        "collection_path": str(current_collection_path),
        "collection_sha256": current_collection_sha256,
    }
    validate_teacher_shard(payload, split=split, task=task, config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return output_path


def _require_test_permission(splits: Iterable[str], after_stage1_export: str | Path | None) -> None:
    if "test" not in set(splits):
        return
    if after_stage1_export is None or not Path(after_stage1_export).expanduser().is_file():
        raise Stage1ContractError(
            "test teacher targets are sealed until a fixed Stage-1 adapter export exists; "
            "pass --after-stage1-export after step 5K"
        )


def extract_all(
    *,
    dynamic_config: Model3O2DynamicConfig,
    output_root: str | Path,
    device: str | torch.device,
    splits: Iterable[str] = ("train", "validation"),
    tasks: Iterable[str] | None = None,
    after_stage1_export: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = Stage1DataConfig()
    config.validate()
    selected_splits = tuple(str(value) for value in splits)
    if not selected_splits or any(value not in {"train", "validation", "test"} for value in selected_splits):
        raise ValueError(f"unsupported Stage-1 teacher splits: {selected_splits}")
    _require_test_permission(selected_splits, after_stage1_export)
    root = Path(output_root).expanduser().resolve()
    selected_tasks = task_subset_for_suite(dynamic_config.evaluation.suite, tasks)
    full_suite = selected_tasks == task_names_for_suite(dynamic_config.evaluation.suite)
    teacher = OriginalWanE0Teacher.create(dynamic_config, device=device)
    entries: list[dict[str, Any]] = []
    try:
        for split in selected_splits:
            for task in selected_tasks:
                for demo_id in range(50):
                    if config.split_for_demo(demo_id) != split:
                        continue
                    collection_path = root / "collection" / split / task / f"demo_{demo_id:03d}.pt"
                    if not collection_path.is_file():
                        raise FileNotFoundError(f"missing collection shard: {collection_path}")
                    output_path = root / "teacher" / split / task / f"demo_{demo_id:03d}.pt"
                    result = extract_demo_shard(
                        teacher=teacher,
                        collection_path=collection_path,
                        output_path=output_path,
                        split=split,
                        task=task,
                        overwrite=overwrite,
                    )
                    entries.append(
                        {
                            "split": split,
                            "task": task,
                            "demo_id": demo_id,
                            "path": str(result),
                            "sha256": sha256_file(result),
                        }
                    )
                    print(f"[dynamic-teacher:{split}] task={task} demo={demo_id + 1}/50", flush=True)
    finally:
        teacher.close()
    split_counts = {
        split: len([entry for entry in entries if entry["split"] == split]) * config.states_per_demo
        for split in selected_splits
    }
    result = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_e0_teacher_summary",
        "splits": list(selected_splits),
        "tasks": list(selected_tasks),
        "teacher_identity": teacher.identity,
        "shards": entries,
        "split_state_counts": split_counts,
        "test_read": "test" in selected_splits,
    }
    if full_suite and selected_splits == ("train", "validation"):
        _write_json(root / "teacher" / "summary.json", result)
    # A single stable summary file is intentionally only emitted for the
    # train/validation pretraining path.  Test diagnostics may keep their own
    # manifest without contaminating the train-cache read ledger.
    if full_suite and selected_splits != ("train", "validation"):
        _write_json(root / "teacher" / "test_or_aux_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--after-stage1-export", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = extract_all(
        dynamic_config=load_config(args.config),
        output_root=args.output_root,
        device=args.device,
        splits=args.splits,
        tasks=args.tasks,
        after_stage1_export=args.after_stage1_export,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
