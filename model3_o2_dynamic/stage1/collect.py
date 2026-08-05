"""Collect four exact-state LIBERO branch trajectories for Dynamic Stage 1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from model3_o2_dynamic.config import Model3O2DynamicConfig, load_config

from .common import (
    COLLECTION_RENDERER_CONTEXT_CONTRACT,
    SOURCE_RUNTIME_RESTORE_CONTRACT,
    STATIC_MODEL_STATE_ARRAYS,
    branch_actions,
    capture_static_model_state as _capture_static_model_state,
    close_env,
    images_from_obs,
    libero_root,
    motion_target_hints,
    object_hints,
    proprio_from_obs,
    restore_source_runtime,
    restore_static_model_state as _restore_static_model_state,
    sha256_file,
    step_with_render_ticks,
    target_pose,
    torch_record,
)
from .contracts import (
    BRANCH_NAMES,
    SPLIT_STATE_COUNTS,
    Stage1DataConfig,
    Stage1ContractError,
    task_names_for_suite,
)
from .selection import _require_test_permission, _validate_selection_payload, normalize_splits, split_label


BRANCH_RESTORE_SEED = 42
EXPERT_RENDER_MAX_ATTEMPTS = 4


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_env(*, bddl_path: Path, resolution: int) -> Any:
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=int(resolution),
        camera_widths=int(resolution),
        hard_reset=True,
    )
    env.seed(42)
    return env


def _static_model_state_path(root: Path, tag: str, task: str) -> Path:
    return root / "selection" / tag / f"{task}.static_model_state.pt"


def _load_static_model_state(
    *,
    path: Path,
    task: str,
    splits: tuple[str, ...],
    bddl_path: Path,
    selection_path: Path,
    env: Any,
) -> tuple[dict[str, np.ndarray], str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing selection static model-state snapshot: {path}")
    selection = torch.load(selection_path, map_location="cpu", weights_only=False)
    snapshot_sha256 = sha256_file(path)
    if (
        selection.get("source_restore_contract") != SOURCE_RUNTIME_RESTORE_CONTRACT
        or selection.get("static_model_state_path") != str(path)
        or selection.get("static_model_state_sha256") != snapshot_sha256
    ):
        raise Stage1ContractError(f"selection static model-state provenance changed: {selection_path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("artifact_kind") != "stage1_static_model_state"
        or payload.get("track_id") != "model3_o2_dynamic"
        or payload.get("source_restore_contract") != SOURCE_RUNTIME_RESTORE_CONTRACT
        or payload.get("task") != task
        or tuple(payload.get("splits", ())) != splits
        or payload.get("bddl_file") != str(bddl_path)
        or payload.get("source_hdf5_sha256") != selection.get("source_hdf5_sha256")
    ):
        raise Stage1ContractError(f"invalid static model-state provenance: {path}")
    arrays = payload.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != set(STATIC_MODEL_STATE_ARRAYS):
        raise Stage1ContractError(f"invalid static model-state arrays: {path}")
    values = {}
    for name in STATIC_MODEL_STATE_ARRAYS:
        value = arrays[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise Stage1ContractError(f"invalid static model-state tensor {name}: {path}")
        values[name] = value.numpy().copy()
    _restore_static_model_state(env, values)
    return values, snapshot_sha256


def _restore_branch(
    env: Any,
    source_state: np.ndarray,
    config: Stage1DataConfig,
    static_model_state: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Restore fixed visual-model and complete physical state before a branch."""

    seed = getattr(env, "seed", None)
    if not callable(seed):
        raise TypeError("LIBERO environment does not expose a callable seed method")
    seed(BRANCH_RESTORE_SEED)
    env.reset()
    _restore_static_model_state(env, static_model_state)
    return restore_source_runtime(env, source_state, tolerance=config.restore_state_max_abs)


def _run_branch(
    env: Any,
    *,
    source_state: np.ndarray,
    actions: np.ndarray,
    config: Stage1DataConfig,
    static_model_state: dict[str, np.ndarray],
) -> dict[str, Any]:
    restored = _restore_branch(env, source_state, config, static_model_state)
    initial_obs = restored["obs"]
    current_images = images_from_obs(initial_obs)
    current_proprio = proprio_from_obs(initial_obs)
    hints = motion_target_hints(env)
    render_images: list[np.ndarray] = []
    done_steps: list[int] = []
    capture_substeps: list[tuple[int, ...]] = []
    final_obs = initial_obs
    for action_step, action in enumerate(actions, start=1):
        final_obs, _, done, _, captures, substeps = step_with_render_ticks(
            env,
            action,
            render_ticks_per_action=config.render_ticks_per_action,
        )
        capture_substeps.append(substeps)
        render_images.extend(images_from_obs(obs) for obs in captures)
        if done:
            done_steps.append(action_step)
    expected_ticks = config.action_horizon * config.render_ticks_per_action
    if len(render_images) != expected_ticks:
        raise RuntimeError(f"branch render coverage mismatch: {len(render_images)} vs {expected_ticks}")
    render_tensor = np.stack(render_images, axis=0)
    stage_indices = [tick - 1 for tick in config.stage_render_ticks]
    stage_images = render_tensor[stage_indices]
    target_position, _, target_key = target_pose(final_obs, hints)
    terminal_state = np.asarray(env.get_sim_state(), dtype=np.float64)
    if not np.isfinite(terminal_state).all() or not np.isfinite(target_position).all():
        raise RuntimeError("branch terminal simulator state is non-finite")
    return {
        "restore_max_abs": float(restored["restore_max_abs"]),
        "current_images": current_images,
        "current_proprio": current_proprio,
        "render_tick_images": render_tensor,
        "stage_images": stage_images,
        "render_capture_substeps": capture_substeps,
        "terminal_state": terminal_state,
        "target_position_key": target_key,
        "target_position": target_position,
        "done_steps": done_steps,
    }


def _build_record(
    env: Any,
    selection_record: dict[str, Any],
    config: Stage1DataConfig,
    static_model_state: dict[str, np.ndarray],
) -> dict[str, Any]:
    source_state = np.asarray(selection_record["source_state"], dtype=np.float64)
    expert = np.asarray(selection_record["expert_actions"], dtype=np.float32)
    actions_by_branch = branch_actions(
        expert,
        task_position=int(selection_record["task_position"]),
        demo_id=int(selection_record["demo_id"]),
        config=config,
    )
    branches = {
        name: _run_branch(
            env,
            source_state=source_state,
            actions=actions_by_branch[name],
            config=config,
            static_model_state=static_model_state,
        )
        for name in BRANCH_NAMES
    }
    expert_attempts = [branches["expert"]]
    terminal_error = 0.0
    expert_branch: dict[str, Any] | None = None
    expert_repeat_attempt_count = 1
    for attempt_number in range(2, EXPERT_RENDER_MAX_ATTEMPTS + 1):
        repeat = _run_branch(
            env,
            source_state=source_state,
            actions=actions_by_branch["expert"],
            config=config,
            static_model_state=static_model_state,
        )
        candidate_terminal_error = float(
            np.max(np.abs(expert_attempts[0]["terminal_state"] - repeat["terminal_state"]))
        )
        terminal_error = max(terminal_error, candidate_terminal_error)
        if candidate_terminal_error > config.repeat_terminal_state_max_abs:
            raise RuntimeError(
                f"same-state expert repeat terminal error {candidate_terminal_error:.3e} exceeds "
                f"{config.repeat_terminal_state_max_abs:.3e}"
            )
        for prior in expert_attempts:
            render_equal = bool(np.array_equal(prior["render_tick_images"], repeat["render_tick_images"]))
            current_equal = bool(np.array_equal(prior["current_images"], repeat["current_images"]))
            if render_equal and current_equal:
                expert_branch = prior
                branches["expert"] = expert_branch
                expert_repeat_attempt_count = attempt_number
                break
        if expert_branch is not None:
            break
        expert_attempts.append(repeat)
    if expert_branch is None:
        reference = expert_attempts[0]
        last = expert_attempts[-1]
        current_delta = np.abs(reference["current_images"].astype(np.int16) - last["current_images"].astype(np.int16))
        render_delta = np.abs(
            reference["render_tick_images"].astype(np.int16) - last["render_tick_images"].astype(np.int16)
        )
        raise RuntimeError(
            "same-state expert repeat did not reach exact renderer consensus "
            f"after {EXPERT_RENDER_MAX_ATTEMPTS} attempts "
            f"(sample_id={selection_record['sample_id']} demo_id={selection_record['demo_id']} "
            f"progress_bin={selection_record['progress_bin']} source_index={selection_record['source_index']} "
            f"current_max_abs={int(current_delta.max())} "
            f"current_changed={int(np.count_nonzero(current_delta))} "
            f"render_max_abs={int(render_delta.max())} "
            f"render_changed={int(np.count_nonzero(render_delta))})"
        )
    proprio_candidates = [branches[name]["current_proprio"] for name in BRANCH_NAMES]
    if not all(np.array_equal(proprio_candidates[0], item) for item in proprio_candidates[1:]):
        raise RuntimeError("same-state branches do not share identical O2 proprio")
    target_keys = [branches[name]["target_position_key"] for name in BRANCH_NAMES]
    if len(set(target_keys)) != 1:
        raise RuntimeError(f"branch target object identity changed: {target_keys}")
    return {
        **{
            key: selection_record[key]
            for key in (
                "sample_id",
                "task",
                "task_position",
                "instruction",
                "demo_id",
                "split",
                "source_index",
                "episode_progress",
                "progress_bin",
                "motion_labels",
                "target_position_key",
            )
        },
        "source_state": torch_record(source_state),
        "branch_names": list(BRANCH_NAMES),
        "canonical_current_branch": "expert",
        "actions": torch_record(
            np.stack([actions_by_branch[name] for name in BRANCH_NAMES]), dtype=torch.float32
        ),
        "current_images": torch_record(expert_branch["current_images"]),
        "current_proprio_raw": torch_record(expert_branch["current_proprio"], dtype=torch.float32),
        "render_tick_images": torch_record(
            np.stack([branches[name]["render_tick_images"] for name in BRANCH_NAMES])
        ),
        "stage_images": torch_record(
            np.stack([branches[name]["stage_images"] for name in BRANCH_NAMES])
        ),
        "terminal_states": torch_record(
            np.stack([branches[name]["terminal_state"] for name in BRANCH_NAMES]), dtype=torch.float64
        ),
        "target_positions": torch_record(
            np.stack([branches[name]["target_position"] for name in BRANCH_NAMES]), dtype=torch.float64
        ),
        "restore_max_abs": max(float(branches[name]["restore_max_abs"]) for name in BRANCH_NAMES),
        "repeat_terminal_state_max_abs": terminal_error,
        "repeat_render_images_exact": True,
        "repeat_current_images_exact": True,
        "expert_repeat_attempt_count": expert_repeat_attempt_count,
        "render_capture_substeps": expert_branch["render_capture_substeps"],
        "done_steps": {name: list(branches[name]["done_steps"]) for name in BRANCH_NAMES},
    }


def validate_collection_shard(payload: dict[str, Any], *, config: Stage1DataConfig) -> None:
    if payload.get("artifact_kind") != "stage1_branch_collection_shard":
        raise Stage1ContractError("wrong collection artifact kind")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != config.states_per_demo:
        raise Stage1ContractError("one collection shard must contain exactly ten source states")
    demo_id = payload.get("demo_id")
    if not isinstance(demo_id, int):
        raise Stage1ContractError("collection shard lacks integer demo_id")
    if not isinstance(payload.get("static_model_state_sha256"), str):
        raise Stage1ContractError("collection shard lacks static model-state provenance")
    if payload.get("source_restore_contract") != SOURCE_RUNTIME_RESTORE_CONTRACT:
        raise Stage1ContractError("collection shard source-runtime restore contract changed")
    if payload.get("renderer_context_contract") != COLLECTION_RENDERER_CONTEXT_CONTRACT:
        raise Stage1ContractError("collection shard renderer-context contract changed")
    if [int(record["progress_bin"]) for record in records] != list(range(10)):
        raise Stage1ContractError("collection shard progress-bin order changed")
    for record in records:
        if list(record.get("branch_names", ())) != list(BRANCH_NAMES):
            raise Stage1ContractError("collection branch order changed")
        if record.get("canonical_current_branch") != "expert":
            raise Stage1ContractError("collection current-source branch changed")
        if tuple(record["actions"].shape) != (4, 8, 7):
            raise Stage1ContractError("collection action tensor shape changed")
        if tuple(record["current_proprio_raw"].shape) != (8,):
            raise Stage1ContractError("collection raw proprio shape changed")
        if tuple(record["render_tick_images"].shape[0:3]) != (4, 32, 2):
            raise Stage1ContractError("collection requires four branches x 32 ticks x two cameras")
        if tuple(record["stage_images"].shape[0:3]) != (4, 4, 2):
            raise Stage1ContractError("collection stage-image alignment changed")
        if float(record["restore_max_abs"]) > config.restore_state_max_abs:
            raise Stage1ContractError("collection restore error exceeds contract")
        if float(record["repeat_terminal_state_max_abs"]) > config.repeat_terminal_state_max_abs:
            raise Stage1ContractError("collection repeat terminal error exceeds contract")
        if not record.get("repeat_render_images_exact") or not record.get("repeat_current_images_exact"):
            raise Stage1ContractError("collection repeatability has not passed exact image identity")
        attempt_count = record.get("expert_repeat_attempt_count")
        if not isinstance(attempt_count, int) or not 2 <= attempt_count <= EXPERT_RENDER_MAX_ATTEMPTS:
            raise Stage1ContractError("collection expert repeat attempt count changed")


def _task_manifest_path(root: Path, tag: str, task: str) -> Path:
    return root / "collection" / tag / f"{task}.manifest.json"


def _matches_current_collection_provenance(
    payload: dict[str, Any],
    *,
    task: str,
    split: str,
    demo_id: int,
    selection_path: Path,
    selection_sha256: str,
    static_model_path: Path,
    static_model_state_sha256: str,
    config: Stage1DataConfig,
) -> bool:
    """Allow resume only for a fully current, structurally validated shard."""

    return (
        payload.get("schema_version") == 1
        and payload.get("track_id") == "model3_o2_dynamic"
        and payload.get("artifact_kind") == "stage1_branch_collection_shard"
        and payload.get("task") == task
        and payload.get("split") == split
        and payload.get("demo_id") == demo_id
        and payload.get("selection_path") == str(selection_path)
        and payload.get("selection_sha256") == selection_sha256
        and payload.get("source_restore_contract") == SOURCE_RUNTIME_RESTORE_CONTRACT
        and payload.get("renderer_context_contract") == COLLECTION_RENDERER_CONTEXT_CONTRACT
        and payload.get("static_model_state_path") == str(static_model_path)
        and payload.get("static_model_state_sha256") == static_model_state_sha256
        and payload.get("data_contract") == config.as_dict()
    )


def collect_task(
    *,
    dynamic_config: Model3O2DynamicConfig,
    task: str,
    output_root: str | Path,
    splits: Iterable[str] = ("train", "validation"),
    after_stage1_export: str | Path | None = None,
    libero_root_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one bounded collection shard per demonstration, never a giant RAM payload."""

    config = Stage1DataConfig()
    config.validate()
    selected_splits = normalize_splits(splits)
    _require_test_permission(selected_splits, after_stage1_export)
    task_names = task_names_for_suite(dynamic_config.evaluation.suite)
    if task not in task_names:
        raise ValueError(f"unknown Dynamic Stage-1 task for {dynamic_config.evaluation.suite}: {task}")
    root = Path(output_root).expanduser().resolve()
    tag = split_label(selected_splits)
    selection_path = root / "selection" / tag / f"{task}.pt"
    if not selection_path.is_file():
        raise FileNotFoundError(f"missing source selection for {task}: {selection_path}")
    selection = torch.load(selection_path, map_location="cpu", weights_only=False)
    _validate_selection_payload(selection, task=task, config=config, expected_splits=selected_splits)
    selection_sha256 = sha256_file(selection_path)
    bddl_path = Path(selection["bddl_file"])
    if not bddl_path.is_file():
        # Resolve early so a stale selection never causes partial collection.
        _ = libero_root(libero_root_path)
        raise FileNotFoundError(f"selection references missing BDDL file: {bddl_path}")
    selected_demo_ids = [
        demo_id
        for demo_id in range(config.demos_per_task)
        if config.split_for_demo(demo_id) in selected_splits
    ]
    records_by_demo: dict[int, list[dict[str, Any]]] = {demo_id: [] for demo_id in selected_demo_ids}
    for row in selection["records"]:
        records_by_demo[int(row["demo_id"])].append(row)
    started = time.time()
    shard_entries: list[dict[str, Any]] = []
    env = _make_env(bddl_path=bddl_path, resolution=config.render_resolution)
    try:
        static_model_path = _static_model_state_path(root, tag, task)
        static_model_state, static_model_state_sha256 = _load_static_model_state(
            path=static_model_path,
            task=task,
            splits=selected_splits,
            bddl_path=bddl_path,
            selection_path=selection_path,
            env=env,
        )
        for demo_id in selected_demo_ids:
            split = config.split_for_demo(demo_id)
            shard_path = root / "collection" / split / task / f"demo_{demo_id:03d}.pt"
            reusable = False
            if shard_path.is_file() and not overwrite:
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                if _matches_current_collection_provenance(
                    shard,
                    task=task,
                    split=split,
                    demo_id=demo_id,
                    selection_path=selection_path,
                    selection_sha256=selection_sha256,
                    static_model_path=static_model_path,
                    static_model_state_sha256=static_model_state_sha256,
                    config=config,
                ):
                    validate_collection_shard(shard, config=config)
                    reusable = True
            if not reusable:
                selected = sorted(records_by_demo[demo_id], key=lambda row: int(row["progress_bin"]))
                if len(selected) != config.states_per_demo:
                    raise Stage1ContractError(f"{task} demo {demo_id} selection cardinality changed")
                rows = [_build_record(env, row, config, static_model_state) for row in selected]
                shard = {
                    "schema_version": 1,
                    "track_id": "model3_o2_dynamic",
                    "artifact_kind": "stage1_branch_collection_shard",
                    "task": task,
                    "task_position": int(selection["task_position"]),
                    "split": split,
                    "demo_id": demo_id,
                    "selection_path": str(selection_path),
                    "selection_sha256": selection_sha256,
                    "source_restore_contract": SOURCE_RUNTIME_RESTORE_CONTRACT,
                    "renderer_context_contract": COLLECTION_RENDERER_CONTEXT_CONTRACT,
                    "static_model_state_path": str(static_model_path),
                    "static_model_state_sha256": static_model_state_sha256,
                    "data_contract": config.as_dict(),
                    "records": rows,
                    "started_unix": started,
                    "completed_unix": time.time(),
                }
                validate_collection_shard(shard, config=config)
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = shard_path.with_suffix(".partial.pt")
                torch.save(shard, temporary)
                temporary.replace(shard_path)
            shard_entries.append(
                {
                    "split": split,
                    "demo_id": demo_id,
                    "path": str(shard_path),
                    "sha256": sha256_file(shard_path),
                    "record_count": config.states_per_demo,
                }
            )
            print(f"[dynamic-collect:{task}] demo={demo_id + 1}/50", flush=True)
    finally:
        close_env(env)
    split_counts = {
        split: sum(entry["record_count"] for entry in shard_entries if entry["split"] == split)
        for split in selected_splits
    }
    expected_split_counts = {
        split: SPLIT_STATE_COUNTS[split] // len(task_names)
        for split in selected_splits
    }
    if split_counts != expected_split_counts:
        raise Stage1ContractError(f"{task} collection split count mismatch: {split_counts}")
    manifest = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_branch_collection_task_manifest",
        "task": task,
        "splits": list(selected_splits),
        "selection_path": str(selection_path),
        "selection_sha256": selection_sha256,
        "source_restore_contract": SOURCE_RUNTIME_RESTORE_CONTRACT,
        "renderer_context_contract": COLLECTION_RENDERER_CONTEXT_CONTRACT,
        "static_model_state_path": str(static_model_path),
        "static_model_state_sha256": static_model_state_sha256,
        "data_contract": config.as_dict(),
        "shards": shard_entries,
        "state_count": sum(split_counts.values()),
        "branch_trajectory_count": sum(split_counts.values()) * len(BRANCH_NAMES),
        "per_camera_rgb_frame_count": sum(split_counts.values()) * len(BRANCH_NAMES) * 32 * 2,
        "split_counts": split_counts,
        "started_unix": started,
        "completed_unix": time.time(),
    }
    _write_json(_task_manifest_path(root, tag, task), manifest)
    return manifest


def collect_all(
    *,
    dynamic_config: Model3O2DynamicConfig,
    output_root: str | Path,
    splits: Iterable[str] = ("train", "validation"),
    after_stage1_export: str | Path | None = None,
    libero_root_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    selected_splits = normalize_splits(splits)
    _require_test_permission(selected_splits, after_stage1_export)
    tag = split_label(selected_splits)
    manifests = [
        collect_task(
            dynamic_config=dynamic_config,
            task=task,
            output_root=root,
            splits=selected_splits,
            after_stage1_export=after_stage1_export,
            libero_root_path=libero_root_path,
            overwrite=overwrite,
        )
        for task in task_names_for_suite(dynamic_config.evaluation.suite)
    ]
    summary = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_branch_collection_summary",
        "splits": list(selected_splits),
        "task_manifests": [
            str(_task_manifest_path(root, tag, task))
            for task in task_names_for_suite(dynamic_config.evaluation.suite)
        ],
        "state_count": sum(int(item["state_count"]) for item in manifests),
        "branch_trajectory_count": sum(int(item["branch_trajectory_count"]) for item in manifests),
        "per_camera_rgb_frame_count": sum(int(item["per_camera_rgb_frame_count"]) for item in manifests),
        "split_counts": {
            split: sum(int(item["split_counts"][split]) for item in manifests)
            for split in selected_splits
        },
    }
    expected = {
        "state_count": sum(SPLIT_STATE_COUNTS[split] for split in selected_splits),
        "branch_trajectory_count": sum(SPLIT_STATE_COUNTS[split] for split in selected_splits) * len(BRANCH_NAMES),
        "per_camera_rgb_frame_count": sum(SPLIT_STATE_COUNTS[split] for split in selected_splits) * len(BRANCH_NAMES) * 32 * 2,
        "split_counts": {split: SPLIT_STATE_COUNTS[split] for split in selected_splits},
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise Stage1ContractError(f"formal collection summary mismatch for {key}: {summary[key]} vs {value}")
    _write_json(root / "collection" / tag / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--after-stage1-export", type=Path)
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dynamic = load_config(args.config)
    if args.task:
        result: Any = collect_task(
            dynamic_config=dynamic,
            task=args.task,
            output_root=args.output_root,
            splits=args.splits,
            after_stage1_export=args.after_stage1_export,
            libero_root_path=args.libero_root,
            overwrite=args.overwrite,
        )
    else:
        result = collect_all(
            dynamic_config=dynamic,
            output_root=args.output_root,
            splits=args.splits,
            after_stage1_export=args.after_stage1_export,
            libero_root_path=args.libero_root,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
