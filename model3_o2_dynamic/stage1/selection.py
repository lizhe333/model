"""Deterministic motion-aware source-state selection for Dynamic Stage 1."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch

from model3_o2_dynamic.config import Model3O2DynamicConfig, load_config

from .common import (
    SOURCE_RUNTIME_RESTORE_CONTRACT,
    capture_static_model_state,
    close_env,
    libero_root,
    motion_target_hints,
    motion_labels,
    motion_snapshot,
    object_hints,
    restore_source_runtime,
    restore_static_model_state,
    sample_id,
    sha256_file,
    split_for_demo,
    torch_record,
)
from .contracts import (
    SPLIT_STATE_COUNTS,
    Stage1DataConfig,
    Stage1ContractError,
    demonstration_directory_for_suite,
    task_filenames_for_suite,
    task_names_for_suite,
)


MOTION_LABELS = ("object_motion", "robot_motion", "camera_motion", "contact_interaction", "static_control")
SUPPORTED_SPLITS = ("train", "validation", "test")


class CoverageFailure(Stage1ContractError):
    """Raised before collection if deterministic candidates cannot meet coverage quotas."""


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _static_model_state_path(root: Path, tag: str, task: str) -> Path:
    return root / "selection" / tag / f"{task}.static_model_state.pt"


def _static_model_state_payload(
    *,
    task: str,
    splits: tuple[str, ...],
    bddl_path: Path,
    source_hdf5: Path,
    source_hdf5_sha256: str,
    values: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_static_model_state",
        "source_restore_contract": SOURCE_RUNTIME_RESTORE_CONTRACT,
        "task": task,
        "splits": list(splits),
        "bddl_file": str(bddl_path),
        "source_hdf5": str(source_hdf5),
        "source_hdf5_sha256": source_hdf5_sha256,
        "arrays": {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in values.items()},
    }


def _load_static_model_state(
    *,
    path: Path,
    task: str,
    splits: tuple[str, ...],
    bddl_path: Path,
    source_hdf5: Path,
    source_hdf5_sha256: str,
    env: Any,
    overwrite: bool,
) -> tuple[dict[str, np.ndarray], str]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        valid = (
            payload.get("artifact_kind") == "stage1_static_model_state"
            and payload.get("track_id") == "model3_o2_dynamic"
            and payload.get("source_restore_contract") == SOURCE_RUNTIME_RESTORE_CONTRACT
            and payload.get("task") == task
            and tuple(payload.get("splits", ())) == splits
            and payload.get("bddl_file") == str(bddl_path)
            and payload.get("source_hdf5") == str(source_hdf5)
            and payload.get("source_hdf5_sha256") == source_hdf5_sha256
        )
        arrays = payload.get("arrays") if valid else None
        if isinstance(arrays, dict):
            try:
                values = {
                    str(name): tensor.numpy().copy()
                    for name, tensor in arrays.items()
                    if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                }
                restore_static_model_state(env, values)
                return values, sha256_file(path)
            except (Stage1ContractError, TypeError, ValueError):
                pass
        if not overwrite:
            raise Stage1ContractError(f"invalid static model-state provenance: {path}")

    seed = getattr(env, "seed", None)
    if not callable(seed):
        raise TypeError("LIBERO environment does not expose a callable seed method")
    seed(42)
    env.reset()
    values = capture_static_model_state(env)
    payload = _static_model_state_payload(
        task=task,
        splits=splits,
        bddl_path=bddl_path,
        source_hdf5=source_hdf5,
        source_hdf5_sha256=source_hdf5_sha256,
        values=values,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    temporary.replace(path)
    return values, sha256_file(path)


def _demo_key(demo_id: int) -> str:
    return f"demo_{int(demo_id)}"


def normalize_splits(splits: Iterable[str]) -> tuple[str, ...]:
    """Return the canonical split order and reject an accidental test read."""

    requested = {str(value) for value in splits}
    unknown = requested - set(SUPPORTED_SPLITS)
    if unknown or not requested:
        raise ValueError(f"unsupported Dynamic Stage-1 selection splits: {sorted(requested)}")
    return tuple(split for split in SUPPORTED_SPLITS if split in requested)


def split_label(splits: Iterable[str]) -> str:
    return "_".join(normalize_splits(splits))


def _require_test_permission(
    splits: Iterable[str],
    after_stage1_export: str | Path | None,
) -> None:
    if "test" not in normalize_splits(splits):
        return
    if after_stage1_export is None or not Path(after_stage1_export).expanduser().is_file():
        raise Stage1ContractError(
            "test source selection is sealed until the fixed Stage-1 adapter export exists; "
            "pass --after-stage1-export after step 5K"
        )


def _coverage_requirements(total: int, config: Stage1DataConfig) -> dict[str, int]:
    return {
        "object_motion": int(np.ceil(config.min_object_motion_fraction * total)),
        "robot_motion": int(np.ceil(config.min_robot_motion_fraction * total)),
        "camera_motion": int(np.ceil(config.min_camera_motion_fraction * total)),
        "contact_interaction": int(np.ceil(config.min_contact_interaction_fraction * total)),
        "static_control_max": int(np.floor(config.max_static_control_fraction * total)),
    }


def coverage_summary(records: Iterable[dict[str, Any]], config: Stage1DataConfig) -> dict[str, Any]:
    values = list(records)
    counts = Counter()
    for record in values:
        labels = record.get("motion_labels")
        if not isinstance(labels, dict):
            raise Stage1ContractError("selected record lacks motion_labels")
        for label in MOTION_LABELS:
            counts[label] += int(bool(labels.get(label, False)))
    total = len(values)
    required = _coverage_requirements(total, config)
    passed = (
        counts["object_motion"] >= required["object_motion"]
        and counts["robot_motion"] >= required["robot_motion"]
        and counts["camera_motion"] >= required["camera_motion"]
        and counts["contact_interaction"] >= required["contact_interaction"]
        and counts["static_control"] <= required["static_control_max"]
    )
    return {
        "state_count": total,
        "counts": {name: int(counts[name]) for name in MOTION_LABELS},
        "fractions": {name: (float(counts[name]) / total if total else 0.0) for name in MOTION_LABELS},
        "requirements": required,
        "passed": passed,
    }


def _score_candidate(
    candidate: dict[str, Any],
    *,
    current_counts: Counter,
    requirements: dict[str, int],
) -> float:
    """Deficit-aware score; ties are resolved separately by earliest timestep."""

    labels = candidate["motion_labels"]
    score = 0.0
    for label in ("object_motion", "robot_motion", "camera_motion", "contact_interaction"):
        deficit = max(0, requirements[label] - int(current_counts[label]))
        if labels.get(label, False) and deficit:
            # Normalized deficits prevent a large robot quota from drowning out
            # contact candidates near the end of greedy selection.
            score += float(deficit) / max(requirements[label], 1)
    if labels.get("static_control", False):
        overflow = max(0, int(current_counts["static_control"]) + 1 - requirements["static_control_max"])
        score -= 2.0 + float(overflow)
    return score


def _selection_counts(records: Iterable[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for record in records:
        labels = record["motion_labels"]
        for label in MOTION_LABELS:
            counts[label] += int(bool(labels.get(label, False)))
    return counts


def _coverage_penalty(counts: Counter, requirements: dict[str, int]) -> int:
    minimums = ("object_motion", "robot_motion", "camera_motion", "contact_interaction")
    return sum(max(0, int(requirements[label]) - int(counts[label])) for label in minimums) + max(
        0,
        int(counts["static_control"]) - int(requirements["static_control_max"]),
    )


def _repair_coverage_selection(
    selected: list[dict[str, Any]],
    candidates_by_demo: dict[int, dict[int, list[dict[str, Any]]]],
    *,
    config: Stage1DataConfig,
) -> list[dict[str, Any]]:
    """Deterministically exchange same-bin candidates until quotas pass or prove no repair exists."""

    requirements = _coverage_requirements(len(selected), config)
    counts = _selection_counts(selected)
    slot_order = [
        (demo_id, progress_bin)
        for demo_id in sorted(candidates_by_demo)
        for progress_bin in range(config.states_per_demo)
    ]
    if len(slot_order) != len(selected):
        raise CoverageFailure("selection slot cardinality changed before coverage repair")
    slots = {slot: index for index, slot in enumerate(slot_order)}
    minimums = ("object_motion", "robot_motion", "camera_motion", "contact_interaction")
    while True:
        penalty = _coverage_penalty(counts, requirements)
        if penalty == 0:
            return selected
        best: tuple[tuple[int, int, int, int], int, dict[str, Any], Counter] | None = None
        for (demo_id, progress_bin), index in sorted(slots.items()):
            current = selected[index]
            current_labels = current["motion_labels"]
            for candidate in sorted(
                candidates_by_demo[demo_id][progress_bin], key=lambda item: int(item["source_index"])
            ):
                if int(candidate["source_index"]) == int(current["source_index"]):
                    continue
                candidate_labels = candidate["motion_labels"]
                trial_counts = counts.copy()
                for label in MOTION_LABELS:
                    trial_counts[label] += int(bool(candidate_labels.get(label, False))) - int(
                        bool(current_labels.get(label, False))
                    )
                trial_penalty = _coverage_penalty(trial_counts, requirements)
                if trial_penalty >= penalty:
                    continue
                if any(
                    int(counts[label]) >= int(requirements[label])
                    and int(trial_counts[label]) < int(requirements[label])
                    for label in minimums
                ):
                    continue
                if (
                    int(counts["static_control"]) <= int(requirements["static_control_max"])
                    and int(trial_counts["static_control"]) > int(requirements["static_control_max"])
                ):
                    continue
                choice_key = (trial_penalty, int(candidate["source_index"]), demo_id, progress_bin)
                candidate_choice = (choice_key, index, candidate, trial_counts)
                if best is None or candidate_choice[0] < best[0]:
                    best = candidate_choice
        if best is None:
            return selected
        _, index, replacement, counts = best
        selected[index] = replacement


def select_candidates_for_split(
    candidates_by_demo: dict[int, dict[int, list[dict[str, Any]]]],
    *,
    config: Stage1DataConfig,
) -> list[dict[str, Any]]:
    """Select one earliest tie-broken candidate in each progress bin of each demo."""

    if not candidates_by_demo:
        raise CoverageFailure("cannot select an empty split")
    expected = len(candidates_by_demo) * config.states_per_demo
    requirements = _coverage_requirements(expected, config)
    counts: Counter = Counter()
    output: list[dict[str, Any]] = []
    for demo_id in sorted(candidates_by_demo):
        bins = candidates_by_demo[demo_id]
        if tuple(sorted(bins)) != tuple(range(config.states_per_demo)):
            raise CoverageFailure(
                f"demo {demo_id} does not expose exactly ten non-empty progress bins: {sorted(bins)}"
            )
        for progress_bin in range(config.states_per_demo):
            candidates = bins[progress_bin]
            if not candidates:
                raise CoverageFailure(f"demo {demo_id} progress bin {progress_bin} is empty")
            chosen = max(
                candidates,
                key=lambda item: (
                    _score_candidate(item, current_counts=counts, requirements=requirements),
                    -int(item["source_index"]),
                ),
            )
            output.append(chosen)
            for label in MOTION_LABELS:
                counts[label] += int(bool(chosen["motion_labels"].get(label, False)))
    output = _repair_coverage_selection(output, candidates_by_demo, config=config)
    summary = coverage_summary(output, config)
    if not summary["passed"]:
        raise CoverageFailure(
            "deterministic motion-aware selection cannot satisfy coverage quota: "
            + json.dumps(summary, sort_keys=True)
        )
    return output


def _make_env(*, bddl_path: Path, resolution: int) -> Any:
    # Keep tests importable on machines without MuJoCo/LIBERO.
    from libero.libero.envs import OffScreenRenderEnv

    # Source selection reads simulator state, contacts, and wrist-camera
    # extrinsics only. It never consumes RGB pixels; disabling camera
    # observables avoids redundant rendering. Every candidate is subsequently
    # restored from a complete MuJoCo state and checked exactly, so a soft
    # reset preserves the selection contract while avoiding XML reconstruction
    # for each legal source timestep.
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=int(resolution),
        camera_widths=int(resolution),
        use_camera_obs=False,
        hard_reset=False,
    )
    env.seed(42)
    return env


def _candidate_motion_labels(
    env: Any,
    *,
    source_state: np.ndarray,
    expert_actions: np.ndarray,
    config: Stage1DataConfig,
    static_model_state: dict[str, np.ndarray],
) -> tuple[dict[str, bool], str]:
    seed = getattr(env, "seed", None)
    if not callable(seed):
        raise TypeError("LIBERO environment does not expose a callable seed method")
    seed(42)
    env.reset()
    restore_static_model_state(env, static_model_state)
    restored = restore_source_runtime(env, source_state, tolerance=config.restore_state_max_abs)
    hints = motion_target_hints(env)
    trace = [motion_snapshot(env, restored["obs"], hints)]
    for action in expert_actions:
        obs, _, _, _ = env.step(action)
        trace.append(motion_snapshot(env, obs, hints))
    return motion_labels(trace, config), str(trace[0]["target_position_key"])


def _enumerate_demo_candidates(
    env: Any,
    *,
    task: str,
    task_position: int,
    demo_id: int,
    states: np.ndarray,
    actions: np.ndarray,
    instruction: str,
    config: Stage1DataConfig,
    static_model_state: dict[str, np.ndarray],
) -> dict[int, list[dict[str, Any]]]:
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise Stage1ContractError(f"{task} demo {demo_id} action tensor must be [T,7]")
    max_source = int(len(actions) - config.action_horizon)
    if max_source < config.states_per_demo - 1:
        raise CoverageFailure(
            f"{task} demo {demo_id} has only {max_source + 1} legal source timesteps; need ten bins"
        )
    if len(states) <= max_source:
        raise Stage1ContractError(
            f"{task} demo {demo_id} state/action alignment lacks a state for source {max_source}"
        )
    bins: dict[int, list[dict[str, Any]]] = {index: [] for index in range(config.states_per_demo)}
    denominator = max(max_source, 1)
    for source_index in range(max_source + 1):
        progress = float(source_index) / denominator
        progress_bin = min(int(progress * config.states_per_demo), config.states_per_demo - 1)
        source_state = np.asarray(states[source_index], dtype=np.float64).copy()
        expert = np.asarray(actions[source_index : source_index + config.action_horizon], dtype=np.float32).copy()
        labels, target_key = _candidate_motion_labels(
            env,
            source_state=source_state,
            expert_actions=expert,
            config=config,
            static_model_state=static_model_state,
        )
        bins[progress_bin].append(
            {
                "task": task,
                "task_position": int(task_position),
                "instruction": instruction,
                "demo_id": int(demo_id),
                "split": split_for_demo(demo_id, config),
                "source_index": int(source_index),
                "episode_progress": progress,
                "progress_bin": int(progress_bin),
                "source_state": torch_record(source_state),
                "expert_actions": torch_record(expert, dtype=torch.float32),
                "motion_labels": labels,
                "target_position_key": target_key,
            }
        )
    return bins


def _validate_selection_payload(
    payload: dict[str, Any],
    *,
    task: str,
    config: Stage1DataConfig,
    expected_splits: Iterable[str] | None = None,
) -> None:
    if payload.get("artifact_kind") != "stage1_source_selection" or payload.get("task") != task:
        raise Stage1ContractError(f"invalid Stage 1 selection identity for {task}")
    if payload.get("source_restore_contract") != SOURCE_RUNTIME_RESTORE_CONTRACT:
        raise Stage1ContractError(f"{task} selection source-runtime restore contract changed")
    if not isinstance(payload.get("static_model_state_path"), str) or not isinstance(
        payload.get("static_model_state_sha256"), str
    ):
        raise Stage1ContractError(f"{task} selection lacks static model-state provenance")
    selected_splits = normalize_splits(payload.get("splits", ()))
    if expected_splits is not None and selected_splits != normalize_splits(expected_splits):
        raise Stage1ContractError(
            f"{task} selection split identity changed: {selected_splits} vs {normalize_splits(expected_splits)}"
        )
    records = payload.get("records")
    selected_demo_ids = [
        demo_id
        for demo_id in range(config.demos_per_task)
        if config.split_for_demo(demo_id) in selected_splits
    ]
    expected_count = len(selected_demo_ids) * config.states_per_demo
    if not isinstance(records, list) or len(records) != expected_count:
        raise Stage1ContractError(f"{task} selection must contain {expected_count} records")
    observed = [(int(row["demo_id"]), int(row["progress_bin"])) for row in records]
    expected = [(demo, progress_bin) for demo in selected_demo_ids for progress_bin in range(10)]
    if observed != expected:
        raise Stage1ContractError(f"{task} selection demo/progress order is not split-stable")
    for row in records:
        if not isinstance(row.get("source_state"), torch.Tensor) or not isinstance(row.get("expert_actions"), torch.Tensor):
            raise Stage1ContractError("selection source state/action must be tensors")
        if tuple(row["expert_actions"].shape) != (8, 7):
            raise Stage1ContractError("selection action shape changed")
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in selected_splits}
    for row in records:
        if row["split"] not in grouped:
            raise Stage1ContractError(f"{task} selection contains an unrequested split")
        grouped[row["split"]].append(row)
    for split in selected_splits:
        split_expected_count = sum(
            1 for demo_id in selected_demo_ids if config.split_for_demo(demo_id) == split
        ) * config.states_per_demo
        if len(grouped[split]) != split_expected_count:
            raise Stage1ContractError(f"{task} {split} selection count changed")
        if not coverage_summary(grouped[split], config)["passed"]:
            raise CoverageFailure(f"{task} {split} selection no longer meets motion coverage")


def select_task(
    *,
    dynamic_config: Model3O2DynamicConfig,
    task: str,
    output_root: str | Path,
    splits: Iterable[str] = ("train", "validation"),
    after_stage1_export: str | Path | None = None,
    libero_root_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Replay all legal expert continuations and seal one selected state/bin."""

    config = Stage1DataConfig()
    config.validate()
    selected_splits = normalize_splits(splits)
    _require_test_permission(selected_splits, after_stage1_export)
    suite = dynamic_config.evaluation.suite
    task_names = task_names_for_suite(suite)
    if task not in task_names:
        raise ValueError(f"unknown Dynamic Stage-1 task for {suite}: {task}")
    task_position = task_names.index(task)
    root = Path(output_root).expanduser().resolve()
    output_path = root / "selection" / split_label(selected_splits) / f"{task}.pt"
    if output_path.is_file() and not overwrite:
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        _validate_selection_payload(payload, task=task, config=config, expected_splits=selected_splits)
        return output_path

    libero = libero_root(libero_root_path)
    source_path = (
        libero
        / "datasets"
        / demonstration_directory_for_suite(suite)
        / task_filenames_for_suite(suite)[task]
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"missing {suite} demonstration HDF5: {source_path}")
    started = time.time()
    source_hdf5_sha256 = sha256_file(source_path)
    selected_records: list[dict[str, Any]] = []
    candidate_counts: dict[int, int] = {}
    with h5py.File(source_path, "r") as source:
        data = source["data"]
        bddl_relative = str(data.attrs["bddl_file_name"])
        bddl_path = libero / bddl_relative
        if not bddl_path.is_file():
            raise FileNotFoundError(f"missing LIBERO BDDL: {bddl_path}")
        problem_info = json.loads(data.attrs["problem_info"])
        instruction = str(problem_info["language_instruction"])
        env = _make_env(bddl_path=bddl_path, resolution=config.render_resolution)
        try:
            static_model_path = _static_model_state_path(root, split_label(selected_splits), task)
            static_model_state, static_model_state_sha256 = _load_static_model_state(
                path=static_model_path,
                task=task,
                splits=selected_splits,
                bddl_path=bddl_path,
                source_hdf5=source_path,
                source_hdf5_sha256=source_hdf5_sha256,
                env=env,
                overwrite=overwrite,
            )
            by_split: dict[str, dict[int, dict[int, list[dict[str, Any]]]]] = {
                split: {} for split in selected_splits
            }
            for demo_id in range(config.demos_per_task):
                demo_split = split_for_demo(demo_id, config)
                if demo_split not in selected_splits:
                    continue
                key = _demo_key(demo_id)
                if key not in data:
                    raise Stage1ContractError(f"{task} lacks required HDF5 demonstration {key}")
                group = data[key]
                candidates = _enumerate_demo_candidates(
                    env,
                    task=task,
                    task_position=task_position,
                    demo_id=demo_id,
                    states=np.asarray(group["states"], dtype=np.float64),
                    actions=np.asarray(group["actions"], dtype=np.float32),
                    instruction=instruction,
                    config=config,
                    static_model_state=static_model_state,
                )
                candidate_counts[demo_id] = sum(len(value) for value in candidates.values())
                by_split[demo_split][demo_id] = candidates
                print(f"[dynamic-select:{task}] demo={demo_id + 1}/50 candidates={candidate_counts[demo_id]}", flush=True)
            for split in selected_splits:
                chosen = select_candidates_for_split(by_split[split], config=config)
                for row in chosen:
                    row["sample_id"] = sample_id(task_position, row["demo_id"], row["progress_bin"])
                selected_records.extend(chosen)
        finally:
            close_env(env)
    selected_records.sort(key=lambda row: (int(row["demo_id"]), int(row["progress_bin"])))
    payload = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_source_selection",
        "task": task,
        "task_position": task_position,
        "splits": list(selected_splits),
        "source_hdf5": str(source_path),
        "source_hdf5_sha256": source_hdf5_sha256,
        "bddl_file": str(bddl_path),
        "source_restore_contract": SOURCE_RUNTIME_RESTORE_CONTRACT,
        "static_model_state_path": str(static_model_path),
        "static_model_state_sha256": static_model_state_sha256,
        "data_contract": config.as_dict(),
        "candidate_counts_by_demo": candidate_counts,
        "records": selected_records,
        "started_unix": started,
        "completed_unix": time.time(),
    }
    _validate_selection_payload(payload, task=task, config=config, expected_splits=selected_splits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    _write_json(
        output_path.with_suffix(".manifest.json"),
        {
            key: value
            for key, value in payload.items()
            if key not in {"records", "candidate_counts_by_demo"}
        }
        | {
            "record_count": len(selected_records),
            "candidate_counts_by_demo": candidate_counts,
            "coverage_by_split": {
                split: coverage_summary(
                    [row for row in selected_records if row["split"] == split], config
                )
                for split in selected_splits
            },
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
        },
    )
    return output_path


def select_all(
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
    paths = [
        select_task(
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
    config = Stage1DataConfig()
    all_records: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _validate_selection_payload(
            payload,
            task=str(payload["task"]),
            config=config,
            expected_splits=selected_splits,
        )
        all_records.extend(payload["records"])
    summary = {
        split: coverage_summary([row for row in all_records if row["split"] == split], config)
        for split in selected_splits
    }
    expected_count = sum(SPLIT_STATE_COUNTS[split] for split in selected_splits)
    if len(all_records) != expected_count or not all(value["passed"] for value in summary.values()):
        raise CoverageFailure(f"formal global selection failed: count={len(all_records)}, coverage={summary}")
    result = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_source_selection_summary",
        "splits": list(selected_splits),
        "task_paths": [str(path) for path in paths],
        "task_shas": {path.stem: sha256_file(path) for path in paths},
        "state_count": len(all_records),
        "split_counts": {
            split: sum(1 for row in all_records if row["split"] == split)
            for split in selected_splits
        },
        "coverage_by_split": summary,
    }
    _write_json(root / "selection" / split_label(selected_splits) / "summary.json", result)
    return result


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
        result: Any = str(
            select_task(
                dynamic_config=dynamic,
                task=args.task,
                output_root=args.output_root,
                splits=args.splits,
                after_stage1_export=args.after_stage1_export,
                libero_root_path=args.libero_root,
                overwrite=args.overwrite,
            )
        )
    else:
        result = select_all(
            dynamic_config=dynamic,
            output_root=args.output_root,
            splits=args.splits,
            after_stage1_export=args.after_stage1_export,
            libero_root_path=args.libero_root,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, sort_keys=True) if isinstance(result, dict) else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
