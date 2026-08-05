"""Hard cache and optimizer contracts for Dynamic O2 response warmup."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch


LAYERS = (8, 16, 24)
BRANCH_NAMES = ("expert", "local_delta", "gripper_flip", "zero")
RESPONSE_BRANCH_NAMES = BRANCH_NAMES[:3]
CURRENT_SHAPE = (392, 1536)
ACTION_SHAPE = (8, 7)
TARGET_SHAPE = (4, 256)

# Stage-1 task order is part of sample identity.  Keep the formal suite maps
# here, rather than borrowing the historical four-task Gate-0 config, so the
# Dynamic pipeline cannot silently fall back to a smaller diagnostic set.
OBJECT_TASKS: tuple[tuple[str, str], ...] = (
    ("pick_up_the_alphabet_soup_and_place_it_in_the_basket", "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"),
    ("pick_up_the_cream_cheese_and_place_it_in_the_basket", "pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5"),
    ("pick_up_the_salad_dressing_and_place_it_in_the_basket", "pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5"),
    ("pick_up_the_bbq_sauce_and_place_it_in_the_basket", "pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5"),
    ("pick_up_the_ketchup_and_place_it_in_the_basket", "pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5"),
    ("pick_up_the_tomato_sauce_and_place_it_in_the_basket", "pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5"),
    ("pick_up_the_butter_and_place_it_in_the_basket", "pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5"),
    ("pick_up_the_milk_and_place_it_in_the_basket", "pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5"),
    ("pick_up_the_chocolate_pudding_and_place_it_in_the_basket", "pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5"),
    ("pick_up_the_orange_juice_and_place_it_in_the_basket", "pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5"),
)
OBJECT_TASK_NAMES = tuple(name for name, _ in OBJECT_TASKS)
OBJECT_TASK_FILENAMES = dict(OBJECT_TASKS)

# This order is the registered LIBERO_10 evaluator order.  It is intentionally
# not filesystem order: ``task_position`` enters deterministic perturbations
# and therefore must remain aligned with the Long suite's policy/evaluator
# identity.
LONG_TASKS: tuple[tuple[str, str], ...] = (
    (
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5",
    ),
    (
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5",
    ),
    (
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
    ),
    (
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5",
    ),
    (
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5",
    ),
    (
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
    ),
    (
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5",
    ),
    (
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5",
    ),
    (
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5",
    ),
    (
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5",
    ),
)
LONG_TASK_NAMES = tuple(name for name, _ in LONG_TASKS)
LONG_TASK_FILENAMES = dict(LONG_TASKS)

TASKS_BY_SUITE: dict[str, tuple[tuple[str, str], ...]] = {
    "libero_object": OBJECT_TASKS,
    "libero_10": LONG_TASKS,
}
DEMONSTRATION_DIRECTORY_BY_SUITE = {
    "libero_object": "libero_object",
    "libero_10": "libero_10",
}


def task_specs_for_suite(suite: str) -> tuple[tuple[str, str], ...]:
    try:
        return TASKS_BY_SUITE[str(suite)]
    except KeyError as error:
        raise Stage1ContractError(
            f"Dynamic Stage 1 supports only {sorted(TASKS_BY_SUITE)}, got {suite!r}"
        ) from error


def task_names_for_suite(suite: str) -> tuple[str, ...]:
    return tuple(name for name, _ in task_specs_for_suite(suite))


def task_subset_for_suite(suite: str, tasks: Iterable[str] | None = None) -> tuple[str, ...]:
    """Resolve an optional task subset in the registered suite order."""

    available = task_names_for_suite(suite)
    if tasks is None:
        return available
    requested = tuple(dict.fromkeys(str(task) for task in tasks))
    if not requested:
        raise Stage1ContractError("task subset must not be empty")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise Stage1ContractError(f"unknown Dynamic Stage-1 tasks for {suite}: {unknown}")
    return tuple(task for task in available if task in requested)


def task_filenames_for_suite(suite: str) -> dict[str, str]:
    return dict(task_specs_for_suite(suite))


def demonstration_directory_for_suite(suite: str) -> str:
    try:
        return DEMONSTRATION_DIRECTORY_BY_SUITE[str(suite)]
    except KeyError as error:
        raise Stage1ContractError(
            f"Dynamic Stage 1 has no demonstration directory for suite {suite!r}"
        ) from error

SPLIT_BY_DEMO = (
    ("train", range(0, 40)),
    ("validation", range(40, 45)),
    ("test", range(45, 50)),
)
SPLIT_STATE_COUNTS = {"train": 4_000, "validation": 500, "test": 500}
SPLIT_DEMO_COUNTS = {"train": 40, "validation": 5, "test": 5}


@dataclass(frozen=True)
class Stage1DataConfig:
    """Non-negotiable simulator, branch, and target geometry for Stage 1."""

    demos_per_task: int = 50
    states_per_demo: int = 10
    action_horizon: int = 8
    render_ticks_per_action: int = 4
    stage_render_ticks: tuple[int, ...] = (8, 16, 24, 32)
    render_resolution: int = 256
    model_view_resolution: int = 224
    motion_perturbation: float = 0.20
    zero_gripper_action: float = -1.0
    restore_state_max_abs: float = 1.0e-10
    repeat_terminal_state_max_abs: float = 1.0e-8
    object_motion_m: float = 0.01
    robot_motion_m: float = 0.02
    camera_motion_m: float = 0.02
    rotation_motion_degrees: float = 10.0
    max_static_control_fraction: float = 0.10
    min_object_motion_fraction: float = 0.30
    min_robot_motion_fraction: float = 0.40
    min_camera_motion_fraction: float = 0.30
    min_contact_interaction_fraction: float = 0.25
    fit_noise_seeds: tuple[int, ...] = (84001, 84002, 84003, 84004)
    heldout_noise_seeds: tuple[int, ...] = (84005, 84006)
    teacher_timestep: int = 250
    projection_seed: int = 83021
    projection_dim: int = 256

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def split_for_demo(self, demo_id: int) -> str:
        for split, demo_ids in SPLIT_BY_DEMO:
            if int(demo_id) in demo_ids:
                return split
        raise Stage1ContractError(f"demo_id {demo_id} is outside the formal 0--49 range")

    def validate(self) -> None:
        errors: list[str] = []
        if self.demos_per_task != 50 or self.states_per_demo != 10:
            errors.append("Stage 1 requires 50 demonstrations and 10 states per demonstration")
        if self.action_horizon != 8 or self.render_ticks_per_action != 4:
            errors.append("Stage 1 requires an 8-action horizon with four render ticks per action")
        if self.stage_render_ticks != (8, 16, 24, 32):
            errors.append("Stage 1 process positions must be render ticks 8,16,24,32")
        if self.motion_perturbation != 0.20 or self.zero_gripper_action != -1.0:
            errors.append("Stage 1 branch perturbation / zero-action semantics changed")
        if self.fit_noise_seeds != (84001, 84002, 84003, 84004):
            errors.append("Stage 1 fit-noise seeds changed")
        if self.heldout_noise_seeds != (84005, 84006):
            errors.append("Stage 1 held-out-noise seeds changed")
        if self.teacher_timestep != 250 or self.projection_seed != 83021 or self.projection_dim != 256:
            errors.append("Stage 1 E0 teacher geometry changed")
        if errors:
            raise Stage1ContractError("Invalid Stage 1 data contract:\n- " + "\n- ".join(errors))


class Stage1ContractError(ValueError):
    """Raised before any Stage 1 tensor can be used incorrectly."""


@dataclass(frozen=True)
class Stage1TrainConfig:
    seed: int = 85101
    max_steps: int = 5000
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    beta_anchor: float = 1.0
    teacher_timestep: int = 250
    validation_every: int = 250
    checkpoint_steps: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000)
    predictor_width: int = 128
    max_predictor_parameters: int = 2_500_000

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Stage1TrainConfig":
        values = dict(raw)
        if "checkpoint_steps" in values:
            values["checkpoint_steps"] = tuple(int(value) for value in values["checkpoint_steps"])
        result = cls(**values)
        errors: list[str] = []
        if result.seed != 85101:
            errors.append("Stage 1 seed must be 85101")
        if result.max_steps != 5000:
            errors.append("Stage 1 must run exactly 5000 optimizer steps")
        if result.batch_size != 64:
            errors.append("Stage 1 effective batch size must be 64")
        if result.learning_rate != 3.0e-4:
            errors.append("Stage 1 learning rate must be 3e-4")
        if result.weight_decay != 1.0e-4:
            errors.append("Stage 1 weight decay must be 1e-4")
        if result.beta_anchor != 1.0:
            errors.append("Stage 1 anchor coefficient beta must be 1")
        if result.teacher_timestep != 250:
            errors.append("Stage 1 teacher timestep must be 250")
        if result.validation_every != 250:
            errors.append("Stage 1 validation cadence must be 250 steps")
        if result.checkpoint_steps != (1000, 2000, 3000, 4000, 5000):
            errors.append("Stage 1 checkpoint steps must be 1K,2K,3K,4K,5K")
        if result.max_predictor_parameters > 2_500_000:
            errors.append("Stage 1 predictor parameter cap cannot exceed 2.5M")
        if errors:
            raise Stage1ContractError("Invalid Stage 1 contract:\n- " + "\n- ".join(errors))
        return result

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _tensor_shape(payload: dict[str, Any], key: str, errors: list[str]) -> tuple[int, ...] | None:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        errors.append(f"cache `{key}` must be a torch.Tensor")
        return None
    if not torch.isfinite(value.float()).all():
        errors.append(f"cache `{key}` contains non-finite values")
    return tuple(int(dim) for dim in value.shape)


def _current_hidden_shape(
    payload: dict[str, Any],
    *,
    count: int,
    errors: list[str],
) -> tuple[int, ...] | None:
    """Accept either a compact test tensor or the formal mmap-able carrier shards.

    A formal cache contains $4{,}500\times3\times392\times1536$ BF16 values,
    which is about 16 GiB.  Loading that tensor just to draw a batch would make
    the fixed Stage-1 experiment needlessly fragile.  The shard form keeps the
    exact uncompressed carrier while allowing the trainer to map only the
    selected per-demo tensors.  Small unit tests may still use a contiguous
    ``current_hidden`` tensor.
    """

    value = payload.get("current_hidden")
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value.float()).all():
            errors.append("cache `current_hidden` contains non-finite values")
        return tuple(int(dim) for dim in value.shape)
    if value is not None:
        errors.append("cache `current_hidden` must be a torch.Tensor or null in sharded mode")
        return None
    shards = payload.get("current_hidden_shards")
    index = payload.get("current_hidden_index")
    if not isinstance(shards, list) or not shards:
        errors.append("sharded cache requires non-empty current_hidden_shards")
        return None
    if not isinstance(index, torch.Tensor) or tuple(index.shape) != (count, 2):
        errors.append("sharded cache current_hidden_index must be [N,2]")
        return None
    if index.dtype not in {torch.int32, torch.int64}:
        errors.append("sharded cache current_hidden_index must be an integer tensor")
        return None
    total = 0
    for shard_idx, shard in enumerate(shards):
        if not isinstance(shard, dict):
            errors.append(f"current_hidden_shards[{shard_idx}] must be a dict")
            continue
        shape = tuple(int(dim) for dim in shard.get("shape", ()))
        if len(shape) != 4 or shape[1:] != (len(LAYERS), *CURRENT_SHAPE):
            errors.append(
                f"current_hidden_shards[{shard_idx}] shape must be [M,3,392,1536], got {shape}"
            )
            continue
        if shape[0] <= 0:
            errors.append(f"current_hidden_shards[{shard_idx}] is empty")
            continue
        if not isinstance(shard.get("path"), str) or not shard["path"]:
            errors.append(f"current_hidden_shards[{shard_idx}] lacks an absolute path")
        if not isinstance(shard.get("sha256"), str) or len(shard["sha256"]) != 64:
            errors.append(f"current_hidden_shards[{shard_idx}] lacks a SHA-256")
        total += shape[0]
    if len(shards) and index.numel():
        shard_ids = index[:, 0].long()
        local_ids = index[:, 1].long()
        if bool((shard_ids < 0).any()) or bool((shard_ids >= len(shards)).any()):
            errors.append("current_hidden_index refers to an invalid shard")
        else:
            for shard_idx, shard in enumerate(shards):
                shape = tuple(int(dim) for dim in shard.get("shape", ()))
                if len(shape) == 4:
                    selected = local_ids[shard_ids == shard_idx]
                    if selected.numel() and (
                        bool((selected < 0).any()) or bool((selected >= shape[0]).any())
                    ):
                        errors.append(f"current_hidden_index has an invalid local index for shard {shard_idx}")
    if total < count:
        errors.append("current_hidden shards contain fewer rows than cache records")
    return (count, len(LAYERS), *CURRENT_SHAPE)


def validate_response_cache(
    payload: dict[str, Any],
    *,
    require_trainable_splits_only: bool,
) -> dict[str, Any]:
    """Validate the frozen E0 cache before it can enter Stage 1.

    The train cache intentionally contains no test state or held-out-noise
    target.  A separately materialized test cache may pass with
    ``require_trainable_splits_only=False`` after fixed step-5K export exists.
    """

    if not isinstance(payload, dict):
        raise Stage1ContractError("Stage 1 response cache must be a dictionary")
    errors: list[str] = []
    _require(payload.get("schema_version") == 1, "cache schema_version must be 1", errors)
    _require(payload.get("track_id") == "model3_o2_dynamic", "wrong cache track_id", errors)
    _require(payload.get("artifact_kind") == "stage1_response_cache", "wrong cache artifact_kind", errors)
    _require(tuple(payload.get("layers", ())) == LAYERS, "cache layers must be [8,16,24]", errors)
    _require(tuple(payload.get("branch_names", ())) == BRANCH_NAMES, "wrong cache branch order", errors)
    _require(
        tuple(payload.get("response_branch_names", ())) == RESPONSE_BRANCH_NAMES,
        "cache response branches must be expert/local_delta/gripper_flip",
        errors,
    )
    _require(payload.get("teacher_timestep") == 250, "cache teacher timestep must be 250", errors)
    _require(payload.get("target_space") == "e0_global_projected_standardized", "wrong Stage 1 target space", errors)
    _require(payload.get("normalization_fit_split") == "train", "normalization must be fit on train only", errors)
    _require(payload.get("common_branch_noise") is True, "teacher noise must be common across branches", errors)
    _require(isinstance(payload.get("source_identity"), dict), "cache source_identity is required", errors)
    _require(isinstance(payload.get("teacher_identity"), dict), "cache teacher_identity is required", errors)
    _require(
        isinstance(payload.get("action_normalization_identity"), dict),
        "cache action_normalization_identity is required",
        errors,
    )
    action_shape = _tensor_shape(payload, "actions", errors)
    target_shape = _tensor_shape(payload, "response_targets", errors)
    mean_shape = _tensor_shape(payload, "normalization_mean", errors)
    std_shape = _tensor_shape(payload, "normalization_std", errors)
    records = payload.get("records")
    if not isinstance(records, list):
        errors.append("cache records must be a list")
        records = []
    count = len(records)
    current_shape = _current_hidden_shape(payload, count=count, errors=errors)
    _require(
        current_shape == (count, len(LAYERS), *CURRENT_SHAPE),
        f"current_hidden must be [N,3,392,1536], got {current_shape}",
        errors,
    )
    _require(
        action_shape == (count, len(BRANCH_NAMES), *ACTION_SHAPE),
        f"actions must be [N,4,8,7], got {action_shape}",
        errors,
    )
    _require(
        target_shape == (count, len(LAYERS), len(RESPONSE_BRANCH_NAMES), *TARGET_SHAPE),
        f"response_targets must be [N,3,3,4,256], got {target_shape}",
        errors,
    )
    _require(
        mean_shape == (len(LAYERS), *TARGET_SHAPE),
        f"normalization_mean must be [3,4,256], got {mean_shape}",
        errors,
    )
    _require(
        std_shape == (len(LAYERS), *TARGET_SHAPE),
        f"normalization_std must be [3,4,256], got {std_shape}",
        errors,
    )
    if isinstance(payload.get("normalization_std"), torch.Tensor):
        _require(
            bool((payload["normalization_std"] > 0).all()),
            "normalization_std must be strictly positive",
            errors,
        )
    splits = []
    demo_split: dict[tuple[str, int], str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {index} must be a dict")
            continue
        split = record.get("split")
        if split not in {"train", "validation", "test"}:
            errors.append(f"record {index} has invalid split {split!r}")
            continue
        if not isinstance(record.get("demo_id"), int) or not isinstance(record.get("task"), str):
            errors.append(f"record {index} must carry task and integer demo_id")
            continue
        key = (record["task"], record["demo_id"])
        prior = demo_split.setdefault(key, split)
        if prior != split:
            errors.append(f"demonstration {key} crosses split boundaries")
        splits.append(split)
    if require_trainable_splits_only:
        _require("test" not in splits, "train/validation cache must not include test examples", errors)
        _require(payload.get("test_state_count", 0) == 0, "train cache must report zero test states", errors)
        _require(
            payload.get("heldout_noise_target_count", 0) == 0,
            "train cache must report zero held-out-noise targets",
            errors,
        )
        _require("train" in splits and "validation" in splits, "cache needs train and validation records", errors)
    if errors:
        raise Stage1ContractError("Stage 1 response-cache validation failed:\n- " + "\n- ".join(errors))
    return {
        "passed": True,
        "record_count": count,
        "split_counts": {split: splits.count(split) for split in ("train", "validation", "test")},
        "source_identity": payload["source_identity"],
        "teacher_identity": payload["teacher_identity"],
        "action_normalization_identity": payload["action_normalization_identity"],
    }
