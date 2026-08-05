"""Simulator-side primitives shared by Dynamic Stage-1 preparation stages.

Nothing in this module imports a model.  Source selection and same-state
branch rollout consequently cannot inspect teacher targets or learned outputs.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .contracts import Stage1DataConfig, Stage1ContractError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUNTIME_RESTORE_CONTRACT = "seeded_static_osc_source_sync_v1"
COLLECTION_RENDERER_CONTEXT_CONTRACT = "seeded_hard_reset_expert_consensus_renderer_v1"
STATIC_MODEL_STATE_ARRAYS = (
    "body_pos",
    "body_quat",
    "geom_pos",
    "geom_quat",
    "site_pos",
    "site_quat",
    "cam_pos",
    "cam_quat",
    "geom_rgba",
    "site_rgba",
    "mat_rgba",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def libero_root(value: str | Path | None = None) -> Path:
    root = Path(value or os.environ.get("LIBERO_ROOT", "/data/users/lizhe/LIBERO")).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"LIBERO_ROOT does not exist: {root}. Set LIBERO_ROOT or pass --libero-root."
        )
    return root


def capture_static_model_state(env: Any) -> dict[str, np.ndarray]:
    """Copy model arrays omitted by LIBERO's flattened simulator state."""

    raw = getattr(env, "env", None)
    model = getattr(getattr(raw, "sim", None), "model", None)
    if model is None:
        raise TypeError("LIBERO environment does not expose a MuJoCo model")
    values: dict[str, np.ndarray] = {}
    for name in STATIC_MODEL_STATE_ARRAYS:
        value = getattr(model, name, None)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"MuJoCo model does not expose array {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"MuJoCo model array {name} contains non-finite values")
        values[name] = np.asarray(value).copy()
    return values


def restore_static_model_state(env: Any, values: dict[str, np.ndarray]) -> None:
    """Restore the canonical fixed-fixture model state before a source replay."""

    raw = getattr(env, "env", None)
    sim = getattr(raw, "sim", None)
    model = getattr(sim, "model", None)
    if sim is None or model is None:
        raise TypeError("LIBERO environment does not expose a MuJoCo model")
    if set(values) != set(STATIC_MODEL_STATE_ARRAYS):
        raise Stage1ContractError("static model-state array set changed")
    for name in STATIC_MODEL_STATE_ARRAYS:
        target = getattr(model, name, None)
        value = np.asarray(values[name])
        if not isinstance(target, np.ndarray) or target.shape != value.shape or target.dtype != value.dtype:
            raise Stage1ContractError(f"static model-state shape or dtype changed for {name}")
        target[...] = value
    sim.forward()


def images_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Return the two deployment-aligned, 180-degree-rotated RGB views."""

    values: list[np.ndarray] = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if key not in obs:
            raise KeyError(f"simulator observation is missing {key}")
        image = np.ascontiguousarray(np.asarray(obs[key])[::-1, ::-1])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"unexpected simulator image {key}: {image.shape} {image.dtype}")
        values.append(image)
    return np.stack(values, axis=0)


def _quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Match the LIBERO evaluator's $(x,y,z,w)$ quaternion conversion."""

    quat = np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(1.0 - float(quat[3] * quat[3]), 0.0))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator


def proprio_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Build the exact raw eight-dimensional LIBERO evaluator proprio state."""

    required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(f"cannot build O2 proprio; missing {missing}")
    result = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3),
            _quat_to_axis_angle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(2),
        )
    ).astype(np.float32)
    if result.shape != (8,) or not np.isfinite(result).all():
        raise ValueError(f"invalid O2 proprio state: {result.shape}")
    return result


def split_for_demo(demo_id: int, config: Stage1DataConfig) -> str:
    return config.split_for_demo(int(demo_id))


def sample_id(task_position: int, demo_id: int, source_slot: int) -> int:
    """Unique deterministic state id used by branches and common teacher noise."""

    if task_position < 0 or demo_id < 0 or source_slot < 0:
        raise ValueError("sample-id coordinates must be non-negative")
    return int(task_position) * 500 + int(demo_id) * 10 + int(source_slot)


def branch_actions(
    expert: np.ndarray,
    *,
    task_position: int,
    demo_id: int,
    config: Stage1DataConfig,
) -> dict[str, np.ndarray]:
    """Create the four frozen same-state branches without random perturbations."""

    if tuple(expert.shape) != (config.action_horizon, 7):
        raise ValueError(f"expert action chunk must be [8,7], got {expert.shape}")
    if not np.isfinite(expert).all():
        raise ValueError("expert action chunk contains non-finite values")
    base = np.clip(np.asarray(expert, dtype=np.float32), -1.0, 1.0)
    # The perturbation coordinate is intentionally constant across all ten
    # states in a demonstration, as frozen in the Dynamic Stage-1 contract.
    branch_key = int(task_position) * 50 + int(demo_id)
    dimension = branch_key % 6
    sign = 1.0 if ((branch_key // 6) % 2 == 0) else -1.0
    local_delta = base.copy()
    local_delta[:, dimension] = np.clip(
        local_delta[:, dimension] + sign * config.motion_perturbation,
        -1.0,
        1.0,
    )
    gripper_flip = base.copy()
    gripper_flip[:, -1] = -gripper_flip[:, -1]
    zero = np.zeros_like(base)
    zero[:, -1] = config.zero_gripper_action
    outputs = {
        "expert": base,
        "local_delta": local_delta,
        "gripper_flip": gripper_flip,
        "zero": zero,
    }
    for name, action in outputs.items():
        if tuple(action.shape) != (config.action_horizon, 7) or not np.isfinite(action).all():
            raise ValueError(f"invalid {name} branch action")
    return outputs


def restore_state(env: Any, source_state: np.ndarray, *, tolerance: float) -> dict[str, Any]:
    env.reset()
    return restore_source_runtime(env, source_state, tolerance=tolerance)


def restore_source_runtime(env: Any, source_state: np.ndarray, *, tolerance: float) -> dict[str, Any]:
    """Restore the serialized source plus OSC and force state omitted by HDF5.

    LIBERO's flattened state is only time, qpos, and qvel. Reset-created OSC
    goals and applied forces would otherwise make the first replayed action
    depend on the reset state rather than on the demonstration source state.
    """

    raw = _underlying_env(env)
    sim = raw.sim
    setter = getattr(env, "set_state", None)
    if callable(setter):
        setter(source_state)
    else:
        sim.set_state_from_flattened(source_state)
    for name in ("ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied"):
        value = getattr(sim.data, name, None)
        if isinstance(value, np.ndarray):
            value.fill(0.0)
    sim.forward()
    for robot in getattr(raw, "robots", ()):
        controller = getattr(robot, "controller", None)
        synchronize = getattr(controller, "update_initial_joints", None)
        if not callable(synchronize):
            continue
        joints = np.asarray(getattr(robot, "_joint_positions"), dtype=np.float64).copy()
        synchronize(joints)
        initial_joint = getattr(controller, "initial_joint", None)
        if initial_joint is not None and not np.array_equal(np.asarray(initial_joint), joints):
            raise RuntimeError("OSC controller initial joints do not match restored source state")
    check_success = getattr(raw, "_check_success", None)
    if callable(check_success):
        check_success()
    post_process = getattr(raw, "_post_process", None)
    if callable(post_process):
        post_process()
    raw._update_observables(force=True)
    obs = raw._get_observations()
    restored = np.asarray(env.get_sim_state(), dtype=np.float64)
    expected = np.asarray(source_state, dtype=np.float64)
    if restored.shape != expected.shape:
        raise RuntimeError(f"simulator restore width changed: {restored.shape} vs {expected.shape}")
    error = float(np.max(np.abs(restored - expected)))
    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(
            f"simulator restore exceeded contract: max_abs={error:.3e}, tolerance={tolerance:.3e}"
        )
    return {"obs": obs, "restore_max_abs": error}


def _underlying_env(env: Any) -> Any:
    raw = getattr(env, "env", None)
    if raw is None:
        raise TypeError("Dynamic Stage-1 renderer requires LIBERO ControlEnv.env")
    required = (
        "sim",
        "control_timestep",
        "model_timestep",
        "_pre_action",
        "_post_action",
        "_update_observables",
        "_get_observations",
    )
    missing = [name for name in required if not hasattr(raw, name)]
    if missing:
        raise TypeError(f"LIBERO/robosuite step internals changed; missing {missing}")
    return raw


def _render_substep_indices(substeps: int, render_ticks_per_action: int) -> tuple[int, ...]:
    if substeps < render_ticks_per_action:
        raise RuntimeError(
            f"control period has only {substeps} simulator substeps; cannot record "
            f"{render_ticks_per_action} ordered render ticks"
        )
    # The control period in the pinned robosuite runtime has 25 physics ticks.
    # The four target phases are therefore the nearest fixed substeps to
    # 1/4, 1/2, 3/4 and 1.  This changes no action command or duration.
    indices = tuple(int(round(position * substeps / render_ticks_per_action)) for position in range(1, render_ticks_per_action + 1))
    if len(set(indices)) != render_ticks_per_action or indices[-1] != substeps:
        raise RuntimeError(f"could not build four ordered render substeps from {substeps}: {indices}")
    return indices


def step_with_render_ticks(
    env: Any,
    action: np.ndarray,
    *,
    render_ticks_per_action: int,
) -> tuple[dict[str, Any], float, bool, dict[str, Any], list[dict[str, Any]], tuple[int, ...]]:
    """Replay exactly one normal LIBERO control action while sampling substeps.

    This mirrors robosuite's ``MujocoEnv.step`` loop instead of issuing four
    environment actions.  Hence the control command, control duration, state
    evolution, and final observation remain those of one original action.
    """

    raw = _underlying_env(env)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if getattr(raw, "action_dim", None) == 4 and action.size > 4:
        action = np.concatenate((action[:3], action[-1:]), axis=-1)
    if action.size != int(raw.action_dim):
        raise ValueError(f"environment got action width {action.size}, expected {raw.action_dim}")
    if not np.isfinite(action).all():
        raise ValueError("non-finite simulator action")
    if bool(getattr(raw, "done", False)):
        raise RuntimeError("attempted to step a terminated simulator episode")
    ratio = float(raw.control_timestep) / float(raw.model_timestep)
    substeps = int(ratio)
    if not math.isclose(ratio, float(substeps), rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"non-integral robosuite control/model timestep ratio: {ratio}")
    capture_indices = _render_substep_indices(substeps, render_ticks_per_action)
    captures: list[dict[str, Any]] = []
    raw.timestep += 1
    policy_step = True
    for substep in range(1, substeps + 1):
        raw.sim.forward()
        raw._pre_action(action, policy_step)
        raw.sim.step()
        raw._update_observables()
        if substep in capture_indices:
            captures.append(raw._get_observations())
        policy_step = False
    raw.cur_time += raw.control_timestep
    reward, done, info = raw._post_action(action)
    # LIBERO's BDDL wrapper replaces robosuite's horizon result with task
    # success, exactly as its public ``step`` method does.
    if hasattr(raw, "_check_success"):
        done = raw._check_success()
    if getattr(raw, "viewer", None) is not None and getattr(raw, "renderer", None) != "mujoco":
        raw.viewer.update()
    final_obs = raw.viewer._get_observations() if getattr(raw, "viewer_get_obs", False) else raw._get_observations()
    if len(captures) != render_ticks_per_action:
        raise RuntimeError("render-tick coverage mismatch")
    return final_obs, float(reward), bool(done), dict(info), captures, capture_indices


def object_hints(env: Any) -> tuple[str, ...]:
    value = getattr(env, "obj_of_interest", ())
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Iterable):
        raise ValueError("LIBERO environment exposes an invalid obj_of_interest")
    hints = tuple(str(item) for item in value)
    if not hints:
        raise ValueError("LIBERO environment exposes no obj_of_interest")
    return hints


def motion_target_hints(env: Any) -> tuple[str, ...]:
    """Return BDDL goal actors in stable order, with a legacy-safe fallback.

    ``obj_of_interest`` includes both manipulated objects and fixed containers.
    Stage-1 motion coverage must instead track the first object argument of
    every BDDL goal predicate, so compound tasks correctly consider either
    manipulated object without spuriously treating a container as the target.
    """

    interest = object_hints(env)
    by_lower = {value.lower(): value for value in interest}
    parsed = getattr(_underlying_env(env), "parsed_problem", None)
    goal_state = parsed.get("goal_state", ()) if isinstance(parsed, dict) else ()
    selected: list[str] = []
    for predicate in goal_state:
        if not isinstance(predicate, (list, tuple)) or len(predicate) < 2:
            continue
        candidate = str(predicate[1])
        canonical = by_lower.get(candidate.lower())
        if canonical is not None and canonical not in selected:
            selected.append(canonical)
    return tuple(selected) if selected else interest


def target_poses(
    obs: dict[str, Any],
    hints: Iterable[str],
) -> tuple[tuple[np.ndarray, np.ndarray | None, str], ...]:
    """Read every exact BDDL target-object pose in stable hint order."""

    hint_values = tuple(dict.fromkeys(str(value) for value in hints))
    if not hint_values:
        raise KeyError("cannot identify target poses from an empty hint set")
    observed_by_lower = {str(key).lower(): str(key) for key in obs}
    poses: list[tuple[np.ndarray, np.ndarray | None, str]] = []
    unavailable: list[str] = []
    for hint in hint_values:
        position_key = observed_by_lower.get(f"{hint.lower()}_pos")
        if position_key is None or np.asarray(obs[position_key]).size != 3:
            # Fixtures such as a stove can be a BDDL goal actor without
            # exposing an object pose. They remain valid contact hints but
            # cannot contribute to translation / rotation motion.
            unavailable.append(f"{hint}_pos")
            continue
        prefix = position_key.removesuffix("_pos")
        quaternion_key = observed_by_lower.get(f"{prefix.lower()}_quat")
        quaternion = None
        if quaternion_key is not None and np.asarray(obs[quaternion_key]).size == 4:
            quaternion = np.asarray(obs[quaternion_key], dtype=np.float64).reshape(4).copy()
        poses.append(
            (
                np.asarray(obs[position_key], dtype=np.float64).reshape(3).copy(),
                quaternion,
                position_key,
            )
        )
    if not poses:
        raise KeyError(
            f"cannot identify any exact target pose from hints={hint_values}; unavailable={unavailable}; "
            f"observed={sorted(obs)}"
        )
    return tuple(poses)


def target_pose(obs: dict[str, Any], hints: Iterable[str]) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return the stable primary target pose used only for provenance fields."""

    return target_poses(obs, hints)[0]


def _camera_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    raw = _underlying_env(env)
    model = raw.sim.model
    names = ["robot0_eye_in_hand", "robot0_eye_in_hand_camera", "eye_in_hand"]
    camera_id: int | None = None
    for name in names:
        try:
            camera_id = int(model.camera_name2id(name))
            break
        except Exception:
            continue
    if camera_id is None or camera_id < 0:
        available = []
        for index in range(int(getattr(model, "ncam", 0))):
            try:
                available.append(str(model.camera_id2name(index)))
            except Exception:
                pass
        raise KeyError(f"cannot locate wrist camera; available={available}")
    return (
        np.asarray(raw.sim.data.cam_xpos[camera_id], dtype=np.float64).reshape(3).copy(),
        np.asarray(raw.sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def _contact_signature(env: Any, hints: Iterable[str]) -> frozenset[tuple[str, str]]:
    raw = _underlying_env(env)
    target_hints = tuple(str(value).lower() for value in hints)
    pairs: set[tuple[str, str]] = set()
    for contact in raw.sim.data.contact[: raw.sim.data.ncon]:
        left = raw.sim.model.geom_id2name(contact.geom1)
        right = raw.sim.model.geom_id2name(contact.geom2)
        if left is None or right is None:
            continue
        names = (str(left), str(right))
        if any(hint in name.lower() for hint in target_hints for name in names):
            pairs.add(tuple(sorted(names)))
    return frozenset(pairs)


def motion_snapshot(env: Any, obs: dict[str, Any], hints: Iterable[str]) -> dict[str, Any]:
    targets = target_poses(obs, hints)
    target_position, target_quaternion, target_key = targets[0]
    wrist_position, wrist_rotation = _camera_pose(env)
    return {
        "target_position": target_position,
        "target_quaternion": target_quaternion,
        "target_position_key": target_key,
        "target_poses": targets,
        "eef_position": np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3).copy(),
        "eef_quaternion": np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4).copy(),
        "wrist_position": wrist_position,
        "wrist_rotation": wrist_rotation,
        "target_contact_signature": _contact_signature(env, hints),
    }


def _quat_rotation_delta_degrees(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    a = np.asarray(left, dtype=np.float64).reshape(4)
    b = np.asarray(right, dtype=np.float64).reshape(4)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        raise ValueError("invalid zero-norm quaternion in motion trace")
    cosine = float(np.clip(abs(np.dot(a, b)) / denominator, -1.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def _matrix_rotation_delta_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def motion_labels(trace: list[dict[str, Any]], config: Stage1DataConfig) -> dict[str, bool]:
    if len(trace) < 2:
        raise ValueError("motion trace needs the source state and at least one continuation state")
    initial = trace[0]
    object_motion = False
    robot_motion = False
    camera_motion = False
    contact_interaction = False
    reference_contacts = initial["target_contact_signature"]
    initial_targets = tuple(initial["target_poses"])
    initial_keys = tuple(item[2] for item in initial_targets)
    for current in trace[1:]:
        current_targets = tuple(current["target_poses"])
        current_keys = tuple(item[2] for item in current_targets)
        if current_keys != initial_keys:
            raise RuntimeError(f"target object identity changed within a trace: {initial_keys} vs {current_keys}")
        object_motion |= any(
            float(np.linalg.norm(current_target[0] - initial_target[0])) >= config.object_motion_m
            or _quat_rotation_delta_degrees(current_target[1], initial_target[1]) >= config.rotation_motion_degrees
            for initial_target, current_target in zip(initial_targets, current_targets, strict=True)
        )
        robot_motion |= (
            float(np.linalg.norm(current["eef_position"] - initial["eef_position"])) >= config.robot_motion_m
            or _quat_rotation_delta_degrees(current["eef_quaternion"], initial["eef_quaternion"])
            >= config.rotation_motion_degrees
        )
        camera_motion |= (
            float(np.linalg.norm(current["wrist_position"] - initial["wrist_position"])) >= config.camera_motion_m
            or _matrix_rotation_delta_degrees(current["wrist_rotation"], initial["wrist_rotation"])
            >= config.rotation_motion_degrees
        )
        contact_interaction |= current["target_contact_signature"] != reference_contacts
    static_control = not (object_motion or robot_motion or camera_motion or contact_interaction)
    return {
        "object_motion": bool(object_motion),
        "robot_motion": bool(robot_motion),
        "camera_motion": bool(camera_motion),
        "contact_interaction": bool(contact_interaction),
        "static_control": bool(static_control),
    }


def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def torch_record(value: np.ndarray, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(value))
    return tensor if dtype is None else tensor.to(dtype=dtype)
