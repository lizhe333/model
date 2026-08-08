"""Task-aware deterministic analysis of recorded LIBERO simulator states."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from side_model3_adapter_v2.failure_taxonomy import FAILURE_TAXONOMY, UNDETERMINED_STAGE


STATE_ANALYZER_NAME = "libero_state_rules_v1"


@dataclass(frozen=True)
class RuleThresholds:
    close_action_min: float
    open_action_max: float
    grasp_proximity_m: float
    alignment_attempt_max_distance_m: float
    target_static_m: float
    eef_retreat_m: float
    carried_translation_m: float
    co_motion_relative_m: float
    destination_approach_m: float
    disturbance_m: float
    disturbance_persistence_records: int
    grasp_observation_window_policy_steps: int
    repeated_attempt_count: int
    min_recovery_queries: int
    mechanism_joint_motion: float


class TraceEvidenceError(ValueError):
    """Raised when a trace lacks evidence needed for a deterministic rule."""


ANALYSIS_COLUMNS = (
    "checkpoint_step",
    "task",
    "trial",
    "task_description",
    "success",
    "primary_failure",
    "secondary_failure",
    "outcome_awareness_failure",
    "recovery_failure",
    "furthest_stage",
    "confidence",
    "needs_manual_review",
    "rule_labels",
    "rule_events",
    "short_evidence",
    "state_path",
    "analyzer",
)

SUMMARY_COLUMNS = (
    "checkpoint_step",
    "task",
    "failure_type",
    "episode_count",
    "analyzed_failures_in_group",
    "proportion",
    "mean_confidence",
    "outcome_awareness_count",
    "recovery_count",
)

RULE_SUMMARY_COLUMNS = (
    "checkpoint_step",
    "task",
    "rule_label",
    "episode_count",
    "analyzed_failures_in_group",
    "proportion",
)


def _predicate_key(predicate: Any, arguments: Sequence[Any]) -> tuple[str, ...]:
    return (str(predicate).lower(), *(str(value) for value in arguments))


def _float_vector(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _position(record: dict[str, Any], name: str) -> tuple[float, ...] | None:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return None
    entry = objects.get(name)
    if not isinstance(entry, dict):
        return None
    return _float_vector(entry.get("position"))


def _eef_position(record: dict[str, Any]) -> tuple[float, ...] | None:
    robot = record.get("robot")
    if not isinstance(robot, dict):
        return None
    return _float_vector(robot.get("eef_position"))


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise TraceEvidenceError("incompatible vector dimensions in state trace")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _difference(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    if len(left) != len(right):
        raise TraceEvidenceError("incompatible vector dimensions in state trace")
    return tuple(a - b for a, b in zip(left, right))


def _action_value(record: dict[str, Any]) -> float | None:
    action = _float_vector(record.get("action"))
    if not action:
        return None
    return action[-1]


def _is_close(record: dict[str, Any], thresholds: RuleThresholds) -> bool:
    value = _action_value(record)
    return value is not None and value >= thresholds.close_action_min


def _is_open(record: dict[str, Any], thresholds: RuleThresholds) -> bool:
    value = _action_value(record)
    return value is not None and value <= thresholds.open_action_max


def _policy_indices(records: Sequence[dict[str, Any]]) -> list[int]:
    return [index for index, record in enumerate(records) if record.get("phase") == "policy"]


def _query_id(record: dict[str, Any], fallback: int) -> int:
    value = record.get("policy_query_index")
    if isinstance(value, int):
        return value
    return fallback


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"state trace must be a JSON object: {path}")
    return value


def load_state_task_rules(path: Path) -> dict[int, dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("schema_version") != 2 or payload.get("suite") != "libero_10":
        raise ValueError("state task rules must be schema version 2 for libero_10")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("state task rules must contain a tasks list")

    rules: dict[int, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("state task entries must be mappings")
        task_id = int(task["task_id"])
        if task_id in rules:
            raise ValueError(f"duplicate task rule for task {task_id}")
        goals = task.get("goal_predicates")
        requirements = task.get("stage_requirements")
        if not isinstance(goals, list) or not isinstance(requirements, list):
            raise ValueError(f"task {task_id} is missing machine-readable stage rules")
        goal_ids = {str(goal["id"]) for goal in goals}
        stage_ids = {str(stage["id"]) for stage in task.get("stage_states", [])}
        required_stage_ids = {str(stage["id"]) for stage in requirements}
        if stage_ids != required_stage_ids:
            raise ValueError(f"task {task_id} stage requirements do not match stage states")
        for stage in requirements:
            mentioned = set(stage.get("required_true", [])) | set(stage.get("required_false", []))
            if not mentioned.issubset(goal_ids):
                raise ValueError(f"task {task_id} stage references an unknown goal predicate")
        for prerequisite in task.get("physical_prerequisites", []):
            required_goal = prerequisite.get("required_goal")
            dependent_goal = prerequisite.get("dependent_goal")
            if required_goal not in goal_ids or dependent_goal not in goal_ids:
                raise ValueError(f"task {task_id} physical prerequisite references an unknown goal")
        rules[task_id] = task
    if set(rules) != set(range(10)):
        raise ValueError("state task rules must cover LIBERO Long task ids 0 through 9")
    return rules


def load_rule_thresholds(path: Path) -> RuleThresholds:
    payload = _load_json(path)
    if payload.get("schema_version") != 1 or payload.get("suite") != "libero_10":
        raise ValueError("state rule thresholds must be schema version 1 for libero_10")
    fields = tuple(RuleThresholds.__dataclass_fields__)
    try:
        values = {field: payload[field] for field in fields}
    except KeyError as exc:
        raise ValueError(f"state rule thresholds missing {exc.args[0]!r}") from exc
    try:
        thresholds = RuleThresholds(
            **{
                field: int(values[field])
                if field
                in {
                    "disturbance_persistence_records",
                    "grasp_observation_window_policy_steps",
                    "repeated_attempt_count",
                    "min_recovery_queries",
                }
                else float(values[field])
                for field in fields
            }
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("state rule thresholds must be numeric") from exc
    if (
        thresholds.grasp_proximity_m <= 0.0
        or thresholds.alignment_attempt_max_distance_m < thresholds.grasp_proximity_m
        or thresholds.repeated_attempt_count < 2
        or thresholds.disturbance_persistence_records < 1
        or thresholds.grasp_observation_window_policy_steps < 1
        or thresholds.min_recovery_queries < 1
    ):
        raise ValueError("state rule thresholds contain an invalid range")
    return thresholds


def load_state_trace(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != 1 or payload.get("task_suite") != "libero_10":
        raise ValueError(f"unsupported simulator-state trace: {path}")
    if not isinstance(payload.get("records"), list) or not payload["records"]:
        raise ValueError(f"state trace has no records: {path}")
    return payload


def scan_state_traces(state_dir: Path) -> list[Path]:
    if not state_dir.is_dir():
        raise ValueError(f"state directory does not exist: {state_dir}")
    paths = sorted(state_dir.glob("task_*/episode_*.json"))
    if not paths:
        raise ValueError(f"no state trace files found under: {state_dir}")
    return paths


def _goal_truths(
    record: dict[str, Any],
    task_rule: dict[str, Any],
) -> dict[str, bool]:
    expected = {
        _predicate_key(goal["predicate"], goal["args"]): str(goal["id"])
        for goal in task_rule["goal_predicates"]
    }
    observed: dict[str, bool] = {}
    predicates = record.get("task_predicates")
    if not isinstance(predicates, list):
        raise TraceEvidenceError("record is missing task predicates")
    for predicate in predicates:
        if not isinstance(predicate, dict):
            continue
        key = _predicate_key(predicate.get("predicate"), predicate.get("arguments", []))
        goal_id = expected.get(key)
        if goal_id is not None:
            observed[goal_id] = bool(predicate.get("satisfied"))
    missing = set(expected.values()).difference(observed)
    if missing:
        raise TraceEvidenceError(f"record is missing registered goal predicates: {sorted(missing)}")
    return observed


def _stage_for_truths(task_rule: dict[str, Any], truths: dict[str, bool]) -> dict[str, Any] | None:
    matches = []
    for stage in task_rule["stage_requirements"]:
        true_goals = stage.get("required_true", [])
        false_goals = stage.get("required_false", [])
        if all(truths[goal] for goal in true_goals) and all(
            not truths[goal] for goal in false_goals
        ):
            matches.append(stage)
    return matches[0] if len(matches) == 1 else None


def _stage_timeline(
    records: Sequence[dict[str, Any]], task_rule: dict[str, Any]
) -> tuple[dict[int, dict[str, bool]], str]:
    truth_timeline: dict[int, dict[str, bool]] = {}
    best_stage: dict[str, Any] | None = None
    best_index = -1
    for index, record in enumerate(records):
        truths = _goal_truths(record, task_rule)
        truth_timeline[index] = truths
        stage = _stage_for_truths(task_rule, truths)
        if stage is None:
            continue
        rank = int(stage.get("progress_rank", 0))
        if best_stage is None or (rank, index) > (
            int(best_stage.get("progress_rank", 0)),
            best_index,
        ):
            best_stage = stage
            best_index = index
    return truth_timeline, (
        UNDETERMINED_STAGE if best_stage is None else str(best_stage["id"])
    )


def _spatial_goals(task_rule: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        goal
        for goal in task_rule["goal_predicates"]
        if str(goal["predicate"]).lower() in {"in", "on"}
    ]


def _mechanism_goals(task_rule: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        goal
        for goal in task_rule["goal_predicates"]
        if str(goal["predicate"]).lower() not in {"in", "on"}
    ]


def _policy_baseline_index(policy_indices: Sequence[int]) -> int:
    if not policy_indices:
        raise TraceEvidenceError("trace contains no policy records")
    return max(0, policy_indices[0] - 1)


def _transport_event(
    records: Sequence[dict[str, Any]],
    policy_indices: Sequence[int],
    *,
    baseline_index: int,
    target: str,
    thresholds: RuleThresholds,
) -> dict[str, Any] | None:
    initial_object = _position(records[baseline_index], target)
    initial_eef = _eef_position(records[baseline_index])
    if initial_object is None or initial_eef is None:
        raise TraceEvidenceError(f"missing policy-start pose for target {target}")
    initial_relative = _difference(initial_eef, initial_object)
    for record_index in policy_indices:
        object_position = _position(records[record_index], target)
        eef_position = _eef_position(records[record_index])
        if object_position is None or eef_position is None:
            continue
        object_motion = _distance(object_position, initial_object)
        eef_motion = _distance(eef_position, initial_eef)
        relative_error = _distance(
            _difference(eef_position, object_position), initial_relative
        )
        if (
            object_motion >= thresholds.carried_translation_m
            and eef_motion >= thresholds.carried_translation_m
            and relative_error <= thresholds.co_motion_relative_m
        ):
            return {
                "record_index": record_index,
                "target": target,
                "object_motion_m": round(object_motion, 6),
                "eef_motion_m": round(eef_motion, 6),
                "relative_error_m": round(relative_error, 6),
            }
    return None


def _collect_grasp_attempts(
    records: Sequence[dict[str, Any]],
    policy_indices: Sequence[int],
    truths_by_record: dict[int, dict[str, bool]],
    spatial_goals: Sequence[dict[str, Any]],
    thresholds: RuleThresholds,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    previous_close = False
    for policy_offset, record_index in enumerate(policy_indices):
        record = records[record_index]
        close = _is_close(record, thresholds)
        if close and not previous_close:
            eef_position = _eef_position(record)
            candidates = []
            if eef_position is not None:
                for goal in spatial_goals:
                    if truths_by_record[record_index][str(goal["id"])]:
                        continue
                    target = str(goal["args"][0])
                    target_position = _position(record, target)
                    if target_position is None:
                        continue
                    distance = _distance(eef_position, target_position)
                    if distance <= thresholds.alignment_attempt_max_distance_m:
                        candidates.append((distance, goal, target_position))
            if candidates:
                distance, goal, target_position = min(candidates, key=lambda item: item[0])
                max_object_motion = 0.0
                max_eef_motion = 0.0
                carried = False
                start_eef = eef_position
                lookahead = policy_indices[
                    policy_offset + 1 : policy_offset + 1 + thresholds.grasp_observation_window_policy_steps
                ]
                initial_relative = _difference(start_eef, target_position)
                for later_index in lookahead:
                    later_object = _position(records[later_index], str(goal["args"][0]))
                    later_eef = _eef_position(records[later_index])
                    if later_object is None or later_eef is None:
                        continue
                    object_motion = _distance(later_object, target_position)
                    eef_motion = _distance(later_eef, start_eef)
                    max_object_motion = max(max_object_motion, object_motion)
                    max_eef_motion = max(max_eef_motion, eef_motion)
                    relative_error = _distance(
                        _difference(later_eef, later_object), initial_relative
                    )
                    if (
                        object_motion >= thresholds.carried_translation_m
                        and eef_motion >= thresholds.carried_translation_m
                        and relative_error <= thresholds.co_motion_relative_m
                    ):
                        carried = True
                near_target = distance <= thresholds.grasp_proximity_m
                failed = (
                    near_target
                    and max_object_motion <= thresholds.target_static_m
                    and max_eef_motion >= thresholds.eef_retreat_m
                )
                attempts.append(
                    {
                        "record_index": record_index,
                        "query_index": _query_id(record, policy_offset + 1),
                        "goal_id": str(goal["id"]),
                        "target": str(goal["args"][0]),
                        "destination": str(goal["args"][1]),
                        "target_distance_m": round(distance, 6),
                        "near_target": near_target,
                        "object_motion_m": round(max_object_motion, 6),
                        "eef_motion_m": round(max_eef_motion, 6),
                        "carried": carried,
                        "failed": failed,
                    }
                )
        previous_close = close
    return attempts


def _target_departed_after(
    records: Sequence[dict[str, Any]],
    policy_indices: Sequence[int],
    *,
    target: str,
    record_index: int,
    thresholds: RuleThresholds,
) -> bool:
    try:
        start_offset = list(policy_indices).index(record_index)
    except ValueError:
        return False
    for later_index in policy_indices[start_offset + 1 :]:
        eef_position = _eef_position(records[later_index])
        target_position = _position(records[later_index], target)
        if eef_position is None or target_position is None:
            continue
        if _is_open(records[later_index], thresholds) or _distance(
            eef_position, target_position
        ) >= thresholds.grasp_proximity_m + thresholds.eef_retreat_m:
            return True
    return False


def _placement_events(
    records: Sequence[dict[str, Any]],
    policy_indices: Sequence[int],
    truths_by_record: dict[int, dict[str, bool]],
    spatial_goals: Sequence[dict[str, Any]],
    transports: dict[str, dict[str, Any] | None],
    task_rule: dict[str, Any],
    thresholds: RuleThresholds,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    final_truths = truths_by_record[len(records) - 1]
    for goal in spatial_goals:
        goal_id = str(goal["id"])
        target = str(goal["args"][0])
        ever_satisfied = any(truths[goal_id] for truths in truths_by_record.values())
        transport = transports[goal_id]
        if final_truths[goal_id] or (not ever_satisfied and transport is None):
            continue
        if transport is not None and not ever_satisfied and not _target_departed_after(
            records,
            policy_indices,
            target=target,
            record_index=int(transport["record_index"]),
            thresholds=thresholds,
        ):
            continue
        events.append(
            {
                "label": "placement_failure",
                "record_index": len(records) - 1,
                "goal_id": goal_id,
                "target": target,
                "reason": "goal_predicate_false_after_transport",
            }
        )

    for prerequisite in task_rule.get("physical_prerequisites", []):
        required_goal = str(prerequisite["required_goal"])
        dependent_goal = str(prerequisite["dependent_goal"])
        for record_index, truths in truths_by_record.items():
            if truths[dependent_goal] and not truths[required_goal]:
                events.append(
                    {
                        "label": "placement_failure",
                        "record_index": record_index,
                        "goal_id": required_goal,
                        "dependent_goal": dependent_goal,
                        "reason": "physical_prerequisite_violated",
                    }
                )
                break
    return events


def _mechanism_joint_moved(
    records: Sequence[dict[str, Any]],
    policy_indices: Sequence[int],
    *,
    baseline_index: int,
    mechanism_name: str,
    thresholds: RuleThresholds,
) -> bool:
    baseline_objects = records[baseline_index].get("objects", {})
    if not isinstance(baseline_objects, dict):
        return False
    mechanism = baseline_objects.get(mechanism_name)
    if not isinstance(mechanism, dict):
        return False
    joint_names = mechanism.get("joint_names")
    baseline_joints = records[baseline_index].get("joints")
    if not isinstance(joint_names, list) or not isinstance(baseline_joints, dict):
        return False
    for joint_name in joint_names:
        baseline = _float_vector(baseline_joints.get(joint_name))
        if baseline is None:
            continue
        for record_index in policy_indices:
            joints = records[record_index].get("joints")
            if not isinstance(joints, dict):
                continue
            current = _float_vector(joints.get(joint_name))
            if current is not None and _distance(current, baseline) >= thresholds.mechanism_joint_motion:
                return True
    return False


def _mechanism_events(
    records: Sequence[dict[str, Any]],
    policy_indices: Sequence[int],
    truths_by_record: dict[int, dict[str, bool]],
    spatial_goals: Sequence[dict[str, Any]],
    mechanism_goals: Sequence[dict[str, Any]],
    *,
    baseline_index: int,
    thresholds: RuleThresholds,
) -> list[dict[str, Any]]:
    final_truths = truths_by_record[len(records) - 1]
    spatial_ready = all(
        any(truths[str(goal["id"])] for truths in truths_by_record.values())
        for goal in spatial_goals
    )
    events: list[dict[str, Any]] = []
    for goal in mechanism_goals:
        goal_id = str(goal["id"])
        mechanism_name = str(goal["args"][0])
        if final_truths[goal_id] or not spatial_ready:
            continue
        if _mechanism_joint_moved(
            records,
            policy_indices,
            baseline_index=baseline_index,
            mechanism_name=mechanism_name,
            thresholds=thresholds,
        ):
            events.append(
                {
                    "label": "mechanism_interaction_failure",
                    "record_index": len(records) - 1,
                    "goal_id": goal_id,
                    "mechanism": mechanism_name,
                }
            )
    return events


def _initial_invariant_events(
    records: Sequence[dict[str, Any]],
    truths_by_record: dict[int, dict[str, bool]],
    task_rule: dict[str, Any],
) -> list[dict[str, Any]]:
    goal_by_predicate = {
        _predicate_key(goal["predicate"], goal["args"]): str(goal["id"])
        for goal in task_rule["goal_predicates"]
    }
    events: list[dict[str, Any]] = []
    for predicate in task_rule.get("initial_task_predicates", []):
        if not isinstance(predicate, dict):
            continue
        goal_id = goal_by_predicate.get(
            _predicate_key(predicate.get("predicate"), predicate.get("args", []))
        )
        if goal_id is None or not truths_by_record[0][goal_id]:
            continue
        for record_index, truths in truths_by_record.items():
            if not truths[goal_id]:
                events.append(
                    {
                        "label": "mechanism_interaction_failure",
                        "record_index": record_index,
                        "goal_id": goal_id,
                        "reason": "required_initial_predicate_lost",
                    }
                )
                break
    return events


def _disturbance_events(
    records: Sequence[dict[str, Any]],
    policy_indices: Sequence[int],
    task_rule: dict[str, Any],
    *,
    baseline_index: int,
    thresholds: RuleThresholds,
) -> list[dict[str, Any]]:
    baseline_objects = records[baseline_index].get("objects")
    if not isinstance(baseline_objects, dict):
        return []

    events: list[dict[str, Any]] = []
    for name in sorted(str(value) for value in task_rule.get("distractors", [])):
        entry = baseline_objects.get(name)
        if not isinstance(entry, dict) or entry.get("kind") != "movable_object":
            continue
        baseline_position = _float_vector(entry.get("position"))
        if baseline_position is None:
            continue
        consecutive = 0
        max_motion = 0.0
        for record_index in policy_indices:
            position = _position(records[record_index], str(name))
            if position is None:
                consecutive = 0
                continue
            motion = _distance(position, baseline_position)
            max_motion = max(max_motion, motion)
            if motion >= thresholds.disturbance_m:
                consecutive += 1
                if consecutive == thresholds.disturbance_persistence_records:
                    events.append(
                        {
                            "label": "environment_disturbance",
                            "record_index": record_index,
                            "object": str(name),
                            "motion_m": round(max_motion, 6),
                        }
                    )
                    break
            else:
                consecutive = 0
    return events


def _outcome_awareness_events(
    records: Sequence[dict[str, Any]],
    policy_indices: Sequence[int],
    truths_by_record: dict[int, dict[str, bool]],
    spatial_goals: Sequence[dict[str, Any]],
    attempts: Sequence[dict[str, Any]],
    transports: dict[str, dict[str, Any] | None],
    prerequisite_violations: Sequence[dict[str, Any]],
    thresholds: RuleThresholds,
) -> list[dict[str, Any]]:
    events = [
        {
            "label": "outcome_awareness_failure",
            "record_index": event["record_index"],
            "goal_id": event["goal_id"],
            "dependent_goal": event["dependent_goal"],
            "reason": "downstream_mechanism_completed_before_placement",
        }
        for event in prerequisite_violations
    ]
    for attempt in attempts:
        if not attempt["failed"]:
            continue
        goal_id = str(attempt["goal_id"])
        target = str(attempt["target"])
        destination = str(attempt["destination"])
        failure_index = int(attempt["record_index"])
        failure_query = int(attempt["query_index"])
        if transports.get(goal_id) is not None:
            continue
        later_indices = [
            index
            for offset, index in enumerate(policy_indices)
            if index > failure_index
            and _query_id(records[index], offset + 1) > failure_query
        ]
        other_targets_carried = any(
            transport is not None and int(transport["record_index"]) > failure_index
            for other_goal_id, transport in transports.items()
            if other_goal_id != goal_id
        )
        if other_targets_carried:
            continue
        target_at_failure = _position(records[failure_index], target)
        if target_at_failure is None:
            continue
        for record_index in later_indices:
            if truths_by_record[record_index][goal_id]:
                break
            eef_position = _eef_position(records[record_index])
            destination_position = _position(records[record_index], destination)
            target_position = _position(records[record_index], target)
            if eef_position is None or destination_position is None or target_position is None:
                continue
            if (
                _distance(eef_position, destination_position)
                <= thresholds.destination_approach_m
                and _distance(target_position, target_at_failure) <= thresholds.target_static_m
            ):
                events.append(
                    {
                        "label": "outcome_awareness_failure",
                        "record_index": record_index,
                        "goal_id": goal_id,
                        "target": target,
                        "reason": "empty_gripper_approached_destination_after_failed_grasp",
                    }
                )
                break
    return events


def _recovery_event(
    records: Sequence[dict[str, Any]],
    policy_indices: Sequence[int],
    truths_by_record: dict[int, dict[str, bool]],
    attempts: Sequence[dict[str, Any]],
    alignment_events: Sequence[dict[str, Any]],
    disturbance_events: Sequence[dict[str, Any]],
    thresholds: RuleThresholds,
) -> dict[str, Any] | None:
    if not alignment_events or not disturbance_events:
        return None
    alignment_targets = {str(event["target"]) for event in alignment_events}
    alignment_attempts = [
        attempt for attempt in attempts if str(attempt["target"]) in alignment_targets
    ]
    candidates = [
        int(attempt["record_index"]) for attempt in alignment_attempts if attempt["failed"]
    ]
    if not candidates:
        return None
    failure_index = min(candidates)
    failure_offset = next(
        offset for offset, index in enumerate(policy_indices) if index >= failure_index
    )
    failure_query = _query_id(records[policy_indices[failure_offset]], failure_offset + 1)
    later_indices = [
        index
        for offset, index in enumerate(policy_indices)
        if index > failure_index and _query_id(records[index], offset + 1) > failure_query
    ]
    later_queries = {
        _query_id(records[index], offset + 1)
        for offset, index in enumerate(policy_indices)
        if index in later_indices
    }
    if len(later_queries) < thresholds.min_recovery_queries:
        return None
    alignment_goal_ids = {str(attempt["goal_id"]) for attempt in alignment_attempts}
    recovered = any(
        any(truths[goal_id] for goal_id in alignment_goal_ids)
        for index, truths in truths_by_record.items()
        if index > failure_index
    )
    if recovered:
        return None
    return {
        "label": "recovery_failure",
        "record_index": later_indices[-1],
        "reason": "repeated_failed_grasp_after_environment_disturbance",
    }


def _event_summary(events: Sequence[dict[str, Any]]) -> str:
    snippets = []
    for event in events[:3]:
        label = event["label"]
        record_index = event.get("record_index")
        target = event.get("target") or event.get("object") or event.get("goal_id")
        suffix = "" if target is None else f" target={target}"
        snippets.append(f"{label} record={record_index}{suffix}")
    return "; ".join(snippets)


def _manual_analysis(
    payload: dict[str, Any],
    *,
    state_path: Path,
    checkpoint_step: int,
    task_rule: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "checkpoint_step": checkpoint_step,
        "task": int(payload["task_id"]),
        "trial": int(payload["trial_id"]),
        "task_description": str(task_rule["language"]),
        "success": bool(payload.get("success")),
        "primary_failure": "其他/无法判断",
        "secondary_failure": None,
        "outcome_awareness_failure": False,
        "recovery_failure": False,
        "furthest_stage": UNDETERMINED_STAGE,
        "confidence": 0.0,
        "needs_manual_review": True,
        "rule_labels": [],
        "rule_events": [],
        "short_evidence": f"insufficient_state_evidence: {reason}",
        "state_path": str(state_path.resolve()),
        "analyzer": STATE_ANALYZER_NAME,
    }


def analyze_state_trace(
    payload: dict[str, Any],
    *,
    state_path: Path,
    checkpoint_step: int,
    task_rules: dict[int, dict[str, Any]],
    thresholds: RuleThresholds,
) -> dict[str, Any]:
    task_id = int(payload["task_id"])
    if task_id not in task_rules:
        raise ValueError(f"no state task rule registered for task {task_id}")
    task_rule = task_rules[task_id]
    try:
        records = payload["records"]
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise TraceEvidenceError("trace records are malformed")
        policy_indices = _policy_indices(records)
        baseline_index = _policy_baseline_index(policy_indices)
        truths_by_record, furthest_stage = _stage_timeline(records, task_rule)
        spatial_goals = _spatial_goals(task_rule)
        mechanism_goals = _mechanism_goals(task_rule)
        transports = {
            str(goal["id"]): _transport_event(
                records,
                policy_indices,
                baseline_index=baseline_index,
                target=str(goal["args"][0]),
                thresholds=thresholds,
            )
            for goal in spatial_goals
        }
        attempts = _collect_grasp_attempts(
            records,
            policy_indices,
            truths_by_record,
            spatial_goals,
            thresholds,
        )
        attempts_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for attempt in attempts:
            attempts_by_target[str(attempt["target"])].append(attempt)
        alignment_events = []
        for target, target_attempts in attempts_by_target.items():
            if (
                len(target_attempts) >= thresholds.repeated_attempt_count
                and not any(bool(attempt["carried"]) for attempt in target_attempts)
                and all(
                    bool(attempt["failed"]) or not bool(attempt["near_target"])
                    for attempt in target_attempts
                )
            ):
                alignment_events.append(
                    {
                        "label": "grasp_alignment_failure",
                        "record_index": target_attempts[-1]["record_index"],
                        "target": target,
                        "attempt_count": len(target_attempts),
                        "evidence_basis": "kinematic_trajectory_proxy",
                    }
                )
        alignment_targets = {str(event["target"]) for event in alignment_events}
        grasp_events = [
            {
                "label": "grasp_failed",
                "record_index": attempt["record_index"],
                "goal_id": attempt["goal_id"],
                "target": attempt["target"],
                "target_distance_m": attempt["target_distance_m"],
                "object_motion_m": attempt["object_motion_m"],
                "eef_motion_m": attempt["eef_motion_m"],
                "evidence_basis": "kinematic_trajectory_proxy",
            }
            for attempt in attempts
            if attempt["failed"]
            and str(attempt["target"]) not in alignment_targets
            and transports.get(str(attempt["goal_id"])) is None
        ]
        placement_events = _placement_events(
            records,
            policy_indices,
            truths_by_record,
            spatial_goals,
            transports,
            task_rule,
            thresholds,
        )
        prerequisite_violations = [
            event
            for event in placement_events
            if event.get("reason") == "physical_prerequisite_violated"
        ]
        mechanism_events = [
            *_initial_invariant_events(records, truths_by_record, task_rule),
            *_mechanism_events(
                records,
                policy_indices,
                truths_by_record,
                spatial_goals,
                mechanism_goals,
                baseline_index=baseline_index,
                thresholds=thresholds,
            ),
        ]
        disturbance_events = _disturbance_events(
            records,
            policy_indices,
            task_rule,
            baseline_index=baseline_index,
            thresholds=thresholds,
        )
        outcome_events = _outcome_awareness_events(
            records,
            policy_indices,
            truths_by_record,
            spatial_goals,
            attempts,
            transports,
            prerequisite_violations,
            thresholds,
        )
        recovery_event = _recovery_event(
            records,
            policy_indices,
            truths_by_record,
            attempts,
            alignment_events,
            disturbance_events,
            thresholds,
        )

        events = [
            *placement_events,
            *mechanism_events,
            *alignment_events,
            *grasp_events,
            *disturbance_events,
            *outcome_events,
        ]
        if recovery_event is not None:
            events.append(recovery_event)
        events.sort(key=lambda event: (int(event["record_index"]), str(event["label"])))
        labels = list(dict.fromkeys(str(event["label"]) for event in events))

        if placement_events:
            primary_failure = "搬运/放置失败"
            confidence = 0.98 if prerequisite_violations else 0.9
        elif mechanism_events:
            primary_failure = "机构交互失败"
            confidence = 0.92
        elif alignment_events:
            primary_failure = "接近/位姿对齐失败"
            confidence = 0.85
        elif grasp_events:
            primary_failure = "抓取/接触失败"
            confidence = 0.8
        elif outcome_events:
            primary_failure = "动作结果判断错误"
            confidence = 0.85
        elif recovery_event is not None:
            primary_failure = "偏差累积/恢复失败"
            confidence = 0.8
        else:
            primary_failure = "其他/无法判断"
            confidence = 0.0

        if outcome_events and primary_failure != "动作结果判断错误":
            secondary_failure = "动作结果判断错误"
        elif recovery_event is not None and primary_failure != "偏差累积/恢复失败":
            secondary_failure = "偏差累积/恢复失败"
        else:
            secondary_failure = None
        needs_manual_review = primary_failure == "其他/无法判断" or furthest_stage == UNDETERMINED_STAGE
        return {
            "checkpoint_step": checkpoint_step,
            "task": task_id,
            "trial": int(payload["trial_id"]),
            "task_description": str(task_rule["language"]),
            "success": bool(payload.get("success")),
            "primary_failure": primary_failure,
            "secondary_failure": secondary_failure,
            "outcome_awareness_failure": bool(outcome_events),
            "recovery_failure": recovery_event is not None,
            "furthest_stage": furthest_stage,
            "confidence": confidence,
            "needs_manual_review": needs_manual_review,
            "rule_labels": labels,
            "rule_events": events,
            "short_evidence": _event_summary(events)
            if events
            else "insufficient_state_evidence: no deterministic rule matched",
            "state_path": str(state_path.resolve()),
            "analyzer": STATE_ANALYZER_NAME,
        }
    except TraceEvidenceError as exc:
        return _manual_analysis(
            payload,
            state_path=state_path,
            checkpoint_step=checkpoint_step,
            task_rule=task_rule,
            reason=str(exc),
        )


def build_state_manifest(
    payload: dict[str, Any], *, state_path: Path, checkpoint_step: int
) -> dict[str, Any]:
    records = payload.get("records", [])
    return {
        "checkpoint_step": checkpoint_step,
        "task": int(payload["task_id"]),
        "trial": int(payload["trial_id"]),
        "success": bool(payload.get("success")),
        "task_description": str(payload.get("task_description", "")),
        "state_path": str(state_path.resolve()),
        "record_count": len(records) if isinstance(records, list) else 0,
        "environment_steps": int(payload.get("environment_steps", 0)),
    }


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key in ("rule_labels", "rule_events"):
                if key in csv_row:
                    csv_row[key] = json.dumps(csv_row[key], ensure_ascii=False)
            writer.writerow(csv_row)


def build_failure_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    totals: dict[tuple[int, str], int] = defaultdict(int)
    for row in rows:
        checkpoint_step = int(row["checkpoint_step"])
        task = str(row["task"])
        failure_type = str(row["primary_failure"])
        grouped[(checkpoint_step, task, failure_type)].append(row)
        grouped[(checkpoint_step, "ALL", failure_type)].append(row)
        totals[(checkpoint_step, task)] += 1
        totals[(checkpoint_step, "ALL")] += 1
    output = []
    for (checkpoint_step, task, failure_type), items in sorted(grouped.items()):
        total = totals[(checkpoint_step, task)]
        output.append(
            {
                "checkpoint_step": checkpoint_step,
                "task": task,
                "failure_type": failure_type,
                "episode_count": len(items),
                "analyzed_failures_in_group": total,
                "proportion": len(items) / total,
                "mean_confidence": sum(float(item["confidence"]) for item in items)
                / len(items),
                "outcome_awareness_count": sum(
                    bool(item["outcome_awareness_failure"]) for item in items
                ),
                "recovery_count": sum(bool(item["recovery_failure"]) for item in items),
            }
        )
    return output


def build_rule_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], int] = defaultdict(int)
    totals: dict[tuple[int, str], int] = defaultdict(int)
    for row in rows:
        checkpoint_step = int(row["checkpoint_step"])
        task = str(row["task"])
        for scope in (task, "ALL"):
            totals[(checkpoint_step, scope)] += 1
            for label in row["rule_labels"]:
                grouped[(checkpoint_step, scope, str(label))] += 1
    return [
        {
            "checkpoint_step": checkpoint_step,
            "task": task,
            "rule_label": label,
            "episode_count": count,
            "analyzed_failures_in_group": totals[(checkpoint_step, task)],
            "proportion": count / totals[(checkpoint_step, task)],
        }
        for (checkpoint_step, task, label), count in sorted(grouped.items())
    ]


def write_state_analysis_outputs(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    thresholds: RuleThresholds,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, output_dir / "state_failure_analysis.jsonl")
    _write_csv(output_dir / "state_failure_analysis.csv", rows, ANALYSIS_COLUMNS)
    _write_csv(
        output_dir / "state_failure_summary.csv",
        build_failure_summary(rows),
        SUMMARY_COLUMNS,
    )
    _write_csv(
        output_dir / "state_rule_summary.csv",
        build_rule_summary(rows),
        RULE_SUMMARY_COLUMNS,
    )
    _write_csv(
        output_dir / "state_needs_manual_review.csv",
        [row for row in rows if row["needs_manual_review"]],
        ANALYSIS_COLUMNS,
    )
    (output_dir / "state_rule_thresholds.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "analyzer": STATE_ANALYZER_NAME,
                "thresholds": asdict(thresholds),
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
