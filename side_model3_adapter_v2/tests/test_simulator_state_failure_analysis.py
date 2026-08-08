from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from side_model3_adapter_v2.state_failure_analysis import (
    analyze_state_trace,
    build_state_manifest,
    load_rule_thresholds,
    load_state_task_rules,
    write_state_analysis_outputs,
)
from side_model3_adapter_v2.scripts.analyze_libero_state_failures import main as state_cli_main


STAGE_RULES = (
    Path(__file__).resolve().parents[1] / "configs" / "libero_long_stage_rules.json"
)
THRESHOLDS = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "libero_long_state_rule_thresholds.json"
)
RULES = load_state_task_rules(STAGE_RULES)
RULE_THRESHOLDS = load_rule_thresholds(THRESHOLDS)


def _base_positions(task_rule: dict[str, Any]) -> dict[str, list[float]]:
    positions: dict[str, list[float]] = {}
    for index, goal in enumerate(task_rule["goal_predicates"]):
        if str(goal["predicate"]).lower() not in {"in", "on"}:
            continue
        positions[str(goal["args"][0])] = [0.4 * index, 0.0, 0.0]
        positions.setdefault(str(goal["args"][1]), [1.0 + 0.4 * index, 0.0, 0.0])
    for index, goal in enumerate(task_rule["goal_predicates"]):
        if str(goal["predicate"]).lower() in {"in", "on"}:
            continue
        positions.setdefault(str(goal["args"][0]), [1.5 + 0.2 * index, 0.0, 0.0])
    for index, name in enumerate(task_rule.get("distractors", [])):
        positions[str(name)] = [0.0, 0.5 + 0.2 * index, 0.0]
    return positions


def _record(
    task_rule: dict[str, Any],
    *,
    phase: str,
    action: float | None,
    query: int | None = None,
    eef: tuple[float, float, float] = (0.0, 0.0, 0.04),
    positions: dict[str, list[float]] | None = None,
    truths: dict[str, bool] | None = None,
    mechanism_joint: float = 0.0,
) -> dict[str, Any]:
    object_positions = _base_positions(task_rule)
    object_positions.update(positions or {})
    truths = truths or {}
    spatial_goals = [
        goal
        for goal in task_rule["goal_predicates"]
        if str(goal["predicate"]).lower() in {"in", "on"}
    ]
    spatial_targets = {str(goal["args"][0]) for goal in spatial_goals}
    spatial_destinations = {str(goal["args"][1]) for goal in spatial_goals}
    mechanisms = {
        str(goal["args"][0])
        for goal in task_rule["goal_predicates"]
        if str(goal["predicate"]).lower() not in {"in", "on"}
    }
    objects = {}
    for name, position in object_positions.items():
        if name in spatial_targets:
            kind = "movable_object"
            parent = None
            joint_names = [f"{name}_joint0"]
        elif name in spatial_destinations:
            kind = "task_site"
            parent = f"{name}_parent"
            joint_names = ["mechanism_joint"] if name in mechanisms else []
        elif name in mechanisms:
            kind = "fixture"
            parent = None
            joint_names = ["mechanism_joint"]
        else:
            kind = "movable_object"
            parent = None
            joint_names = [f"{name}_joint0"]
        objects[name] = {
            "kind": kind,
            "position": position,
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "joint_names": joint_names,
        }
        if parent is not None:
            objects[name]["parent"] = parent
    return {
        "step": 0,
        "sim_time": 0.0,
        "phase": phase,
        "policy_query_index": query,
        "action_chunk_index": 0 if query is not None else None,
        "robot": {
            "eef_position": list(eef),
            "eef_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "gripper_qpos": [0.02, -0.02],
        },
        "action": None if action is None else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, action],
        "objects": objects,
        "joints": {"mechanism_joint": [mechanism_joint]},
        "task_predicates": [
            {
                "predicate": str(goal["predicate"]).lower(),
                "arguments": list(goal["args"]),
                "satisfied": bool(truths.get(str(goal["id"]), False)),
            }
            for goal in task_rule["goal_predicates"]
        ],
        "success": False,
    }


def _trace(task_id: int, records: list[dict[str, Any]], *, trial_id: int = 0) -> dict[str, Any]:
    for step, record in enumerate(records):
        record["step"] = step
        record["sim_time"] = 0.05 * step
    return {
        "schema_version": 1,
        "task_suite": "libero_10",
        "task_id": task_id,
        "trial_id": trial_id,
        "task_description": RULES[task_id]["language"],
        "success": False,
        "environment_steps": len(records) - 1,
        "records": records,
    }


def _analyze(payload: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    state_path = tmp_path / f"task_{payload['task_id']}_episode_{payload['trial_id']}.json"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    return analyze_state_trace(
        payload,
        state_path=state_path,
        checkpoint_step=75000,
        task_rules=RULES,
        thresholds=RULE_THRESHOLDS,
    )


def test_failed_grasp_then_empty_gripper_approaches_basket(tmp_path: Path) -> None:
    rule = RULES[0]
    payload = _trace(
        0,
        [
            _record(rule, phase="initial", action=None),
            _record(rule, phase="settling", action=-1.0),
            _record(rule, phase="policy", action=1.0, query=1),
            _record(rule, phase="policy", action=-1.0, query=2, eef=(0.25, 0.0, 0.04)),
            _record(rule, phase="policy", action=-1.0, query=3, eef=(1.0, 0.0, 0.04)),
        ],
    )

    row = _analyze(payload, tmp_path)

    assert row["primary_failure"] == "抓取/接触失败"
    assert row["secondary_failure"] == "动作结果判断错误"
    assert row["outcome_awareness_failure"] is True
    assert "grasp_failed" in row["rule_labels"]
    assert "outcome_awareness_failure" in row["rule_labels"]


@pytest.mark.parametrize("task_id", [3, 9])
def test_closing_before_placement_is_task_aware_outcome_failure(
    task_id: int, tmp_path: Path
) -> None:
    rule = RULES[task_id]
    spatial_goal = next(
        goal for goal in rule["goal_predicates"] if goal["predicate"].lower() in {"in", "on"}
    )
    mechanism_goal = next(
        goal for goal in rule["goal_predicates"] if goal["predicate"].lower() not in {"in", "on"}
    )
    payload = _trace(
        task_id,
        [
            _record(rule, phase="initial", action=None),
            _record(rule, phase="settling", action=-1.0),
            _record(
                rule,
                phase="policy",
                action=-1.0,
                query=1,
                truths={str(mechanism_goal["id"]): True, str(spatial_goal["id"]): False},
                mechanism_joint=0.1,
            ),
            _record(
                rule,
                phase="policy",
                action=-1.0,
                query=2,
                truths={str(mechanism_goal["id"]): True, str(spatial_goal["id"]): False},
                mechanism_joint=0.1,
            ),
        ],
    )

    row = _analyze(payload, tmp_path)

    assert row["primary_failure"] == "搬运/放置失败"
    assert row["secondary_failure"] == "动作结果判断错误"
    assert row["outcome_awareness_failure"] is True
    assert row["recovery_failure"] is False
    assert {"placement_failure", "outcome_awareness_failure"}.issubset(row["rule_labels"])


def test_transport_then_release_with_false_native_predicate_is_placement_failure(
    tmp_path: Path,
) -> None:
    rule = RULES[5]
    payload = _trace(
        5,
        [
            _record(rule, phase="initial", action=None),
            _record(rule, phase="settling", action=-1.0),
            _record(
                rule,
                phase="policy",
                action=1.0,
                query=1,
                eef=(0.05, 0.0, 0.04),
                positions={"black_book_1": [0.05, 0.0, 0.0]},
            ),
            _record(
                rule,
                phase="policy",
                action=-1.0,
                query=2,
                eef=(0.25, 0.0, 0.04),
                positions={"black_book_1": [0.05, 0.0, 0.0]},
            ),
        ],
    )

    row = _analyze(payload, tmp_path)

    assert row["primary_failure"] == "搬运/放置失败"
    assert "placement_failure" in row["rule_labels"]
    assert "grasp_failed" not in row["rule_labels"]


def test_task8_lost_stove_invariant_is_mechanism_failure(tmp_path: Path) -> None:
    rule = RULES[8]
    stove_goal_id = "stove_on"
    initial_truths = {stove_goal_id: True}
    payload = _trace(
        8,
        [
            _record(rule, phase="initial", action=None, truths=initial_truths),
            _record(rule, phase="settling", action=-1.0, truths=initial_truths),
            _record(rule, phase="policy", action=-1.0, query=1),
        ],
    )

    row = _analyze(payload, tmp_path)

    assert row["primary_failure"] == "机构交互失败"
    assert row["furthest_stage"] == "stove_on_lost"
    assert row["rule_events"][0]["reason"] == "required_initial_predicate_lost"


def test_task7_repeated_failed_grasps_disturbance_and_recovery(tmp_path: Path) -> None:
    rule = RULES[7]
    disturbed = {"tomato_sauce_1": [0.05, 0.5, 0.0]}
    payload = _trace(
        7,
        [
            _record(rule, phase="initial", action=None),
            _record(rule, phase="settling", action=-1.0),
            _record(rule, phase="policy", action=1.0, query=1),
            _record(rule, phase="policy", action=-1.0, query=2, eef=(0.25, 0.0, 0.04)),
            _record(rule, phase="policy", action=1.0, query=3, positions=disturbed),
            _record(
                rule,
                phase="policy",
                action=-1.0,
                query=4,
                eef=(0.25, 0.0, 0.04),
                positions=disturbed,
            ),
            _record(rule, phase="policy", action=1.0, query=5, positions=disturbed),
            _record(
                rule,
                phase="policy",
                action=-1.0,
                query=6,
                eef=(0.25, 0.0, 0.04),
                positions=disturbed,
            ),
            _record(
                rule,
                phase="policy",
                action=-1.0,
                query=7,
                eef=(0.25, 0.0, 0.04),
                positions=disturbed,
            ),
        ],
    )

    row = _analyze(payload, tmp_path)

    assert row["primary_failure"] == "接近/位姿对齐失败"
    assert row["secondary_failure"] == "偏差累积/恢复失败"
    assert row["recovery_failure"] is True
    assert {
        "grasp_alignment_failure",
        "environment_disturbance",
        "recovery_failure",
    }.issubset(row["rule_labels"])


def test_missing_native_predicate_routes_to_manual_review(tmp_path: Path) -> None:
    rule = RULES[5]
    payload = _trace(
        5,
        [
            _record(rule, phase="initial", action=None),
            _record(rule, phase="settling", action=-1.0),
            _record(rule, phase="policy", action=1.0, query=1),
        ],
    )
    payload["records"][2]["task_predicates"] = []

    row = _analyze(payload, tmp_path)

    assert row["needs_manual_review"] is True
    assert row["primary_failure"] == "其他/无法判断"
    assert "missing registered goal predicates" in row["short_evidence"]


def test_analysis_outputs_preserve_manifest_and_rule_summary(tmp_path: Path) -> None:
    rule = RULES[5]
    payload = _trace(
        5,
        [
            _record(rule, phase="initial", action=None),
            _record(rule, phase="settling", action=-1.0),
            _record(rule, phase="policy", action=1.0, query=1),
            _record(rule, phase="policy", action=-1.0, query=2, eef=(0.25, 0.0, 0.04)),
        ],
    )
    state_path = tmp_path / "task_05" / "episode_000.json"
    state_path.parent.mkdir()
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    row = _analyze(payload, tmp_path)
    output_dir = tmp_path / "state_failure_analysis"
    write_state_analysis_outputs([row], output_dir, thresholds=RULE_THRESHOLDS)
    manifest = build_state_manifest(
        payload, state_path=state_path, checkpoint_step=75000
    )

    assert manifest["record_count"] == len(payload["records"])
    with (output_dir / "state_failure_analysis.csv").open(encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["primary_failure"] == "抓取/接触失败"
    with (output_dir / "state_rule_summary.csv").open(encoding="utf-8") as handle:
        labels = {row["rule_label"] for row in csv.DictReader(handle)}
    assert "grasp_failed" in labels


def test_state_cli_scans_trace_and_writes_failure_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rule = RULES[5]
    payload = _trace(
        5,
        [
            _record(rule, phase="initial", action=None),
            _record(rule, phase="settling", action=-1.0),
            _record(rule, phase="policy", action=1.0, query=1),
            _record(rule, phase="policy", action=-1.0, query=2, eef=(0.25, 0.0, 0.04)),
        ],
    )
    state_dir = tmp_path / "libero_10" / "simulator_states"
    state_path = state_dir / "task_05" / "episode_000.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_libero_state_failures.py",
            "--state-dir",
            str(state_dir),
            "--checkpoint-step",
            "75000",
            "--output-dir",
            str(output_dir),
        ],
    )

    state_cli_main()

    assert (output_dir / "state_failure_manifest.jsonl").exists()
    with (output_dir / "state_failure_analysis.csv").open(encoding="utf-8") as handle:
        row = list(csv.DictReader(handle))[0]
    assert row["task"] == "5"
    assert row["primary_failure"] == "抓取/接触失败"
