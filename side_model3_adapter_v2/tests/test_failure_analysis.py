from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from side_model3_adapter_v2.failure_analysis import (
    FAILURE_TAXONOMY,
    annotate_manual_review,
    attach_and_validate_analyses,
    create_storyboard,
    load_stage_rules,
    parse_failure_video_path,
    select_storyboard_frame_indices,
    write_analysis_outputs,
)


STAGE_RULES = (
    Path(__file__).resolve().parents[1] / "configs" / "libero_long_stage_rules.json"
)


def _write_test_video(path: Path, frame_count: int = 32) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (128, 64),
    )
    assert writer.isOpened()
    for frame_index in range(frame_count):
        frame = np.full((64, 128, 3), frame_index * 4, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_parse_failure_video_path_and_checkpoint() -> None:
    path = Path(
        "/tmp/solver10_step_075000/eval_output/libero_10/videos/"
        "2026_08_08-08_39_11--episode=task1_trial2--success=False--"
        "task=put_both_the_cream_cheese_box_and_the_butter_in_th.mp4"
    )
    episode = parse_failure_video_path(path)
    assert episode.checkpoint_step == 75000
    assert episode.task == 1
    assert episode.trial == 2


def test_storyboard_keeps_sixteen_uniform_frames_plus_endpoints(tmp_path: Path) -> None:
    video_path = tmp_path / (
        "2026_08_08-08_39_11--episode=task2_trial5--success=False--"
        "task=turn_on_the_stove_and_put_the_moka_pot_on_it.mp4"
    )
    _write_test_video(video_path)
    episode = parse_failure_video_path(video_path)
    storyboard_path = tmp_path / "storyboard.jpg"
    metadata = create_storyboard(episode, storyboard_path, tile_width=128)

    assert storyboard_path.exists()
    assert len(metadata["frame_indices"]) == 18
    assert metadata["frame_indices"][0] == 0
    assert metadata["frame_indices"][-1] == 31
    assert metadata["timestamps_seconds"][0] == 0.0
    assert metadata["timestamps_seconds"][-1] == 31 / 8.0


def test_uniform_frame_indices_are_unique_and_include_endpoints() -> None:
    indices = select_storyboard_frame_indices(700)
    assert len(indices) == 18
    assert len(set(indices)) == 18
    assert indices[0] == 0
    assert indices[-1] == 699


def test_analysis_validation_aggregation_and_manual_review(tmp_path: Path) -> None:
    stage_rules = load_stage_rules(STAGE_RULES)
    manifest = [
        {
            "checkpoint_step": 75000,
            "task": 1,
            "trial": 2,
            "video_path": "/tmp/failure.mp4",
            "storyboard_path": "/tmp/storyboard.jpg",
        },
        {
            "checkpoint_step": 75000,
            "task": 2,
            "trial": 5,
            "video_path": "/tmp/failure2.mp4",
            "storyboard_path": "/tmp/storyboard2.jpg",
        },
    ]
    analyses = [
        {
            "checkpoint_step": 75000,
            "task": 1,
            "trial": 2,
            "primary_failure": "动作结果判断错误",
            "secondary_failure": "偏差累积/恢复失败",
            "outcome_awareness_failure": True,
            "recovery_failure": True,
            "furthest_stage": "cream_cheese_in_basket_only",
            "confidence": 0.9,
            "short_evidence": "未抓住第二件物体后仍执行放置序列。",
        },
        {
            "checkpoint_step": 75000,
            "task": 2,
            "trial": 5,
            "primary_failure": FAILURE_TAXONOMY[1],
            "secondary_failure": None,
            "outcome_awareness_failure": False,
            "recovery_failure": False,
            "furthest_stage": "stove_on_only",
            "confidence": 0.65,
            "short_evidence": "末端执行器未对准目标接触位置。",
        },
    ]
    rows = attach_and_validate_analyses(
        manifest,
        analyses,
        stage_rules,
        analyzer="test_vlm",
    )
    rows = write_analysis_outputs(rows, tmp_path)

    assert rows[0]["needs_manual_review"] is False
    assert rows[1]["needs_manual_review"] is True

    with (tmp_path / "failure_analysis.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    with (tmp_path / "needs_manual_review.csv").open(encoding="utf-8") as handle:
        manual_rows = list(csv.DictReader(handle))
    assert len(manual_rows) == 1
    assert manual_rows[0]["task"] == "2"
    with (tmp_path / "failure_summary.csv").open(encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert any(row["task"] == "ALL" for row in summary_rows)


def test_manual_review_catches_uncertain_stage_primary_and_evidence() -> None:
    rows = [
        {
            "confidence": 0.99,
            "furthest_stage": "无法判断",
            "primary_failure": "接近/位姿对齐失败",
            "short_evidence": "柜体遮挡导致阶段无法判断。",
        },
        {
            "confidence": 0.99,
            "furthest_stage": "stove_on_only",
            "primary_failure": "其他/无法判断",
            "short_evidence": "视觉证据充分。",
        },
        {
            "confidence": 0.75,
            "furthest_stage": "stove_on_only",
            "primary_failure": "接近/位姿对齐失败",
            "short_evidence": "目标仍在初始位置。",
        },
    ]
    annotated = annotate_manual_review(rows)

    assert [row["needs_manual_review"] for row in annotated] == [True, True, False]


def test_analysis_requires_one_record_per_manifest_episode() -> None:
    stage_rules = load_stage_rules(STAGE_RULES)
    manifest = [
        {
            "checkpoint_step": 75000,
            "task": 1,
            "trial": 2,
            "video_path": "/tmp/failure.mp4",
            "storyboard_path": "/tmp/storyboard.jpg",
        }
    ]
    with pytest.raises(ValueError, match="Missing analysis records"):
        attach_and_validate_analyses(
            manifest,
            [],
            stage_rules,
            analyzer="test_vlm",
        )
