from pathlib import Path

import pytest

from model3.scripts.finalize_spatial_eval import (
    RunContract,
    TaskResult,
    ValidationError,
    VideoResult,
    _build_metric_rows,
    _build_validation_payload,
    _load_videos,
)
from model3.video_retention import (
    expected_retained_episode_ids,
    should_retain_rollout_video,
)


def test_retains_every_failure_and_one_in_ten_successes() -> None:
    assert should_retain_rollout_video(trial_idx=0, success=True)
    assert should_retain_rollout_video(trial_idx=10, success=True)
    assert not should_retain_rollout_video(trial_idx=1, success=True)
    assert should_retain_rollout_video(trial_idx=1, success=False)

    retained = expected_retained_episode_ids(
        successes=[0, 2, 10, 11, 20],
        failures=[1, 3],
        success_video_stride=10,
    )
    assert retained == frozenset({0, 1, 3, 10, 20})


@pytest.mark.parametrize(
    ("trial_idx", "success", "stride"),
    [(-1, True, 10), (0, True, 0), (0, True, -1), (0, 1, 10)],
)
def test_rejects_invalid_retention_inputs(trial_idx, success, stride) -> None:
    with pytest.raises(ValueError):
        should_retain_rollout_video(
            trial_idx=trial_idx,
            success=success,
            success_video_stride=stride,
        )


def _contract(tmp_path: Path) -> RunContract:
    return RunContract(
        run_root=tmp_path,
        output_dir=tmp_path / "eval_output",
        track_id="model3",
        dataset="LIBERO",
        suite="libero_spatial",
        evidence_scope="local_training_eval",
        num_tasks=1,
        trials_per_task=3,
        total_episodes=3,
        max_steps=400,
        seed=42,
        seed_source="test",
        checkpoint_path=tmp_path / "checkpoints/weights/step_1.pt",
        checkpoint_sha256="a" * 64,
        training_config_path=tmp_path / "config.yaml",
        gpu_ids=(0,),
        success_video_stride=10,
        save_all_failure_videos=True,
        video_retention_declared=True,
    )


def test_sparse_videos_do_not_remove_metric_rows_or_fabricate_steps(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    result = TaskResult(
        path=tmp_path / "task.json",
        gpu_id=0,
        task_id=0,
        task_name="test task",
        successes=frozenset({0, 1}),
        failures=frozenset({2}),
        saved_video_episodes=frozenset({0, 2}),
        duration_seconds=12.0,
    )
    videos = {
        (0, 0): VideoResult(
            path=tmp_path / "eval_output/libero_spatial/videos/success.mp4",
            success=True,
            steps_executed=8,
        ),
        (0, 2): VideoResult(
            path=tmp_path / "eval_output/libero_spatial/videos/failure.mp4",
            success=False,
            steps_executed=400,
        ),
    }

    episode_rows, task_rows = _build_metric_rows(contract, {0: result}, videos)

    assert len(episode_rows) == 3
    assert episode_rows[1]["success"] == "true"
    assert episode_rows[1]["steps_executed"] == ""
    assert episode_rows[1]["video_path"] == ""
    assert episode_rows[2]["success"] == "false"
    assert episode_rows[2]["steps_executed"] == 400
    assert task_rows[0]["success_rate"] == "0.666667"
    assert task_rows[0]["mean_steps"] == ""

    validation = _build_validation_payload(
        contract,
        {0: result},
        videos,
        {"method_id": "test", "model_class": "test", "step": 1},
        b"episodes",
        b"tasks",
    )
    assert validation["results"]["valid_episodes"] == 3
    assert validation["results"]["rollout_videos"] == 2
    assert validation["results"]["successful_rollout_videos"] == 1
    assert validation["results"]["failed_rollout_videos"] == 1
    assert validation["checks"]["all_failure_videos_retained"]


def test_finalizer_rejects_a_missing_failure_video(tmp_path: Path, monkeypatch) -> None:
    contract = _contract(tmp_path)
    result = TaskResult(
        path=tmp_path / "task.json",
        gpu_id=0,
        task_id=0,
        task_name="test task",
        successes=frozenset({0, 1}),
        failures=frozenset({2}),
        saved_video_episodes=frozenset({0, 2}),
        duration_seconds=12.0,
    )
    video_dir = contract.output_dir / contract.suite / "videos"
    video_dir.mkdir(parents=True)
    success_path = video_dir / (
        "rollout--episode=task0_trial0--success=True--task=test_task.mp4"
    )
    success_path.touch()
    monkeypatch.setattr(
        "model3.scripts.finalize_spatial_eval._count_video_frames",
        lambda path: 8,
    )

    with pytest.raises(ValidationError, match="failed episodes are missing"):
        _load_videos(contract, {0: result})
