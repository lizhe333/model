"""Deterministic rollout-video retention policy for Model3 evaluation."""

from __future__ import annotations

from collections.abc import Iterable


def should_retain_rollout_video(
    *,
    trial_idx: int,
    success: bool,
    success_video_stride: int = 10,
) -> bool:
    """Retain every failure and one deterministic stride of successful trials."""

    if type(trial_idx) is not int or trial_idx < 0:
        raise ValueError(f"trial_idx must be a non-negative integer, got {trial_idx!r}")
    if type(success) is not bool:
        raise ValueError(f"success must be a boolean, got {success!r}")
    if type(success_video_stride) is not int or success_video_stride < 1:
        raise ValueError(
            "success_video_stride must be a positive integer, "
            f"got {success_video_stride!r}"
        )
    return not success or trial_idx % success_video_stride == 0


def expected_retained_episode_ids(
    *,
    successes: Iterable[int],
    failures: Iterable[int],
    success_video_stride: int,
) -> frozenset[int]:
    """Return the exact episode ids required by the retention policy."""

    success_ids = frozenset(successes)
    failure_ids = frozenset(failures)
    retained_success_ids = {
        episode_id
        for episode_id in success_ids
        if should_retain_rollout_video(
            trial_idx=episode_id,
            success=True,
            success_video_stride=success_video_stride,
        )
    }
    for episode_id in failure_ids:
        should_retain_rollout_video(
            trial_idx=episode_id,
            success=False,
            success_video_stride=success_video_stride,
        )
    return frozenset(retained_success_ids | failure_ids)
