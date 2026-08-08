from __future__ import annotations

import pytest

from side_model3_adapter_v2.scripts.eval_transition_action_dependence import (
    classify_action_dependence,
    evenly_spaced_indices,
    summarize_condition_rows,
)


def test_evenly_spaced_indices_are_unique_and_span_dataset() -> None:
    indices = evenly_spaced_indices(1_000, 8)
    assert len(indices) == 8
    assert len(set(indices)) == 8
    assert indices == sorted(indices)
    assert indices[0] < 100
    assert indices[-1] > 900


def test_evenly_spaced_indices_reject_impossible_count() -> None:
    with pytest.raises(ValueError):
        evenly_spaced_indices(4, 5)


def _rows(
    *,
    gt: list[float],
    shuffle: list[float],
    zero: list[float],
) -> list[dict[str, float]]:
    return [
        {
            "loss_gt": gt[index],
            "loss_shuffle": shuffle[index],
            "loss_zero": zero[index],
            "prediction_change_shuffle": 0.2,
            "prediction_change_zero": 0.3,
            "action_rms_change_shuffle": 0.4,
            "action_rms_change_zero": 0.5,
        }
        for index in range(len(gt))
    ]


def test_summary_and_strong_gate() -> None:
    horizon = summarize_condition_rows(
        _rows(
            gt=[1.0, 1.0, 1.0, 1.0],
            shuffle=[2.0, 2.0, 2.0, 2.0],
            zero=[3.0, 3.0, 3.0, 3.0],
        )
    )
    summaries = {"4": horizon, "8": horizon}
    assert horizon["conditions"]["shuffle"]["loss_ratio_of_means_to_gt"] == 2.0
    assert classify_action_dependence(summaries) == (
        "sufficient_action_dependence_for_v2_bridge"
    )


def test_weak_and_mixed_gates() -> None:
    weak = summarize_condition_rows(
        _rows(
            gt=[1.0, 1.0, 1.0, 1.0],
            shuffle=[1.05, 1.05, 1.05, 1.05],
            zero=[1.09, 1.09, 1.09, 1.09],
        )
    )
    assert classify_action_dependence({"4": weak, "8": weak}) == (
        "weak_action_dependence_do_not_launch_v2_bridge"
    )

    mixed = summarize_condition_rows(
        _rows(
            gt=[1.0, 1.0, 1.0, 1.0],
            shuffle=[1.5, 1.5, 1.5, 1.5],
            zero=[1.05, 1.05, 1.05, 1.05],
        )
    )
    assert classify_action_dependence({"4": mixed, "8": mixed}) == (
        "mixed_action_dependence_requires_judgment"
    )
