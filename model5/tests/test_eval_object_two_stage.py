from argparse import Namespace
from pathlib import Path

from model5.scripts.eval_object_two_stage import ObjectEvaluation


def test_summarizer_environment_is_bound_to_condition_checkpoint(tmp_path: Path) -> None:
    train_run = tmp_path / "train"
    checkpoint = train_run / "checkpoints" / "weights" / "step_010000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (train_run / "config.yaml").write_text("seed: 42\n", encoding="utf-8")

    evaluation = ObjectEvaluation(
        Namespace(
            root=tmp_path,
            vendor=tmp_path / "vendor",
            train_run=train_run,
            libero_root=tmp_path / "libero",
            python=Path("/usr/bin/python3"),
            run_root=tmp_path / "evaluation",
        )
    )

    environment = evaluation.environment(2, tmp_path / "condition", 10_000)

    assert environment["CKPT"] == str(checkpoint)
    assert environment["CONFIG"] == str(train_run / "config.yaml")
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"


def test_each_solver_condition_is_retained_as_local_evaluation_evidence(tmp_path: Path) -> None:
    train_run = tmp_path / "train"
    checkpoint = train_run / "checkpoints" / "weights" / "step_010000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    evaluation = ObjectEvaluation(
        Namespace(
            root=tmp_path,
            vendor=tmp_path / "vendor",
            train_run=train_run,
            libero_root=tmp_path / "libero",
            python=Path("/usr/bin/python3"),
            run_root=tmp_path / "evaluation",
        )
    )
    evaluation.records[10_000] = {
        "step": 10_000,
        "path": str(checkpoint),
        "sha256": "a" * 64,
    }
    condition = tmp_path / "solver5"

    evaluation.make_condition(condition, 10_000, 5)

    manifest = __import__("json").loads(
        (condition / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_type"] == "per_checkpoint_solver_eval"
    assert manifest["evidence_scope"] == "local_training_eval"
