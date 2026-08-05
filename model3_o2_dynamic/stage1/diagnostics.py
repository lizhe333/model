"""Post-export, held-out-noise Stage-1 response diagnostics.

These metrics are deliberately *soft*: they are recorded after the fixed
step-5K adapter export and never select a checkpoint, tune a hyperparameter,
or decide whether Stage 2 is legal.  The module keeps test carriers lazy so
the held-out split is not folded into the train/validation cache.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from model3_o2_dynamic.config import Model3O2DynamicConfig, load_config
from model3_o2_dynamic.prepare_stage2 import verify_adapter_export

from .action_normalization import load_official_o2_normalizer
from .cache import CurrentHiddenReader, current_hidden, load_response_cache, sha256_file, standardized_targets
from .carrier import parent_config_path, validate_carrier_shard
from .collect import validate_collection_shard
from .contracts import LAYERS, Stage1DataConfig, Stage1ContractError, task_names_for_suite
from .export import tensor_state_sha256
from .teacher import validate_teacher_shard
from model3_o2_dynamic.models.response_predictor import TokenResponsePredictor


DIAGNOSTIC_SEED = 85201
DIAGNOSTIC_STEPS = 1000
DIAGNOSTIC_BATCH_SIZE = 64
DIAGNOSTIC_LEARNING_RATE = 3.0e-4
DIAGNOSTIC_WEIGHT_DECAY = 1.0e-4


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class HeldoutResponseData:
    payload: dict[str, Any]
    source_identity: dict[str, Any]
    teacher_identity: dict[str, Any]
    action_normalization_identity: dict[str, Any]


def _same_record(left: dict[str, Any], right: dict[str, Any], *, source: str) -> None:
    for key in ("sample_id", "task", "demo_id", "split", "source_index", "progress_bin"):
        if left.get(key) != right.get(key):
            raise Stage1ContractError(f"held-out {source} record alignment failed at {key}")


def _heldout_targets(teacher: dict[str, Any], config: Stage1DataConfig) -> torch.Tensor:
    e0 = teacher["e0_global"].float()
    expected = (
        config.states_per_demo,
        len(LAYERS),
        len(config.heldout_noise_seeds),
        4,
        len(config.stage_render_ticks),
        config.projection_dim,
    )
    if tuple(e0.shape) != expected:
        raise Stage1ContractError(f"held-out E0 geometry changed: {tuple(e0.shape)} vs {expected}")
    mean = e0.mean(dim=2)
    response = mean[:, :, :3] - mean[:, :, 3:4]
    if tuple(response.shape) != (config.states_per_demo, 3, 3, 4, 256):
        raise Stage1ContractError(f"held-out response geometry changed: {tuple(response.shape)}")
    return response


def build_heldout_response_data(
    *,
    dynamic_config: Model3O2DynamicConfig,
    output_root: str | Path,
    train_cache_path: str | Path,
) -> HeldoutResponseData:
    """Read the sealed test shards only after an adapter export exists."""

    config = Stage1DataConfig()
    root = Path(output_root).expanduser().resolve()
    train_cache, cache_identity = load_response_cache(train_cache_path, require_trainable_splits_only=True)
    normalizer = load_official_o2_normalizer(parent_config_path=parent_config_path(dynamic_config))
    action_identity = normalizer.identity()
    if train_cache["action_normalization_identity"] != action_identity:
        raise Stage1ContractError("held-out diagnostics normalizer differs from the sealed train cache")

    actions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    index: list[list[int]] = []
    source_identity: dict[str, Any] | None = None
    teacher_identity: dict[str, Any] | None = None
    for task in task_names_for_suite(dynamic_config.evaluation.suite):
        for demo_id in range(45, 50):
            collection_path = root / "collection" / "test" / task / f"demo_{demo_id:03d}.pt"
            teacher_path = root / "teacher" / "test" / task / f"demo_{demo_id:03d}.pt"
            carrier_path = root / "carrier" / "test" / task / f"demo_{demo_id:03d}.pt"
            for path, label in ((collection_path, "collection"), (teacher_path, "teacher"), (carrier_path, "carrier")):
                if not path.is_file():
                    raise FileNotFoundError(f"missing held-out {label} shard: {path}")
            collection = torch.load(collection_path, map_location="cpu", weights_only=False)
            teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
            carrier = torch.load(carrier_path, map_location="cpu", weights_only=False)
            validate_collection_shard(collection, config=config)
            validate_teacher_shard(teacher, split="test", task=task, config=config)
            validate_carrier_shard(carrier, split="test", task=task, config=config)
            if carrier.get("source_identity") != train_cache["source_identity"]:
                raise Stage1ContractError("held-out carrier source identity differs from train cache")
            if carrier.get("action_normalization_identity") != action_identity:
                raise Stage1ContractError("held-out carrier action normalizer differs from train cache")
            if source_identity is None:
                source_identity = dict(carrier["source_identity"])
            elif source_identity != carrier["source_identity"]:
                raise Stage1ContractError("held-out carrier source identity differs across shards")
            if teacher_identity is None:
                teacher_identity = dict(teacher["teacher_identity"])
            elif teacher_identity != teacher["teacher_identity"]:
                raise Stage1ContractError("held-out teacher identity differs across shards")
            shard_index = len(shards)
            shards.append(
                {
                    "path": str(carrier_path),
                    "sha256": sha256_file(carrier_path),
                    "shape": list(carrier["current_hidden"].shape),
                    "dtype": str(carrier["current_hidden"].dtype),
                    "task": task,
                    "split": "test",
                    "demo_id": demo_id,
                }
            )
            actions.append(
                normalizer.normalize_action(
                    torch.stack([record["actions"] for record in collection["records"]], dim=0)
                )
            )
            targets.append(_heldout_targets(teacher, config))
            for local_index, (collection_record, teacher_record, carrier_record) in enumerate(
                zip(collection["records"], teacher["records"], carrier["records"])
            ):
                _same_record(collection_record, teacher_record, source="teacher")
                _same_record(collection_record, carrier_record, source="carrier")
                records.append(
                    {
                        key: collection_record[key]
                        for key in (
                            "sample_id",
                            "task",
                            "demo_id",
                            "split",
                            "source_index",
                            "progress_bin",
                        )
                    }
                )
                index.append([shard_index, local_index])
    if source_identity is None or teacher_identity is None:
        raise Stage1ContractError("held-out diagnostic build found no test shards")
    action_tensor = torch.cat(actions, dim=0).contiguous()
    target_tensor = torch.cat(targets, dim=0).contiguous()
    index_tensor = torch.tensor(index, dtype=torch.int64)
    if tuple(action_tensor.shape) != (500, 4, 8, 7):
        raise Stage1ContractError(f"held-out action geometry changed: {tuple(action_tensor.shape)}")
    if tuple(target_tensor.shape) != (500, 3, 3, 4, 256):
        raise Stage1ContractError(f"held-out target geometry changed: {tuple(target_tensor.shape)}")
    if tuple(index_tensor.shape) != (500, 2) or len(records) != 500:
        raise Stage1ContractError("held-out carrier index geometry changed")
    payload = {
        "current_hidden": None,
        "current_hidden_shards": shards,
        "current_hidden_index": index_tensor,
        "actions": action_tensor,
        "response_targets": target_tensor,
        "normalization_mean": train_cache["normalization_mean"],
        "normalization_std": train_cache["normalization_std"],
        "records": records,
    }
    # CurrentHiddenReader verifies each shard hash lazily, before it is used.
    payload["_current_hidden_reader"] = CurrentHiddenReader(payload)
    if cache_identity["split_counts"] != {"train": 4000, "validation": 500, "test": 0}:
        raise Stage1ContractError("held-out diagnostics require the exact sealed train/validation cache")
    return HeldoutResponseData(
        payload=payload,
        source_identity=source_identity,
        teacher_identity=teacher_identity,
        action_normalization_identity=action_identity,
    )


def _verify_fixed_stage1_export(
    *,
    stage1_output_dir: str | Path,
    adapter_export: str | Path,
    expected_parent_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1_root = Path(stage1_output_dir).expanduser().resolve()
    checkpoint_path = stage1_root / "audit_checkpoints" / "step_005000.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing fixed Stage-1 audit checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_kind") != "stage1_response_audit_checkpoint" or checkpoint.get("step") != 5000:
        raise Stage1ContractError("held-out diagnostics require the fixed Stage-1 5K audit checkpoint")
    export_identity = verify_adapter_export(adapter_export, expected_parent_sha256=expected_parent_sha256)
    adapter_state = {
        name.removeprefix("adapters."): value
        for name, value in checkpoint["model_state_dict"].items()
        if name.startswith("adapters.")
    }
    if tensor_state_sha256(adapter_state) != export_identity["adapter_state_sha256"]:
        raise Stage1ContractError("fixed 5K audit checkpoint adapters differ from the deployment-only export")
    return checkpoint, export_identity


class ConditionalPredictorBank(torch.nn.Module):
    """Three small $Q_l$ diagnostic models with one fixed input condition."""

    def __init__(self, *, condition: str, width: int) -> None:
        super().__init__()
        predictor_condition = "full" if condition == "shuffled" else condition
        self.condition = condition
        self.predictors = torch.nn.ModuleDict(
            {
                str(layer): TokenResponsePredictor(width=width, condition=predictor_condition)
                for layer in LAYERS
            }
        )
        oversized = {
            layer: predictor.parameter_count()
            for layer, predictor in ((layer, self.predictors[str(layer)]) for layer in LAYERS)
            if predictor.parameter_count() > 2_500_000
        }
        if oversized:
            raise Stage1ContractError(f"held-out diagnostic Q parameter cap exceeded: {oversized}")

    def forward(
        self,
        *,
        current_hidden: torch.Tensor,
        actions: torch.Tensor,
        teacher_timestep: int,
    ) -> torch.Tensor:
        if tuple(current_hidden.shape[1:]) != (3, 392, 1536):
            raise ValueError(f"diagnostic current hidden must be [B,3,392,1536], got {tuple(current_hidden.shape)}")
        if tuple(actions.shape[1:]) != (4, 8, 7):
            raise ValueError(f"diagnostic actions must be [B,4,8,7], got {tuple(actions.shape)}")
        batch = int(current_hidden.shape[0])
        timestep = torch.full(
            (batch,),
            float(teacher_timestep),
            device=current_hidden.device,
            dtype=torch.float32,
        )
        layer_values: list[torch.Tensor] = []
        for position, layer in enumerate(LAYERS):
            predictor = self.predictors[str(layer)]
            branch_values = [
                predictor(current_hidden[:, position], actions[:, branch], timestep)
                for branch in range(3)
            ]
            layer_values.append(torch.stack(branch_values, dim=1))
        return torch.stack(layer_values, dim=1)


def _diagnostic_response_loss(
    model: ConditionalPredictorBank,
    *,
    current_hidden: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    teacher_timestep: int,
    beta_anchor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted = model(
        current_hidden=current_hidden,
        actions=actions,
        teacher_timestep=teacher_timestep,
    )
    if tuple(predicted.shape) != tuple(targets.shape):
        raise RuntimeError(f"diagnostic prediction/target mismatch: {tuple(predicted.shape)} vs {tuple(targets.shape)}")
    total = predicted.new_zeros((), dtype=torch.float32)
    metrics: dict[str, torch.Tensor] = {}
    for position, layer in enumerate(LAYERS):
        layer_prediction = predicted[:, position]
        layer_target = targets[:, position]
        anchor = (layer_prediction - layer_target).float().square().mean()
        local = (
            (layer_prediction[:, 1:] - layer_prediction[:, :1])
            - (layer_target[:, 1:] - layer_target[:, :1])
        ).float().square().mean()
        total = total + local + float(beta_anchor) * anchor
        metrics[f"loss_anchor_l{layer}"] = anchor.detach()
        metrics[f"loss_local_l{layer}"] = local.detach()
    metrics["loss_response"] = total.detach()
    return total, metrics


def _conditioned_batch(
    *,
    payload: dict[str, Any],
    indices: torch.Tensor,
    mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(payload["records"])
    if mode == "action_only":
        # Action-only Q explicitly ignores these values, but its shared input
        # validator keeps the official carrier geometry visible in the audit.
        current = torch.empty((1, 3, 392, 1536), device=device, dtype=torch.float32).expand(
            len(indices), -1, -1, -1
        )
        actions = payload["actions"][indices]
    elif mode == "state_only":
        current = current_hidden(payload, indices).to(device=device, dtype=torch.float32)
        actions = torch.zeros((len(indices), 4, 8, 7), device=device, dtype=torch.float32)
    elif mode == "full":
        current = current_hidden(payload, indices).to(device=device, dtype=torch.float32)
        actions = payload["actions"][indices]
    elif mode == "shuffled":
        current = current_hidden(payload, indices).to(device=device, dtype=torch.float32)
        actions = payload["actions"][(indices + 1) % count]
    else:
        raise ValueError(f"unsupported held-out diagnostic mode: {mode}")
    return current, actions.to(device=device, dtype=torch.float32)


@torch.no_grad()
def _evaluate_predictor(
    *,
    model: ConditionalPredictorBank,
    payload: dict[str, Any],
    mode: str,
    device: torch.device,
    teacher_timestep: int,
    beta_anchor: float,
    batch_size: int,
    evaluation_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    indices = (
        torch.arange(len(payload["records"]), dtype=torch.long)
        if evaluation_indices is None
        else evaluation_indices.detach().to(device="cpu", dtype=torch.long).flatten()
    )
    count = len(indices)
    if count == 0:
        raise Stage1ContractError("cannot evaluate an empty held-out diagnostic split")
    totals: dict[str, float] = {}
    for start in range(0, count, batch_size):
        selected = indices[start : start + batch_size]
        current, actions = _conditioned_batch(
            payload=payload,
            indices=selected,
            mode=mode,
            device=device,
        )
        targets = standardized_targets(payload, selected, device=device)
        _, metrics = _diagnostic_response_loss(
            model,
            current_hidden=current,
            actions=actions,
            targets=targets,
            teacher_timestep=teacher_timestep,
            beta_anchor=beta_anchor,
        )
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * len(selected)
    return {name: value / count for name, value in totals.items()}


def _train_predictor_diagnostic(
    *,
    mode: str,
    train_payload: dict[str, Any],
    validation_indices: torch.Tensor,
    test_payload: dict[str, Any],
    predictor_width: int,
    teacher_timestep: int,
    beta_anchor: float,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = ConditionalPredictorBank(condition=mode, width=predictor_width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DIAGNOSTIC_LEARNING_RATE,
        weight_decay=DIAGNOSTIC_WEIGHT_DECAY,
    )
    train_indices = torch.tensor(
        [index for index, record in enumerate(train_payload["records"]) if record["split"] == "train"],
        dtype=torch.long,
    )
    if len(train_indices) != 4000 or len(validation_indices) != 500:
        raise Stage1ContractError("diagnostic split cardinality differs from the sealed Stage-1 cache")
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    last_metrics: dict[str, float] = {}
    model.train()
    for _ in range(int(steps)):
        positions = torch.randint(0, len(train_indices), (int(batch_size),), generator=generator)
        selected = train_indices[positions]
        current, actions = _conditioned_batch(
            payload=train_payload,
            indices=selected,
            mode=mode,
            device=device,
        )
        targets = standardized_targets(train_payload, selected, device=device)
        loss, metrics = _diagnostic_response_loss(
            model,
            current_hidden=current,
            actions=actions,
            targets=targets,
            teacher_timestep=teacher_timestep,
            beta_anchor=beta_anchor,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_metrics = {name: float(value) for name, value in metrics.items()}
    model.eval()
    return {
        "seed": int(seed),
        "train_steps": int(steps),
        "predictor_parameter_counts": {
            str(layer): int(model.predictors[str(layer)].parameter_count()) for layer in LAYERS
        },
        "last_train_metrics": last_metrics,
        "validation": _evaluate_predictor(
            model=model,
            payload=train_payload,
            mode=mode,
            device=device,
            teacher_timestep=teacher_timestep,
            beta_anchor=beta_anchor,
            batch_size=batch_size,
            evaluation_indices=validation_indices,
        ),
        "test": _evaluate_predictor(
            model=model,
            payload=test_payload,
            mode=mode,
            device=device,
            teacher_timestep=teacher_timestep,
            beta_anchor=beta_anchor,
            batch_size=batch_size,
        ),
    }


def run_heldout_input_ablations(
    *,
    dynamic_config: Model3O2DynamicConfig,
    output_root: str | Path,
    train_cache_path: str | Path,
    stage1_output_dir: str | Path,
    adapter_export: str | Path,
    device: str | torch.device = "cuda:0",
    batch_size: int = DIAGNOSTIC_BATCH_SIZE,
) -> dict[str, Any]:
    """Fit four sealed small-$Q$ input diagnostics, then score held-out test."""

    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("held-out Dynamic Stage-1 diagnostics require a CUDA device")
    stage1_config_raw = dynamic_config.stage1
    predictor_width = int(stage1_config_raw["predictor_width"])
    heldout = build_heldout_response_data(
        dynamic_config=dynamic_config,
        output_root=output_root,
        train_cache_path=train_cache_path,
    )
    train_cache, cache_identity = load_response_cache(
        train_cache_path,
        require_trainable_splits_only=True,
    )
    checkpoint, export_identity = _verify_fixed_stage1_export(
        stage1_output_dir=stage1_output_dir,
        adapter_export=adapter_export,
        expected_parent_sha256=dynamic_config.initialization.model3_checkpoint_sha256,
    )
    if cache_identity["split_counts"] != {"train": 4000, "validation": 500, "test": 0}:
        raise Stage1ContractError("diagnostics require the sealed 4K/500 train/validation cache")
    validation_indices = torch.tensor(
        [index for index, record in enumerate(train_cache["records"]) if record["split"] == "validation"],
        dtype=torch.long,
    )
    modes = ("full", "action_only", "state_only", "shuffled")
    metrics = {
        mode: _train_predictor_diagnostic(
            mode=mode,
            train_payload=train_cache,
            validation_indices=validation_indices,
            test_payload=heldout.payload,
            predictor_width=predictor_width,
            teacher_timestep=int(stage1_config_raw["teacher_timestep"]),
            beta_anchor=float(stage1_config_raw["beta_anchor"]),
            device=target_device,
            seed=DIAGNOSTIC_SEED + modes.index(mode),
            steps=DIAGNOSTIC_STEPS,
            batch_size=int(batch_size),
        )
        for mode in modes
    }
    root = Path(output_root).expanduser().resolve()
    result = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_heldout_input_ablation_metrics",
        "diagnostic_kind": "four_small_q_conditional_predictability_models",
        "soft_diagnostic_only": True,
        "test_read_after_fixed_stage1_export": True,
        "test_state_count": len(heldout.payload["records"]),
        "heldout_noise_seeds": list(Stage1DataConfig().heldout_noise_seeds),
        "stage1_audit_checkpoint": str(Path(stage1_output_dir).expanduser().resolve() / "audit_checkpoints" / "step_005000.pt"),
        "stage1_audit_checkpoint_sha256": sha256_file(Path(stage1_output_dir).expanduser().resolve() / "audit_checkpoints" / "step_005000.pt"),
        "adapter_export": export_identity,
        "source_identity": heldout.source_identity,
        "teacher_identity": heldout.teacher_identity,
        "action_normalization_identity": heldout.action_normalization_identity,
        "diagnostic_optimizer": {
            "seed_base": DIAGNOSTIC_SEED,
            "steps_per_model": DIAGNOSTIC_STEPS,
            "batch_size": int(batch_size),
            "learning_rate": DIAGNOSTIC_LEARNING_RATE,
            "weight_decay": DIAGNOSTIC_WEIGHT_DECAY,
            "uses_response_adapters": False,
        },
        "metrics": metrics,
        "mode_definitions": {
            "full": "small Q_l receives matched current hidden and normalized action",
            "action_only": "small Q_l receives normalized action only",
            "state_only": "small Q_l receives current hidden only",
            "shuffled": "small Q_l receives matched current hidden and deterministically mismatched action",
        },
        "stage1_checkpoint_step": int(checkpoint["step"]),
    }
    _write_json(root / "diagnostics" / "heldout_input_ablations.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--stage1-output-dir", type=Path, required=True)
    parser.add_argument("--adapter-export", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DIAGNOSTIC_BATCH_SIZE)
    args = parser.parse_args()
    print(
        json.dumps(
            run_heldout_input_ablations(
                dynamic_config=load_config(args.config),
                output_root=args.output_root,
                train_cache_path=args.train_cache,
                stage1_output_dir=args.stage1_output_dir,
                adapter_export=args.adapter_export,
                device=args.device,
                batch_size=args.batch_size,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
