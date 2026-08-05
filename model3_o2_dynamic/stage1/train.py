"""Train only $A_8/A_{16}/A_{24}$ and temporary $Q_8/Q_{16}/Q_{24}$."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from model3_o2_dynamic.models.response_adapter import ResponseAdapterBank
from model3_o2_dynamic.models.response_predictor import TokenResponsePredictor

from .cache import current_hidden, load_response_cache, split_indices, standardized_targets
from .contracts import LAYERS, Stage1ContractError, Stage1TrainConfig
from .export import save_adapter_export, sha256_file


class ResponseWarmupModel(nn.Module):
    """Training-only container.  It is never used by the deployed policy."""

    def __init__(self, *, predictor_width: int) -> None:
        super().__init__()
        self.adapters = ResponseAdapterBank()
        self.predictors = nn.ModuleDict(
            {
                str(layer): TokenResponsePredictor(width=predictor_width)
                for layer in LAYERS
            }
        )

    def predictor_parameter_counts(self) -> dict[int, int]:
        return {
            layer: self.predictors[str(layer)].parameter_count()
            for layer in LAYERS
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _rng_payload(generator: torch.Generator) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "index_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng(payload: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    generator.set_state(payload["index_generator"])
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _batch_indices(
    indices: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    positions = torch.randint(0, len(indices), (batch_size,), generator=generator)
    return indices[positions]


def response_loss(
    model: ResponseWarmupModel,
    *,
    current_hidden: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    teacher_timestep: int,
    beta_anchor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Implement the frozen anchor and local-response objectives exactly.

    ``current_hidden`` is deliberately detached here.  The only parameters in
    ``model`` are the three A/Q pairs, so a successful backward is also an
    ownership assertion that no Wan/O2 parameter can receive Stage-1 gradient.
    """

    if tuple(current_hidden.shape[1:]) != (3, 392, 1536):
        raise ValueError(f"current_hidden must be [B,3,392,1536], got {tuple(current_hidden.shape)}")
    if tuple(actions.shape[1:]) != (4, 8, 7):
        raise ValueError(f"actions must be [B,4,8,7], got {tuple(actions.shape)}")
    if tuple(targets.shape[1:]) != (3, 3, 4, 256):
        raise ValueError(f"targets must be [B,3,3,4,256], got {tuple(targets.shape)}")
    batch = current_hidden.shape[0]
    timestep = torch.full(
        (batch,), float(teacher_timestep), device=current_hidden.device, dtype=torch.float32
    )
    total = current_hidden.new_zeros((), dtype=torch.float32)
    metrics: dict[str, torch.Tensor] = {}
    for position, layer in enumerate(LAYERS):
        # Explicit stop-gradient at h_l.  Stage 1 deliberately gives Q only
        # the deployment adapter's residual, never the identity skip B_l=h_l+r.
        # Otherwise a temporary Q could solve the response task directly from
        # h_l and leave the exported adapter effectively untrained.
        h = current_hidden[:, position].detach()
        residual = model.adapters.residual(layer, h)
        predictor = model.predictors[str(layer)]
        predicted = torch.stack(
            [predictor(residual, actions[:, branch], timestep) for branch in range(3)], dim=1
        )
        target = targets[:, position]
        anchor = (predicted - target).float().square().mean()
        local_prediction = predicted[:, 1:] - predicted[:, :1]
        local_target = target[:, 1:] - target[:, :1]
        local = (local_prediction - local_target).float().square().mean()
        layer_total = local + float(beta_anchor) * anchor
        total = total + layer_total
        metrics[f"loss_anchor_l{layer}"] = anchor.detach()
        metrics[f"loss_local_l{layer}"] = local.detach()
        residual_norm = residual.float().norm(dim=-1).mean()
        carrier_norm = h.float().norm(dim=-1).mean()
        metrics[f"residual_norm_l{layer}"] = residual_norm.detach()
        metrics[f"carrier_norm_l{layer}"] = carrier_norm.detach()
        metrics[f"residual_ratio_l{layer}"] = (residual_norm / carrier_norm.clamp_min(1.0e-12)).detach()
    metrics["loss_response"] = total.detach()
    return total, metrics


@torch.no_grad()
def validation_metrics(
    model: ResponseWarmupModel,
    cache: dict[str, Any],
    indices: torch.Tensor,
    *,
    device: torch.device,
    config: Stage1TrainConfig,
    batch_size: int = 64,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        current = current_hidden(cache, selected).to(device=device, dtype=torch.float32)
        actions = cache["actions"][selected].to(device=device, dtype=torch.float32)
        targets = standardized_targets(cache, selected, device=device)
        _, metrics = response_loss(
            model,
            current_hidden=current,
            actions=actions,
            targets=targets,
            teacher_timestep=config.teacher_timestep,
            beta_anchor=config.beta_anchor,
        )
        local_count = len(selected)
        count += local_count
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * local_count
    if count == 0:
        raise Stage1ContractError("validation split is empty")
    return {key: value / count for key, value in totals.items()}


def _checkpoint_payload(
    *,
    model: ResponseWarmupModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Stage1TrainConfig,
    cache_identity: dict[str, Any],
    index_generator: torch.Generator,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_response_audit_checkpoint",
        "method_id": "model3_o2_dynamic_response_prewarm_v1",
        "contains_predictors": True,
        "stage1_predictor_input": "adapter_residual_only",
        "step": int(step),
        "stage1_config": config.as_dict(),
        "cache_identity": cache_identity,
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _rng_payload(index_generator),
        "history": list(history),
    }


def _assert_predictor_budget(model: ResponseWarmupModel, maximum: int) -> dict[int, int]:
    counts = model.predictor_parameter_counts()
    too_large = {layer: count for layer, count in counts.items() if count > maximum}
    if too_large:
        raise Stage1ContractError(
            f"Stage 1 predictor parameter cap exceeded: {too_large}, cap={maximum}"
        )
    return counts


def _assert_first_backward_gradients(model: ResponseWarmupModel) -> None:
    missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    if missing:
        raise RuntimeError(
            "Stage 1 response loss failed to reach A/Q parameters: " + ", ".join(missing)
        )


def train_stage1(
    *,
    cache_path: str | Path,
    output_dir: str | Path,
    config: Stage1TrainConfig,
    device: str | torch.device = "cuda:0",
    resume: str | Path | None = None,
    allow_nonformal_smoke: bool = False,
) -> dict[str, Any]:
    """Run the fixed 5K optimizer-step warmup and emit adapter-only export."""

    cache, cache_identity = load_response_cache(cache_path, require_trainable_splits_only=True)
    split_counts = cache_identity["split_counts"]
    if not allow_nonformal_smoke and split_counts != {"train": 4000, "validation": 500, "test": 0}:
        raise Stage1ContractError(
            "formal Stage 1 requires exactly 4000 train / 500 validation source states; "
            f"got {split_counts}"
        )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {target_device}, but CUDA is unavailable")
    _seed_everything(config.seed)
    train_indices = split_indices(cache, "train")
    validation_indices = split_indices(cache, "validation")
    output = Path(output_dir).expanduser().resolve()
    checkpoint_dir = output / "audit_checkpoints"
    export_dir = output / "adapter_export"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    model = ResponseWarmupModel(predictor_width=config.predictor_width).to(target_device)
    parameter_counts = _assert_predictor_budget(model, config.max_predictor_parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    index_generator = torch.Generator(device="cpu").manual_seed(config.seed + 17)
    step = 0
    history: list[dict[str, Any]] = []
    if resume is not None:
        resume_path = Path(resume).expanduser().resolve()
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_payload.get("artifact_kind") != "stage1_response_audit_checkpoint":
            raise Stage1ContractError("Stage 1 resume must be an audit checkpoint")
        if resume_payload.get("stage1_predictor_input") != "adapter_residual_only":
            raise Stage1ContractError(
                "Stage 1 resume does not prove that Q received only the adapter residual"
            )
        if resume_payload.get("stage1_config") != config.as_dict():
            raise Stage1ContractError("Stage 1 resume config does not match the frozen config")
        if resume_payload.get("cache_identity", {}).get("sha256") != cache_identity["sha256"]:
            raise Stage1ContractError("Stage 1 resume cache identity does not match")
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        _restore_rng(resume_payload["rng_state"], index_generator)
        step = int(resume_payload["step"])
        history = list(resume_payload.get("history", []))
        if step < 0 or step >= config.max_steps:
            raise Stage1ContractError(f"invalid Stage 1 resume step: {step}")
    started = time.time()
    first_backward_checked = False
    while step < config.max_steps:
        model.train()
        selected = _batch_indices(
            train_indices,
            batch_size=config.batch_size,
            generator=index_generator,
        )
        current = current_hidden(cache, selected).to(device=target_device, dtype=torch.float32)
        actions = cache["actions"][selected].to(device=target_device, dtype=torch.float32)
        targets = standardized_targets(cache, selected, device=target_device)
        loss, metrics = response_loss(
            model,
            current_hidden=current,
            actions=actions,
            targets=targets,
            teacher_timestep=config.teacher_timestep,
            beta_anchor=config.beta_anchor,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not first_backward_checked:
            _assert_first_backward_gradients(model)
            first_backward_checked = True
        optimizer.step()
        step += 1
        if step % config.validation_every == 0 or step == config.max_steps:
            val = validation_metrics(
                model,
                cache,
                validation_indices,
                device=target_device,
                config=config,
                batch_size=config.batch_size,
            )
            event = {
                "step": step,
                "train_loss_response": float(metrics["loss_response"]),
                **{f"train_{key}": float(value) for key, value in metrics.items() if key != "loss_response"},
                **{f"validation_{key}": float(value) for key, value in val.items()},
            }
            history.append(event)
            _json_write(output / "metrics.json", {"history": history})
        if step in config.checkpoint_steps:
            checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                    cache_identity=cache_identity,
                    index_generator=index_generator,
                    history=history,
                ),
                checkpoint_path,
            )
    final_checkpoint = checkpoint_dir / "step_005000.pt"
    if not final_checkpoint.is_file():
        raise RuntimeError("Stage 1 final audit checkpoint was not produced")
    normalization_identity = {
        "target_space": cache["target_space"],
        "normalization_fit_split": cache["normalization_fit_split"],
        "normalization_mean_shape": list(cache["normalization_mean"].shape),
        "normalization_std_shape": list(cache["normalization_std"].shape),
        "cache_sha256": cache_identity["sha256"],
    }
    export_path = export_dir / "stage1_adapter_step_005000.pt"
    export_payload = save_adapter_export(
        export_path,
        model.adapters,
        source_identity=cache["source_identity"],
        normalization_identity=normalization_identity,
    )
    result = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "status": "complete",
        "stage": "response_warmup",
        "optimizer_steps": step,
        "contains_predictors_in_stage2_export": False,
        "stage1_predictor_input": "adapter_residual_only",
        "predictor_parameter_counts": parameter_counts,
        "cache_identity": cache_identity,
        "final_audit_checkpoint": str(final_checkpoint),
        "final_audit_checkpoint_sha256": sha256_file(final_checkpoint),
        "adapter_export": str(export_path),
        "adapter_export_sha256": export_payload["file_sha256"],
        "adapter_state_sha256": export_payload["response_adapter_state_sha256"],
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    _json_write(output / "stage1_result.json", result)
    return result


def _load_cli_config(path: Path) -> Stage1TrainConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    section = raw.get("stage1", raw)
    if not isinstance(section, dict):
        raise Stage1ContractError("Stage 1 config section must be an object")
    return Stage1TrainConfig.from_mapping(section)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-nonformal-smoke", action="store_true")
    args = parser.parse_args()
    result = train_stage1(
        cache_path=args.cache,
        output_dir=args.output_dir,
        config=_load_cli_config(args.config),
        device=args.device,
        resume=args.resume,
        allow_nonformal_smoke=bool(args.allow_nonformal_smoke),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
