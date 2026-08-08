"""Measure whether the trained transition predictor depends on its action input."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


HORIZONS = (4, 8)
CONDITIONS = ("gt", "shuffle", "zero")


def evenly_spaced_indices(dataset_size: int, sample_count: int) -> list[int]:
    """Return deterministic midpoint indices spanning the complete dataset."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if sample_count <= 0 or sample_count > dataset_size:
        raise ValueError("sample_count must be in [1, dataset_size]")
    indices = [
        ((2 * position + 1) * dataset_size) // (2 * sample_count)
        for position in range(sample_count)
    ]
    if len(set(indices)) != sample_count:
        raise ValueError("evenly spaced selection produced duplicate indices")
    return indices


class IndexedSelection(Dataset):
    """Expose source indices together with samples from a fixed selection."""

    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        source_index = self.indices[position]
        return source_index, self.dataset[source_index]


@dataclass
class StateRecord:
    dataset_index: int
    action: torch.Tensor
    control_state: torch.Tensor
    target_states: dict[int, torch.Tensor]
    valid: dict[int, bool]


def normalized_prediction_change(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    """Per-sample RMS change after the same normalization used by state loss."""

    reference_norm = F.layer_norm(reference.float(), (reference.shape[-1],))
    candidate_norm = F.layer_norm(candidate.float(), (candidate.shape[-1],))
    return (candidate_norm - reference_norm).square().mean(dim=(1, 2)).sqrt()


def summarize_condition_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty horizon")
    gt = torch.tensor([row["loss_gt"] for row in rows], dtype=torch.float64)
    result: dict[str, Any] = {
        "valid_samples": len(rows),
        "conditions": {
            "gt": {
                "mean_loss": float(gt.mean().item()),
                "median_loss": float(gt.median().item()),
            }
        },
    }
    denominator = gt.clamp_min(1.0e-12)
    for condition in ("shuffle", "zero"):
        control = torch.tensor(
            [row[f"loss_{condition}"] for row in rows], dtype=torch.float64
        )
        prediction_change = torch.tensor(
            [row[f"prediction_change_{condition}"] for row in rows],
            dtype=torch.float64,
        )
        action_change = torch.tensor(
            [row[f"action_rms_change_{condition}"] for row in rows],
            dtype=torch.float64,
        )
        ratio_of_means = float((control.mean() / gt.mean().clamp_min(1.0e-12)).item())
        result["conditions"][condition] = {
            "mean_loss": float(control.mean().item()),
            "median_loss": float(control.median().item()),
            "loss_ratio_of_means_to_gt": ratio_of_means,
            "relative_mean_loss_increase_over_gt": ratio_of_means - 1.0,
            "mean_per_sample_loss_ratio_to_gt": float(
                (control / denominator).mean().item()
            ),
            "fraction_loss_exceeds_gt": float((control > gt).double().mean().item()),
            "mean_normalized_prediction_change_from_gt": float(
                prediction_change.mean().item()
            ),
            "mean_action_rms_change_from_gt": float(action_change.mean().item()),
        }
    return result


def classify_action_dependence(horizon_summaries: dict[str, dict[str, Any]]) -> str:
    strong = True
    weak = True
    for horizon in HORIZONS:
        summary = horizon_summaries[str(horizon)]
        for condition in ("shuffle", "zero"):
            metrics = summary["conditions"][condition]
            ratio = float(metrics["loss_ratio_of_means_to_gt"])
            fraction = float(metrics["fraction_loss_exceeds_gt"])
            strong = strong and ratio >= 1.25 and fraction >= 0.75
            weak = weak and ratio < 1.10
    if strong:
        return "sufficient_action_dependence_for_v2_bridge"
    if weak:
        return "weak_action_dependence_do_not_launch_v2_bridge"
    return "mixed_action_dependence_requires_judgment"


@torch.inference_mode()
def collect_state_records(model, loader: DataLoader) -> list[StateRecord]:
    records: list[StateRecord] = []
    for batch_number, (dataset_indices, sample) in enumerate(loader, start=1):
        inputs = model.build_inputs(sample)
        current_wan_states = model._extract_adapted_wan_states(
            observation_latents=inputs["current_latents"],
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            use_target_adapters=False,
        )
        control_state, _ = model.online_predictive_encoder(
            current_wan_states,
            inputs["proprio"][0],
        )
        target_states: dict[int, torch.Tensor] = {}
        for horizon in HORIZONS:
            future_wan_states = model._extract_adapted_wan_states(
                observation_latents=inputs["future_latents"][horizon],
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                use_target_adapters=True,
            )
            target_state, _ = model.target_predictive_encoder(
                future_wan_states,
                inputs["proprio"][horizon],
            )
            target_states[horizon] = target_state

        batch_size = int(inputs["action"].shape[0])
        for row in range(batch_size):
            records.append(
                StateRecord(
                    dataset_index=int(dataset_indices[row].item()),
                    action=inputs["action"][row].detach().float().cpu(),
                    control_state=control_state[row].detach().cpu(),
                    target_states={
                        horizon: target_states[horizon][row].detach().float().cpu()
                        for horizon in HORIZONS
                    },
                    valid={
                        horizon: bool(inputs["future_valid"][horizon][row].item())
                        for horizon in HORIZONS
                    },
                )
            )
        print(
            f"[extract] batch={batch_number}/{len(loader)} records={len(records)}",
            flush=True,
        )
    return records


@torch.inference_mode()
def evaluate_horizon(
    model,
    records: list[StateRecord],
    *,
    horizon: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    valid_records = [record for record in records if record.valid[horizon]]
    if len(valid_records) < 2:
        raise ValueError(f"horizon {horizon} has fewer than two valid samples")

    actions = torch.stack([record.action for record in valid_records])
    controls = torch.stack([record.control_state for record in valid_records])
    targets = torch.stack([record.target_states[horizon] for record in valid_records])
    dataset_indices = torch.tensor(
        [record.dataset_index for record in valid_records], dtype=torch.long
    )
    shuffled_actions = torch.roll(actions, shifts=1, dims=0)
    shuffled_indices = torch.roll(dataset_indices, shifts=1, dims=0)
    zero_actions = torch.zeros_like(actions)

    output_rows: list[dict[str, Any]] = []
    device = model.device
    dtype = model.torch_dtype
    for start in range(0, len(valid_records), batch_size):
        stop = min(start + batch_size, len(valid_records))
        control_batch = controls[start:stop].to(device=device, dtype=dtype)
        target_batch = targets[start:stop].to(device=device, dtype=torch.float32)
        action_batches = {
            "gt": actions[start:stop].to(device=device, dtype=dtype),
            "shuffle": shuffled_actions[start:stop].to(device=device, dtype=dtype),
            "zero": zero_actions[start:stop].to(device=device, dtype=dtype),
        }
        predictions: dict[str, torch.Tensor] = {}
        losses: dict[str, torch.Tensor] = {}
        for condition in CONDITIONS:
            action_tokens = model.action_chunk_encoder(
                action_batches[condition], horizon=horizon
            )
            predictions[condition] = model.transition_predictor(
                control_batch, action_tokens
            )
            losses[condition] = model._future_state_loss_per_sample(
                predictions[condition], target_batch
            )

        shuffle_prediction_change = normalized_prediction_change(
            predictions["gt"], predictions["shuffle"]
        )
        zero_prediction_change = normalized_prediction_change(
            predictions["gt"], predictions["zero"]
        )
        shuffle_action_change = (
            action_batches["gt"][:, :horizon] - action_batches["shuffle"][:, :horizon]
        ).float().square().mean(dim=(1, 2)).sqrt()
        zero_action_change = (
            action_batches["gt"][:, :horizon] - action_batches["zero"][:, :horizon]
        ).float().square().mean(dim=(1, 2)).sqrt()

        for offset in range(stop - start):
            absolute = start + offset
            output_rows.append(
                {
                    "dataset_index": int(dataset_indices[absolute].item()),
                    "shuffle_donor_dataset_index": int(
                        shuffled_indices[absolute].item()
                    ),
                    "horizon": horizon,
                    "loss_gt": float(losses["gt"][offset].item()),
                    "loss_shuffle": float(losses["shuffle"][offset].item()),
                    "loss_zero": float(losses["zero"][offset].item()),
                    "prediction_change_shuffle": float(
                        shuffle_prediction_change[offset].item()
                    ),
                    "prediction_change_zero": float(
                        zero_prediction_change[offset].item()
                    ),
                    "action_rms_change_shuffle": float(
                        shuffle_action_change[offset].item()
                    ),
                    "action_rms_change_zero": float(
                        zero_action_change[offset].item()
                    ),
                }
            )
        print(
            f"[transition] horizon={horizon} samples={stop}/{len(valid_records)}",
            flush=True,
        )
    return output_rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_evidence(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    dataset_size: int,
    selected_indices: list[int],
    checkpoint_payload: dict[str, Any],
    all_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    (output_dir / "commands.txt").write_text(command + "\n", encoding="utf-8")
    gpu_name = "none"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.device(args.device))
    (output_dir / "environment.md").write_text(
        "\n".join(
            [
                "# Environment",
                "",
                f"- generated_at: `{datetime.now().astimezone().isoformat()}`",
                f"- python: `{platform.python_version()}`",
                f"- torch: `{torch.__version__}`",
                f"- device: `{args.device}`",
                f"- gpu: `{gpu_name}`",
                "- optimizer/backward: `disabled`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "track_id": "side_model3_adapter_v2",
        "diagnostic_id": "side_model3_adapter_v2_action_dependence_gate_v1",
        "evidence_label": "offline_training_distribution_diagnostic",
        "training_config": str(Path(args.training_config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint_payload["step"]),
        "method_id": checkpoint_payload["method_id"],
        "model_class": checkpoint_payload["model_class"],
        "dataset_size": dataset_size,
        "selection": {
            "strategy": "deterministic_evenly_spaced_midpoints",
            "requested_samples": args.num_samples,
            "first_index": selected_indices[0],
            "last_index": selected_indices[-1],
        },
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "horizons": list(HORIZONS),
        "conditions": list(CONDITIONS),
        "status": "complete",
    }
    write_json(output_dir / "run_manifest.json", manifest)
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_dir / "summary.json", summary)

    report_lines = [
        "# Side-Model3-Adapter-v2 Action-Dependence Gate",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
        "| Horizon | Condition | Mean loss | Ratio to GT | Fraction worse than GT | Prediction change |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        horizon_summary = summary["horizons"][str(horizon)]
        gt_metrics = horizon_summary["conditions"]["gt"]
        report_lines.append(
            f"| {horizon} | GT | {gt_metrics['mean_loss']:.6f} | 1.000 | - | 0.000000 |"
        )
        for condition in ("shuffle", "zero"):
            metrics = horizon_summary["conditions"][condition]
            report_lines.append(
                f"| {horizon} | {condition} | {metrics['mean_loss']:.6f} | "
                f"{metrics['loss_ratio_of_means_to_gt']:.3f} | "
                f"{metrics['fraction_loss_exceeds_gt']:.3f} | "
                f"{metrics['mean_normalized_prediction_change_from_gt']:.6f} |"
            )
    report_lines.extend(
        [
            "",
            "This is a training-distribution action-sensitivity diagnostic. It does not modify",
            "the checkpoint and does not establish closed-loop benefit or recovery behavior.",
            "",
        ]
    )
    (output_dir / "run_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from model3.third_party.light_wam.src.lightwam.runtime import (
        _mixed_precision_to_model_dtype,
        _normalize_mixed_precision,
        build_datasets,
    )
    from model3.third_party.light_wam.src.lightwam.utils import misc

    training_config = Path(args.training_config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    cfg = OmegaConf.load(training_config)
    misc.register_work_dir(str(training_config.parent))
    model_dtype = _mixed_precision_to_model_dtype(
        _normalize_mixed_precision(cfg.mixed_precision)
    )
    model = instantiate(
        cfg.model,
        model_dtype=model_dtype,
        device=args.device,
    )
    checkpoint_payload = model.load_checkpoint(checkpoint)
    if int(checkpoint_payload.get("step", -1)) != args.checkpoint_step:
        raise ValueError("checkpoint step does not match --checkpoint-step")
    model.eval()

    train_dataset, _ = build_datasets(cfg.data)
    selected_indices = evenly_spaced_indices(len(train_dataset), args.num_samples)
    loader = DataLoader(
        IndexedSelection(train_dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    records = collect_state_records(model, loader)
    all_rows: list[dict[str, Any]] = []
    horizon_summaries: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        rows = evaluate_horizon(
            model,
            records,
            horizon=horizon,
            batch_size=args.batch_size,
        )
        all_rows.extend(rows)
        horizon_summaries[str(horizon)] = summarize_condition_rows(rows)

    summary = {
        "schema_version": 1,
        "decision": classify_action_dependence(horizon_summaries),
        "thresholds": {
            "sufficient_loss_ratio": 1.25,
            "sufficient_fraction_control_worse": 0.75,
            "weak_loss_ratio": 1.10,
        },
        "horizons": horizon_summaries,
    }
    write_evidence(
        output_dir=output_dir,
        args=args,
        dataset_size=len(train_dataset),
        selected_indices=selected_indices,
        checkpoint_payload=checkpoint_payload,
        all_rows=all_rows,
        summary=summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
