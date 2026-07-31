#!/usr/bin/env python3
"""Combine two single-model latency reports into one comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    baseline = load(args.baseline)
    treatment = load(args.treatment)
    baseline_protocol = baseline["protocol"]
    treatment_protocol = treatment["protocol"]
    controlled_keys = (
        "gpu_name",
        "dtype",
        "input_shape",
        "input_count",
        "input_seed",
        "prompt_preencoded",
        "warmup_calls",
        "measured_calls",
        "cuda_synchronize",
    )
    mismatches = {
        key: [baseline_protocol.get(key), treatment_protocol.get(key)]
        for key in controlled_keys
        if baseline_protocol.get(key) != treatment_protocol.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Controlled latency protocol mismatch: {mismatches}")

    baseline_total = baseline["summary"]["timings"]["total"]
    treatment_total = treatment["summary"]["timings"]["total"]
    baseline_amortized = baseline["summary"]["amortized_policy_latency_per_executed_step_ms"]
    treatment_amortized = treatment["summary"]["amortized_policy_latency_per_executed_step_ms"]
    comparison = {
        "schema_version": 1,
        "baseline": baseline["model_label"],
        "treatment": treatment["model_label"],
        "controlled_protocol": {key: baseline_protocol[key] for key in controlled_keys},
        "deployment_protocol": {
            baseline["model_label"]: {
                "action_horizon": baseline_protocol["action_horizon"],
                "replan_steps": baseline_protocol["replan_steps"],
                "num_inference_steps": baseline_protocol["num_inference_steps"],
            },
            treatment["model_label"]: {
                "action_horizon": treatment_protocol["action_horizon"],
                "replan_steps": treatment_protocol["replan_steps"],
                "num_inference_steps": treatment_protocol["num_inference_steps"],
            },
        },
        "results": {
            baseline["model_label"]: {
                "total_mean_ms": baseline_total["mean_ms"],
                "total_median_ms": baseline_total["median_ms"],
                "total_p95_ms": baseline_total["p95_ms"],
                "amortized_ms_per_executed_step": baseline_amortized,
            },
            treatment["model_label"]: {
                "total_mean_ms": treatment_total["mean_ms"],
                "total_median_ms": treatment_total["median_ms"],
                "total_p95_ms": treatment_total["p95_ms"],
                "amortized_ms_per_executed_step": treatment_amortized,
            },
        },
        "ratios_treatment_over_baseline": {
            "total_mean": treatment_total["mean_ms"] / baseline_total["mean_ms"],
            "total_p95": treatment_total["p95_ms"] / baseline_total["p95_ms"],
            "amortized_per_executed_step": treatment_amortized / baseline_amortized,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    labels = [baseline["model_label"], treatment["model_label"]]
    reports = [baseline, treatment]
    lines = [
        "# Light-WAM vs Model3 Long Inference Latency",
        "",
        "Both models ran sequentially on the same GPU with pre-encoded prompt context, "
        "10 warmup calls, 50 measured calls, and CUDA synchronization.",
        "",
        "| Model | Mean plan call (ms) | Median (ms) | P95 (ms) | Replan steps | Amortized (ms/control step) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, report in zip(labels, reports):
        total = report["summary"]["timings"]["total"]
        amortized = report["summary"]["amortized_policy_latency_per_executed_step_ms"]
        lines.append(
            f"| {label} | {total['mean_ms']:.3f} | {total['median_ms']:.3f} | "
            f"{total['p95_ms']:.3f} | {report['protocol']['replan_steps']} | {amortized:.3f} |"
        )
    ratios = comparison["ratios_treatment_over_baseline"]
    lines.extend(
        [
            "",
            f"Treatment/baseline mean planning-call ratio: `{ratios['total_mean']:.3f}x`.",
            f"Treatment/baseline amortized control-step ratio: "
            f"`{ratios['amortized_per_executed_step']:.3f}x`.",
            "",
            "This is a deployed-policy latency comparison, not a single-variable action-head ablation.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
