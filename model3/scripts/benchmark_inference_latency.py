#!/usr/bin/env python3
"""Benchmark one WAM checkpoint's deployed action-planning latency."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


DEFAULT_TASK = "put both the alphabet soup and the tomato sauce in the basket"
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, required=True)
    parser.add_argument("--replan-steps", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--input-count", type=int, default=8)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--input-seed", type=int, default=20260726)
    parser.add_argument("--action-seed", type=int, default=42)
    parser.add_argument("--task-description", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
    }


def model_dtype(mixed_precision: str) -> torch.dtype:
    precision = str(mixed_precision).strip().lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "no":
        return torch.float32
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.input_count < 1:
        raise ValueError("warmup, iterations, and input-count must be positive")
    if args.replan_steps > args.action_horizon:
        raise ValueError("replan-steps cannot exceed action-horizon")
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be multiples of 16")
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("config and checkpoint must exist")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The reference latency protocol requires CUDA")

    cfg = OmegaConf.load(args.config)
    cfg.model.load_text_encoder = True
    dtype = model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=dtype, device=str(device))
    model.load_checkpoint(str(args.checkpoint))
    model = model.to(device).eval()
    if not callable(getattr(model, "benchmark_infer_action", None)):
        raise TypeError(f"{type(model).__name__} does not expose benchmark_infer_action")

    prompt = DEFAULT_PROMPT.format(task=args.task_description)
    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is None:
        raise RuntimeError("Latency protocol requires one-time prompt pre-encoding")
    text_encoder.to(device)
    with torch.no_grad():
        context, context_mask = model.encode_prompt(prompt)
    context = context.detach().cpu()
    context_mask = context_mask.detach().cpu()
    text_encoder.to("cpu")
    sync(device)
    torch.cuda.empty_cache()

    generator = torch.Generator(device="cpu").manual_seed(args.input_seed)
    images = torch.rand(
        (args.input_count, 3, args.height, args.width),
        generator=generator,
        dtype=torch.float32,
    ).mul_(2.0).sub_(1.0).to(device=device, dtype=dtype)
    proprio_dim = int(getattr(model, "proprio_dim", 0) or 0)
    proprio = torch.randn(
        (args.input_count, proprio_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)

    call_kwargs: dict[str, Any] = {
        "prompt": None,
        "action_horizon": args.action_horizon,
        "context": context,
        "context_mask": context_mask,
        "negative_prompt": "",
        "text_cfg_scale": 1.0,
        "num_inference_steps": args.num_inference_steps,
        "sigma_shift": None,
        "rand_device": "cpu",
        "tiled": False,
    }

    for call_idx in range(args.warmup):
        sample_idx = call_idx % args.input_count
        model.benchmark_infer_action(
            input_image=images[sample_idx : sample_idx + 1],
            proprio=proprio[sample_idx : sample_idx + 1],
            seed=args.action_seed + call_idx,
            **call_kwargs,
        )
    sync(device)

    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    raw_calls: list[dict[str, Any]] = []
    for call_idx in range(args.iterations):
        sample_idx = call_idx % args.input_count
        result = model.benchmark_infer_action(
            input_image=images[sample_idx : sample_idx + 1],
            proprio=proprio[sample_idx : sample_idx + 1],
            seed=args.action_seed + args.warmup + call_idx,
            **call_kwargs,
        )
        action = result["action"]
        if tuple(action.shape) != (args.action_horizon, 7):
            raise RuntimeError(f"Unexpected action shape: {tuple(action.shape)}")
        raw_calls.append(
            {
                "index": call_idx,
                "input_index": sample_idx,
                "timings_ms": {
                    key: float(value) * 1000.0
                    for key, value in result["timings_s"].items()
                },
            }
        )
    sync(device)

    timing_keys = sorted(raw_calls[0]["timings_ms"])
    if "total" not in timing_keys:
        raise RuntimeError("benchmark_infer_action did not return a total timing")
    timing_summary = {
        key: summarize([call["timings_ms"][key] for call in raw_calls])
        for key in timing_keys
    }
    total_mean_ms = timing_summary["total"]["mean_ms"]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_label": args.model_label,
        "model_class": type(model).__name__,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "config": str(args.config.resolve()),
        "protocol": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "dtype": str(dtype),
            "input_shape": [1, 3, args.height, args.width],
            "input_count": args.input_count,
            "input_seed": args.input_seed,
            "prompt_preencoded": True,
            "warmup_calls": args.warmup,
            "measured_calls": args.iterations,
            "cuda_synchronize": True,
            "action_horizon": args.action_horizon,
            "replan_steps": args.replan_steps,
            "num_inference_steps": args.num_inference_steps,
        },
        "summary": {
            "timings": timing_summary,
            "amortized_policy_latency_per_executed_step_ms": total_mean_ms
            / args.replan_steps,
            "policy_limited_control_hz": 1000.0
            / (total_mean_ms / args.replan_steps),
        },
        "memory_bytes": {
            "baseline_allocated": baseline_allocated,
            "baseline_reserved": baseline_reserved,
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
        },
        "raw_calls": raw_calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.model_label}: total_mean={total_mean_ms:.3f} ms "
        f"total_p95={timing_summary['total']['p95_ms']:.3f} ms "
        f"amortized={payload['summary']['amortized_policy_latency_per_executed_step_ms']:.3f} ms/step",
        flush=True,
    )
    print(f"output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
