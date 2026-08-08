"""Analyze failed LIBERO Long rollouts from simulator-ground-truth traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from side_model3_adapter_v2.state_failure_analysis import (
    analyze_state_trace,
    build_state_manifest,
    load_rule_thresholds,
    load_state_task_rules,
    load_state_trace,
    scan_state_traces,
    write_jsonl,
    write_state_analysis_outputs,
)


DEFAULT_STAGE_RULES = (
    Path(__file__).resolve().parents[1] / "configs" / "libero_long_stage_rules.json"
)
DEFAULT_THRESHOLDS = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "libero_long_state_rule_thresholds.json"
)


def _parse_episode_selector(value: str) -> tuple[int, int]:
    fields = value.split(":")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError(
            f"invalid episode selector {value!r}; expected task:trial"
        )
    try:
        return int(fields[0]), int(fields[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid episode selector {value!r}; expected integer task:trial"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-dir",
        type=Path,
        action="append",
        required=True,
        help="LIBERO suite simulator_states directory; repeat only for disjoint shards.",
    )
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-rules", type=Path, default=DEFAULT_STAGE_RULES)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--episode",
        type=_parse_episode_selector,
        action="append",
        default=[],
        metavar="TASK:TRIAL",
        help="Select an episode; repeat as needed.",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    task_rules = load_state_task_rules(args.stage_rules.resolve())
    thresholds = load_rule_thresholds(args.thresholds.resolve())
    selectors = set(args.episode)
    traces: list[tuple[Path, dict]] = []
    seen_episode_ids: set[tuple[int, int]] = set()
    for state_dir in args.state_dir:
        for state_path in scan_state_traces(state_dir.resolve()):
            payload = load_state_trace(state_path)
            episode_id = (int(payload["task_id"]), int(payload["trial_id"]))
            if selectors and episode_id not in selectors:
                continue
            if episode_id in seen_episode_ids:
                raise ValueError(
                    "duplicate task/trial simulator-state trace across --state-dir inputs: "
                    f"{episode_id[0]}:{episode_id[1]}"
                )
            seen_episode_ids.add(episode_id)
            traces.append((state_path.resolve(), payload))

    traces.sort(key=lambda item: (int(item[1]["task_id"]), int(item[1]["trial_id"])))
    if args.limit is not None:
        traces = traces[: args.limit]
    if not traces:
        raise ValueError("no simulator-state traces matched the requested scan")

    manifest = []
    rows = []
    skipped_successes = 0
    for state_path, payload in traces:
        if bool(payload.get("success")):
            skipped_successes += 1
            continue
        manifest.append(
            build_state_manifest(
                payload,
                state_path=state_path,
                checkpoint_step=args.checkpoint_step,
            )
        )
        rows.append(
            analyze_state_trace(
                payload,
                state_path=state_path,
                checkpoint_step=args.checkpoint_step,
                task_rules=task_rules,
                thresholds=thresholds,
            )
        )

    output_dir = args.output_dir.resolve()
    write_jsonl(manifest, output_dir / "state_failure_manifest.jsonl")
    write_state_analysis_outputs(rows, output_dir, thresholds=thresholds)
    print(
        json.dumps(
            {
                "scanned_episodes": len(traces),
                "analyzed_episodes": len(rows),
                "skipped_successes": skipped_successes,
                "manifest": str(output_dir / "state_failure_manifest.jsonl"),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
