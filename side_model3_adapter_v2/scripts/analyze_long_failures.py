"""Create LIBERO Long failure storyboards and structured analysis tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from side_model3_adapter_v2.failure_analysis import (
    attach_and_validate_analyses,
    build_vlm_request,
    create_storyboard,
    load_stage_rules,
    read_jsonl,
    run_vlm_command,
    scan_failure_videos,
    write_analysis_outputs,
    write_jsonl,
)


DEFAULT_STAGE_RULES = (
    Path(__file__).resolve().parents[1] / "configs" / "libero_long_stage_rules.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-rules", type=Path, default=DEFAULT_STAGE_RULES)
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help="Select task:trial or checkpoint_step:task:trial; repeat as needed.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--analysis-jsonl", type=Path)
    parser.add_argument(
        "--vlm-command",
        help="Persistent command that reads request JSONL on stdin and writes one structured JSON object per line.",
    )
    parser.add_argument("--analyzer-name", default="external_vlm")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    storyboard_dir = output_dir / "storyboards"
    stage_rules = load_stage_rules(args.stage_rules.resolve())
    episodes = scan_failure_videos(
        [path.resolve() for path in args.video_dir],
        selectors=args.episode,
        limit=args.limit,
    )

    manifest = []
    for episode in episodes:
        if episode.task not in stage_rules:
            raise ValueError(f"No stage rule registered for task {episode.task}")
        storyboard_path = storyboard_dir / f"{episode.artifact_stem}_storyboard.jpg"
        metadata = create_storyboard(episode, storyboard_path)
        manifest.append(
            {
                "checkpoint_step": episode.checkpoint_step,
                "task": episode.task,
                "trial": episode.trial,
                "video_path": str(episode.video_path),
                "storyboard_path": str(storyboard_path.resolve()),
                **metadata,
            }
        )

    manifest_path = output_dir / "failure_manifest.jsonl"
    write_jsonl(manifest, manifest_path)
    requests = [
        build_vlm_request(record, stage_rules[int(record["task"])])
        for record in manifest
    ]
    write_jsonl(requests, output_dir / "vlm_requests.jsonl")
    (output_dir / "failure_response_schema.json").write_text(
        json.dumps(
            {
                "taxonomy": list(
                    requests[0]["response_schema"]["properties"]["primary_failure"]["enum"]
                ),
                "per_task_response_schemas": {
                    str(request["task"]): request["response_schema"] for request in requests
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    raw_analyses = None
    analyzer_name = args.analyzer_name
    if args.analysis_jsonl is not None and args.vlm_command is not None:
        raise ValueError("Choose either --analysis-jsonl or --vlm-command")
    if args.analysis_jsonl is not None:
        raw_analyses = read_jsonl(args.analysis_jsonl.resolve())
    elif args.vlm_command is not None:
        raw_analyses = run_vlm_command(args.vlm_command, requests)

    if raw_analyses is not None:
        write_jsonl(raw_analyses, output_dir / "vlm_outputs.jsonl")
        rows = attach_and_validate_analyses(
            manifest,
            raw_analyses,
            stage_rules,
            analyzer=analyzer_name,
        )
        rows = write_analysis_outputs(
            rows,
            output_dir,
            confidence_threshold=args.confidence_threshold,
        )
        write_jsonl(rows, output_dir / "failure_analysis.jsonl")

    print(
        json.dumps(
            {
                "prepared_episodes": len(manifest),
                "manifest": str(manifest_path),
                "analysis_rows": 0 if raw_analyses is None else len(raw_analyses),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
