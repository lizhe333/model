"""Evaluation-only LIBERO Long failure-video analysis utilities."""

from __future__ import annotations

import csv
import json
import math
import re
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from side_model3_adapter_v2.failure_taxonomy import (
    FAILURE_TAXONOMY,
    MANUAL_REVIEW_EVIDENCE_MARKERS,
    UNDETERMINED_STAGE,
)

VIDEO_NAME_RE = re.compile(
    r"--episode=task(?P<task>\d+)_trial(?P<trial>\d+)"
    r"--success=(?P<success>True|False)--task=(?P<description>.+)\.mp4$"
)
CHECKPOINT_RE = re.compile(r"solver10_step_(?P<step>\d+)")


@dataclass(frozen=True)
class FailureEpisode:
    checkpoint_step: int | None
    task: int
    trial: int
    video_path: Path
    task_slug: str

    @property
    def episode_key(self) -> tuple[int | None, int, int]:
        return self.checkpoint_step, self.task, self.trial

    @property
    def artifact_stem(self) -> str:
        step = "unknown" if self.checkpoint_step is None else f"{self.checkpoint_step:06d}"
        return f"step_{step}_task{self.task:02d}_trial{self.trial:02d}"


def parse_failure_video_path(path: Path) -> FailureEpisode:
    match = VIDEO_NAME_RE.search(path.name)
    if match is None:
        raise ValueError(f"Unrecognized rollout video filename: {path.name}")
    if match.group("success") != "False":
        raise ValueError(f"Video is not a failed rollout: {path.name}")
    checkpoint_match = CHECKPOINT_RE.search(str(path))
    checkpoint_step = (
        None if checkpoint_match is None else int(checkpoint_match.group("step"))
    )
    return FailureEpisode(
        checkpoint_step=checkpoint_step,
        task=int(match.group("task")),
        trial=int(match.group("trial")),
        video_path=path.resolve(),
        task_slug=match.group("description"),
    )


def _parse_selector(selector: str) -> tuple[int | None, int, int]:
    fields = selector.split(":")
    if len(fields) == 2:
        return None, int(fields[0]), int(fields[1])
    if len(fields) == 3:
        return int(fields[0]), int(fields[1]), int(fields[2])
    raise ValueError(
        f"Invalid episode selector {selector!r}; expected task:trial or step:task:trial"
    )


def scan_failure_videos(
    video_dirs: Sequence[Path],
    *,
    selectors: Sequence[str] = (),
    limit: int | None = None,
) -> list[FailureEpisode]:
    selected_keys = {_parse_selector(value) for value in selectors}
    episodes: list[FailureEpisode] = []
    for video_dir in video_dirs:
        for path in sorted(video_dir.rglob("*.mp4")):
            if "--success=False--" not in path.name:
                continue
            episode = parse_failure_video_path(path)
            if selected_keys:
                exact_key = episode.episode_key
                step_agnostic_key = (None, episode.task, episode.trial)
                if exact_key not in selected_keys and step_agnostic_key not in selected_keys:
                    continue
            episodes.append(episode)

    episodes.sort(
        key=lambda item: (
            -1 if item.checkpoint_step is None else item.checkpoint_step,
            item.task,
            item.trial,
            str(item.video_path),
        )
    )
    keys = [episode.episode_key for episode in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate checkpoint/task/trial failure videos found")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        episodes = episodes[:limit]
    if not episodes:
        raise ValueError("No failed rollout videos matched the requested scan")
    return episodes


def select_storyboard_frame_indices(
    frame_count: int,
    *,
    interior_count: int = 16,
) -> list[int]:
    if frame_count < interior_count + 2:
        raise ValueError(
            f"Video has {frame_count} frames; need at least {interior_count + 2}"
        )
    interior = np.linspace(1, frame_count - 2, num=interior_count)
    interior_indices = [int(round(value)) for value in interior]
    return [0, *interior_indices, frame_count - 1]


def _read_video_frame(capture: cv2.VideoCapture, frame_index: int) -> Image.Image:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Failed to decode video frame {frame_index}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _make_storyboard_tile(
    frame: Image.Image,
    *,
    frame_index: int,
    timestamp_seconds: float,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    caption_height = 30
    resized = frame.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_width, tile_height + caption_height), "black")
    tile.paste(resized, (0, caption_height))
    draw = ImageDraw.Draw(tile)
    caption = f"frame {frame_index:04d}   t={timestamp_seconds:06.2f}s"
    draw.text((8, 7), caption, fill="white", font=ImageFont.load_default())
    return tile


def create_storyboard(
    episode: FailureEpisode,
    storyboard_path: Path,
    *,
    interior_count: int = 16,
    columns: int = 6,
    tile_width: int = 384,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(episode.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {episode.video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or fps <= 0 or source_width <= 0 or source_height <= 0:
            raise RuntimeError(f"Invalid video metadata: {episode.video_path}")
        frame_indices = select_storyboard_frame_indices(
            frame_count,
            interior_count=interior_count,
        )
        tile_height = max(1, round(tile_width * source_height / source_width))
        tiles = [
            _make_storyboard_tile(
                _read_video_frame(capture, frame_index),
                frame_index=frame_index,
                timestamp_seconds=frame_index / fps,
                tile_width=tile_width,
                tile_height=tile_height,
            )
            for frame_index in frame_indices
        ]
    finally:
        capture.release()

    rows = math.ceil(len(tiles) / columns)
    board = Image.new(
        "RGB",
        (columns * tiles[0].width, rows * tiles[0].height),
        (24, 24, 24),
    )
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile.width
        y = (index // columns) * tile.height
        board.paste(tile, (x, y))
    storyboard_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(storyboard_path, format="JPEG", quality=92, optimize=True)
    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps,
        "frame_indices": frame_indices,
        "timestamps_seconds": [frame_index / fps for frame_index in frame_indices],
        "storyboard_width": board.width,
        "storyboard_height": board.height,
    }


def load_stage_rules(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Stage rules must contain a tasks list")
    rules: dict[int, dict[str, Any]] = {}
    for task in tasks:
        task_id = int(task["task_id"])
        if task_id in rules:
            raise ValueError(f"Duplicate stage rule for task {task_id}")
        stage_ids = [stage["id"] for stage in task["stage_states"]]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError(f"Duplicate stage id for task {task_id}")
        rules[task_id] = task
    if set(rules) != set(range(10)):
        raise ValueError("LIBERO Long stage rules must cover task ids 0 through 9")
    return rules


def build_response_schema(allowed_stages: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "primary_failure",
            "secondary_failure",
            "outcome_awareness_failure",
            "recovery_failure",
            "furthest_stage",
            "confidence",
            "short_evidence",
        ],
        "properties": {
            "primary_failure": {"type": "string", "enum": list(FAILURE_TAXONOMY)},
            "secondary_failure": {
                "anyOf": [
                    {"type": "string", "enum": list(FAILURE_TAXONOMY)},
                    {"type": "null"},
                ]
            },
            "outcome_awareness_failure": {"type": "boolean"},
            "recovery_failure": {"type": "boolean"},
            "furthest_stage": {
                "type": "string",
                "enum": [*allowed_stages, UNDETERMINED_STAGE],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "short_evidence": {"type": "string", "minLength": 1},
        },
    }


def _taxonomy_prompt() -> str:
    return "\n".join(f"- {value}" for value in FAILURE_TAXONOMY)


def build_vlm_request(
    manifest_record: dict[str, Any],
    task_rule: dict[str, Any],
) -> dict[str, Any]:
    allowed_stages = [stage["id"] for stage in task_rule["stage_states"]]
    system_prompt = (
        "你在分析一个 LIBERO Long 失败 rollout 的时间 storyboard。"
        "只使用可见证据和给定 BDDL 任务规则，不要猜测被遮挡的接触、抓取或 predicate。"
        "必须按时间顺序回溯 primary_failure：它是第一个导致任务最终失败的关键错误，"
        "绝不是视频最后一个失败动作。"
        "例如 task 3 中，若碗没有进抽屉，后续关抽屉失败是次要现象，"
        "primary_failure 应为搬运/放置失败。"
        "recovery_failure=true 仅在明显异常已经发生、之后存在新的观察和纠正机会、"
        "但策略仍未根据异常状态恢复时使用；单纯反复抓不到不能算 recovery failure。"
        "outcome_awareness_failure=true 仅在动作客观失败后，策略却按成功状态进入不相容的下一阶段时使用；"
        "抓空后重新抓取说明知道失败，不能标 true。"
        "secondary_failure 只记录后续独立错误。"
        "若 storyboard 不足以判断，降低 confidence 并使用其他/无法判断或无法判断阶段。"
    )
    episode_prompt = (
        f"任务 {task_rule['task_id']}: {task_rule['language']}\n"
        f"正式目标 predicates: {json.dumps(task_rule['goal_predicates'], ensure_ascii=False)}\n"
        f"允许的阶段状态: {json.dumps(task_rule['stage_states'], ensure_ascii=False)}\n"
        f"阶段顺序规则: {task_rule['ordering']}\n"
        f"固定 failure taxonomy:\n{_taxonomy_prompt()}\n"
        "请阅读 storyboard 中从左到右、从上到下的时间序列，并按 response_schema 输出一个 JSON 对象。"
    )
    return {
        "checkpoint_step": manifest_record.get("checkpoint_step"),
        "task": manifest_record["task"],
        "trial": manifest_record["trial"],
        "image_path": manifest_record["storyboard_path"],
        "system_prompt": system_prompt,
        "episode_prompt": episode_prompt,
        "response_schema": build_response_schema(allowed_stages),
    }


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def validate_analysis_record(
    record: dict[str, Any],
    task_rule: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "primary_failure",
        "secondary_failure",
        "outcome_awareness_failure",
        "recovery_failure",
        "furthest_stage",
        "confidence",
        "short_evidence",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Analysis record is missing fields: {sorted(missing)}")
    if record["primary_failure"] not in FAILURE_TAXONOMY:
        raise ValueError(f"Invalid primary_failure: {record['primary_failure']}")
    secondary = record["secondary_failure"]
    if secondary is not None and secondary not in FAILURE_TAXONOMY:
        raise ValueError(f"Invalid secondary_failure: {secondary}")
    if not isinstance(record["outcome_awareness_failure"], bool):
        raise ValueError("outcome_awareness_failure must be boolean")
    if not isinstance(record["recovery_failure"], bool):
        raise ValueError("recovery_failure must be boolean")
    allowed_stages = {stage["id"] for stage in task_rule["stage_states"]}
    allowed_stages.add(UNDETERMINED_STAGE)
    if record["furthest_stage"] not in allowed_stages:
        raise ValueError(
            f"Invalid stage {record['furthest_stage']!r} for task {task_rule['task_id']}"
        )
    confidence = float(record["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    evidence = str(record["short_evidence"]).strip()
    if not evidence:
        raise ValueError("short_evidence must not be empty")
    normalized = dict(record)
    normalized["confidence"] = confidence
    normalized["short_evidence"] = evidence
    return normalized


def run_vlm_command(
    command: str,
    requests: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run one persistent VLM process over JSONL stdin and collect JSONL stdout."""
    argv = shlex.split(command)
    if not argv:
        raise ValueError("vlm command is empty")
    request_jsonl = "".join(
        json.dumps(request, ensure_ascii=False) + "\n" for request in requests
    )
    completed = subprocess.run(
        argv,
        input=request_jsonl,
        text=True,
        capture_output=True,
        check=True,
    )
    response_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(response_lines) != len(requests):
        raise ValueError(
            "VLM command must emit one JSON object per request line; "
            f"received {len(response_lines)} responses for {len(requests)} requests"
        )
    outputs = []
    for request, response_line in zip(requests, response_lines):
        response = json.loads(response_line)
        outputs.append(
            {
                **response,
                "checkpoint_step": request.get("checkpoint_step"),
                "task": request["task"],
                "trial": request["trial"],
            }
        )
    return outputs


ANALYSIS_COLUMNS = (
    "checkpoint_step",
    "task",
    "trial",
    "task_description",
    "primary_failure",
    "secondary_failure",
    "outcome_awareness_failure",
    "recovery_failure",
    "furthest_stage",
    "confidence",
    "needs_manual_review",
    "short_evidence",
    "video_path",
    "storyboard_path",
    "analyzer",
)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def attach_and_validate_analyses(
    manifest_records: Sequence[dict[str, Any]],
    raw_analyses: Sequence[dict[str, Any]],
    stage_rules: dict[int, dict[str, Any]],
    *,
    analyzer: str,
) -> list[dict[str, Any]]:
    manifest_by_key = {
        (record.get("checkpoint_step"), int(record["task"]), int(record["trial"])): record
        for record in manifest_records
    }
    rows = []
    seen = set()
    for raw in raw_analyses:
        key = (raw.get("checkpoint_step"), int(raw["task"]), int(raw["trial"]))
        if key not in manifest_by_key:
            raise ValueError(f"Analysis does not match the prepared manifest: {key}")
        if key in seen:
            raise ValueError(f"Duplicate analysis record: {key}")
        seen.add(key)
        manifest = manifest_by_key[key]
        task_id = int(manifest["task"])
        analysis = validate_analysis_record(raw, stage_rules[task_id])
        rows.append(
            {
                "checkpoint_step": manifest.get("checkpoint_step"),
                "task": task_id,
                "trial": int(manifest["trial"]),
                "task_description": stage_rules[task_id]["language"],
                "primary_failure": analysis["primary_failure"],
                "secondary_failure": analysis["secondary_failure"],
                "outcome_awareness_failure": analysis["outcome_awareness_failure"],
                "recovery_failure": analysis["recovery_failure"],
                "furthest_stage": analysis["furthest_stage"],
                "confidence": analysis["confidence"],
                "short_evidence": analysis["short_evidence"],
                "video_path": manifest["video_path"],
                "storyboard_path": manifest["storyboard_path"],
                "analyzer": raw.get("analyzer", analyzer),
            }
        )
    missing = set(manifest_by_key).difference(seen)
    if missing:
        raise ValueError(f"Missing analysis records for prepared episodes: {sorted(missing)}")
    rows.sort(
        key=lambda row: (
            -1 if row["checkpoint_step"] is None else row["checkpoint_step"],
            row["task"],
            row["trial"],
        )
    )
    return rows


def annotate_manual_review(
    rows: Sequence[dict[str, Any]],
    *,
    confidence_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0,1]")
    annotated = []
    for row in rows:
        evidence = str(row["short_evidence"])
        needs_manual_review = (
            float(row["confidence"]) < confidence_threshold
            or row["furthest_stage"] == UNDETERMINED_STAGE
            or row["primary_failure"] == "其他/无法判断"
            or any(marker in evidence for marker in MANUAL_REVIEW_EVIDENCE_MARKERS)
        )
        annotated.append({**row, "needs_manual_review": needs_manual_review})
    return annotated


SUMMARY_COLUMNS = (
    "checkpoint_step",
    "task",
    "failure_type",
    "episode_count",
    "analyzed_failures_in_group",
    "proportion",
    "mean_confidence",
    "outcome_awareness_count",
    "recovery_count",
)


def build_failure_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int | None, str, str], list[dict[str, Any]]] = defaultdict(list)
    group_totals: dict[tuple[int | None, str], int] = defaultdict(int)
    for row in rows:
        step = row["checkpoint_step"]
        task = str(row["task"])
        failure_type = row["primary_failure"]
        grouped[(step, task, failure_type)].append(row)
        grouped[(step, "ALL", failure_type)].append(row)
        group_totals[(step, task)] += 1
        group_totals[(step, "ALL")] += 1

    output = []
    for (step, task, failure_type), items in sorted(
        grouped.items(),
        key=lambda item: (
            -1 if item[0][0] is None else item[0][0],
            item[0][1] == "ALL",
            item[0][1],
            FAILURE_TAXONOMY.index(item[0][2]),
        ),
    ):
        total = group_totals[(step, task)]
        output.append(
            {
                "checkpoint_step": step,
                "task": task,
                "failure_type": failure_type,
                "episode_count": len(items),
                "analyzed_failures_in_group": total,
                "proportion": len(items) / total,
                "mean_confidence": sum(float(item["confidence"]) for item in items)
                / len(items),
                "outcome_awareness_count": sum(
                    bool(item["outcome_awareness_failure"]) for item in items
                ),
                "recovery_count": sum(bool(item["recovery_failure"]) for item in items),
            }
        )
    return output


def write_analysis_outputs(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    confidence_threshold: float = 0.7,
) -> list[dict[str, Any]]:
    annotated_rows = annotate_manual_review(
        rows,
        confidence_threshold=confidence_threshold,
    )
    _write_csv(output_dir / "failure_analysis.csv", annotated_rows, ANALYSIS_COLUMNS)
    _write_csv(
        output_dir / "failure_summary.csv",
        build_failure_summary(annotated_rows),
        SUMMARY_COLUMNS,
    )
    manual_rows = [row for row in annotated_rows if row["needs_manual_review"]]
    _write_csv(
        output_dir / "needs_manual_review.csv",
        manual_rows,
        ANALYSIS_COLUMNS,
    )
    return annotated_rows
