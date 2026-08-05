"""One explicit, restartable launcher for Dynamic O2 response prewarm.

``--execute`` is intentionally required.  A plain invocation validates the
frozen template and prints the exact sequence without collecting simulator
data, opening a GPU, writing a cache, or starting either training stage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from .config import Model3O2DynamicConfig, load_config
from .contracts import validate_contract
from .launch import launch as launch_stage2
from .prepare_stage2 import prepare_stage2_config
from .stage1.carrier import extract_all as extract_carriers
from .stage1.collect import collect_all
from .stage1.diagnostics import run_heldout_input_ablations
from .stage1.preflight import preflight_demonstrations
from .stage1.prepare import prepare_train_validation_cache
from .stage1.selection import select_all
from .stage1.teacher import extract_all as extract_teacher
from .stage1.train import train_stage1
from .stage1.contracts import Stage1ContractError, Stage1TrainConfig


STAGES = (
    "preflight_stage1_demonstrations",
    "select_train_validation",
    "collect_train_validation",
    "carrier_train_validation",
    "teacher_train_validation",
    "prepare_train_validation_cache",
    "stage1_response_warmup",
    "select_test",
    "collect_test",
    "carrier_test",
    "teacher_test",
    "heldout_diagnostics",
    "materialize_stage2_config",
    "stage2_joint_training",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _validate_run_id(run_id: str) -> str:
    value = str(run_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("run-id must contain only letters, digits, '.', '_' or '-'")
    return value


def pipeline_paths(config: Model3O2DynamicConfig, run_id: str) -> dict[str, Path]:
    clean_id = _validate_run_id(run_id)
    root = config.evidence_root
    data_root = root / f"{clean_id}_stage1_data"
    stage1_output = root / f"{clean_id}_stage1_train"
    pipeline_root = root / f"{clean_id}_pipeline"
    return {
        "pipeline_root": pipeline_root,
        "stage1_data_root": data_root,
        "stage1_output": stage1_output,
        "train_cache": data_root / "prepared" / "train_validation_response_cache.pt",
        "adapter_export": stage1_output / "adapter_export" / "stage1_adapter_step_005000.pt",
        "stage2_config": pipeline_root / "stage2" / "stage2_config.json",
    }


def build_pipeline_plan(
    *,
    config_path: str | Path,
    run_id: str,
    stage1_device: str,
) -> dict[str, Any]:
    template_path = Path(config_path).expanduser().resolve()
    config = load_config(template_path)
    contract = validate_contract(config, check_paths=False)
    paths = pipeline_paths(config, run_id)
    return {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "dynamic_response_prewarm_pipeline_plan",
        "template_config": str(template_path),
        "template_config_sha256": _sha256(template_path),
        "run_id": _validate_run_id(run_id),
        "stage1_device": str(stage1_device),
        "stage2_physical_gpu_ids": list(config.base.training.gpu_ids),
        "contract": contract,
        "paths": {key: str(value) for key, value in paths.items()},
        "stages": list(STAGES),
        "test_isolation": {
            "file_integrity_before_any_split_payload": [
                "preflight_stage1_demonstrations",
            ],
            "before_stage1_export": [
                "select_train_validation",
                "collect_train_validation",
                "carrier_train_validation",
                "teacher_train_validation",
                "prepare_train_validation_cache",
                "stage1_response_warmup",
            ],
            "after_fixed_stage1_export_only": [
                "select_test",
                "collect_test",
                "carrier_test",
                "teacher_test",
                "heldout_diagnostics",
            ],
        },
        "execute_required": True,
    }


def _load_complete_stage1(output: Path) -> dict[str, Any] | None:
    result_path = output / "stage1_result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("track_id") != "model3_o2_dynamic"
        or result.get("status") != "complete"
        or result.get("optimizer_steps") != 5000
    ):
        raise Stage1ContractError(f"existing Stage-1 result is not a complete fixed 5K run: {result_path}")
    export = Path(result.get("adapter_export", ""))
    if not export.is_file():
        raise Stage1ContractError("complete Stage-1 result does not reference its adapter export")
    return result


def _existing_stage2_config(path: Path, adapter_export: Path) -> None:
    if not path.is_file():
        return
    materialized = load_config(path)
    validate_contract(materialized, check_paths=True)
    if materialized.initialization.response_adapter_export != adapter_export.resolve():
        raise ValueError("existing Stage-2 config points at a different Stage-1 adapter export")


def _initialize_manifest(
    *,
    plan: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    root = paths["pipeline_root"]
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "pipeline_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("track_id") != "model3_o2_dynamic"
            or existing.get("template_config_sha256") != plan["template_config_sha256"]
            or existing.get("run_id") != plan["run_id"]
        ):
            raise ValueError("existing Dynamic pipeline manifest does not match this config/run-id")
        return existing
    manifest = {
        **plan,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_at": None,
        "status": "active",
        "events": [],
    }
    _write_json(manifest_path, manifest)
    return manifest


def _record_event(
    *,
    manifest: dict[str, Any],
    paths: dict[str, Path],
    stage: str,
    status: str,
    detail: dict[str, Any] | None = None,
) -> None:
    manifest.setdefault("events", []).append(
        {
            "stage": stage,
            "status": status,
            "at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "detail": detail or {},
        }
    )
    _write_json(paths["pipeline_root"] / "pipeline_manifest.json", manifest)


def run_pipeline(
    *,
    config_path: str | Path,
    run_id: str,
    execute: bool,
    stage1_device: str = "cuda:0",
    libero_root: str | Path | None = None,
    stop_after: str | None = None,
    stage1_resume: str | Path | None = None,
    stage2_resume: str | Path | None = None,
    skip_heldout_diagnostics: bool = False,
) -> dict[str, Any] | int:
    """Run the sealed stages in order, or return a no-write dry plan."""

    plan = build_pipeline_plan(config_path=config_path, run_id=run_id, stage1_device=stage1_device)
    if not execute:
        return plan
    if stop_after is not None and stop_after not in STAGES:
        raise ValueError(f"unknown --stop-after stage: {stop_after}")
    template_path = Path(config_path).expanduser().resolve()
    dynamic = load_config(template_path)
    # The parent Model3/O2 contract treats this as a normal backend cache
    # directory.  Create it only for an explicit execution; the dry plan above
    # deliberately remains no-write.
    dynamic.base.backend.hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    # Perform all formal filesystem legality checks before simulator/model work.
    validate_contract(dynamic, check_paths=True)
    paths = pipeline_paths(dynamic, run_id)
    manifest = _initialize_manifest(plan=plan, paths=paths)
    data_root = paths["stage1_data_root"]
    stage1_output = paths["stage1_output"]

    def run_stage(name: str, function: Callable[[], Any]) -> Any:
        _record_event(manifest=manifest, paths=paths, stage=name, status="started")
        try:
            result = function()
        except Exception as error:
            _record_event(
                manifest=manifest,
                paths=paths,
                stage=name,
                status="failed",
                detail={"error_type": type(error).__name__, "error": str(error)},
            )
            manifest["status"] = "failed"
            manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            _write_json(paths["pipeline_root"] / "pipeline_manifest.json", manifest)
            raise
        detail = result if isinstance(result, dict) else {"result": str(result)}
        _record_event(manifest=manifest, paths=paths, stage=name, status="complete", detail=detail)
        return result

    def stop_if_requested(name: str) -> dict[str, Any] | None:
        if stop_after != name:
            return None
        manifest["status"] = "stopped_after_requested_stage"
        manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(paths["pipeline_root"] / "pipeline_manifest.json", manifest)
        return manifest

    run_stage(
        "preflight_stage1_demonstrations",
        lambda: preflight_demonstrations(
            dynamic_config=dynamic,
            output_root=data_root,
            libero_root_path=libero_root,
        ),
    )
    if (stopped := stop_if_requested("preflight_stage1_demonstrations")) is not None:
        return stopped

    run_stage(
        "select_train_validation",
        lambda: select_all(
            dynamic_config=dynamic,
            output_root=data_root,
            splits=("train", "validation"),
            libero_root_path=libero_root,
        ),
    )
    if (stopped := stop_if_requested("select_train_validation")) is not None:
        return stopped
    run_stage(
        "collect_train_validation",
        lambda: collect_all(
            dynamic_config=dynamic,
            output_root=data_root,
            splits=("train", "validation"),
            libero_root_path=libero_root,
        ),
    )
    if (stopped := stop_if_requested("collect_train_validation")) is not None:
        return stopped
    run_stage(
        "carrier_train_validation",
        lambda: extract_carriers(
            dynamic_config=dynamic,
            output_root=data_root,
            device=stage1_device,
            splits=("train", "validation"),
        ),
    )
    if (stopped := stop_if_requested("carrier_train_validation")) is not None:
        return stopped
    run_stage(
        "teacher_train_validation",
        lambda: extract_teacher(
            dynamic_config=dynamic,
            output_root=data_root,
            device=stage1_device,
            splits=("train", "validation"),
        ),
    )
    if (stopped := stop_if_requested("teacher_train_validation")) is not None:
        return stopped
    cache_path = run_stage(
        "prepare_train_validation_cache",
        lambda: prepare_train_validation_cache(dynamic_config=dynamic, output_root=data_root),
    )
    if (stopped := stop_if_requested("prepare_train_validation_cache")) is not None:
        return stopped

    complete_stage1 = _load_complete_stage1(stage1_output)
    if complete_stage1 is None:
        stage1_config = Stage1TrainConfig.from_mapping(dynamic.stage1)
        stage1_result = run_stage(
            "stage1_response_warmup",
            lambda: train_stage1(
                cache_path=cache_path,
                output_dir=stage1_output,
                config=stage1_config,
                device=stage1_device,
                resume=stage1_resume,
            ),
        )
    else:
        stage1_result = complete_stage1
        _record_event(
            manifest=manifest,
            paths=paths,
            stage="stage1_response_warmup",
            status="reused_complete_fixed_5k_export",
            detail={"adapter_export": stage1_result["adapter_export"]},
        )
    adapter_export = Path(stage1_result["adapter_export"]).expanduser().resolve()
    if adapter_export != paths["adapter_export"].resolve():
        raise Stage1ContractError("Stage-1 result adapter export path is outside this pipeline run")
    if (stopped := stop_if_requested("stage1_response_warmup")) is not None:
        return stopped

    if not skip_heldout_diagnostics:
        run_stage(
            "select_test",
            lambda: select_all(
                dynamic_config=dynamic,
                output_root=data_root,
                splits=("test",),
                after_stage1_export=adapter_export,
                libero_root_path=libero_root,
            ),
        )
        if (stopped := stop_if_requested("select_test")) is not None:
            return stopped
        run_stage(
            "collect_test",
            lambda: collect_all(
                dynamic_config=dynamic,
                output_root=data_root,
                splits=("test",),
                after_stage1_export=adapter_export,
                libero_root_path=libero_root,
            ),
        )
        if (stopped := stop_if_requested("collect_test")) is not None:
            return stopped
        run_stage(
            "carrier_test",
            lambda: extract_carriers(
                dynamic_config=dynamic,
                output_root=data_root,
                device=stage1_device,
                splits=("test",),
                after_stage1_export=adapter_export,
            ),
        )
        if (stopped := stop_if_requested("carrier_test")) is not None:
            return stopped
        run_stage(
            "teacher_test",
            lambda: extract_teacher(
                dynamic_config=dynamic,
                output_root=data_root,
                device=stage1_device,
                splits=("test",),
                after_stage1_export=adapter_export,
            ),
        )
        if (stopped := stop_if_requested("teacher_test")) is not None:
            return stopped
        run_stage(
            "heldout_diagnostics",
            lambda: run_heldout_input_ablations(
                dynamic_config=dynamic,
                output_root=data_root,
                train_cache_path=cache_path,
                stage1_output_dir=stage1_output,
                adapter_export=adapter_export,
                device=stage1_device,
            ),
        )
        if (stopped := stop_if_requested("heldout_diagnostics")) is not None:
            return stopped
    elif stop_after in {"select_test", "collect_test", "carrier_test", "teacher_test", "heldout_diagnostics"}:
        raise ValueError("cannot stop after a skipped held-out diagnostic stage")

    if not paths["stage2_config"].is_file():
        run_stage(
            "materialize_stage2_config",
            lambda: prepare_stage2_config(
                template_config=template_path,
                adapter_export=adapter_export,
                output_config=paths["stage2_config"],
            ),
        )
    else:
        _existing_stage2_config(paths["stage2_config"], adapter_export)
        _record_event(
            manifest=manifest,
            paths=paths,
            stage="materialize_stage2_config",
            status="reused_materialized_config",
            detail={"stage2_config": str(paths["stage2_config"])},
        )
    if (stopped := stop_if_requested("materialize_stage2_config")) is not None:
        return stopped
    _record_event(
        manifest=manifest,
        paths=paths,
        stage="stage2_joint_training",
        status="launching",
        detail={"stage2_config": str(paths["stage2_config"]), "resume": None if stage2_resume is None else str(stage2_resume)},
    )
    try:
        return_code = launch_stage2(
            paths["stage2_config"],
            _validate_run_id(run_id),
            dry_run=False,
            resume=None if stage2_resume is None else Path(stage2_resume),
        )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _record_event(
            manifest=manifest,
            paths=paths,
            stage="stage2_joint_training",
            status="failed",
            detail={"error_type": type(error).__name__, "error": str(error)},
        )
        raise
    manifest["status"] = "complete" if return_code == 0 else "stage2_nonzero_exit"
    manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _record_event(
        manifest=manifest,
        paths=paths,
        stage="stage2_joint_training",
        status="complete" if return_code == 0 else "failed",
        detail={"return_code": int(return_code)},
    )
    return int(return_code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage1-device", default="cuda:0")
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--stop-after", choices=STAGES)
    parser.add_argument("--stage1-resume", type=Path)
    parser.add_argument("--stage2-resume", type=Path)
    parser.add_argument("--skip-heldout-diagnostics", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(
        config_path=args.config,
        run_id=args.run_id,
        execute=bool(args.execute),
        stage1_device=args.stage1_device,
        libero_root=args.libero_root,
        stop_after=args.stop_after,
        stage1_resume=args.stage1_resume,
        stage2_resume=args.stage2_resume,
        skip_heldout_diagnostics=bool(args.skip_heldout_diagnostics),
    )
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
