"""Cache the exact Dynamic O2 step-0 current hidden carrier $h_l(s)$."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from model3_o2_dynamic.config import Model3O2DynamicConfig, load_config
from model3_o2_dynamic.models import Model3O2DynamicWAM
from model3_o2_dynamic.runtime import create_model3_o2_dynamic_wam

from .action_normalization import OfficialO2Normalizer, load_official_o2_normalizer
from .collect import validate_collection_shard
from .common import sha256_file
from .contracts import LAYERS, Stage1DataConfig, Stage1ContractError, task_names_for_suite, task_subset_for_suite


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parent_config_path(dynamic: Model3O2DynamicConfig) -> Path:
    candidate = dynamic.initialization.model3_checkpoint.parent.parent.parent / "config.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(f"cannot locate shared Model3 parent config: {candidate}")
    return candidate


def _seed_exact_o2_initialization(seed: int) -> None:
    """Match O2 gate creation before every clean Dynamic step-0 construction."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dynamic_query_config() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "hydra"
        / "model"
        / "model3_o2_dynamic_response_query_flow.yaml"
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Dynamic Hydra model override must be a mapping")
    return raw


def instantiate_clean_dynamic_step0(
    dynamic: Model3O2DynamicConfig,
    *,
    device: str | torch.device,
) -> Model3O2DynamicWAM:
    """Construct the exact Treatment O2 gate state used by Stage 2 step zero."""

    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Dynamic current-carrier extraction requires a CUDA device")
    # Match the original-Wan teacher's explicit local checkpoint root. Without
    # this, DiffSynth falls back to ``./checkpoints`` and may attempt a network
    # download despite the pinned Wan components already being available.
    checkpoint_root = Path(dynamic.base.backend.checkpoint_root).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Dynamic carrier checkpoint root is missing: {checkpoint_root}")
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(checkpoint_root)
    parent_path = parent_config_path(dynamic)
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    if not isinstance(parent, dict) or not isinstance(parent.get("model"), dict):
        raise ValueError("shared parent config lacks model mapping")
    kwargs = copy.deepcopy(parent["model"])
    kwargs.pop("_target_", None)
    override = _dynamic_query_config()
    kwargs["action_query_policy_config"] = copy.deepcopy(override["action_query_policy_config"])
    kwargs["response_adapter_config"] = copy.deepcopy(override["response_adapter_config"])
    # ``yaml.safe_load`` intentionally does not resolve Hydra environment
    # interpolations.  Bind the selected Dynamic config directly so carrier
    # construction has the same global Object/Long schedule identity as the
    # eventual Stage-2 model.
    kwargs["dynamic_response_schedule"] = {
        "freeze_through_step": int(dynamic.schedule.freeze_through_step),
        "first_adapter_update_step": int(dynamic.schedule.first_adapter_update_step),
        "adapter_lr_scale": float(dynamic.schedule.adapter_lr_scale),
        "gate_freeze_through_step": int(dynamic.schedule.gate_freeze_through_step),
        "first_gate_update_step": int(dynamic.schedule.first_gate_update_step),
        "gate_lr_scale": float(dynamic.schedule.gate_lr_scale),
    }
    kwargs["model3_warmstart_path"] = str(dynamic.initialization.model3_checkpoint)
    kwargs["model3_warmstart_sha256"] = dynamic.initialization.model3_checkpoint_sha256
    kwargs["model3_warmstart_step"] = dynamic.initialization.model3_checkpoint_step
    # The cached language context is the exact deployed prompt representation;
    # avoiding a duplicate T5 instance does not alter h_l and keeps the one-GPU
    # formal carrier writer within memory.
    kwargs["load_text_encoder"] = False
    _seed_exact_o2_initialization(dynamic.base.training.seed)
    model = create_model3_o2_dynamic_wam(
        **kwargs,
        model_dtype=torch.bfloat16,
        device=str(target_device),
    )
    if not isinstance(model, Model3O2DynamicWAM):
        raise RuntimeError(f"Dynamic factory returned {type(model).__name__}")
    model.eval().requires_grad_(False)
    expected = model.o2_gate_initialization_sha256
    if model.original_o2_tensor_sha256() != expected:
        raise RuntimeError("clean Dynamic O2 step-0 hash changed immediately after construction")
    return model


@dataclass
class DynamicCarrierExtractor:
    dynamic: Model3O2DynamicConfig
    device: torch.device
    model: Model3O2DynamicWAM
    parent_config: Any
    processor: Any
    normalizer: OfficialO2Normalizer
    source_identity: dict[str, Any]

    @classmethod
    def create(
        cls,
        dynamic: Model3O2DynamicConfig,
        *,
        device: str | torch.device,
    ) -> "DynamicCarrierExtractor":
        model = instantiate_clean_dynamic_step0(dynamic, device=device)
        parent_path = parent_config_path(dynamic)
        parent_cfg = OmegaConf.load(parent_path)
        processor_cfg = parent_cfg.data.train.processor
        processor = instantiate(processor_cfg).eval()
        normalizer = load_official_o2_normalizer(parent_config_path=parent_path)
        source_identity = {
            "track_id": "model3_o2_dynamic",
            "method_id": model.method_id,
            "model_class": type(model).__name__,
            "model3_warmstart_path": str(dynamic.initialization.model3_checkpoint),
            "model3_warmstart_sha256": dynamic.initialization.model3_checkpoint_sha256,
            "model3_warmstart_step": dynamic.initialization.model3_checkpoint_step,
            "original_o2_tensor_sha256": model.o2_gate_initialization_sha256,
            "o2_gate_initialization_seed": int(dynamic.base.training.seed),
            "current_timestep": 0,
            "current_precision": "torch.bfloat16",
            "current_hidden_layers": list(LAYERS),
            "current_hidden_shape": [3, 392, 1536],
            "response_adapters_disabled_for_carrier": True,
            "parent_config_path": str(parent_path),
            "parent_config_sha256": sha256_file(parent_path),
        }
        return cls(
            dynamic=dynamic,
            device=torch.device(device),
            model=model,
            parent_config=parent_cfg,
            processor=processor,
            normalizer=normalizer,
            source_identity=source_identity,
        )

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def _input_image(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor) or tuple(images.shape[:1]) != (2,) or images.shape[-1] != 3:
            raise ValueError(f"collection current images must be [2,H,W,3], got {getattr(images, 'shape', None)}")
        image_batch = {
            "image": images[0].permute(2, 0, 1).unsqueeze(0).to(dtype=torch.uint8),
            "wrist_image": images[1].permute(2, 0, 1).unsqueeze(0).to(dtype=torch.uint8),
        }
        pixels = self.processor.build_pixel_values_from_episode_images({"images": image_batch})
        if tuple(pixels.shape[:3]) != (2, 1, 3):
            raise RuntimeError(f"parent processor camera output changed: {tuple(pixels.shape)}")
        concatenation = self.parent_config.data.train.get("concat_multi_camera", "horizontal")
        if concatenation != "horizontal":
            raise RuntimeError(f"Dynamic carrier requires shared horizontal camera concatenation, got {concatenation}")
        video = torch.cat([pixels[0], pixels[1]], dim=-1)
        video_size = tuple(int(value) for value in self.parent_config.data.train.video_size)
        if tuple(video.shape[1:]) != (3, *video_size):
            raise RuntimeError(
                f"parent processor image geometry changed: got {tuple(video.shape)}, expected [1,3,{video_size[0]},{video_size[1]}]"
            )
        # The official evaluator's final resize/crop is identity at 224x448;
        # assert that rather than introducing a duplicate interpolation path.
        # Its final Normalize(mean=.5, std=.5) remains required, however.
        return video.float().sub(0.5).div(0.5).to(device=self.device, dtype=torch.bfloat16)

    @torch.inference_mode()
    def encode_record(self, record: dict[str, Any]) -> torch.Tensor:
        image = self._input_image(record["current_images"])
        raw_proprio = record["current_proprio_raw"].reshape(1, -1)
        proprio = self.normalizer.normalize_proprio(raw_proprio).to(device=self.device, dtype=torch.bfloat16)
        from analysis.action_response_local.teacher import _load_context

        context_cpu, mask_cpu, _ = _load_context(
            str(record["instruction"]), Path(self.dynamic.base.data.text_embedding_cache_dir)
        )
        context, context_mask = self.model._prepare_model3_context(
            prompt=None,
            context=context_cpu.unsqueeze(0),
            context_mask=mask_cpu.unsqueeze(0),
            proprio=proprio,
        )
        observation_latents = self.model._encode_input_image_latents_tensor(input_image=image, tiled=False)
        fuse_flag = bool(getattr(self.model.video_expert, "fuse_vae_embedding_in_latents", False))
        with self.model.response_adapters_disabled():
            layer_states = self.model._build_action_layer_states(
                observation_latents=observation_latents,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
        if tuple(int(state["layer_idx"]) for state in layer_states) != LAYERS:
            raise RuntimeError("Dynamic carrier layer order is not 8/16/24")
        hidden = torch.stack([state["adapted"][0].detach().cpu() for state in layer_states], dim=0)
        if tuple(hidden.shape) != (3, 392, 1536):
            raise RuntimeError(f"Dynamic carrier shape changed: {tuple(hidden.shape)}")
        if hidden.requires_grad or not torch.isfinite(hidden.float()).all():
            raise RuntimeError("Dynamic carrier must be finite stop-gradient tensors")
        return hidden.to(dtype=torch.bfloat16)


def validate_carrier_shard(payload: dict[str, Any], *, split: str, task: str, config: Stage1DataConfig) -> None:
    if payload.get("artifact_kind") != "stage1_current_carrier_shard":
        raise Stage1ContractError("wrong Dynamic Stage-1 carrier artifact kind")
    if payload.get("track_id") != "model3_o2_dynamic" or payload.get("split") != split or payload.get("task") != task:
        raise Stage1ContractError("carrier shard identity mismatch")
    hidden = payload.get("current_hidden")
    if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (config.states_per_demo, 3, 392, 1536):
        raise Stage1ContractError(f"carrier tensor shape mismatch: {getattr(hidden, 'shape', None)}")
    if hidden.dtype != torch.bfloat16 or not torch.isfinite(hidden.float()).all():
        raise Stage1ContractError("carrier must be finite BF16 tensors")
    source = payload.get("source_identity")
    if not isinstance(source, dict) or source.get("response_adapters_disabled_for_carrier") is not True:
        raise Stage1ContractError("carrier was not explicitly extracted before Dynamic response adapters")
    if not isinstance(source.get("original_o2_tensor_sha256"), str):
        raise Stage1ContractError("carrier is missing exact O2 step-0 hash")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != config.states_per_demo:
        raise Stage1ContractError("carrier records must cover one ten-state demonstration shard")


def _matches_current_collection_provenance(
    payload: object,
    *,
    collection_path: Path,
    collection_sha256: str,
) -> bool:
    """Return whether an existing carrier was derived from this exact input shard."""

    if not isinstance(payload, dict):
        return False
    return (
        payload.get("collection_path") == str(collection_path.expanduser().resolve())
        and payload.get("collection_sha256") == collection_sha256
    )


def _can_reuse_existing_carrier(
    *,
    output_path: Path,
    split: str,
    task: str,
    config: Stage1DataConfig,
    collection_path: Path,
    collection_sha256: str,
) -> bool:
    """Validate an existing carrier before deciding whether resume may reuse it."""

    try:
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        validate_carrier_shard(payload, split=split, task=task, config=config)
    except Exception:
        # A malformed or stale cache is not a legal resume input.  Leave it in
        # place until the replacement is completely written and atomically
        # moved into place below.
        return False
    return _matches_current_collection_provenance(
        payload,
        collection_path=collection_path,
        collection_sha256=collection_sha256,
    )


def _require_test_permission(splits: Iterable[str], after_stage1_export: str | Path | None) -> None:
    if "test" not in set(splits):
        return
    if after_stage1_export is None or not Path(after_stage1_export).expanduser().is_file():
        raise Stage1ContractError(
            "test current carriers are sealed until the fixed Stage-1 adapter export exists; "
            "pass --after-stage1-export after step 5K"
        )


def extract_demo_shard(
    *,
    extractor: DynamicCarrierExtractor,
    collection_path: Path,
    output_path: Path,
    split: str,
    task: str,
    overwrite: bool,
) -> Path:
    config = Stage1DataConfig()
    current_collection_path = collection_path.expanduser().resolve()
    current_collection_sha256 = sha256_file(current_collection_path)
    if output_path.is_file() and not overwrite and _can_reuse_existing_carrier(
        output_path=output_path,
        split=split,
        task=task,
        config=config,
        collection_path=current_collection_path,
        collection_sha256=current_collection_sha256,
    ):
        return output_path
    collection = torch.load(current_collection_path, map_location="cpu", weights_only=False)
    validate_collection_shard(collection, config=config)
    if collection.get("task") != task or collection.get("split") != split:
        raise Stage1ContractError("collection / carrier split identity mismatch")
    hidden = torch.stack([extractor.encode_record(record) for record in collection["records"]], dim=0)
    if extractor.model.original_o2_tensor_sha256() != extractor.source_identity["original_o2_tensor_sha256"]:
        raise RuntimeError("Stage-1 current-carrier extraction changed a frozen O2 tensor")
    payload = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_current_carrier_shard",
        "task": task,
        "split": split,
        "demo_id": int(collection["demo_id"]),
        "layers": list(LAYERS),
        "current_hidden": hidden,
        "records": [
            {
                key: record[key]
                for key in ("sample_id", "task", "demo_id", "split", "source_index", "progress_bin", "motion_labels")
            }
            for record in collection["records"]
        ],
        "source_identity": extractor.source_identity,
        "action_normalization_identity": extractor.normalizer.identity(),
        "collection_path": str(current_collection_path),
        "collection_sha256": current_collection_sha256,
    }
    validate_carrier_shard(payload, split=split, task=task, config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return output_path


def extract_all(
    *,
    dynamic_config: Model3O2DynamicConfig,
    output_root: str | Path,
    device: str | torch.device,
    splits: Iterable[str] = ("train", "validation"),
    tasks: Iterable[str] | None = None,
    after_stage1_export: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = Stage1DataConfig()
    config.validate()
    selected_splits = tuple(str(value) for value in splits)
    if not selected_splits or any(value not in {"train", "validation", "test"} for value in selected_splits):
        raise ValueError(f"unsupported Stage-1 carrier splits: {selected_splits}")
    _require_test_permission(selected_splits, after_stage1_export)
    root = Path(output_root).expanduser().resolve()
    selected_tasks = task_subset_for_suite(dynamic_config.evaluation.suite, tasks)
    full_suite = selected_tasks == task_names_for_suite(dynamic_config.evaluation.suite)
    extractor = DynamicCarrierExtractor.create(dynamic_config, device=device)
    entries: list[dict[str, Any]] = []
    try:
        for split in selected_splits:
            for task in selected_tasks:
                for demo_id in range(50):
                    if config.split_for_demo(demo_id) != split:
                        continue
                    collection_path = root / "collection" / split / task / f"demo_{demo_id:03d}.pt"
                    if not collection_path.is_file():
                        raise FileNotFoundError(f"missing collection shard: {collection_path}")
                    output_path = root / "carrier" / split / task / f"demo_{demo_id:03d}.pt"
                    result = extract_demo_shard(
                        extractor=extractor,
                        collection_path=collection_path,
                        output_path=output_path,
                        split=split,
                        task=task,
                        overwrite=overwrite,
                    )
                    entries.append(
                        {
                            "split": split,
                            "task": task,
                            "demo_id": demo_id,
                            "path": str(result),
                            "sha256": sha256_file(result),
                        }
                    )
                    print(f"[dynamic-carrier:{split}] task={task} demo={demo_id + 1}/50", flush=True)
    finally:
        source_identity = dict(extractor.source_identity)
        normalizer_identity = extractor.normalizer.identity()
        extractor.close()
    result = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_current_carrier_summary",
        "splits": list(selected_splits),
        "tasks": list(selected_tasks),
        "source_identity": source_identity,
        "action_normalization_identity": normalizer_identity,
        "shards": entries,
        "split_state_counts": {
            split: len([entry for entry in entries if entry["split"] == split]) * config.states_per_demo
            for split in selected_splits
        },
        "test_read": "test" in selected_splits,
    }
    if full_suite and selected_splits == ("train", "validation"):
        _write_json(root / "carrier" / "summary.json", result)
    elif full_suite:
        _write_json(root / "carrier" / "test_or_aux_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--after-stage1-export", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = extract_all(
        dynamic_config=load_config(args.config),
        output_root=args.output_root,
        device=args.device,
        splits=args.splits,
        tasks=args.tasks,
        after_stage1_export=args.after_stage1_export,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
