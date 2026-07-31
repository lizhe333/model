"""Materialize matched Stage 2 configs after Model5-80K exists."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
STAGE1_CONFIG = ROOT / "model5_o2/configs/libero_long_stage1_model5_80k.json"
MODEL5_METHOD = "model5_asymmetric_tri_timestep_query_flow_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_parent(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Model5-80K checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Model5-80K payload must be a dictionary")
    if payload.get("method_id") != MODEL5_METHOD:
        raise ValueError(f"wrong parent method: {payload.get('method_id')!r}")
    if payload.get("model_class") != "Model5WAM":
        raise ValueError(f"wrong parent class: {payload.get('model_class')!r}")
    if payload.get("step") != 80_000:
        raise ValueError(f"parent checkpoint must be step 80000, got {payload.get('step')!r}")
    expected_features = {
        "temporal_scope": "current_plus_noisy_future",
        "fixed_future_timestep": 1000,
        "num_future_latent_slots": 8,
        "spatial_downsample_factor": 1,
    }
    if payload.get("action_feature_config") != expected_features:
        raise ValueError("Model5-80K parent changed the nine-slot temporal contract")
    return _sha256(path)


def _stage2_raw(
    base: dict,
    *,
    role: str,
    hydra_model: str,
    checkpoint: Path,
    sha256: str,
    port: int,
) -> dict:
    raw = deepcopy(base)
    # Generated configs live under run evidence, not beside the source config;
    # pin the repository root so relative backend/data paths remain correct.
    raw["project_root"] = str(ROOT)
    raw["stage_role"] = role
    raw["backend"]["hydra_model"] = hydra_model
    raw["initialization"] = {
        "mode": "model_only_warmstart",
        "model5_checkpoint": str(checkpoint),
        "model5_checkpoint_sha256": sha256,
        "model5_checkpoint_step": 80000,
    }
    raw["training"].update(
        {
            "main_process_port": port,
            "max_steps": 10000,
            "save_every": 5000,
            "warmup_steps": 1000,
        }
    )
    if role == "stage2_model5_o2":
        raw["architecture"].update(
            {
                "query_trace_readout": "layer_separable_gated_residual",
                "readout_num_layers": 3,
                "readout_query_dim": 512,
                "readout_gate_type": "querywise_scalar",
                "readout_identity_init": True,
            }
        )
    return raw


def prepare(parent: Path, output_dir: Path) -> list[Path]:
    sha256 = _verify_parent(parent)
    base = json.loads(STAGE1_CONFIG.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=False)
    configs = {
        "libero_long_stage2_model5_control.json": _stage2_raw(
            base,
            role="stage2_model5_control",
            hydra_model="model5_o2_stage2_model5_control_query_flow",
            checkpoint=parent.resolve(),
            sha256=sha256,
            port=29614,
        ),
        "libero_long_stage2_model5_o2.json": _stage2_raw(
            base,
            role="stage2_model5_o2",
            hydra_model="model5_o2_layer_aware_temporal_query_flow",
            checkpoint=parent.resolve(),
            sha256=sha256,
            port=29615,
        ),
    }
    written: list[Path] = []
    for name, raw in configs.items():
        destination = output_dir / name
        destination.write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(destination)
    (output_dir / "model5_80k_parent_identity.json").write_text(
        json.dumps(
            {"path": str(parent.resolve()), "sha256": sha256, "step": 80000},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in prepare(args.parent.resolve(), args.output_dir.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
