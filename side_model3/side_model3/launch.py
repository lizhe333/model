"""Suite-neutral Side-Model3 contract preflight.

No formal dataset, budget, checkpoint, or evidence run is registered here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import SideModel3Config, default_config, load_config
from .contracts import ContractError, validate_contract


PACKAGE_ROOT = Path(__file__).resolve().parent
HYDRA_MODEL_PATH = PACKAGE_ROOT / "configs/hydra/model/side_model3_v1.yaml"


def _top_level_hydra_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def build_preflight(config: SideModel3Config | None = None) -> dict[str, Any]:
    """Validate method and Hydra wiring without importing or loading Wan."""

    config = default_config() if config is None else config
    contract = validate_contract(config)
    if not HYDRA_MODEL_PATH.is_file():
        raise ContractError(f"missing Side-Model3 Hydra model: {HYDRA_MODEL_PATH}")
    hydra_model = _top_level_hydra_values(HYDRA_MODEL_PATH)
    if hydra_model.get("_target_") != "side_model3.runtime.create_side_model3_wam":
        raise ContractError("Side-Model3 Hydra target is not the registered factory")
    if hydra_model.get("wam_adapter") not in {"null", "~"}:
        raise ContractError("Side-Model3 Hydra model must remove the inherited Wan adapter")
    if hydra_model.get("state_fusion_action_expert_config") not in {"null", "~"}:
        raise ContractError("Side-Model3 Hydra model must remove inherited StateFusion")

    return {
        "passed": True,
        "mode": "method_contract_preflight",
        "formal_launch_registered": False,
        "loads_model_weights": False,
        "hydra_model_path": str(HYDRA_MODEL_PATH),
        "hydra_target": hydra_model["_target_"],
        "contract": contract,
    }


def launch(
    config_path: str | Path | None = None,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Run the only authorized launcher mode: a side-effect-free preflight."""

    if not dry_run:
        raise ContractError(
            "Side-Model3 has no registered suite or formal execution contract; "
            "only --dry-run is available"
        )
    config = default_config() if config_path is None else load_config(config_path)
    result = build_preflight(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Side-Model3 v1 wiring")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="optional method-only JSON override; no suite fields are accepted",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("Side-Model3 currently exposes only --dry-run preflight")
    launch(args.config, dry_run=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
