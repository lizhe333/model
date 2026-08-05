"""Response-cache loading and E0 target standardization helpers."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from .contracts import Stage1ContractError, validate_response_cache


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class CurrentHiddenReader:
    """Lazily map formal per-demo carrier shards without materializing 16+ GiB."""

    def __init__(self, payload: dict[str, Any], *, max_cached_shards: int = 8) -> None:
        self.payload = payload
        if int(max_cached_shards) <= 0:
            raise ValueError("max_cached_shards must be positive")
        self.max_cached_shards = int(max_cached_shards)
        self._loaded: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._verified: set[int] = set()

    def _shard(self, shard_index: int) -> torch.Tensor:
        if shard_index in self._loaded:
            self._loaded.move_to_end(shard_index)
            return self._loaded[shard_index]
        entry = self.payload["current_hidden_shards"][shard_index]
        path = Path(entry["path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Stage-1 current carrier shard is missing: {path}")
        if shard_index not in self._verified:
            actual = sha256_file(path)
            if actual != entry["sha256"]:
                raise Stage1ContractError(
                    f"Stage-1 current carrier shard SHA mismatch: {path}; expected {entry['sha256']}, got {actual}"
                )
            self._verified.add(shard_index)
        try:
            shard_payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        except (TypeError, RuntimeError):
            # Older torch builds may not expose mmap.  Correctness wins over
            # memory locality when a legacy runtime is used.
            shard_payload = torch.load(path, map_location="cpu", weights_only=False)
        tensor = shard_payload.get("current_hidden") if isinstance(shard_payload, dict) else None
        expected_shape = tuple(int(value) for value in entry["shape"])
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
            raise Stage1ContractError(f"carrier shard tensor shape mismatch for {path}")
        if tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor.float()).all():
            raise Stage1ContractError(f"carrier shard tensor is not finite BF16: {path}")
        self._loaded[shard_index] = tensor
        self._loaded.move_to_end(shard_index)
        # A random batch can touch dozens of ten-state shards.  Keeping every
        # one would quietly rematerialize the complete 16+ GiB carrier cache.
        # Values have already been copied into the requested batch tensor, so
        # evicting the least-recent mapping preserves correctness and bounds
        # host memory/mmap handles.
        while len(self._loaded) > self.max_cached_shards:
            self._loaded.popitem(last=False)
        return tensor

    def read(self, indices: torch.Tensor) -> torch.Tensor:
        selected = indices.detach().to(device="cpu", dtype=torch.long).flatten()
        if selected.numel() == 0:
            raise ValueError("cannot read an empty Stage-1 current-hidden batch")
        mapping = self.payload["current_hidden_index"][selected]
        output: torch.Tensor | None = None
        for shard_index in torch.unique(mapping[:, 0], sorted=True).tolist():
            positions = torch.nonzero(mapping[:, 0] == int(shard_index), as_tuple=False).flatten()
            locals_ = mapping[positions, 1].long()
            values = self._shard(int(shard_index))[locals_]
            if output is None:
                output = torch.empty(
                    (selected.numel(), *values.shape[1:]), dtype=values.dtype, device="cpu"
                )
            output[positions] = values
        if output is None or tuple(output.shape[1:]) != (3, 392, 1536):
            raise Stage1ContractError("failed to construct a valid Stage-1 carrier batch")
        return output


def load_response_cache(
    path: str | Path,
    *,
    require_trainable_splits_only: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_path = Path(path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Stage 1 response cache does not exist: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    result = validate_response_cache(
        payload,
        require_trainable_splits_only=require_trainable_splits_only,
    )
    result.update({"path": str(cache_path), "sha256": sha256_file(cache_path)})
    if payload.get("current_hidden") is None:
        payload["_current_hidden_reader"] = CurrentHiddenReader(payload)
    return payload, result


def split_indices(payload: dict[str, Any], split: str) -> torch.Tensor:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    values = [index for index, record in enumerate(payload["records"]) if record["split"] == split]
    if not values:
        raise Stage1ContractError(f"Stage 1 cache has no {split!r} records")
    return torch.tensor(values, dtype=torch.long)


def current_hidden(payload: dict[str, Any], indices: torch.Tensor) -> torch.Tensor:
    """Return $h_{8/16/24}$ for a batch from contiguous or formal sharded cache."""

    tensor = payload.get("current_hidden")
    if isinstance(tensor, torch.Tensor):
        return tensor[indices]
    reader = payload.get("_current_hidden_reader")
    if not isinstance(reader, CurrentHiddenReader):
        reader = CurrentHiddenReader(payload)
        payload["_current_hidden_reader"] = reader
    return reader.read(indices)


def standardized_targets(
    payload: dict[str, Any],
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    targets = payload["response_targets"][indices].to(device=device, dtype=torch.float32)
    mean = payload["normalization_mean"].to(device=device, dtype=torch.float32)
    std = payload["normalization_std"].to(device=device, dtype=torch.float32)
    return (targets - mean[None, :, None]) / std[None, :, None]
