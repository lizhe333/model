import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_single import (
    _denormalize_action,
    _get_max_steps,
    _get_num_video_frames,
    _load_model_checkpoint,
    _maybe_apply_training_run_config,
    _maybe_preencode_task_prompt_context,
    _obs_to_model_input,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _mixed_precision_to_model_dtype,
    _safe_close_env,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
)
from model3.third_party.light_wam.src.lightwam.datasets.lerobot.processors.lightwam_processor import LightWAMProcessor
from model3.third_party.light_wam.src.lightwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from model3.third_party.light_wam.src.lightwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from model3.third_party.light_wam.src.lightwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark


def _sanitize_name(text: str, limit: int = 80) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip().lower())
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("_")
    return sanitized[:limit] if sanitized else "task"


def _wrap_text(text: str, max_chars: int = 64) -> list[str]:
    words = text.strip().split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        items = []
        for value in frame.values():
            if isinstance(value, Image.Image):
                items.append(np.array(value.convert("RGB")))
            else:
                items.append(np.array(value, copy=True))
        return np.concatenate(items, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _resize_rgb_array(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    if pil_image.size != (width, height):
        pil_image = pil_image.resize((width, height), resample=Image.BILINEAR)
    return np.array(pil_image.convert("RGB"))


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _normalize_attention_map(attn: np.ndarray) -> np.ndarray:
    attn = attn.astype(np.float32)
    attn = attn - float(attn.min())
    denom = float(attn.max())
    if denom <= 1e-8:
        return np.zeros_like(attn, dtype=np.float32)
    return attn / denom


def _upsample_attention_map(attn_grid: np.ndarray, width: int, height: int) -> np.ndarray:
    attn_grid = _normalize_attention_map(attn_grid)
    attn_uint8 = np.clip(attn_grid * 255.0, 0, 255).astype(np.uint8)
    resized = Image.fromarray(attn_uint8).resize((width, height), resample=Image.BILINEAR)
    return np.array(resized).astype(np.float32) / 255.0


def _heatmap_to_rgb(attn_map: np.ndarray) -> np.ndarray:
    x = np.clip(attn_map, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _make_overlay(image_rgb: np.ndarray, attn_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = image_rgb.astype(np.float32) / 255.0
    heat = _heatmap_to_rgb(attn_map)
    overlay = (1.0 - alpha) * base + alpha * heat
    return np.clip(overlay * 255.0, 0, 255).astype(np.uint8)


def _make_topk_masked_overlay(
    image_rgb: np.ndarray,
    attn_map: np.ndarray,
    *,
    topk_ratio: float,
    alpha: float = 0.55,
    background_dim: float = 0.28,
) -> tuple[np.ndarray, np.ndarray]:
    topk_ratio = float(np.clip(topk_ratio, 1e-4, 1.0))
    flat = attn_map.reshape(-1)
    k = max(1, int(np.ceil(flat.size * topk_ratio)))
    threshold = float(np.partition(flat, flat.size - k)[flat.size - k])
    mask = attn_map >= threshold

    base = image_rgb.astype(np.float32) / 255.0
    heat = _heatmap_to_rgb(attn_map)
    highlight = (1.0 - alpha) * base + alpha * heat
    masked = base * background_dim
    masked[mask] = highlight[mask]
    return np.clip(masked * 255.0, 0, 255).astype(np.uint8), mask.astype(np.uint8)


def _build_layer_panel(
    *,
    task_description: str,
    current_rgb: np.ndarray,
    summary_overlays: list[tuple[int, np.ndarray]],
    top_query_overlays: list[tuple[int, np.ndarray]],
    topk_overlays: list[tuple[int, np.ndarray]],
    control_step: Optional[int] = None,
) -> np.ndarray:
    target_h, target_w = current_rgb.shape[:2]
    margin = 24
    gap = 16
    label_col_w = 135
    row_label_h = 28
    title_lines = _wrap_text(f'Task instruction: "{task_description}"', max_chars=72)
    title_h = 26 * len(title_lines) + 16
    col_layers = [layer_idx for layer_idx, _ in summary_overlays]
    if [layer_idx for layer_idx, _ in top_query_overlays] != col_layers or [
        layer_idx for layer_idx, _ in topk_overlays
    ] != col_layers:
        raise ValueError("Summary/top-query/top-k overlay layers must align.")
    num_cols = max(1, len(col_layers))
    num_rows = 4
    canvas_w = margin * 2 + label_col_w + num_cols * target_w + max(0, num_cols - 1) * gap
    canvas_h = margin * 2 + title_h + row_label_h + num_rows * target_h + max(0, num_rows - 1) * gap + 16
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    text_y = margin
    for line in title_lines:
        draw.text((margin, text_y), line, fill=(0, 0, 0))
        text_y += 26
    if control_step is not None:
        draw.text((margin, text_y), f"Control step: {control_step}", fill=(80, 80, 80))

    header_y = margin + title_h
    base_x = margin + label_col_w
    base_y = header_y + row_label_h
    for col_idx, layer_idx in enumerate(col_layers):
        x = base_x + col_idx * (target_w + gap)
        draw.text((x + 8, header_y), f"Layer {layer_idx}", fill=(0, 0, 0))

    row_labels = ["Current Obs.", "Summary", "Top query", "Top-k masked"]
    row_tiles = [
        [current_rgb for _ in range(num_cols)],
        [overlay for _, overlay in summary_overlays],
        [overlay for _, overlay in top_query_overlays],
        [overlay for _, overlay in topk_overlays],
    ]
    for row_idx, row_label in enumerate(row_labels):
        y = base_y + row_idx * (target_h + gap)
        draw.text((margin, y + 8), row_label, fill=(0, 0, 0))
        for col_idx, tile in enumerate(row_tiles[row_idx]):
            x = base_x + col_idx * (target_w + gap)
            canvas.paste(Image.fromarray(tile), (x, y))

    return np.array(canvas)


def _postprocess_action_chunk(
    action_tensor: torch.Tensor,
    processor: LightWAMProcessor,
    *,
    binarize_gripper: bool,
) -> np.ndarray:
    action = _denormalize_action(action_tensor, processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])
    return action


@torch.no_grad()
def _extract_query_pool_attention(
    model: torch.nn.Module,
    *,
    prompt: Optional[str],
    input_image: torch.Tensor,
    proprio: Optional[torch.Tensor],
    context: Optional[torch.Tensor],
    context_mask: Optional[torch.Tensor],
) -> list[dict[str, Any]]:
    if not bool(getattr(model, "uses_state_fusion_action_expert")()):
        raise ValueError("Qualitative attention export currently supports state-fusion Light-WAM only.")

    input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
    first_frame_latents = model._encode_input_image_latents_tensor(
        input_image=input_image,
        tiled=False,
    )
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    use_prompt = prompt is not None
    use_context = context is not None or context_mask is not None
    if use_prompt and use_context:
        raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
    if not use_prompt and not use_context:
        raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

    if use_prompt:
        context, context_mask = model.encode_prompt(prompt)
    else:
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=model.device, dtype=torch.bool, non_blocking=True)

    if proprio is not None:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )

    timestep_video = torch.zeros(
        (first_frame_latents.shape[0],),
        dtype=first_frame_latents.dtype,
        device=first_frame_latents.device,
    )
    video_pre = model._build_action_observation_video_pre(
        observation_latents=first_frame_latents,
        timestep_video=timestep_video,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=fuse_flag,
    )
    _ = model.video_expert.forward_backbone(video_pre)
    fusion_inputs = model._build_multilayer_action_fusion_inputs()
    return model.state_fusion_action_expert.summarize_pooling_attention(fusion_inputs)


def _infer_token_grid_shape(
    *,
    model: torch.nn.Module,
    input_h: int,
    input_w: int,
    token_count: int,
) -> tuple[int, int]:
    latent_h = input_h // int(model.vae.upsampling_factor)
    latent_w = input_w // int(model.vae.upsampling_factor)
    if bool(getattr(model, "apply_video_latent_downsample_to_action_branch", False)):
        factor = int(getattr(model, "video_latent_spatial_downsample_factor", 1))
        latent_h //= factor
        latent_w //= factor
    patch_size = tuple(getattr(model.video_expert, "patch_size", (1, 2, 2)))
    token_h = latent_h // int(patch_size[1])
    token_w = latent_w // int(patch_size[2])
    if token_h * token_w != token_count:
        raise ValueError(
            f"Token grid mismatch: inferred {token_h}x{token_w}={token_h * token_w}, token_count={token_count}"
        )
    return token_h, token_w


def _save_attention_capture(
    *,
    task_id: int,
    capture_idx: int,
    control_step: int,
    task_description: str,
    current_rgb: np.ndarray,
    attention_summaries: list[dict[str, Any]],
    model: torch.nn.Module,
    input_h: int,
    input_w: int,
    task_dir: Path,
    panel_output_dir: Path,
    topk_ratio: float,
) -> dict[str, Any]:
    capture_dir = task_dir / f"capture_{capture_idx:03d}_step_{control_step:04d}"
    capture_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb(capture_dir / "current_frame.png", current_rgb)

    summary_overlays: list[tuple[int, np.ndarray]] = []
    top_query_overlays: list[tuple[int, np.ndarray]] = []
    topk_overlays: list[tuple[int, np.ndarray]] = []
    capture_meta_layers: list[dict[str, Any]] = []
    for layer_summary in attention_summaries:
        layer_idx = int(layer_summary["layer_idx"])
        if "adapted" not in layer_summary["sources"]:
            continue
        source_summary = layer_summary["sources"]["adapted"]
        summary_attention = (
            source_summary["summary_attention"][0]
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        per_query_attention = (
            source_summary["per_query_attention"][0]
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        query_importance = (
            source_summary["query_importance"]
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        top_query_idx = int(source_summary["top_query_idx"].detach().cpu().item())
        top_query_attention = per_query_attention[top_query_idx]
        token_h, token_w = _infer_token_grid_shape(
            model=model,
            input_h=input_h,
            input_w=input_w,
            token_count=int(summary_attention.shape[0]),
        )
        summary_grid = summary_attention.reshape(token_h, token_w)
        top_query_grid = top_query_attention.reshape(token_h, token_w)
        summary_map = _upsample_attention_map(summary_grid, input_w, input_h)
        top_query_map = _upsample_attention_map(top_query_grid, input_w, input_h)

        summary_heatmap_rgb = (_heatmap_to_rgb(summary_map) * 255.0).astype(np.uint8)
        summary_overlay_rgb = _make_overlay(current_rgb, summary_map, alpha=0.45)
        top_query_heatmap_rgb = (_heatmap_to_rgb(top_query_map) * 255.0).astype(np.uint8)
        top_query_overlay_rgb = _make_overlay(current_rgb, top_query_map, alpha=0.45)
        topk_overlay_rgb, topk_mask = _make_topk_masked_overlay(
            current_rgb,
            summary_map,
            topk_ratio=topk_ratio,
        )

        _save_rgb(capture_dir / f"layer_{layer_idx:02d}_summary_overlay.png", summary_overlay_rgb)
        _save_rgb(capture_dir / f"layer_{layer_idx:02d}_summary_heatmap.png", summary_heatmap_rgb)
        _save_rgb(capture_dir / f"layer_{layer_idx:02d}_top_query_overlay.png", top_query_overlay_rgb)
        _save_rgb(capture_dir / f"layer_{layer_idx:02d}_top_query_heatmap.png", top_query_heatmap_rgb)
        _save_rgb(capture_dir / f"layer_{layer_idx:02d}_topk_masked_overlay.png", topk_overlay_rgb)
        np.save(capture_dir / f"layer_{layer_idx:02d}_summary_token_attention.npy", summary_grid.astype(np.float32))
        np.save(capture_dir / f"layer_{layer_idx:02d}_summary_image_attention.npy", summary_map.astype(np.float32))
        np.save(capture_dir / f"layer_{layer_idx:02d}_top_query_token_attention.npy", top_query_grid.astype(np.float32))
        np.save(capture_dir / f"layer_{layer_idx:02d}_top_query_image_attention.npy", top_query_map.astype(np.float32))
        np.save(capture_dir / f"layer_{layer_idx:02d}_topk_mask.npy", topk_mask.astype(np.uint8))
        np.save(capture_dir / f"layer_{layer_idx:02d}_query_importance.npy", query_importance.astype(np.float32))
        summary_overlays.append((layer_idx, summary_overlay_rgb))
        top_query_overlays.append((layer_idx, top_query_overlay_rgb))
        topk_overlays.append((layer_idx, topk_overlay_rgb))
        capture_meta_layers.append(
            {
                "layer_idx": layer_idx,
                "token_grid_shape": [token_h, token_w],
                "num_tokens": int(summary_attention.shape[0]),
                "top_query_idx": top_query_idx,
                "topk_ratio": float(topk_ratio),
            }
        )

    panel = _build_layer_panel(
        task_description=task_description,
        current_rgb=current_rgb,
        summary_overlays=summary_overlays,
        top_query_overlays=top_query_overlays,
        topk_overlays=topk_overlays,
        control_step=control_step,
    )
    panel_path = capture_dir / "attention_panel.png"
    _save_rgb(panel_path, panel)
    _save_rgb(
        panel_output_dir / f"task_{task_id:02d}_capture_{capture_idx:03d}_step_{control_step:04d}.png",
        panel,
    )
    return {
        "capture_idx": capture_idx,
        "control_step": control_step,
        "capture_dir": str(capture_dir),
        "panel_path": str(panel_path),
        "layers": capture_meta_layers,
    }


def _run_single_task(
    *,
    task,
    task_id: int,
    initial_state: Any,
    model: torch.nn.Module,
    processor: LightWAMProcessor,
    cfg: DictConfig,
    model_device: str,
    output_dir: Path,
    panel_output_dir: Path,
) -> dict[str, Any]:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 30))
    capture_interval = int(cfg.EVALUATION.get("qualitative_capture_interval", 16))
    topk_ratio = float(cfg.EVALUATION.get("qualitative_attention_topk_ratio", 0.15))
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 10))
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    action_horizon = (
        int(cfg.EVALUATION.action_horizon)
        if cfg.EVALUATION.get("action_horizon") is not None
        else int(cfg.data.train.num_frames) - 1
    )
    num_inference_steps = (
        int(cfg.EVALUATION.num_inference_steps)
        if cfg.EVALUATION.get("num_inference_steps") is not None
        else int(cfg.get("eval_num_inference_steps", 20))
    )
    sigma_shift = (
        None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.sigma_shift)
    )
    binarize_gripper = bool(cfg.EVALUATION.get("binarize_gripper", False))
    tiled = bool(cfg.EVALUATION.get("tiled", False))
    input_h = int(cfg.data.train.video_size[0])
    input_w = int(cfg.data.train.video_size[1])
    task_dir = output_dir / f"task_{task_id:02d}_{_sanitize_name(task_description)}"
    task_dir.mkdir(parents=True, exist_ok=True)

    cached_prompt_context, cached_prompt_context_mask = _maybe_preencode_task_prompt_context(
        task_description=task_description,
        model=model,
        cfg=cfg,
        model_device=model_device,
    )

    env.reset()
    obs = env.set_init_state(initial_state)
    pending_actions: list[list[float]] = []
    capture_summaries: list[dict[str, Any]] = []
    control_step = 0
    done = False
    query_seed = None if cfg.get("seed") is None else int(cfg.seed)
    prompt = None if cached_prompt_context is not None else DEFAULT_PROMPT.format(task=task_description)
    try:
        total_steps = max_steps + num_steps_wait
        for sim_step in range(total_steps):
            if sim_step < num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action())
                if done:
                    break
                continue

            raw_frame = _frame_to_rgb_array(get_libero_image(obs))
            current_rgb = _resize_rgb_array(raw_frame, input_w, input_h)
            image_tensor = None
            proprio = None

            if control_step % capture_interval == 0:
                image_tensor, proprio, _ = _obs_to_model_input(
                    obs,
                    cfg=cfg,
                    processor=processor,
                    width=input_w,
                    height=input_h,
                    device=model_device,
                    dtype=model.torch_dtype,
                )
                attention_summaries = _extract_query_pool_attention(
                    model,
                    prompt=prompt,
                    input_image=image_tensor,
                    proprio=proprio,
                    context=cached_prompt_context,
                    context_mask=cached_prompt_context_mask,
                )
                capture_summary = _save_attention_capture(
                    task_id=task_id,
                    capture_idx=len(capture_summaries),
                    control_step=control_step,
                    task_description=task_description,
                    current_rgb=current_rgb,
                    attention_summaries=attention_summaries,
                    model=model,
                    input_h=input_h,
                    input_w=input_w,
                    task_dir=task_dir,
                    panel_output_dir=panel_output_dir,
                    topk_ratio=topk_ratio,
                )
                capture_summaries.append(capture_summary)

            if len(pending_actions) == 0:
                if image_tensor is None or proprio is None:
                    image_tensor, proprio, _ = _obs_to_model_input(
                        obs,
                        cfg=cfg,
                        processor=processor,
                        width=input_w,
                        height=input_h,
                        device=model_device,
                        dtype=model.torch_dtype,
                    )
                action_pred = model.infer_action(
                    prompt=prompt,
                    input_image=image_tensor,
                    action_horizon=action_horizon,
                    proprio=proprio,
                    context=cached_prompt_context,
                    context_mask=cached_prompt_context_mask,
                    negative_prompt=str(cfg.EVALUATION.get("negative_prompt", "")),
                    text_cfg_scale=float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
                    num_inference_steps=num_inference_steps,
                    sigma_shift=sigma_shift,
                    seed=query_seed,
                    rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
                    tiled=tiled,
                )
                action_chunk = _postprocess_action_chunk(
                    action_pred["action"],
                    processor,
                    binarize_gripper=binarize_gripper,
                )
                pending_actions = action_chunk[:replan_steps].tolist()

            obs, _, done, _ = env.step(pending_actions.pop(0))
            control_step += 1
            if done:
                break
    finally:
        _safe_close_env(env)

    task_summary = {
        "task_id": task_id,
        "task_description": task_description,
        "task_dir": str(task_dir),
        "success": bool(done),
        "num_captures": len(capture_summaries),
        "captures": capture_summaries,
    }
    (task_dir / "task_summary.json").write_text(json.dumps(task_summary, indent=2), encoding="utf-8")
    return task_summary


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")

    cfg = _maybe_apply_training_run_config(cfg)
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: LightWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    output_dir = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_output_dir = output_dir / cfg.EVALUATION.task_suite_name / "query_attention"
    suite_output_dir.mkdir(parents=True, exist_ok=True)
    panel_output_dir = suite_output_dir / "attention_panels"
    panel_output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    num_tasks = int(task_suite.n_tasks)
    max_tasks = int(cfg.EVALUATION.get("qualitative_num_tasks", num_tasks))
    task_ids = list(range(min(num_tasks, max_tasks)))

    logging.info(
        "Start LIBERO learned-query attention export | suite=%s tasks=%s ckpt=%s output=%s",
        cfg.EVALUATION.task_suite_name,
        task_ids,
        cfg.ckpt,
        suite_output_dir,
    )

    all_task_summaries = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if len(initial_states) < 1:
            raise ValueError(f"No initial states found for task_id={task_id}.")
        task_summary = _run_single_task(
            task=task,
            task_id=task_id,
            initial_state=initial_states[0],
            model=model,
            processor=processor,
            cfg=cfg,
            model_device=model_device,
            output_dir=suite_output_dir,
            panel_output_dir=panel_output_dir,
        )
        all_task_summaries.append(task_summary)

    summary = {
        "task_suite_name": cfg.EVALUATION.task_suite_name,
        "task_ids": task_ids,
        "ckpt": str(cfg.ckpt),
        "dataset_stats_path": str(dataset_stats_path),
        "output_dir": str(suite_output_dir),
        "duration_sec": time.time() - start_time,
        "tasks": all_task_summaries,
    }
    (suite_output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Saved attention summary: %s", suite_output_dir / "summary.json")


if __name__ == "__main__":
    main()
