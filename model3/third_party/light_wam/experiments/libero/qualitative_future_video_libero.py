import json
import logging
import os
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
    _validate_visualize_future_video_cfg,
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


def _wrap_text(text: str, max_chars: int = 60) -> list[str]:
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


def _compose_qualitative_panel(
    *,
    task_description: str,
    current_frame: np.ndarray,
    pred_frames_by_offset: dict[int, np.ndarray],
    gt_frames_by_offset: dict[int, np.ndarray],
    panel_offsets: list[int],
) -> np.ndarray:
    target_h, target_w = current_frame.shape[:2]
    label_col_w = 190
    margin = 24
    gap = 12
    title_lines = _wrap_text(f'Task instruction: "{task_description}"', max_chars=72)
    title_h = 26 * len(title_lines) + 16
    row_header_h = 34
    row_h = target_h
    num_cols = len(panel_offsets)
    canvas_w = margin * 2 + label_col_w + num_cols * target_w + (num_cols - 1) * gap
    canvas_h = margin * 2 + title_h + row_header_h + 3 * row_h + 2 * gap + 24
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    text_x = margin
    text_y = margin
    for line in title_lines:
        draw.text((text_x, text_y), line, fill=(0, 0, 0))
        text_y += 26

    row_labels = [
        ("Current Obs.", None),
        ("GT Future", None),
        ("Pred. Future", None),
    ]
    header_y = margin + title_h
    base_y = header_y + row_header_h
    for row_idx, (row_label, row_time) in enumerate(row_labels):
        row_y = base_y + row_idx * (row_h + gap)
        draw.text((margin, row_y + 8), row_label, fill=(0, 0, 0))
        if row_time is not None:
            draw.text((margin, row_y + 32), row_time, fill=(90, 90, 90))

    current_img = Image.fromarray(current_frame)
    canvas.paste(current_img, (margin + label_col_w, base_y))
    draw.text((margin + label_col_w + 8, header_y), "t=0", fill=(0, 0, 0))

    blank_tile = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    for col_idx, offset in enumerate(panel_offsets):
        x = margin + label_col_w + col_idx * (target_w + gap)
        draw.text((x + 8, header_y), f"t={offset}", fill=(0, 0, 0))
        gt = gt_frames_by_offset.get(offset)
        pred = pred_frames_by_offset.get(offset)
        gt_img = Image.fromarray(blank_tile if gt is None else gt)
        pred_img = Image.fromarray(blank_tile if pred is None else pred)
        canvas.paste(gt_img, (x, base_y + (row_h + gap)))
        canvas.paste(pred_img, (x, base_y + 2 * (row_h + gap)))
        if gt is None:
            draw.text((x + 12, base_y + (row_h + gap) + 12), "N/A", fill=(120, 120, 120))
        if pred is None:
            draw.text((x + 12, base_y + 2 * (row_h + gap) + 12), "N/A", fill=(120, 120, 120))

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
def _infer_future_video_with_latents(
    model: torch.nn.Module,
    *,
    prompt: Optional[str],
    input_image: torch.Tensor,
    num_video_frames: int,
    action_horizon: int,
    proprio: Optional[torch.Tensor],
    context: Optional[torch.Tensor],
    context_mask: Optional[torch.Tensor],
    num_inference_steps: int,
    sigma_shift: Optional[float],
    seed: Optional[int],
    rand_device: str,
    tiled: bool,
) -> dict[str, Any]:
    if not bool(getattr(model, "uses_state_fusion_action_expert")()):
        raise ValueError("Qualitative future-video export currently supports state-fusion Light-WAM only.")

    model.eval()
    if input_image.ndim == 3:
        input_image = input_image.unsqueeze(0)
    if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
        raise ValueError(
            f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
        )

    _, _, height, width = input_image.shape
    checked_h, checked_w, checked_t = model._check_resize_height_width(height, width, num_video_frames)
    if (checked_h, checked_w) != (height, width):
        raise ValueError(
            f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
        )
    if checked_t != num_video_frames:
        raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

    if proprio is not None:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)

    input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
    first_frame_latents = model._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    video_first_frame_latents = first_frame_latents
    apply_spatial_downsample = True
    restore_spatial_resolution = True
    if bool(getattr(model, "_use_lowres_video_training_objective")()):
        video_first_frame_latents, _ = model._maybe_downsample_video_latents_for_backbone(first_frame_latents)
        apply_spatial_downsample = False
        restore_spatial_resolution = False

    latent_t = (num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    latent_h = int(video_first_frame_latents.shape[-2])
    latent_w = int(video_first_frame_latents.shape[-1])

    generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
    latents_video = torch.randn(
        (1, model.vae.model.z_dim, latent_t, latent_h, latent_w),
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    latents_video[:, :, 0:1] = video_first_frame_latents.clone()

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
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )

    pred_action = model._predict_state_fusion_action_from_observation(
        observation_latents=first_frame_latents,
        action_horizon=action_horizon,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=fuse_flag,
    )

    infer_timesteps_video, infer_deltas_video = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=latents_video.dtype,
        shift_override=sigma_shift,
    )
    for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
        timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=model.device)
        pred_video = model._predict_video_only(
            latents_video=latents_video,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            apply_spatial_downsample=apply_spatial_downsample,
            restore_spatial_resolution=restore_spatial_resolution,
        )
        latents_video = model.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
        latents_video[:, :, 0:1] = video_first_frame_latents.clone()

    decoded_frames = model._decode_latents(latents_video, tiled=tiled)
    return {
        "action": pred_action[0].detach().to(device="cpu", dtype=torch.float32),
        "video": decoded_frames,
        "video_latents": latents_video[0].detach().to(device="cpu", dtype=torch.float32),
    }


def _save_clip_bundle(
    clip_dir: Path,
    *,
    task_description: str,
    panel_output_path: Optional[Path],
    current_frame: np.ndarray,
    pred_frames: list[np.ndarray],
    gt_frames_by_offset: dict[int, np.ndarray],
    pred_video_latents: torch.Tensor,
    future_step_offsets: list[int],
    save_latents: bool,
) -> dict[str, Any]:
    clip_dir.mkdir(parents=True, exist_ok=True)
    target_h, target_w = pred_frames[0].shape[:2]
    current_resized = _resize_rgb_array(current_frame, target_w, target_h)
    _save_rgb(clip_dir / "current_frame.png", current_resized)

    latent_frame_count = int(pred_video_latents.shape[1])
    if save_latents:
        for latent_idx in range(latent_frame_count):
            np.save(
                clip_dir / f"pred_latent_frame_{latent_idx:02d}.npy",
                pred_video_latents[:, latent_idx].numpy(),
            )
        torch.save(pred_video_latents, clip_dir / "pred_video_latents.pt")

    available_offsets = []
    pred_frames_by_offset: dict[int, np.ndarray] = {}
    for frame_idx, offset in enumerate(future_step_offsets):
        pred_frame = _resize_rgb_array(pred_frames[frame_idx], target_w, target_h)
        pred_frames_by_offset[int(offset)] = pred_frame
        _save_rgb(clip_dir / f"pred_frame_{frame_idx:02d}_step{offset:03d}.png", pred_frame)

        gt_frame = gt_frames_by_offset.get(offset)
        if gt_frame is None:
            continue
        available_offsets.append(offset)
        gt_resized = _resize_rgb_array(gt_frame, target_w, target_h)
        _save_rgb(clip_dir / f"gt_frame_{frame_idx:02d}_step{offset:03d}.png", gt_resized)
        triptych = np.concatenate([current_resized, pred_frame, gt_resized], axis=1)
        _save_rgb(clip_dir / f"triplet_{frame_idx:02d}_step{offset:03d}.png", triptych)

    panel_offsets = [8, 16, 24, 32]
    qualitative_panel = _compose_qualitative_panel(
        task_description=task_description,
        current_frame=current_resized,
        pred_frames_by_offset=pred_frames_by_offset,
        gt_frames_by_offset={
            int(offset): _resize_rgb_array(gt_frame, target_w, target_h)
            for offset, gt_frame in gt_frames_by_offset.items()
        },
        panel_offsets=panel_offsets,
    )
    _save_rgb(clip_dir / "qualitative_panel.png", qualitative_panel)
    if panel_output_path is not None:
        _save_rgb(panel_output_path, qualitative_panel)

    meta = {
        "future_step_offsets": future_step_offsets,
        "available_gt_offsets": available_offsets,
        "num_pred_frames": len(pred_frames),
        "latent_shape": list(pred_video_latents.shape),
        "num_latent_frames": latent_frame_count,
        "panel_offsets": panel_offsets,
    }
    (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


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
    num_video_frames = _get_num_video_frames(cfg)
    future_step_offsets = [
        frame_idx * int(cfg.data.train.action_video_freq_ratio) for frame_idx in range(num_video_frames)
    ]
    capture_interval = int(cfg.EVALUATION.get("qualitative_capture_interval", 16))
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 10))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 30))
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
    input_h = int(cfg.data.train.video_size[0])
    input_w = int(cfg.data.train.video_size[1])
    sigma_shift = (
        None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.sigma_shift)
    )
    save_latents = bool(cfg.EVALUATION.get("qualitative_save_latents", True))
    binarize_gripper = bool(cfg.EVALUATION.get("binarize_gripper", False))
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
    clip_records: list[dict[str, Any]] = []
    control_step = 0
    done = False
    query_seed = None if cfg.get("seed") is None else int(cfg.seed)

    try:
        total_steps = max_steps + num_steps_wait
        for sim_step in range(total_steps):
            if sim_step < num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action())
                if done:
                    break
                continue

            current_imgs = get_libero_image(obs)
            current_frame = _frame_to_rgb_array(current_imgs)
            for clip in clip_records:
                offset = control_step - int(clip["anchor_step"])
                if offset in clip["future_step_offsets"] and offset not in clip["gt_frames_by_offset"]:
                    clip["gt_frames_by_offset"][offset] = current_frame.copy()

            future_payload = None
            if control_step % capture_interval == 0:
                prompt = None if cached_prompt_context is not None else DEFAULT_PROMPT.format(task=task_description)
                image_tensor, proprio, _ = _obs_to_model_input(
                    obs,
                    cfg=cfg,
                    processor=processor,
                    width=input_w,
                    height=input_h,
                    device=model_device,
                    dtype=model.torch_dtype,
                )
                future_payload = _infer_future_video_with_latents(
                    model,
                    prompt=prompt,
                    input_image=image_tensor,
                    num_video_frames=num_video_frames,
                    action_horizon=action_horizon,
                    proprio=proprio,
                    context=cached_prompt_context,
                    context_mask=cached_prompt_context_mask,
                    num_inference_steps=num_inference_steps,
                    sigma_shift=sigma_shift,
                    seed=query_seed,
                    rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
                    tiled=bool(cfg.EVALUATION.get("tiled", False)),
                )
                pred_frames = [np.array(frame.convert("RGB")) for frame in future_payload["video"]]
                clip_records.append(
                    {
                        "clip_index": len(clip_records),
                        "anchor_step": control_step,
                        "current_frame": current_frame.copy(),
                        "pred_frames": pred_frames,
                        "pred_video_latents": future_payload["video_latents"].clone(),
                        "future_step_offsets": list(future_step_offsets),
                        "gt_frames_by_offset": {0: current_frame.copy()},
                    }
                )

            if len(pending_actions) == 0:
                if future_payload is not None:
                    action_chunk = _postprocess_action_chunk(
                        future_payload["action"],
                        processor,
                        binarize_gripper=binarize_gripper,
                    )
                else:
                    prompt = None if cached_prompt_context is not None else DEFAULT_PROMPT.format(task=task_description)
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
                        tiled=bool(cfg.EVALUATION.get("tiled", False)),
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

    clip_summaries = []
    for clip in clip_records:
        clip_dir = task_dir / f"clip_{int(clip['clip_index']):03d}_step_{int(clip['anchor_step']):04d}"
        panel_output_path = (
            panel_output_dir
            / f"task_{task_id:02d}_clip_{int(clip['clip_index']):03d}_step_{int(clip['anchor_step']):04d}.png"
        )
        clip_meta = _save_clip_bundle(
            clip_dir,
            task_description=task_description,
            panel_output_path=panel_output_path,
            current_frame=clip["current_frame"],
            pred_frames=clip["pred_frames"],
            gt_frames_by_offset=clip["gt_frames_by_offset"],
            pred_video_latents=clip["pred_video_latents"],
            future_step_offsets=clip["future_step_offsets"],
            save_latents=save_latents,
        )
        clip_meta.update(
            {
                "clip_dir": str(clip_dir),
                "anchor_step": int(clip["anchor_step"]),
            }
        )
        clip_summaries.append(clip_meta)

    task_summary = {
        "task_id": task_id,
        "task_description": task_description,
        "num_clips": len(clip_summaries),
        "success": bool(done),
        "task_dir": str(task_dir),
        "clips": clip_summaries,
    }
    (task_dir / "task_summary.json").write_text(json.dumps(task_summary, indent=2), encoding="utf-8")
    return task_summary


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")

    cfg = _maybe_apply_training_run_config(cfg)
    _validate_visualize_future_video_cfg(cfg)
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
    suite_output_dir = output_dir / cfg.EVALUATION.task_suite_name
    suite_output_dir.mkdir(parents=True, exist_ok=True)
    panel_output_dir = suite_output_dir / "qualitative_panels"
    panel_output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    num_tasks = int(task_suite.n_tasks)
    max_tasks = int(cfg.EVALUATION.get("qualitative_num_tasks", num_tasks))
    task_ids = list(range(min(num_tasks, max_tasks)))

    logging.info(
        "Start LIBERO future-video qualitative export | suite=%s tasks=%s ckpt=%s output=%s",
        cfg.EVALUATION.task_suite_name,
        task_ids,
        cfg.ckpt,
        output_dir,
    )

    all_task_summaries = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if len(initial_states) < 1:
            raise ValueError(f"No initial states found for task_id={task_id}.")
        logging.info("Running task_id=%s / %s", task_id, cfg.EVALUATION.task_suite_name)
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
        "output_dir": str(output_dir),
        "duration_sec": time.time() - start_time,
        "tasks": all_task_summaries,
    }
    (suite_output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logging.info("Saved qualitative summary: %s", suite_output_dir / "summary.json")


if __name__ == "__main__":
    main()
