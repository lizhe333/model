"""Model3 with the action-flow decoder replaced by direct L1 regression."""

from __future__ import annotations

from typing import Any, Optional

import torch

from model3.models.model3_wam import Model3WAM

from .vla_query_regression_action_expert import VLAQueryRegressionActionExpert


class Model3RegressionWAM(Model3WAM):
    """Matched Model3 query encoder with a deterministic regression decoder."""

    method_id = VLAQueryRegressionActionExpert.method_id
    action_inference_timing_key = "model_action_regression_head"
    allow_legacy_policy_config_without_method_id = True

    @classmethod
    def _build_action_policy(
        cls,
        *,
        video_hidden_dim: int,
        action_dim: int,
        num_fusion_layers: int,
        proprio_dim: Optional[int],
        action_query_policy_config: dict[str, Any],
        device: str,
        torch_dtype: torch.dtype,
    ) -> VLAQueryRegressionActionExpert:
        return VLAQueryRegressionActionExpert(
            video_hidden_dim=video_hidden_dim,
            action_dim=action_dim,
            num_fusion_layers=num_fusion_layers,
            proprio_dim=proprio_dim,
            **dict(action_query_policy_config),
        ).to(device=device, dtype=torch_dtype)

    @property
    def action_policy(self) -> VLAQueryRegressionActionExpert:
        expert = self.state_fusion_action_expert
        if not isinstance(expert, VLAQueryRegressionActionExpert):
            raise RuntimeError(
                "Model3RegressionWAM requires VLAQueryRegressionActionExpert; "
                f"got {type(expert).__name__ if expert is not None else None}."
            )
        return expert

    def _compute_regression_action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if pred_action.shape != target_action.shape:
            raise ValueError(
                f"predicted and target action shapes differ: {pred_action.shape} vs {target_action.shape}"
            )
        action_loss_token = (
            pred_action.float() - target_action.float()
        ).abs().mean(dim=2)
        step_weights = self._build_action_temporal_weights(
            action_horizon=int(action_loss_token.shape[1]),
            device=action_loss_token.device,
            dtype=action_loss_token.dtype,
        ).view(1, -1)
        if action_is_pad is None:
            weight_sum = step_weights.sum(dim=1).clamp(min=1e-6)
            return (action_loss_token * step_weights).sum(dim=1) / weight_sum
        valid = (~action_is_pad).to(
            device=action_loss_token.device,
            dtype=action_loss_token.dtype,
        )
        weighted_valid = valid * step_weights
        valid_sum = weighted_valid.sum(dim=1).clamp(min=1e-6)
        return (action_loss_token * weighted_valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        """Joint future-video flow and direct normalized-action L1 objective."""

        self._reset_timing_breakdown()
        total_timing = self._timing_start()
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        video_supervision_latents = self._build_video_training_supervision_latents(
            input_latents
        )
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        policy_horizon = int(self.action_policy.training_action_horizon)
        if int(action.shape[1]) < policy_horizon:
            raise ValueError(
                f"Training chunk has {action.shape[1]} actions; regression requires {policy_horizon}."
            )
        action_target = action[:, :policy_horizon]
        action_is_pad = inputs["action_is_pad"]
        if action_is_pad is not None:
            action_is_pad = action_is_pad[:, :policy_horizon]

        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=input_latents.shape[0],
            device=self.device,
            dtype=video_supervision_latents.dtype,
        )
        video_targets = self._prepare_video_training_targets(
            video_supervision_latents=video_supervision_latents,
            timestep_video=timestep_video,
            first_frame_latents=inputs["first_frame_latents"],
        )
        pred_video = self._predict_video_only(
            latents_video=video_targets["latents_video"],
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_flag,
            apply_spatial_downsample=video_targets["apply_spatial_downsample"],
            restore_spatial_resolution=video_targets["restore_spatial_resolution"],
        )

        observation_latents = inputs["first_frame_latents"]
        if observation_latents is None:
            observation_latents = input_latents[:, :, 0:1]
        layer_states = self._build_action_layer_states(
            observation_latents=observation_latents,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        action_timing = self._timing_start()
        pred_action = self.action_policy(
            layer_states=layer_states,
            proprio=inputs["proprio"],
        )
        self._timing_end("model3_regression_action_policy", action_timing)

        include_initial_video_step = inputs["first_frame_latents"] is None
        target_video = video_targets["target_video"]
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        video_loss_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            device=video_loss_per_sample.device,
            dtype=video_loss_per_sample.dtype,
        )
        loss_video = (video_loss_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_regression_action_loss_per_sample(
            pred_action=pred_action,
            target_action=action_target,
            action_is_pad=action_is_pad,
        )
        loss_action = action_loss_per_sample.mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        metrics = self._build_loss_dict(loss_video=loss_video, loss_action=loss_action)
        self._timing_end("training_loss_total", total_timing)
        if self.enable_timing_breakdown:
            metrics.update(self._get_timing_breakdown_metrics())
        return loss_total, metrics

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 1,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        return super().infer_action(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
