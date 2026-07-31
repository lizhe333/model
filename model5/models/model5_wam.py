"""Model5 policy built on a clean, vendored Light-WAM baseline."""

from __future__ import annotations

from typing import Any, Optional

import torch

from model5.third_party.light_wam.src.lightwam.models.wan22.helpers.loader import (
    apply_video_backbone_preset,
    load_wan_video_components,
    resolve_video_backbone_type,
    sync_action_dit_config_with_video_backbone,
)
from model5.third_party.light_wam.src.lightwam.models.wan22.lightwam import (
    DisabledActionExpert,
    LightWAM,
)
from model5.third_party.light_wam.src.lightwam.models.wan22.mot import MoT
from model5.third_party.light_wam.src.lightwam.utils.logging_config import get_logger

from .vla_query_dit_action_expert import VLAQueryDiTActionExpert


logger = get_logger(__name__)


class Model5WAM(LightWAM):
    """Model3-derived query WAM with high-resolution noisy-future features."""

    method_id = VLAQueryDiTActionExpert.method_id
    SUPPORTED_ACTION_FEATURE_SCOPES = {
        "current_only",
        "current_plus_noisy_future",
    }

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        action_query_policy_config: dict[str, Any],
        action_feature_config: dict[str, Any],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        video_backbone_type: str = "wan2_2_ti2v",
        video_backbone_name: str | None = None,
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        video_latent_spatial_downsample_factor: int = 1,
        apply_video_latent_downsample_to_action_branch: bool = False,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        use_first_frame_residual_video_target: bool = False,
        action_temporal_weighting_enabled: bool = False,
        action_temporal_weighting_num_prefix_steps: Optional[int] = None,
        action_temporal_weighting_prefix_weight: float = 1.0,
        action_temporal_weighting_tail_weight: float = 1.0,
        wam_adapter: dict[str, Any] | None = None,
    ) -> "Model5WAM":
        if video_dit_config is None or "text_dim" not in video_dit_config:
            raise ValueError("Model5 requires `video_dit_config` with `text_dim`.")
        if not isinstance(action_query_policy_config, dict) or not action_query_policy_config:
            raise ValueError("Model5 requires a non-empty `action_query_policy_config`.")
        if not isinstance(action_feature_config, dict) or not action_feature_config:
            raise ValueError("Model5 requires a non-empty `action_feature_config`.")

        temporal_scope = str(action_feature_config.get("temporal_scope", "")).strip()
        if temporal_scope not in cls.SUPPORTED_ACTION_FEATURE_SCOPES:
            raise ValueError(
                "`action_feature_config.temporal_scope` must be one of "
                f"{sorted(cls.SUPPORTED_ACTION_FEATURE_SCOPES)}, got {temporal_scope!r}."
            )
        fixed_future_timestep = int(
            action_feature_config.get("fixed_future_timestep", video_num_train_timesteps)
        )
        if fixed_future_timestep != int(video_num_train_timesteps):
            raise ValueError(
                "Model5 fixes tau_f to the maximum Wan training timestep; "
                f"expected {video_num_train_timesteps}, got {fixed_future_timestep}."
            )
        num_future_latent_slots = int(
            action_feature_config.get("num_future_latent_slots", 8)
        )
        if num_future_latent_slots <= 0:
            raise ValueError("`num_future_latent_slots` must be positive.")
        feature_downsample_factor = int(
            action_feature_config.get(
                "spatial_downsample_factor",
                1,
            )
        )
        supported_feature_factors = {
            1,
            int(video_latent_spatial_downsample_factor),
        }
        if feature_downsample_factor not in supported_feature_factors:
            raise ValueError(
                "Model5 action-feature spatial factor must be high resolution "
                "(factor=1) or match the video branch for the separately named "
                "efficiency diagnostic. "
                f"got feature={feature_downsample_factor}."
            )

        resolved_backbone_type = resolve_video_backbone_type(video_backbone_type)
        video_dit_config = apply_video_backbone_preset(
            dit_config=dict(video_dit_config),
            video_backbone_type=resolved_backbone_type,
        )
        action_dit_config = sync_action_dit_config_with_video_backbone(
            action_dit_config={} if action_dit_config is None else dict(action_dit_config),
            video_dit_config=video_dit_config,
        )
        adapter_config = {} if wam_adapter is None else dict(wam_adapter)
        use_wam_adapter = bool(adapter_config.get("use_wam_adapter", False))
        remove_original_action_expert = bool(
            adapter_config.get("remove_original_action_expert", False)
        )
        if not use_wam_adapter or not remove_original_action_expert:
            raise ValueError(
                "Model5 requires `use_wam_adapter=true` and "
                "`remove_original_action_expert=true`."
            )

        video_dit_config["use_wam_adapter"] = True
        video_dit_config["adapter_layer_indices"] = adapter_config.get(
            "adapter_layer_indices"
        )
        video_dit_config["adapter_dim"] = int(adapter_config.get("adapter_dim", 128))
        video_dit_config["adapter_scale"] = float(adapter_config.get("adapter_scale", 1.0))

        use_backbone_lora = bool(adapter_config.get("use_backbone_lora", False))
        if use_backbone_lora:
            video_dit_config["use_backbone_lora"] = True
            video_dit_config["lora_layer_indices"] = adapter_config.get(
                "lora_layer_indices"
            )
            video_dit_config["lora_target_modules"] = adapter_config.get(
                "lora_target_modules",
                ["ffn.0", "ffn.2"],
            )
            video_dit_config["lora_rank"] = int(adapter_config.get("lora_rank", 16))
            video_dit_config["lora_alpha"] = float(adapter_config.get("lora_alpha", 16.0))
            video_dit_config["lora_dropout"] = float(
                adapter_config.get("lora_dropout", 0.0)
            )

        components = load_wan_video_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            video_backbone_type=resolved_backbone_type,
            video_backbone_name=video_backbone_name,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        adapter_layers = tuple(getattr(video_expert, "adapter_layer_indices", ()))
        if not adapter_layers:
            raise ValueError("Model5 requires at least one Wan adapter layer.")

        action_policy = VLAQueryDiTActionExpert(
            video_hidden_dim=int(video_dit_config["hidden_dim"]),
            action_dim=int(action_dit_config["action_dim"]),
            num_fusion_layers=len(adapter_layers),
            proprio_dim=proprio_dim,
            **dict(action_query_policy_config),
        ).to(device=device, dtype=torch_dtype)
        action_expert = DisabledActionExpert(
            action_dim=int(action_dit_config["action_dim"])
        )
        mot = MoT(
            mixtures={"video": video_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_backbone_type=resolved_backbone_type,
            video_latent_spatial_downsample_factor=video_latent_spatial_downsample_factor,
            apply_video_latent_downsample_to_action_branch=(
                apply_video_latent_downsample_to_action_branch
            ),
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            use_first_frame_residual_video_target=use_first_frame_residual_video_target,
            action_temporal_weighting_enabled=action_temporal_weighting_enabled,
            action_temporal_weighting_num_prefix_steps=(
                action_temporal_weighting_num_prefix_steps
            ),
            action_temporal_weighting_prefix_weight=action_temporal_weighting_prefix_weight,
            action_temporal_weighting_tail_weight=action_temporal_weighting_tail_weight,
            use_wam_adapter=use_wam_adapter,
            freeze_backbone=bool(adapter_config.get("freeze_backbone", True)),
            remove_original_action_expert=remove_original_action_expert,
            state_fusion_action_expert=action_policy,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": "NOT_INSTANTIATED_FOR_MODEL5",
            "action_dit_pretrained_path": action_dit_pretrained_path,
        }
        model.action_feature_temporal_scope = temporal_scope
        model.action_feature_fixed_future_timestep = fixed_future_timestep
        model.action_feature_num_future_latent_slots = num_future_latent_slots
        model.action_feature_spatial_downsample_factor = feature_downsample_factor
        model._last_action_feature_diagnostics: dict[str, Any] = {}
        return model

    @property
    def action_policy(self) -> VLAQueryDiTActionExpert:
        expert = self.state_fusion_action_expert
        if not isinstance(expert, VLAQueryDiTActionExpert):
            raise RuntimeError(
                "Model5WAM requires VLAQueryDiTActionExpert; "
                f"got {type(expert).__name__ if expert is not None else None}."
            )
        return expert

    def configure_trainable_modules(self):
        _ = self.action_policy
        return super().configure_trainable_modules()

    def build_inputs(self, sample, tiled: bool = False):
        inputs = super().build_inputs(sample, tiled=tiled)
        proprio = sample.get("proprio")
        if self.proprio_dim is None:
            inputs["proprio"] = None
        else:
            if proprio is None or proprio.ndim != 3:
                raise ValueError("Model5 requires `proprio` with shape [B, T, D].")
            inputs["proprio"] = proprio[:, 0].to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
        return inputs

    def action_feature_config_dict(self) -> dict[str, Any]:
        return {
            "temporal_scope": self.action_feature_temporal_scope,
            "fixed_future_timestep": self.action_feature_fixed_future_timestep,
            "num_future_latent_slots": self.action_feature_num_future_latent_slots,
            "spatial_downsample_factor": self.action_feature_spatial_downsample_factor,
        }

    def get_last_action_feature_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_action_feature_diagnostics)

    def gradient_smoke_summary(self) -> dict[str, float | bool]:
        group_patterns = {
            "query_encoder": ("state_fusion_action_expert.query_encoder",),
            "wan_adapter": ("video_expert.wam_adapters",),
            "wan_lora": ("lora_", ".lora"),
        }
        summary: dict[str, float | bool] = {}
        for group_name, patterns in group_patterns.items():
            squared_norm = 0.0
            tensors_with_grad = 0
            for name, parameter in self.named_parameters():
                if not parameter.requires_grad or not any(pattern in name for pattern in patterns):
                    continue
                if parameter.grad is None:
                    continue
                tensors_with_grad += 1
                squared_norm += float(parameter.grad.detach().float().square().sum().item())
            summary[f"gradient/{group_name}_has_grad"] = tensors_with_grad > 0
            summary[f"gradient/{group_name}_norm"] = squared_norm**0.5

        frozen_base_has_grad = any(
            parameter.grad is not None
            for _, parameter in self.video_expert.named_parameters()
            if not parameter.requires_grad
        )
        summary["gradient/frozen_wan_base_has_grad"] = frozen_base_has_grad
        return summary

    def _build_action_feature_latents(
        self,
        *,
        observation_latents: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        noise_device: Optional[torch.device | str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation_latents.ndim != 5 or int(observation_latents.shape[2]) != 1:
            raise ValueError(
                "Model5 action conditioning requires [B, C, 1, H, W] observation latents."
            )

        feature_factor = int(self.action_feature_spatial_downsample_factor)
        if feature_factor == 1:
            current_latents = observation_latents
        elif feature_factor == int(self.video_latent_spatial_downsample_factor):
            current_latents, _ = self._maybe_downsample_video_latents_for_backbone(
                observation_latents
            )
        else:
            raise ValueError(
                "Model5 action-feature spatial factor must be 1 or match the video "
                f"branch; got feature={feature_factor}, "
                f"video={self.video_latent_spatial_downsample_factor}."
            )
        future_slots = (
            0
            if self.action_feature_temporal_scope == "current_only"
            else self.action_feature_num_future_latent_slots
        )
        if future_slots:
            future_shape = (
                current_latents.shape[0],
                current_latents.shape[1],
                future_slots,
                current_latents.shape[3],
                current_latents.shape[4],
            )
            if generator is None:
                future_noise = torch.randn(
                    future_shape,
                    device=current_latents.device,
                    dtype=current_latents.dtype,
                )
            else:
                random_device = torch.device("cpu" if noise_device is None else noise_device)
                future_noise = torch.randn(
                    future_shape,
                    generator=generator,
                    device=random_device,
                    dtype=torch.float32,
                ).to(device=current_latents.device, dtype=current_latents.dtype)
            feature_latents = torch.cat([current_latents, future_noise], dim=2)
        else:
            feature_latents = current_latents

        timestep_video = torch.full(
            (feature_latents.shape[0],),
            float(self.action_feature_fixed_future_timestep),
            dtype=feature_latents.dtype,
            device=feature_latents.device,
        )
        slot_timesteps = torch.full(
            (feature_latents.shape[0], feature_latents.shape[2]),
            float(self.action_feature_fixed_future_timestep),
            dtype=feature_latents.dtype,
            device=feature_latents.device,
        )
        slot_timesteps[:, 0] = 0
        return feature_latents, timestep_video, slot_timesteps

    def _build_action_layer_states(
        self,
        *,
        observation_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        feature_generator: Optional[torch.Generator] = None,
        feature_noise_device: Optional[torch.device | str] = None,
    ) -> list[dict[str, Any]]:
        feature_latents, timestep_video, slot_timesteps = (
            self._build_action_feature_latents(
                observation_latents=observation_latents,
                generator=feature_generator,
                noise_device=feature_noise_device,
            )
        )
        video_pre, _ = self._build_video_pre(
            latents_video=feature_latents,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            apply_spatial_downsample=False,
        )
        backbone_timing = self._timing_start()
        _ = self.video_expert.forward_backbone(video_pre)
        self._timing_end("action_feature_backbone", backbone_timing)
        layer_states = self._build_multilayer_action_fusion_inputs()
        hidden_tokens = [
            int(layer_state[self.action_policy.query_encoder.feature_source].shape[1])
            for layer_state in layer_states
        ]
        self._last_action_feature_diagnostics = {
            "temporal_scope": self.action_feature_temporal_scope,
            "latent_shape": tuple(int(value) for value in feature_latents.shape),
            "latent_slots": int(feature_latents.shape[2]),
            "future_slots": int(feature_latents.shape[2]) - 1,
            "fixed_future_timestep": int(self.action_feature_fixed_future_timestep),
            "slot_timesteps": tuple(
                int(value) for value in slot_timesteps[0].detach().to("cpu").tolist()
            ),
            "hidden_tokens_per_layer": tuple(hidden_tokens),
        }
        return layer_states

    def training_loss(self, sample, tiled: bool = False):
        """Joint future-video flow and query-conditioned action-flow objective."""

        self._reset_timing_breakdown()
        total_timing = self._timing_start()
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        video_supervision_latents = self._build_video_training_supervision_latents(
            input_latents
        )
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        policy_horizon = int(self.action_policy.training_action_horizon)
        if int(action.shape[1]) < policy_horizon:
            raise ValueError(
                f"Training chunk has {action.shape[1]} actions; model5 requires {policy_horizon}."
            )
        action_target = action[:, :policy_horizon]
        action_is_pad = inputs["action_is_pad"]
        if action_is_pad is not None:
            action_is_pad = action_is_pad[:, :policy_horizon]

        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=video_supervision_latents.dtype,
        )
        #把首帧恢复成干净的latent并从预测目标中去除
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

        action_noise = torch.randn_like(action_target)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action_target.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action_target,
            action_noise,
            timestep_action,
        )
        target_action_velocity = self.train_action_scheduler.training_target(
            action_target,
            action_noise,
            timestep_action,
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
        pred_action_velocity = self.action_policy(
            layer_states=layer_states,
            noisy_action=noisy_action,
            timestep=timestep_action,
            proprio=inputs["proprio"],
        )
        self._timing_end("model5_action_policy", action_timing)

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

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action_velocity,
            target_action=target_action_velocity,
            action_is_pad=action_is_pad,
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        metrics = self._build_loss_dict(loss_video=loss_video, loss_action=loss_action)
        diagnostics = self.get_last_action_feature_diagnostics()
        metrics.update(
            {
                "feature/latent_slots": float(diagnostics["latent_slots"]),
                "feature/future_slots": float(diagnostics["future_slots"]),
                "feature/fixed_future_timestep": float(
                    diagnostics["fixed_future_timestep"]
                ),
                "feature/hidden_tokens": float(
                    diagnostics["hidden_tokens_per_layer"][0]
                ),
            }
        )
        if self.device.type == "cuda" and torch.cuda.is_available():
            metrics["feature/cuda_peak_memory_mb"] = float(
                torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
            )
        self._timing_end("training_loss_total", total_timing)
        if self.enable_timing_breakdown:
            metrics.update(self._get_timing_breakdown_metrics())
        return loss_total, metrics

    def _prepare_model5_context(
        self,
        *,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt == use_context:
            raise ValueError("Provide exactly one of prompt or context/context_mask.")
        if use_prompt:
            prepared_context, prepared_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("Both context and context_mask are required.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError("Context must be [B, L, D] with mask [B, L].")
            prepared_context = context.to(device=self.device, dtype=self.torch_dtype)
            prepared_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            prepared_context, prepared_mask = self._append_proprio_to_context(
                context=prepared_context,
                context_mask=prepared_mask,
                proprio=proprio,
            )
        return prepared_context, prepared_mask

    @torch.no_grad()
    def infer_action_from_latents(
        self,
        *,
        observation_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        if observation_latents.ndim != 5 or int(observation_latents.shape[2]) != 1:
            raise ValueError(
                "Model5 infer_action_from_latents expects [B,C,1,H,W] observation latents."
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError("Prepared context must be [B,L,D] with mask [B,L].")
        if int(context.shape[0]) != int(observation_latents.shape[0]):
            raise ValueError("Observation latent and context batch sizes must match.")

        feature_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(int(seed))
        )
        layer_states = self._build_action_layer_states(
            observation_latents=observation_latents,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
            feature_generator=feature_generator,
            feature_noise_device=rand_device,
        )
        action_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(int(seed) + 1)
        )
        action = self.action_policy.sample(
            layer_states=layer_states,
            action_horizon=action_horizon,
            scheduler=self.infer_action_scheduler,
            num_inference_steps=num_inference_steps,
            proprio=proprio,
            generator=action_generator,
            noise_device=rand_device,
            sigma_shift=sigma_shift,
        )
        return {"action": action.detach().to(device="cpu", dtype=torch.float32)}

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
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError("Model5 infer_action expects one RGB image [1, 3, H, W].")
        if input_image.shape[-2] % 16 or input_image.shape[-1] % 16:
            raise ValueError("Input image height and width must be multiples of 16.")

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("proprio was provided but this model has no proprio encoder.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.proprio_dim):
                raise ValueError(f"proprio must have shape [1, {self.proprio_dim}].")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        elif self.proprio_dim is not None:
            raise ValueError("Model5 requires proprio at inference time.")

        prepared_context, prepared_mask = self._prepare_model5_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        observation_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )
        result = self.infer_action_from_latents(
            observation_latents=observation_latents,
            context=prepared_context,
            context_mask=prepared_mask,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            proprio=proprio,
            seed=seed,
            rand_device=rand_device,
            sigma_shift=sigma_shift,
        )
        return {"action": result["action"][0]}

    def log_parameter_summary(self):
        action_policy = self.action_policy
        self.state_fusion_action_expert = None
        try:
            super().log_parameter_summary()
        finally:
            self.state_fusion_action_expert = action_policy
        query_encoder = action_policy.query_encoder
        policy_total, policy_trainable = self._count_module_parameters(action_policy)
        logger.info(
            "Model5 action policy: queries=%s query_dim=%s heads=%s "
            "bridge_depth=%s feature_source=%s horizon=%s total=%s trainable=%s",
            query_encoder.num_action_queries,
            query_encoder.query_dim,
            query_encoder.num_heads,
            query_encoder.bridge_depth,
            query_encoder.feature_source,
            action_policy.action_horizon,
            self._format_param_count(policy_total),
            self._format_param_count(policy_trainable),
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "method_id": self.method_id,
            "model_class": type(self).__name__,
            "mot": self.mot.state_dict(),
            "action_policy_state_dict": self.action_policy.state_dict(),
            "action_policy_config": self.action_policy.config_dict(),
            "action_feature_config": self.action_feature_config_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        checkpoint_method = payload.get("method_id")
        if checkpoint_method != self.method_id:
            raise ValueError(
                f"Model5 checkpoint method_id must be {self.method_id!r}, "
                f"got {checkpoint_method!r} from {path}."
            )
        checkpoint_feature_config = payload.get("action_feature_config")
        if checkpoint_feature_config != self.action_feature_config_dict():
            raise ValueError(
                "Model5 checkpoint action_feature_config mismatch: "
                f"expected {self.action_feature_config_dict()}, "
                f"got {checkpoint_feature_config}."
            )
        if payload.get("model_class") != type(self).__name__:
            raise ValueError(
                f"Model5 checkpoint model_class must be {type(self).__name__!r}, "
                f"got {payload.get('model_class')!r}."
            )
        if "mot" not in payload or not isinstance(payload["mot"], dict):
            raise ValueError(f"Model5 checkpoint is missing a valid `mot` state: {path}")
        mot_state = self.mot.state_dict()
        incompatible = self.mot.load_state_dict(payload["mot"], strict=False)
        self._log_load_state_dict_result(
            module_name="mot",
            model_state_dict=mot_state,
            checkpoint_state_dict=payload["mot"],
            incompatible_keys=incompatible,
        )

        checkpoint_policy_config = payload.get("action_policy_config")
        if not isinstance(checkpoint_policy_config, dict):
            raise ValueError(f"Model5 checkpoint is missing `action_policy_config`: {path}")
        if checkpoint_policy_config.get("method_id") != self.method_id:
            raise ValueError("Model5 action policy config has a mismatched method_id")
        policy_state = payload.get("action_policy_state_dict")
        if not isinstance(policy_state, dict):
            raise ValueError(f"Model5 checkpoint is missing `action_policy_state_dict`: {path}")
        self.action_policy.load_state_dict(policy_state, strict=True)

        if self.proprio_encoder is not None:
            proprio_state = payload.get("proprio_encoder")
            if not isinstance(proprio_state, dict):
                raise ValueError(f"Model5 checkpoint is missing `proprio_encoder`: {path}")
            self.proprio_encoder.load_state_dict(proprio_state, strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
