"""Direct Side-Model3 copy with three trainable in-Wan residual adapters."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from model3.third_party.light_wam.src.lightwam.models.wan22.helpers.loader import (
    apply_video_backbone_preset,
    load_wan_video_components,
    resolve_video_backbone_type,
    sync_action_dit_config_with_video_backbone,
)
from model3.third_party.light_wam.src.lightwam.models.wan22.lightwam import (
    DisabledActionExpert,
    LightWAM,
)
from model3.third_party.light_wam.src.lightwam.models.wan22.mot import MoT
from model3.third_party.light_wam.src.lightwam.utils.logging_config import get_logger

from .action_dit import SideModel3ActionDiT
from .ema_target import EMATargetPredictiveEncoder, EMATargetWanAdapters
from .future_latent_change_head import MultiHorizonFutureLatentChangeHead
from .ladder_side_encoder import LadderSideEncoder
from .latent_transition import LatentTransitionPredictor, MultiHorizonActionChunkEncoder
from .visual_anchor_resampler import VisualAnchorActionFusion, VisualAnchorResampler


logger = get_logger(__name__)


class SideModel3AdapterWAM(LightWAM):
    """Copied Side-Model3 policy with minimal simulator-domain Wan PEFT."""

    method_id = "side_model3_adapter_wan_residual_ladder_flow_v1"
    selected_wan_layers = (8, 16, 20, 24, 29)
    adapter_layer_indices = (8, 16, 24)
    adapter_dim = 256
    adapter_scale = 1.0
    target_action_offsets = (4, 8)

    def __init__(
        self,
        *,
        side_encoder: LadderSideEncoder,
        action_policy: SideModel3ActionDiT,
        visual_resampler: VisualAnchorResampler,
        visual_fusion: VisualAnchorActionFusion,
        action_chunk_encoder: MultiHorizonActionChunkEncoder,
        transition_predictor: LatentTransitionPredictor,
        latent_change_head: MultiHorizonFutureLatentChangeHead,
        ema_decay: float,
        proprio_dim: int,
        loss_weights: dict[str, float],
        **lightwam_kwargs: Any,
    ) -> None:
        super().__init__(proprio_dim=None, **lightwam_kwargs)
        self.proprio_dim = int(proprio_dim)
        self.proprio_encoder = None
        self.online_predictive_encoder = side_encoder
        self.target_predictive_encoder = EMATargetPredictiveEncoder(
            self.online_predictive_encoder,
            decay=ema_decay,
        )
        self.target_wan_adapters = EMATargetWanAdapters(
            self.video_expert.wam_adapters,
            decay=ema_decay,
        )
        self.action_policy = action_policy
        self.visual_resampler = visual_resampler
        self.visual_fusion = visual_fusion
        self.action_chunk_encoder = action_chunk_encoder
        self.transition_predictor = transition_predictor
        self.latent_change_head = latent_change_head
        self.loss_weights = {name: float(value) for name, value in loss_weights.items()}
        self.ema_update_count = 0
        self._ema_optimizer_hook_handle = None
        self.to(device=self.device, dtype=self.torch_dtype)
        self.target_predictive_encoder.float()
        self.target_wan_adapters.float()
        self.target_predictive_encoder.requires_grad_(False)
        self.target_predictive_encoder.eval()
        self.target_wan_adapters.requires_grad_(False)
        self.target_wan_adapters.eval()

    @classmethod
    def from_wan21_pretrained(
        cls,
        *,
        side_encoder_config: dict[str, Any],
        action_policy_config: dict[str, Any],
        visual_anchor_config: dict[str, Any],
        transition_config: dict[str, Any],
        latent_change_config: dict[str, Any],
        wam_adapter_config: dict[str, Any],
        loss_weights: dict[str, float],
        ema_decay: float = 0.996,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        video_backbone_type: str = "wan2_1_t2v",
        video_backbone_name: str | None = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 128,
        load_text_encoder: bool = True,
        proprio_dim: int = 8,
        redirect_common_files: bool = False,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = False,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
    ) -> "SideModel3AdapterWAM":
        if video_dit_config is None or "text_dim" not in video_dit_config:
            raise ValueError("Side-Model3-Adapter requires video_dit_config with text_dim")
        resolved_backbone_type = resolve_video_backbone_type(video_backbone_type)
        video_config = apply_video_backbone_preset(
            dit_config=dict(video_dit_config),
            video_backbone_type=resolved_backbone_type,
        )
        adapter_config = dict(wam_adapter_config)
        adapter_layers = tuple(int(value) for value in adapter_config["adapter_layer_indices"])
        if adapter_layers != cls.adapter_layer_indices:
            raise ValueError("Side-Model3-Adapter requires Wan adapters at layers 8,16,24")
        if int(adapter_config["adapter_dim"]) != cls.adapter_dim:
            raise ValueError("Side-Model3-Adapter requires adapter_dim=256")
        if float(adapter_config["adapter_scale"]) != cls.adapter_scale:
            raise ValueError("Side-Model3-Adapter requires adapter_scale=1.0")
        video_config["use_wam_adapter"] = True
        video_config["adapter_layer_indices"] = list(adapter_layers)
        video_config["adapter_dim"] = cls.adapter_dim
        video_config["adapter_scale"] = cls.adapter_scale
        video_config["use_backbone_lora"] = False
        video_config["use_gradient_checkpointing"] = False
        action_config = sync_action_dit_config_with_video_backbone(
            action_dit_config={} if action_dit_config is None else dict(action_dit_config),
            video_dit_config=video_config,
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
            dit_config=video_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        if not bool(getattr(video_expert, "use_wam_adapter", False)):
            raise ValueError("Side-Model3-Adapter did not instantiate Wan adapters")
        if tuple(getattr(video_expert, "adapter_layer_indices", ())) != cls.adapter_layer_indices:
            raise ValueError("Side-Model3-Adapter Wan adapter layers changed")
        if bool(getattr(video_expert, "has_backbone_lora", lambda: False)()):
            raise ValueError("Side-Model3-Adapter must not instantiate Wan LoRA")
        if max(cls.selected_wan_layers) >= len(video_expert.blocks):
            raise ValueError("Wan backbone does not expose all Side-Model3 layers")

        video_hidden_dim = int(video_config["hidden_dim"])
        action_dim = int(action_config["action_dim"])
        side_config = dict(side_encoder_config)
        if tuple(side_config.get("layer_indices", cls.selected_wan_layers)) != cls.selected_wan_layers:
            raise ValueError("Side-Model3-Adapter v1 requires Wan layers 8,16,20,24,29")
        side_config.update(
            {
                "video_hidden_dim": video_hidden_dim,
                "proprio_dim": int(proprio_dim),
                "layer_indices": cls.selected_wan_layers,
            }
        )
        side_encoder = LadderSideEncoder(**side_config).to(
            device=device, dtype=torch_dtype
        )

        policy_config = dict(action_policy_config)
        policy_config.update(
            {
                "action_dim": action_dim,
                "context_dim": side_encoder.hidden_dim,
            }
        )
        action_policy = SideModel3ActionDiT(**policy_config).to(
            device=device, dtype=torch_dtype
        )

        visual_config = dict(visual_anchor_config)
        visual_resampler = VisualAnchorResampler(
            video_hidden_dim=video_hidden_dim,
            hidden_dim=side_encoder.hidden_dim,
            num_anchors=int(visual_config.get("num_anchors", 16)),
            num_heads=int(visual_config.get("num_heads", 8)),
            ffn_dim=int(visual_config.get("ffn_dim", 2048)),
        ).to(device=device, dtype=torch_dtype)
        visual_fusion = VisualAnchorActionFusion(
            num_slots=side_encoder.num_slots,
            hidden_dim=side_encoder.hidden_dim,
            num_heads=int(visual_config.get("num_heads", 8)),
        ).to(device=device, dtype=torch_dtype)

        transition_values = dict(transition_config)
        action_chunk_encoder = MultiHorizonActionChunkEncoder(
            action_dim=action_dim,
            hidden_dim=side_encoder.hidden_dim,
            max_horizon=action_policy.action_horizon,
        ).to(device=device, dtype=torch_dtype)
        transition_predictor = LatentTransitionPredictor(
            hidden_dim=side_encoder.hidden_dim,
            num_heads=int(transition_values.get("num_heads", 8)),
            ffn_dim=int(transition_values.get("ffn_dim", 2048)),
            num_blocks=int(transition_values.get("num_blocks", 2)),
        ).to(device=device, dtype=torch_dtype)

        latent_values = dict(latent_change_config)
        latent_channels = int(getattr(components.vae, "z_dim", components.vae.model.z_dim))
        latent_change_head = MultiHorizonFutureLatentChangeHead(
            latent_channels=latent_channels,
            grid_height=int(latent_values["grid_height"]),
            grid_width=int(latent_values["grid_width"]),
            hidden_dim=side_encoder.hidden_dim,
            num_heads=int(latent_values.get("num_heads", 8)),
            ffn_dim=int(latent_values.get("ffn_dim", 2048)),
        ).to(device=device, dtype=torch_dtype)

        disabled_action_expert = DisabledActionExpert(action_dim=action_dim)
        mot = MoT(
            mixtures={"video": video_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=disabled_action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_config["text_dim"]),
            device=device,
            torch_dtype=torch_dtype,
            video_backbone_type=resolved_backbone_type,
            video_latent_spatial_downsample_factor=1,
            apply_video_latent_downsample_to_action_branch=False,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=0.0,
            loss_lambda_action=float(loss_weights["action"]),
            use_first_frame_residual_video_target=False,
            action_temporal_weighting_enabled=False,
            use_wam_adapter=True,
            freeze_backbone=True,
            remove_original_action_expert=False,
            state_fusion_action_expert=None,
            side_encoder=side_encoder,
            action_policy=action_policy,
            visual_resampler=visual_resampler,
            visual_fusion=visual_fusion,
            action_chunk_encoder=action_chunk_encoder,
            transition_predictor=transition_predictor,
            latent_change_head=latent_change_head,
            ema_decay=ema_decay,
            proprio_dim=proprio_dim,
            loss_weights=loss_weights,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": "SIDE_MODEL3_ADAPTER_COPIED_ACTION_DIT",
        }
        return model

    @torch.no_grad()
    def update_ema_after_optimizer_step(self) -> None:
        self.target_predictive_encoder.update(self.online_predictive_encoder)
        self.target_wan_adapters.update(self.video_expert.wam_adapters)
        self.ema_update_count += 1

    def register_ema_optimizer_hook(self, optimizer: Any):
        """Update the target after each optimizer step that actually executes."""

        optimizer_with_hooks = getattr(optimizer, "optimizer", optimizer)
        self._ema_optimizer_hook_handle = optimizer_with_hooks.register_step_post_hook(
            lambda _optimizer, _args, _kwargs: self.update_ema_after_optimizer_step()
        )
        return self._ema_optimizer_hook_handle

    def configure_trainable_modules(self):
        self.eval()
        self.requires_grad_(False)
        trainable_modules = (
            self.online_predictive_encoder,
            self.action_policy,
            self.visual_resampler,
            self.visual_fusion,
            self.action_chunk_encoder,
            self.transition_predictor,
            self.latent_change_head,
        )
        for module in trainable_modules:
            module.train()
            module.requires_grad_(True)
        self.video_expert.eval()
        self.video_expert.wam_adapters.train()
        self.video_expert.wam_adapters.requires_grad_(True)
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.target_predictive_encoder.requires_grad_(False)
        self.target_predictive_encoder.eval()
        self.target_wan_adapters.requires_grad_(False)
        self.target_wan_adapters.eval()

    def _extract_adapted_wan_states(
        self,
        *,
        observation_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        use_target_adapters: bool,
    ) -> list[torch.Tensor]:
        if observation_latents.ndim != 5 or int(observation_latents.shape[2]) != 1:
            raise ValueError("Side-Model3-Adapter Wan input must be [B,C,1,H,W]")
        if not bool(getattr(self.video_expert, "use_wam_adapter", False)):
            raise RuntimeError("Side-Model3-Adapter requires Wan adapters at runtime")
        if bool(getattr(self.video_expert, "has_backbone_lora", lambda: False)()):
            raise RuntimeError("Side-Model3-Adapter must not instantiate Wan LoRA")

        timestep = torch.zeros(
            observation_latents.shape[0],
            device=observation_latents.device,
            dtype=observation_latents.dtype,
        )
        video_pre = self._build_action_observation_video_pre(
            observation_latents=observation_latents,
            timestep_video=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        tokens = video_pre["tokens"]
        self_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=tokens.device,
        )
        captured: dict[int, torch.Tensor] = {}
        for layer_index, block in enumerate(self.video_expert.blocks):
            tokens = block(
                tokens,
                video_pre["context"],
                video_pre["t_mod"],
                video_pre["freqs"],
                context_mask=video_pre["context_mask"],
                self_attn_mask=self_attention_mask,
            )
            if layer_index in self.adapter_layer_indices:
                if use_target_adapters:
                    tokens = self.target_wan_adapters.apply(layer_index, tokens)
                else:
                    tokens, _ = self.video_expert.wam_adapters[str(layer_index)](tokens)
            if layer_index in self.selected_wan_layers:
                captured[layer_index] = tokens
        states = [captured[index] for index in self.selected_wan_layers]
        if use_target_adapters:
            return [state.detach() for state in states]
        return states

    def build_inputs(self, sample, tiled: bool = False) -> dict[str, Any]:
        video = sample.get("video")
        if video is None:
            if sample.get("video_latents") is not None:
                raise ValueError(
                    "Side-Model3-Adapter requires raw video so t/t+4/t+8 are encoded independently"
                )
            raise ValueError("Side-Model3-Adapter training requires sample['video']")
        if video.ndim != 5 or int(video.shape[1]) != 3 or int(video.shape[2]) < 3:
            raise ValueError("video must be [B,3,T,H,W] with at least three sampled frames")
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        action = sample.get("action")
        proprio = sample.get("proprio")
        if context is None or context_mask is None or action is None or proprio is None:
            raise ValueError("Side-Model3-Adapter requires context, context_mask, action, and proprio")
        if action.ndim != 3 or int(action.shape[1]) < 8:
            raise ValueError("Side-Model3-Adapter requires at least eight expert actions")
        if proprio.ndim != 3 or int(proprio.shape[1]) <= max(self.target_action_offsets):
            raise ValueError("proprio must cover environment offsets 0,4,8")

        video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        observation_latents = []
        for frame_position in range(3):
            with torch.no_grad():
                observation_latents.append(
                    self._encode_video_latents(
                        video[:, :, frame_position : frame_position + 1],
                        tiled=tiled,
                    ).detach()
                )

        image_is_pad = sample.get("image_is_pad")
        future_valid = {
            4: torch.ones(video.shape[0], device=self.device, dtype=torch.bool),
            8: torch.ones(video.shape[0], device=self.device, dtype=torch.bool),
        }
        if image_is_pad is not None:
            if image_is_pad.ndim != 2 or int(image_is_pad.shape[1]) < 3:
                raise ValueError("image_is_pad must cover the first three sampled frames")
            future_valid = {
                4: ~image_is_pad[:, 1].to(device=self.device, dtype=torch.bool),
                8: ~image_is_pad[:, 2].to(device=self.device, dtype=torch.bool),
            }

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad[:, :8].to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
            for horizon in self.target_action_offsets:
                future_valid[horizon] = future_valid[horizon] & ~action_is_pad[
                    :, :horizon
                ].any(dim=1)
        proprio_is_pad = sample.get("proprio_is_pad")
        if proprio_is_pad is not None:
            proprio_is_pad = proprio_is_pad.to(device=self.device, dtype=torch.bool)
            for horizon in self.target_action_offsets:
                future_valid[horizon] = (
                    future_valid[horizon] & ~proprio_is_pad[:, horizon]
                )
        return {
            "context": context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            ),
            "context_mask": context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            ),
            "current_latents": observation_latents[0],
            "future_latents": {4: observation_latents[1], 8: observation_latents[2]},
            "proprio": {
                0: proprio[:, 0].to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                ),
                4: proprio[:, 4].to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                ),
                8: proprio[:, 8].to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                ),
            },
            "action": action[:, :8].to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            ),
            "action_is_pad": action_is_pad,
            "future_valid": future_valid,
        }

    @staticmethod
    def _masked_batch_mean(loss_per_sample: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
        if valid is None:
            return loss_per_sample.mean()
        weights = valid.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype)
        return (loss_per_sample * weights).sum() / weights.sum().clamp(min=1.0)

    @staticmethod
    def _future_state_loss_per_sample(
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        predicted_norm = F.layer_norm(predicted.float(), (predicted.shape[-1],))
        target_norm = F.layer_norm(target.float(), (target.shape[-1],))
        cosine = 1.0 - F.cosine_similarity(predicted_norm, target_norm, dim=-1)
        smooth = F.smooth_l1_loss(
            predicted_norm,
            target_norm,
            reduction="none",
        ).mean(dim=(1, 2))
        return cosine.mean(dim=1) + 0.1 * smooth

    @staticmethod
    def _latent_change_target(
        current_latents: torch.Tensor,
        future_latents: torch.Tensor,
    ) -> torch.Tensor:
        current = current_latents.squeeze(2)
        future = future_latents.squeeze(2)
        return F.avg_pool2d(future - current, kernel_size=2, stride=2)

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]

        current_wan_states = self._extract_adapted_wan_states(
            observation_latents=inputs["current_latents"],
            context=context,
            context_mask=context_mask,
            use_target_adapters=False,
        )
        control_state, _ = self.online_predictive_encoder(
            current_wan_states,
            inputs["proprio"][0],
        )
        visual_anchors = self.visual_resampler(current_wan_states[-1])
        action_state = self.visual_fusion(control_state, visual_anchors)

        action_noise = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=action.shape[0],
            device=action.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action,
            action_noise,
            timestep_action,
        )
        target_action_velocity = self.train_action_scheduler.training_target(
            action,
            action_noise,
            timestep_action,
        )
        predicted_action_velocity = self.action_policy(
            action_state,
            noisy_action,
            timestep_action,
        )
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=predicted_action_velocity,
            target_action=target_action_velocity,
            action_is_pad=inputs["action_is_pad"],
        )
        action_weight = self.train_action_scheduler.training_weight(
            timestep_action
        ).to(device=action.device, dtype=action_loss_per_sample.dtype)
        loss_action = (action_loss_per_sample * action_weight).mean()

        losses: dict[str, torch.Tensor] = {"action": loss_action}
        for horizon in self.target_action_offsets:
            action_tokens = self.action_chunk_encoder(action, horizon=horizon)
            predicted_state = self.transition_predictor(control_state, action_tokens)
            with torch.no_grad():
                future_wan_states = self._extract_adapted_wan_states(
                    observation_latents=inputs["future_latents"][horizon],
                    context=context,
                    context_mask=context_mask,
                    use_target_adapters=True,
                )
                target_state, _ = self.target_predictive_encoder(
                    future_wan_states,
                    inputs["proprio"][horizon],
                )
            state_per_sample = self._future_state_loss_per_sample(
                predicted_state,
                target_state.detach(),
            )
            losses[f"state_{horizon}"] = self._masked_batch_mean(
                state_per_sample,
                inputs["future_valid"][horizon],
            )

            predicted_latent_change = self.latent_change_head(
                predicted_state,
                horizon=horizon,
            )
            target_latent_change = self._latent_change_target(
                inputs["current_latents"],
                inputs["future_latents"][horizon],
            ).detach()
            if predicted_latent_change.shape != target_latent_change.shape:
                raise ValueError(
                    "latent-change head grid does not match pooled VAE target: "
                    f"{tuple(predicted_latent_change.shape)} vs "
                    f"{tuple(target_latent_change.shape)}"
                )
            latent_per_sample = F.smooth_l1_loss(
                predicted_latent_change.float(),
                target_latent_change.float(),
                reduction="none",
            ).mean(dim=(1, 2, 3))
            losses[f"latent_{horizon}"] = self._masked_batch_mean(
                latent_per_sample,
                inputs["future_valid"][horizon],
            )

        total = sum(self.loss_weights[name] * value for name, value in losses.items())
        metrics = {
            f"loss_{name}": float(value.detach().item())
            for name, value in losses.items()
        }
        metrics["loss_total"] = float(total.detach().item())
        metrics["ema_updates"] = float(self.ema_update_count)
        return total, metrics

    def _prepare_side_context(
        self,
        *,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (prompt is None) == (context is None and context_mask is None):
            raise ValueError("provide exactly one of prompt or context/context_mask")
        if prompt is not None:
            return self.encode_prompt(prompt)
        if context is None or context_mask is None:
            raise ValueError("context and context_mask are both required")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        return (
            context.to(device=self.device, dtype=self.torch_dtype),
            context_mask.to(device=self.device, dtype=torch.bool),
        )

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
        if proprio is None:
            raise ValueError("Side-Model3-Adapter requires current proprioception")
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if tuple(proprio.shape) != (1, self.proprio_dim):
            raise ValueError(f"proprio must be [1,{self.proprio_dim}]")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or tuple(input_image.shape[:2]) != (1, 3):
            raise ValueError("input_image must be one RGB image [1,3,H,W]")
        prepared_context, prepared_mask = self._prepare_side_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
        )
        observation_latents = self._encode_input_image_latents_tensor(
            input_image=input_image.to(device=self.device, dtype=self.torch_dtype),
            tiled=tiled,
        )
        wan_states = self._extract_adapted_wan_states(
            observation_latents=observation_latents,
            context=prepared_context,
            context_mask=prepared_mask,
            use_target_adapters=False,
        )
        control_state, _ = self.online_predictive_encoder(
            wan_states,
            proprio.to(device=self.device, dtype=self.torch_dtype),
        )
        anchors = self.visual_resampler(wan_states[-1])
        action_state = self.visual_fusion(control_state, anchors)
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        action = self.action_policy.sample(
            action_state,
            action_horizon=action_horizon,
            scheduler=self.infer_action_scheduler,
            num_inference_steps=num_inference_steps,
            generator=generator,
            noise_device=rand_device,
            sigma_shift=sigma_shift,
        )
        return {"action": action[0].detach().to(device="cpu", dtype=torch.float32)}

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "selected_wan_layers": list(self.selected_wan_layers),
            "adapter_layer_indices": list(self.video_expert.adapter_layer_indices),
            "adapter_dim": int(self.video_expert.adapter_dim),
            "adapter_scale": float(self.video_expert.adapter_scale),
            "side_encoder": self.online_predictive_encoder.config_dict(),
            "action_policy": self.action_policy.config_dict(),
            "ema_decay": self.target_predictive_encoder.decay,
            "loss_weights": self.loss_weights,
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "method_id": self.method_id,
            "model_class": type(self).__name__,
            "step": step,
            "mot": self.mot.state_dict(),
            "online_wan_adapters": self.video_expert.wam_adapters.state_dict(),
            "target_wan_adapters": self.target_wan_adapters.state_dict(),
            "online_predictive_encoder": self.online_predictive_encoder.state_dict(),
            "target_predictive_encoder": self.target_predictive_encoder.state_dict(),
            "action_policy": self.action_policy.state_dict(),
            "visual_resampler": self.visual_resampler.state_dict(),
            "visual_fusion": self.visual_fusion.state_dict(),
            "action_chunk_encoder": self.action_chunk_encoder.state_dict(),
            "transition_predictor": self.transition_predictor.state_dict(),
            "latent_change_head": self.latent_change_head.state_dict(),
            "checkpoint_config": self.checkpoint_config(),
            "ema_update_count": self.ema_update_count,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if payload.get("method_id") != self.method_id:
            raise ValueError("Side-Model3-Adapter checkpoint method identity mismatch")
        if payload.get("model_class") != type(self).__name__:
            raise ValueError("Side-Model3-Adapter checkpoint class identity mismatch")
        expected_config = self.checkpoint_config()
        if payload.get("checkpoint_config") != expected_config:
            raise ValueError("Side-Model3-Adapter checkpoint tensor contract mismatch")
        self.mot.load_state_dict(payload["mot"], strict=True)
        self.video_expert.wam_adapters.load_state_dict(
            payload["online_wan_adapters"], strict=True
        )
        self.target_wan_adapters.load_state_dict(
            payload["target_wan_adapters"], strict=True
        )
        self.online_predictive_encoder.load_state_dict(
            payload["online_predictive_encoder"], strict=True
        )
        self.target_predictive_encoder.load_state_dict(
            payload["target_predictive_encoder"], strict=True
        )
        self.action_policy.load_state_dict(payload["action_policy"], strict=True)
        self.visual_resampler.load_state_dict(payload["visual_resampler"], strict=True)
        self.visual_fusion.load_state_dict(payload["visual_fusion"], strict=True)
        self.action_chunk_encoder.load_state_dict(
            payload["action_chunk_encoder"], strict=True
        )
        self.transition_predictor.load_state_dict(
            payload["transition_predictor"], strict=True
        )
        self.latent_change_head.load_state_dict(
            payload["latent_change_head"], strict=True
        )
        self.ema_update_count = int(payload.get("ema_update_count", 0))
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def load_model3_action_dit_warmstart(self, path) -> None:
        payload = torch.load(path, map_location="cpu")
        expected_method = "model3_vla_recurrent_query_flow_v1"
        if payload.get("method_id") != expected_method:
            raise ValueError("Action-DiT warm start must be a Model3 checkpoint")
        if payload.get("model_class") != "Model3WAM":
            raise ValueError("Action-DiT warm start must come from Model3WAM")
        policy_config = payload.get("action_policy_config")
        if not isinstance(policy_config, dict) or policy_config.get("method_id") != expected_method:
            raise ValueError("Model3 Action-DiT warm start has a mismatched policy identity")
        policy_state = payload.get("action_policy_state_dict")
        if not isinstance(policy_state, dict):
            raise ValueError("Model3 warm start has no action policy state")
        self.action_policy.load_model3_action_dit_state(policy_state)

    def log_parameter_summary(self):
        module_names = (
            "online_predictive_encoder",
            "target_predictive_encoder",
            "target_wan_adapters",
            "visual_resampler",
            "visual_fusion",
            "action_chunk_encoder",
            "transition_predictor",
            "latent_change_head",
            "action_policy",
            "video_expert",
        )
        for name in module_names:
            total, trainable = self._count_module_parameters(getattr(self, name))
            logger.info(
                "Side-Model3-Adapter module=%s total=%s trainable=%s",
                name,
                self._format_param_count(total),
                self._format_param_count(trainable),
            )
