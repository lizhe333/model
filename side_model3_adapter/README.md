# Side-Model3-Adapter v1

`side_model3_adapter/` 是 `side_model3/` 的直接代码副本，与 `side_model3/`、
`model3/` 平行放置。它实现 `specs/24-side-model3-adapter-v1.md`，目标是保留
Side-Model3 全部结构，只增加少量 Wan 内 residual adapter，提高适配模拟器、
双相机拼接和机器人视觉分布的能力。

## 直接复制边界

本包的主体代码直接照抄 `side_model3/`：

- 五阶段 Ladder Side Encoder；
- O2-style Trace Fusion；
- Visual Anchor Resampler 和 gated visual residual；
- Model3-style $16$ 层 Action-DiT；
- $h=4/8$ Action Chunk Encoder 和共享 Transition Predictor；
- Future Latent Change Head；
- raw-video $t/t+4/t+8$ 数据对齐；
- 五项损失、动作采样、checkpoint 和 trainer 结构；
- 原有 focused tests。

这不是重新设计 Side-Model3。v1 只修改以下必要差异：

1. 方法、类、Hydra、checkpoint 和 evidence identity 独立；
2. Wan blocks $8/16/24$ 启用 vendored zero-init `ResidualAdapter`；
3. online current Wan forward 保留到 adapter 的梯度；
4. future target 使用 FP32 EMA adapter copies；
5. Side/Visual 读取 adapted Wan states 时不再切断 adapter 梯度。

`side_model3/` 本身没有被修改。
逐文件来源与必要差异记录在 `COPY_LINEAGE.md`。

## 在线路径

```text
current RGB + language + current proprioception
-> frozen Wan VAE
-> frozen Wan base parameters at timestep 0
-> online residual adapters after blocks 8/16/24
-> post-adapter H8/H16/H24 and propagated H20/H29
-> copied five-stage Ladder + Trace Fusion
-> 64 control slots
-> copied 16-anchor visual residual
-> copied 16-layer Action-DiT
-> 8 x 7 action chunk
```

每个 residual adapter 为：

```text
LayerNorm(3072)
-> Linear(3072, 256)
-> GELU
-> zero-init Linear(256, 3072)
-> residual add, scale 1.0
```

因此 adapter 初始化时严格等价于 identity。不实例化 Wan LoRA，不训练 Wan 原始
参数，不恢复 StateFusion、future-video head 或 future-video flow loss。
三个 production adapters 合计新增 $4{,}747{,}008$ 个 optimizer-trainable 参数；
FP32 EMA copies 参数量相同，但不进入 optimizer。

## Target 与 EMA

Online Predictive Encoder 包含 online Wan adapters 和复制的 Ladder/Trace。
Target 路径只复制这三个 adapters 与 Ladder/Trace，不复制完整 Wan。

训练中的 $t+4$、$t+8$ target 为：

```text
future RGB + same language
-> frozen VAE
-> frozen Wan blocks + FP32 EMA adapters
-> FP32 EMA Ladder/Trace
-> detached future control-state target
```

两组 target 都以 decay $0.996$ 在实际执行的 optimizer step 后更新一次。梯度累积
的 micro-batch 和 skipped step 不更新。

## 保持不变的损失和数据

总损失直接复制 Side-Model3：

$$
\mathcal L =
1.00\mathcal L_{\mathrm{action}}+
0.25\mathcal L_{\mathrm{state},4}+
0.50\mathcal L_{\mathrm{state},8}+
0.10\mathcal L_{\mathrm{latent},4}+
0.20\mathcal L_{\mathrm{latent},8}.
$$

训练仍要求 raw processed video。配置固定检查 `num_frames=33`、
`global_sample_stride=1`、`action_video_freq_ratio=4` 和 $224\times448$ 双相机
横向拼接，使 sampled positions $0/1/2$ 对齐 environment offsets $0/4/8$。
joint-video latent cache 不可用于本模型。

## 梯度边界

训练参数只有：

- online Wan residual adapters at $8/16/24$；
- copied online Ladder/Trace；
- copied Visual Anchor/Visual Fusion；
- copied Action-DiT；
- copied action chunk、transition 和 latent-change modules。

Wan 原始 blocks、VAE、text encoder、EMA adapters 和 EMA Ladder/Trace 全部无
optimizer state。未来 target forward 全程无梯度。

## 代码结构

- `models/side_model3_adapter_wam.py`：直接复制的主编排与 adapter 差异。
- `models/ladder_side_encoder.py`：复制的 Ladder；仅保留 adapter 输入梯度。
- `models/visual_anchor_resampler.py`：复制的视觉旁路；仅保留 adapter 输入梯度。
- `models/action_dit.py`：直接复制，架构不变。
- `models/latent_transition.py`：直接复制，架构不变。
- `models/future_latent_change_head.py`：直接复制，架构不变。
- `models/ema_target.py`：复制 Side EMA，并增加 EMA Wan adapter bank。
- `runtime.py`：复制 factory，固定三层 adapter 并继续拒绝 LoRA/StateFusion。
- `trainer.py`：复制 optimizer-step EMA trainer。
- `configs/hydra/model/side_model3_adapter_v1.yaml`：独立 Hydra identity。

## Hydra 与轻量检查

把 `side_model3_adapter/configs/hydra` 加入 Hydra search path，然后选择
`model=side_model3_adapter_v1`。本包不会向 vendored Light-WAM 配置目录写文件。

不加载权重的检查命令：

```bash
python -m side_model3_adapter.scripts.preflight --dry-run
```

## 当前证据状态

当前 v1 实现、25 项 focused tests、copy-equivalence、Hydra compose 和 dry-run
preflight 已通过。没有选择 LIBERO suite、训练预算、checkpoint selection 或闭环
gate，也没有 Side-Model3-Adapter 训练、保留 checkpoint、rollout 或性能证据。
adapter 是提高达到 Model3 水平概率的工程假设，不是已验证结论。
