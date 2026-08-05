# Side-Model3 v1 模型构建说明

## 1. 文档目的

本文档定义接下来要构建的 `Side-Model3 v1`。内容只覆盖模型组件、信息流、初始化、训练目标、参数更新范围和部署路径，不包含实验矩阵、评价门槛或任务选择。

Side-Model3 v1 以 Model3 为直接父模型。它保留 Model3 的动作生成架构和训练范式，但完全移除 Wan 内部的机器人适配，将机器人动态学习迁移到 Wan 之外的侧路中。

## 2. 模型身份

Side-Model3 v1 的核心定义是：

> Wan 只提供当前观测的通用视频表征；机器人状态、动作条件动态和未来变化预测全部由独立侧路承担。

必须同时满足以下约束：

- Wan2.1-T2V-1.3B 的原始参数完全冻结；
- 不实例化 Wan LoRA；
- 不实例化 Wan residual WAM adapter；
- Wan 前向使用 `torch.no_grad()`；
- Wan hidden state 进入侧路前显式停止梯度；
- 侧路输出不会写回 Wan 的后续 block；
- Wan 只编码单个干净观测，不加入 noisy future slot；
- 动作推理不读取真实未来观测。

本版本不采用后半段 residual adapter、policy-owned noise canvas、direct-regression 动作头或 shuffled-action ranking loss。它们不是 Side-Model3 v1 的组成部分。

## 3. 整体结构

部署路径如下：

```text
当前双视角观测
    -> Frozen Wan VAE
    -> 单个 clean current latent
    -> Frozen Wan current-only forward
    -> H8/H16/H20/H24/H29
    -> 五阶段 Ladder Side Encoder
    -> O2-style Layer-Aware Trace Fusion
    -> 64 个 control-state slots

H29 current tokens
    -> Visual Anchor Resampler
    -> 16 个 visual anchor tokens
    -> Gated Visual Residual Fusion
    -> action-conditioned 64 slots

64 个 action-conditioned slots
    -> 原 Model3 16 层 Action-DiT
    -> 8-step action chunk
```

训练时额外建立两个时间尺度的潜在动态路径：

```text
当前 control-state + 前 4 步 expert action
    -> Latent Transition Predictor
    -> 预测 t+4 control-state
    -> 预测 t+4 低分辨率 VAE latent 变化

当前 control-state + 完整 8 步 expert action
    -> 同一个 Latent Transition Predictor
    -> 预测 t+8 control-state
    -> 预测 t+8 低分辨率 VAE latent 变化
```

真实 `t+4` 和 `t+8` 观测只用于构造停止梯度的训练目标。

## 4. 从 Model3 继承和删除的组件

### 4.1 直接继承

以下组件和数据契约保持与 Model3 一致：

- Wan2.1-T2V-1.3B、Wan VAE、文本编码器和 tokenizer；
- 双视角 `224 x 224` 图像预处理和相机拼接方式；
- 语言条件和本体状态输入；
- 8-step action chunk 和动作归一化约定；
- 16 层、hidden dim 512 的 Model3 Action-DiT；
- action flow-matching 目标、动作 scheduler 和采样过程；
- current-only 动作推理契约。

### 4.2 删除

以下 Model3 组件不进入 Side-Model3 v1：

- 30 层 rank-64 Wan LoRA；
- Wan 第 8、16、24 层的 residual WAM adapter；
- 读取 adapted/delta state 的旧输入契约；
- 通过可训练 Wan 完成的 future-video flow 分支；
- 任何穿过 Wan block 的反向传播路径。

### 4.3 新增或扩展

新增或扩展以下模块：

- 五阶段 Ladder Side Encoder；
- O2-style Layer-Aware Trace Fusion；
- Visual Anchor Resampler 和 Gated Visual Residual Fusion；
- Multi-Horizon Action Chunk Encoder；
- Latent Transition Predictor；
- EMA Target Predictive Encoder；
- Multi-Horizon Future Latent Change Head。

## 5. Current-Only Frozen Wan 路径

当前、`t+4` 和 `t+8` 观测都使用相同的 Frozen Wan 输入协议：

- 每次只输入一个单时刻双视角观测；
- VAE 输出只包含一个 clean observation latent；
- 视频特征时间步固定为 `0`；
- 使用 Model3 的 `first_frame_causal` 注意力约定；
- 不添加 Gaussian future slot；
- 不提供真实未来 token；
- 不执行 future denoising。

未来观测是在各自目标时刻被当作新的当前观测单独编码，而不是与 `t` 时刻观测共同输入 Wan。

第一版读取以下五个原始 backbone state：

```text
{8, 16, 20, 24, 29}
```

每个状态保留完整空间 token。每个读取位置配置独立的线性投影和 LayerNorm，将 Wan hidden dim 映射到统一的 512 维侧路空间。

读取接口只读不写。侧路状态不作为残差加入 Wan，也不成为任何后续 Wan block 的输入。

## 6. Ladder Side Encoder

### 6.1 Control-State 初始化

Side Encoder 维护一组共享的 control-state slots：

- slot 数量：64；
- hidden dim：512；
- 初始状态：可学习参数，按 batch 扩展；
- 本体状态：先经过无仿射 LayerNorm，再线性映射到 512 维，然后广播并加入全部 64 个初始 slots；
- 层位置：五个读取阶段分别具有一个 `[1, 1, 512]` 的可学习 layer embedding。

语言通过 Frozen Wan 的语言条件进入五层视频特征。第一版不建立独立的高带宽语言旁路。

### 6.2 五阶段更新

64 个 control-state slots 按顺序读取 `H8 -> H16 -> H20 -> H24 -> H29`。每个阶段只包含一个更新 block，配置固定为：

- hidden dim：512；
- cross-attention heads：8；
- query self-attention heads：8；
- FFN hidden dim：2048；
- activation：GELU；
- dropout：0；
- normalization：Pre-LayerNorm。

每个阶段执行：

1. 投影并归一化当前 Wan token；
2. 向 control-state slots 加入当前 layer embedding；
3. slots 通过 cross-attention 读取当前层空间 token；
4. slots 进行一次 self-attention；
5. slots 通过 FFN 完成非线性更新；
6. 将本阶段更新乘以一个可学习标量 residual gate 后加入输入 slots；
7. 保存当前阶段的 slot 快照。

每阶段 residual gate 的有效初值为 `0.1`，使所有读取模块从第一步开始获得梯度，同时限制随机初始化对状态的扰动。

### 6.3 O2-Style Trace Fusion

模型保留五个阶段的完整状态轨迹，而不是只使用最终状态。H29 阶段状态作为主状态；H8、H16、H20 和 H24 状态分别通过独立的 `512 -> 512` 投影形成 residual route。

每条早期 residual route 具有形状 `[1, 64, 1]` 的 query-wise gate。投影输出层和 gate 都零初始化，因此 Trace Fusion 初始时严格等价于只使用 H29 阶段状态。

融合后的 control-state 形状固定为：

```text
[batch, 64, 512]
```

该状态同时供动作分支和 Latent Transition Predictor 使用。

## 7. Visual Anchor 与动作侧视觉残差

64 个 control-state slots 主要承担控制抽象和动态预测。为保留物体纹理、边缘、姿态和精确空间位置，动作分支额外读取 H29 的原始当前帧特征。

### 7.1 Visual Anchor Resampler

Visual Anchor Resampler 的配置为：

- 16 个可学习 visual queries；
- hidden dim：512；
- cross-attention heads：8；
- self-attention heads：8；
- FFN hidden dim：2048；
- dropout：0；
- 输入：投影到 512 维的 H29 spatial tokens。

输出为：

```text
visual_anchors: [batch, 16, 512]
```

### 7.2 Gated Visual Residual Fusion

Visual anchors 不直接拼接进 Action-DiT context，避免零值 token 仍改变 cross-attention softmax 分母。

动作分支先执行一次独立的视觉 cross-attention：

```text
visual_residual = CrossAttention(
    query=control_state,
    key=visual_anchors,
    value=visual_anchors
)
```

该 cross-attention 使用 8 heads。其输出经过一个零初始化的 query-wise gate，再以残差方式加入 control-state：

```text
action_state = control_state + gated_visual_residual
```

因此初始化时 `action_state` 严格等于 `control_state`，视觉旁路不会改变原 Action-DiT 的 64-token context 结构。训练后，动作损失可以逐步打开视觉细节残差。

Visual anchors 和 visual residual 只服务动作分支。Latent Transition Predictor 不预测 visual anchors，也不把它们定义为潜在世界状态的一部分。

## 8. 动作生成分支

动作头保持原 Model3 Action-DiT，不替换为 direct regression：

- 16 个 Action-DiT blocks；
- hidden dim：512；
- attention heads：8；
- FFN hidden dim：2048；
- action horizon：8；
- noisy action token 作为 action-flow 输入；
- 使用原 Model3 时间步嵌入、动作位置编码、输出头和 flow scheduler；
- cross-attention context 为 64 个 `action_state` slots。

动作损失更新 Action-DiT、Online Ladder Side Encoder、Trace Fusion、Visual Anchor Resampler 和 Gated Visual Residual Fusion，但不会进入 Wan。

模型结构不依赖训练好的 Model3 checkpoint。默认使用与 Model3 相同的 Action-DiT 初始化。为了快速构建原型，可以只 warm-start Model3 Action-DiT 权重；Wan LoRA、WAM adapter 和旧 query encoder 权重不属于本模型。

## 9. Multi-Horizon Latent Transition

### 9.1 Action Chunk Encoder

完整 8-step expert action chunk 被编码为 8 个 512 维动作 token：

- 每一步动作通过共享线性层映射；
- 加入动作位置 embedding；
- 保留动作顺序，不提前池化；
- 加入 horizon embedding，区分 `h=4` 和 `h=8`。

`h=4` 分支读取前 4 个动作 token；`h=8` 分支读取全部 8 个动作 token。

### 9.2 Latent Transition Predictor

两个 horizon 共用一个 Transition Predictor。该模块包含两个顺序执行的 Transformer/Cross-Attention blocks，每个 block 配置为：

- hidden dim：512；
- action cross-attention heads：8；
- state self-attention heads：8；
- FFN hidden dim：2048；
- activation：GELU；
- dropout：0；
- normalization：Pre-LayerNorm。

每个 block 先让当前 control-state 读取动作 token，再执行 state self-attention 和 FFN。最终线性层输出 64 个未来状态残差，并与当前 control-state 相加。

输出包括：

```text
predicted_state_t4: [batch, 64, 512]
predicted_state_t8: [batch, 64, 512]
```

Transition Predictor 不读取真实未来观测，也不访问未来 Wan hidden state。

## 10. EMA Future-State Targets

### 10.1 Online 与 Target Predictive Encoder

`Online Predictive Encoder` 被明确定义为以下整体：

```text
Ladder Side Encoder + O2-Style Trace Fusion
```

`Target Predictive Encoder` 是 Online Predictive Encoder 的完整深拷贝，包含 Ladder 和 Trace Fusion，但不包含 Visual Anchor Resampler、视觉残差、Action-DiT 或 Transition Predictor。

Target Predictive Encoder：

- 初始化时完整复制 Online Predictive Encoder 参数和 buffer；
- 不加入优化器；
- 所有输出停止梯度；
- 每次 optimizer step 后以固定 `EMA decay = 0.996` 更新参数；
- 非浮点 buffer 直接从 Online 分支复制。

EMA 不复制 Wan。Online 和 Target 分支共享同一个完全冻结的 Wan。

### 10.2 未来目标构建

真实 `t+4` 和 `t+8` RGB 观测分别作为单时刻观测进入 Frozen Wan。对于当前 Model3 的动作/视频频率约定：

```text
video frame index 1 -> action offset 4
video frame index 2 -> action offset 8
```

目标路径为：

```text
future observation + same language + future proprioception
    -> Frozen Wan VAE
    -> single clean future latent
    -> Frozen Wan current-only forward at timestep 0
    -> H8/H16/H20/H24/H29
    -> Target Predictive Encoder
    -> target_state_t4 / target_state_t8
    -> stop-gradient
```

### 10.3 Future-State Matching

预测状态和目标状态分别经过无仿射 LayerNorm。每个 horizon 的状态匹配由两部分组成：

- 对 64 个对应 slots 计算平均 cosine distance；
- 对归一化后的对应 slots 计算 mean Smooth L1，系数为 `0.1`。

slot 顺序由 Online/Target 共享的初始 query 索引和相同 Ladder 更新顺序确定，不进行最优匹配或 slot permutation。

## 11. Multi-Horizon Future Latent Change Head

EMA 状态目标会随 Online Predictive Encoder 缓慢变化，因此模型同时保留一个固定 VAE 坐标系下的稠密未来目标。

### 11.1 固定目标

真实 `t+4` 和 `t+8` 观测经过冻结 Wan VAE，分别构造：

```text
latent_delta_t4 = latent_t4 - latent_t
latent_delta_t8 = latent_t8 - latent_t
```

空间下采样固定使用 `torch.nn.functional.avg_pool2d`，`kernel_size=2`、`stride=2`，只作用于 VAE latent 的 H/W 维度。目标不经过 EMA encoder，因此不会随侧路参数漂移。

### 11.2 预测头

两个 horizon 共用一个 Future Latent Change Head，并通过 horizon embedding 区分输出。该模块配置为：

- 与下采样后 latent 网格数量一致的可学习 spatial queries；
- hidden dim：512；
- cross-attention heads：8；
- self-attention heads：8；
- FFN hidden dim：2048；
- dropout：0；
- 二维位置 embedding；
- 输出通道等于 Wan VAE latent channel。

spatial queries 读取相应的 predicted future state，随后映射并 reshape 为低分辨率 latent change map。每个 horizon 使用 mean Smooth L1 与固定 latent-delta 目标匹配。

该预测头不恢复 RGB，也不执行视频扩散或未来去噪。

## 12. 训练目标

Side-Model3 v1 的默认总目标由五项组成：

```text
1.00 * action flow-matching
0.25 * t+4 future-state matching
0.50 * t+8 future-state matching
0.10 * t+4 latent-change prediction
0.20 * t+8 latent-change prediction
```

每项损失先在自己的有效 token、空间位置和 batch 维度上取 mean，再乘以上述固定系数。五项损失从训练开始同时启用，不使用 action-only warmup。

模型不包含 shuffled-action ranking loss，也不包含通过 Wan 计算的 future-video flow loss。

## 13. 训练信息流

每个训练 batch 按以下顺序执行：

1. 取得当前、`t+4`、`t+8` RGB 观测，语言，当前/未来本体状态和 8-step expert action；
2. 在 `no_grad` 下运行当前观测的 Frozen Wan；
3. Online Predictive Encoder 生成当前 control-state；
4. Visual Anchor Resampler 和 Gated Visual Residual Fusion 生成 action-state；
5. 原 Model3 Action-DiT 使用 64 个 action-state slots 计算 action flow prediction；
6. Action Chunk Encoder 分别构造前 4 步和完整 8 步动作条件；
7. Transition Predictor 生成 predicted state at `t+4` 和 `t+8`；
8. 在 `no_grad` 下分别运行真实 `t+4` 和 `t+8` 观测的 Frozen Wan；
9. Target Predictive Encoder 生成停止梯度的两个未来状态目标；
10. Future Latent Change Head 从两个预测未来状态解码低分辨率 VAE latent 变化；
11. 按固定系数组合五项损失并反向传播；
12. 优化器更新后，以 `0.996` EMA 更新 Target Predictive Encoder。

## 14. 参数和梯度契约

| 模块 | 训练时运行 | 优化器更新 | 部署保留 |
|---|---|---|---|
| Wan VAE | 是 | 否 | 是 |
| 文本编码器 | 是或使用缓存 | 否 | 是 |
| Wan2.1 Video-DiT | 是，`no_grad` | 否 | 是 |
| Wan LoRA | 不实例化 | 否 | 否 |
| Wan residual WAM adapter | 不实例化 | 否 | 否 |
| Online Ladder Side Encoder | 是 | 是 | 是 |
| O2-style Trace Fusion | 是 | 是 | 是 |
| Visual Anchor Resampler | 是 | 是 | 是 |
| Gated Visual Residual Fusion | 是 | 是 | 是 |
| Target Predictive Encoder | 是 | 否，仅 EMA 更新 | 否 |
| Multi-Horizon Action Chunk Encoder | 是 | 是 | 否 |
| Latent Transition Predictor | 是 | 是 | 否 |
| Future Latent Change Head | 是 | 是 | 否 |
| Model3 Action-DiT | 是 | 是 | 是 |

以下情况均违反 Side-Model3 v1 的模型定义：

- Wan 任意参数出现优化器更新；
- Wan block 上存在 LoRA 或 residual adapter；
- 侧路输出被写回 Wan；
- Wan 输入包含真实或 Gaussian future slot；
- 动作头读取真实未来观测；
- Target Predictive Encoder 通过反向传播更新；
- 推理时运行 transition 或 future-target 分支。

## 15. 推理路径

部署时只运行：

```text
当前观测、语言、当前 proprioception
    -> Frozen VAE
    -> single clean current latent
    -> Frozen Wan current-only forward at timestep 0
    -> H8/H16/H20/H24/H29
    -> Online Ladder Side Encoder
    -> O2-style Trace Fusion
    -> 64 control-state slots

H29 current tokens
    -> Visual Anchor Resampler
    -> Gated Visual Residual Fusion
    -> 64 action-state slots

64 action-state slots
    -> 原 Model3 Action-DiT sampling
    -> 8-step action chunk
```

推理时不运行：

- 真实未来观测编码；
- Target Predictive Encoder；
- Multi-Horizon Action Chunk Encoder；
- Latent Transition Predictor；
- Future Latent Change Head；
- future-video generation 或 future denoising。

## 16. 建议代码边界

实现放在独立的 `side_model3/` 包内，不直接改变已经验证的 `model3/`、`model3_o2/` 或 `model5/`：

```text
side_model3/
├── README.md
├── config.py
├── contracts.py
├── launch.py
├── runtime.py
├── configs/
└── models/
    ├── side_model3_wam.py
    ├── ladder_side_encoder.py
    ├── visual_anchor_resampler.py
    ├── latent_transition.py
    ├── future_latent_change_head.py
    └── ema_target.py
```

模块职责：

- `side_model3_wam.py`：组织训练与推理信息流，并复用 Model3 基础组件；
- `ladder_side_encoder.py`：完成五层 Wan 读取、64-slot 更新和 O2-style trace fusion；
- `visual_anchor_resampler.py`：产生 16 个 visual anchors，并将其作为 gated residual 注入动作状态；
- `latent_transition.py`：编码动作前缀并预测 `t+4/t+8` 未来状态；
- `future_latent_change_head.py`：预测固定 VAE 坐标系中的多时间尺度 latent 变化；
- `ema_target.py`：初始化和更新 Target Predictive Encoder，并实施 stop-gradient；
- `contracts.py`：检查 Wan 完全冻结、无 LoRA/adapter、无侧路回写、目标时间对齐和推理路径无未来输入。

实现可以复用：

- Model3 的数据输入、动作归一化、Action-DiT、flow scheduler 和动作采样；
- Model3 的 current-only Wan 前向和 selected-layer state 缓存接口；
- Model3 O2 的 layer-aware trace fusion 初始化思想。

Side-Model3 必须拥有独立的配置、method id、checkpoint identity 和输出目录，避免与已有模型权重或运行记录混用。

