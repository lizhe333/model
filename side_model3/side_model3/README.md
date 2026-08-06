# Side-Model3 v1

`side_model3/` 是与 `model3/` 平行的独立方法包，实现
`specs/23-side-model3-v1.md` 冻结的模型合同。它保留 Model3 的 $16$ 层
Action-DiT 动作流解码器，但把机器人表征和多时间尺度潜在动态学习全部移到
Wan 外部；当前没有选择 LIBERO suite、训练预算、checkpoint 或闭环门槛。

## 方法边界

在线路径只读取一个当前双视角观测：

```text
current RGB + language + current proprioception
-> frozen Wan VAE
-> frozen current-only Wan forward at timestep 0
-> raw H8/H16/H20/H24/H29 states
-> five-stage Ladder Side Encoder
-> identity-initialized trace fusion
-> 64 control slots
-> 16 visual anchors from H29
-> identity-initialized visual residual
-> Model3-style 16-layer Action-DiT
-> 8 x 7 action chunk
```

Wan2.1-T2V-1.3B、VAE 和文本编码器始终冻结。Wan 内不实例化 LoRA、WAM
residual adapter 或 future-video head；每次 Wan 前向都在无梯度上下文中运行，
读出的 hidden states 在进入侧路前停止梯度，侧路输出也不会写回 Wan。

训练额外把真实 $t+4$ 和 $t+8$ 观测分别当成新的单帧当前观测编码。EMA target
encoder 生成未来 control-state 目标，共享 transition predictor 分别读取动作前
$4$ 步和前 $8$ 步，并预测两个 horizon 的 future state 与固定 VAE latent change。
真实未来观测只构造停止梯度的 target，不进入在线动作路径。

EMA 不通过参数版本号或下一批 loss 推测优化器状态。正式训练集成必须使用
`SideModel3Trainer`（或显式调用 `register_ema_optimizer_hook`），使 target 只在
实际执行的 optimizer step 之后更新一次；梯度累积的中间 micro-batch 不更新。
EMA target 使用 FP32 参数与浮点 buffer，避免 decay 为 $0.996$ 时的微小更新被
BF16 量化吞掉。

总损失为

$$
\mathcal L =
1.00\mathcal L_{\mathrm{action}}+
0.25\mathcal L_{\mathrm{state},4}+
0.50\mathcal L_{\mathrm{state},8}+
0.10\mathcal L_{\mathrm{latent},4}+
0.20\mathcal L_{\mathrm{latent},8}.
$$

## 代码结构

- `models/side_model3_wam.py`：组织 frozen-Wan、动作、预测、EMA 和 checkpoint 路径。
- `models/ladder_side_encoder.py`：五阶段 Ladder 和 exact-final-state trace fusion。
- `models/visual_anchor_resampler.py`：H29 visual anchors 与动作侧 gated residual。
- `models/action_dit.py`：独立的 Model3 架构 Action-DiT。
- `models/latent_transition.py`：多 horizon 动作编码和共享 transition predictor。
- `models/future_latent_change_head.py`：固定 VAE 坐标系中的 latent-change 预测。
- `models/ema_target.py`：完整 online predictive encoder 的 EMA 副本。
- `runtime.py`：Hydra factory；拒绝任何 Wan adapter、LoRA 或 StateFusion 配置。
- `trainer.py`：把 FP32 EMA target 更新绑定到实际执行的 optimizer step。
- `config.py`、`contracts.py`：suite-neutral 方法配置和静态/实时模型合同检查。
- `configs/hydra/model/side_model3_v1.yaml`：继承 vendored Light-WAM 公共字段的独立模型配置。
- `launch.py`、`scripts/preflight.py`：只读 dry-run preflight，不启动训练。

## 张量与数据合同

- 输入为双相机拼接后的 RGB，形状为 $B\times3\times1\times224\times448$。
- 五阶段 Ladder trace 为 $B\times5\times64\times512$，部署 control state 为
  $B\times64\times512$。
- 第一个、第二个和第三个采样视频帧分别对应环境 offset $0$、$4$ 和 $8$；
  proprioception 使用相同 offset。
- 动作 target 只取前 $8$ 步，生产动作形状为 $B\times8\times7$。
- 训练必须读取 raw processed video；Model3 的 joint-video latent cache 不满足
  三个观测独立 VAE/Wan 编码的合同。

## Hydra factory

模型配置组名是 `side_model3_v1`，目标为
`side_model3.runtime.create_side_model3_wam`。配置显式将继承的 `wam_adapter` 和
`state_fusion_action_expert_config` 置空；factory 也会拒绝从调用方重新打开
adapter、LoRA 或 StateFusion。

与 vendored Light-WAM 的主配置组合时，需要把
`side_model3/configs/hydra` 加入 Hydra search path，再选择
`model=side_model3_v1`；本包不向 vendored 配置目录写入文件。

默认 $224\times448$ 输入经 Wan VAE 得到 $28\times56$ latent grid，固定
$2\times2$ average pooling 后，latent-change head 使用 $14\times28$ queries。
可选 prototype warm start 只能通过 `model3_action_dit_warmstart_path` 加载兼容的
Action-DiT 权重，不加载 Model3 Wan LoRA、adapter 或旧 query encoder。

训练接入使用 `SideModel3Trainer` 时会固定检查 `num_frames=33`、
`global_sample_stride=1`、`action_video_freq_ratio=4`、raw-video 输入以及
$224\times448$ 双相机横向拼接，确保采样位置 $0/1/2$ 实际对应环境 offset
$0/4/8$。本包仍不替用户选择 suite 或训练预算。

## 轻量检查

以下命令只检查方法合同与 Hydra wiring，不下载权重、不构造 Wan，也不创建 run：

```bash
python -m side_model3.scripts.preflight --dry-run
```

也可以让 `validate_contract(model=model)` 检查已经构造的实例，确认 live Wan/VAE
冻结、无 adapter/LoRA、EMA target 无梯度且 method/loss/layer identity 正确。

## 当前证据状态

当前 v1 实现、17 项 focused tests、Hydra compose 和 dry-run 方法检查已通过。
没有 Side-Model3 训练、保留 checkpoint、正式 rollout、闭环成功率或效率结果；
本包中的测试和 dry-run 不能作为性能证据。正式执行前仍需单独冻结 suite、
raw-video 数据路径、训练预算、评测合同和 evidence run。
