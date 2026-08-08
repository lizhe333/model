# Side-Model3-Adapter-v2

`side_model3_adapter_v2/` 是 `side_model3_adapter/` 的直接代码副本。它实现
`specs/29-side-model3-adapter-v2-direct-wm-action.md`，保留 v1 的完整模型、参数、
五项原始损失和推理路径，只增加训练期 direct WM-to-action future-state auxiliary。

## 直接复制边界

本包的主体代码直接复制 `side_model3_adapter/`：

- 五阶段 Ladder Side Encoder；
- O2-style Trace Fusion；
- Visual Anchor Resampler 和 gated visual residual；
- Model3-style $16$ 层 Action-DiT；
- $h=4/8$ Action Chunk Encoder 和共享 Transition Predictor；
- Future Latent Change Head；
- independent-observation cached $t/t+4/t+8$ 数据对齐；
- 五项损失、动作采样、checkpoint 和 trainer 结构；
- 原有 focused tests。

这不是重新设计 Side-Model3。v2 只增加以下必要差异：

1. 方法、类、Hydra、checkpoint 和 evidence identity 独立；
2. Action-DiT velocity 按 flow 公式重建 predicted clean action；
3. predicted action 经 branch-local detached-parameter $E/T$ 到 EMA future state；
4. 仅新分支使用逐样本 $w(\sigma)$ masked reduction，并按 $0\to0.1$ warmup；
5. v1 的 Wan adapters、Side/Visual 路径、五项原始 loss 和 inference 不变。

`side_model3_adapter/` 本身没有被修改。
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
cached future VAE latent + same language
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

训练复用 Side-Model3 的 independent-observation VAE cache。该 cache 位于 Wan
adapter 之前，保存三次独立单帧 VAE 编码后的 $[C_v,3,H_v,W_v]$，因此不会共享任何
Base 或 Adapter 模型状态。配置固定检查 `num_frames=33`、
`global_sample_stride=1`、`action_video_freq_ratio=4` 和 $224\times448$ 双相机
横向拼接，使 sampled positions $0/1/2$ 对齐 environment offsets $0/4/8$。
Model3 joint-video cache 不可用于本模型。

## 梯度边界

训练参数只有：

- online Wan residual adapters at $8/16/24$；
- copied online Ladder/Trace；
- copied Visual Anchor/Visual Fusion；
- copied Action-DiT；
- copied action chunk、transition 和 latent-change modules。

Wan 原始 blocks、VAE、text encoder、EMA adapters 和 EMA Ladder/Trace 全部无
optimizer state。未来 target forward 全程无梯度。

新 predicted-action 分支使用 `torch.func.functional_call` 配合 detached $E/T$
参数映射，保留 $\partial L_{\mathrm{dyn-action}}/\partial\widehat a_0$；不得使用
`torch.no_grad()` 或全局切换 `requires_grad`。预测 clean action 不新增 clamp。

## v2 训练路径

```text
noisy expert action + flow timestep
-> existing Action-DiT velocity
-> predicted clean action
-> functional detached-parameter Action Chunk Encoder
-> functional detached-parameter Transition Predictor
-> EMA future-state target
-> weighted masked L_dyn-action
-> Action-DiT and its conditioning route
```

GT action branch 仍正常训练 Action Chunk Encoder / Transition Predictor；新的
branch 不累积它们的参数梯度。推理完全不调用这条路径。

## 代码结构

- `models/side_model3_adapter_v2_wam.py`：v2 主编排与 direct WM-to-action 分支。
- `models/ladder_side_encoder.py`：复制的 Ladder；仅保留 adapter 输入梯度。
- `models/visual_anchor_resampler.py`：复制的视觉旁路；仅保留 adapter 输入梯度。
- `models/action_dit.py`：直接复制，架构不变。
- `models/latent_transition.py`：直接复制，架构不变。
- `models/future_latent_change_head.py`：直接复制，架构不变。
- `models/ema_target.py`：复制 Side EMA，并增加 EMA Wan adapter bank。
- `runtime.py`：复制 factory，固定三层 adapter 并继续拒绝 LoRA/StateFusion。
- `data.py`：复制 Side cache reader，严格读取共享的 pre-Wan independent cache。
- `trainer.py`：复制并同步 DeepSpeed optimizer-step EMA trainer。
- `backend_train.py`、`scripts/train_backend.sh`、`scripts/train_object.sh`：复制
  Side-Model3 的 cached Object backend 和训练入口。
- `scripts/run_object_pilot.sh`：先执行真实 $B=16$ smoke，OOM 时回退到
  $B=8$/GA-$2$，然后启动对应的 $40\text{K}$ Object run。
- `configs/hydra/model/side_model3_adapter_v2.yaml`：独立 Hydra identity。

## Hydra 与轻量检查

把 `side_model3_adapter_v2/configs/hydra` 加入 Hydra search path，然后选择
`model=side_model3_adapter_v2`。本包不会向 vendored Light-WAM 配置目录写文件。

不加载权重的检查命令：

```bash
python -m side_model3_adapter_v2.scripts.preflight --dry-run
```

v2 首个执行合同是 `specs/29-side-model3-adapter-v2-direct-wm-action.md` 中的
Long matched C0/C1 local-$10\text{K}$ trial。加载 v1 Long step-$90\text{K}$ parent
时，通过 `V1_WARMSTART_PATH` 显式传入模型-only warm start：

```bash
V1_WARMSTART_PATH=<v1-step-90000-checkpoint> \\
  bash side_model3_adapter_v2/scripts/train_long.sh
```

## 当前证据状态

当前 v2 direct implementation、focused tests、shared-cache strict validation、
Hydra compose 和 dry-run preflight 已通过。没有 v2 Object/Long checkpoint、训练或
rollout；v1 的 rollout summary 仍属于 parent evidence，不能归因给 v2。

在实现 v2 之前，Long step-$90\text{K}$ checkpoint 完成了 $512$ 样本的
action-dependence gate。GT state loss 在 $h=4/8$ 为 $0.005180/0.005369$；shuffle
action 分别升至 $13.338/27.630$ 倍，zero action 分别升至 $6.568/17.033$ 倍。
决策是 `sufficient_action_dependence_for_v2_bridge`：现有 Transition Predictor
明确依赖 action，不是只读 current state 的 future predictor。诊断脚本为
`scripts/eval_transition_action_dependence.py`，合同和边界位于
`specs/28-side-model3-adapter-v2-action-dependence-gate.md`。

## LIBERO Long Failure Analysis

Long rollout 的离线 failure analysis 位于
`side_model3_adapter_v2/scripts/analyze_long_failures.py`。它只读取 evaluator 已保存的
`success=False` MP4，不会改训练代码、checkpoint 或 success metric。每个 failure
生成首尾帧加 $16$ 个均匀内部采样帧的 timestamped storyboard，并写入
`failure_manifest.jsonl` 和 `vlm_requests.jsonl`。

VLM integration 使用一个小的进程协议：`--vlm-command` 指向的命令从 stdin 读取一条
request JSON（含 `image_path`、中文 prompt 和 JSON schema），并在 stdout 返回一个
JSON object。也可以用 `--analysis-jsonl` 回灌已有的 structured output。pipeline 会
校验 taxonomy、task-specific `furthest_stage`、confidence 和 episode identity，然后写出
`failure_analysis.jsonl`、`failure_analysis.csv`、`failure_summary.csv` 及低于
$0.75$ 的 `needs_manual_review.csv`。此外，`furthest_stage=无法判断`、
`primary_failure=其他/无法判断` 或 evidence 出现不确定措辞时也会自动标记人工复核。

所有 Long task 的 goal predicate 和允许 progress state 位于
`configs/libero_long_stage_rules.json`。它把 BDDL goal 当作终态 conjunction；两个独立
物体的完成顺序不是 stage error。第一批三个 video 的 dry run 已保存为 run artifact；
全量 failure batch 仍需用户确认后才运行。后续 simulator-ground-truth logging 的现况与
缺口见 `docs/libero_long_state_logging_design.md`。

## LIBERO State-Trace Failure Analysis

新启动的 Long eval 会为每个 episode 保存 simulator-state JSON。对这类新证据，运行
task-aware deterministic analyzer：

```bash
conda run --no-capture-output -n lightwam-libero-eval \
  python -m side_model3_adapter.scripts.analyze_libero_state_failures \
  --state-dir <eval_output>/libero_10/simulator_states \
  --checkpoint-step 75000 \
  --output-dir <eval_output>/state_failure_analysis
```

它只分析 `success=false` trace，并以 LIBERO 原生 predicate 为放置、关 drawer/door 和
stove 状态的真值。Task $3$/$9$ 的放置后关闭前置关系、无序双物体任务，以及 Task $7$ 的
配置化 distractor 都来自 `configs/libero_long_stage_rules.json`。结果包含逐 episode 的
`state_failure_analysis.jsonl/csv`、failure/rule summary、manual-review 表和可复现的
阈值文件。`grasp_failed` 与 `grasp_alignment_failure` 是动作和轨迹代理，不是接触传感器
结论。Task $8$ 若失去初始 stove-on predicate 则直接保留为机构交互失败证据。

历史 $75\text{K}/80\text{K}$ Long 输出没有 state trace，不能用此命令事后补出原因；要
获得该层诊断，必须在已启用 recorder 的新 output 目录重新 eval。
