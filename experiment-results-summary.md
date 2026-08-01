# 实验结果总览

更新时间：2026-08-01。

本文只汇总 Light-WAM、Model3、Model3 O2 和 Model5。除“论文官方”一栏外，
LIBERO 结果均为本地评测：10 tasks × 50 trials，共 500 episodes。`B` 为单卡
batch size，`GA` 为 gradient accumulation，`H/R` 为动作输出长度/执行后重规划
步数，`S` 为 Action-DiT solver steps。

## 一页结果

### Light-WAM：论文官方与本地复现

| 来源 | Spatial | Object | Goal | Long | 平均 | 配置 |
|---|---:|---:|---:|---:|---:|---|
| Light-WAM 论文官方 | 98.2% | 99.6% | 97.8% | 93.0% | 97.2% | `P-LW` |
| 我们复现发布权重 | 483/500 (96.6%) | 497/500 (99.4%) | 477/500 (95.4%) | 461/500 (92.2%) | 95.9% | `E-LW` |

论文官方数值来自 [Light-WAM, arXiv:2606.08242](https://arxiv.org/abs/2606.08242)。
本地 Object 另做过 `H8/R8` 控制，结果为 494/500；原生 `H32/R10` 为
497/500。

### 我们的模型

| 模型 | Suite / checkpoint | 结果 | 配置 |
|---|---|---:|---|
| Model3 | Spatial 60K, S10 | 488/500 (97.6%) | `M3-Spatial` |
| Model3 | Object 10K / 15K / 20K, S10 | 221 / 433 / 440（各 500） | `M3-Object` |
| Model3 | Object 20K, S5 / S10 / S20 | 467 / 440 / 349（各 500） | `M3-Object`，仅改 S |
| Model3 | Goal 固定 60K, S10 | 473/500 (94.6%) | `M3-Goal` |
| Model3 | Goal 65K / 70K / 75K / 80K, S10 | 480 / 482 / 476 / 475（各 500） | 同 `M3-Goal`；70K 为后验曲线最佳点 |
| Model3 | Long 80K, S10 | 478/500 (95.6%) | `M3-Long` |
| Model3 O2 | Spatial local 5K / 10K, S10 | 481 / **489**（各 500） | `O2-Spatial` |
| Model3 O2 | Object local 10K / 20K / 35K, S10 | 442 / 464 / **492**（各 500） | `O2-Object` |
| Model3 O2 | Object local 35K, S5 | 489/500 (97.8%) | 同 `O2-Object`，仅改 S |
| Model3 O2 | Long local 5K / 10K, S10 | 436 / **476**（各 500） | `O2-Long` |
| Model5 | Object 10K / 15K / 20K, S10 | 400 / **466** / 459（各 500） | `M5-Object` |
| Model5 | Object 10K / 15K / 20K, S5 | 448 / **478** / 454（各 500） | 同 `M5-Object`，仅改 S |
| Model5 | Long | 150K 训练进行中，暂无闭环结果 | `M5-Long` |

## 配置索引

### `P-LW`：Light-WAM 论文官方

- Wan2.1-T2V-1.3B frozen backbone；全层 LoRA；WAM adapters 位于
  layers 8/16/24；每层 16 queries；StateFusion 直接回归动作。
- future-video latent 做 2× spatial downsampling。
- AdamW，LR `1e-4`，weight decay `1e-2`，cosine，warmup 1K；global batch
  64；4×H100。
- 论文选择的 checkpoint：Spatial/Goal 60K，Object 12.5K，Long 80K。

### `E-LW`：我们复现 Light-WAM 发布权重

- 不重新训练；使用发布 checkpoint 及其随附配置，seed 42，4×RTX 4090
  并行完成 500 episodes。
- 原生推理为 StateFusion 单次动作预测，`H32/R10`。
- checkpoint：Spatial 55K、Object 12.5K、Goal 60K、Long 80K。

### `M3-*`：Model3 共同配置

- Wan2.1-T2V-1.3B frozen base；30 层 rank-64 LoRA；adapters/hidden
  layers 8/16/24；64 recurrent queries；16-layer Action-DiT；video/action
  flow joint loss。
- 双相机 224×224，BF16，4×RTX 4090，global batch 64，warmup 1K，
  checkpoint 每 5K；`H8/R8/S10`（标注其他 S 的行除外）。

| 配置 | 与 `M3-*` 共同配置相比的 suite 参数 |
|---|---|
| `M3-Spatial` | B16/GA1，LR `2e-4`，训练 60K，episode limit 400 |
| `M3-Object` | B16/GA1，LR `1e-4`，保留到 20K，episode limit 400 |
| `M3-Goal` | B16/GA1，LR `2e-4`；固定比较点 60K，另评 65K–80K，episode limit 400 |
| `M3-Long` | B16/GA1，LR `1e-4`，训练 80K，episode limit 700 |

### `O2-*`：Model3 O2

- 同对应 `M3-*`，仅增加 layer-aware `q1/q2/q3` readout；Action-DiT、
  loss、`H8/R8/S10` 不变。
- 从对应 Model3 checkpoint 做 model-only warm start；optimizer、scheduler、
  dataloader 与 RNG 全部重新开始；B16/GA1，LR `1e-4`。

| 配置 | 唯一 suite 差异 |
|---|---|
| `O2-Spatial` | parent = Model3 Spatial 60K；O2-local 10K |
| `O2-Object` | parent = Model3 Object 20K；训练并保留到 O2-local 35K |
| `O2-Long` | parent = Model3 Long 80K；O2-local 10K |

### `M5-*`：Model5

- 同 Model3 query + Action-DiT 主体；action-feature branch 改为
  `[current clean latent, one Gaussian noisy future latent]`，显式 temporal
  timesteps `[0,1000]`；queries 读取 layers 8/16/24 的完整时序 tokens。
- B8/GA2，global batch 64，BF16，关闭 gradient checkpointing，4×RTX 4090，
  `H8/R8`。

| 配置 | 唯一 suite 差异 |
|---|---|
| `M5-Object` | LR `2e-4`，150K 预算；训练在完整 20K checkpoint 后暂停；S5 和 S10 均完成 3-checkpoint × 500 episodes |
| `M5-Long` | LR `1e-4`，150K 预算；fresh Wan-base initialization；训练进行中 |

### LRD-WAM：G1 通过，G2 未通过

LRD-WAM G0/G1 仅使用原始 pretrained Wan2.1-T2V-1.3B。Object 420 episodes
与 Long 300 episodes 的逐样本 rank-8 residual explained energy 均通过冻结门槛，
因此 G1 判定为 `pass_input_conditional_low_rank`；这只支持输入条件/任务条件低秩，
不支持统一全局机器人子空间。

随后冻结并执行的 Object+Long G2 覆盖 3 个 seed、18 个 dynamics、54 个 G2-A
probe 和 15 个 16-layer Action-DiT。G2-A 的 LR-P3 rank-8 在 Object 与 Long
都没有优于 frozen-base P0。G2-B 在 Long 上相对 P0 的 pooled 改善为 8.26%
（95% CI 6.23%–10.29%），相对 P5 为 6.39%（3.55%–9.30%）；但 Object
只有 0.74% 与 0.51%，区间均跨 0，也未达到预注册的 5% 门槛。Long D2 未通过
D1 非劣检验，Object 的 shuffled-delta 干预退化也未达到 2%。最终判定为
`fail_stop_before_g3`：保留 G1 表示层结论，拒绝“可部署且动作充分的 delta code”
主张，不启动 G3 或闭环评测。

完整中文报告：[`lrd-wam-g2-object-long-result.md`](lrd-wam-g2-object-long-result.md)。

## 证据边界

- Light-WAM 论文官方值是论文报告值，不是我们的本地复现。
- 本地 Light-WAM 使用官方发布权重；Model3/O2/Model5 使用本地训练权重。
- Model3 Spatial 60K 历史上通过严格评测，但成功 eval ledger 已删除，当前属于
  `recorded_not_locally_auditable`；其余表中已完成条件均有本地终端校验记录。
- Model3 Goal 70K 是查看 65K–80K 曲线后的 best observed，不替代固定 60K
  比较点。
- 不跨不同 checkpoint、solver 或训练路径直接宣称结构性优越；表格只陈述观测结果。

详细结果页：[`model3_o2/Spatial.md`](model3_o2/Spatial.md)、
[`model3_o2/Long.md`](model3_o2/Long.md)、[`model5/Object.md`](model5/Object.md)、
[`lrd-wam-g2-object-long-result.md`](lrd-wam-g2-object-long-result.md)。
