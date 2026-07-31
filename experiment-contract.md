# Efficient Video-WM-to-WAM Adaptation

## Experiment Contract

> 执行优先级：**O2 → D → L → C → B → minimal A/R sanity**。
> 研究叙事见 [research-proposal.md](research-proposal.md)，证据边界见
> [evidence-and-related-work.md](evidence-and-related-work.md)。

## 0. Status and Authority

- 当前状态：**Model3 O2 主线已获用户批准，Matrix D 为第一执行矩阵**。
- 正式 carrier `C* = model3_o2_layer_aware_query_flow`。
- G0 的 carrier-selection 已完成；Long 对 parent 的 `δ=2%` non-inferiority
  certification 仍等待 paired CI，但不阻塞 Matrix D。
- 历史 Model3/Regression/O2 数字是 evidence anchors，不能替代新的 matched controls。

## 1. G0: Fixed O2 Carrier

### 1.1 Frozen Carrier Record

| 字段 | 当前冻结内容 | 状态 |
|---|---|---|
| Carrier | `model3_o2_layer_aware_query_flow` | 已选定 |
| Video backbone | `Wan-AI/Wan2.1-T2V-1.3B`，frozen base + all-layer rank-64 LoRA/adapters | 已记录 |
| Interface | layers 8/16/24 recurrent `q1/q2/q3` + layer-separable gated residual readout，exact-q3 identity init | 已记录 |
| Action carrier | 16-layer Action-DiT，flow-matching action objective，solver 10 | 已记录 |
| Action/replanning | policy horizon 8，executed/replan 8；video/action horizon 32 | 已记录 |
| Long initialization | strict model-only warm start from Model3 Long 80K，parent SHA `65680089...1d68`；fresh optimizer/scheduler/dataloader/sampler/RNG | 已记录 |
| Long training | 4×RTX 4090，B16/GA1，global batch 64，BF16，10K O2-local steps | 已记录 |
| Long evaluation | 10 tasks × 50 trials，seed 42，H8/R8，solver 10，700-step limit | 已记录 |
| Selected Long checkpoint | O2 10K，SHA `9653d5c5...8375f`，476/500 (95.2%) | 已验证 |
| Parent comparison | Model3 Long 80K，478/500 (95.6%)；same 500 task/trial identities | 已验证 |
| Spatial initialization | strict model-only warm start from Model3 Spatial 60K，parent SHA `67ccb8f4...1cf9b`；fresh state | 已记录 |
| Spatial evaluation | 10 tasks × 50 trials，seed 42，H8/R8，solver 10，400-step limit | 已记录 |
| Selected Spatial checkpoint | O2 10K，SHA `45e01052...415b`，489/500 (97.8%) | 已验证 |
| Spatial parent comparison | Model3 Spatial 60K，488/500 (97.6%)；parent ledger 已删除，不能 paired test | 历史 aggregate only |
| Evidence | [`model3_o2/Long.md`](model3_o2/Long.md)，[`model3_o2/Spatial.md`](model3_o2/Spatial.md) | source mirror 已登记 |

SHA 在表中为可读性缩写；执行时必须使用对应结果页和 config 中的完整值。

### 1.2 Open Preflight Fields

首次 server run 前补齐：

- server absolute repo path、backend commit SHA 与 dirty status；
- evaluator commit、action normalization 与 gripper convention；
- Matrix D 的 measured cost unit、checkpoint set 与 training-seed policy；
- O2 Long 与 parent 的逐 task、逐 initial-state paired outcomes，用于 paired CI。

### 1.3 Carrier Decision

O2 Object 35K 达到 492/500，O2 Long 10K 达到 476/500，O2 Spatial 10K 达到
489/500。Long 在相同 500 个 identities 上接近 parent 478/500；Spatial 与历史 parent
488/500 的 aggregate 差为 +1 success，但 parent ledger 已删除。该证据支持
`C* = Model3 O2` 与跨 suite portability，不支持 Long 或 Spatial improvement。
Regression 与历史 Model3 退为 reference tracks。

## 2. Common Contract and Primary Estimands

除声明 treatment 外，D/L/C/B/A/R 共享：

- O2 pretrained base、initialization、action carrier 与 solver；
- action horizon、executed prefix、replanning cadence、normalization 与 gripper convention；
- data split、windowing、episode grouping、task weighting 与 evaluator；
- GPU type、precision、batch contract、checkpoint rule 与 formal initial-state IDs；
- parameters、compute、memory、throughput、latency 与逐 episode outcomes 的记录格式。

| Estimand | Matched contrast | 回答什么 | 不能回答什么 |
|---|---|---|---|
| E-Temporal | D2 vs D1 | noisy future slots 是否有闭环增益 | 哪些深度提供信息 |
| E-Depth | L8/L16/L24 | tested depths 中控制信息在哪里可读 | 多层应怎样组合 |
| E-Composition | C0/C1/C2/C3 | selected depths 应怎样组合 | 哪些 PEFT blocks 必须更新 |
| E-Localized-Capacity | B0/B1/B1.5/B2 | 只适配 selected blocks 是否足够 | 连续空间绝对 minimum |
| E-A/R-Sanity | A-S pair、R-S pair | 主结论是否依赖 objective/routing | 完整 schedule/routing 最优解 |

任何 variant 若改变两个以上 estimand 字段，必须拆分 treatment，不能继续称为 matched pair。

## 3. Matrix D: Temporal Canvas

Matrix D 使用 registered O2 layer-aware interface，并作为第一个正式实验矩阵。

| ID | Action-feature input | 角色 |
|---|---|---|
| D0 | clean current latent，历史 O2/Model3 contract | historical reference only |
| D1 | clean current latent，与 D2 相同 timestep/position/code path | matched current-only control |
| D2 | D1 + policy-owned Gaussian-noise future slots | temporal-canvas treatment |

### 3.1 Matched Controls

- D1/D2 使用相同 O2 PEFT、layers 8/16/24 readout、action decoder、loss、训练预算与 solver；
- future slots 不读取 expert future video/action/reward/simulator state；
- D2 只运行一次 Wan，不做 iterative video denoising；
- 固定 environment initial state 与 action-solver RNG，并建立
  `episode_id -> feature_noise_seed` 确定性映射；
- 使用额外 noise-seed mappings 做敏感性分析；
- 报告额外 latent tokens、peak memory、throughput 与 plan-call latency。

### 3.2 Decision

- D1 对 D2 non-inferior：后续 L/C/B 固定 current-only D1；
- D2 显著提高闭环且收益覆盖成本：后续矩阵固定 D2；
- D2 收益不稳定或只改善 offline loss：关闭 temporal-canvas 路线，选择 D1。

## 4. Matrix L: Control-Depth Localization

固定 Matrix D 胜出 temporal contract、O2 all-layer PEFT 和 action carrier。L 只改变读取深度。

| ID | Hidden-state depth | Readout |
|---|---:|---|
| L8 | 8 | matched single-depth readout |
| L16 | 16 | matched single-depth readout |
| L24 | 24 | matched single-depth readout |
| L-O2 | 8/16/24 | registered multi-depth O2 reference |

### 4.1 Matched Controls

- L8/L16/L24 的 query 数、hidden width、projection depth、output tokens、action head、
  trainable parameters、steps 与 optimizer 相同；
- 不在 L 中改变 aggregation family；每个 single-depth variant 使用同一种 readout；
- primary evidence 是 paired closed-loop success 与 task-wise discordant outcomes；
- action-chunk recovery、progress/contact probes 只解释信息类型，不替代闭环；
- 第一轮只测试 O2 注册的 8/16/24。没有清晰结果时最多增加一个预声明
  early/mid/late bracket，不做 30-layer 无边界 sweep。

### 4.2 Output Contract

Matrix L 输出：

- best tested single depth `l*`；
- 各 depth 相对 L-O2 的 paired success difference；
- task-level complementarity map；
- 进入 Matrix C 的冻结 depth set `S_L`。

不能仅因某层 probe 最好就纳入 `S_L`；它必须具有闭环可读性或与 `l*` 存在预声明的
task-level complementarity。

## 5. Matrix C: Selected-Depth Composition

Matrix C 固定 `S_L`，只改变 aggregation/readout。

| ID | Depth set | Aggregation/readout | 作用 |
|---|---|---|---|
| C0 | `{l*}` | matched single-depth readout | best-single baseline |
| C1 | `S_L` | parameter-matched simple/shared pooling | 测试无需层身份的组合 |
| C2 | `S_L` | layer-separable compact composition | 测试层身份收益 |
| C3 | `S_L` | O2-style gated residual readout | high-capacity aggregation reference |

### 5.1 Controls and Decisions

- 固定 D contract、O2 PEFT、action decoder、query 数、hidden width、output tokens、steps；
- C0/C1/C2 参数量必须匹配，或提供明确 parameter-matched control；
- 报告 interface FLOPs、读取 tokens、memory 与 latency；
- C0 non-inferior：选择 single-depth interface；
- C1/C2 优于 C0：支持 depth complementarity；
- C2 优于 C1：支持保留 layer identity；
- C3 不优于 compact C2：复杂 gated residual 不是必要条件。

Matrix C 不允许重新选择 depths；若结果要求改变 `S_L`，返回 Matrix L 并记录一次 protocol
amendment，不能在同一 C sweep 中 post-hoc 调整。

## 6. Matrix B: Selected-Layer PEFT

固定 D winner、`S_L` 与 C winner 后，才改变 Video-DiT adaptation scope。

| ID | Video-DiT adaptation | 解释 |
|---|---|---|
| B0 | Wan base/PEFT 全冻结，只训练 interface/head | no-PEFT control |
| B1 | 仅适配 `S_L` 对应 blocks | selected-layer candidate |
| B1.5 | B1 + 一个预声明相邻 block bracket；仅当 B1 失败时启用 | 定位边界 |
| B2 | O2 all-layer rank-64 PEFT | performance/cost upper anchor |

### 6.1 Mapping and Decision Rules

- `depth -> adapted block` mapping 在运行 B 前冻结；B1 不得更新未声明 blocks；
- B1.5 的邻域方向与 block IDs 预声明，只允许一次；
- B0 对 B2 non-inferior：称 B0 为 **cheapest tested configuration**；
- B0 失败而 B1 non-inferior：称 B1 为 **smallest tested selected-layer PEFT**；
- B1.5 成功报告边界 `(B1, B1.5]`，失败报告 `(B1.5, B2]`；
- 每个候选报告 trainable parameters、optimizer states、memory、GPU hours、throughput 与
  closed-loop success。

### 6.2 Mandatory PEFT × Interface Check

| | C winner | O2-style C3 |
|---|---:|---:|
| cheapest viable B | ✓ | ✓ |
| all-layer B2 | ✓ | ✓ |

若存在明显补偿关系，报告联合 Pareto，不单独宣称 B-small 或 C-compact 足够。

## 7. Minimal Matrix A/R Sanity Checks

A/R 在 D/L/C/B 冻结后运行，不再承担 discovery priority。

### 7.1 A-Sanity

| ID | Objective | PEFT route |
|---|---|---|
| A-S0 | registered `L_video + L_action` | final B route |
| A-S1 | action-only | final B route |

- 固定 D/L/C/B、initialization、action updates、batch contract 与 checkpoint rule；
- 单列 video forward/backward 成本；
- 不加入 warmup variants，不恢复 A0–A3 schedule grid。

### 7.2 R-Sanity

| ID | Video loss → selected PEFT | Action loss → selected PEFT | Interface/head receives action loss |
|---|---:|---:|---:|
| R-S0 | 是 | 是 | 是 |
| R-S1 | 是 | 否 | 是 |

- forward graph、loss scalars、steps 与 optimizer 保持一致，只改变 action-loss detach/mask；
- 若 B0 胜出、WM PEFT 冻结，R sanity 标记为 not applicable；
- gradient norm/cosine 只作解释，不替代闭环；
- sanity pair 若不反转 D/L/C/B 结论，停止 A/R；若反转，只扩大 claim conditioning，是否
  新开完整 A/R matrix 需要单独批准。

## 8. Evaluation and Statistics

### 8.1 Endpoints and Contrasts

- Primary endpoint：paired closed-loop task success；
- Primary contrasts：D2 vs D1；L8/L16/L24；C winner vs C0/C3；B1/B0 vs B2；
- A-S0/A-S1 与 R-S0/R-S1 是 secondary sanity contrasts；
- offline losses、probes、gradient statistics 与 telemetry 不能替代闭环。

### 8.2 Non-Inferiority

- `δ = 2.0` percentage points 已对 O2 Long comparison 冻结，不得 post-hoc 修改；
- 每个新 compact contrast 在结果前预声明适用 margin；默认沿用 `δ=2 pp` 时必须明确记录；
- 报告 `Δ = p_candidate - p_reference` 的 95% paired CI；
- 只有 CI 下界 `> -δ` 才判定 non-inferior；
- exact McNemar test 可辅助 paired difference/superiority，不替代 non-inferiority。

#### O2 Long vs Parent Record

| Quantity | Value |
|---|---:|
| O2 Long 10K | 476/500 (95.2%) |
| Model3 Long 80K parent | 478/500 (95.6%) |
| Observed paired difference | `-0.4 pp` |
| Both succeed / O2 only / parent only / both fail | 459 / 17 / 19 / 5 |
| Exact two-sided McNemar `p` | `0.8679394004284404` |

当前可以写“未检测到显著差异，且表现出较强 portability”，不能写“已通过 `δ=2%`
non-inferiority”。正式判定仍需逐 task paired outcomes 的 stratified bootstrap CI；当前
source mirror 只有汇总 2×2 表，故状态为 **pending**。

### 8.3 Confidence and Randomness

- paired CI 按 task 分层，并保持相同 initial-state pair；
- 区分 rollout、feature-noise 与 training-seed variance；
- primary D/L/C/B 结论至少需要两个 training seeds，或一个 seed 加第二 suite 的一致方向；
- 同一次训练的 checkpoints 不算独立重复；
- checkpoint winner 标记为 `best observed on a predeclared checkpoint set`。

### 8.4 Formal Evaluation

- 每个 suite：10 tasks × 50 trials；主结论至少覆盖 Long 与 Object；
- 固定 evaluator commit、initial-state IDs、task order 与 terminal validation；
- 保留逐 episode outcome、checkpoint SHA、config 与 command；
- telemetry 可记录接近、抓取、抬起、运输、释放和 timeout，但不得输入策略。

## 9. Cost Accounting

| 成本 | 必须报告 |
|---|---|
| Training | GPU type/count、accelerator-hours、steps、samples、profiler FLOPs（若可用） |
| WM compute | Wan forwards/backwards、latent tokens、layers |
| Memory | peak allocated/reserved、optimizer-state memory |
| Parameters | base、PEFT、interface、action head 分项 |
| Deployment | forwards/chunk、solver steps、plan-call latency、throughput |

“相同 steps”不能称为 compute-matched；不同 temporal grid、depth set 或 interface path 必须
使用实测 accelerator-hours、throughput 或 FLOPs 核算。

## 10. Go/No-Go Gates

### G0：O2 Fixed Carrier

- O2 已冻结；Long paired CI 作为并行 paper-claim certification，不阻塞 D。

### G1：Matrix D

- 先决定 D1 current-only 或 D2 temporal canvas；
- 没有闭环收益或成本不合算时选择 D1，禁止用 offline loss 保留 D2。

### G2：Matrix L

- 在 D winner 上运行 L8/L16/L24；
- 输出 `l*`、`S_L` 与 task-level complementarity；
- 没有清晰结果时最多一次 depth bracket，不做全层 sweep。

### G3：Matrix C

- 固定 `S_L` 后比较 C0/C1/C2/C3；
- C 中不重新选择 layers。

### G4：Matrix B

- 固定 D/L/C 后运行 B0/B1/B2；B1 失败只允许 B1.5；
- 完成 PEFT × interface 2×2。

### G5：Minimal A/R Sanity

- 只运行 A-S0/A-S1 与适用时的 R-S0/R-S1；
- sanity 不反转主结论即停止，不扩展完整 A/R grid。

### G6：Generalization

- recipe 冻结后覆盖 Long 与 Object，再选择 Spatial 或 Goal；
- LIBERO-Plus/RoboTwin 不用于高频搜索。

## 11. Execution Approval and Artifacts

用户已批准 O2 主线及上述优先级。每次 server launch 前，handoff packet 必须包含：

- 当前 Gate、唯一 treatment diff 与前序 frozen decisions；
- config diff、tensor/loss/gradient/optimizer path；
- 预算、早停规则、checkpoint set 与 evaluation command；
- 输出目录、artifact naming、日志与逐 episode outcomes；
- 任何超出 D/L/C/B/minimal A/R 的新 scope 的单独批准。

当前最近一步是：**补齐 Matrix D preflight，执行 D1/D2；paired CI retrieval 与 D 并行，
不改变主实验优先级。**
