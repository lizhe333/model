# Efficient Video-WM-to-WAM Adaptation

## Experiment Contract

> 用途：执行、复现与 Go/No-Go 决策。研究叙事见
> [research-proposal.md](research-proposal.md)，文献与历史证据见
> [evidence-and-related-work.md](evidence-and-related-work.md)。

## 0. Status and Authority

- 当前状态：**Model3 O2 主线已获用户批准，进入 G1 执行准备**。
- 正式 carrier `C*` 已选为 `model3_o2_layer_aware_query_flow`；Long 与 Object 的正式
  500-episode 结果已进入本地 source mirror。
- G0 的 carrier-selection 部分已经完成；Long 对 parent 的 `δ=2%` non-inferiority
  certification 仍等待合同规定的 paired CI。
- 历史结果只能作为 evidence/diagnostic anchor，不能自动替代新的 matched control。

## 1. G0 Carrier Freeze

### 1.1 Frozen O2 Carrier Record

| 字段 | 当前冻结内容 | 状态 |
|---|---|---|
| Carrier | `model3_o2_layer_aware_query_flow` | 已选定 |
| Video backbone | `Wan-AI/Wan2.1-T2V-1.3B`，frozen base + all-layer rank-64 LoRA/adapters | 已记录 |
| Interface | layers 8/16/24 recurrent `q1/q2/q3` + layer-separable gated residual readout，exact-q3 identity init | 已记录 |
| Action carrier | 16-layer Action-DiT，flow-matching action objective，solver 10 | 已记录 |
| Action/replanning | policy horizon 8，executed/replan 8；video/action horizon 32 | 已记录 |
| Long initialization | strict model-only warm start from Model3 Long 80K，parent SHA `65680089...1d68`；fresh optimizer/scheduler/dataloader/sampler/RNG | 已记录 |
| Long training | 4×RTX 4090，B16/GA1，global batch 64，BF16，10K O2-local steps | 已记录 |
| Long evaluation | LIBERO Long，10 tasks × 50 trials，seed 42，H8/R8，solver 10，700-step limit | 已记录 |
| Selected Long checkpoint | O2 10K，SHA `9653d5c5...8375f`，476/500 (95.2%) | 已验证 |
| Parent comparison | Model3 Long 80K，478/500 (95.6%)；same 500 task/trial identities | 已验证 |
| Evidence | [`model3_o2/Long.md`](model3_o2/Long.md)；full artifacts at `runs/I-003/model3_o2/2026-07-31_model3_o2_long_5k_10k_eval500/` | source mirror 已登记 |

SHA 在主表中为可读性缩写；执行与验证时必须使用 `model3_o2/Long.md` 和 config 中的完整值。

### 1.2 Open Reproducibility Fields

以下字段不推翻 O2 主线选择，但必须在首次 A/R server handoff 的 preflight 中补齐：

- server absolute repo path、backend commit SHA 与 dirty status；
- evaluator commit；
- action normalization 与 gripper convention 的最终字段；
- A/R 的 measured cost unit、checkpoint set 与 training-seed policy；
- O2 Long 与 parent 的逐 task、逐 initial-state paired outcomes，用于合同规定的 CI。

### 1.3 Carrier Decision

1. O2 Object 35K 在 solver 10 达到 492/500，且 O2 Long 10K 达到 476/500；
2. Long 在相同 500 个 task/trial identities 上接近 parent 的 478/500，支持跨 suite
   portability，但没有证明 Long improvement；
3. 因此 `C* = Model3 O2`，Regression 与历史 Model3 退为 reference tracks；
4. A0–A3、R0–R3 和 B0–B2 必须在统一 O2 carrier、initialization 与 evaluator contract
   下训练，历史 O2/Model3 数字不能替代新的 matched controls；
5. Long non-inferiority 是 paper-claim certification subgate，不再阻塞 O2 主线执行。

## 2. Common Experimental Contract

除声明的 treatment 外，A/R/B/C/D 共享：

- pretrained Video-DiT base 与 Model3 O2 `C*` initialization；
- action carrier、action horizon、supervised/executed prefix 与 replanning cadence；
- 数据 split、windowing、episode grouping、task weighting 与 action normalization；
- GPU type、precision、software stack、batch contract 与 evaluator；
- checkpoint rule、formal task set 和 initial-state IDs；
- 参数、compute、memory、throughput、latency 与逐 episode outcome 的记录格式。

任何 variant 若改变上述项目，必须单列为新的 estimand，不能继续视为 matched pair。

### 2.1 Primary Estimands

| Estimand | 对照 | 回答什么 | 不能回答什么 |
|---|---|---|---|
| E-Schedule-Budget | A0–A3 相同 accelerator-hour budget | 给定总预算的 success-cost Pareto | future supervision 的纯增量效应 |
| E-Schedule-Mechanism | 固定 action-stage updates，单列 Stage V 成本 | warmup/future supervision 是否改变闭环能力 | 等预算最优资源分配 |
| E-Routing | 固定 objective、initialization、capacity，仅改 loss→PEFT | video/action gradients 应进入哪里 | 不同 capacity 的结论 |
| E-Capacity | B0/B1/B1.5/B2 | 最便宜的已测试可行 PEFT 与边界区间 | 连续空间中的绝对 minimum |

## 3. Matrix A: Supervision Schedule

所有变体共享 `C*` 的 pretrained base、action carrier、数据与评测协议。Stage V
checkpoint 是 treatment 的组成部分，因此 A2/A3 进入 Stage A 时不要求与 A0/A1 拥有
相同 PEFT 权重。

| ID | Stage V | Stage A | 默认 PEFT routing | 作用 |
|---|---|---|---|---|
| A0 | 无 | action-only | action → PEFT | 无 robot-video supervision control |
| A1 | 无独立 warmup | joint `L_video + L_action` | video + action → PEFT | matched joint control |
| A2 | robot-video warmup | action-only | action → PEFT | acquisition → specialization |
| A3 | robot-video warmup | joint `L_video + L_action` | video + action → PEFT | warmup + persistent joint dynamics |

### 3.1 A-Budget

- 固定 GPU type、precision、software stack 与 total accelerator-hours；
- 优先用 profiler-measured total FLOPs；不可用时至少报告 Wan token
  forward/backward counts；
- action-only 变体在相同时间内允许获得更多 action updates；
- 输出 success、accelerator-hours、peak memory、samples/second 与 total action updates。

### 3.2 A-Mechanism

- 固定 Stage A action updates、batch contract 与 checkpoint rule；
- A2/A3 的 Stage V 成本额外列出，不藏入“相同 steps”；
- 该对照用于 future supervision 的增量判断，不用于宣称等预算最优。

### 3.3 A 控制要求

- A1 必须在 `C*` 上新训练，不得复用历史 flow-Model3 数字；
- pilot 只能用于早停，不能形成论文结论；
- A0–A3 使用相同 task weighting、action normalization 与 evaluator；
- 记录每条 loss 的梯度接收模块，以及每步实际 forward/backward 路径与 token 数。

## 4. Matrix R: Gradient Routing

R variants 使用相同 Stage A initialization、PEFT capacity、interface 和 action carrier。
若 A 选择 warmup，四个变体从同一 Stage V checkpoint 开始。

| ID | Video loss → WM PEFT | Action loss → WM PEFT | Interface/head 接收 action loss | 作用 |
|---|---:|---:|---:|---|
| R0 | 是 | 是 | 是 | shared-gradient joint |
| R1 | 是 | 否 | 是 | video 塑造 WM，action 只塑造 interface/head |
| R2 | 否 | 是 | 是 | action-specific WM adaptation |
| R3 | 否 | 否 | 是 | frozen WM readout control |

控制与诊断：

- R0/R1 的 forward graph、loss scalar、optimizer steps 相同，仅改变 action-loss
  detach/optimizer mask；
- R2/R3 的 action-only forward graph、loss scalar、optimizer steps 相同；
- R0/R1 与 R2/R3 不是纯 routing contrast，不用跨 pair 差值解释 gradient effect；
- R2/R3 不用伪造的 zero video loss 消耗额外 forward；
- 记录共享 PEFT 上 `grad(L_video)`、`grad(L_action)` 的 norm 与 cosine similarity；
- gradient statistics 只解释机制，不能替代闭环结果。

## 5. Matrix B: PEFT Capacity Bracket

固定 A/R 胜出 schedule/routing、action carrier 与一个预声明 interface：

| ID | Video-DiT adaptation | 解释 |
|---|---|---|
| B0 | Wan base/PEFT 全冻结，只训练 interface/head | frozen lower-cost control |
| B1 | 一个预声明 compact partial-layer/low-rank PEFT | compact candidate |
| B1.5 | 仅当 B1 失败时启用的一个 bracket | 定位可行边界 |
| B2 | 当前 all-layer/rank-64 PEFT upper bound | performance/cost anchor |

预声明规则：

1. B0 对 B2 non-inferior：称 B0 为 **cheapest tested configuration**，不宣称 WM PEFT
   必要；
2. B0 失败、B1 non-inferior：称 B1 为 **smallest tested non-frozen PEFT**，不称绝对
   minimum；
3. B1 失败、B2 成功：只允许增加一个 B1.5，不展开完整 rank × layer grid；
4. B1.5 成功时报告边界 `(B1, B1.5]`，失败时报告 `(B1.5, B2]`；
5. 每个候选报告 trainable parameters、optimizer states、peak memory、GPU hours、
   throughput 与 closed-loop success。

## 6. Matrix C: Video-to-Action Interface

Matrix C 分两步执行，避免同时改变 layer bandwidth 与 aggregation。

| ID | 读取层 | Aggregation/readout | 回答的问题 |
|---|---|---|---|
| C0 | 单层 | 与 C1 相同的 query/readout contract | 单层是否足够 |
| C1 | layers 8/16/24 | 与 C0 相同的 query/readout contract | 多层信息是否有独立收益 |
| C2 | layers 8/16/24 | layer-separable compact readout | 保留层身份是否优于共享 aggregation |
| C3（条件） | layers 8/16/24 | 参数匹配的简单 pooling | query/recurrent structure 是否必要 |

参数匹配合同：

- 固定 action decoder、query 数、hidden width、输出 token 数与 training steps；
- C0/C1 的 per-layer projection 匹配总参数，或另给 parameter-matched control；
- 报告读取 token 数、层数、interface FLOPs、peak memory 与 plan-call latency；
- EnFold-style generator-state prediction 不进入 Matrix C。

### 6.1 Mandatory PEFT × Interface Check

顺序搜索后运行一次 2×2 sanity check：

| | `C-simple` | `C-current` |
|---|---:|---:|
| cheapest viable/smallest-tested PEFT | ✓ | ✓ |
| current PEFT upper bound B2 | ✓ | ✓ |

若出现明显补偿关系，报告两条联合 Pareto 路径，不单独宣称“B-small 足够”或
“C-simple 足够”。

## 7. Matrix D: Action-Feature Temporal Input

仅在 A/R/B recipe 与 interface 冻结后执行。

| ID | Action-feature latent grid | 角色 |
|---|---|---|
| D0 | clean current latent，历史 Model3 contract | historical reference |
| D1 | clean current latent，与 D2 相同 timestep/position/code path | matched current-only control |
| D2 | D1 + policy-owned Gaussian-noise future slots | noisy temporal canvas treatment |

控制要求：

- D1/D2 使用相同 PEFT、interface、action decoder、loss、训练预算与 solver；
- future slots 不读取 expert future video/action/reward/simulator state；
- D2 只运行一次 Wan，不进行 iterative video denoising；
- primary evaluation 固定 environment initial state、action-solver RNG，并建立
  `episode_id -> feature_noise_seed` 的确定性映射；
- 用额外 noise-seed mappings 做敏感性分析；
- 只有 D2 稳定提高闭环且收益覆盖 token/memory/latency 成本时才保留。

## 8. Evaluation and Statistics

### 8.1 Endpoints and Contrasts

- Primary endpoint：paired closed-loop task success；
- Primary contrasts：A2 vs A0、A2 vs A1/A3（若 A2 为候选）、R1 vs R0 或 R2 vs R3、
  B-small vs B2；
- C/D 是 secondary contrasts；
- future-video loss、action loss、probe MSE、gradient cosine 与 telemetry 只作诊断。

### 8.2 Non-Inferiority

- `δ = 2.0` percentage points 已在 O2 Long 结果前预声明，现对该 comparison 正式冻结，
  不得 post-hoc 修改；
- 报告 `Δ = p_candidate - p_reference` 的 95% paired confidence interval；
- 只有 CI 下界 `> -δ` 才判定 non-inferior；
- exact McNemar test 可辅助 paired difference/superiority，不替代 non-inferiority。

#### Current O2 Long vs Parent Record

| Quantity | Value |
|---|---:|
| O2 Long 10K | 476/500 (95.2%) |
| Model3 Long 80K parent | 478/500 (95.6%) |
| Observed paired difference `Δ` | `-0.4 pp` |
| Both succeed | 459 |
| O2 only | 17 |
| Parent only | 19 |
| Both fail | 5 |
| Exact two-sided McNemar `p` | `0.8679394004284404` |

McNemar 检验回答 discordant outcomes 是否呈现显著不对称；它不计算 `Δ` 相对 `-δ` 的
置信下界。因此，当前可以写“未检测到显著差异，且表现出较强 portability”，不能写
“已通过 `δ=2%` non-inferiority”。正式判定仍需用相同 500 个 episode 的逐 task paired
outcomes 执行预声明的 task-stratified paired bootstrap；只有 95% CI 下界 `> -2 pp` 才
通过。当前 source mirror 只有汇总 2×2 表，没有逐 task outcomes，故该子门状态为
**pending**。

### 8.3 Confidence and Randomness

- paired CI 按 task 分层，并保持相同 initial-state pair 做 bootstrap；
- 区分 rollout、feature-noise 与 training-seed variance；
- primary A/R/B 结论至少需要两个独立 training seeds，或一个 seed 加第二 suite 的一致
  方向；优先两个 seeds；
- 同一次训练的不同 checkpoint 不算独立重复；
- checkpoint sweep 的胜出结果标记为
  `best observed on a predeclared checkpoint set`。

### 8.4 Formal Evaluation

- 每个 formal suite：10 tasks × 50 trials；
- 主结论至少覆盖 Long 与 Object，再选 Spatial 或 Goal 作为额外 suite；
- LIBERO-Plus/RoboTwin 仅在 recipe 冻结后评估，不用于高频搜索；
- 固定 evaluator commit、initial-state IDs、task order 与 terminal validation；
- 保留逐 episode outcome、checkpoint SHA、config 与 command；
- evaluator 可记录接近、抓取、抬起、运输、释放、timeout 等 telemetry，但不得将其输入
  策略，总体 task success 始终是主指标。

## 9. Cost Accounting

| 成本 | 必须报告 |
|---|---|
| Training | GPU type/count、accelerator-hours、steps、samples、profiler FLOPs（若可用） |
| WM compute | future/action Wan forwards、backwards、latent token 数、层数 |
| Memory | peak allocated/reserved、optimizer-state memory |
| Parameters | base、PEFT、interface、action head 分项 |
| Deployment | Video-DiT forwards/chunk、action solver steps、plan-call latency、throughput |

“相同 steps”不能称为 compute-matched。不同 token grid、future branch 或 solver path 必须
使用实测 accelerator-hours、throughput 或 profiler FLOPs 重新核算。

## 10. Go/No-Go Gates

### G0：O2 Carrier Selected / NI Certification Open

- `C* = Model3 O2` 已冻结为主线 carrier；
- Long selected checkpoint 与 Long/Object 正式评测结果已登记；
- 首次 A/R handoff 补齐 backend/evaluator commits、action normalization、cost unit、
  checkpoint set 与 seed policy；
- Long paired CI 是 formal non-inferiority claim 的未完成子门，不阻塞 G1 执行。

### G1：Supervision and Routing Insight

1. 以 O2 carrier 启动 A/R preflight，用短 pilot 排除明显失败的 A variants；
2. 对存活 variants 执行 A-Budget 与 A-Mechanism；
3. 在固定 schedule/initialization 下执行 R0–R3；
4. 至少用第二 training seed 或第二 suite 确认方向。

若 A0 最优且 video-adapted variants 无稳定收益，停止 dynamics-supervision claim。若
routing 无稳定差异，不扩展新的 gradient mechanism。

### G2：Capacity and Deployment Boundary

1. 仅当 G1 形成可复现的非显然规律时，运行 B0/B1/B2；
2. B1 失败时只允许一个 B1.5；
3. 冻结 B recipe 后运行 C0/C1，再按需运行 C2/C3；
4. 完成 PEFT × interface 2×2；
5. 最后才运行 D1/D2。

若 A/R 规律弱但 B 显示显著成本优势，项目降级为 parameter-efficient/system study；若
两者都不成立，停止方法扩展。

## 11. Execution Approval and Artifacts

用户已于 2026-07-31 批准以 O2 为主线执行。在每次新训练启动前，handoff packet 仍必须
包含：

- G0 evidence table；
- 每个 variant 的 config diff 与 tensor/loss/gradient/optimizer 路径；
- 预算、早停规则、checkpoint set 与 formal evaluation command；
- 输出目录、artifact naming、日志与逐 episode 结果位置；
- 本次 O2 主线批准记录，以及任何超出 A/R/B/C/D 合同的新 scope 的单独批准。

当前最近一步是：**在 O2 carrier 上完成 A/R preflight 并执行 Matrix A + R；同时从
正式结果目录取回逐 task paired outcomes，补算 Long 的 `δ=2%` paired CI。**
