# 草案：Video-WM 到 WAM 的高效微调

> [!IMPORTANT]
> 本文件是 2026-07-31 拆分前的合并草案，仅用于追溯，不再作为主 Proposal。
> 当前版本请阅读 [research-proposal.md](research-proposal.md)；实验执行细节见
> [experiment-contract.md](experiment-contract.md)；文献、历史结果与审计边界见
> [evidence-and-related-work.md](evidence-and-related-work.md)。
> 本归档正文中的旧 G0 结论已经过时：当前决定以 Model3 O2 为主线载体；Long 的
> $\delta = 2\%$ non-inferiority 仍等待合同规定的 paired CI。
> 本归档正文中的旧实验顺序也已经过时；当前优先级为
> **O2 → D → L → C → B → minimal A/R sanity checks**。

## 高效 WM-to-WAM 实验 Proposal

## 0. 文档状态与证据边界

- 当前状态：**内部实验合同，等待用户审核；不自动启动新训练**。
- Core idea：**在有限训练与部署预算下，高效微调预训练 Video-WM，使其获得有竞争力的 WAM 闭环控制能力。**
- 本文优先回答同一 Video-DiT 母体内的 fine-tuning question，不提出新的
  generator-state distillation 模块。
- EnFold 已封闭 multi-level generator-state prediction、current-only student 和
  generator-free control 作为新方法的空间，但它属于不同 adaptation regime，只在
  Scope Boundary 中讨论，不进入主方法矩阵和主实验 Gate。
- 当前本地工作区没有 `model3_o2` 或最新 Regression Long 的权威 server return、
  checkpoint 记录和正式评测报告。因此，本文不采信外部对话中“当前已推进到某个
  model3_o2 阶段”的说法；正式实验载体由 G0 根据服务器证据重新冻结。
- 历史 Model3、Model3 Regression、Model5 结果可作 evidence/diagnostic anchor，
  不能自动充当新的 matched control。

本文是主 Proposal。详细 related-work matrix、历史消融、文献纳入模板和“不做事项”
放在附录，避免主线被已有结果和方法名淹没。

---

## 1. Gap、核心问题与研究范围

### 1.1 从现有模型暴露的问题出发

Light-WAM、FastWAM、DiT4DiT、DeVA、VidMan 等工作已经证明：预训练 Video-DiT
可以通过机器人数据与动作目标被适配成有效 WAM。它们却同时改变了：

- future-video supervision 的 target 与使用阶段；
- Video-DiT 的可训练范围；
- video/action gradients 的流向；
- Video-to-Action interface；
- 动作前向的 temporal latent grid；
- action decoder 与部署计算。

因此，跨论文性能只能说明多种完整方案可行，不能回答在固定 Video-DiT、数据、
action carrier 与评测协议后，**怎样微调最有效、收益来自哪里、成本花在哪里**。

核心问题是：

> **How can we efficiently fine-tune a pretrained Video-DiT into a
> high-performing World-Action Model under a constrained resource budget?**

中文表述：

> **在有限训练与部署预算下，如何组织 supervision、gradient routing 与 PEFT，
> 才能把预训练 Video-DiT 高效微调成高性能 WAM？**

“高效”不是只看 trainable parameters。候选模型必须先达到预声明闭环性能，再比较：

1. trainable parameters；
2. total accelerator-hours 与 measured training compute；
3. peak training memory；
4. online Video-DiT forward/token cost；
5. plan-call latency。

### 1.2 主技术路线：in-place adaptation

主 related-work/method comparison 只讨论 **在线仍保留 Video-DiT 或 video generator
计算的 in-place adaptation**：

| 路线 | 代表工作 | 与本文的关系 |
|---|---|---|
| Frozen backbone + PEFT/interface | Light-WAM | 证明轻量适配可行，但没有给出相同母体内的 supervision/routing 因果比较 |
| Joint video/action objectives | FastWAM、DiT4DiT、DeVA | 证明持续 predictive objective 可行，但接口、容量和梯度路径同时变化 |
| Robot-video acquisition → action specialization | VidMan | 支持 staged training，但其 backbone、数据和任务不能直接外推到 Wan/LIBERO |
| Compressed/asynchronous future use | Efficient-WAM、AHA-WAM | 提供系统效率参考，不直接回答当前 fine-tuning schedule |

EnFold 不放入该表。它用 training-time generator states 监督独立 current-only DINO
encoder，task gradients 不进入 encoder，部署时不运行 generator。它与 in-place
Video-DiT fine-tuning 不是同一组可交换变量。

### 1.3 两个主问题与两个部署侧问题

#### Primary RQ1：Dynamics supervision 应如何组织？

- action-only、从头 joint、video warmup 后 action-only、video warmup 后 joint，
  哪种 schedule 在相同总成本下最好？
- 在固定 action-stage updates 时，future supervision 是否仍有增量价值？
- video/action gradients 应该如何进入共享 PEFT？

#### Primary RQ2：获得该收益需要多少 Video-DiT adaptation capacity？

- frozen backbone 是否足够？
- 一个 compact PEFT 能否在预声明 non-inferiority margin 内接近当前 PEFT 上界？
- 如果 compact candidate 失败，性能边界位于哪里？

#### Secondary RQ3：最低成本的在线 Video-to-Action interface 是什么？

先固定 aggregation 只改变读取层数，再固定读取层数改变 aggregation，避免把
multi-level information 与 query architecture 混为一谈。

#### Secondary RQ4：动作特征前向是否需要 noisy future slots？

只有 RQ1/RQ2 的主线成立、载体和接口已冻结后，才比较 current-only 与
policy-owned noisy future slots。该问题用于确认 temporal canvas 的部署成本边界，
不与 RQ1/RQ2 平起平坐。

主研究顺序：

```text
fixed carrier
  -> supervision schedule
  -> gradient routing
  -> PEFT capacity
  -> interface cost
  -> noisy temporal slots
```

---

## 2. Primary estimands 与结果解释

### 2.1 三个 primary estimands

| Estimand | 对照 | 回答什么 | 不能回答什么 |
|---|---|---|---|
| **E-Schedule-Budget** | A0/A1/A2/A3 在相同 accelerator-hour budget 下 | 给定总训练预算，哪种 schedule 的 success-cost Pareto 最好 | future supervision 的纯增量因果效应 |
| **E-Schedule-Mechanism** | 固定 action-stage updates，额外报告 Stage V 成本 | future supervision/warmup 是否改变闭环能力 | 等总成本下的最优资源分配 |
| **E-Routing** | 固定 objective、initialization、capacity，只改变 loss→PEFT 路径 | video/action gradients 应进入哪里 | 不同 PEFT 容量之间的结论 |
| **E-Capacity** | frozen、compact、conditional bracket、current upper bound | 最便宜的已测试可行 PEFT 与边界区间 | 连续空间中的绝对 minimum |

前两个 schedule estimands 都要报告。不能用等预算结果直接写成“future supervision
具有因果增益”，也不能用固定 action updates 的结果掩盖额外 warmup 成本。

### 2.2 Claim decision tree

Proposal 不预设最终答案。论文 claim 由结果条件决定：

| 结果模式 | 可支持的设计规律 |
|---|---|
| A2 > A0 且 A2 ≥ A1/A3 | future prediction 更像 dynamics-acquisition scaffold，而非必须持续的 auxiliary objective |
| A3 > A2 | persistent joint dynamics 在 action adaptation 中仍有独立价值 |
| R1 > R0 | action gradient 进入 WM PEFT 会产生干扰，支持 gradient isolation |
| R2 > R3 | action-specific PEFT adaptation 优于只读 frozen dynamics |
| B-small non-inferior to B2 | 控制特化只需稀疏 adaptation capacity |
| C-single non-inferior to C-multi | 高带宽多层接口不是必要条件 |
| C-multi > C-single，但 C-compact ≈ C-current | 多层信息必要，复杂 recurrent readout 不必要 |
| D2 不优于 D1 且成本更高 | noisy temporal canvas 不是控制所必需 |

强论文需要其中至少两到三项结果形成同一条规律，例如：

> **Acquire dynamics broadly, specialize control sparsely.**

如果 A/R 没有稳定规律、只有 PEFT/latency Pareto，则论文应降级为
parameter-efficient/system-efficiency study；不能用实验数量包装机制 insight。

---

## 3. G0：冻结实验载体

### 3.1 为什么必须先冻结 carrier

旧 Model3 使用 flow action decoder；Regression、`model3_o2` 或其他近期分支可能使用
不同 action head、checkpoint 或训练路径。若 carrier 改变，旧 Model3 joint 结果不能
直接充当 A1。

在服务器权威状态核对前，统一用占位符 $C^\ast$ 表示最终 carrier。G0 必须记录：

| 字段 | 必需内容 |
|---|---|
| Repository | 绝对路径、remote、branch、commit SHA、dirty status |
| Initialization | Wan base、PEFT checkpoint、interface/head initialization |
| Action carrier | regression / flow / other，参数量与 solver |
| Action contract | horizon、supervised prefix、normalization、gripper convention |
| Replanning | executed prefix、cadence、action-solver steps |
| Data | dataset path、split、windowing、episode grouping |
| Evaluation | evaluator commit、task set、initial-state IDs、terminal validation |
| Existing evidence | checkpoint SHA、500-episode result、latency protocol、日志位置 |

### 3.2 Carrier 选择规则

1. 只比较在同一 evaluator 和 action contract 下得到的正式结果；
2. 若 direct regression 相对 flow 在闭环上达到预声明 non-inferiority 且 latency 更低，
   可选 regression carrier；
3. 若 `model3_o2` 是当前最强候选，必须先返回上述完整证据，不能根据分支名或阶段性
   指标直接选用；
4. carrier 一旦冻结，A0/A1/A2/A3、R0–R3 和 B0–B2 必须重新在 $C^\ast$ 上训练；
5. 旧 Model3/Regression/Model5 数字只作为 historical anchor。

当前本地结论：**无法确认 `model3_o2` 或 Regression Long 的最新终态，G0 未完成。**

---

## 4. 主实验矩阵

### 4.1 Matrix A：Supervision schedule

所有变体共享 $C^\ast$ 的 pretrained Wan base、action carrier、随机初始化合同、数据和
评测协议。Stage V checkpoint 是 treatment 本身，因此 A2/A3 进入 Stage A 时不要求
与 A0/A1 拥有相同 PEFT 权重。

| ID | Stage V | Stage A | 默认 PEFT routing | 作用 |
|---|---|---|---|---|
| A0 | 无 | action-only | action → PEFT | 无 robot-video supervision control |
| A1 | 无独立 warmup | joint $L_{\mathrm{video}} + L_{\mathrm{action}}$ | video + action → PEFT | fixed-carrier joint control |
| A2 | robot-video warmup | action-only | action → PEFT | dynamics acquisition → control specialization |
| A3 | robot-video warmup | joint $L_{\mathrm{video}} + L_{\mathrm{action}}$ | video + action → PEFT | warmup + persistent joint dynamics |

#### A-Budget：等总成本

- 固定相同 GPU type、precision、software stack 和 total accelerator-hours；
- 优先使用 profiler-measured total training FLOPs；若不可用，至少报告
  Wan token-forward/backward counts；
- action-only 在相同时间内允许获得更多 action updates；
- 输出 success、GPU hours、peak memory、samples/second 和 total action updates。

#### A-Mechanism：固定 action updates

- 固定 Stage A 的 action update 数、batch contract 与 checkpoint rule；
- A2/A3 的 Stage V 成本额外列出，不藏入相同 steps；
- 该对照用于判断 future supervision 的增量价值，不用于宣称等预算最优。

#### Matrix A 控制要求

- A1 必须是在 $C^\ast$ 上新训练的 matched joint control，不得复用旧 flow-Model3 数字；
- pilot 只能早停，不能形成论文结论；
- A0–A3 使用相同 task weighting、action normalization 和 evaluator；
- 记录每条 loss 对哪些模块有梯度、每步实际 forward/backward 路径与 token 数。

### 4.2 Matrix R：Gradient routing

Matrix R 使用同一个 Stage A initialization、PEFT capacity、interface 和 action carrier。
它包含两个 matched pair：R0/R1 在 joint objective 下只改变 action gradient 是否进入
WM PEFT；R2/R3 在 action-only objective 下只改变 action gradient 是否进入 WM PEFT。
若 Matrix A 选择了 warmup，则四个 R 变体从同一个 Stage V checkpoint 开始。

| ID | Video loss → WM PEFT | Action loss → WM PEFT | Interface/head 接收 action loss | 作用 |
|---|---:|---:|---:|---|
| R0 | 是 | 是 | 是 | shared-gradient joint |
| R1 | 是 | 否 | 是 | video 塑造 WM，action 只塑造 interface/head |
| R2 | 否 | 是 | 是 | action-specific WM adaptation |
| R3 | 否 | 否 | 是 | frozen WM readout control |

控制与诊断：

- R0/R1 的 forward graph、loss scalar、optimizer steps 保持一致，只改变
  action-loss detach/optimizer mask；
- R2/R3 的 action-only forward graph、loss scalar、optimizer steps 保持一致；
- R0/R1 与 R2/R3 之间不是纯 routing contrast，不能用其差值单独解释 gradient effect；
- R2/R3 不以伪造的 zero video loss 消耗额外 forward；
- 记录共享 PEFT 上 $\operatorname{grad}(L_{\mathrm{video}})$、$\operatorname{grad}(L_{\mathrm{action}})$ 的 norm 与 cosine similarity；
- gradient statistics 只解释机制，不能替代闭环结果。

### 4.3 Matrix B：PEFT capacity bracket

固定 A/R 的胜出 schedule/routing、action carrier 和一个预声明 interface：

| ID | Video-DiT adaptation | 解释 |
|---|---|---|
| B0 | Wan base/PEFT 全冻结，只训练 interface/head | frozen lower-cost control |
| B1 | 一个预声明 compact partial-layer/low-rank PEFT | compact candidate |
| B1.5 | 仅当 B1 失败时启用；位于 B1 与 B2 之间的一个 bracket | 定位可行边界 |
| B2 | 当前 all-layer/rank-64 PEFT 上界 | performance/cost upper anchor |

预声明规则：

1. B0 对 B2 non-inferior 时，结论是 frozen readout 为 **cheapest tested
   configuration**，当前实验没有证明 WM PEFT 必要；
2. B0 失败而 B1 non-inferior 时，B1 只能称为 **smallest tested non-frozen PEFT**，
   不能称为绝对 minimum；
3. B1 失败、B2 成功时，只允许增加一个 B1.5，不展开完整 rank × layer grid；
4. B1.5 成功时，报告边界位于 `(B1, B1.5]`；失败时报告位于 `(B1.5, B2]`；
5. 每个候选报告 trainable parameters、optimizer states、peak memory、GPU hours、
   throughput 与 closed-loop success。

### 4.4 Matrix C：Video-to-Action interface

Matrix C 分两步执行，避免同时改变 layer bandwidth 与 aggregation。

| ID | 读取层 | Aggregation/readout | 回答的问题 |
|---|---|---|---|
| C0 | 单层 | 与 C1 相同的 query/readout contract | 单层是否足够 |
| C1 | layers 8/16/24 | 与 C0 相同的 query/readout contract | 多层信息是否有独立收益 |
| C2 | layers 8/16/24 | layer-separable compact readout | 保留层身份是否优于共享 aggregation |
| C3（条件） | layers 8/16/24 | 参数匹配的简单 pooling | query/recurrent structure 是否必要 |

参数匹配合同：

- 固定 action decoder、query 数、hidden width、输出 token 数与 training steps；
- C0/C1 的 per-layer projection 需匹配总参数或另给 parameter-matched control；
- 报告读取 token 数、层数、interface FLOPs、peak memory 与 plan-call latency；
- EnFold-style generator-state prediction 不进入 Matrix C。

### 4.5 强制 PEFT × interface 交互检查

顺序搜索后，必须运行一个 2×2 sanity check：

| | 简单接口 `C-simple` | 当前三层接口 `C-current` |
|---|---:|---:|
| cheapest viable/smallest-tested PEFT | ✓ | ✓ |
| current PEFT upper bound B2 | ✓ | ✓ |

该检查只回答是否存在明显补偿关系：

- 小 PEFT 是否必须依赖更宽接口；
- 大 PEFT 是否允许更窄接口。

若存在强交互，不能单独宣称“B-small 足够”或“C-simple 足够”，而应报告两条 Pareto
路径。

### 4.6 Matrix D：Action-feature temporal input（条件）

只有 G1–G4 完成后才执行：

| ID | Action-feature latent grid | 角色 |
|---|---|---|
| D0 | clean current latent，历史 Model3 contract | historical reference |
| D1 | clean current latent，使用与 D2 相同的 feature timestep/position/code path | matched current-only control |
| D2 | D1 + policy-owned Gaussian-noise future slots | noisy temporal canvas treatment |

控制要求：

- D1/D2 使用相同 PEFT、interface、action decoder、loss、训练预算和 solver；
- future slots 不得读取 expert future video/action/reward/simulator state；
- D2 只运行一次 Wan，不做 iterative video denoising；
- primary evaluation 固定 environment initial state、action-solver RNG，并建立
  `episode_id -> feature_noise_seed` 的确定性映射；
- 额外使用多套 noise-seed mapping 做敏感性分析；
- 只有 D2 稳定提高闭环且收益覆盖额外 token/memory/latency 时才保留。

---

## 5. 执行顺序与 Go/No-Go Gates

### G0：Carrier freeze

- 完成服务器 repo/checkpoint/evaluator 证据核对；
- 固定 $C^\ast$、action contract、primary suite、cost unit 与统计 margin；
- 未完成 G0，不启动 A/R/B。

### G1：Supervision schedule

1. 用短 pilot 排除明显失败的 A 变体；
2. 对存活变体分别执行 A-Budget 与 A-Mechanism；
3. 至少用第二个独立 training seed 或第二个 suite 确认方向；
4. checkpoint stability 只作辅助证据，不算独立确认。

若 A0 最优且 video-adapted variants 无稳定收益，停止 dynamics-supervision claim，
论文缩小为 PEFT/system efficiency。

### G2：Gradient-routing insight gate

- 在固定 schedule/initialization 下执行 R0–R3；
- A + R 必须共同形成一个可复现的非显然规律，才继续机制论文主线；
- 若 routing 无稳定差异，不继续扩展新的 gradient mechanism。

### G3：PEFT capacity boundary

- 运行 B0/B1/B2；
- B1 失败时只允许一次 B1.5 bracket；
- 使用 non-inferiority，而不是“p 值不显著”判断 compact PEFT 是否保留性能。

### G4：Interface 与交互

- 先执行 C0/C1，再执行 C2；C3 仅在需要时运行；
- 完成强制 PEFT × interface 2×2；
- 若 interface 与 PEFT 存在强补偿关系，报告联合 Pareto，不给出单轴结论。

### G5：Noisy temporal input

- D1 是 D2 的必要 matched control；
- D2 没有稳定收益或成本过高时关闭 noisy-future 路线；
- 不把 D 作为论文主贡献。

### G6：Generalization

- formal result：10 tasks × 50 trials；
- 主结论至少覆盖 Long 与 Object；
- 再选择 Spatial 或 Goal 作为额外 suite；
- LIBERO-Plus/RoboTwin 只在 recipe 冻结后评估，不用于高频搜索。

---

## 6. 统计与成本合同

### 6.1 Primary endpoints 与 contrasts

- Primary endpoint：paired closed-loop task success；
- Primary contrasts：
  - A2 vs A0、A2 vs A1/A3（若 A2 为候选）；
  - R1 vs R0 或 R2 vs R3；
  - B-small vs B2；
- C/D 属 secondary contrasts；
- future-video loss、action loss、probe MSE、gradient cosine 和 telemetry 都是诊断，
  不能替代闭环。

### 6.2 Non-inferiority

- 暂定 non-inferiority margin：$\delta = 2.0\,\mathrm{pp}$；
- G0 必须在查看新实验结果前冻结或收紧 $\delta$，之后不得 post-hoc 修改；
- 对 paired comparison 报告
  $\Delta = p_{\mathrm{candidate}} - p_{\mathrm{reference}}$ 的 95% paired confidence interval；
- 只有 CI 下界 $> -\delta$ 才判定 non-inferior；
- exact McNemar test 用于 paired difference/superiority 的辅助分析，不替代
  non-inferiority。

### 6.3 Confidence interval 与随机性

- paired CI 使用按 task 分层、保持相同 initial-state pair 的 bootstrap；
- 区分 rollout variance、feature-noise variance 与 training-seed variance；
- primary A/R/B 结论至少需要两个独立 training seeds 或一个 seed 加第二 suite 的
  一致方向；优先两个 seeds；
- 同一次训练的不同 checkpoint 不算独立重复；
- checkpoint sweep 的胜出结果标记为
  `best observed on a predeclared checkpoint set`。

### 6.4 Cost accounting

| 成本 | 必须报告 |
|---|---|
| Training | GPU type/count、accelerator-hours、steps、samples、profiler FLOPs（若可用） |
| WM compute | future/action Wan forwards、backwards、latent token 数、层数 |
| Memory | peak allocated/reserved、optimizer-state memory |
| Parameters | base、PEFT、interface、action head 分项 |
| Deployment | Video-DiT forwards/chunk、action solver steps、plan-call latency、throughput |

“相同 steps”不能称为 compute-matched。不同 token grid、future branch 或 solver path
必须使用实测 accelerator-hours/throughput 重新核算。

### 6.5 Evaluation integrity

- 固定 evaluator commit、initial-state IDs、task order 与 terminal validation；
- 保留逐 episode outcome、checkpoint SHA、config 与 command；
- evaluator 可以记录接近、抓取、抬起、运输、释放、timeout 等 telemetry，但不得
  把它们输入策略；
- telemetry 只解释改进阶段，总体 task success 仍是主指标。

---

## 7. 论文贡献与降级路径

### 7.1 理想的三项贡献

1. **Causal fine-tuning study**：在相同 Video-DiT/carrier 下隔离 supervision schedule
   与 video/action gradient routing；
2. **Efficient adaptation frontier**：建立 closed-loop success 对 trainable parameters、
   GPU hours、memory 与 latency 的 Pareto，并通过 bracket 定位可行 PEFT 边界；
3. **Deployment boundary**：验证保持主收益所需的最低接口带宽，以及 noisy temporal
   canvas 是否必要。

### 7.2 强、中、弱结果

| 结果等级 | 条件 | 论文定位 |
|---|---|---|
| 强 | A/R 揭示统一机制规律，B/C 显示收益可由稀疏 adaptation/compact interface 保留 | mechanism + efficient adaptation paper |
| 中 | A/R 规律弱，但 B-small 在成本上显著占优且闭环 non-inferior | parameter-efficient/system paper |
| 弱 | schedule/routing 无稳定规律，小 PEFT/简单接口明显掉点 | 技术报告或 negative study，不扩展方法 |

### 7.3 结果条件化的 paper-facing claim

不在实验前固定结论。若 staged + sparse specialization 成立，优先使用：

> **Future prediction is most useful as a dynamics-acquisition stage rather
> than a permanently coupled control objective; subsequent action specialization
> requires only sparse Video-DiT adaptation and a compact online interface.**

若 persistent joint 或 gradient isolation 胜出，则根据 Claim Decision Tree 改写，
不强行使用上述句子。

---

## 8. 当前最小执行计划

1. **完成 G0 状态核对**：从服务器返回最新 `model3` / `model3_regression` /
   `model3_o2` 的 repo SHA、checkpoint、正式评测、action contract 与 latency；
2. **冻结实验合同**：carrier、primary suite、$\delta$、cost unit、checkpoint set、
   training-seed policy；
3. **实现前审计**：为 A0–A3 与 R0–R3 画出 tensor/loss/gradient/optimizer 路径，
   确认每个 treatment 只改变声明的变量；
4. **优先运行 A + R pilot**：先判断能否产生论文 insight；
5. **A/R 通过后才运行 B**；C、D 不得提前抢资源；
6. 用户明确批准后，才创建或更新 server-side handoff/experiment packet。

当前最近一步不是实现 EnFold、ASDB、Matrix C 或 noisy slots，而是：

> **冻结 carrier，并让 Matrix A + Matrix R 成为可执行、可计费、可证伪的实验合同。**

---

# Appendix A. Related Work 与 Scope Boundary

## A.1 In-place adaptation matrix

| 方法 | WM 适配 | Supervision/routing | 在线接口 | 推理路径 | 本文边界 |
|---|---|---|---|---|---|
| Light-WAM | frozen Wan base + LoRA/adapters | joint video/action；具体 routing 需代码核对 | selected layers + state fusion | current observation Wan once | 强 PEFT baseline，但非 matched causal study |
| FastWAM | 需继续核对统一成本 | joint video/action flow | Action-DiT 逐层读取 video K/V | video prefill + action denoising | 高带宽接口，不给出最低成本边界 |
| DiT4DiT | joint configuration | video/action experts 联合训练 | Video-DiT hidden condition | noisy future grid + action denoising | future slots 与接口同时变化 |
| DeVA | Video2World DiT + Action Expert 均训练 | warmup + joint | multi-layer/multi-timestep transfer | joint future/action process | 容量和接口较重 |
| VidMan | Stage 2 可更新或冻结 VDT | video pretrain → action-only | layer-wise action adapter | fixed noisy video latents，更新 action | 支持 staged hypothesis |
| Model3 historical | frozen Wan base + all-layer rank-64 LoRA/adapters | joint $L_{\mathrm{video}} + L_{\mathrm{action}}$ | recurrent queries over layers 8/16/24 | current observation Wan once + action solver | 历史 upper anchor；新实验需固定 carrier 重训 |

## A.2 EnFold：Scope 与 novelty boundary

EnFold 的训练/推理结构是：

```text
training:
  teacher-forced real future
  -> Cosmos generator multi-level states
  -> timestep-conditioned target for current-only DINO encoder
  -> detached action readout

deployment:
  current-only DINO encoder
  -> action head
  -> no generator execution
```

它与本文 former ASDB 候选的 method-level overlap：

- selected multi-level generator states；
- timestep-conditioned state prediction；
- stop-gradient teacher/readout contract；
- current-only predictive representation；
- generator-free control。

因此本文不能把这些机制重新包装为方法创新。但 EnFold 同时改变 online backbone、
predictive target、gradient contract、action representation 与 deployment path，不能
放入 in-place PEFT/interface matrix，也不应设置正式 Matrix E/Gate。

若未来资源充足，可在 Discussion 中提出 matched system comparison；它是独立扩展，
不是当前 Proposal 的必需实验。

---

# Appendix B. 历史证据与张力

## B.1 本地历史结果

| 结果 | 数值 | 证据边界 |
|---|---:|---|
| Model3 Long step 80K | 478/500，95.6% | 历史 flow-carrier 正式结果 |
| Released Light-WAM Long | 461/500，92.2% | 本地发布权重复测 |
| Model3 Object flow-10 | 440/500，88.0% | 历史固定配置 |
| Model3 Object flow-5 | 467/500，93.4% | post-hoc solver diagnostic |
| Model3 Regression Object step 20K | 467/500，93.4% | predeclared checkpoint set 中 best observed |
| Released Light-WAM Object | 497/500，99.4% | 本地发布权重复测 |
| Model3 plan-call latency | 232.994 ms | 历史受控测试 |
| Light-WAM plan-call latency | 70.327 ms | 历史受控测试 |

这些数字不能替代 $C^\ast$ 上重新训练的 A/R/B controls。

## B.2 DeVA 与 VidMan 的张力

DeVA paper-reported RoboCasa ablation：

| 变体 | Success |
|---|---:|
| Action only | 19.8% |
| Goal-image prediction | 25.8% |
| Future video + unified backbone | 36.8% |
| Future video + decoupled multi-level transfer | 66.0% |
| + affordance/depth guidance | 72.0% |

这些结果同时改变 expert、interface 和容量，不能直接证明 decoupling 因果。

VidMan paper-reported CALVIN ablation：

| 变体 | Avg. Len. |
|---|---:|
| Stage 2 $L_{\mathrm{video}} + L_{\mathrm{action}}$ | 2.70 |
| Stage 2 action-only | 3.42 |
| Frozen VDT + adapter/head | 2.98 |
| Action loss 更新 VDT + adapter/head | 3.42 |

两篇工作共同产生 Matrix A/R 的研究动机：future-video objective 可能帮助 dynamics
acquisition，但持续 joint gradient 也可能与 control specialization 冲突。

## B.3 已关闭解释

- cadence 不是历史 Model3/Light-WAM Object gap 的主因；
- Light-WAM 与 Model3 的直接 action prefix 差异不能继续作为主要解释；
- frozen spatial C3/C3-add 缺少 correspondence control，不能扩展；
- flow/regression 互补失败集合不足以支持双头 router；
- Regression 与 flow 来自独立训练路径，不能把差异归因于 decoder-only。

---

# Appendix C. 新文献纳入模板

```text
工作名称 / 版本 / checkpoint：
Video backbone：
训练期 generator 与部署期 encoder 是否相同：
Video objective：
Action objective / decoder：
Predictive target：
训练是 staged、joint 还是独立：
L_video 是否更新 WM / PEFT：
L_action 是否更新 WM / PEFT：
task loss 是否更新 predictive encoder：
接口读取哪些层 / timesteps / tokens：
推理是否运行 generator / future branch：
trainable params / GPU hours / memory / latency：
最关键消融：
消融是否匹配参数、训练预算和推理协议：
能够支持的结论：
不能支持的结论：
与 RQ1–RQ4 的关系：
```

如果论文未公开某项，填“未报告”，不得根据方法名推断。

---

# Appendix D. 明确不做的事情

- 不把 EnFold 放入主 in-place adaptation matrix；
- 不创建 EnFold-style Matrix E 或把 fold-out 设为当前正式 Gate；
- 不重复提出 ASDB、多层 generator-state prediction 或 current-only student；
- 不把旧 Model3 flow 结果当作新 carrier 的 A1；
- 不用相同 steps 冒充 compute-matched；
- 不把 checkpoint 当作独立 seed；
- 不把“差异不显著”当作 non-inferiority；
- 不把一个 B1 候选称为绝对 minimum；
- 不让 Matrix C 同时改变层数、aggregation、recurrence 与容量；
- 不跳过 PEFT × interface 交互检查；
- 不在 A/R insight gate 前扩展 C/D；
- 不无条件跑满所有 80K/150K 实验；
- 不用 probe、offline loss 或 gradient cosine 代替闭环成功率；
- 不在用户批准前启动新增训练或修改 server project routing。
