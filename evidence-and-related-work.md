# Efficient Video-WM-to-WAM Adaptation

## Evidence, Related Work, and Audit Boundary

> 用途：保存 related-work evidence、历史结果、已关闭解释与新文献审计记录。研究叙事见
> [research-proposal.md](research-proposal.md)，执行合同见
> [experiment-contract.md](experiment-contract.md)。

## 1. Evidence Boundary

- 本文主线只研究 **in-place adaptation**：部署时仍运行被适配的 Video-DiT 或 video
  generator computation。
- 当前主线 carrier 是 **Model3 O2**。其 checkpoint identity 与 Object/Long/Spatial 正式结果
  已进入 source mirror；当前实验顺序为 D → L → C → B → minimal A/R sanity，所有新
  controls 都必须在统一 O2 合同下执行。
- 历史 Model3、Model3 Regression 和 Model5 退为 diagnostic/reference tracks。
- 论文没有公开的字段记为“未报告”，不得根据方法名或结构图推断。
- 本文不提出 generator-state distillation、独立 current-only student 或 generator-free
  control 新模块。

## 2. Related-Work Matrix

| 方法 | WM 适配 | Supervision / routing | 在线接口与推理 | 对本文的作用 |
|---|---|---|---|---|
| Light-WAM | frozen Wan base + LoRA/adapters | joint video/action；具体 routing 待代码核对 | selected layers + state fusion；current observation Wan once | 证明轻量适配可行；不是 temporal/depth 的 matched causal study |
| FastWAM | 统一成本仍需核对 | joint video/action flow | Action-DiT 逐层读取 video K/V；video prefill + action denoising | 展示持续 predictive objective 与高带宽接口 |
| DiT4DiT | joint configuration | video/action experts 联合训练 | Video-DiT hidden condition；noisy future grid + action denoising | future slots、接口与 objective 同时变化 |
| DeVA | Video2World DiT + Action Expert 均训练 | warmup + joint | multi-layer/multi-timestep transfer；joint future/action process | 支持重容量方案，但不能单独归因 decoupling |
| VidMan | Stage 2 可更新或冻结 VDT | video pretrain → action-only | layer-wise action adapter；fixed noisy video latents | 支持 staged hypothesis 与 action-gradient 研究 |
| Efficient-WAM / AHA-WAM | compressed/asynchronous future use | 各自效率机制 | 面向训练或推理成本优化 | 系统效率参考，不直接回答 temporal/depth attribution |
| Model3 O2（active carrier） | frozen Wan base + all-layer rank-64 LoRA/adapters | joint future-video/action flow | recurrent layers 8/16/24 + layer-aware `q1/q2/q3` readout；H8/R8、solver 10 | 当前 D/L/C/B 与 A/R sanity 的统一载体 |
| Model3 historical | frozen Wan base + all-layer rank-64 LoRA/adapters | joint $L_{\mathrm{video}} + L_{\mathrm{action}}$ | recurrent queries over layers 8/16/24；current Wan once + action solver | 历史 upper anchor；新 carrier 上必须重训 |

跨论文共同缺口不是“Video-DiT 能否做 WAM”，而是：在固定强 carrier 后，动作前向是否
需要 temporal canvas、控制信息在哪些深度可读、这些深度应如何组合，以及是否只适配这些
blocks 就足以保持闭环性能。Supervision 与 gradient routing 退为最后的 sanity boundary。

## 3. EnFold Scope and Novelty Boundary

EnFold 的结构是：

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

它覆盖了此前候选中的以下 method space：

- selected multi-level generator states；
- timestep-conditioned state prediction；
- stop-gradient teacher/readout contract；
- current-only predictive representation；
- generator-free control。

因此，本文不能把这些机制重新包装为新方法。另一方面，EnFold 同时改变 online backbone、
predictive target、gradient contract、action representation 与 deployment path，与本文的
in-place PEFT treatments 不是同一组可交换变量。

主 Proposal 只保留一句 scope boundary；本文档保留完整边界。EnFold 不进入 Matrix C，
不设置正式 Matrix E/Gate。未来如做 matched system comparison，应作为独立扩展并重新申请
实验资源。

## 4. Local Evidence and Carrier Decision

| 结果 | 数值 | 证据边界 |
|---|---:|---|
| Model3 O2 Object step 35K, solver 10 | 492/500，98.4% | predeclared set 中 best observed；支持高性能 carrier 选择 |
| Model3 O2 Long step 10K | 476/500，95.2% | validated selected checkpoint；强 portability，不是 Long improvement |
| Model3 O2 Spatial step 10K | 489/500，97.8% | validated selected checkpoint；较历史 parent +1 success，不是已证明的 improvement |
| Model3 Long step 80K | 478/500，95.6% | 历史 flow-carrier 正式结果 |
| Model3 Spatial step 60K | 488/500，97.6% | 历史 strict pass；eval ledger 已删除，当前不可本地审计 |
| Released Light-WAM Long | 461/500，92.2% | 本地发布权重复测 |
| Model3 Object flow-10 | 440/500，88.0% | 历史固定配置 |
| Model3 Object flow-5 | 467/500，93.4% | post-hoc solver diagnostic |
| Model5 Object step 15K, solver 10 | 466/500，93.2% | one-slot $[0, 1000]$ temporal treatment；three-checkpoint two-solver sweep，terminally validated |
| Model5 Object step 15K, solver 5 | 478/500，95.6% | same checkpoint/protocol; terminally validated matched solver result |
| Model3 Regression Object step 20K | 467/500，93.4% | predeclared checkpoint set 中 best observed |
| Released Light-WAM Object | 497/500，99.4% | 本地发布权重复测 |
| Model3 plan-call latency | 232.994 ms | 历史受控测试 |
| Light-WAM plan-call latency | 70.327 ms | 历史受控测试 |

O2 的 Object 高表现，以及 Long/Spatial 接近 parent 的表现，共同支持其成为 $C^\ast$，但不能
替代 $C^\ast$ 上重新训练的 D/L/C/B controls。特别是 flow 与 regression 来自不同训练路径，
其差异不能归因于 decoder-only treatment。

### 4.1 Model5 Object Two-Solver Sweep

Model5 Object evaluates one clean current slot plus one policy-owned noisy
future slot at explicit Wan timesteps $[0, 1000]$. The 10K/15K/20K checkpoints
all completed terminal validation for the same 500 Object task/trial identities
at both action solvers:

| Checkpoint | Solver 10 | Solver 5 |
|---|---:|---:|
| 10K | 400/500 | 448/500 |
| 15K | 466/500 | 478/500 |
| 20K | 459/500 | 454/500 |

Solver 10 executed first as a resource schedule only. Both are retained as
formal matched results, and this sweep does not impose an automatic cross-solver
selection rule. Step 15K is best observed under both settings. Matched outcome
tables are: 10K `(396 both-success, 4 solver-10-only, 52 solver-5-only, 48
both-fail; p=1.10e-11)`, 15K `(461, 5, 17, 17; p=0.0169)`, and 20K `(441, 18,
13, 28; p=0.4731)`. This is one trained treatment family, not an independent
seed-level comparison of solver or temporal-slot effects. The source mirror
does not include checkpoints, videos, or raw rollout logs; see
[`model5/Object.md`](model5/Object.md) for the retained-evidence boundary.

### 4.2 O2 Long Paired Comparison

O2 Long 10K 与固定 Model3 Long 80K parent 使用相同 500 个 task/trial identities：

| Matched outcome | Episodes |
|---|---:|
| Both succeed | 459 |
| O2 only | 17 |
| Model3 only | 19 |
| Both fail | 5 |

观测差为 $\Delta = 95.2\% - 95.6\% = -0.4\,\mathrm{pp}$，exact two-sided McNemar
$p = 0.8679394004284404$。据此只能说没有检测到显著差异，并且 O2 表现出较强的 Long
portability；不能说 O2 改进了 Long，也不能说已经通过 $\delta = 2\%$ non-inferiority。

合同要求按 task 分层、保持相同 initial-state pair 的 bootstrap CI，并以 95% CI 下界
$> -2\,\mathrm{pp}$ 为通过条件。当前 mirror 未包含逐 task paired outcomes，因此 formal
non-inferiority 状态为 **pending**。权威汇总见 [`model3_o2/Long.md`](model3_o2/Long.md)。

### 4.3 O2 Spatial Historical Comparison

O2 Spatial 的 predeclared 5K/10K checkpoints 分别达到 481/500（96.2%）和
489/500（97.8%），10K 被选中。固定 Model3 Spatial 60K 历史结果为
488/500（97.6%），所以观测差仅为 $+0.2\,\mathrm{pp}$。

Model3 结果曾通过 strict finalization，但成功 eval 目录与 episode ledger 后来被删除；当前
只能登记为 `recorded_not_locally_auditable`。因此无法重建 paired McNemar 或 stratified
paired CI，不能把多出的 1 次成功写成 superiority。权威汇总见
[`model3_o2/Spatial.md`](model3_o2/Spatial.md)。

## 5. External Evidence Tension

### 5.1 DeVA

DeVA paper-reported RoboCasa ablation：

| Variant | Success |
|---|---:|
| Action only | 19.8% |
| Goal-image prediction | 25.8% |
| Future video + unified backbone | 36.8% |
| Future video + decoupled multi-level transfer | 66.0% |
| + affordance/depth guidance | 72.0% |

这些结果同时改变 expert、interface 与 capacity，支持 future modeling 的价值，但不能单独
证明 decoupling 或某条 gradient route 的因果作用。

### 5.2 VidMan

VidMan paper-reported CALVIN ablation：

| Variant | Avg. Len. |
|---|---:|
| Stage 2 $L_{\mathrm{video}} + L_{\mathrm{action}}$ | 2.70 |
| Stage 2 action-only | 3.42 |
| Frozen VDT + adapter/head | 2.98 |
| Action loss 更新 VDT + adapter/head | 3.42 |

DeVA 与 VidMan 说明 future-video objective 和 gradient route 可能影响控制，但两篇论文的
backbone、数据、action interface、compute 与 schedule 并不匹配。由于当前主线优先回答
temporal/depth/interface/adaptation path，这组证据只支撑最后的 minimal A/R sanity checks，
不再支撑前置的大规模 A/R discovery matrix。

## 6. Closed Interpretations

以下解释不再作为当前 Proposal 的扩展方向：

- cadence 不是历史 Model3/Light-WAM Object gap 的主因；
- Light-WAM 与 Model3 的 direct action prefix 差异不能继续作为主要解释；
- frozen spatial C3/C3-add 缺少 correspondence control，不继续扩展；
- flow/regression 的互补失败集合不足以支持双头 router；
- Regression 与 flow 来自独立训练路径，不能将差异解释为 decoder-only；
- I-003 的旧方法新颖性已被 Light-WAM 覆盖，不因本 Proposal 重新激活。
- output-space endpoint rank-$8$ residual 的 input/task-conditional 结构只作为
  LRD-WAM 诊断证据保留；G2-R2 与 D1/P5 容量对照说明原 future-field、
  residual-specific 方法主张尚未成立，但不足以证明 D1/current-only 辅助信息在闭环
  必然没有价值。任何小型闭环必须使用独立 current-only 合同，且不能继承原方法主张。

## 7. New-Paper Intake Template

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
接口读取哪些 layers / timesteps / tokens：
推理是否运行 generator / future branch：
trainable params / GPU hours / memory / latency：
最关键消融：
消融是否匹配参数、训练预算和推理协议：
能够支持的结论：
不能支持的结论：
与 Matrix D/L/C/B 或 A/R sanity 的关系：
证据来源：用户已读笔记 / AI 预读 / 原论文 / 外部检索：
```

## 8. Audit Checklist: Explicitly Out of Scope

- 不把 EnFold 放入主 in-place adaptation matrix；
- 不创建 EnFold-style Matrix E，或重复提出 ASDB/current-only student；
- 不把旧 Model3 flow 结果当作新 carrier 的 A1；
- 不用相同 steps 冒充 compute-matched；
- 不把 checkpoint 当作独立 seed；
- 不把 McNemar $p > 0.05$ 或“差异不显著”当作 non-inferiority；
- 不把一个 B1 候选称为绝对 minimum；
- 不在 Matrix D 冻结 temporal contract 前运行 L；
- 不让 Matrix L 同时改变 depth、aggregation、PEFT 与 action head；
- 不让 Matrix C 重新选择 layers，或同时改变 aggregation 与 capacity；
- 不在 Matrix L/C 冻结前定义 selected-layer B1；
- 不跳过 PEFT × interface 交互检查；
- 不让 A/R sanity 抢占 D/L/C/B 的主预算，或未经单独批准恢复完整 A/R grid；
- 不以新 predictor、rank、fusion、Action-DiT 或 D2/noisy-future 闭环扩展重新激活
  尚未成立的 output-space endpoint rank-$8$ residual 核心路线；单独冻结的
  D1/current-only pilot 只验证 auxiliary information 的闭环转化；
- 不无条件跑满所有 80K/150K 实验；
- 不用 probe、offline loss 或 gradient cosine 代替闭环成功率；
- O2 主线虽已获批准，但不在缺少 preflight/复现字段时启动 server run，也不借此扩展合同
  之外的新方法 scope。
