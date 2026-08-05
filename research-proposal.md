# Efficient Video-WM-to-WAM Adaptation

## Research Proposal

> 当前执行主线：**Model3 O2 → Matrix D → Matrix L → Matrix C → Matrix B → minimal A/R sanity checks**。
> O2 carrier 已选定；Long non-inferiority 统计子门仍待 paired CI。
> 执行约束见 [experiment-contract.md](experiment-contract.md)，文献与历史证据见
> [evidence-and-related-work.md](evidence-and-related-work.md)。
>
> LRD-WAM“显式低秩机器人视频动力学增量”的原始 future-field 主张尚未成立。G1 的
> input/task-conditional rank-$8$ 结构只保留为诊断发现；G2、G2-R2 与 D1 容量对照
> 不支持把 output-space endpoint residual 作为 residual-specific 的核心部署机制，原
> D2/noisy-future G3 继续停止。这些离线证据尚不足以判定相关 current-only 表示在闭环
> 一定没有价值，因此另行冻结了只测试 D1/current-only auxiliary information 的小型闭环，见
> [d1-current-only-auxiliary-closed-loop-experiment.md](d1-current-only-auxiliary-closed-loop-experiment.md)。历史设计见
> [low-rank-delta-dynamics-wam-experiment.md](low-rank-delta-dynamics-wam-experiment.md)，
> 终态结果见
> [lrd-wam-g2r2-d1-carrier-result.md](lrd-wam-g2r2-d1-carrier-result.md)。
>
> LRD-WAM Gate 结果之后形成的后续候选“训练期未来引导的稀疏 WM 适配”见
> [future-guided-sparse-wm-adaptation.md](future-guided-sparse-wm-adaptation.md)。该文档记录
> Model5-carrier 架构与证伪边界，同样不自动替换或启动当前 O2 主线。
>
> 新候选方法“同状态多动作的动作响应过程监督 + 局部 Video-DiT 适配”见
> [action-response-process-local-adaptation.md](action-response-process-local-adaptation.md)。该方案明确区分
> 部署时保留的当前状态适配模块与仅训练期存在的动作查询头，并将 O2 仅作为架构合同和性能上限；
> 它不替换已批准的 D/L/C/B 顺序，也不自动授权新训练。

## 1. Problem and Scope

预训练 Video-DiT 已经能够作为 World-Action Model（WAM）的强初始化，但现有系统通常
同时改变 temporal latent canvas、hidden-state depth、multi-layer aggregation、Video-DiT
adaptation capacity，以及 video/action objective。跨论文结果因此无法回答：一个强 WAM
究竟依赖哪些 temporal tokens、哪些深度的控制信息，以及哪些层真正需要被适配。

本文固定 **Model3 O2** 作为强 carrier。O2 在 Object、Long、Spatial 上分别达到
492/500、476/500、489/500；Long parent 为 478/500，历史 Spatial parent 为
488/500。固定 carrier 后，研究不再优先搜索 supervision schedule，而是沿着 action
information path 逐级缩小系统：

```text
temporal input
  -> informative depths
  -> layer composition
  -> adapted layers
  -> minimal objective/routing sanity checks
```

核心问题是：

> **Which temporal and depth-wise representation path is actually required
> for control, and how little of the Video-DiT must be adapted once that path
> is identified?**

本文只研究 **in-place adaptation**：部署时仍运行 Video-DiT。EnFold 的 fold-out、
current-only student 与 generator-free deployment 定义 novelty boundary，不进入主矩阵。

“高效”要求候选模型先达到预声明闭环性能，再比较 trainable parameters、
accelerator-hours、peak memory、在线 token/forward cost 与 plan-call latency。

## 2. Research Questions and Priority

### RQ-D：动作前向是否需要 temporal canvas？

首先比较 matched current-only input 与 policy-owned noisy future slots。该实验决定后续
所有层分析应在什么 temporal input contract 下进行，不能放在接口与 PEFT 搜索之后。

### RQ-L：控制相关信息在哪些深度出现？

在冻结 D 结果后，对 O2 已注册的 layers 8/16/24 做 parameter-matched single-depth
readout。Matrix L 的目标是定位 **tested depths** 中的控制可读性与任务互补性，不根据
offline probe 直接宣称某层对闭环必要。

### RQ-C：应该怎样组合这些层？

只在 Matrix L 确认有用的深度上比较 best single layer、simple shared aggregation、
layer-separable composition 与当前 O2 gated residual readout。这样把“读取哪些层”和
“怎样组合”拆成两个问题。

### RQ-B：只适配这些层是否足够？

固定 D/L/C recipe 后，比较 frozen WM、仅适配 L-selected blocks、一个条件式邻域 bracket，
以及 O2 all-layer rank-64 PEFT upper bound。目标是找到 smallest tested viable
selected-layer adaptation，而不是搜索连续空间中的绝对 minimum。

### Minimal A/R：主结论是否依赖 objective 或 gradient route？

A/R 不再是大规模 discovery matrix。最后只运行两个 matched sanity pairs：joint versus
action-only，以及 action gradient 是否进入 selected PEFT。如果 sanity check 反转主结论，
则 D/L/C/B claim 必须条件化；否则不扩展完整 schedule/routing grid。

## 3. Experimental Design

### 3.1 Fixed Carrier：Model3 O2

O2 保持 parent Model3 的 Wan PEFT、future-video loss、16-layer Action-DiT、flow action
objective 与 H8/R8 部署合同，只将 recurrent `q1/q2/q3` trace 改为显式 layer-aware
readout。Long 的 `McNemar p=0.8679394` 表示未检测到与 parent 的显著差异，但不等于已
通过 $\delta = 2\%$ non-inferiority；正式 claim 仍等待 paired CI。Spatial 比历史 parent 多 1 次
成功，但 parent ledger 已删除，所以该差异只能描述为保持 parent-level performance，不能
写成 superiority。

### 3.2 Matrix D：Temporal Canvas

| Variant | Action-feature input | 研究角色 |
|---|---|---|
| D1 | clean current latent，matched timestep/position/code path | current-only control |
| D2 | D1 + policy-owned Gaussian-noise future slots | temporal-canvas treatment |

D1/D2 固定 O2 PEFT、当前 layer-aware interface、action decoder、loss、预算与 solver。
D2 不读取 expert future information，也不运行 iterative video denoising。只有 D2 的闭环
收益稳定且覆盖额外 token、memory 和 latency 成本时，后续矩阵才保留 temporal canvas。

### 3.3 Matrix L：Depth Localization

| Variant | Read depth | Readout contract |
|---|---:|---|
| L8 | layer 8 | matched single-depth readout |
| L16 | layer 16 | matched single-depth readout |
| L24 | layer 24 | matched single-depth readout |
| L-O2 | layers 8/16/24 | registered O2 multi-depth reference |

L8/L16/L24 固定 temporal canvas、PEFT、action head、query 数、hidden width、训练预算与
参数量，只改变 hidden-state depth。Offline probes 可解释信息类型，但正式 localization
由 paired closed-loop success 与 per-task complementarity 决定。

### 3.4 Matrix C：Layer Composition

| Variant | 输入 | Aggregation |
|---|---|---|
| C0 | L 中的 best single depth | single-depth baseline |
| C1 | L-selected depths | parameter-matched simple/shared pooling |
| C2 | L-selected depths | layer-separable compact composition |
| C3 | L-selected depths | O2-style gated residual readout |

Matrix C 不再改变读取深度集合。若 C0 已对 multi-depth variants non-inferior，则优先单层；
若 multi-depth 有收益，再判断收益来自层身份保留还是复杂 gated composition。

### 3.5 Matrix B：Selected-Layer Adaptation

| Variant | Video-DiT adaptation | 研究角色 |
|---|---|---|
| B0 | frozen WM，只训练 interface/head | no-PEFT control |
| B1 | 仅适配 L-selected blocks | selected-layer candidate |
| B1.5 | B1 + 一层预声明相邻 block，仅在 B1 失败时启用 | boundary bracket |
| B2 | O2 all-layer rank-64 PEFT | performance/cost upper anchor |

B1/B1.5 的 block mapping 必须在查看结果前冻结。B0/B1 只有相对 B2 的 paired 95% CI
下界高于 $-\delta$ 时才判为 non-inferior，结论只写 cheapest/smallest tested configuration。

### 3.6 Minimal A/R Sanity Checks

| Pair | 只改变什么 | 目的 |
|---|---|---|
| A-S0 vs A-S1 | registered joint objective vs action-only；固定 action updates | 检查 final recipe 是否依赖 video objective |
| R-S0 vs R-S1 | action loss 是否更新 selected PEFT；interface/head 始终接收 action loss | 检查 selected-layer claim 是否依赖 gradient route |

若 B0 胜出、WM PEFT 全冻结，则 R sanity 记为 not applicable。除非 sanity pair 显著反转
D/L/C/B 结论，否则不恢复原 A0–A3 或 R0–R3 全矩阵。

## 4. Evaluation and Decision Rules

1. Primary endpoint 是 paired closed-loop task success；probe、loss 与 gradient statistics
   只作诊断。
2. Primary contrasts 按顺序为 D2 vs D1、L single-depth comparisons、C variants、
   B-selected vs B2；A/R 是 secondary sanity contrasts。
3. Formal evaluation 为每个 suite 10 tasks × 50 trials，并固定 initial states 与 evaluator。
4. $\delta = 2\,\mathrm{pp}$ 的 non-inferiority 必须由 paired 95% CI 判定；McNemar $p > 0.05$ 不等于
   non-inferiority。
5. 所有变体同时报告 success、parameters、accelerator-hours、memory、token cost 与
   latency；相同 steps 不视为 compute-matched。

| 结果模式 | 可支持的解释 |
|---|---|
| D1 non-inferior to D2 | current-only input 足够，temporal canvas 可删除 |
| D2 稳定优于 D1 且覆盖成本 | noisy temporal canvas 对控制有实际价值 |
| 一个或少数 L depths 保持性能 | 控制信息在 tested depths 中具有局部可读性 |
| multi-depth C 优于 best single | 深度间信息具有闭环互补性 |
| B1 non-inferior to B2 | 只适配 control-relevant blocks 足以保持性能 |
| A/R sanity 反转主结果 | 主结论依赖 objective/routing，必须缩小 claim scope |

## 5. Expected Contributions and Downgrade Paths

理想结果支持三项贡献：

1. **Temporal requirement**：确认强 WAM 是否需要 noisy future slots；
2. **Depth-wise control path**：定位 control-readable depths，并隔离 layer selection 与
   aggregation；
3. **Localized adaptation frontier**：验证只适配这些深度是否能保持 O2 闭环性能。

若 D/L/C 给出稳定结构规律，但 B1 失败，论文仍可定位为 representation/interface study；
若只有 B 显示成本优势，则降级为 parameter-efficient adaptation study；若 D/L/C/B 都
没有稳定规律，则保留为 negative study，不用 A/R 扩展实验数量包装机制贡献。

## 6. Execution Roadmap

```text
G0: O2 fixed as the strong carrier; Long paired-CI certification remains parallel
G1: Matrix D — decide the temporal canvas
G2: Matrix L — localize control-readable depths
G3: Matrix C — compose only the selected depths
G4: Matrix B — adapt only those depths
G5: minimal A/R sanity checks
```

- 每个 Gate 只冻结下一阶段需要的变量；后续矩阵不得回改前序 treatment。
- Matrix L 不扩展为无边界全层搜索；先验证 O2 注册的 8/16/24 depths。
- Matrix C 不同时改变 depth set 与 aggregation。
- Matrix B 失败时只允许一个 B1.5 邻域 bracket。
- A/R 只负责 sanity，不抢占 D/L/C/B 的主预算。
- 用户已批准 O2 主线执行；每次 server launch 仍必须满足
  [experiment-contract.md](experiment-contract.md) 的 preflight 与 artifact contract。
