# Efficient Video-WM-to-WAM Adaptation

## Research Proposal

> 状态：内部研究 Proposal；尚未启动新训练。  
> 执行约束见 [experiment-contract.md](experiment-contract.md)，文献与历史证据见
> [evidence-and-related-work.md](evidence-and-related-work.md)。

## 1. Problem and Scope

预训练 video diffusion models 为 World-Action Model（WAM）提供了强大的视觉与动态先验。
现有工作已经证明 Video-DiT 可以通过机器人数据和动作目标转化为有效策略，但这些系统
通常同时改变四类因素：future-video supervision 的使用阶段、video/action gradients 的
流向、Video-DiT 的 adaptation capacity，以及 predictive features 暴露给 action policy
的方式。跨论文性能因此只能证明多种完整系统都可行，不能说明各项改动中哪些真正必要。

本文研究受控的 **in-place adaptation**：在线控制时仍保留同一个 Video-DiT，在固定
pretrained backbone、action carrier、数据与评测协议后，依次隔离 supervision schedule、
gradient routing 和 PEFT capacity。核心问题是：

> **When should a Video-WM learn dynamics, and how much of it must be adapted
> for control?**

换言之，我们希望判断 future prediction 应当持续参与控制训练，还是主要承担早期
dynamics acquisition；随后再寻找保持闭环性能所需的最小已测试适配容量与在线接口成本。

本文所称“高效”不是仅减少 trainable parameters。候选方案必须先达到预声明的闭环性能
标准，再联合比较 trainable parameters、accelerator-hours、peak training memory、在线
Video-DiT token/forward cost 与 plan-call latency。

EnFold 属于另一种 fold-out regime：它用 generator computation 监督独立的 current-only
encoder，并在部署时移除 generator。因此，它定义本文的 novelty boundary，而不是
in-place 主矩阵中的 matched baseline。

## 2. Core Hypothesis and Research Questions

本文的待验证统一假设是：

> **Acquire dynamics broadly, specialize control sparsely.**

它不是预设结论。只有当 supervision、routing 和 capacity 的受控结果共同支持时，才能成为
论文主张。

### Primary RQ1：什么时候学习 dynamics，梯度应如何流动？

我们比较 action-only、从头 joint、video warmup 后 action-only，以及 video warmup 后
joint 四种 schedule；随后固定 objective、initialization 和 PEFT capacity，隔离
video/action losses 是否更新共享 PEFT。该问题同时回答：

- 在相同总训练成本下，哪种 schedule 的 success-cost Pareto 最好；
- 在固定 action-stage updates 时，future supervision 是否仍有增量价值；
- control specialization 是否需要 action gradients 进入 Video-DiT PEFT。

### Primary RQ2：至少需要多少 Video-DiT adaptation capacity？

在 RQ1 冻结的 recipe 上，比较 frozen backbone、一个预声明 compact PEFT、一个条件式
bracket，以及当前 PEFT upper bound。目标不是宣称搜索连续空间中的绝对 minimum，而是
确定 **smallest tested viable configuration** 及其性能—成本边界。

### Secondary RQ3/RQ4：收益能否通过更便宜的在线路径保留？

主线成立后，再检查最低 Video-to-Action interface bandwidth，以及 action forward 是否
需要 noisy future slots。两者只界定部署成本，不作为独立机制贡献，也不应先于 RQ1/RQ2
消耗主要实验预算。

## 3. Experimental Design

正式实验先冻结统一载体 `C*`。`C*` 的 backbone、action head、checkpoint、数据、
action contract 和 evaluator 均由服务器证据确定；历史 Model3、Regression 或
`model3_o2` 结果不能自动充当 matched control。

### 3.1 Matrix A：Supervision Schedule

| Variant | Dynamics stage | Action stage | 研究角色 |
|---|---|---|---|
| A0 | none | action-only | 无 robot-video supervision 的对照 |
| A1 | none | joint video/action | 持续 predictive objective |
| A2 | video warmup | action-only | dynamics acquisition 后控制特化 |
| A3 | video warmup | joint video/action | warmup 后继续 predictive objective |

Matrix A 同时报告两个 estimand：

- **Budget-matched**：固定总 accelerator-hour budget，回答给定预算下的最优资源分配；
- **Action-update-matched**：固定 action-stage updates 并单列 warmup 成本，回答 future
  supervision 的增量价值。

两种比较不能互相替代：等预算结果不能直接证明 future supervision 的因果收益，固定
action updates 的结果也不能隐藏额外 warmup 成本。

### 3.2 Matrix R：Gradient Routing

| Variant | Video loss → PEFT | Action loss → PEFT | 研究角色 |
|---|---:|---:|---|
| R0 | ✓ | ✓ | shared-gradient joint |
| R1 | ✓ | ✗ | video 塑造 WM，action 只训练 interface/head |
| R2 | ✗ | ✓ | action-specific WM adaptation |
| R3 | ✗ | ✗ | frozen WM readout control |

R0/R1 是 joint objective 下的 matched pair；R2/R3 是 action-only objective 下的
matched pair。跨 pair 的差异混合了 objective 与 routing，不能单独解释为 gradient effect。

### 3.3 Matrix B：PEFT Capacity

| Variant | Capacity | 角色 |
|---|---|---|
| B0 | frozen WM，仅训练 interface/head | lower-cost control |
| B1 | 预声明 compact PEFT | 首个低成本候选 |
| B1.5 | 仅当 B1 失败时启用的单个 bracket | 定位边界区间 |
| B2 | 当前 PEFT upper bound | performance/cost anchor |

Compact candidate 只有在相对 B2 的 paired 95% confidence interval 下界高于
`-δ` 时才判为 non-inferior。B0 或 B1 成功时，只称为 cheapest/smallest tested
configuration，不使用“绝对最小”表述。

### 3.4 Deployment Boundary

当 A/R/B recipe 冻结后，Matrix C 先在同一 aggregation contract 下比较单层与多层读取，
再固定读取层数比较 compact aggregation；随后执行一次 PEFT × interface 的 2×2 检查，
避免把小 PEFT 与宽接口之间的补偿关系误写成单轴结论。

最后，Matrix D 以 matched code path 比较 current-only 与 policy-owned noisy future
slots。D 只回答额外 temporal canvas 的收益是否覆盖 token、memory 和 latency 成本，
不引入 expert future video、action、reward 或 simulator state。

## 4. Evaluation and Decision Rules

主文采用以下四条判定原则：

1. primary endpoint 是 paired closed-loop task success；offline loss、probe 与 gradient
   statistics 仅作诊断；
2. formal evaluation 为每个 suite 的 10 tasks × 50 trials，并固定 initial states 与
   evaluator；
3. compact variant 使用 G0 预声明的 `δ = 2.0` percentage points 进行 paired
   non-inferiority 判断；“差异不显著”不等于 non-inferior；
4. 所有结论同时报告 success、trainable parameters、accelerator-hours、peak memory 和
   plan-call latency；相同 steps 不视为 compute-matched。

结果对应的论文解释如下：

| 结果模式 | 可支持的解释 |
|---|---|
| staged 优于 action-only/joint | future prediction 主要用于 dynamics acquisition |
| persistent joint 优于 staged | predictive objective 在 control specialization 中仍有必要 |
| gradient isolation 优于 shared gradients | representation learning 与 control specialization 存在优化干扰 |
| compact PEFT/interface 保持性能 | dynamics acquisition 与在线控制所需 capacity 可以解耦 |

若 A/R 没有稳定规律，但 B 显示显著 Pareto 优势，论文降级为 parameter-efficient
adaptation study；若机制规律与效率边界都不成立，则停止方法扩展并报告 negative study。

## 5. Expected Contributions

理想结果支持三项贡献：

1. **Controlled fine-tuning study**：在同一 Video-DiT 和 action carrier 上隔离
   supervision schedule 与 gradient routing；
2. **Efficient adaptation frontier**：给出闭环性能相对 parameters、training compute、
   memory 与 latency 的 Pareto，并定位最小已测试可行 PEFT 区间；
3. **Deployment boundary**：说明保持主收益所需的接口带宽，以及 noisy temporal slots
   是否必要。

如果 staged + sparse specialization 得到共同支持，paper-facing claim 才可写为：

> Future prediction is most useful as a dynamics-acquisition stage rather than
> a permanently coupled control objective; subsequent action specialization
> requires only sparse Video-DiT adaptation and a compact online interface.

若 persistent joint 或其他 routing 胜出，则按真实结果重写，不强行维持上述叙事。

## 6. Execution Roadmap

```text
G0: freeze carrier and evaluation contract
G1: run Matrix A + R and decide whether a mechanism insight exists
G2: if G1 succeeds, run Matrix B, then secondary C/D deployment checks
```

- G0 未完成，不启动 A/R/B；
- A/R 没有形成可复现规律时，不扩展新的 gradient mechanism；
- B 只允许一个条件式 B1.5，不展开无边界的 rank × layer 搜索；
- C/D 只能在 A/R/B recipe 冻结后执行；
- 新训练与 server-side handoff 仍需用户明确批准。
