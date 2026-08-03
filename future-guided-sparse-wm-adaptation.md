# Future-Guided Sparse WM Adaptation

## 候选架构说明

> 记录日期：2026-08-02  
> 当前状态：**用户认可并要求落盘的候选研究想法；尚未替换已批准的 Model3 O2 主线，也未授权新训练。**  
> 研究核心：在受限训练与部署预算下，高效微调预训练 Video-WM，使其获得有竞争力的 WAM 闭环控制能力。稀疏选层是实现该目标的机制和成本轴，不是为了追求“参数最少”而牺牲控制性能。

## 0. 一句话想法

> 使用训练期真实未来作为 privileged supervision，定位并训练少数真正需要机器人化的 Video-WM 层；部署时删除 future-aware 分支，只保留一次 Wan 前向、少量 adapter 和原 Model5 动作接口。

暂用名称 **Future-Guided Sparse WM Adaptation**。在得到闭环正结果和完整 novelty check 前，不固定论文方法名或首创性表述。

## 1. 为什么从 LRD-WAM 转向这条路线

已有 LRD-WAM Gate 结果支持两条不同结论：

1. Wan 到 LIBERO 真实未来目标的逐样本 residual 确实具有明显低秩结构；
2. 真实未来 residual 的 oracle 表示含有很强动作信息，但 predicted low-rank residual 不能跨 Object/Long 稳定成为动作 code，也不能稳定优于参数匹配的普通 side reader。

因此，当前证据不支持继续把“显式预测低秩视频 residual，并让动作头读取该 residual”作为核心方法。更合理的保留方式是：

> 真实未来能够揭示当前 WM 内部缺失的控制相关信息，但未来信息应当用于指导 WM 的内部适配，而不是在部署时继续预测一个显式 LRDelta。

这条路线不删除旧 LRD-WAM 负结果，也不把 Gate1 的低秩现象改写成已验证的方法依据。旧实验仍是“为什么不直接使用 residual code”的证据边界。

## 2. 核心研究问题与假设

核心问题不是“哪种 readout 更漂亮”，也不是“绝对最少能训练几个参数”，而是：

> **在固定动作接口和明确资源预算下，训练期未来信息能否识别并修正 Video-WM 中少数控制相关计算，使较小的 WM 适配范围达到 all-layer PEFT 的闭环性能？**

工作假设为：

- 预训练 Wan 已包含大部分视觉、语义与运动先验；
- 机器人数据不一定需要重写全部 Video-DiT，只需修正少数中后层或跨层计算；
- 真实未来提供的价值不是一个可直接部署的 residual code，而是指出当前表示缺少哪些动作相关动力学信息；
- future-aware supervision 与 action loss 共同作用时，少量 adapter 可能学到比固定选层或普通稀疏 PEFT 更有效的修正。

这些均是待验证假设，不能由 Gate1/Gate2 直接推出。

## 3. 架构

### 3.1 保留 Model5 动作主干

主实验以当前真实代码审计后的 Model5 为准。对话中的暂定合同是：

```text
current clean latent + one Gaussian future-noise slot
                    │ temporal timestep [0, 1000]
                    ▼
                   Wan
                    │
          hidden states H8/H16/H24
                    │
          recurrent 64-query readout
                    │
               action memory
                    │
               Action-DiT
                    │
               action chunk
```

第一版固定 `H8/H16/H24 → recurrent queries → Action-DiT`，不同时发明新的 readout。这样闭环差异才能主要归因于 WM adaptation，而不是接口容量变化。

### 3.2 少数层 gated adapters

将 Wan blocks 划成少量预声明候选组，在每组代表层或候选层加入轻量 residual adapter：

$$
H'_l = H_l + s_l A_l(H_l),
$$

其中 \(A_l\) 是小型 adapter，\(s_l\) 是可学习 gate。Wan 原始权重默认冻结；adapter、gate、现有 query readout 与 Action-DiT 按实验合同训练。

gate 的作用是预算约束下的结构选择，而不是再构造一套动作接口。候选组数、adapter width、gate 参数化和可训练 block 必须在正式结果前冻结。

### 3.3 训练期 future-aware 分支

同一个训练样本增加一个仅训练期存在的 privileged pass：

```text
Teacher view: current latent + real future latent
                         │
                 shared / EMA Wan pass
                         │
               future-aware targets T_l
                         │ stop-gradient
                         ▼
Student view: current latent + Gaussian future slot
                         │
                sparsely adapted Wan
                         │
               student states S_l
```

teacher 不必复制一整套独立 Wan，但 teacher view、noise/timestep、共享权重或 EMA 方式必须在实现前冻结。真实未来只能生成 stop-gradient 监督目标，不得进入部署动作前向。

为避免逼迫 student 恢复当前观测中不可知的全部未来细节，默认不逐元素对齐完整 Wan hidden。优先对齐经过轻量投影后的控制相关摘要：

$$
\mathcal L_{\mathrm{align}}
=
\sum_l s_l
\left\|
P_l(S_l)-\operatorname{sg}(P_l(T_l))
\right\|_2^2.
$$

`P_l` 的目标空间仍需通过代码与实验合同确定。候选包括压缩 hidden summary、query memory 或相同 noisy action 上的 action-flow target；第一轮只选一种主目标，不能把多种蒸馏 loss 一起加入后再归因。

### 3.4 总训练目标

候选目标为：

$$
\mathcal L
=
\mathcal L_{\mathrm{action}}
+\lambda_v\mathcal L_{\mathrm{video}}
+\lambda_f\mathcal L_{\mathrm{align}}
+\lambda_b\sum_l c_l |s_l|.
$$

- `L_action`：保留 Model5 的 action flow-matching objective；
- `L_video`：保留与 matched baseline 一致的 future-video supervision；
- `L_align`：将训练期 future-aware 信息写入 student adapters；
- `L_budget`：按 adapter 参数、反传成本或实测训练成本惩罚同时开启过多层。

必须记录每条 loss 到 Wan base、adapter、gate、query readout 和 Action-DiT 的真实梯度路径。`c_l` 不能只用参数量代替全部效率成本；至少同时报告 backward depth、optimizer state、峰值显存与每步耗时。

## 4. 选层与正式训练流程

层选择不能退化为不断手工试层，也不能只按 gate 数值下结论。

### Stage S0：冻结 carrier

记录当前 Model5 的仓库 commit、checkpoint、数据、action contract、solver、评测 initial states、H8/R8、video/action loss、实际 hidden shapes 和梯度路径。历史 Model5 数字只能作为参考，不能自动充当新实验 matched baseline。

### Stage S1：受预算约束的稀疏发现

1. 将 Wan 划分为约 6 个预声明候选层组；
2. 在候选组中放置相同规格 adapter 与 gate；
3. 用短预算运行 `action + video + future-align + budget`；
4. 记录 gate、adapter norm、梯度、训练成本和小规模闭环信号；
5. 对候选组做 leave-one-group-out 删除干预。

层重要性至少同时参考：

- 删除该 adapter 后的闭环或 action loss 退化；
- future-aware gap 是否缩小；
- 单位参数、显存和 GPU-hour 收益；
- 多 seed 下选择是否稳定。

### Stage S2：冻结稀疏结构并重训

选择少数候选组后，删除其余 adapters。正式比较应从同一个初始 checkpoint 重新训练，而不是直接把 discovery run 的最佳状态当作无偏正式结果。

第一轮只允许一个预声明的 adapter 数量预算，例如保留 2–3 个候选组；若失败，只增加一个相邻容量 bracket，不展开全层组合搜索。

## 5. 部署路径

```text
current observation + language + one Gaussian future slot
                         │
               Wan + selected adapters
                         │
                    H8/H16/H24
                         │
              existing recurrent queries
                         │
                    Action-DiT
                         │
                    action chunk
```

部署时：

- 不使用真实未来；
- 不保留 teacher/EMA 分支；
- 不预测显式 LRDelta；
- 不运行迭代未来视频生成；
- 不增加第二套 action readout；
- 相对 Model5 只保留少数已选择 adapters 的额外计算。

## 6. 必须有的对照

核心内部矩阵固定同一 Model5 readout、Action-DiT、数据和评测：

| Variant | WM adaptation | Future-aware alignment | 作用 |
|---|---|---:|---|
| F0 | frozen Wan；只训练 interface/head | 否 | frozen-WM 下界 |
| F1 | 当前 Model5 all-layer PEFT | 否 | 性能/成本上界 |
| F2 | 固定少数层 adapters | 否 | 手工 sparse PEFT |
| F3 | 参数匹配的随机层 adapters | 否 | 排除“任意少量容量都够” |
| F4 | 稀疏 gate 自动选层 | 否 | 分离 sparse selection 本身 |
| F5 | future-guided sparse selection | 是 | 候选完整方法 |

若 discovery 后只比较 F5 与 F1，不能证明 future guidance、稀疏选择或层位置中的哪一项有效。F2–F4 可先做低成本 screening，但论文主结论至少需要 F1、F4、F5 的 matched closed-loop 比较。

外部系统比较可以包含 Light-WAM，但不能用跨论文数字替代内部 matched controls。方法成立后，可将 selected-adapter recipe 移植到 Light-WAM-style readout，检查它是否依赖 Model5 接口；该移植属于后续泛化，不进入第一轮结构发现。

## 7. 主要指标与决策规则

主要终点是 paired closed-loop success。离线 action MSE、future alignment 和 gate 大小只作机制诊断。

所有变体至少报告：

- WM 内部与总 trainable parameters；
- 实际反传经过的 Wan blocks；
- optimizer state 与 peak training memory；
- samples/second、每步时间和总 accelerator-hours；
- 达到目标成功率所需训练样本与 GPU-hours；
- 部署 token/forward 数与 plan-call latency；
- 每 task、每 initial-state 的 paired outcome 与置信区间。

结果解释必须遵守：

| 结果 | 允许的结论 |
|---|---|
| F5 > F4，且 F5 对 F1 non-inferior、成本更低 | future guidance 帮助找到控制有效的低成本 WM 适配子集 |
| F4/F5 都约等于 F1，但彼此无差异 | 稀疏适配可行；future guidance 未证明有独特价值 |
| F5 > F4，但成本不优于 F1 | future-aware supervision 可能有效；不能声称 adaptation efficiency |
| F2/F3/F4/F5 相近 | 层位置/自动选择未证明重要，收益可能只是额外 adapter 容量 |
| 选择层跨 seed 不稳定 | 只能报告 budgeted sparse adaptation，不能声称存在统一控制关键层 |
| 仅 Model5 readout 上成立 | 结论必须限定为 interface-dependent |

## 8. 与 Light-WAM 的边界

两者处于同一技术家族，都使用预训练 Video-WM、机器人数据、future-video supervision、PEFT 和 hidden-to-action 接口，因此不能仅凭“adapter + future loss”声称新方法。

区别必须由研究变量和证据体现：

| 维度 | Light-WAM | 本候选路线 |
|---|---|---|
| 主要问题 | 构建整体高效的 WAM 系统 | 在固定接口与资源预算下，WM 应在哪里、如何被适配 |
| WM adaptation | 固定 PEFT/adapter recipe | future-guided、预算约束的稀疏层选择与适配 |
| Future signal | future-video training objective | video objective + future-aware internal target，用于定位适配子集 |
| Readout | StateFusionActionExpert 是系统设计的一部分 | 第一阶段固定 Model5 readout，隔离 WM adaptation |
| 主要输出 | WAM success/latency/总成本 | success–WM-adaptation-cost Pareto 与层选择稳定性 |

若 F5 不能稳定优于参数匹配的 F4，这条路线只能作为 Light-WAM 风格 PEFT 的缩减实验，不能形成独立方法贡献。

## 9. 与当前 O2/D/L/C/B Proposal 的关系

当前已批准主线仍是：

```text
Model3 O2 -> Matrix D -> Matrix L -> Matrix C -> Matrix B
          -> minimal A/R sanity checks
```

本文档记录的是一条 **Model5-carrier 候选方法分支**，不会静默替换上述执行合同。两者存在明显交叉：

- Matrix D 决定 noisy future slot 是否值得保留；
- Matrix L/B 通过顺序实验先定位可读层，再测试 selected-layer PEFT；
- 本路线则让 future-aware supervision 在预算约束下参与适配层选择。

正式激活前必须选择一种关系：

1. **后续方法分支**：先完成当前 D/L/C/B，再用其结果定义候选层组，测试 future guidance 是否优于普通 sparse selection；或
2. **独立 Model5 carrier**：重新执行 G0，并用 F0–F5 建立独立 matched contract。

不得把 Model3 O2 与历史 Model5 的现成数字拼接成同一因果矩阵，也不得同时改变 carrier、readout、adapter、teacher target 和 action head 后把收益归因于 future-guided adaptation。

## 10. 最小下一步

在申请训练前只做以下准备：

1. 审计当前 Model5 代码与真实 checkpoint，确认单 future slot、timesteps、H8/H16/H24、query memory、video/action loss 和梯度路径；
2. 冻结 6 个左右候选层组、adapter 规格、teacher view、唯一 alignment target 与 budget cost；
3. 写出 F0/F1/F4/F5 的 matched experiment contract、参数与显存估算；
4. 明确它是接在现有 D/L/C/B 后，还是独立重开 Model5 carrier；
5. 未经用户再次批准，不启动长训练或正式闭环评测。
