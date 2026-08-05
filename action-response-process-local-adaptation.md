# 动作响应过程监督的局部 Video-DiT-to-WAM 适配

## 候选方法方案

> 记录日期：2026-08-03  
> 当前状态：**Gate 0 v1 的合同性 No-Go 保留；Gate 0B 已终态验证动作响应条件期望可学习；successor 合同 `specs/21-model3-o2-dynamic-response-prewarm.md` 已冻结，但实现、训练和闭环均未开始。**  
> 研究核心：在受限训练和部署预算下，把预训练 Video-DiT 低成本适配成高性能 WAM。动作响应监督和局部训练是实现手段，不以最少参数或新 readout 为主贡献。

## 0. 一句话方法

> 从同一个物理状态执行多个动作，用它们造成的真实未来过程差异监督 O2 已读取的少数 Video-DiT 层；训练时用动作查询这些差异，部署时删除查询头，只保留当前状态适配模块和原 O2 动作接口。

暂用描述 **动作响应过程监督的局部适配**。在闭环结果和完整 novelty check 之前，不固定英文方法名，也不使用“因果动力学”“最小充分状态”或“首个”等强表述。

## 1. 核心问题与假设

### 1.1 核心问题

预训练 Video-DiT 可能已经包含丰富的视觉、运动和未来信息，但这些信息主要按“接下来通常发生什么”组织，不一定按“改变机器人动作会怎样改变未来”组织。

本文要回答：

> **能否利用少量同状态多动作未来，把 Video-DiT 的当前状态特征改造成足以回答动作响应问题的表示，并通过局部训练避免完整大模型的端到端反传？**

### 1.2 待验证假设

- Video-DiT 不一定缺少动作信息，真正缺少的可能是便于动作头利用的动作响应结构；
- 同状态多动作的未来差异比单条真实未来更能隔离动作造成的变化；
- 保留时间顺序的变化过程比时间平均后的终点差异更有利于接触、抓取和滑动控制；
- 第 8、16、24 层的局部适配可能在较低反传成本下接近全层适配的闭环性能。

以上均不能由直觉或离线损失直接证明，必须通过 matched closed-loop 对照验证。

## 2. 实验载体边界

### 2.1 复用 O2 架构，不复用已训练 O2 作为方法起点

第一版复用 O2 的固定结构：

- `Wan2.1-T2V-1.3B` Video-DiT；
- 第 8、16、24 层特征接口；
- 已注册的 O2 多层组合方式；
- 16 层 Action-DiT、动作 flow-matching 目标与 solver；
- 相同的 action horizon、执行/重规划节奏、归一化和评测协议。

但正式效率比较中，所有内部方法必须从同一个预训练 Wan 起点和同一个动作头初始状态开始。已经训练成 WAM 的 O2 checkpoint 只作为性能上限，不能作为本文方法的初始化；否则实验只能说明“如何继续修改一个 WAM”，不能说明“如何把 Video-DiT 低成本适配成 WAM”。

### 2.2 固定动作接口

第一轮不改变 O2 的读取层、query 数、组合方式、动作头宽度和推理 solver。本文只研究：

1. 用什么未来目标监督当前状态适配模块；
2. 是否可以只在局部反传；
3. 在闭环性能不明显下降时实际节省多少总适配成本。

## 3. 同状态多动作数据

### 3.1 分支轨迹

若 LIBERO/robosuite wrapper 能保存并恢复完整 simulator state，则从同一个物理状态执行：

1. 专家动作 `a`；
2. 小幅位置或旋转扰动 `a + delta_a`；
3. 夹爪开合扰动；
4. 零动作 `a_0`。

每条分支使用相同控制步数、观测频率和相机状态。第一轮在 Object 中选择少量任务和数百个状态，优先覆盖接近、接触、闭合、抬升、滑动和释放阶段。

“同状态”必须指完整 simulator state 相同，而不是两张图像看起来相似。若当前环境不能可靠保存和恢复状态，本方法的干预性主张暂不成立，只能降级为动作对比监督。

### 3.2 数据成本

分支 rollout、环境重置、未来特征计算和缓存都计入总适配成本。不能只报告反传时间，而忽略额外轨迹和 teacher forward。

## 4. 动作响应过程目标

### 4.1 未来教师只负责测量

冻结 Video-DiT，在固定 layer、denoising timestep、噪声和时空布局下编码每条真实未来：

$$
H_l(y_a),\quad H_l(y_{a+\delta a}),\quad H_l(y_{a_0}).
$$

teacher 不接收梯度，其结果提前缓存。future teacher 本身不是创新；本文研究的是由同状态多动作关系定义的监督目标。

### 4.2 保留时间顺序

不能一开始把整个未来平均成一个向量。第一版将相同执行时长按控制步对齐，并压缩为四个固定阶段：

```text
动作开始 -> 接近/接触 -> 物体响应 -> 动作结束
```

对每层得到：

$$
H_l(y_a)=
[h^a_{l,1},h^a_{l,2},h^a_{l,3},h^a_{l,4}].
$$

时间压缩方式在结果前冻结；空间 token 可以统一压缩，但不得跨四个时间阶段做全局平均。

### 4.3 两类差分

以零动作作为全局响应参照：

$$
R_l(s,a)=H_l(y_a)-H_l(y_{a_0}).
$$

以小扰动作为局部敏感性参照：

$$
D_l(s,a,\delta a)=H_l(y_{a+\delta a})-H_l(y_a).
$$

默认将小扰动差分作为主要目标，零动作差分作为辅助锚点。原因是“运动减静止”可能过于容易，而相邻动作之间的差异更接近控制决策真正需要区分的变化。

如果时间平均目标与过程目标表现相同，只能将方法称为“动作结果监督”，不能声称学到了动作响应过程。

## 5. 两个明确分开的模块

### 5.1 部署时保留的当前状态适配模块

在第 8、16、24 层分别加入小型残差适配模块：

$$
B_l(s)=h_l(s)+A_l(h_l(s)).
$$

`A_l` 只接收当前状态特征，不接收动作。`B_l(s)` 同时用于训练期局部任务和原 O2 动作接口。

第一版默认把 `B_l` 作为该层暴露给 O2 接口的适配特征，但不把它重新写回后续 Video-DiT blocks。这样三个局部模块可以独立训练，避免串行组合漂移；后续若研究 in-place 回写，必须作为新的独立变量。

`B_l(s)` 的操作性含义是：一个小预测头应能借助它回答同一状态下多个动作的响应。不能仅凭可视化宣称它已经表示了“可控性结构”。

### 5.2 仅训练期存在的动作响应预测头

每层增加一个小预测头：

$$
\widehat R_l(s,a)=Q_l(B_l(s),a).
$$

只有 `Q_l` 接收动作。`Q_l` 用于预测四阶段动作响应，并通过它把监督梯度传给 `A_l`。训练完成后删除 `Q_l`，保留 `A_l` 和 `B_l`。

为避免 `Q_l` 只根据动作分布猜结果：

- `Q_l` 的容量必须小且所有变体参数匹配；
- 数据同时包含同状态不同动作、不同状态相似动作；
- 必须运行 `Q(a)` 的 action-only 诊断；
- 必须运行状态错配诊断 `Q(B(s'),a)`。

若 action-only 或状态错配模型接近完整模型，说明 `A_l` 没有学到状态依赖的动作响应基础。

## 6. 局部训练目标与梯度路径

每层局部过程损失可以写为：

$$
\mathcal L^l_{\mathrm{anchor}}
=
\left\|
Q_l(B_l(s),a)-\operatorname{sg}(R_l(s,a))
\right\|_2^2,
$$

$$
\mathcal L^l_{\mathrm{local}}
=
\left\|
Q_l(B_l(s),a+\delta a)-Q_l(B_l(s),a)
-\operatorname{sg}(D_l(s,a,\delta a))
\right\|_2^2.
$$

总局部目标为：

$$
\mathcal L_{\mathrm{response}}
=
\sum_{l\in\{8,16,24\}}
\left(
\mathcal L^l_{\mathrm{local}}
+\beta\mathcal L^l_{\mathrm{anchor}}
\right).
$$

训练局部模块时：

- 在 `h_l(s)` 处停止梯度；
- 只更新当前层 `A_l` 和 `Q_l`；
- 不更新 Wan base，不让局部损失穿过上游 blocks；
- 各层使用相同的 adapter 和 predictor 规格。

局部损失完成后删除 `Q_l`，用 `B_8/B_16/B_24` 训练固定 O2 动作头。默认先冻结 `A_l`；只有在动作头训练后闭环明显不足时，才允许一个预声明的短整体校准阶段。

## 7. 完整训练流程

### Stage 0：数据与信号门

1. 验证 simulator state 可以完全恢复；
2. 采集小规模同状态多动作分支；
3. 缓存第 8、16、24 层四阶段 future states；
4. 检查同状态动作差分是否显著高于 teacher/noise 重复误差；
5. 检查动作差分是否只由机械臂外观运动主导。

若动作差分不稳定，或不同随机 teacher/noise 的变化与动作变化同量级，则停止方法开发。

### Stage 1：局部动作响应训练

分别训练 `A_8/Q_8`、`A_16/Q_16`、`A_24/Q_24`。所有 future features 从缓存读取，不运行 teacher backward。

### Stage 2：动作头训练

删除 `Q_l`，固定 O2 readout 和 Action-DiT 架构，从共同初始化训练动作头。第一轮冻结 `A_l`，避免把局部方法重新变成全局端到端训练。

### Stage 3：短整体校准

只有 Stage 2 不能达到预声明闭环门槛时才运行。校准预算上限默认不超过 O2 全层训练 GPU-hours 的 10%；超出即判定局部训练未形成有效成本优势。

### 部署

```text
当前观测
  -> 一次 Wan 前向
  -> 第 8/16/24 层当前状态适配 A_l
  -> 原 O2 多层接口
  -> 原 O2 Action-DiT
  -> action chunk
```

部署时没有真实未来、分支动作、future teacher 或 `Q_l`，也不增加第二套动作头。

## 8. 基线与消融

### 8.1 先比较监督目标

所有变体使用相同的 Wan 起点、动作头初始化、8/16/24 adapters、多动作数据、teacher forward 数、参数量和训练预算。

| ID | 监督目标 | 研究作用 |
|---|---|---|
| T0 | 只有动作监督 | 判断是否根本不需要未来监督 |
| T1 | 直接预测普通未来过程特征 | 普通 future-teacher 基线 |
| T2 | 时间平均后的动作结果差分 | 判断终点结果是否已经足够 |
| T3 | 四阶段动作响应过程差分 | 候选主目标 |

T1 也必须使用相同分支数据和相同 teacher 计算次数，只是不构造动作差分，避免把 T3 的收益误归因于额外数据。

### 8.2 再比较梯度路径

| ID | 监督目标 | 训练方式 | 研究作用 |
|---|---|---|---|
| G0 | T3 | selected adapters 端到端反传 | 性能参考 |
| G1 | T3 | 第 8/16/24 层局部训练 | 候选完整方法 |
| G2 | T3 | 局部训练 + 短整体校准 | 检查低成本校准能否修复差距 |
| U | O2 原训练方案 | all-layer PEFT | 强性能/成本上限 |

### 8.3 防 shortcut 诊断

| ID | 输入 | 目的 |
|---|---|---|
| D0 | 只有动作 `Q(a)` | 检查预测头是否只记动作分布 |
| D1 | 错配状态与动作 `Q(B(s'),a)` | 检查表示是否包含状态依赖关系 |
| D2 | 正确状态与打乱时间顺序的目标 | 检查过程顺序是否真正被使用 |

进入论文完整实验后，应增加参数匹配的 AGRA-like 语义对齐目标，判断收益是否来自动作响应关系，而不是普通语义特征对齐。Light-WAM 作为同类外部强参考；EnFold 改变 teacher/student 和部署路径，不进入内部 matched matrix。

## 9. 评测与成本合同

### 9.1 主要终点

- paired closed-loop task success；
- 第一阶段 Object，成功后再验证 Long；
- 每个 suite 固定任务、initial states、动作噪声、solver 和终止规则；
- 默认沿用预声明的 `delta = 2 percentage points` 非劣界，正式运行前再次冻结。

离线差分预测误差、动作-未来匹配和表示可视化只作诊断，不能替代闭环。

### 9.2 过程与鲁棒性诊断

- 接触前后动作响应区分；
- 抓取成功、抓空、滑动和错误方向的区分；
- 时间阶段打乱后的性能下降；
- 背景或光照变化下的稳定性；
- 动作小扰动下表示与动作输出的合理变化。

### 9.3 总成本

每个变体必须报告：

- 分支轨迹数量与 simulator 时间；
- teacher forward 与缓存成本；
- Wan backward blocks、backward FLOPs 和每步时间；
- trainable parameters、optimizer state 和峰值显存；
- 总 GPU-hours 与达到目标成功率所需 GPU-hours；
- 部署一次 plan 的 forward 数、延迟和吞吐。

“相同训练步数”不能称为相同成本。本文的效率结论必须计算数据采集、teacher 缓存、局部训练、动作头训练和整体校准的总和。

## 10. 决策规则

| 结果 | 允许的结论 |
|---|---|
| T3 > T2 且 T3 > T1 | 时间化动作响应监督具有独立价值 |
| T3 约等于 T2 | 只能称为动作结果监督，不能声称过程建模 |
| T3 约等于 T1 | 动作差分没有证明优于普通未来蒸馏 |
| G1 对 G0 non-inferior 且成本明显更低 | 局部训练能保留目标收益 |
| G2 才成功且校准成本不超过上限 | 允许称为局部预训练 + 短校准 |
| G2 校准消耗大部分节省 | 不得声称低成本局部适配 |
| G1/G2 对 U non-inferior，且总成本至少约降低 2 倍 | 支持低成本 Video-DiT-to-WAM 适配主张 |
| D0/D1 接近完整模型 | 状态适配模块未证明学到状态依赖的动作响应基础 |
| 只改善离线指标、不改善闭环 | 停止方法扩展，保留为负结果 |

## 11. 与现有工作的边界

### EnFold

EnFold 用真实未来 generator states 监督 current-only encoder，保留的是可从当前状态预测的未来生成表示。本文要求当前状态表示支持由动作查询的响应函数，监督来自同状态多动作之间的时间化差异，而不是复制单条真实未来状态。

若 T3 不能优于 T1，这一区别没有实证意义，本方法将退化为 EnFold-like future-state distillation 的变体。

### AGRA

AGRA 通过语义特征对齐改善动作头对交互区域的关注和无关扰动鲁棒性。本文的候选增量是同状态多动作响应过程，而不是语义对齐。正式论文必须加入 AGRA-like matched target，不能只做文字区分。

### Light-WAM

Light-WAM 已覆盖 Wan、LoRA/adapter、未来视频监督、8/16/24 多层读取和直接动作解码。因此“少数 adapter + future loss”不是新贡献。本文只有在 T3 相对普通未来监督有效、G1 相对端到端训练更省且闭环非劣时，才形成独立方法增量。

### VERA

VERA 在图像运动空间学习 action-to-motion Jacobian，并将视频计划翻译为动作。本文不保留视频 rollout 或图像 Jacobian，而是用训练期动作分支塑造 Video-DiT 当前状态接口。若最终方法退化为显式运动到动作翻译器，需要重新检查与 VERA 的重合。

## 12. 与当前研究主线的关系

当前已批准执行顺序仍是：

```text
Model3 O2 -> Matrix D -> Matrix L -> Matrix C -> Matrix B
          -> minimal A/R sanity checks
```

本文档是一条独立的候选方法分支，不自动替换该顺序。当前 D/L/C/B 可以提供：

- 是否保留 noisy future slots；
- 第 8/16/24 层中哪些深度值得读取；
- 固定接口后 selected-layer adaptation 的性能边界。

本分支已经为 Gate 0 v1 和 Gate 0B 分别冻结合同。Gate 0 v1 失败后按原合同在 Stage 1 前停止；Gate 0B 没有事后改变 v1 数值或门槛，而是用多噪声动作差分与 held-out 条件预测重新检验“低 raw ratio 是否等于不可学习”。Gate 0B 通过只允许另行冻结局部 adapter Stage 1，不能直接进入完整集成。

## 13. 已执行的最小路径

1. 已审计并验证 LIBERO/robosuite 完整 simulator state 保存与恢复；
2. 已在四个 Object 任务的 $192$ 个状态上复现专家、局部扰动、夹爪翻转和零动作分支，共 $768$ 条分支；
3. 已缓存原始 Wan 第 8/16/24 层四阶段特征，并比较动作差分与固定 teacher 的替代噪声差分；
4. Gate 0 信号门失败，因此按合同停止，没有运行 `Q(a)` 的 Stage 1 学习、测试集读取或完整集成。

## 14. 终态结果（2026-08-03）

运行证据位于
`runs/I-003/action_response_local/20260803_gate0_stage1_v1/`。状态恢复、
重复 rollout 和像素复现误差均为 0，且四个任务的晚期目标物体响应比例均为
$1.0$，说明同状态动作干预链路成立。

但第 8、16、24 层的 pooled action/noise ratio 中位数分别只有 $0.6087$、
$0.5967$、$0.6351$，对应 bootstrap $95\%$ 区间下界分别为 $0.5267$、
$0.5319$、$0.5767$。四个任务在三个层上全部未达到预先冻结的
median $\ge 2.0$、lower CI $>1.0$ 门槛。

唯一合同结论为 `no_go_complete_o2_integration`。它只表示 v1 的冻结授权门没有
通过；不能据此证明监督不可学习。Stage 1 未执行是 v1 硬门控的预期终态，不是
缺失实验。后续 Gate 0B 保留该 artifact，并独立检验其科学解释。

## 15. Gate 0B 条件可预测性结果（2026-08-03）

Gate 0B 运行位于
`runs/I-003/action_response_local/20260803_gate0b_conditional_predictability_v1/`。
它复用 $192$ 个状态和 $768$ 条分支，对每条未来使用四个拟合噪声和两个封存
测试噪声；同一状态、阶段和噪声下的所有动作分支严格共享噪声。

审计确认 Gate 0 v1 的四阶段和全局投影与原 Stage 1 目标形状一致，但 v1
ratio 没有使用 train-split target 标准化，并在差分前全局平均 $14\times28$
tokens。其 noise denominator 虽然只改变 seed，却测量 expert 绝对特征变化，
不是 $\Delta$ 本身的跨噪声误差。因此 v1 ratio 不是直接的 learnability test。

Gate 0B 在 $128/32/32$ 个 train/validation/test states 上完成 $144$ 个冻结 Wan
的小预测器工作项。唯一一次测试使用未见状态与 seed $84005/84006$。原 Stage 1
标准化全局空间 E0 中，Full 相对 Action-only、State-only、Shuffled 的 pooled
MSE 改善为 $14.99\%$、$9.40\%$、$14.79\%$，bootstrap $95\%$ 下界分别为
$12.76\%$、$7.86\%$、$13.11\%$；三个层和三个训练 seed 全部通过。

终态决定为 `proceed_stage1_exact_space`。这证明固定 timestep $250$ 下的 E0
动作响应条件期望可学习，否定“raw ratio 低所以监督不可学习”的广泛推断。
E1 局部网格虽然有正向信号，但相对 Action-only 和 State-only 的 pooled 改善
只有 $2.85\%$ 和 $3.72\%$，未达到 $5\%$ 门槛，因而不支持替换 E0。

Gate 0B 没有训练 adapter、动作头或完整策略。下一步若执行，必须围绕 E0、
四噪声均值、相同控制和新的未见证据边界冻结独立 Stage 1 合同。

## 16. Successor 合同修订（2026-08-03）

用户已经确认新的 `model3_o2_dynamic` 两阶段合同。它不覆盖本文早期的候选流程，
而是把 Gate 0B 的授权落实为一个单独注册的 treatment。

原 Model3 O2 已经包含共享 parent 和 gate-stage 两段：先由 Model3 联合训练得到
Object step $20\text{K}$ query-pretrained parent，再新增 exact-q3 O2
layer-aware gate 并执行 O2-local 联合训练。Dynamic treatment 在这两段之间插入
response warmup：

```text
共享 Model3 Object step-20K parent
-> 构造并冻结与 Base 相同初始化的 O2 gate
-> response Stage 1: 只训练 A8/A16/A24 和 Q8/Q16/Q24
-> 删除 Q
-> O2 gate-stage Stage 2: 恢复原 video + action 联合训练
```

Stage 1 使用一个 LIBERO/robosuite simulator、$5{,}000$ 个确定性运动感知 source
states、每个状态四条分支、共 $20{,}000$ 条 trajectories、约 $1.28$M
per-camera frames，并固定训练 $5\text{K}$ optimizer steps。source-state selection
必须覆盖 object motion、robot motion、wrist-camera motion 和 contact transition，
静态控制样本占比受限，不允许均匀随机采样。

Stage 2 完整保留原 O2 全层 PEFT、query、gate、Action-DiT、future-video loss 和
action loss。保存完整 O2-local step-$5\text{K}$ checkpoint 后解冻 response
adapters；其学习率固定为同期其他 PEFT 的 $0.1$，原 optimizer、scheduler、
dataloader 和 RNG 不得重置。第一轮 Object treatment 匹配 Base 的
$35\text{K}$ O2-local budget 和 $\{10\text{K},20\text{K},35\text{K}\}$
评测集合。

唯一完整合同、hard legality checks、tensor/gradient 语义、证据目录和主张边界
以 `specs/21-model3-o2-dynamic-response-prewarm.md` 为准。
