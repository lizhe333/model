# Low-Rank Delta Dynamics WAM

## 实验 Proposal 与执行合同

> 文档状态：**候选方法分支；评审后已冻结主 carrier 与分阶段实验合同，但不自动启动训练。**
>
> 当前名称：**Low-Rank Delta Dynamics WAM（LRD-WAM）**。
>
> 现有主线仍以 [research-proposal.md](research-proposal.md) 与
> [experiment-contract.md](experiment-contract.md) 为准。本分支首先只运行不训练新
> WAM 的 G0/G1 诊断；D2 是本方法的主 carrier，D1 是 current-only matched control。
> 任何 G2 之后的新训练都需要单独冻结 server carrier 与预算。

## 0. 一句话目标

冻结原始预训练 Wan，不把动作速度错误地写成视频速度的残差，而是在部署一致的
D2 输入上显式学习一个**低秩的机器人视频动力学增量**，并让动作模型读取这一个增量本身：

\[
Y_{\mathrm{robot}}
=
Y_{\mathrm{pre}}+\Delta Y,
\qquad
\operatorname{rank}(\Delta Y)\le r.
\]

最终目标不是“参数少且 LIBERO 分数不错”，而是验证一个可证伪的科学命题：

> 从通用 Video-DiT 到机器人 WAM 的任务增量，是否在视频向量场中具有稳定、可压缩的
> 低维结构；这份增量是否既能修正机器人未来动力学，也足以作为闭环动作生成的控制表示？

## 1. 数学定义与严格边界

### 1.1 基础 checkpoint 与两个 oracle 残差

主方法的 \(V_{\mathrm{base}}\) 只允许是**原始预训练 Wan**。Model3 O2 不作为 base，
只作为强性能 baseline 与次级分析 checkpoint；否则“机器人增量低秩”会被改写成
“O2 之后剩余误差低秩”，两者不是同一个命题。Action-DiT 初始化在所有 matched 方法间统一。

在完全相同的 latent normalization、D2 input、target、token layout 与 flow
parameterization 下，冻结 checkpoint 的 pre-unpatchify 输出分别记为
\(Y_{\mathrm{pre}}\) 与 \(Y_{\mathrm{O2}}\)，真实机器人 target 为 \(Y^*\)。G1 同时计算：

\[
R_{\mathrm{pre}}^*=Y^*-Y_{\mathrm{pre}},
\qquad
R_{\mathrm{O2}}^*=Y^*-Y_{\mathrm{O2}}.
\]

本文关于“从通用视频先验到机器人动力学的低秩增量”的主 claim 只由
\(R_{\mathrm{pre}}^*\) 支撑；\(R_{\mathrm{O2}}^*\) 只回答 O2 适配后还剩下什么结构。
不能拿不同 timestep、target 定义、视频 horizon 或 action endpoint 的输出直接相减。

### 1.2 严格 rank 所在空间：Wan 最终线性投影之前

先做零训练 shape audit，再冻结 rank 定义。Wan head 的 modulated token 记为：

\[
G_0\in\mathbb R^{N\times D},
\qquad
W_{\mathrm{head}}\in\mathbb R^{D\times C_p},
\qquad
Y_{\mathrm{pre}}=G_0W_{\mathrm{head}}.
\]

对 [`model5/third_party/light_wam/src/lightwam/models/wan22/wan_video_dit.py`](model5/third_party/light_wam/src/lightwam/models/wan22/wan_video_dit.py)
与 Wan2.1 loader preset 的静态审计显示，Object carrier 的原始 YAML 仍保留 5B
占位维度 `3072 -> 48`，但 `video_backbone_type=wan2_1_t2v` 会在实例化前严格覆盖为
\(D=1536\)、`out_dim=16`，因此最终线性层输出
\(C_p=16\times1\times2\times2=64\)。G0 必须在真实 batch 上确认该 runtime preset、\(N\)
以及动作 endpoint 对应的 token 索引；不得把未生效的 YAML 占位值 `3072 -> 192`
写入 rank 合同。候选方法不以“LoRA rank 小”间接声称场低秩，而是直接预测
pre-head token correction：

\[
\Delta G_{\phi}=P_{\phi}(h)Q_{\phi}(h)^\top,
\qquad
P_{\phi}(h)\in\mathbb R^{N\times r},
\qquad
Q_{\phi}(h)\in\mathbb R^{D\times r}.
\]

冻结原始 Wan head 后：

\[
Y_{\mathrm{robot}}
=(G_0+\Delta G_{\phi})W_{\mathrm{head}},
\qquad
\Delta Y=\Delta G_{\phi}W_{\mathrm{head}},
\qquad
\operatorname{rank}(\Delta Y)\le\operatorname{rank}(\Delta G_{\phi})\le r.
\]

严格 rank claim 位于 \(N\times C_p\) 的 pre-unpatchify token-output 矩阵。最终
`unpatchify` 会重排张量元素，因此不能把重排后的某个二维 latent view 的数值 rank 也声称为
必然不超过 \(r\)。初始方法中 \(h\) 来自冻结 Wan 并显式 `stop-gradient`；允许梯度进入 Wan
的变体必须归入单独 PEFT baseline。

### 1.3 归一化 rank 注册

shape audit 完成后，以归一化容量而非任意绝对整数注册：

\[
\rho\in\left\{\frac1{32},\frac1{16},\frac18,\frac14\right\},
\qquad
r(\rho)=\max\left(1,\left\lfloor\rho\min(N,D,C_p)\right\rceil\right).
\]

最终整数 \(r\) 必须随审计表一起写入配置；查看测试结果后不得改换 \(\rho\) 或取整规则。

### 1.4 联合 WAM 场，而不是“动作 = 视频 + 残差”

视频 latent 与 robot action 不在同一个空间，所以下式被明确禁止：

\[
v_{\mathrm{action}}=v_{\mathrm{video}}+\Delta v.
\]

严谨的联合表达是：

\[
u_{\mathrm{WAM}}
=
\begin{bmatrix}
Y_{\mathrm{pre}}\\
0
\end{bmatrix}
+
\begin{bmatrix}
\Delta Y_{\phi}\\
u_a
\end{bmatrix}.
\]

其中，\(\Delta Y_{\phi}\) 是同空间的视频动力学修正；\(u_a\) 是在动作空间中新建的
action flow。本文研究的是：视频增量能否成为生成 \(u_a\) 的充分条件。

### 1.5 动作必须读取同一个 delta，而不是并排另建分支

直接读取 \(P,Q\) 存在 factorization gauge ambiguity：不同 factor basis 可以生成完全
相同的 \(PQ^\top\)，却给动作头不同的坐标。为避免把任意 factor 编号误当成控制语义，
动作接口读取 factorization-invariant 的 delta summary：

\[
C_{\Delta}
=
A(h)^\top\Delta G
=
\left(A(h)^\top P\right)Q^\top
\in\mathbb R^{M\times D},
\]

再投影为与现有 Action-DiT 匹配的 query memory。这里 \(A(h)\) 是只依赖当前 base
features 的 \(M\) 个 query weights。该计算可以利用低秩乘法完成，不要求物化完整
\(N\times D\) 矩阵；随后统一投影为现有 Action-DiT 所需的 `[B,64,512]` memory。

这个合同保证：

- 视频残差 loss 与动作条件读取的是同一个 \(\Delta G\)；
- 对保持 \(PQ^\top\) 不变的 factor reparameterization，\(C_\Delta\) 也不变；
- 不能用一个与 \(\Delta G\) 无关的 action branch 冒充“共享 delta code”。

### 1.6 从 ScaleResfusion 类比中能搬什么、不能搬什么

本想法借用的是图像修复工作中的研究范式：预训练生成场已经包含大部分通用能力，新任务
可以被写成原场附近的 task-specific correction，再检查该 correction 是否适合紧凑适配。

可以搬入本文的部分是：

- `pretrained field + task residual` 的问题定义；
- 冻结基础生成先验，只学习受限容量修正；
- 用 field drift、rank sweep 与 full-update control 验证“只需小修正”，而不是只报参数量。

明确不能直接搬入的部分是：

- 图像修复前后都在 image-latent velocity 空间，而 video velocity 与 action velocity 不同域；
- ScaleResfusion 的 residual flow path 与 exact acceleration point 不自动存在于 WAM；
- 图像修复中的低清—高清残差不能被替换成“视频—动作残差”；
- LoRA 的参数 rank 不能证明本文的输出 delta field 低秩。

因此，ScaleResfusion 在这里是 hypothesis source 与分析范式，不是可直接复用的数学方法。
正式论文还必须单独核对相关工作的原文、代码与 novelty boundary。

## 2. 与现有 Model3/O2/Model5 的关系

| 现有对象 | 在本实验中的唯一角色 | 不能据此声称什么 |
|---|---|---|
| 原始预训练 Wan | 唯一 \(V_{\mathrm{base}}\) 与所有方法的共同初始化 | 不能被 O2 checkpoint 替换 |
| Model3 O2 | matched retraining 的强性能 baseline；另作 G1-secondary 分析 checkpoint | 多层 readout 的强表现不证明从预训练 Wan 出发的机器人增量低秩 |
| Model5-style D2 | **LRD-WAM 最终方法 carrier**：one policy-owned Gaussian future slot，temporal `[0,1000]` | D2 本身是待检验的方法组成，不是已证明的强 baseline |
| 已训练 Model5 Object | historical positive pilot；15K 下 solver 10 为 466/500、solver 5 为 478/500 | 单 seed、无 D1 对照、无显式 rank-constrained field，不能充当主方法证据 |
| D1 current-only | 与 D2 配对的 temporal control | 若 D1 显著优于 D2，不能继续包装 future-field LRD-WAM |
| parameter-matched LoRA | 相同容量的 PEFT control | 参数空间 LoRA rank 不等于输出场 rank |
| Matrix L/C/B | hidden depth、composition 与 selected-layer PEFT 的既有分析主线 | 不能代替 D1/D2 与输出低秩对照 |
| Light-WAM | 强 PEFT/轻量 WAM 外部锚点 | 本工作不能重新声称“首次用 PEFT 将 Wan 变成 WAM” |

本分支不修改现有 D/L/C/B 的历史结论，但不再等待“任意 D winner”决定方法 carrier：
D2 是主方法路径，D1 是 matched control。若 D1 在相同预算下显著优于 D2，停止
future-field LRD-WAM；只有另行定义、命名并验证 current-only hidden-space delta，才可开新分支，
不能静默把 D1 改称同一个方法。

## 3. 核心 Claims 与对应证伪条件

### Claim 1：机器人视频场增量具有稳定的低秩结构

支持证据必须同时包含：

- 从原始预训练 Wan 出发的 \(R_{\mathrm{pre}}^*\) 在 pre-unpatchify
  \(N\times C_p\) 空间中奇异值快速衰减；
- Object 与 Long 在**部署同构的 D2 网格**上存在相近的归一化有效 rank 区间；
- 不同任务可以使用不同 basis；核心要求是所需 rank 稳定，而不是强行假设一个全局固定
  子空间；
- 该结构不是 raw target velocity、本身的 token/channel smoothness 或固定 D2 token geometry 造成的
  平凡低秩；
- 低秩结构不只出现在少数 task、少数 diffusion timestep 或非接触阶段。

证伪条件：只有 O2 residual、单个 suite 或少数样本低秩；换 task 后所需归一化 rank 接近
full rank；entry-permuted、random-pair 或 raw-velocity controls 具有同样谱形状。

### Claim 2：可部署的 delta code 是控制有效表示

支持证据必须来自只输入 current observation、language 与 **D2 policy-owned Gaussian
future slot** 的路径，temporal grid 固定为 `[0,1000]`。D1 不增加 future slot，只作为
current-only control。oracle residual 的 target 由真实未来定义，只能说明 representation
potential，不能证明在线可用。

证伪条件：oracle code 能预测 action，但 deployable code 不能；base hidden 或
parameter-matched side adapter 与 delta code 等效；zero-delta、episode-shuffled delta 或
detached-independent side feature 不降低性能。

### Claim 3：显式低秩 delta 带来真实 adaptation efficiency

候选方法必须先在闭环性能上达到预声明门槛，再比较成本。仅减少 trainable parameters
不足以支持效率 claim；冻结大模型仍可能保留昂贵 forward。

证伪条件：性能接近只发生在更大 Action-DiT、更多 steps 或额外 VDT forwards 下；
训练 GPU-hours、peak memory 或达到目标成功率的时间没有实际改善；优势只来自缓存但
未计入缓存生成成本。

## 4. 总体路线

```text
G0  冻结 carrier、残差定义、数据与成本口径
 ↓
G1  不训练新 WAM：部署同构 D2 residual 谱、shape 与泄漏审计
 ↓  只有“低秩结构真实存在”才继续
G2  小规模 deployable delta-code 预测与 action probe
 ↓  只有“无未来泄漏仍可读动作”才继续
G3  LIBERO Object matched closed-loop method test
 ↓  只有“性能门槛 + 成本优势”同时成立才继续
G4  Long / Spatial 跨 suite 验证
 ↓
G5  第二 benchmark / 第二 backbone 与机制干预
```

G1 失败时停止方法开发，保留为 negative representation result；不得通过增大 action head
或扩大 rank 搜索把低秩假设包装回来。

## 5. G0：Preflight 与残差合同冻结

开始任何诊断前记录：

| 字段 | 必须冻结的内容 |
|---|---|
| Base video checkpoint | 原始预训练 Wan checkpoint、commit SHA、precision、VAE 与 latent normalization；这是唯一主 \(V_{\mathrm{base}}\) |
| Analysis checkpoint | Model3 O2 checkpoint SHA；只作强 baseline 与 \(R_{\mathrm{O2}}^*\) 次级分析，不作主 base |
| Action decoder | 所有方法统一的 Action-DiT initialization、architecture、H8/R8、solver 与训练配置 |
| Temporal carrier | D2：one current latent + one policy-owned Gaussian future probe，temporal `[0,1000]`，同空间分辨率；D1 是 current-only control |
| Video target | true future 只定义 H8 chunk 对应 endpoint 的 flow target；冻结 noise convention，不把真实未来送入 Wan；future input noise 与 scheduler target 必须复用同一 tensor |
| Endpoint mapping | `H8 actions [0,8) -> raw observation frame 8 -> raw offsets [0,2,4,6,8] 的 5-frame VAE clip -> VAE latent slot 1 -> D2 future patch-token indices`；必须验证该 clip 的 current latent 与已有 cache slot 0 一致。现有 33-frame、ratio-4 cache 的 latent slots 1/2 分别止于 raw frame 16/32，不能冒充 H8 target |
| Conditioning | current cameras、language；proprioception 是否使用要统一；明确禁止 expert future action/reward/state |
| Head/shape audit | 运行时记录 \(N,D,C_p\)、head 输入/输出 shape、展平顺序、unpatchify 前后 shape；冻结的 Wan2.1 Object runtime 预期为 `1536 -> 64`，同时记录被 loader 覆盖的 YAML 占位值 `3072 -> 192` |
| Rank grid | \(\rho\in\{1/32,1/16,1/8,1/4\}\)，shape audit 后导出并冻结每个整数 \(r\) |
| Split | episode-heldout train/validation/test；task 与 suite 标识 |
| Cost | GPU type/count、forward/backward 数、缓存成本、峰值显存与 accelerator-hours |
| Randomness | video noise、feature noise、action solver 与 rollout seed 分开记录 |

本地 artifact 已确认 Model5 Object 在 15K 的单 seed 结果为 solver 10 `466/500`
（93.2%）、solver 5 `478/500`（95.6%）；15K 在两种 solver 下均优于 10K/20K。G0 仍需在
server 侧复核 checkpoint、commit、逐 episode 结果和实际 forward 计数；复核前它只作
historical pilot，不进入 LRD-WAM 主结果表。

## 6. G1：Deployment-Aligned Residual Spectrum Diagnostic

### 6.1 G1-primary：严格复现部署网格

第一轮使用 LIBERO Object 与 Long 的 episode-heldout 演示窗口。每个样本只构造
`one current latent + one Gaussian future probe`，temporal grid 固定 `[0,1000]`，空间分辨率
与部署相同；目标未来严格对应 H8 action chunk 的 endpoint。真实 future 只生成监督 target，
不作为 Wan 输入或 action condition。在同一窗口上保存：

\[
Y_{\mathrm{pre}},\qquad Y^*,\qquad R_{\mathrm{pre}}^*=Y^*-Y_{\mathrm{pre}}.
\]

主分析至少包含：

1. **Per-sample token-output SVD**：每个
   \(R_{\mathrm{pre}}^*\in\mathbb R^{N\times C_p}\) 的谱；
2. **Across-sample subspace SVD/PCA**：检查不同样本是否共享低维 basis，还是需要
   input-conditional basis；
3. **Cross-task subspace overlap**：用 principal angles 或 projection overlap 判断 basis
   的可迁移程度；低 overlap 不自动否定逐输入低秩，但会把 claim 限定为
   input-conditional low rank；
4. **Endpoint/phase stability**：按 task、动作阶段以及接触/非接触阶段分层，确认结果不是
   少数 endpoint 或简单非接触片段造成。

G1-primary 不扫描 sampled timestep，不输入多 future slots，也不使用完整未来视频。这样它与
最终 D2 方法共享输入和目标几何。

### 6.2 G1-secondary：机制分析，不决定主 Go/No-Go

只有 G1-primary 完成后，才可选做 full-video、sampled-timestep 或 multi-slot 谱分析，并在
完全相同合同允许时计算 \(R_{\mathrm{O2}}^*\)。这些结果只回答低秩结构随 timestep、horizon
和已适配 checkpoint 如何变化，不能替代 G1-primary，也不能为其失败“补票”。

主指标为累计解释能量：

\[
E(r)
=
\frac{\sum_{i=1}^{r}\sigma_i^2}
{\sum_i\sigma_i^2}.
\]

### 6.3 必要 controls

| Control | 排除的替代解释 |
|---|---|
| SVD of \(Y^*\) | 所有 video velocity 天然都低秩，而不是 task delta 特殊 |
| SVD of \(Y_{\mathrm{pre}}\) | base field 本身的谱结构被误写成 residual 结构 |
| entry-permuted \(R_{\mathrm{pre}}^*\) | 低秩来自相同边际分布而非时空—通道结构 |
| randomly paired \(Y^* - Y_{\mathrm{pre}}\) | 任意两个速度场之差都呈现相同谱衰减 |
| task/phase-stratified results | 聚合平均掩盖少数困难阶段的高 rank |
| natural-video residual（若同合同数据可得） | 机器人 residual 是否具有域特异性；没有该数据时明确记为未验证 |

### 6.4 G1 Go/No-Go

进入 G2 需要同时满足：

- 存在一个预注册归一化 rank \(\rho\le1/8\)，在 Object 与 Long 的 G1-primary 上
  median \(E(r(\rho))\ge0.80\)，且
  10th-percentile \(E(r(\rho))\ge0.60\)；
- 相同 rank 下，真实 residual 的 median \(E(r(\rho))\) 至少比 entry-permuted control 高
  10 percentage points；
- 有效 rank 不只由单个 task、endpoint 或动作阶段支撑；
- 在 episode-heldout 窗口上仍保持相同 rank 规律。若 across-sample 没有稳定共享 basis，
  必须将 Claim 1 写成 **input-conditional low rank**，并由 G2 的 heldout predictor 检验
  这些条件化 factors 是否可学习，不能声称存在一个全局机器人子空间。

阈值、shape 与 \(\rho\to r\) 映射必须在查看完整测试结果前冻结。G1-secondary 的任何
sampled-timestep 或 O2 结果均不参与主 Go/No-Go。

## 7. G2：Deployable Delta-Code Probe

### 7.1 泄漏隔离

G2 以 D2 policy view 为唯一主路径：

- **D2 policy view（primary）**：只使用 current observation、language 与 one
  policy-owned Gaussian future slot，temporal `[0,1000]`；
- **D1 policy view（temporal control）**：current-only，不创建 future slot；
- **Oracle view（secondary upper bound）**：真实 future 只构造 residual target，不进入主方法
  输入；若用 oracle SVD code 做 probe，必须单独标记为不可部署上限。

动作标签只进入 action loss，不得作为 delta adapter 的输入。所有 action probe 使用
episode-heldout split；同一 episode 的相邻窗口不能跨 train/test。

### 7.2 G2-A：缓存表示的廉价 probe

先缓存冻结表示，对每一种 conditioner 分别训练 parameter-matched 的 linear probe 与
2-layer MLP probe；使用相同 split、输入 token 数、输出 action target、训练步数与参数预算。

| ID | Dynamics representation | Action condition | 研究问题 |
|---|---|---|---|
| LR-P0 | frozen base | parameter-matched base-hidden pooling | 原始视频表示是否已经足够 |
| LR-P1 | oracle truncated SVD residual | invariant \(C_\Delta\) | 低秩 residual 的表示上限 |
| LR-P2 | predicted full-rank residual | invariant residual pooling | residual target 本身是否有用 |
| LR-P3 | predicted rank-constrained residual | invariant \(C_\Delta\) | 无未来泄漏时低秩 delta 是否可读 |
| LR-P5 | parameter-matched LoRA/side adapter | matched action head | 收益是否只是相同容量的小 adapter |

LR-P1 不能进入方法主表或闭环主 claim。G2-A 用 LR-P0/P2/P3/P5 选择一个饱和 rank，
LR-P2 只提供 full-rank upper anchor。对 LR-P3 额外做三个低成本 intervention：

1. **zero-delta**：令 \(C_\Delta=0\)；
2. **episode-shuffled delta**：在不同 episode 间打乱 \(C_\Delta\)，保持边际尺度；
3. **detached-independent feature**：用同参数量、从 base hidden 独立产生且不受
   \(\mathcal L_\Delta\) 约束的 side feature 替代 \(C_\Delta\)。

若三种干预均不显著伤害 probe，动作很可能绕过 delta，不能进入 shared-code claim。

### 7.3 G2-B：matched Action-DiT probe

再把 G2-A 的 LR-P0/P2/P3/P5 接到**相同初始化、相同 architecture、相同 action-flow
objective、相同 solver/steps** 的 Model3/O2 Action-DiT；只有 conditioner 不同。G2-A 防止
强 Action-DiT 掩盖表示差异，G2-B 检查差异在现有动作生成器中是否仍成立。不得为 LR-P3
单独放大 decoder 或增加训练步数。

### 7.4 G2 Go/No-Go

只有 LR-P3 在 G2-A 与 G2-B 的 Object、Long episode-heldout 数据上都稳定优于
LR-P0/LR-P5、不劣于 LR-P2，并且 rank 增长出现清晰饱和区间时进入 G3。zero/shuffle/
detached-independent intervention 必须形成方向一致的性能下降。若只有 oracle LR-P1 强，
结论是“未来 residual 含动作信息，但当前观测无法可靠预测”，停止闭环方法训练。若 D1
显著优于 D2，则停止 future-field LRD-WAM，不把 D1 静默替换成主方法。

## 8. G3：Object Closed-Loop Matched Matrix

所有候选从同一个原始预训练 Wan initialization 与同一个 Action-DiT initialization 开始，
固定 D2 输入、数据、action-flow objective、H8/R8、solver 10、checkpoint set、optimizer、
batch、训练样本数与评测 initial states。Model3 O2 必须在该合同上 matched retraining，不能
直接拿历史 checkpoint 代替。

| ID | Video-DiT adaptation | Dynamics output | Action condition | 作用 |
|---|---|---|---|---|
| LR-M0 | frozen base | none | matched base hidden | no-dynamics-adaptation lower bound |
| LR-M1 | all-layer rank-64 LoRA | ordinary adapted field | O2 readout | current strong upper anchor |
| LR-M2 | parameter-matched LoRA | ordinary adapted field | matched action interface | 参数空间低秩 baseline |
| LR-M3 | frozen base + full-rank side adapter | full-rank \(\Delta G\) | invariant delta pooling | 显式 residual、无 rank constraint |
| LR-M4 | frozen base + low-rank delta | rank-\(r\) \(\Delta G\) | independent base-hidden branch | low-rank dynamics、无共享 code |
| LR-M5 | frozen base + low-rank delta | rank-\(r\) \(\Delta G\) | proposed invariant \(C_\Delta\) | 完整方法 |
| LR-M6 | frozen base | none | action-space residual head | residual 放在 action space 的对照 |

### 8.1 分阶段淘汰，而不是一次跑满矩阵

- **G3-stage 1（screening）**：只跑 LR-M0/M1/M2/M3/M5，单 training seed；在开跑前从
  10K--20K 中冻结一个统一训练步数，所有方法共用。用 G2 选出的唯一 rank，不重新 grid
  search。目的只是淘汰明显失败的方法，不作论文级方差结论。
- **G3-stage 2（confirmation）**：只保留 LR-M1/M2/M5，至少 3 个 training seeds，固定
  10 tasks × 50 trials（500 episodes）与相同 initial-state IDs。
- **机制补充**：仅当 LR-M5 通过 stage 2 的性能与成本门槛后，才运行 LR-M4/M6。
  LR-M4 与 LR-M5 回答共享动力学—动作表示是否必要；LR-M6 只回答 action-space residual
  能否解释收益。它们不是前置大矩阵。

LR-M3 与 LR-M5 回答 output rank constraint；LR-M2 与 LR-M5 回答参数空间 LoRA 和
输出空间 delta 的区别。动作头大小不得为 LR-M5 单独增加。

### 8.2 训练 loss

动力学分支：

\[
\mathcal L_{\Delta}
=
\left\|Y_{\mathrm{pre}}+\Delta G_{\phi}W_{\mathrm{head}}-Y^*\right\|_2^2.
\]

动作分支保持与 O2 匹配的 action flow loss：

\[
\mathcal L
=
\lambda_{\Delta}\mathcal L_{\Delta}
+
\lambda_a\mathcal L_{\mathrm{action}}.
\]

第一主对比不引入额外 GAN、DMD、distillation teacher 或更大的 action decoder。若
factor normalization、orthogonality 或 scale-balance regularizer 对优化必需，必须单列
系数并提供关闭该项的稳定性消融，不能把额外正则收益归因于低秩结构。

### 8.3 G3 成功门槛

LR-M5 只有同时满足以下条件才进入 G4：

1. 相对 LR-M1/O2 anchor 的 paired 95% CI 下界高于预注册 non-inferiority margin
   `-2 pp`；
2. stage 2 至少 3 个 training seeds，固定 10 tasks × 50 trials 与相同 initial-state IDs；
3. 相比 LR-M1，VDT-side trainable parameters 至少减少 5 倍；
4. peak training memory 至少降低 30%，或达到同一成功率门槛的 accelerator-hours 至少
   降低 2 倍；
5. 一次 action chunk 仍只运行一次 frozen VDT forward，不增加 video denoising rollout；
6. G2 的 zero/shuffle/detached-independent intervention 必须显示动作依赖 delta；若要在
   论文中保留强 shared delta-code claim，再补 LR-M4，并要求 LR-M5 相对 LR-M4 的 paired
   95% CI 下界高于 0；若二者等效，只能保留低秩 dynamics claim；
7. LR-M5 至少对 LR-M2 non-inferior，并在成功率、达到门槛的 GPU-hours 或显存中形成
   一个严格 Pareto 改善；否则普通 parameter-matched PEFT 已足够解释结果。

若只满足性能，不满足成本，定位为 representation method；若只满足成本但性能未过
non-inferiority，定位为 compression result；两者都不满足则停止。

## 9. G4/G5：泛化与机制验证

### G4：跨 suite

冻结 Object 选出的 architecture 与 rank-selection rule，不用 Long 测试集重新挑 rank。
依次验证：

1. LIBERO Long；
2. LIBERO Spatial；
3. 数据量 25%/50%/100% 的 adaptation curve。

不同 suite 可以重新运行预声明的 calibration rule，但必须计入总 GPU-hours；不能用完整
grid search 后只报告最终廉价模型。

### G5：跨系统与机制干预

只有 G4 通过后再选择：

- RoboTwin 2.0 或另一机器人 benchmark；
- 第二个 Video-DiT/backbone 或公开 WAM 代码基；
- residual energy 与物体运动、interaction region、contact phase 的空间对应；
- rank component intervention：删除、置换或缩放 component，观察动作与闭环阶段变化；
- cross-task subspace transfer：Object basis 是否能低成本初始化 Long/Spatial。

“第二 benchmark”与“第二 backbone”至少完成一个，才适合提出 backbone-independent 或
general WAM adaptation claim。

## 10. 训练与部署信息流合同

### 10.1 训练

```text
current latent + one policy-owned Gaussian future probe + language
                    temporal grid [0,1000]
                              ↓
                    one frozen Wan forward
                              ↓
              pre-head G0 and frozen head W_head
                              ↓
                    low-rank delta adapter
                       ΔG=P[N,r]Q[D,r]^T
                       ↙                 ↘
 Y_robot=(G0+ΔG)W_head              C_delta=A(h)^TΔG
           ↓                               ↓
 endpoint video-flow target             Action-DiT
           ↓                               ↓
   dynamics residual loss            action flow loss
```

- delta adapter 不接收 expert action；
- true future 只用于构造与 H8 chunk endpoint 对齐的 video-flow target，不进入 Wan 输入、
  delta adapter 输入或 action condition；
- action loss 是否反传到 delta adapter 必须显式注册；
- 原始 Wan backbone 与最终 head 都冻结；必须验证无 optimizer entry、无非零 gradient、
  无权重漂移；
- 主方法每个 action chunk 只运行一次 Wan forward，同时服务 dynamics target 与 action；
- 只有 delta module、invariant pooling/interface 与 matched Action-DiT 更新；
- 历史 Model5 的 sampled-timestep full-video flow branch 不属于主方法。若作为
  G1-secondary/G5 ablation 加入，它会产生第二次 Wan forward，必须单列方法名、显存、
  latency 与 accelerator-hours，不能并入 LRD-WAM 主效率结果。

### 10.2 部署

```text
current latent + one policy-owned Gaussian future probe + language
                    temporal grid [0,1000]
                         ↓
               one frozen Video-DiT forward
                         ↓
              pre-head G0 + low-rank ΔG
                         ↓
                   invariant C_delta
                         ↓
                Action-DiT, solver 10
                         ↓
                 H8 action chunk, R8
```

训练和部署使用相同 D2 输入：Gaussian future probe 都由 policy 自己采样，不读取真实未来。
部署不生成未来 RGB、不运行 iterative video solver。若最终需要两个 VDT forwards 才达到
性能，必须作为不同方法重新登记，并重新计算 latency/efficiency。D1 部署只作为
current-only control，不是 LRD-WAM 主方法。

## 11. 统计、成本与结果记录

### 11.1 主要指标

| 类别 | 指标 |
|---|---|
| Representation | G1-primary 上 \(R_{\mathrm{pre}}^*\) 的 \(E(r(\rho))\)、normalized effective rank、cross-task principal angles、residual reconstruction error；O2/sampled-timestep 单列 secondary |
| Action | episode-heldout action-flow loss/probe；正式结论使用 closed-loop success |
| Closed loop | paired success difference、task-stratified 95% CI、discordant outcomes、failure stage |
| Parameters | frozen base、delta adapter、interface、Action-DiT 分项；同时报告 VDT-side 与 total trainable |
| Training | GPU type/count、steps、samples、accelerator-hours、samples/s、peak allocated/reserved memory |
| Deployment | VDT forwards/chunk、solver steps、latent tokens、plan-call latency、throughput |
| Cache | cache 生成 GPU-hours、磁盘体积、可复用次数与摊销/不摊销两种成本 |

### 11.2 禁止的统计替代

- checkpoint 不能当独立 seed；
- McNemar `p>0.05` 不能替代 non-inferiority CI；
- offline MSE、action probe 或谱能量不能替代闭环；
- 相同训练 steps 不能称为 compute-matched；
- 在 test suite 上挑 rank 后不能再把同一结果称为泛化。

## 12. 预期结果表模板

| Method | Rank | Object | Long | VDT-side trainable | Total trainable | Peak GB | GPU h to threshold | Plan ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 frozen + base hidden | — |  |  |  |  |  |  |  |
| D2 frozen + base hidden | — |  |  |  |  |  |  |  |
| O2 matched all-layer LoRA | 64/parameter rank |  |  |  |  |  |  |  |
| Parameter-matched LoRA |  |  |  |  |  |  |  |  |
| Full-rank pre-head delta | full |  |  |  |  |  |  |  |
| Low-rank delta, no sharing |  |  |  |  |  |  |  |  |
| **LRD-WAM** |  |  |  |  |  |  |  |  |

另附三张不能省略的图：

1. `normalized rank → residual energy / reconstruction`，主图按 suite/task/phase 分层，
   sampled-timestep 仅放 secondary；
2. `rank → closed-loop success / GPU-hours` Pareto；
3. `action success → video field drift`，比较 frozen、LoRA、full-rank delta 与 LRD-WAM。

## 13. 论文级结论的降级路径

| 观察 | 允许的结论 | 不允许的包装 |
|---|---|---|
| G1-primary 谱不低秩 | 从原始 Wan 出发、在 D2 部署合同下没有稳定低秩证据 | 用 O2 或 sampled-timestep secondary 结果继续称“低秩场方法” |
| D1 显著优于 D2 | policy-owned future slot 不是有效 temporal carrier | 静默把 current-only hidden delta 改称同一个 LRD-WAM |
| Oracle 强、policy view 弱 | 真实未来 residual 含动作信息，但不可部署预测 | 用 oracle probe 声称 WAM 有效 |
| Low-rank ≈ full-rank，但不优于 base hidden | delta 可压缩，但不是更好的动作接口 | “动力学增量是控制充分表示” |
| LR-M5 性能过门、成本不过门 | representation method | efficient adaptation method |
| LR-M5 成本下降、性能未过门 | compression/efficiency trade-off | non-inferior high-performance WAM |
| Object 成立、Long 失败 | suite-specific method | general VDT-to-WAM principle |
| 多 suite + matched controls + 成本均成立 | method paper 候选 | 在未做 novelty search 前声称“首次” |

## 14. 最小下一步

### 14.1 2026-08-01 Object G0/G1 preliminary execution record

零训练 Object G0 与预备 G1 已在
`runs/I-003/model5/20260801_lrd_wam_object_g0_g1/` 完成并通过 retained artifact
validator。G0 只加载原始 Wan2.1-T2V-1.3B，实测
`N_future=392, D=1536, C_p=64`、future slice `[392,784)`、rank grid
`r={2,4,8,16}`；target layout、Wan unpatchify 与 H8 clip current/cache current
round trip 的 max error 均为 0。checkpoint structure 为 825/825 compatible，且
11 个跨 embedding/block/head 的抽样 tensor 在 BF16 load 后逐元素完全一致。

8-sample smoke 与 64-sample integration 按顺序完成。64 样本在 `rho=1/8, r=8`
时 real residual 的 median/p10 `E(r)` 为 83.54%/82.17%；entry-permuted 与
random-pair median 分别为 21.54% 与 27.32%。这些结果表明 Object 正式 G1 的
实现与合同已具备启动条件，但仍只是 preliminary readiness evidence：没有运行正式
256--512 Object、Long、across-sample PCA、phase 分层或 natural-video control，因而不能
作正式 G1 Go/No-Go，更不能启动 G2。

### 14.2 2026-08-01 formal Object G1 protocol

用户已单独授权启动正式 Object G1。运行前冻结以下内容，查看正式结果后不得修改：

- 正式样本数为 420，位于预声明的 256--512 范围内。运行时数据身份以
  `libero_object_no_noops_lerobot/meta/info.json` 和 `episodes.jsonl` 为准：两者均只包含
  457 个 Object episode，而不是 500；这 457 个 episode 全部支持无 padding 的完整 H8
  window。各 task 可用 episode 数为 `47/43/45/50/46/42/45/47/47/45`，因此正式集合按
  最小 task 覆盖固定为 `10 tasks x 42 episodes`。其余 37 个有效 episode 仅因 task balance
  未使用；不存在 43 个被 runtime adapter 排除的不完整 episode，也不补齐或重复采样；
- 每个 task 的 42 个 episode 用 seed 13407 固定抽样，并按 episode 严格拆分为
  train/validation/test `24/9/9`，总计 `240/90/90`。同一 episode 只贡献一个 window，
  不跨 split；
- 每个 task、每个 split 内将 episode 等分为 early/middle/late 三组，window start 分别位于
  该 episode 有效 H8 start range 的 10%/50%/90% 位置。该标签只表示 normalized episode
  progress，不冒充 contact/non-contact annotation；
- per-sample SVD、raw target、raw pretrained、entry-permuted、random-pair controls、
  `rho -> r`、noise 与 shuffle seed 均保持 8/64 preliminary 合同不变；
- across-sample PCA 将每个 `392 x 64` residual 按冻结 layout 展平，只在 240 个 train
  episode 上拟合 centered basis，并在 validation/test 上报告 rank 2/4/8/16 projection
  energy；不得用 validation/test 重选 basis 或 rank；
- cross-task overlap 只使用各 task 的 train episodes，在 rank 8 flattened-residual PCA
  basis 上报告 principal angles 与 mean squared cosine overlap；
- task、split 与 early/middle/late 均报告 rank-energy 分层。Object-only 正式结果仍不能
  单独触发 G2；proposal 的最终 G1 Go/No-Go 继续要求 matched Long；
- formal artifact root 固定为
  `runs/I-003/model5/20260801_lrd_wam_object_g1_formal420/`。本次仍不启动 Long、G2、
  Action-DiT 训练或闭环评测。

### 14.3 2026-08-01 formal Object G1 result

正式运行在 `runs/I-003/model5/20260801_lrd_wam_object_g1_formal420/` 完成，terminal
manifest 为 `complete_stopped_after_object_formal420`，独立 validator 为 `pass`。420 个
样本来自 10 个 task 各 42 个唯一 episode，split 为 `240/90/90`，normalized
early/middle/late 各 140。正式 G0 复跑继续通过：只加载原始 Wan2.1-T2V-1.3B，825/825
checkpoint keys compatible，11 个抽样 tensor 在 BF16 load 后逐元素相等；所有 target
layout、unpatchify、future slice、head、noise reuse 与 current-latent audit error 均为 0。

per-sample SVD 的 median explained energy 为：

| Quantity | r=2 | r=4 | r=8 | r=16 |
|---|---:|---:|---:|---:|
| real residual | 54.99% | 74.46% | 83.26% | 90.47% |
| entry-shuffled residual | 5.90% | 11.39% | 21.52% | 39.29% |
| random-pair residual | 11.62% | 18.13% | 27.87% | 44.49% |
| raw target | 21.24% | 28.88% | 37.81% | 52.37% |
| raw pretrained output | 13.31% | 18.39% | 27.79% | 44.17% |

在主检查 `rho=1/8, r=8`，real residual median/p10 为 83.26%/81.55%，相对
entry-shuffled median 高 61.73 pp。10 个 task 的 real median 为 82.61%--84.27%，最低
task p10 为 80.45%；early/middle/late median 为 83.11%/83.85%/82.55%，最低 phase p10
为 80.46%。因此冻结的 Object-only per-sample/task/normalized-phase signal 为 `pass`。

该结论不能外推为一个强共享低维 basis。只用 240 个 train episode 拟合的 centered global
PCA 在 rank 2/4/8/16 解释 train 总 centered variance 的 17.61%/26.55%/36.72%/46.84%；
held-out validation/test 的 per-sample centered projection median 在 rank 8 仅为
31.03%/30.08%，rank 16 为 37.64%/35.69%。包含 train mean 后的 raw reconstruction
median 较高：validation/test rank 8 为 74.94%/75.27%，rank 16 为 77.37%/77.65%。
45 个 rank-8 task-basis pair 的 mean squared-cosine overlap median 为 0.1675，mean
principal angle median 为 69.56 度，支持 task-specific basis 而不是单一共享 basis 的解释。

本次共执行 421 次 Wan 与 421 次 VAE forward，用时 152.54 秒，CUDA allocated peak
为 3.619 GiB；没有 optimizer、backward、robot checkpoint、adapter/LoRA、Long、G2、
Action-DiT 训练或闭环评测，也没有保存 residual/basis tensor。proposal-level G1
Go/No-Go 仍为 `not_evaluable_without_matched_long`。natural-video control 与经过验证的
contact/non-contact phase labels 仍不可用。

### 14.4 后续最小步骤

当前仍不应直接实现完整 LRD-WAM。下一步是：

1. 不启动 G2、Action-DiT 训练或闭环评测；Object 正式结果本身不能触发这些阶段；
2. 若另行授权 matched Long，则先冻结 Long 的数据可用数、task balance 与同合同 D2
   selection，再在 Object 与 Long 的同合同 D2 窗口上提取
   \(Y_{\mathrm{pre}},Y^*,R_{\mathrm{pre}}^*\)，只以 G1-primary 决定 Go/No-Go；
3. 只在 Object 与 Long 都通过冻结的 cross-suite G1 gate 后，才为 G2-A/G2-B 与 three
   bypass interventions 编写新的执行合同；
4. 只有 G2 通过，才用一个冻结 rank 申请 G3-stage 1 的单 seed Object screening；
5. stage 1 淘汰后才为 LR-M1/M2/M5 申请 3-seed、500-episode confirmation 预算；
6. LR-M4/M6 只在 LR-M5 通过后作为机制补充。

这条顺序把最便宜的证伪实验放在前面，并把核心主张固定为：

\[
\boxed{
\text{Pretrained Wan Dynamics on D2}
+
\text{Pre-Head Rank-Constrained Robot Delta}
\rightarrow
\text{Control-Sufficient Delta Code}
}
\]

而不是“给 Model3 再加一个 LoRA 或 action head”。
