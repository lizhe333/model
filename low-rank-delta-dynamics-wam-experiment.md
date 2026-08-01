# Low-Rank Delta Dynamics WAM

## 实验 Proposal 与执行合同

> 文档状态：**候选方法分支，已完成实验设计；不自动启动训练。**
>
> 当前名称：**Low-Rank Delta Dynamics WAM（LRD-WAM）**。
>
> 现有主线仍以 [research-proposal.md](research-proposal.md) 与
> [experiment-contract.md](experiment-contract.md) 为准。本分支首先只运行不训练新
> WAM 的 G0/G1 诊断；任何 G2 之后的新训练都需要单独冻结 server carrier 与预算。

## 0. 一句话目标

冻结通用 Video-DiT，不把动作速度错误地写成视频速度的残差，而是显式学习一个
**低秩的机器人视频动力学增量**，并让动作模型读取这一个增量本身：

\[
V_{\mathrm{robot}}
=
V_{\mathrm{base}}+\Delta V,
\qquad
\operatorname{rank}(\Delta V)\le r.
\]

最终目标不是“参数少且 LIBERO 分数不错”，而是验证一个可证伪的科学命题：

> 从通用 Video-DiT 到机器人 WAM 的任务增量，是否在视频向量场中具有稳定、可压缩的
> 低维结构；这份增量是否既能修正机器人未来动力学，也足以作为闭环动作生成的控制表示？

## 1. 数学定义与严格边界

### 1.1 基础视频场与 oracle 残差

对同一当前观测、语言条件、video-flow timestep 与 noisy future latent，冻结的
Video-DiT 输出：

\[
V_0
=
f_{\theta}(z_s,s\mid z_{\mathrm{cur}},l)
\in \mathbb R^{N\times C}.
\]

其中，\(N\) 是展平后的时空 token 数，\(C\) 是 latent channel 数。机器人演示中的真实
future latent 定义同一 flow contract 下的监督目标 \(V^*\)。oracle 机器人增量为：

\[
R^*=V^*-V_0.
\]

所有 residual 对比必须在完全相同的 latent normalization、timestep、conditioning、
token layout 与 flow parameterization 下计算。不能拿不同 timestep、不同 target 定义或
不同视频 horizon 的输出直接相减。

### 1.2 显式 rank-constrained delta field

候选方法不依靠“LoRA rank 小”间接声称场低秩，而是直接预测：

\[
\Delta V_{\phi}
=
P_{\phi}(h)Q_{\phi}(h)^\top,
\]

其中：

\[
P_{\phi}(h)\in\mathbb R^{N\times r},
\qquad
Q_{\phi}(h)\in\mathbb R^{C\times r},
\qquad
r\ll\min(N,C).
\]

初始方法中，\(h\) 是从冻结 Video-DiT 得到并显式 `stop-gradient` 的 base features；若某个
变体允许梯度进入 Video-DiT，它必须归入单独的 PEFT baseline，不能继续计作 frozen-delta
方法。

因此，输出增量本身满足：

\[
\operatorname{rank}(\Delta V_{\phi})\le r.
\]

第一轮注册的 rank 集合为：

\[
r\in\{2,4,8,16,32,64\}.
\]

### 1.3 联合 WAM 场，而不是“动作 = 视频 + 残差”

视频 latent 与 robot action 不在同一个空间，所以下式被明确禁止：

\[
v_{\mathrm{action}}=v_{\mathrm{video}}+\Delta v.
\]

严谨的联合表达是：

\[
u_{\mathrm{WAM}}
=
\begin{bmatrix}
V_0\\
0
\end{bmatrix}
+
\begin{bmatrix}
\Delta V_{\phi}\\
u_a
\end{bmatrix}.
\]

其中，\(\Delta V_{\phi}\) 是同空间的视频动力学修正；\(u_a\) 是在动作空间中新建的
action flow。本文研究的是：视频增量能否成为生成 \(u_a\) 的充分条件。

### 1.4 动作必须读取同一个 delta，而不是并排另建分支

直接读取 \(P,Q\) 存在 factorization gauge ambiguity：不同 factor basis 可以生成完全
相同的 \(PQ^\top\)，却给动作头不同的坐标。为避免把任意 factor 编号误当成控制语义，
动作接口读取 factorization-invariant 的 delta summary：

\[
C_{\Delta}
=
A(h)^\top\Delta V
=
\left(A(h)^\top P\right)Q^\top
\in\mathbb R^{M\times C},
\]

再投影为与现有 Action-DiT 匹配的 query memory。这里 \(A(h)\) 是只依赖当前 base
features 的 \(M\) 个 query weights。该计算可以利用低秩乘法完成，不要求物化完整
\(N\times C\) 矩阵。

这个合同保证：

- 视频残差 loss 与动作条件读取的是同一个 \(\Delta V\)；
- 对保持 \(PQ^\top\) 不变的 factor reparameterization，\(C_\Delta\) 也不变；
- 不能用一个与 \(\Delta V\) 无关的 action branch 冒充“共享 delta code”。

### 1.5 从 ScaleResfusion 类比中能搬什么、不能搬什么

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

| 现有对象 | 在本实验中的角色 | 不能据此声称什么 |
|---|---|---|
| Model3 | all-block rank-64 LoRA、future-video flow 与 16-layer Action-DiT 的历史强 baseline | 旧 checkpoint 不能代替新 carrier 上的 matched retraining |
| Model3 O2 | 当前注册强 carrier；Object 492/500、Long 476/500 | 多层 readout 的强表现不证明机器人增量低秩 |
| Model5 Object | 对话记录中的 D2-like positive pilot：one noisy future slot，15K 约 95.6% | 不是 D1/D2 因果对照，也没有显式 rank-constrained delta field；正式引用前须复核服务器 artifact |
| Matrix D | 决定部署侧使用 current-only 还是 policy-owned noisy future slot | Model5 单边结果不能替代 D1 vs D2 |
| Matrix L/C/B | hidden depth、composition 与 selected-layer PEFT 的分析主线 | 参数空间 LoRA rank 不等于输出向量场 rank |
| Light-WAM | 强 PEFT/轻量 WAM 外部锚点 | 本工作不能重新声称“首次用 PEFT 将 Wan 变成 WAM” |

本分支不修改现有 D/L/C/B 的已批准执行顺序。若 Matrix D 已有正式 winner，LRD-WAM
直接继承该 temporal input contract；若尚无 winner，G1 的 oracle 谱诊断可先做，但任何
闭环 LRD-WAM 训练必须等待 D 合同冻结，或单独批准一个明确的 D2-only pilot。

## 3. 核心 Claims 与对应证伪条件

### Claim 1：机器人视频场增量具有稳定的低秩结构

支持证据必须同时包含：

- oracle residual 的奇异值快速衰减；
- Object 与 Long 上存在相近的有效 rank 区间；
- 不同任务可以使用不同 basis；核心要求是所需 rank 稳定，而不是强行假设一个全局固定
  子空间；
- 该结构不是 raw target velocity、本身的 token/channel smoothness 或 timestep 造成的
  平凡低秩；
- 低秩结构不只出现在少数 task、少数 diffusion timestep 或非接触阶段。

证伪条件：只有单个 suite、单个 timestep 或少数样本低秩；换 task 后所需 rank 接近
full rank；entry-permuted、random-pair 或 raw-velocity controls 具有同样谱形状。

### Claim 2：可部署的 delta code 是控制有效表示

支持证据必须来自只输入 current observation、language 与 Matrix D winner 所定义的
policy-owned temporal input 的路径：D1 不增加 future slot，D2 只使用 Gaussian noise。
oracle residual 包含真实未来，只能说明 representation potential，不能证明在线可用。

证伪条件：oracle code 能预测 action，但 deployable code 不能；base hidden 或
parameter-matched side adapter 与 delta code 等效；打断 delta/action sharing 后性能不变。

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
G1  不训练新 WAM：oracle residual 谱与泄漏审计
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
| Base | Wan checkpoint、commit SHA、precision、VAE 与 latent normalization |
| Carrier | Model3/O2 checkpoint SHA；是否只作分析 checkpoint 或训练 initialization |
| Video target | flow parameterization、future horizon、timestep sampler、noise convention |
| Conditioning | current cameras、language、proprioception；明确禁止 expert future action/reward/state |
| Tensor layout | \(N\) 的 time/height/width 展平顺序与 \(C\) 定义 |
| Split | episode-heldout train/validation/test；task 与 suite 标识 |
| Cost | GPU type/count、forward/backward 数、缓存成本、峰值显存与 accelerator-hours |
| Randomness | video noise、feature noise、action solver 与 rollout seed 分开记录 |

G0 还必须复核对话中的 Model5 Object `15K / 95.6%` 所对应的真实 checkpoint、配置、
评测任务、solver、H/R 与逐 episode 结果。在复核前它只写作 historical conversation
anchor，不进入主结果表。

## 6. G1：Oracle Residual Spectrum Diagnostic

### 6.1 数据与提取

第一轮使用 LIBERO Object 与 Long 的 episode-heldout 演示窗口。在同一窗口上同时保存：

\[
V_0,\qquad V^*,\qquad R^*=V^*-V_0.
\]

按 suite、task、diffusion timestep bin、动作阶段以及接触/非接触阶段分层。分析至少包含：

1. **Per-sample token-channel SVD**：每个 \(R^*\in\mathbb R^{N\times C}\) 的谱；
2. **Across-sample subspace SVD/PCA**：检查不同样本是否共享低维 basis，还是需要
   input-conditional basis；
3. **Cross-task subspace overlap**：用 principal angles 或 projection overlap 判断 basis
   的可迁移程度；低 overlap 不自动否定逐输入低秩，但会把 claim 限定为
   input-conditional low rank；
4. **Timestep stability**：避免只在接近纯噪声或接近干净 future 的区间得出结论。

主指标为累计解释能量：

\[
E(r)
=
\frac{\sum_{i=1}^{r}\sigma_i^2}
{\sum_i\sigma_i^2}.
\]

### 6.2 必要 controls

| Control | 排除的替代解释 |
|---|---|
| SVD of \(V^*\) | 所有 video velocity 天然都低秩，而不是 task delta 特殊 |
| SVD of \(V_0\) | base field 本身的谱结构被误写成 residual 结构 |
| entry-permuted \(R^*\) | 低秩来自相同边际分布而非时空—通道结构 |
| randomly paired \(V^* - V_0\) | 任意两个速度场之差都呈现相同谱衰减 |
| task/timestep-stratified results | 聚合平均掩盖少数困难阶段的高 rank |
| natural-video residual（若同合同数据可得） | 机器人 residual 是否具有域特异性；没有该数据时明确记为未验证 |

### 6.3 G1 Go/No-Go

进入 G2 需要同时满足：

- 存在一个预注册 \(r\le32\)，在 Object 与 Long 上的 median \(E(r)\ge0.80\)，且
  10th-percentile \(E(r)\ge0.60\)；
- 相同 rank 下，真实 residual 的 median \(E(r)\) 至少比 entry-permuted control 高
  10 percentage points；
- 有效 rank 不只由单个 timestep bin 或单个 task 支撑；
- 在 episode-heldout 窗口上仍保持相同 rank 规律。若 across-sample 没有稳定共享 basis，
  必须将 Claim 1 写成 **input-conditional low rank**，并由 G2 的 heldout predictor 检验
  这些条件化 factors 是否可学习，不能声称存在一个全局机器人子空间。

阈值必须在查看完整测试结果前冻结。若 tensor shape 使 `r=32` 已超过
`0.1×min(N,C)`，则先按 normalized rank 重新注册，不能用过大的绝对 rank 制造通过。

## 7. G2：Deployable Delta-Code Probe

### 7.1 泄漏隔离

G2 分成两个明确视角：

- **Oracle view**：由真实 future 构造 noisy latent，只用于 residual reconstruction 与
  representation upper bound；
- **Policy view**：只使用 current observation、language 与 Matrix D winner 所定义的
  policy-owned temporal input；D1 为 current-only，D2 才加入 Gaussian future slot。两者
  都不得读取真实 future、expert action、reward 或 simulator state。

动作标签只进入 action loss，不得作为 delta adapter 的输入。所有 action probe 使用
episode-heldout split；同一 episode 的相邻窗口不能跨 train/test。

### 7.2 小规模对照

| ID | Dynamics representation | Action condition | 研究问题 |
|---|---|---|---|
| LR-P0 | frozen base | parameter-matched base-hidden pooling | 原始视频表示是否已经足够 |
| LR-P1 | oracle truncated SVD residual | invariant \(C_\Delta\) | 低秩 residual 的表示上限 |
| LR-P2 | predicted full-rank residual | invariant residual pooling | residual target 本身是否有用 |
| LR-P3 | predicted rank-constrained residual | invariant \(C_\Delta\) | 无未来泄漏时低秩 delta 是否可读 |
| LR-P4 | predicted low-rank residual | 独立 base-hidden action branch | delta/action sharing 是否必要 |
| LR-P5 | parameter-matched LoRA/side adapter | matched action head | 收益是否只是相同容量的小 adapter |

LR-P1 不能进入方法主表或闭环主 claim。G2 的关键对比是
LR-P3 vs LR-P0/LR-P4/LR-P5；LR-P2 只提供
full-rank upper anchor。

### 7.3 G2 Go/No-Go

只有 LR-P3 在 Object 与 Long episode-heldout 数据上稳定优于
LR-P0/LR-P4/LR-P5，且 rank 增长出现清晰饱和区间时进入 G3。若只有 oracle LR-P1 强，
结论是“未来 residual 含动作信息，但当前观测无法可靠预测”，停止闭环方法训练。

## 8. G3：Object Closed-Loop Matched Matrix

所有候选从同一个 Wan initialization 开始，固定数据、Action-DiT、action-flow objective、
H8/R8、solver 10、checkpoint set、optimizer、batch、训练样本数与评测 initial states。

| ID | Video-DiT adaptation | Dynamics output | Action condition | 作用 |
|---|---|---|---|---|
| LR-M0 | frozen base | none | matched base hidden | no-dynamics-adaptation lower bound |
| LR-M1 | all-layer rank-64 LoRA | ordinary adapted field | O2 readout | current strong upper anchor |
| LR-M2 | parameter-matched LoRA | ordinary adapted field | matched action interface | 参数空间低秩 baseline |
| LR-M3 | frozen base + full-rank side adapter | full-rank \(\Delta V\) | invariant delta pooling | 显式 residual、无 rank constraint |
| LR-M4 | frozen base + low-rank delta | rank-\(r\) \(\Delta V\) | independent base-hidden branch | low-rank dynamics、无共享 code |
| LR-M5 | frozen base + low-rank delta | rank-\(r\) \(\Delta V\) | proposed invariant \(C_\Delta\) | 完整方法 |
| LR-M6 | frozen base | none | action-space residual head | residual 放在 action space 的对照 |

LR-M3 与 LR-M5 回答 output rank constraint；LR-M2 与 LR-M5 回答参数空间 LoRA 和
输出空间 delta 的区别；LR-M4 与 LR-M5 回答共享动力学—动作表示是否必要。动作头大小
不得为 LR-M5 单独增加。

### 8.1 训练 loss

动力学分支：

\[
\mathcal L_{\Delta}
=
\left\|V_0+\Delta V_{\phi}-V^*\right\|_2^2.
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

### 8.2 G3 成功门槛

LR-M5 只有同时满足以下条件才进入 G4：

1. 相对 LR-M1/O2 anchor 的 paired 95% CI 下界高于预注册 non-inferiority margin
   `-2 pp`；
2. 至少 3 个 training seeds，固定 10 tasks × 50 trials 与相同 initial-state IDs；
3. 相比 LR-M1，VDT-side trainable parameters 至少减少 5 倍；
4. peak training memory 至少降低 30%，或达到同一成功率门槛的 accelerator-hours 至少
   降低 2 倍；
5. 一次 action chunk 仍只运行一次 frozen VDT forward，不增加 video denoising rollout；
6. 为支持“delta 是控制表示”，LR-M5 相对 LR-M4 的 paired 95% CI 下界必须高于 0；
   若二者等效，只能保留低秩 dynamics claim，删除 shared delta-code claim；
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
current observation + language + noised real future latent
                         ↓
                  frozen Video-DiT
                         ↓
                   V0 and base h
                         ↓
              low-rank delta adapter
                 P[N,r] · Q[C,r]^T
                  ↙               ↘
         video residual loss    invariant C_delta
                                      ↓
                                 Action-DiT
                                      ↓
                              action flow loss
```

- delta adapter 不接收 expert action；
- true future 只用于构造 video-flow target 与训练监督；
- action loss 是否反传到 delta adapter 必须显式注册；
- frozen VDT 必须验证无 optimizer entry、无非零 gradient、无权重漂移。

### 10.2 部署

```text
current observation + language + D-winner temporal input
                         ↓
              one frozen Video-DiT forward
                         ↓
                low-rank factors P,Q
                         ↓
                  invariant C_delta
                         ↓
                Action-DiT, solver 10
                         ↓
                 H8 action chunk, R8
```

部署不读取真实未来、不生成未来 RGB、不运行 iterative video solver。若最终需要两个 VDT
forwards 才达到性能，必须作为不同方法重新登记，并重新计算 latency/efficiency。

## 11. 统计、成本与结果记录

### 11.1 主要指标

| 类别 | 指标 |
|---|---|
| Representation | \(E(r)\)、effective rank、cross-task principal angles、residual reconstruction error |
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
| Frozen + base hidden | — |  |  |  |  |  |  |  |
| O2 all-layer LoRA | 64/parameter rank |  |  |  |  |  |  |  |
| Parameter-matched LoRA |  |  |  |  |  |  |  |  |
| Full-rank delta | full |  |  |  |  |  |  |  |
| Low-rank delta, no sharing |  |  |  |  |  |  |  |  |
| **LRD-WAM** |  |  |  |  |  |  |  |  |

另附三张不能省略的图：

1. `rank → residual energy / reconstruction`，按 suite 与 timestep 分层；
2. `rank → closed-loop success / GPU-hours` Pareto；
3. `action success → video field drift`，比较 frozen、LoRA、full-rank delta 与 LRD-WAM。

## 13. 论文级结论的降级路径

| 观察 | 允许的结论 | 不允许的包装 |
|---|---|---|
| G1 谱不低秩 | robot residual 在该 carrier/合同下没有稳定低秩证据 | 继续称“低秩场方法” |
| Oracle 强、policy view 弱 | 真实未来 residual 含动作信息，但不可部署预测 | 用 oracle probe 声称 WAM 有效 |
| Low-rank ≈ full-rank，但不优于 base hidden | delta 可压缩，但不是更好的动作接口 | “动力学增量是控制充分表示” |
| LR-M5 性能过门、成本不过门 | representation method | efficient adaptation method |
| LR-M5 成本下降、性能未过门 | compression/efficiency trade-off | non-inferior high-performance WAM |
| Object 成立、Long 失败 | suite-specific method | general VDT-to-WAM principle |
| 多 suite + matched controls + 成本均成立 | method paper 候选 | 在未做 novelty search 前声称“首次” |

## 14. 最小下一步

当前不应直接实现完整 LRD-WAM。最小、信息增益最高的下一步是：

1. 完成 G0，对齐并复核 Model3/O2/Model5 的真实 server artifacts；
2. 在 Object 与 Long 的同合同窗口上提取 \(V_0,V^*,R^*\)；
3. 完成 G1 的 per-sample、across-sample 与 control spectra；
4. 只在 G1 通过后，为 G2 编写 server handoff 与小规模 deployable probe；
5. 只有 G2 通过，才申请 G3 的 Object matched training 预算。

这条顺序把最便宜的证伪实验放在前面，并把核心主张固定为：

\[
\boxed{
\text{Pretrained Video Dynamics}
+
\text{Rank-Constrained Robot Delta}
\rightarrow
\text{Control-Sufficient Delta Code}
}
\]

而不是“给 Model3 再加一个 LoRA 或 action head”。
