# LRD-WAM G2 Object + Long 正式结果与 G2-R 后续诊断

更新时间：2026-08-02。

> **终态覆盖说明。** 本页正文按时间记录 G2 与 G2-R 的阶段性状态；其中
> “D1 不授权”与 D1 audit `not_triggered` 只描述当时的 G2-R 决策。随后单独冻结的
> [G2-R2 gated audit](lrd-wam-g2r2-gated-residual-complementarity-result.md) 与
> [D1 residual-branch matrix](lrd-wam-g2r2-d1-carrier-result.md) 均已完成。D1
> 终态为 `d1_gain_not_residual_specific`。2026-08-03 的后续判断将其表述为：原版
> future-field 低秩方法主张尚未成立，但现有离线证据不足以证明 current-only 辅助表示
> 在闭环一定无价值。本页不授权原 G3/D2；另行冻结的小型 D1/current-only 闭环见
> [独立实验页](d1-current-only-auxiliary-closed-loop-experiment.md)。

## 一句话结论

**G2 未通过，G2-R 也未建立 residual complement 主张；均不进入原 G3。**

G1 证明了 pretrained Wan 的逐输入 future residual 具有稳定低秩结构；G2
进一步检验这种结构能否形成部署可用、对动作充分的 delta code。结果显示：
G2-A 在 Object 和 Long 都失败；G2-B 虽在 Long 上显著改善，但在 Object 上
没有达到门槛，而且 D2 没有稳定优于 D1。因此不能把 G1 的表示层低秩结论升级为
跨套件的控制充分性结论。

后续 G2-R 没有重开 G2，也没有读取旧 G2 test。它保留相同的 frozen-base
memory，单独检验 $I(action;\ delta \mid P0)>0$ 是否能由 predicted residual
补足。Object 的 oracle residual 有离线互补信号，但所有 predicted residual
分支都不满足 useful 和 specific 门槛；Long 连 oracle viability 都未通过。
因此 G2-R 的跨套件判定是
`close_deployable_delta_complement_route`，不授权 D1、A7、Action-DiT、G3 或
闭环 rollout。

## 冻结合同

- 数据：LIBERO Object 420 episodes / 1680 个 H8 windows；Long 300 episodes /
  1200 个 H8 windows。每个 episode 固定四个窗口。
- split：Object `240/90/90`，Long `180/60/60`，按 episode 隔离
  train/validation/test。
- seeds：`4201 / 4202 / 4203`。
- D2：一个 current latent + 一个 Gaussian future slot，temporal
  $[0, 1000]$，`action=None`；D1：current-only，temporal $[0]$。
- 真实 future 只构造 target，不进入 Wan 输入；future noise 与 scheduler target
  使用同一份 noise。
- $(N, D, C_p) = (392, 1536, 64)$，future token slice $[392, 784)$，rank grid
  $\{2, 4, 8, 16\}$，主 rank 为 8。
- 主对照：LR-P0 / LR-P2 / LR-P3 / LR-P5；LR-P1 仅作 oracle；D1-LR-P3
  仅作 temporal control。
- 干预：zero delta、episode-shuffled delta、detached independent；不重训 decoder。
- G2-A：linear 与两层 MLP probe；G2-B：固定 16-layer Action-DiT、4 次 flow
  draw、10-step solver。
- 主门槛：P3 相对 P0/P5 三个 seed 同方向、中位改善至少 5%、paired 95% CI
  下界大于 0；干预退化中位数至少 2%；同时满足 P2、D1 与 rank saturation
  合同。

唯一 video base 是原始 pretrained Wan2.1-T2V-1.3B。没有加载 Model3、O2、
Model5 或其他机器人 checkpoint；没有对 Wan 创建 optimizer 或执行 backward。

## G2-A 结果

下表为 LR-P3 rank-8 相对对照的 test episode-macro MSE 改善；正数代表 P3
更好，负数代表 P3 更差。

| Suite / 对照 | seed 4201 | seed 4202 | seed 4203 | paired pooled 结果 |
|---|---:|---:|---:|---:|
| Object vs P0 | -3.27% | -1.18% | -3.67% | -2.71%，95% CI $[-5.37\%, -0.16\%]$ |
| Object vs P5 | -2.92% | +2.25% | +1.30% | +0.24%，95% CI $[-4.28\%, 4.63\%]$ |
| Object D2 vs D1 | -9.15% | -7.33% | -5.81% | -7.41%，95% CI $[-10.14\%, -4.74\%]$ |
| Long vs P0 | -6.14% | -7.47% | -3.90% | -5.83%，95% CI $[-10.16\%, -2.08\%]$ |
| Long vs P5 | -6.52% | -4.29% | -3.36% | -4.72%，95% CI $[-11.41\%, 1.43\%]$ |
| Long D2 vs D1 | -23.02% | -19.75% | -13.84% | -18.78%，95% CI $[-24.09\%, -13.48\%]$ |

此外，rank-8 对 rank-16 的 saturation 检查在 Object 与 Long 都未全部通过；
linear probe 对 P0 的方向检查也失败。G2-A 因而明确判定为未通过。

## G2-B 结果

| Suite / 对照 | seed 4201 | seed 4202 | seed 4203 | paired pooled 结果 | Gate |
|---|---:|---:|---:|---:|---:|
| Object vs P0 | -0.05% | -0.37% | +2.47% | +0.74%，95% CI $[-1.45\%, 2.75\%]$ | 未通过 |
| Object vs P5 | -1.38% | +2.29% | +0.63% | +0.51%，95% CI $[-1.15\%, 2.14\%]$ | 未通过 |
| Object vs P2 | +1.79% | +0.30% | +2.75% | +1.66%，95% CI $[-0.23\%, 3.45\%]$ | 非劣通过 |
| Object D2 vs D1 | -4.58% | -0.19% | -1.43% | -2.06%，95% CI $[-3.23\%, -0.97\%]$ | 非劣通过 |
| Long vs P0 | +3.95% | +8.71% | +12.03% | +8.26%，95% CI $[6.23\%, 10.29\%]$ | 通过 |
| Long vs P5 | +0.27% | +11.33% | +7.48% | +6.39%，95% CI $[3.55\%, 9.30\%]$ | 通过 |
| Long vs P2 | -3.74% | +1.39% | +7.29% | +1.76%，95% CI $[0.18\%, 3.32\%]$ | 非劣通过 |
| Long D2 vs D1 | -3.90% | -5.40% | -5.21% | -4.80%，95% CI $[-6.49\%, -3.13\%]$ | 未通过 |

G2-B 的关键信息不是“完全没有信号”，而是**信号不能跨 Object 与 Long
稳定成立**。Long 的 P3 提升真实且通过门槛；Object 的效果小、种子方向不一致、
区间跨 0。冻结合同要求两个 suite 全部通过，所以不能用 Long 的正结果覆盖 Object
失败。

## 干预结果

- G2-A 的 zero、shuffle、detached 干预在两个 suite 都通过退化门槛。
- G2-B 的 zero 与 detached 在 Object/Long 都显著退化。
- G2-B Long shuffle 使 loss pooled 上升约 24.36%，通过门槛。
- G2-B Object shuffle 只使 loss pooled 上升约 1.02%，三个 seed 的 loss
  增幅中位数约 0.52%，低于冻结的 2% 门槛，因此未通过。

这说明 delta 通道总体被 decoder 使用，但在 Object 上没有形成足够强、足够
sample-specific 的优势；“有信息被使用”不等于“相对容量匹配对照有稳定控制增益”。

## 审计与成本

- 训练矩阵：18 dynamics + 54 G2-A + 15 G2-B，共 87 个工作项。
- 所有匹配模型的初始状态 SHA 审计通过；87 个 checkpoint/metrics 的数量与 SHA
  校验通过。
- test 只在所有训练完成后首次读取，没有用于 checkpoint、rank 或超参数选择。
- 总成本 `5.0963 accelerator-hours`，低于冻结上限 16 小时。
- 执行时 artifact 终端验证为 `pass`；科学 Gate 为 `fail`。当前归档中的
  `result_validation.json` 仅因后续公共文档变化导致
  `source_identity_current` 不匹配而记为 `fail`，其余检查通过；这是 provenance
  warning，不是实验未完成。科学结论始终为 `fail_stop_before_g3`。
- 未启动 G3、闭环 rollout 或正式机器人方法训练。

## 最终解释与下一步边界

保留的科学结论是：**Wan future residual 在逐输入层面稳定低秩，但当前
LR-P3/D2 构造不能被认定为跨 Object+Long 的部署可用动作充分 code。**

因此下一步不是自动验证 Long、实现 G3 或做闭环。任何继续工作都必须提出新的
科学假设并另行冻结合同，例如解释为什么 Long 有效而 Object 无效，或重新定义
能够隔离 D2 增量信息而不被 D1/current-state shortcut 覆盖的目标与对照。

## G2-R residual complementarity 后续诊断

### 目的与证据边界

G2-R 是在 G2 `fail_stop_before_g3` 之后冻结的 cache-only 诊断，而不是 G2
补票、G3 或新的控制实验。它最多回答在保留 $P0$ 后，residual memory 是否提供
额外的离线 H8 action 信息。所有结果标记为
`diagnostic_on_reused_validation`：G2 的 validation 曾用于 dynamics checkpoint
选择，且 G2 test 已于 2026-08-01 读取，因而本节不构成新的 held-out
generalization 或闭环证据。

- Object 只使用 G2 的 train/validation $240/90$ episodes，即
  $960/360$ 个四窗口 H8 records；Long 只使用 $180/60$ episodes，即
  $720/240$ 个 records。
- mixed parent shards 可为 SHA 校验而反序列化，但实现立即过滤到
  train/validation view。终验记录 test record exposure、test tensor forward 和
  test metric 均为 $0$。
- 父 G2 的两份文档 source-hash drift 由只读 `model` Git-object historical
  source bundle 解决：commit
  `2ce1a15b9bb91d7b67021b9a8f00cc5f938709d8` 精确恢复两份冻结文档的 SHA-256；
  当前用户文档未回滚或覆盖。
- G2-R 不运行 Wan/VAE forward、Wan optimizer/backward、Action-DiT 或 robot
  checkpoint load；终验的这些 runtime counters 均为 $0$。

### 冻结矩阵与实现合同

所有条件使用同一个 $M0_{D2}\in\mathbb{R}^{B\times64\times512}$ base memory，
并在每个 branch 上执行 LayerNorm 和 $512\rightarrow512$ projector；拼接后的
memory 为 $[B,64,1024]$，token mean 与 proprio $[8]$ 合并后经 MLP2 输出
normalized H8 action $[B,8,7]$。A0 在第二 branch projector 后施加不可训练的
exact-zero mask，因此网络结构和 action-stage 参数量与其它条件完全相同。

| ID | 输入 | 角色 |
|---|---|---|
| A0 | $[M0,\ zero]$ | base-only matched 下界 |
| A1 | $[M0,\ M3_{D2}]$ | predicted rank-8 residual 主候选 |
| A2 | $[M0,\ M5]$ | parameter-matched side-adapter 容量对照 |
| A3 | $[M0,\ shuffle(M3_{D2})]$ | A1 checkpoint 的 task-plus-phase deranged validation 干预；不重训 |
| A4 | $[M0,\ M1]$ | oracle rank-8 residual；不可部署上限 |
| A5 | $[M0_{D2},\ M3_{D1}]$ | D1 predicted rank-8 delta branch |
| A6 | $[M0,\ M2]$ | predicted full-rank residual |

完成的初始训练矩阵是 $6\times3=18$ 个作业：conditions A0/A1/A2/A4/A5/A6，
seeds 4201/4202/4203；每项固定 AdamW、lr $10^{-3}$、weight decay $10^{-4}$、
batch size $64$、1,500 optimizer steps、float32 loss/metrics。每个 seed 的六个
条件都通过 fusion/decoder identical-initialization 和 suite-balanced sampler-order
SHA 审计；A0 second branch max-abs 严格为 $0$。A3 与 A5-shuffle 都按
`(task_id, phase_id)` 在不同 episodes 间 deterministic derangement，fixed point、
same-episode、task mismatch 与 phase mismatch 均为 $0$。

### Validation episode-macro MSE

下表为 reused validation 的 episode-macro normalized H8 action MSE，越低越好；
每格依次为 seeds 4201 / 4202 / 4203。

| 条件 | Object | Long |
|---|---:|---:|
| A0 | 0.082184 / 0.077049 / 0.083977 | 0.078462 / 0.069995 / 0.072773 |
| A1 | 0.093241 / 0.087722 / 0.083465 | 0.081986 / 0.077676 / 0.074463 |
| A2 | 0.083009 / 0.090398 / 0.087833 | 0.075881 / 0.080776 / 0.083789 |
| A3 | 0.096403 / 0.090142 / 0.087217 | 0.088357 / 0.079778 / 0.081933 |
| A4 | 0.071230 / 0.069084 / 0.064439 | 0.072328 / 0.070364 / 0.069089 |
| A5 | 0.085244 / 0.089978 / 0.083814 | 0.074745 / 0.069563 / 0.079680 |
| A5-shuffle | 0.093888 / 0.096221 / 0.087422 | 0.087975 / 0.078794 / 0.089535 |
| A6 | 0.087138 / 0.080954 / 0.088841 | 0.078042 / 0.075779 / 0.078665 |

改善定义为 $I(A,B)=(loss_B-loss_A)/loss_B$；正数表示 treatment $A$ 更好。置信
区间使用 10,000 次 task-stratified、episode-clustered paired bootstrap，三个
decoder seeds 的同 episode records 视为一个 cluster。

| 比较 | 三个 seed 改善 | seed 中位数 | pooled 改善与 95% CI | 冻结判读 |
|---|---:|---:|---:|---|
| Object A4 vs A0 | +13.33% / +10.34% / +23.27% | +13.33% | +15.81%，$[10.10\%,21.23\%]$ | oracle pass |
| Long A4 vs A0 | +7.82% / -0.53% / +5.06% | +5.06% | +4.27%，$[-2.39\%,10.31\%]$ | oracle fail |
| Object A1 vs A0 | -13.45% / -13.85% / +0.61% | -13.45% | -8.72%，$[-12.39\%,-5.27\%]$ | useful fail |
| Long A1 vs A0 | -4.49% / -10.97% / -2.32% | -4.49% | -5.83%，$[-10.04\%,-1.85\%]$ | useful fail |
| Object A3 vs A1 | +3.28% / +2.69% / +4.30% degradation | +3.28% | +3.41%，$[-0.20\%,6.99\%]$ | sample-specific fail |
| Long A3 vs A1 | +7.21% / +2.63% / +9.12% degradation | +7.21% | +6.38%，$[3.35\%,9.53\%]$ | shuffle gate pass，但 A1 仍不 useful |

A1 对 A2 的 specific 检查在 Object 与 Long 均未满足三 seed 同向、5% 中位改善和
CI 下界大于零的联合门槛。A5 与 A6 对 A0 也都未满足 useful 条件。因此 Object 虽有
oracle-only signal，却没有部署输入可稳定预测的 complement；Long 则在 oracle 层已
停止。A5 没有得到 D1 candidate，A6 没有得到 rank-8 bottleneck candidate，故完整
D1 audit 和 rank-16 A7 均按合同 `not_triggered`。

### G2-R 最终判读与边界

- Object：`oracle_teacher_only_not_deployable`。
- Long：`close_endpoint_residual_complement`。
- 跨套件：`close_deployable_delta_complement_route`。
- artifact validator：`pass`，共 38 项检查；18/18 initial work items 完整，
  conditional work items 为 $0$。

正确的解释是：Object endpoint residual 在 oracle 形式下携带可以补充 $P0$ 的
离线 action 信息，但现有 D2/D1 rank-8 和 D2 full-rank predictors，以及普通
parameter-matched side capacity，都没有把这种信息变成稳定、跨套件且 sample-specific
的 deployable complement。Long 没有证明 endpoint residual 自身具备所需的 oracle
互补性。这个负结论仅关闭当前 endpoint/output-space residual 加 concat readout 合同；
它不声称 residual 与 action 没有任何因果关系，也不替代新的交叉验证、fresh held-out
或闭环合同。

## 本地产物

当前本地封存的完整机器可读证据位于：

`runs/I-003/ldr_wam/20260801_lrd_wam_g2_object_long/`

该目录在终态后从执行时的
`runs/I-003/model5/20260801_lrd_wam_g2_object_long/` 迁入封存根；artifact 内部
旧路径保留为执行溯源。

主要文件：

- `protocol_contract.json`
- `training_manifest.json`
- `g2a_metrics.json`
- `g2b_metrics.json`
- `g2_decision.json`
- `result_validation.json`
- `source_identity.json`
- `environment.json`
- `run_report.md`

运行产物、数据缓存和 checkpoint 不进入 Git；本报告与冻结合同进入 `model/`
文档仓库。

G2-R 当前本地封存的完整机器可读证据位于：

`runs/I-003/ldr_wam/20260802_lrd_wam_g2r_residual_complementarity/`

其执行时路径为
`runs/I-003/model5/20260802_lrd_wam_g2r_residual_complementarity/`，同样保留在
artifact 内部作为不可变 provenance。

主要文件：

- `g2r_protocol_contract.json`
- `reuse_identity.json`
- `training_manifest.json`
- `validation_metrics.json`
- `g2r_decision.json`
- `result_validation.json`
- `run_report.md`
- `commands.txt`
- `environment.json`
