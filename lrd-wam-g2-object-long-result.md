# LRD-WAM G2 Object + Long 正式结果

更新时间：2026-08-01。

## 一句话结论

**G2 未通过，正式停止在 G2，不进入 G3。**

G1 证明了 pretrained Wan 的逐输入 future residual 具有稳定低秩结构；G2
进一步检验这种结构能否形成部署可用、对动作充分的 delta code。结果显示：
G2-A 在 Object 和 Long 都失败；G2-B 虽在 Long 上显著改善，但在 Object 上
没有达到门槛，而且 D2 没有稳定优于 D1。因此不能把 G1 的表示层低秩结论升级为
跨套件的控制充分性结论。

## 冻结合同

- 数据：LIBERO Object 420 episodes / 1680 个 H8 windows；Long 300 episodes /
  1200 个 H8 windows。每个 episode 固定四个窗口。
- split：Object `240/90/90`，Long `180/60/60`，按 episode 隔离
  train/validation/test。
- seeds：`4201 / 4202 / 4203`。
- D2：一个 current latent + 一个 Gaussian future slot，temporal
  `[0,1000]`，`action=None`；D1：current-only，temporal `[0]`。
- 真实 future 只构造 target，不进入 Wan 输入；future noise 与 scheduler target
  使用同一份 noise。
- `(N,D,C_p)=(392,1536,64)`，future token slice `[392,784)`，rank grid
  `2/4/8/16`，主 rank 为 8。
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
| Object vs P0 | -3.27% | -1.18% | -3.67% | -2.71%，95% CI [-5.37%, -0.16%] |
| Object vs P5 | -2.92% | +2.25% | +1.30% | +0.24%，95% CI [-4.28%, 4.63%] |
| Object D2 vs D1 | -9.15% | -7.33% | -5.81% | -7.41%，95% CI [-10.14%, -4.74%] |
| Long vs P0 | -6.14% | -7.47% | -3.90% | -5.83%，95% CI [-10.16%, -2.08%] |
| Long vs P5 | -6.52% | -4.29% | -3.36% | -4.72%，95% CI [-11.41%, 1.43%] |
| Long D2 vs D1 | -23.02% | -19.75% | -13.84% | -18.78%，95% CI [-24.09%, -13.48%] |

此外，rank-8 对 rank-16 的 saturation 检查在 Object 与 Long 都未全部通过；
linear probe 对 P0 的方向检查也失败。G2-A 因而明确判定为未通过。

## G2-B 结果

| Suite / 对照 | seed 4201 | seed 4202 | seed 4203 | paired pooled 结果 | Gate |
|---|---:|---:|---:|---:|---:|
| Object vs P0 | -0.05% | -0.37% | +2.47% | +0.74%，95% CI [-1.45%, 2.75%] | 未通过 |
| Object vs P5 | -1.38% | +2.29% | +0.63% | +0.51%，95% CI [-1.15%, 2.14%] | 未通过 |
| Object vs P2 | +1.79% | +0.30% | +2.75% | +1.66%，95% CI [-0.23%, 3.45%] | 非劣通过 |
| Object D2 vs D1 | -4.58% | -0.19% | -1.43% | -2.06%，95% CI [-3.23%, -0.97%] | 非劣通过 |
| Long vs P0 | +3.95% | +8.71% | +12.03% | +8.26%，95% CI [6.23%, 10.29%] | 通过 |
| Long vs P5 | +0.27% | +11.33% | +7.48% | +6.39%，95% CI [3.55%, 9.30%] | 通过 |
| Long vs P2 | -3.74% | +1.39% | +7.29% | +1.76%，95% CI [0.18%, 3.32%] | 非劣通过 |
| Long D2 vs D1 | -3.90% | -5.40% | -5.21% | -4.80%，95% CI [-6.49%, -3.13%] | 未通过 |

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
- artifact 终端验证为 `pass`；科学 Gate 为 `fail`。两者含义不同：前者说明
  实验按合同完整执行，后者说明科学假设未达到门槛。
- 未启动 G3、闭环 rollout 或正式机器人方法训练。

## 最终解释与下一步边界

保留的科学结论是：**Wan future residual 在逐输入层面稳定低秩，但当前
LR-P3/D2 构造不能被认定为跨 Object+Long 的部署可用动作充分 code。**

因此下一步不是自动验证 Long、实现 G3 或做闭环。任何继续工作都必须提出新的
科学假设并另行冻结合同，例如解释为什么 Long 有效而 Object 无效，或重新定义
能够隔离 D2 增量信息而不被 D1/current-state shortcut 覆盖的目标与对照。

## 本地产物

完整机器可读证据位于：

`runs/I-003/model5/20260801_lrd_wam_g2_object_long/`

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
