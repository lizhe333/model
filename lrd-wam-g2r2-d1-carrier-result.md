# LRD-WAM G2-R2-D1 Object carrier-matched result

更新时间：2026-08-02。

## 一句话结论

**D1/P3 相对 P0 有强且样本特异的离线 action 增益，但没有稳定优于严格匹配的
D1/P5 容量对照，因此不能把收益归因于低秩 residual。**

冻结决策为 `d1_gain_not_residual_specific`。父 G2 的
`fail_stop_before_g3` 与当前 endpoint-residual deployment route 的关闭结论保持
不变。

## 实验边界

本轮是已完成 [G2-R2 gated residual 诊断](./lrd-wam-g2r2-gated-residual-complementarity-result.md)
之后单独冻结的 Object cache-only D1 矩阵，证据标签为
`diagnostic_on_reused_validation_d1_v1`。

- 仅使用 Object train/validation 的 $240/90$ episodes、$960/360$ H8 windows。
- 保留原 G2-R2 的 D2 P0 base，只将补充分支限定为 D1；没有同时更换 base。
- seeds 为 $4201/4202/4203$；每个训练条件固定 AdamW、$1500$ steps、batch size
  $64$、float32。
- 旧 G2 test 没有进入 retained tensor、batch、forward、loss、prediction 或 metric。
- Wan、VAE、Action-DiT、父实验 G3、robot rollout、rank/fusion sweep 均未启动。

## 匹配矩阵

| 条件 | 输入 | 作用 |
|---|---|---|
| D1-B0 | P0 + strict zero residual | base-only control |
| D1-B1 | P0 + D1/P3 | D1 low-rank treatment |
| D1-B2 | P0 + D1/P5 | D1-input capacity control |
| D1-B3 | P0 + shuffled D1/P3 | D1-B1 checkpoint intervention |
| D1-B4 | P0 + oracle P1 | nondeployable upper bound |

D1/P5 只读取 `d1_tokens`。其 $128\to104\to512$ side network 加
$36$-channel trainable calibration vector，共 $67{,}468$ 个参数，与 D1/P3
representation 参数量精确一致。所有训练条件都实例化并 optimizer-own 同一个完整
fusion、decoder 与 D1/P5 module；同 seed 的初始 state、sampler order 与总参数量
完全一致。

D1-B0 在 gate/value 计算后持续强制 residual 为精确零，因此训练后仍严格为
base-only。D1-B3 在同 task、同 normalized phase 内跨 episode derangement，固定点、
same-episode、task mismatch 与 phase mismatch 均为 $0$，且不修改 D1-B1 checkpoint。

## 直接结果

正数表示 D1/P3 的 Object validation episode-macro normalized H8 action MSE
低于对应 control。paired CI 使用 $10{,}000$ 次 task-stratified、
episode-clustered bootstrap，并把同 episode 的三个 decoder seeds 保持在同一 cluster。

| 比较 | seed 4201 | seed 4202 | seed 4203 | paired pooled 改善与 $95\%$ CI |
|---|---:|---:|---:|---:|
| D1/P3 vs P0 | $+15.31\%$ | $+8.74\%$ | $+11.96\%$ | $+12.03\%$, $[5.75\%,17.75\%]$ |
| D1/P3 vs D1/P5 | $-2.11\%$ | $-6.81\%$ | $+15.07\%$ | $+2.95\%$, $[-2.09\%,7.71\%]$ |
| D1/P3 vs shuffled D1/P3 | $+8.22\%$ | $+7.56\%$ | $+6.87\%$ | $+7.55\%$, $[3.04\%,11.93\%]$ |
| oracle P1 vs P0 | $+31.56\%$ | $+30.70\%$ | $+33.43\%$ | $+31.90\%$, $[26.38\%,37.00\%]$ |

对应的三 seed episode-macro MSE 为：

| 条件 | seed 4201 | seed 4202 | seed 4203 |
|---|---:|---:|---:|
| D1-B0 | $0.092922$ | $0.090873$ | $0.091412$ |
| D1-B1 | $0.078700$ | $0.082934$ | $0.080475$ |
| D1-B2 | $0.077073$ | $0.077645$ | $0.094752$ |
| D1-B3 | $0.085746$ | $0.089716$ | $0.086408$ |
| D1-B4 | $0.063594$ | $0.062979$ | $0.060854$ |

## 判读

Oracle、D1/P3 useful 和 D1/P3 shuffle 三项 gate 均通过。失败点只在
residual specificity：D1/P3 对 D1/P5 的三个 seed 方向不一致，其中两个 seed
明确由普通 D1 side feature 更好；pooled CI 也跨 $0$。

因此保留的结论是：D1 输入中存在可用于 action readout 的逐样本信息，但当前证据
不能说明低秩 residual factorization 是获得该信息的必要机制。D1/P3 的强 base gain
可以被普通、严格参数匹配的 D1 side capacity 解释。

这项结果关闭“仅凭 D1/P3 即可救活当前低秩 residual route”的解释。它不否定未来
独立提出新的 action-aligned predictor 假设，但任何继续都需要新的冻结合同与新的
held-out 边界；本轮不授权直接进入 G3 或 closed loop。

## 本地可复核产物

完整机器可读证据位于：

`runs/I-003/model5/20260802_lrd_wam_g2r2_d1_carrier_matrix/`

主要文件为 `d1_protocol_contract.json`、`reuse_identity.json`、
`d1_metrics.json`、`d1_decision.json`、`d1_run_report.md`、
`training_manifest.json`、`commands.txt`、`environment.json` 和
`result_validation.json`。终端 artifact validation 为 `pass`，accelerator-hours 为
$0.0$，aggregate CPU job wall time 约 $0.280$ 小时。checkpoint、cache 与原始日志不
进入 Git。
