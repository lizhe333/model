# LRD-WAM G2-R2 gated residual complementarity result

更新时间：2026-08-02。

## 结论

G2-R2 已完成，父实验 G2 的结论仍为
`fail_stop_before_g3`。本轮的冻结决策为
`oracle_target_valuable_but_deployment_predictor_or_input_not_sufficient`。

Object 上，oracle future residual 的确含有可补充 base memory 的离线 action
信息；但现有 predicted residual、full-rank residual、current-only residual 与
parameter-matched side input 都没有满足部署所需的联合证据。门控读出也没有可靠地
优于此前的 concat 读出。因此当前 Object endpoint-residual 主张未成立，不启动原
G3、D2/noisy-future 闭环、rank sweep 或泛化 fusion 调参。

这不是“residual 没有价值”的结论：它只说明在当前缓存、输入构造、预测器和
action readout 合同下，oracle 的价值尚未被转化为足够强且 residual-specific 的
部署输入。随后独立冻结的 D1-input residual-branch 矩阵已经完成；它仍保留 D2 P0
base，只替换补充分支，并进一步表明 D1/P3 不能稳定优于严格参数匹配的 D1/P5
容量对照。结合该后续结果，output-space endpoint rank-$8$ residual 不能作为已经
成立的核心部署机制或方法创新。2026-08-03 另行冻结的 D1/current-only 小型闭环只问
辅助控制信息能否转化为闭环收益，不复活 future-field LRD-WAM。

## 证据边界与冻结合同

- 证据标签为 `diagnostic_on_reused_validation_v2`，终端 artifact 验证为 `pass`。
- 主矩阵为 Object validation 的八个 action-readout 条件，每个条件使用 seeds
  `4201`、`4202`、`4203`；predictor-quality 诊断还汇总了 Object 与 Long
  validation cache。
- 所有处理共享 $M0_{D2}\in\mathbb{R}^{B\times64\times512}$ base memory。G0 的
  第二分支是字面 exact-zero，G1 为 D2/P3 rank-$8$ predicted residual，G2 是
  parameter-matched P5 side input，G3 是 G1 的 task-phase deranged shuffle，G4
  为 oracle P1 residual，G5 是 D1/P3 current-only residual，G6 为 D2/P2
  full-rank residual，G7 是 G4 的 task-phase constrained shuffle。
- 旧 G2 test record 没有进入 batch、representation forward、loss、prediction 或
  metric；Wan、VAE、Action-DiT、父实验的 G3 阶段、rank-$16$/$32$ 与 closed-loop
  rollout 均未启动。
- accelerator 使用量为 $0.0$ 小时；CPU 作业墙钟时间合计约 $1.015$ 小时。

改善定义为
$$
I(A,B)=\frac{\operatorname{loss}(B)-\operatorname{loss}(A)}{\operatorname{loss}(B)},
$$
正数表示 treatment $A$ 更好。区间是 task-stratified、episode-clustered paired
bootstrap 的 $95\%$ CI。

## 主结果

下表的 G0--G7 是本轮 G2-R2 action-readout 条件编号，而不是父实验的阶段编号。

| 比较 | pooled 改善与 $95\%$ CI | 三 seed / 门槛判读 |
|---|---:|---|
| G4 oracle P1 vs G0 | $+32.02\%$, $[25.50\%, 37.87\%]$ | oracle viable |
| G7 shuffled oracle vs G4 | $46.99\%$ degradation, $[43.59\%, 50.43\%]$ | oracle sample-specific |
| G1 gated D2/P3 vs G0 | $+5.09\%$, $[0.33\%, 9.63\%]$ | seed median $4.44\%$，低于 strong-Go 的 $5\%$ |
| G1 gated D2/P3 vs G2/P5 | $+3.77\%$, $[-0.97\%, 8.22\%]$ | 未证明 residual-specific，容量对照未排除 |
| G3 shuffled G1 vs G1 | $19.82\%$ degradation, $[15.75\%, 23.78\%]$ | G1 内容具有 sample-specific 性，但不足以转为 deployable 结论 |
| G6 full-rank P2 vs G0 | $-2.58\%$, $[-8.61\%, 3.08\%]$ | full-rank 不 useful |
| G5 D1/P3 vs G1 | $+6.70\%$, $[1.37\%, 11.81\%]$ | 当时触发独立 D1 审计；该审计随后失败于 matched-capacity specificity 并关闭路线 |
| G1 gated vs 原 G2-R concat A1 | $+0.39\%$, $[-4.51\%, 5.23\%]$ | 无可靠 fusion 改善 |

G1 相对 G0 有弱的正向信号，但没有达到三 seed 强-Go 门槛，也没有相对 P5 的
特异性证据；因此不能把它解释为可部署 residual complement。G4/G7 组合则明确
说明 oracle target 有价值且这种价值是逐样本的。

## Predictor 与预算诊断

P2/P3 对 oracle endpoint residual 的重建可解释约 $62\%$--$63\%$ 的能量，且
P3 在 Object 与 Long 的原始重建质量都略优于 P2。与此同时，按 window 计算的
explained energy 与原 action-MSE 改善的相关性接近 $0$。因此 endpoint video
residual 拟合本身不是充分的 action-control 选择准则，也不能把 full-rank 的失败
简单归因于 rank-$8$ 瓶颈。

冻结条件满足两个边界信号后，额外执行了 Object 的
$750/1500/3000$ step 诊断，条件为 G0、G1、G4、G6。它不改变固定 $1500$ step
主矩阵或上述决策；例如 G0 的三个 seed 在 $3000$ step 均比 $1500$ step 更差。

## 后续边界

- 当前 Object endpoint-residual deployment claim 未成立；D1 后续也已完成并失败于
  residual-specific capacity control。原 G3 与 D2/noisy-future 路径继续停止。
- 不继续 low-rank formula、predictor scaling、rank sweep 或 generic fusion tuning，
  也不把该表示作为核心方法创新包装。
- 新的 D1/current-only auxiliary closed-loop pilot 是一个真正不同且单独冻结的问题；
  它不能继承 LRD-WAM 的方法主张、旧 test 或 D2 执行授权。
- 本结果不替代 fresh held-out generalization、closed-loop 或机器人成功证据。

## 可复核产物

本地可复核产物目录为
`runs/I-003/ldr_wam/20260802_lrd_wam_g2r2_gated_residual_complementarity_audit/`。
该目录在终端验证后从执行时的 `runs/I-003/model5/` 位置迁入当前封存根；artifact
内部旧绝对路径保留为不可变执行溯源。
其中 `g2r2_run_report.md`、`g2r2_metrics.json`、`g2r2_decision.json`、
`g2r2_predictor_quality.json`、`reuse_identity.json`、`commands.txt` 和
`result_validation.json` 分别记录结果、统计、决策、预测器诊断、来源 SHA、执行命令
与终验。运行产物、数据缓存、checkpoint 和原始日志不进入本 Git 仓库。

复核命令：

```bash
/data/miniconda3/envs/lightwam-libero-eval/bin/python -m unittest analysis.lrd_wam.tests.test_g2r2
/data/miniconda3/envs/lightwam-libero-eval/bin/python -m analysis.lrd_wam.g2r2 --stage validate --device cpu
```
