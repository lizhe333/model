# D1 Current-Only Auxiliary Closed-Loop Pilot

## 2026-08-03 判断修订

现有证据已经足以判定：原版 future-field LRD-WAM 的核心方法主张尚未成立；但这些离线
结果还不足以判定相关表示在闭环中一定没有价值。

原方法路径继续停止，不再测试：

```text
current + noisy future slot
-> low-rank residual
-> Action-DiT
```

唯一值得进行的小型闭环候选改为：

```text
current-only Wan features
-> D1 predicted auxiliary representation
-> Action-DiT
-> closed-loop action
```

这个候选不复活原版 future-field LRD-WAM，也不再主张 output-space endpoint
rank-$8$ residual 是 residual-specific 的核心控制机制。历史 `D1-LR-P3` 名称只用于
定位已有 predictor 权重与来源；在新实验中，其输出被解释为 current-only predicted
auxiliary representation。

## 科学问题

> 离线发现的 current-only、逐样本辅助控制信息，能否转化为闭环收益？

正结果最多支持一个新的 current-only auxiliary control candidate，不能支持 noisy
future slot、future-field、全局低秩机器人子空间或适配效率主张。

## 最小实验边界

- 只使用 current observation、language 与 current proprioception；Wan temporal grid
  固定为 $[0]$。
- 在线路径不得构造或读取 noisy future slot、true future、future latent、future action
  或 endpoint residual target。
- 先通过 live/current-only feature equivalence、checkpoint strict-load、H8 action
  alignment、normalization 与 simulator step 等硬合法性检查。
- Object 最小试验比较 current-only base control 与 D1 auxiliary treatment，使用相同
  Action-DiT、初始化、训练预算、solver 与配对 rollout identities。
- 最小规模为每个 arm $10$ tasks $\times$ $5$ initial states，即 $50$ paired episodes。
- 只有 treatment 比 control 至少多成功 $5/50$、无单任务至少 $3/5$ 的明显回退且全部
  合法性检查通过，才进入新的多 seed、$500$-episode 合同。

完整执行合同位于根仓库
`specs/20-d1-current-only-auxiliary-closed-loop.md`。本页只记录公开研究判断，不授权
回写历史 LRD-WAM artifact、复用已读取 G2 test 或启动 D2/future-field 扩展。
