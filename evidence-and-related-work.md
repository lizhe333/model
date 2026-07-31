# Efficient Video-WM-to-WAM Adaptation

## Evidence, Related Work, and Audit Boundary

> 用途：保存 related-work evidence、历史结果、已关闭解释与新文献审计记录。研究叙事见
> [research-proposal.md](research-proposal.md)，执行合同见
> [experiment-contract.md](experiment-contract.md)。

## 1. Evidence Boundary

- 本文主线只研究 **in-place adaptation**：部署时仍运行被适配的 Video-DiT 或 video
  generator computation。
- 历史 Model3、Model3 Regression、Model5 结果是 diagnostic anchors，不是当前 `C*`
  上的 matched controls。
- 本地存在 `model3_o2` 等代码，不等于存在权威 server checkpoint、完整评测与当前终态；
  carrier 只能由 experiment contract 的 G0 冻结。
- 论文没有公开的字段记为“未报告”，不得根据方法名或结构图推断。
- 本文不提出 generator-state distillation、独立 current-only student 或 generator-free
  control 新模块。

## 2. Related-Work Matrix

| 方法 | WM 适配 | Supervision / routing | 在线接口与推理 | 对本文的作用 |
|---|---|---|---|---|
| Light-WAM | frozen Wan base + LoRA/adapters | joint video/action；具体 routing 待代码核对 | selected layers + state fusion；current observation Wan once | 证明轻量适配可行；不是 supervision/routing 的 matched causal study |
| FastWAM | 统一成本仍需核对 | joint video/action flow | Action-DiT 逐层读取 video K/V；video prefill + action denoising | 展示持续 predictive objective 与高带宽接口 |
| DiT4DiT | joint configuration | video/action experts 联合训练 | Video-DiT hidden condition；noisy future grid + action denoising | future slots、接口与 objective 同时变化 |
| DeVA | Video2World DiT + Action Expert 均训练 | warmup + joint | multi-layer/multi-timestep transfer；joint future/action process | 支持重容量方案，但不能单独归因 decoupling |
| VidMan | Stage 2 可更新或冻结 VDT | video pretrain → action-only | layer-wise action adapter；fixed noisy video latents | 支持 staged hypothesis 与 action-gradient 研究 |
| Efficient-WAM / AHA-WAM | compressed/asynchronous future use | 各自效率机制 | 面向训练或推理成本优化 | 系统效率参考，不直接回答当前 schedule/routing |
| Model3 historical | frozen Wan base + all-layer rank-64 LoRA/adapters | joint `L_video + L_action` | recurrent queries over layers 8/16/24；current Wan once + action solver | 历史 upper anchor；新 carrier 上必须重训 |

跨论文共同缺口不是“Video-DiT 能否做 WAM”，而是：在固定母体、action carrier、数据与
评测后，future supervision 应何时使用、video/action gradients 应进入哪里，以及保持收益
需要多少 adaptation/interface capacity。

## 3. EnFold Scope and Novelty Boundary

EnFold 的结构是：

```text
training:
  teacher-forced real future
  -> Cosmos generator multi-level states
  -> timestep-conditioned target for current-only DINO encoder
  -> detached action readout

deployment:
  current-only DINO encoder
  -> action head
  -> no generator execution
```

它覆盖了此前候选中的以下 method space：

- selected multi-level generator states；
- timestep-conditioned state prediction；
- stop-gradient teacher/readout contract；
- current-only predictive representation；
- generator-free control。

因此，本文不能把这些机制重新包装为新方法。另一方面，EnFold 同时改变 online backbone、
predictive target、gradient contract、action representation 与 deployment path，与本文的
in-place PEFT treatments 不是同一组可交换变量。

主 Proposal 只保留一句 scope boundary；本文档保留完整边界。EnFold 不进入 Matrix C，
不设置正式 Matrix E/Gate。未来如做 matched system comparison，应作为独立扩展并重新申请
实验资源。

## 4. Historical Local Evidence

| 结果 | 数值 | 证据边界 |
|---|---:|---|
| Model3 Long step 80K | 478/500，95.6% | 历史 flow-carrier 正式结果 |
| Released Light-WAM Long | 461/500，92.2% | 本地发布权重复测 |
| Model3 Object flow-10 | 440/500，88.0% | 历史固定配置 |
| Model3 Object flow-5 | 467/500，93.4% | post-hoc solver diagnostic |
| Model3 Regression Object step 20K | 467/500，93.4% | predeclared checkpoint set 中 best observed |
| Released Light-WAM Object | 497/500，99.4% | 本地发布权重复测 |
| Model3 plan-call latency | 232.994 ms | 历史受控测试 |
| Light-WAM plan-call latency | 70.327 ms | 历史受控测试 |

这些结果说明闭环性能与部署成本之间存在值得研究的张力，但不能替代 `C*` 上重新训练的
A/R/B controls。特别是 flow 与 regression 来自不同训练路径，其差异不能归因于
decoder-only treatment。

## 5. External Evidence Tension

### 5.1 DeVA

DeVA paper-reported RoboCasa ablation：

| Variant | Success |
|---|---:|
| Action only | 19.8% |
| Goal-image prediction | 25.8% |
| Future video + unified backbone | 36.8% |
| Future video + decoupled multi-level transfer | 66.0% |
| + affordance/depth guidance | 72.0% |

这些结果同时改变 expert、interface 与 capacity，支持 future modeling 的价值，但不能单独
证明 decoupling 或某条 gradient route 的因果作用。

### 5.2 VidMan

VidMan paper-reported CALVIN ablation：

| Variant | Avg. Len. |
|---|---:|
| Stage 2 `L_video + L_action` | 2.70 |
| Stage 2 action-only | 3.42 |
| Frozen VDT + adapter/head | 2.98 |
| Action loss 更新 VDT + adapter/head | 3.42 |

DeVA 与 VidMan 共同构成 Matrix A/R 的研究动机：future-video objective 可能帮助 dynamics
acquisition，但持续 joint gradient 也可能干扰 control specialization。两篇论文的
backbone、数据、action interface、compute 与 schedule 并不匹配，因此该推断仍需同一
carrier 内的受控实验验证。

## 6. Closed Interpretations

以下解释不再作为当前 Proposal 的扩展方向：

- cadence 不是历史 Model3/Light-WAM Object gap 的主因；
- Light-WAM 与 Model3 的 direct action prefix 差异不能继续作为主要解释；
- frozen spatial C3/C3-add 缺少 correspondence control，不继续扩展；
- flow/regression 的互补失败集合不足以支持双头 router；
- Regression 与 flow 来自独立训练路径，不能将差异解释为 decoder-only；
- I-003 的旧方法新颖性已被 Light-WAM 覆盖，不因本 Proposal 重新激活。

## 7. New-Paper Intake Template

```text
工作名称 / 版本 / checkpoint：
Video backbone：
训练期 generator 与部署期 encoder 是否相同：
Video objective：
Action objective / decoder：
Predictive target：
训练是 staged、joint 还是独立：
L_video 是否更新 WM / PEFT：
L_action 是否更新 WM / PEFT：
task loss 是否更新 predictive encoder：
接口读取哪些 layers / timesteps / tokens：
推理是否运行 generator / future branch：
trainable params / GPU hours / memory / latency：
最关键消融：
消融是否匹配参数、训练预算和推理协议：
能够支持的结论：
不能支持的结论：
与 RQ1–RQ4 的关系：
证据来源：用户已读笔记 / AI 预读 / 原论文 / 外部检索：
```

## 8. Audit Checklist: Explicitly Out of Scope

- 不把 EnFold 放入主 in-place adaptation matrix；
- 不创建 EnFold-style Matrix E，或重复提出 ASDB/current-only student；
- 不把旧 Model3 flow 结果当作新 carrier 的 A1；
- 不用相同 steps 冒充 compute-matched；
- 不把 checkpoint 当作独立 seed；
- 不把“差异不显著”当作 non-inferiority；
- 不把一个 B1 候选称为绝对 minimum；
- 不让 Matrix C 同时改变层数、aggregation、recurrence 与 capacity；
- 不跳过 PEFT × interface 交互检查；
- 不在 A/R insight gate 前扩展 C/D；
- 不无条件跑满所有 80K/150K 实验；
- 不用 probe、offline loss 或 gradient cosine 代替闭环成功率；
- 不在用户批准前启动新增训练或修改 server project routing。

