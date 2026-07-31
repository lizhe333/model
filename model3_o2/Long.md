# Model3 O2 Long Result

Model3 O2 was tested on LIBERO Long as a portability treatment for the
layer-aware `q1/q2/q3` readout. The architecture, 16-layer Action-DiT,
flow-matching action objective, future-video loss, H8/R8 deployment contract,
and solver-10 inference path were unchanged from the registered O2 method.

## Training Contract

- Parent: validated Model3 Long step 80K.
- Parent checkpoint SHA-256:
  `65680089b942e1e01b30cf51f707079bd0404956c63a166737e10b1984971d68`.
- Initialization: strict model-only warm start with exact-q3 identity readout.
- State: fresh optimizer, scheduler, dataloader, sampler, and RNG.
- Hardware/profile: four RTX 4090 GPUs, B16/GA1, global batch 64, BF16.
- O2-local budget: 10,000 optimizer steps; checkpoints at 5K and 10K.
- Data: cached no-noops LIBERO Long, two 224 x 224 camera views.

## Evaluation Contract

Each checkpoint was evaluated independently on all 10 Long tasks with 50
official initial states per task, for 500 episodes per condition. Both used
seed 42, action horizon/replan 8/8, solver 10, and the 700-step Long limit.
The highest success count was selected; an exact tie would select 5K.

| O2-local checkpoint | SHA-256 | Successes | Status |
|---:|---|---:|---|
| 5K | `43d3fdb2220826fece236cab3a88c1c4926b2af887aa27203c4c18f3f606a86f` | 436/500 (87.2%) | validated candidate |
| 10K | `9653d5c5a2a151bdf2d5ad18f659686139a590f0dd39b57dd88937ba41f8375f` | 476/500 (95.2%) | validated, selected |

Step 10K is `best_observed_on_predeclared_checkpoint_set`, not an untouched
final test. Both terminal validators passed checkpoint path/SHA/method/class,
10-task and 500-episode coverage, complete disjoint outcomes, summary
recomputation, empty failed-worker ledgers, and exact sparse-video retention.

## Fixed Parent Comparison

The selected O2 step-10K result was paired against the fixed validated Model3
Long step-80K result of 478/500 (95.6%) on the same 500 task/trial identities.

| Matched outcome | Episodes |
|---|---:|
| Both succeed | 459 |
| O2 only | 17 |
| Model3 only | 19 |
| Both fail | 5 |

The exact two-sided McNemar p-value is `0.8679394004284404`. O2 is two
successes lower overall and is not significantly different from its parent in
this matched evaluation.

## Interpretation

- The 5K checkpoint drops substantially, so the fresh O2 Long optimization
  needed the full declared 10K budget to recover parent-level behavior.
- Step 10K preserves strong Long performance but does not establish a Long
  improvement.
- This portability result does not by itself prove that explicit layer identity
  caused the separate Object improvement.
- Training beyond O2-local 10K is not supported by this result.
- The shared-GPU evaluation schedule is valid for closed-loop success but not
  for latency or throughput claims.

The full retained evidence lives in the parent research workspace under
`runs/I-003/model3_o2/2026-07-31_model3_o2_long_5k_10k_eval500/`; large
checkpoints, rollout videos, and raw logs are intentionally not included in
this source mirror.
