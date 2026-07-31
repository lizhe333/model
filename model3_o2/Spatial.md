# Model3 O2 Spatial Result

Model3 O2 was tested on LIBERO Spatial as a portability treatment for the
layer-aware `q1/q2/q3` readout. The architecture, 16-layer Action-DiT,
flow-matching action objective, future-video loss, H8/R8 deployment contract,
and solver-10 inference path were unchanged from the registered O2 method.

## Training Contract

- Parent: retained Model3 Spatial step 60K.
- Parent checkpoint SHA-256:
  `67ccb8f4bc3bb25d4474cfbaff6e9d1e98c47edcd88e4341ec2c8744c5b1cf9b`.
- Initialization: strict model-only warm start with exact-q3 identity readout.
- State: fresh optimizer, scheduler, dataloader, sampler, and RNG.
- Hardware/profile: four RTX 4090 GPUs, B16/GA1, global batch 64, BF16.
- O2-local budget: 10,000 optimizer steps; checkpoints at 5K and 10K.
- Data: cached no-noops LIBERO Spatial, two 224 x 224 camera views.

## Evaluation Contract

Each checkpoint was evaluated independently on all 10 Spatial tasks with 50
official initial states per task, for 500 episodes per condition. Both used
seed 42, action horizon/replan 8/8, solver 10, and the 400-step Spatial limit.
The highest success count was selected; an exact tie would select 5K.

| O2-local checkpoint | SHA-256 | Successes | Status |
|---:|---|---:|---|
| 5K | `944c49503639a67b124d46523759f0195c11e2a9ced94b3ac770c90d13ba9f00` | 481/500 (96.2%) | validated candidate |
| 10K | `45e010523c54f22fb42c26b6afa5cfa36abb7c4e43eddbe6d6766798b485415b` | 489/500 (97.8%) | validated, selected |

Step 10K is `best_observed_on_predeclared_checkpoint_set`, not an untouched
final test. Both terminal validators passed checkpoint path/SHA/method/class,
10-task and 500-episode coverage, complete disjoint outcomes, summary
recomputation, empty failed-worker ledgers, and exact sparse-video retention.

## Fixed Parent Comparison

The fixed historical Model3 Spatial step-60K result is 488/500 (97.6%). O2
step 5K is seven successes lower, while selected O2 step 10K is one success
higher:

| Model | Successes | Difference from Model3 |
|---|---:|---:|
| Model3 Spatial 60K | 488/500 (97.6%) | reference |
| Model3 O2 Spatial 5K | 481/500 (96.2%) | -7 (-1.4 pp) |
| Model3 O2 Spatial 10K | 489/500 (97.8%) | +1 (+0.2 pp) |

The Model3 result historically passed strict finalization, and its retained
checkpoint identity was reverified. Its successful evaluation directory and
episode ledger were later deleted, so the aggregate result is
`recorded_not_locally_auditable`. A paired McNemar test or task-stratified
paired confidence interval cannot be reconstructed. The one-success aggregate
difference is therefore descriptive and does not establish superiority.

## Interpretation

- The 5K checkpoint drops below the parent, while 10K recovers parent-level
  Spatial behavior.
- Step 10K preserves strong Spatial performance but does not establish a
  Spatial improvement over Model3.
- The selected result adds evidence that the O2 carrier transfers across
  Object, Long, and Spatial, but does not isolate layer-aware readout as the
  cause of performance.
- Training beyond O2-local 10K is not supported by this result.
- The shared-GPU evaluation schedule is valid for closed-loop success but not
  for latency or throughput claims.

The full retained O2 evidence lives in the parent research workspace under
`runs/I-003/model3_o2/2026-07-31_model3_o2_spatial_5k_10k_eval500/`; large
checkpoints, rollout videos, and raw logs are intentionally not included in
this source mirror.
