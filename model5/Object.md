# Model5 Object: One-Slot Two-Solver Result

## Method And Training Identity

This treatment uses the Model5 temporal action-feature path:

```text
current observation latent + one policy-owned Gaussian future-noise latent
temporal timestep matrix $[0, 1000]$
  -> one Wan feature forward
  -> layers 8/16/24 -> recurrent 64-query memory -> Action-DiT
```

The action-feature branch does not read expert future video. The formal Object
training profile used high-resolution action features, B8/GA2, effective global
batch 64, BF16, and disabled Wan gradient checkpointing. The base Wan weights
remain frozen; Wan LoRA/adapters and the query/action-policy path train.

## Evaluation Protocol

- Suite: LIBERO Object, 10 tasks x 50 trials = 500 episodes per condition.
- Checkpoints: 10K, 15K, and 20K from the same one-slot training run.
- Action horizon/replan: 8/8; maximum episode length: 400.
- Solver settings: 10 and 5 inference steps for every checkpoint.
- Both solver columns passed checkpoint identity, episode coverage, summary,
  rollout-video, and paired-outcome terminal validation.

Solver 10 was executed before solver 5 only to schedule shared GPU capacity. It
does not make solver 10 primary, and solver 5 is not a diagnostic-only result.

## Results

| Checkpoint | Solver 10 | Solver 5 |
|---|---:|---:|
| 10K | 400/500 (80.0%) | 448/500 (89.6%) |
| 15K | 466/500 (93.2%) | 478/500 (95.6%) |
| 20K | 459/500 (91.8%) | 454/500 (90.8%) |

Step 15K is the best observed checkpoint for both tested solver settings. This
does not establish a universal solver choice: the sweep contains one training
seed and only these three checkpoints.

| Checkpoint | Both success | Solver-10 only | Solver-5 only | Both fail | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| 10K | 396 | 4 | 52 | 48 | 1.10e-11 |
| 15K | 461 | 5 | 17 | 17 | 0.0169 |
| 20K | 441 | 18 | 13 | 28 | 0.4731 |

## Evidence Boundary

The retained authoritative evidence is outside this source mirror at
`runs/I-003/model5/20260801_model5_object_10k_15k_20k_solver10_solver5_r1/`.
It contains checkpoint hashes, six condition validators, three paired solver
comparisons, and the run report. Checkpoints, rollout videos, caches, and raw
logs are intentionally excluded from this repository.
