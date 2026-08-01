# Model5

Model5 is a self-contained Model3 successor for an asymmetric tri-timestep B1
comparison. It keeps Model3's video/action flow schedulers and Action-DiT, but
changes the hidden states read by its 64 recurrent action queries.

## Structure

```text
model5/
├── models/           # Model5WAM and VLA-query Action-DiT
├── configs/          # immutable experiment and Hydra manifests
├── scripts/          # user-facing launch commands
├── third_party/
│   └── light_wam/    # clean Light-WAM b2785f66 source snapshot
├── config.py         # typed config loading
├── contracts.py      # scientific/runtime validation
├── runtime.py        # model5 Hydra factory
└── launch.py         # validated launcher and run artifacts
```

Model5 has no runtime source dependency on Model3, Model4, or the outer
`Light-WAM/` checkout. Its infrastructure is a complete copy of
Model3's clean upstream Light-WAM snapshot at
`b2785f66e13fd9987e94ae1ecc1c441d5059c9ae`. Large pretrained weights,
datasets, and precomputed caches remain external experiment assets.

## Architecture

- Wan2.1-T2V-1.3B with frozen base weights;
- rank-64 LoRA over all 30 Wan blocks;
- adapters and real hidden states from layers 8, 16, and 24;
- fixed feature timestep `tau_f=1000`;
- a high-resolution action-feature latent grid containing one clean current
  slot and a configurable positive number of Gaussian-noise future slots;
- factor-2 downsampling only in the unchanged future-video supervision branch;
- dual-view 224 x 224 observations;
- proper future-video flow supervision;
- one recurrent bank of 64 learned action queries;
- an 8-step action chunk, matching Model3;
- a 16-layer action DiT that cross-attends the full query memory and is trained
  with flow matching;
- no StateFusion conditioner.

Training keeps two independent Wan passes: the unchanged Model3 future-video
flow pass at sampled `tau_v`, and the action-feature pass at fixed `tau_f`.
Expert future latents never enter the action-feature pass. Inference constructs
the same current-plus-noise feature grid, runs Wan once, freezes the resulting
query memory, and uses the existing Model3 action-flow sampler.

The action-feature pass sends an explicit temporal timestep tensor shaped
`[B,T]` into Wan `pre_dit`. For the current Object profile this is `[0,1000]`:
the clean current slot is conditioned at zero and the Gaussian future slot at
`tau_f=1000`. Wan expands those temporal values over the corresponding spatial
tokens before building `t/t_mod`; runtime diagnostics report the tensor returned
by `pre_dit` and fail closed if explicit separated-timestep conditioning is not
active.

The default config is the high-resolution treatment
`current_plus_noisy_future`. The matched high-resolution control is
`model5/configs/libero_spatial_current_only.json`.

The future-slot count is an explicit experiment/config value and part of the
strict checkpoint identity. The current Object and Long profiles both use one;
historical eight-slot Spatial/Long artifacts are incompatible with them.

## Validate

```bash
python3 -m model5.launch \
  --config model5/configs/libero_spatial.json \
  --run-id dry-run \
  --dry-run
```

## Train Spatial

```bash
bash model5/scripts/train_spatial.sh
```

The first formal Spatial treatment run was user-authorized on 2026-07-26. Its
active profile is B8/GA2 with Wan block gradient checkpointing, preserving
effective global batch 64 after B16/GA1 and non-checkpointed B8/GA2 exceeded
48 GiB cards during bring-up.

## Train Low-Resolution Efficiency Profile

```bash
bash model5/scripts/train_spatial_lowres_efficiency.sh
```

This is a separately named factor-2 action-feature diagnostic. It keeps all
nine temporal slots and the tri-timestep/query/loss contract, but it is not the
primary high-resolution Model3 comparison. The selected runtime profile is
B8/GA2 without Wan gradient checkpointing, preserving effective global batch
64.

The launcher writes model5-owned metadata under
`runs/I-003/model5/<run-id>/` and backend checkpoints under
`runs/I-003/model5/backend_runs/<run-id>/`.

## Train Long

```bash
bash model5/scripts/train_long.sh
```

The formal Long treatment uses `libero_10`, the complete shared dual-camera
latent cache, high-resolution current-plus-one-noisy-future action features,
four GPUs, B8/GA2, no Wan gradient checkpointing, and a 150,000-step budget.
It starts fresh from the Wan base and does not resume Model3 weights or
historical eight-slot Model5 states.

## Train Object (1 Noisy Slot)

```bash
bash model5/scripts/train_object.sh
```

The formal Object profile uses the complete dual-camera Object cache, GPUs
0-3, high-resolution current-plus-noisy-future features with exactly one noisy
future slot, B8/GA2, no Wan gradient checkpointing, and 150,000 optimizer steps.
It starts fresh from the Wan base; historical eight-slot checkpoints are not
compatible with this profile, nor is the superseded two-slot Object run.

## Evaluate Object Checkpoints

```bash
conda run --no-capture-output -n lightwam-libero-eval \
  python -m model5.scripts.eval_object_two_stage \
  --train-run runs/I-003/model5/backend_runs/<object-run-id> \
  --run-root runs/I-003/model5/<evaluation-run-id>
```

The evaluator requires complete 10K, 15K, and 20K checkpoints. It runs all
three at solver 10 and solver 5; solver 10 runs first only to schedule shared
GPU resources. Both are retained as matched per-checkpoint evaluation results.
Each condition is 10 Object tasks by 50 episodes.

## Validated Object Sweep

The one-slot Object sweep is complete. Its action-feature grid is one clean
current latent plus one policy-owned Gaussian-noise future latent, with Wan
temporal timesteps `[0,1000]`; training used B8/GA2 without Wan gradient
checkpointing. Each of the three checkpoints was evaluated for 500 matched
episodes at both action solvers:

| Checkpoint | Solver 10 | Solver 5 |
|---|---:|---:|
| 10K | 400/500 (80.0%) | 448/500 (89.6%) |
| 15K | 466/500 (93.2%) | 478/500 (95.6%) |
| 20K | 459/500 (91.8%) | 454/500 (90.8%) |

Solver 10 ran before solver 5 only to allocate the shared GPUs. Both columns
are formal matched results, not a primary/diagnostic split. Step 15K is the
best observed checkpoint under both solver settings. The full protocol, paired
outcomes, and evidence boundary are recorded in [Object.md](Object.md).

## Progress Logs

- training: `runs/I-003/model5/backend_runs/<run-id>/logs/training.log`;
- single-GPU smoke while active: `/tmp/<smoke-run-id>/logs/smoke.log`;
- evaluation summary: `<eval-run>/logs/evaluation.log`;
- per-task evaluation: `<eval-run>/solver<steps>_step_<step>/logs/task_logs/libero_object_task<N>.log`.

Launch scripts stream the same output to the terminal and the `.log` file.
Smoke logs are temporary and are removed after their terminal result is copied
into the dated handoff; formal training and evaluation logs remain with their
run evidence.
