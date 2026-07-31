# Model3

Model3 is the primary future-dynamics-constrained WAM baseline. It uses a
VLA-Adapter-inspired action-query interface and real Wan flow supervision.

## Structure

```text
model3/
├── models/           # Model3WAM and VLA-query Action-DiT
├── configs/          # immutable experiment and Hydra manifests
├── scripts/          # user-facing launch commands
├── third_party/
│   └── light_wam/    # clean Light-WAM b2785f66 source snapshot
├── config.py         # typed config loading
├── contracts.py      # scientific/runtime validation
├── runtime.py        # model3 Hydra factory
└── launch.py         # validated launcher and run artifacts
```

Model3 has no runtime source dependency on the outer `Light-WAM/` checkout.
The infrastructure source was copied from clean upstream
Light-WAM commit `b2785f66e13fd9987e94ae1ecc1c441d5059c9ae` with its license and
provenance record. Large pretrained weights, datasets, and precomputed caches
remain external experiment assets.

## Architecture

- Wan2.1-T2V-1.3B with frozen base weights;
- rank-64 LoRA over all 30 Wan blocks;
- adapters and real hidden states from layers 8, 16, and 24;
- dual-view 224 x 224 observations;
- proper future-video flow supervision;
- one recurrent bank of 64 learned action queries;
- an 8-step action chunk;
- a 16-layer action DiT that cross-attends the full query memory and is trained
  with flow matching;
- no StateFusion conditioner.

## Validate

```bash
python3 -m model3.launch \
  --config model3/configs/libero_spatial.json \
  --run-id dry-run \
  --dry-run
```

## Train Spatial

```bash
bash model3/scripts/train_spatial.sh
```

The cached Object and Goal suite entrypoints are
`model3/scripts/train_object.sh` and `model3/scripts/train_goal.sh`.
Only one formal suite may be active at a time; each owns GPUs `0,1,2,3`.

The launcher writes model3-owned metadata under
`runs/I-003/model3/<run-id>/` and backend checkpoints under
`runs/I-003/model3/backend_runs/<run-id>/`.
