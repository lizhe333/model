# Side-Model3-Adapter-v2 Direct-Copy Lineage

`side_model3_adapter_v2/` was initialized by directly copying the complete
`side_model3_adapter/` package on 2026-08-08. It is intentionally a copied
model family, not a refactor that imports and subclasses the parent at runtime.

## Directly copied implementation

These modules retain the Side-Model3 architecture and implementation:

| Adapter package file | Direct source |
|---|---|
| `models/action_dit.py` | `side_model3/models/action_dit.py` |
| `models/latent_transition.py` | `side_model3/models/latent_transition.py` |
| `models/future_latent_change_head.py` | `side_model3/models/future_latent_change_head.py` |
| `models/ladder_side_encoder.py` | `side_model3/models/ladder_side_encoder.py` |
| `models/visual_anchor_resampler.py` | `side_model3/models/visual_anchor_resampler.py` |
| `models/ema_target.py` | `side_model3/models/ema_target.py` |
| `models/side_model3_adapter_v2_wam.py` | `side_model3_adapter/models/side_model3_adapter_wam.py` |
| `config.py`, `contracts.py`, `runtime.py`, `trainer.py`, `launch.py` | corresponding `side_model3_adapter/` files |
| `data.py`, `backend_train.py` | corresponding `side_model3_adapter/` files |
| `scripts/` | corresponding `side_model3_adapter/scripts/` files |
| `tests/` | corresponding `side_model3_adapter/tests/` files |

## Intentional differences

- independent package/class/method/Hydra/checkpoint identities;
- predicted clean-action future-state auxiliary through functional branch-frozen
  Action Chunk Encoder and Transition Predictor;
- v2-local lambda warmup and checkpointed optimizer-step count;
- v2-specific gradient, reduction, warmup, and checkpoint tests;
- the same pre-Wan independent-observation VAE cache and v1 inference boundary.

No v1 component was replaced with a new decoder, new state encoder, new horizon,
new data contract, or inference-time world-model call.
