# Side-Model3 Direct-Copy Lineage

`side_model3_adapter/` was initialized by directly copying the complete
`side_model3/` package on 2026-08-06. It is intentionally a copied model family,
not a refactor that imports and subclasses the parent at runtime.

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
| `models/side_model3_adapter_wam.py` | `side_model3/models/side_model3_wam.py` |
| `config.py`, `contracts.py`, `runtime.py`, `trainer.py`, `launch.py` | corresponding `side_model3/` files |
| `tests/` | corresponding `side_model3/tests/` files |

## Intentional differences

- independent package/class/method/Hydra/checkpoint identities;
- vendored residual adapters enabled only after Wan blocks $8/16/24$;
- online Side/Visual input gradients retained to those adapters;
- FP32 EMA target adapter bank added for future targets;
- runtime and contract checks changed from “no adapter” to the fixed
  $8/16/24$, dim-$256$, scale-$1.0$ adapter contract;
- adapter-specific copy-equivalence, gradient, EMA, checkpoint, and inference
  tests added.

No Side-Model3 component was replaced with a new decoder, new state encoder,
new loss, new horizon, or new data contract.
