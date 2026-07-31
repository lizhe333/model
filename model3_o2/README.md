# Model3 O2

Model3 O2 is the isolated implementation treatment for hypothesis O2 in the
Model3 hypothesis ledger. It keeps Model3's Wan PEFT path, recurrent queries,
future-video loss, 16-layer Action-DiT, action-flow objective, and 8/8 control
contract. Its only architecture change is an explicit layer-aware readout over
the recurrent `q1/q2/q3` trace before the unchanged Action-DiT.

The readout preserves the full `[B,64,512]` memory. It initializes exactly as
the parent q3-only path and adds independent query-wise residual routes from q1
and q2. The initial Object run is pinned to the retained Model3 step-20K model
checkpoint through a strict model-only warm-start contract.

This track has no training or closed-loop evidence yet and is not an accepted
replacement for Model3.

Validate without launching training:

```bash
python3 -m model3_o2.launch \
  --config model3_o2/configs/libero_object.json \
  --run-id dry-run \
  --dry-run
```
