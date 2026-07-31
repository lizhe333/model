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

The initial Object run was safely paused after logging O2 step 35,260. Complete
10K, 20K, and 35K checkpoints and distributed states are retained. The
two-stage closed-loop evaluation is complete and terminally validated. At
solver 10, steps 10K/20K/35K achieved 442/464/492 successes out of 500, so 35K
is the best observed checkpoint on the predeclared set. The selected 35K
checkpoint achieved 489/500 at solver 5. The matched solver comparison contains
485 both-success, 7 solver-10-only, 4 solver-5-only, and 4 both-fail episodes
(`p=0.548828125`). Solver 5 remains diagnostic-only.

## Long Performance Run

The registered Long treatment keeps the current identity initialization and
model-only warm-starts from the validated Model3 Long step-80K checkpoint. It
uses the validated cached Long data profile, a fresh optimizer/scheduler, and a
10K O2-local budget with checkpoints at 5K and 10K. Object optimizer,
scheduler, and dataloader state are not reused.

The formal run completed its declared 10K budget. Terminally validated
solver-10 evaluations achieved 436/500 at step 5K and 476/500 at step 10K, so
10K is selected on the predeclared set. Versus fixed Model3 Long-80K at
478/500, matched outcomes are 459 both-success, 17 O2-only, 19 Model3-only, and
5 both-fail (`p=0.8679394004284404`). The selected O2 result preserves but does
not improve parent Long performance. See [Long.md](Long.md) for the full result
contract and evidence qualifications.

## Spatial Performance Run

The registered Spatial treatment model-only warm-started from the retained
Model3 Spatial step-60K checkpoint and completed a fresh 10K O2-local budget.
Terminally validated solver-10 evaluations achieved 481/500 at step 5K and
489/500 at step 10K, so 10K is selected on the predeclared set.

The historical fixed Model3 Spatial-60K result is 488/500. The selected O2
result is descriptively one success higher, but the parent evaluation ledger
was deleted and cannot support a paired test. The result therefore preserves
parent-level Spatial performance and does not establish an improvement. See
[Spatial.md](Spatial.md) for the full contract and evidence qualifications.

```bash
python3 -m model3_o2.launch \
  --config model3_o2/configs/libero_long_fast.json \
  --run-id dry-run \
  --dry-run
```

Validate without launching training:

```bash
python3 -m model3_o2.launch \
  --config model3_o2/configs/libero_object.json \
  --run-id dry-run \
  --dry-run
```
