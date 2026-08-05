# Model3 O2 Dynamic

`model3_o2_dynamic/` is the isolated successor treatment that adds
action-response-prewarmed current-only residual adapters before the unchanged
Model3 O2 query interface.

The frozen contract is
`specs/21-model3-o2-dynamic-response-prewarm.md`. Current status:

- Dynamic package identity, model/runtime/trainer, Stage 1 data/teacher/cache,
  and the automatic Stage 1 $→$ Stage 2 launcher are implemented;
- Stage 1 predictor provenance is sealed as `adapter_residual_only`: temporary
  $Q_l$ reads only $A_l(\operatorname{stopgrad}(h_l))$, never $h_l$ or
  $B_l=h_l+A_l(h_l)$;
- focused unit/contract tests pass ($26/26$);
- no simulator collection, teacher cache extraction, Wan smoke, Stage 1 run,
  Stage 2 run, checkpoint, or closed-loop evaluation has been started.

## Shared O2 Lineage

Original Model3 O2 already has two stages:

```text
public Wan
-> Model3 joint training to a suite query-pretrained parent
   (PEFT/query/Action-DiT already trained)
-> add exact-q3 O2 layer-aware gate
-> O2-local joint training
```

The Long-first Treatment reuses the validated Model3 Long step-$80\text{K}$
query-pretrained parent. Dynamic response warmup is inserted after this shared
parent and before O2 gate training; future Object uses the same global
residual-only implementation with its own pinned parent.

## Dynamic Treatment

```text
shared Model3 query-pretrained parent
-> instantiate the same exact-q3 O2 gate and freeze it
-> Stage 1: train only A8/A16/A24 and Q8/Q16/Q24 for 5K steps,
   with Q(A(stopgrad(h)), action)
-> delete Q and load adapter-only state into a clean O2 gate-stage model
-> Stage 2: deploy B=h+A(h), restore original video + action joint training
-> keep the original O2 layer_readout gate active from Stage 2 step 1
-> save step 5K, then unfreeze A8/A16/A24 at 0.1x PEFT LR while the gate
   continues training through step 10000
```

Stage 1 uses one LIBERO/robosuite simulator, $5{,}000$ deterministic
motion-aware source states, four same-state branches per source, $20{,}000$
branch trajectories, and approximately $1.28$M per-camera frames. Deployment
keeps only the three response adapters; the training-only predictors are not
part of the policy checkpoint.

## Execution

The pipeline is deliberately dry by default:

```bash
PYTHONPATH=. conda run --no-capture-output -n lightwam-libero-eval \
  python -m model3_o2_dynamic.pipeline \
  --config model3_o2_dynamic/configs/libero_long_fast.json \
  --run-id <run-id>
```

Only `--execute` starts data preparation or training. It isolates test source
selection, branch collection, carrier/teacher extraction, and held-out
diagnostics until after the fixed Stage 1 step-$5\text{K}$ adapter export.
