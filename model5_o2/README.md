# Model5 O2

Model5 O2 tests one architecture change on a trained Model5 carrier: replace
the final-q3 edge into the Action-DiT with the exact-q3-initialized O2
layer-aware readout. The original Model5 nine-slot temporal grid, joint losses,
gradient routing, PEFT, recurrent queries, and Action-DiT remain unchanged.

## Experiment shape

```text
Stage 1: fresh Model5 Long to 80K

Model5-80K
  |- Stage 2A: q3-only Model5 control, fresh optimizer, local 10K
  `- Stage 2B: exact-q3 Model5 O2, fresh optimizer, local 10K
```

The paused legacy Model5 5K state belongs to the earlier 150K scheduler and is
not the default Stage 1 resume source.

## Validate Stage 1

```bash
python3 -m model5_o2.launch \
  --config model5_o2/configs/libero_long_stage1_model5_80k.json \
  --run-id dry-run-model5-o2-stage1 \
  --dry-run
```

## Prepare Stage 2

After Stage 1 step 80K is complete, generate two configs with the same pinned
parent SHA:

```bash
python3 -m model5_o2.prepare_stage2 \
  --parent /absolute/path/to/checkpoints/weights/step_080000.pt \
  --output-dir runs/I-003/model5_o2/generated_stage2_configs
```

Then dry-run and launch each generated config independently. Formal training is
not started by repository validation.
