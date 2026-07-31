# Model3 Regression

This track is the matched direct-action-regression treatment for Model3. It
reuses Model3's current-observation Wan path and recurrent 64-query encoder,
then replaces the 16-layer action-flow DiT and iterative solver with a two-layer
direct decoder trained by masked L1 on the normalized 8 x 7 action chunk.

The track imports the parent Model3 package and its vendored Light-WAM
infrastructure. It owns a distinct runtime target, method id, checkpoint
identity, configuration, evidence root, and backend output root.

## Validate

```bash
python3 -m model3_regression.launch \
  --config model3_regression/configs/libero_object.json \
  --run-id dry-run \
  --dry-run
```

## Train Object

```bash
bash model3_regression/scripts/train_object.sh
```

The formal Object profile starts fresh for 150,000 optimizer steps on GPUs
0-3. It must not resume a Model3 flow checkpoint.

## Train Long

```bash
bash model3_regression/scripts/train_long.sh
```

The matched cached Long control starts fresh for 80,000 optimizer steps on
GPUs 0-3. It reuses the registered Model3 Long latent cache but owns separate
Regression checkpoints and evidence paths. It must not resume Object or Model3
flow weights.
