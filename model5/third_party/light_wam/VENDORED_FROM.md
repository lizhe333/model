# Vendored Light-WAM Baseline

This directory is a complete copy of Model3's clean vendored Light-WAM tree,
not a copy of the locally modified outer `Light-WAM/` checkout.

- Upstream repository: `https://github.com/L1ziang/Light-WAM.git`
- Upstream commit: `b2785f66e13fd9987e94ae1ecc1c441d5059c9ae`
- Original Model3 copy method: `git archive HEAD`
- Model5 copy method: complete source copy from `model3/third_party/light_wam/`
- Model5 copied on: `2026-07-24`
- Original license: `LICENSE` in this directory

Only namespace changes were applied inside the vendored tree so imports and
Hydra targets resolve through `model5.third_party.light_wam`. Model5 policy,
flow-matching action training, checkpoint identity, and inference logic live in
`model5/` outside this third-party directory.

Datasets, checkpoints, caches, simulator installations, and run artifacts are
not vendored. They are external experiment assets, not source dependencies.
Future upstream synchronization must use an explicit clean copy and review;
do not link this directory to either outer Light-WAM checkout.
