# Model3 Goal

Use a pretrained Wan2.1 Video-DiT to obtain a strong WAM on four RTX 4090
48 GB GPUs without full backbone fine-tuning, using a VLA-Adapter-inspired
action-query interface.

Model3 is accepted when a fresh model3-owned checkpoint reaches at least
450/500 valid successes on LIBERO Spatial while satisfying the future-video
flow, action-flow, online leakage, and efficiency contracts in
`specs/06-model3-future-dynamics-wam.md`.

The implementation is self-contained under `model3/`. Its infrastructure copy
comes from clean Light-WAM commit `b2785f66`; it does not import the outer
Light-WAM source tree. Model3 owns the recurrent action-query policy,
flow-matching action objective, strict checkpoint identity, launch
configuration, and evidence boundary. Light-WAM StateFusion remains a
comparison method rather than model3.
