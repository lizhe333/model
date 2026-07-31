# Model5 Goal

Starting from the validated Model3 implementation, test whether recurrent
action queries benefit from Wan hidden states over a Model3-matched
high-resolution clean current latent plus policy-owned noisy future slots at a
fixed diffusion timestep.

The implementation gate is complete when the self-contained Model5 code passes
its tensor, leakage, gradient, checkpoint, and real one-step smoke contracts in
`specs/08-model5-asymmetric-tri-timestep-wam.md`. Formal training and benchmark
acceptance are deferred until GPUs 0-3 are explicitly released for Model5.

The implementation is self-contained under `model5/`. Its infrastructure copy
comes from Model3's clean Light-WAM commit `b2785f66` snapshot; it does not
runtime-import Model3, Model4, or either outer Light-WAM source tree. Model5
owns the fixed-timestep feature grid, recurrent query policy, checkpoint
identity, launch configuration, and evidence boundary.
