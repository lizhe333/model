# Model5 Goal

Starting from the validated Model3 implementation, test whether recurrent
action queries benefit from Wan hidden states over a Model3-matched
high-resolution clean current latent plus policy-owned noisy future slots at a
fixed diffusion timestep.

The implementation gate and the first formal Object sweep are complete. The
one-slot Object 10K/15K/20K checkpoints were terminally validated at both
solver 10 and solver 5; 15K is best observed under both. The fresh one-slot
Long B8/GA2 no-GC treatment is now training from the Wan base. Long has no
closed-loop result yet.

The implementation is self-contained under `model5/`. Its infrastructure copy
comes from Model3's clean Light-WAM commit `b2785f66` snapshot; it does not
runtime-import Model3, Model4, or either outer Light-WAM source tree. Model5
owns the fixed-timestep feature grid, recurrent query policy, checkpoint
identity, launch configuration, and evidence boundary.
