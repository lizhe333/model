# Vendored RoboTwin

This directory vendors code from the upstream RoboTwin repository:

- Upstream project: https://github.com/RoboTwin-Platform/RoboTwin
- Upstream commit: `bf44be51cf5717a5595ce59447f2cf5263d2aa95`
- Upstream license: MIT License

License compliance notes:

- The original upstream license is preserved in [`LICENSE`](./LICENSE).
- Files copied from RoboTwin remain subject to the MIT License in this directory.
- For this project, the Light-WAM-specific RoboTwin policy implementation is maintained in `experiments/robotwin/lightwam_policy/`.

Local modifications:

- RoboTwin is vendored under `third_party/RoboTwin` for easier integration with this project.
- This vendored copy is adapted from the RoboTwin integration previously maintained in Fast-WAM.
