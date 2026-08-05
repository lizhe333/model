#!/usr/bin/env bash
set -euo pipefail

ROOT=${MODEL3_O2_ROOT:-/data/users/lizhe/I-003_fast_video_dit_to_wam}
TRAIN_SESSION=${MODEL3_O2_TRAIN_SESSION:-model3-o2-spatial}
TRAIN_RUN=${MODEL3_O2_TRAIN_RUN:-${ROOT}/runs/I-003/model3_o2/backend_runs/2026-07-31_model3_o2_spatial_from_model3_60k_b16_ga1_10k}
EVAL_RUN=${MODEL3_O2_EVAL_RUN:-${ROOT}/runs/I-003/model3_o2/2026-07-31_model3_o2_spatial_5k_10k_eval500}
PYTHON_BIN=${MODEL3_O2_PYTHON_BIN:-/data/miniconda3/envs/lightwam-libero-eval/bin/python}
FINAL_CHECKPOINT=${TRAIN_RUN}/checkpoints/weights/step_010000.pt

echo "[watch] waiting for exact training session ${TRAIN_SESSION}"
while tmux has-session -t "=${TRAIN_SESSION}" 2>/dev/null; do
  sleep 20
done

if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
  echo "[watch] training ended without complete step-10K checkpoint: ${FINAL_CHECKPOINT}" >&2
  exit 2
fi
if [[ -e "${EVAL_RUN}" ]]; then
  echo "[watch] refusing to overwrite existing evaluation root: ${EVAL_RUN}" >&2
  exit 3
fi

cd "${ROOT}"
echo "[watch] training complete; launching strict O2 Spatial 5K/10K evaluation"
exec "${PYTHON_BIN}" -m model3_o2.scripts.eval_spatial_two_checkpoint \
  --train-run "${TRAIN_RUN}" \
  --run-root "${EVAL_RUN}"
