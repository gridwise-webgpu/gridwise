#!/bin/bash -l
# One Slurm job of misc/slurm/submit.sh: runs its command (one argument, run
# by bash) in the worktree submit.sh prepared, after recording where it ran.
# GRIDWISE_ADAPTER (the GPU type requested) makes Dawn pick that GPU, as
# Vulkan can see every GPU on a node whatever Slurm allocated.
set -euo pipefail
if ! command -v conda >/dev/null; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
  done
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${GRIDWISE_CONDA_ENV:-node22}"
echo "== $(hostname -s), Slurm job ${SLURM_JOB_ID:-?}, gridwise ${GRIDWISE_COMMIT:-?}, GPU ${GRIDWISE_ADAPTER:-any}, node $(node --version), $(date -Is)"
nvidia-smi -L 2>/dev/null || true
node misc/list_adapters.mjs
echo "== $1"
bash -c "$1"
echo "== done $(date -Is)"
