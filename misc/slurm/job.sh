#!/bin/bash -l
# One Slurm job of misc/slurm/submit.sh: runs its command (one argument, run
# by bash) in the worktree submit.sh prepared, after recording where it ran.
# GRIDWISE_ADAPTER (the GPU type requested) makes Dawn pick that GPU, as
# Vulkan can see every GPU on a node whatever Slurm allocated.
#
# Quieting: on an NVIDIA GPU, Vulkan loads only NVIDIA's driver (so Mesa's
# drivers neither list llvmpipe nor fail to open the GPUs Slurm hid, e.g.
# radv's "Could not open device /dev/dri/renderD129" on wario); and Dawn's
# notices that it lowered NVIDIA's dynamic-buffer limits ("artificially
# reduced ... to fit dynamic offset allocation limit"), harmless and printed
# by every device creation, are dropped from stderr.
set -euo pipefail
if ! command -v conda >/dev/null; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
  done
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${GRIDWISE_CONDA_ENV:-node22}"
echo "== $(hostname -s), Slurm job ${SLURM_JOB_ID:-?} ($(nproc) CPUs), gridwise ${GRIDWISE_COMMIT:-?}, GPU ${GRIDWISE_ADAPTER:-any}, node $(node --version), $(date -Is)"
if nvidia-smi -L 2>/dev/null; then
  for icd in /usr/share/vulkan/icd.d/nvidia_icd*.json /etc/vulkan/icd.d/nvidia_icd*.json; do
    if [ -f "$icd" ]; then
      export VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd"
      break
    fi
  done
fi
quiet() { "$@" 2> >(grep -v "artificially reduced from" >&2); }
quiet node misc/list_adapters.mjs
echo "== $1"
quiet bash -c "$1"
echo "== done $(date -Is)"
