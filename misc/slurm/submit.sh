#!/bin/bash
# Run a Gridwise command on GPU nodes through Slurm, from the login node.
#
#   misc/slurm/submit.sh [-r REF] [-n NAME] [-t TIME] NODE:GPU[,NODE:GPU...] 'COMMAND'
#
#   misc/slurm/submit.sh -r origin/onesweep-probe -n probe daisy:v100,wario:a100 \
#     'node misc/onesweep_probe.mjs 24 u32 && node misc/onesweep_probe.mjs 24 u64kv'
#
# The job script is this checkout's misc/slurm/job.sh (sbatch copies it at
# submission), so fixes to it apply to runs of any commit.
# REF (default origin/main; a branch name alone means origin's) is fetched and checked out, once per commit, as a
# git worktree in $GRIDWISE_RUNS/<commit> (default ~/gridwise-runs), where
# `npm ci` installs the dependencies; each job runs COMMAND there, so jobs
# that wait in the queue run the commit they were submitted with, whatever
# happens to this checkout meanwhile. One job per NODE:GPU, each on one GPU
# of that type (sinfo -N -o "%N %P %G" lists them; each node here is its
# own partition). Output: $GRIDWISE_RESULTS (default ~/gridwise-results)/
# NAME-NODE-GPU-JOBID.out. Both directories must be visible from the GPU
# nodes (a shared home directory is). Run this from any clone of gridwise
# on the login node; it needs the conda environment of misc/slurm/job.sh.
set -euo pipefail

ref=origin/main name=job time=01:00:00
while getopts "r:n:t:" opt; do
  case $opt in
    r) ref=$OPTARG ;;
    n) name=$OPTARG ;;
    t) time=$OPTARG ;;
    *) sed -n '2,20p' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
if [ $# -ne 2 ]; then sed -n '2,20p' "$0"; exit 1; fi
gpus=$1 command=$2

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
runs=${GRIDWISE_RUNS:-$HOME/gridwise-runs}
results=${GRIDWISE_RESULTS:-$HOME/gridwise-results}
mkdir -p "$runs" "$results"

git -C "$repo" fetch -q origin
sha=$(git -C "$repo" rev-parse -q --verify "origin/$ref^{commit}" || git -C "$repo" rev-parse --verify "$ref^{commit}")
dir=$runs/${sha:0:12}
if [ ! -f "$dir/.gridwise-ready" ]; then
  [ -d "$dir" ] || git -C "$repo" worktree add -q --detach "$dir" "$sha"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${GRIDWISE_CONDA_ENV:-node22}"
  (cd "$dir" && PUPPETEER_SKIP_DOWNLOAD=1 npm ci --no-audit --no-fund --loglevel=error)
  touch "$dir/.gridwise-ready"
fi
echo "commit ${sha:0:12} ($ref) in $dir"

for spec in ${gpus//,/ }; do
  node=${spec%%:*} gpu=${spec#*:}
  out="$results/$name-$node-$gpu-%j.out"
  id=$(sbatch --parsable -p "$node" -w "$node" --gres="gpu:$gpu:1" -t "$time" \
    --job-name="gw-$name" --chdir="$dir" --output="$out" \
    --export=ALL,GRIDWISE_ADAPTER="$gpu",GRIDWISE_COMMIT="${sha:0:12}" \
    "$(dirname "$0")/job.sh" "$command")
  echo "job $id: $node $gpu -> ${out//%j/$id}"
done
echo "watch with: squeue -u ${USER:-$(whoami)}"
