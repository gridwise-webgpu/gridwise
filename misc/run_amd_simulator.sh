#!/usr/bin/env bash
# Run Gridwise's correctness tests on a simulated AMD GPU, without a GPU.
#
#   misc/run_amd_simulator.sh <gfx1100|gfx1201> [command ...]
#
# rocjitsu (ROCm/rocm-systems, emulation/rocjitsu) intercepts the amdgpu
# kernel-driver interface and executes the GPU's instructions on the CPU.
# Node's webgpu package (Dawn) runs on Vulkan, here Mesa's RADV driver,
# which talks to the simulated GPU. RDNA3 (gfx1100) and RDNA4 (gfx1201) are
# the simulated GPUs RADV supports. Needs Mesa 25.2.8+ and libdrm 2.4.125+.
#
# rocjitsu comes with AMD's ROCm SDK ("TheRock"; pip-installable, see
# .github/workflows/amd-simulator.yml) or can be built from source.
#
# Environment:
#   ROCM_ROOT         an initialized ROCm SDK (`rocm-sdk path --root`); its
#                     rocjitsu (or mirage) and configs are used by default
#   ROCJITSU          the simulator launcher, for a source build
#   ROCJITSU_CONFIGS  directory of rocjitsu GPU configs, for a source build
#   RJ_THREADS        CPU threads for the simulator (default: nproc)
# The default command checks that Dawn sees the simulated AMD GPU, then runs
# the small regression tests for correctness only.
set -euo pipefail

arch=${1:?usage: $0 <gfx1100|gfx1201> [command ...]}
shift
case "$arch" in
  gfx1100) config=gfx1100_w7900.json ;;
  gfx1201) config=gfx1201_r9700.json ;;
  *) echo "$0: unsupported target '$arch'" >&2; exit 2 ;;
esac
if [[ -z ${ROCJITSU:-} && -n ${ROCM_ROOT:-} ]]; then
  ROCJITSU="$ROCM_ROOT/bin/rocjitsu"
  [[ -x $ROCJITSU ]] || ROCJITSU="$ROCM_ROOT/bin/mirage"
fi
rocjitsu=${ROCJITSU:-$(command -v rocjitsu || true)}
[[ -x $rocjitsu ]] || { echo "$0: no simulator; set ROCM_ROOT or ROCJITSU" >&2; exit 2; }
simulator=("$rocjitsu")
[[ $(basename "$rocjitsu") == mirage ]] && simulator+=(run --in-process)
configs=${ROCJITSU_CONFIGS:-${ROCM_ROOT:+$ROCM_ROOT/share/rocjitsu/configs}}
config="${configs:?set ROCM_ROOT or ROCJITSU_CONFIGS}/$config"
[[ -f $config ]] || { echo "$0: missing config $config" >&2; exit 2; }

icd=""
for f in /usr/share/vulkan/icd.d/radeon_icd.x86_64.json /usr/share/vulkan/icd.d/radeon_icd.json; do
  [[ -f $f ]] && { icd=$f; break; }
done
[[ -n $icd ]] || { echo "$0: RADV not found; install mesa-vulkan-drivers" >&2; exit 2; }

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# -eq 0 ]]; then
  # EXPECT_VENDOR: fail rather than pass on a software or unrelated GPU
  set -- bash -c 'EXPECT_VENDOR=amd node misc/list_adapters.mjs && CORRECTNESS_ONLY=true SKIP_LARGE=true node examples/regression_node.mjs'
fi

# a short private directory for the simulator's sockets
runtime_dir=$(mktemp -d "${TMPDIR:-/tmp}/gridwise-rj.XXXXXX")
trap 'rm -rf -- "$runtime_dir"' EXIT

# Use only RADV, with the system Mesa and libdrm, as rocjitsu's own Vulkan
# tests do; nothing inherited may redirect Vulkan to another device.
env -u LD_LIBRARY_PATH -u LD_PRELOAD -u VK_ADD_DRIVER_FILES \
  -u VK_INSTANCE_LAYERS -u VK_LOADER_DRIVERS_SELECT -u VK_LOADER_DRIVERS_DISABLE \
  -u RADV_FORCE_FAMILY \
  VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd" DRI_PRIME= \
  ROCJITSU_RUNTIME_DIR="$runtime_dir" MIRAGE_RUNTIME="$runtime_dir" \
  "${simulator[@]}" --config "$config" --cpu-thread-budget "${RJ_THREADS:-$(nproc)}" -- "$@"
