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
# Environment:
#   ROCJITSU          the rocjitsu launcher (default: rocjitsu on PATH)
#   ROCJITSU_CONFIGS  directory of rocjitsu GPU configs
#   RJ_THREADS        CPU threads for the simulator (default: nproc)
# The default command lists the adapter, then runs the small regression tests
# for correctness only (CORRECTNESS_ONLY=true SKIP_LARGE=true).
set -euo pipefail

arch=${1:?usage: $0 <gfx1100|gfx1201> [command ...]}
shift
case "$arch" in
  gfx1100) config=gfx1100_w7900.json ;;
  gfx1201) config=gfx1201_r9700.json ;;
  *) echo "$0: unsupported target '$arch'" >&2; exit 2 ;;
esac
rocjitsu=${ROCJITSU:-$(command -v rocjitsu)}
config="${ROCJITSU_CONFIGS:?set ROCJITSU_CONFIGS to the rocjitsu configs directory}/$config"
[[ -f $config ]] || { echo "$0: missing config $config" >&2; exit 2; }

icd=""
for f in /usr/share/vulkan/icd.d/radeon_icd.x86_64.json /usr/share/vulkan/icd.d/radeon_icd.json; do
  [[ -f $f ]] && { icd=$f; break; }
done
[[ -n $icd ]] || { echo "$0: RADV not found; install mesa-vulkan-drivers" >&2; exit 2; }

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if [[ $# -eq 0 ]]; then
  set -- bash -c 'node misc/list_adapters.mjs && CORRECTNESS_ONLY=true SKIP_LARGE=true node examples/regression_node.mjs'
fi

# Use only RADV, with the system Mesa and libdrm, as rocjitsu's own Vulkan
# tests do; nothing inherited may redirect Vulkan to another device.
exec env -u LD_LIBRARY_PATH -u LD_PRELOAD -u VK_ICD_FILENAMES -u VK_ADD_DRIVER_FILES \
  -u VK_INSTANCE_LAYERS -u VK_LOADER_DRIVERS_SELECT -u VK_LOADER_DRIVERS_DISABLE \
  -u RADV_FORCE_FAMILY \
  VK_DRIVER_FILES="$icd" DRI_PRIME= \
  "$rocjitsu" --config "$config" --cpu-thread-budget "${RJ_THREADS:-$(nproc)}" -- "$@"
