/* WebGPU in Node (the `webgpu` package, Dawn), set up the way Gridwise's
 * scripts need it. Not part of the library: src/ also runs in browsers and
 * must not import `webgpu`.
 *
 *   import { createNodeDevice } from "../misc/node_device.mjs";
 *   const { device, adapter } = await createNodeDevice();
 *
 * What it takes care of:
 * - GPU timestamps at full resolution (by default Dawn rounds them down to
 *   multiples of 65.536 us; see docs/timing-strategy.md).
 * - Keeping the GPU instance referenced: if create()'s result is garbage
 *   collected, Dawn tears the device down under us (seen as "futex facility
 *   returned an unexpected error code" crashes).
 * - Requesting subgroups and timestamp queries when the adapter has them,
 *   and the adapter's largest workgroup memory and buffer sizes (sort needs
 *   32 KB of workgroup memory).
 * - WebGPU's globals (GPUBufferUsage and friends) on globalThis.
 */
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);

/** flags every Gridwise script passes to create() */
export const NODE_GPU_FLAGS = [
  "enable-dawn-features=use_user_defined_labels_in_backend",
  "disable-dawn-features=timestamp_quantization",
];

const keepAlive = [];

/** A GPU instance (navigator.gpu's equivalent), kept referenced for the
 * life of the process. flags are added to NODE_GPU_FLAGS. */
export function nodeGPU(flags = []) {
  const gpu = create([...NODE_GPU_FLAGS, ...flags]);
  keepAlive.push(gpu);
  return gpu;
}

/** create() flags that pick the adapter whose name contains `name` as a
 * word, ignoring case and with "_" for a space ("a100", "v100", "titan_v"):
 * for nodes with several kinds of GPU, where Vulkan sees them all whatever
 * the scheduler allocated. Default: the GRIDWISE_ADAPTER environment
 * variable (set by misc/slurm/job.sh). [] when neither is set. Dawn lists
 * its adapters when asked for one that does not exist; this asks, then
 * names the match exactly. */
export async function adapterFlags(name = globalThis.process?.env?.GRIDWISE_ADAPTER) {
  if (!name) return [];
  let available = [];
  try {
    await nodeGPU(["adapter=\u0001none"]).requestAdapter();
  } catch (e) {
    available = [...String(e).matchAll(/name: '([^']*)'/g)].map((m) => m[1]);
  }
  const word = new RegExp(`(^|[^a-z0-9])${name.toLowerCase().replace(/_/g, " ")}($|[^a-z0-9])`);
  const match = available.find((n) => word.test(n.toLowerCase()));
  if (!match) throw new Error(`no WebGPU adapter matches "${name}"; available: ${available.join("; ") || "none"}`);
  return [`adapter=${match}`];
}

/** { gpu, adapter, device }.
 *   disableSubgroups: true   do not request subgroups (Gridwise then emulates them)
 *   features: [...]          more required features
 *   limits: {...}            required limits, overriding the defaults
 *   flags: [...]             more create() flags (e.g. "backend=vulkan")
 *   powerPreference          passed to requestAdapter
 * GRIDWISE_ADAPTER in the environment picks the adapter by name
 * (adapterFlags). */
export async function createNodeDevice({ disableSubgroups = false, features = [], limits = {}, flags = [], powerPreference } = {}) {
  const gpu = nodeGPU([...flags, ...(await adapterFlags())]);
  const adapter = await gpu.requestAdapter(powerPreference ? { powerPreference } : undefined);
  if (!adapter) throw new Error("createNodeDevice: no WebGPU adapter");
  const wanted = [...(disableSubgroups ? [] : ["subgroups"]), "timestamp-query", ...features];
  const L = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures: wanted.filter((f) => adapter.features.has(f)),
    requiredLimits: {
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
      ...limits,
    },
  });
  keepAlive.push(adapter, device);
  return { gpu, adapter, device };
}
