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

/** { gpu, adapter, device }.
 *   disableSubgroups: true   do not request subgroups (Gridwise then emulates them)
 *   features: [...]          more required features
 *   limits: {...}            required limits, overriding the defaults
 *   flags: [...]             more create() flags (e.g. "backend=vulkan")
 *   powerPreference          passed to requestAdapter */
export async function createNodeDevice({ disableSubgroups = false, features = [], limits = {}, flags = [], powerPreference } = {}) {
  const gpu = nodeGPU(flags);
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
