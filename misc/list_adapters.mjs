/* Print the WebGPU adapter that Node's webgpu package (Dawn) finds: its
 * description, subgroup sizes and whether it has subgroups and timestamps. */
import { create, globals } from "webgpu";
Object.assign(globalThis, globals);
const adapter = await create([]).requestAdapter();
if (!adapter) {
  console.error("No WebGPU adapter found");
  process.exit(1);
}
const info = adapter.info ?? {};
console.log("WebGPU adapter:", JSON.stringify({
  vendor: info.vendor,
  architecture: info.architecture,
  device: info.device,
  description: info.description,
  subgroupMinSize: info.subgroupMinSize,
  subgroupMaxSize: info.subgroupMaxSize,
  subgroups: adapter.features.has("subgroups"),
  timestampQuery: adapter.features.has("timestamp-query"),
  maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
}));
process.exit(0);
