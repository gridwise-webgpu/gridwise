/* Print the WebGPU adapter that Node's webgpu package (Dawn) finds: its
 * description, subgroup sizes and whether it has subgroups and timestamps.
 * EXPECT_VENDOR=<vendor> makes it fail if the adapter is someone else's
 * (e.g., a software renderer standing in for a simulated GPU). */
import { nodeGPU, adapterFlags } from "./node_device.mjs";
/* GRIDWISE_ADAPTER=<name> picks one adapter by name (see adapterFlags) */
const adapter = await nodeGPU(await adapterFlags()).requestAdapter();
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
const expected = process.env.EXPECT_VENDOR;
if (expected && info.vendor !== expected) {
  console.error(`Expected a "${expected}" adapter, found "${info.vendor}"`);
  process.exit(1);
}
process.exit(0);
