/* MultisplitPerm throughput in Node (Dawn), with GPU timestamps: atomic vs
 * stable ranking.
 *
 *   node examples/multisplitperm_bench_node.mjs [log2 sizes (20,22,24,26)] [trials (10)]
 *
 * Variants, each at its default group size:
 *   atomic          ranking "atomic" (unbiased; output depends on the run)
 *   reproducible    ranking "atomic", reproducible: true (sorts sub-buckets)
 *   stable          ranking "stable" (reproducible without sorting)
 *   pairs atomic    Philox-7 (index, random word) pairs, ranking "atomic"
 *   pairs stable    the same, ranking "stable"
 * Prints G elements/s (median of 5 executes of `trials` dispatches each),
 * after running each variant back to back for 200 ms (GPUs clock down
 * after light work). */
import { create, globals } from "webgpu";
import { MultisplitPerm } from "../src/multisplitperm.mjs";

Object.assign(globalThis, globals);
const logSizes = (process.argv[2] ?? "20,22,24,26").split(",").map(Number);
const trials = +(process.argv[3] ?? 10);

/* keep the GPU instance referenced: if it is garbage-collected, Dawn tears
 * down the device under us */
const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (!adapter.features.has("timestamp-query") || !adapter.features.has("subgroups")) {
  console.error("This benchmark needs timestamp-query and subgroups.");
  process.exit(1);
}
const device = await adapter.requestDevice({
  requiredFeatures: ["timestamp-query", "subgroups"],
  requiredLimits: {
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  },
});
const { vendor, architecture, description, subgroupMinSize, subgroupMaxSize } = adapter.info;
console.log(`adapter: ${vendor} ${architecture} ${description ?? ""}; subgroups ${subgroupMinSize}-${subgroupMaxSize}`);
console.log(`${trials} dispatches per timing, median of 5; G elements/s`);

const philox7 = { key: [0x12345678, 0x9abcdef0], rounds: 7 };
const variants = [
  ["atomic", {}],
  ["reproducible", { reproducible: true }],
  ["stable", { ranking: "stable" }],
  ["pairs atomic", { rng: philox7, carry: "pairs" }],
  ["pairs stable", { rng: philox7, carry: "pairs", ranking: "stable" }],
];

async function measure(n, outputBuffer, options) {
  const perm = new MultisplitPerm({ device, adapter, n, ...options });
  await perm.execute({ outputBuffer }); /* compile */
  const warmUntil = performance.now() + 200;
  while (performance.now() < warmUntil) await perm.execute({ outputBuffer, trials });
  const times = [];
  for (let r = 0; r < 5; r++) {
    await perm.execute({ outputBuffer, trials, enableGPUTiming: true });
    const { gpuTotalTimeNS } = await perm.getTimingResult();
    const ns = Array.isArray(gpuTotalTimeNS) ? gpuTotalTimeNS.reduce((a, b) => a + b, 0) : gpuTotalTimeNS;
    times.push(ns / trials);
  }
  times.sort((a, b) => a - b);
  const config = `${perm.config.passBits.join("+")} bits, groups of ${perm.config.groupSize}`;
  perm.destroy();
  return { geps: n / times[2], config };
}

console.log(`${"n".padEnd(6)}${variants.map(([name]) => name.padStart(14)).join("")}`);
for (const logN of logSizes) {
  const n = 2 ** logN;
  const outputBuffer = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const results = [];
  for (const [, options] of variants) results.push(await measure(n, outputBuffer, options));
  outputBuffer.destroy();
  console.log(`2^${logN}`.padEnd(6) + results.map((r) => r.geps.toFixed(2).padStart(14)).join(""));
  console.log("      " + results.map((r, i) => `${variants[i][0]}: ${r.config}`).join("; "));
}
process.exit(0);
