/* Histogram throughput in Node (Dawn), with GPU timestamps.
 *
 *   node examples/histogram_bench_node.mjs [log2 n (24)] [trials (20)]
 *
 * Like Ashkiani et al.'s GPU Multisplit histogram experiments: f32 samples
 * uniform in [0, 1024), even bins, and range bins with uneven splitters, for
 * a range of bin counts. Range bins run twice, with the levels in workgroup
 * memory and in a storage buffer (levelsInWorkgroup), to compare the two.
 * Prints G samples/s (median of several executes of `trials` dispatches). */
import { create, globals } from "webgpu";
import { Histogram } from "../src/histogram.mjs";
import { EvenBins, RangeBins } from "../src/bins.mjs";

Object.assign(globalThis, globals);
const logN = +(process.argv[2] ?? 24);
const trials = +(process.argv[3] ?? 20);
const n = 2 ** logN;

/* keep the GPU instance referenced: if it is garbage-collected, Dawn tears
 * down the device under us */
const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (!adapter.features.has("timestamp-query")) {
  console.error("This adapter has no timestamp-query; GPU timing is unavailable.");
  process.exit(1);
}
const device = await adapter.requestDevice({
  requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups"] : [])],
  requiredLimits: { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize },
});
console.log(`adapter: ${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description ?? ""}`);
console.log(`n = 2^${logN}, ${trials} dispatches per timing, median of 5`);

/* deterministic pseudo-random samples and splitters */
let state = 0x9e3779b9;
const rand = () => ((state = Math.imul(state ^ (state >>> 15), 0x2c1b3c6d) >>> 0), state / 2 ** 32);
const input = Float32Array.from({ length: n }, () => rand() * 1024);
const inputBuffer = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(inputBuffer, 0, input);

async function measure(bins, options = {}) {
  const outputBuffer = device.createBuffer({
    size: bins.numBins * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const h = new Histogram({ device, bins, ...options });
  await h.execute({ inputBuffer, outputBuffer }); /* compile and warm up */
  const times = [];
  for (let r = 0; r < 5; r++) {
    await h.execute({ inputBuffer, outputBuffer, trials, enableGPUTiming: true });
    const { gpuTotalTimeNS } = await h.getTimingResult();
    const ns = Array.isArray(gpuTotalTimeNS) ? gpuTotalTimeNS.reduce((a, b) => a + b, 0) : gpuTotalTimeNS;
    times.push(ns / trials);
  }
  times.sort((a, b) => a - b);
  const label = `${h.workgroupBins ? "wg" : "global"} bins${bins.needsLevels ? (h.workgroupLevels ? ", wg levels" : ", storage levels") : ""}`;
  h.destroy();
  outputBuffer.destroy();
  return { gsps: n / times[2], label };
}

const fmt = (r) => `${r.gsps.toFixed(2).padStart(7)} (${r.label})`;
console.log("bins   even                         range, levels in workgroup     range, levels in storage");
for (const m of [2, 8, 32, 128, 256, 1024, 4096]) {
  const even = await measure(new EvenBins({ datatype: "f32", numBins: m, lower: 0, upper: 1024 }));
  /* uneven but strictly increasing: one random splitter in each 1024/m slice */
  const levels = Float32Array.from({ length: m + 1 }, (_, i) =>
    i === 0 ? 0 : i === m ? 1024 : ((i - 1 + 0.25 + 0.5 * rand()) * 1024) / m
  );
  const range = new RangeBins({ datatype: "f32", levels });
  const wg = await measure(range, { levelsInWorkgroup: true });
  const storage = await measure(range, { levelsInWorkgroup: false });
  console.log(`${String(m).padStart(4)}   ${fmt(even).padEnd(29)}${fmt(wg).padEnd(31)}${fmt(storage)}`);
}
process.exit(0);
