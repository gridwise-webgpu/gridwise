/* OneSweep sort timing diagnostic (Node + Dawn), per kernel.
 *
 *   node misc/sort_timing_node.mjs [log2 sizes (8,20,24)] [rounds (2)]
 *
 * For each size, key type (u32, i32) and input order (random, presorted), a
 * fresh sorter sorts freshly uploaded keys 5 times; prints the median total
 * GPU time, M keys/s, and the median time of each kernel (global_hist,
 * onesweep_scan, onesweep_pass 0..3). The whole table is repeated `rounds`
 * times to show drift between rounds. Timestamps are full resolution
 * (disable-dawn-features=timestamp_quantization). */
import { create, globals } from "webgpu";
import { OneSweepSort } from "../src/onesweep.mjs";

Object.assign(globalThis, globals);
const logSizes = (process.argv[2] ?? "8,20,24").split(",").map(Number);
const rounds = +(process.argv[3] ?? 2);

/* keep the GPU instance referenced, or Dawn tears down the device */
const gpu = create(["disable-dawn-features=timestamp_quantization"]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups"] : [])],
  requiredLimits: {
    maxComputeWorkgroupStorageSize: 32768,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  },
});
const { vendor, architecture, description } = adapter.info;
console.log(`adapter: ${vendor} ${architecture} ${description ?? ""}`);

let state = 0x2545f491;
const rand = () => (state = (Math.imul(state ^ (state >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0);
const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

async function measure(n, datatype, order) {
  const keys = new Uint32Array(n).map(rand);
  if (order === "presorted") {
    if (datatype === "i32") new Int32Array(keys.buffer).sort();
    else keys.sort();
  }
  const keysInOut = device.createBuffer({ size: n * 4, usage });
  const keysTemp = device.createBuffer({ size: n * 4, usage });
  const sorter = new OneSweepSort({ device, adapter, datatype, inputLength: n, type: "keysonly", direction: "ascending" });
  device.queue.writeBuffer(keysInOut, 0, keys);
  await sorter.execute({ keysInOut, keysTemp }); /* compile */
  const totals = [];
  const perKernel = [];
  for (let r = 0; r < 5; r++) {
    device.queue.writeBuffer(keysInOut, 0, keys); /* fresh input every time */
    await sorter.execute({ keysInOut, keysTemp, enableGPUTiming: true });
    const ns = [(await sorter.getTimingResult()).gpuTotalTimeNS].flat();
    totals.push(ns.reduce((a, b) => a + b, 0));
    ns.forEach((t, i) => (perKernel[i] ??= []).push(t));
  }
  sorter.destroy?.();
  keysInOut.destroy();
  keysTemp.destroy();
  const ms = median(totals) / 1e6;
  return { ms, mkeys: n / ms / 1e3, kernels: perKernel.map((k) => median(k) / 1e6) };
}

console.log("kernels: global_hist, onesweep_scan, onesweep_pass 0..3 (ms, medians of 5)");
for (let round = 1; round <= rounds; round++) {
  console.log(`--- round ${round}`);
  for (const logN of logSizes) {
    for (const datatype of ["u32", "i32"]) {
      for (const order of ["random", "presorted"]) {
        const r = await measure(2 ** logN, datatype, order);
        console.log(
          `2^${logN} ${datatype} ${order.padEnd(9)} total ${r.ms.toFixed(3).padStart(8)} ms ${r.mkeys.toFixed(0).padStart(6)} Mkeys/s | ` +
            r.kernels.map((t) => t.toFixed(3)).join(" ")
        );
      }
    }
  }
}
process.exit(0);
