/* Benchmark Gridwise primitives in Node (Dawn), with primitive.benchmark():
 * a warm-up on each primitive, then the median of 5 timed executes, per
 * kernel when the GPU has timestamp queries.
 *
 *   node misc/benchmark_node.mjs [primitives] [log2 sizes] [--kernels]
 *   node misc/benchmark_node.mjs scan,sort 20,24 --kernels
 *
 * primitives: scan (DLDFScan), hscan (HierarchicalScan), sort (OneSweep,
 * u32 keys), histogram (256 even f32 bins), perm (MultisplitPerm), shun
 * (ShunPerm, parallel Fisher-Yates: one trial per execute), feistel (FeistelPerm,
 * 8 rounds), swapornot (SwapOrNotPerm), mitchell (MitchellPerm, Thrust's
 * scan mode); default all. Sizes default to 20,22,24. --kernels also prints
 * each kernel's time.
 * Throughput is elements per second. Sort sorts in place, so after the
 * first execute it sorts sorted keys.
 *
 * For measurements that must be comparable between runs, lock the GPU's
 * clocks where you can (docs/node_webgpu_regression_plan.md). */
import { createNodeDevice } from "./node_device.mjs";
import { DLDFScan } from "../src/scandldf.mjs";
import { HierarchicalScan } from "../src/scan.mjs";
import { OneSweepSort } from "../src/onesweep.mjs";
import { Histogram } from "../src/histogram.mjs";
import { EvenBins } from "../src/bins.mjs";
import { MultisplitPerm } from "../src/multisplitperm.mjs";
import { ShunPerm } from "../src/shunperm.mjs";
import { FeistelPerm, SwapOrNotPerm, MitchellPerm } from "../src/bijectionperm.mjs";
import { BinOpAdd } from "../src/binop.mjs";

const positional = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const showKernels = process.argv.includes("--kernels");
const which = (positional[0] ?? "scan,hscan,sort,histogram,perm,shun,feistel,swapornot,mitchell").split(",");
const logSizes = (positional[1] ?? "20,22,24").split(",").map(Number);

const { adapter, device } = await createNodeDevice();
const { vendor, architecture, description } = adapter.info;
console.log(`adapter: ${vendor} ${architecture} ${description ?? ""}`);
console.log(`subgroups: ${device.features.has("subgroups")}, GPU timestamps: ${device.features.has("timestamp-query")}`);

const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const buffer = (bytes, fill) => {
  const b = device.createBuffer({ size: bytes, usage });
  if (fill) device.queue.writeBuffer(b, 0, fill);
  return b;
};
let state = 0x2545f491;
const randomU32 = (n) => Uint32Array.from({ length: n }, () => (state = (Math.imul(state ^ (state >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0));

/* each entry: (n) => { primitive, args, destroy } */
const setups = {
  scan: (n) => {
    const inputBuffer = buffer(n * 4, randomU32(n).map((x) => x & 0xff));
    const outputBuffer = buffer(n * 4);
    const primitive = new DLDFScan({ device, adapter, datatype: "u32", type: "exclusive", binop: new BinOpAdd({ datatype: "u32" }) });
    return { primitive, args: { inputBuffer, outputBuffer }, buffers: [inputBuffer, outputBuffer] };
  },
  hscan: (n) => {
    const inputBuffer = buffer(n * 4, randomU32(n).map((x) => x & 0xff));
    const outputBuffer = buffer(n * 4);
    const primitive = new HierarchicalScan({ device, adapter, datatype: "u32", type: "exclusive", binop: new BinOpAdd({ datatype: "u32" }) });
    return { primitive, args: { inputBuffer, outputBuffer }, buffers: [inputBuffer, outputBuffer] };
  },
  sort: (n) => {
    const keysInOut = buffer(n * 4, randomU32(n));
    const keysTemp = buffer(n * 4);
    const primitive = new OneSweepSort({ device, adapter, datatype: "u32", inputLength: n, type: "keysonly", direction: "ascending" });
    return { primitive, args: { keysInOut, keysTemp }, buffers: [keysInOut, keysTemp] };
  },
  histogram: (n) => {
    const inputBuffer = buffer(n * 4, new Float32Array(randomU32(n).map((x) => x / 2 ** 32).buffer));
    const outputBuffer = buffer(256 * 4);
    const primitive = new Histogram({ device, adapter, bins: new EvenBins({ datatype: "f32", numBins: 256, lower: 0, upper: 1 }) });
    return { primitive, args: { inputBuffer, outputBuffer }, buffers: [inputBuffer, outputBuffer] };
  },
  perm: (n) => {
    const outputBuffer = buffer(n * 4);
    const primitive = new MultisplitPerm({ device, adapter, n });
    return { primitive, args: { outputBuffer }, buffers: [outputBuffer] };
  },
  shun: (n) => {
    const outputBuffer = buffer(n * 4);
    const primitive = new ShunPerm({ device, adapter, n });
    return { primitive, args: { outputBuffer }, buffers: [outputBuffer] };
  },
  ...Object.fromEntries(
    [
      ["feistel", (n) => new FeistelPerm({ device, adapter, n, rounds: 8 })],
      ["swapornot", (n) => new SwapOrNotPerm({ device, adapter, n })],
      ["mitchell", (n) => new MitchellPerm({ device, adapter, n })],
    ].map(([name, make]) => [
      name,
      (n) => {
        const outputBuffer = buffer(n * 4);
        return { primitive: make(n), args: { outputBuffer }, buffers: [outputBuffer] };
      },
    ])
  ),
};

for (const name of which) {
  if (!setups[name]) throw new Error(`unknown primitive ${name}; choose from ${Object.keys(setups)}`);
  for (const logN of logSizes) {
    const n = 2 ** logN;
    const { primitive, args, buffers } = setups[name](n);
    const r = await primitive.benchmark(args);
    const ms = r.ns / 1e6;
    console.log(
      `${name.padEnd(10)} 2^${String(logN).padEnd(3)} ${ms.toFixed(3).padStart(9)} ms ${(n / r.ns).toFixed(2).padStart(8)} G el/s` +
        `  (${r.timing} timing, ${r.trials} trials)`
    );
    if (showKernels) {
      for (const k of r.kernels) console.log(`    ${(k.ns / 1e6).toFixed(3).padStart(9)} ms  ${k.label}`);
    }
    primitive.destroy();
    for (const b of buffers) b.destroy();
  }
}
process.exit(0);
