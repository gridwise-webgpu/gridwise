/* What a GPU achieves on the basic memory operations, for performance models
 * of Gridwise primitives: copy bandwidth, OneSweep sort rate, and the cost of
 * random 4-byte reads and writes. All but the copy are GPU-timed, so they
 * need timestamp queries.
 *
 *   import { copyBandwidth, sortRate, randomAccessCost } from "./device_rates.mjs";
 *   const gbps = await copyBandwidth(device, 256 << 20);
 *   const sort = await sortRate(device, n, { datatype: "u64", type: "keyvalue", keyBits: 63 });
 *   const { gather, scatter } = await randomAccessCost(device, n);   // ns each
 *
 * Or on its own, at log2 sizes (default 20,22,24):
 *
 *   node misc/device_rates.mjs 20,22,24
 *
 * Sort and random-access rates depend on n, through the caches: measure them
 * at the size the model is for. */
import { pathToFileURL } from "node:url";
import { OneSweepSort } from "../src/onesweep.mjs";
import { BasePrimitive, Kernel, UniformStruct } from "../src/primitive.mjs";

const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
const storage = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
let seed = 0x2545f491;
const randomU32 = (k) =>
  Uint32Array.from({ length: k }, () => (seed = (Math.imul(seed ^ (seed >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0));

/** Summed GPU kernel time (ns) of one execute() of primitive. */
export async function gpuNS(primitive, args) {
  await primitive.execute({ ...args, enableGPUTiming: true });
  const { gpuTotalTimeNS } = await primitive.getTimingResult();
  return [gpuTotalTimeNS].flat().reduce((a, b) => a + b, 0);
}

/** Median GPU time (ns) of `repeats` executes after one warm-up; `before`
 * runs ahead of each (e.g., to restore unsorted input). */
async function timeGPU(primitive, args, { repeats = 3, before = () => {} } = {}) {
  const times = [];
  for (let i = 0; i <= repeats; i++) {
    before();
    const ns = await gpuNS(primitive, args);
    if (i > 0) times.push(ns);
  }
  return median(times);
}

/** Copy bandwidth in GB/s, counting each byte read once and written once:
 * copyBufferToBuffer between two buffers of `bytes`, CPU-timed over 20
 * copies. Make bytes large enough to leave the caches. */
export async function copyBandwidth(device, bytes) {
  const usage = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const a = device.createBuffer({ size: bytes, usage });
  const b = device.createBuffer({ size: bytes, usage });
  const run = async (reps) => {
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < reps; i++) encoder.copyBufferToBuffer(i % 2 ? b : a, 0, i % 2 ? a : b, 0, bytes);
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - start;
  };
  await run(4);
  const ms = Math.min(await run(20), await run(20));
  a.destroy();
  b.destroy();
  return (2 * bytes * 20) / (ms * 1e6);
}

/** Bytes a OneSweep sort of one element moves: the global histogram reads
 * the key once, then each pass reads and writes key and payload. */
export function sortBytesPerElement({ datatype = "u32", type = "keysonly", keyBits } = {}) {
  const keyBytes = datatype === "u64" ? 8 : 4;
  const bits = keyBits ?? 8 * keyBytes;
  const elementBytes = keyBytes + (type === "keyvalue" ? 4 : 0);
  return keyBytes + 2 * elementBytes * Math.ceil(bits / 8);
}

/** OneSweep's rate on n random keys, in GB/s of sortBytesPerElement(). */
export async function sortRate(device, n, { datatype = "u32", type = "keysonly", keyBits, adapter } = {}) {
  const keyBytes = datatype === "u64" ? 8 : 4;
  const keys = randomU32((keyBytes / 4) * n);
  const buffer = (bytes) => device.createBuffer({ size: bytes, usage: storage() });
  const args = { keysInOut: buffer(keyBytes * n), keysTemp: buffer(keyBytes * n) };
  if (type === "keyvalue") Object.assign(args, { payloadInOut: buffer(4 * n), payloadTemp: buffer(4 * n) });
  const sort = new OneSweepSort({ device, adapter, datatype, type, direction: "ascending", ...(keyBits && { KEY_BITS: keyBits }) });
  /* sorts in place: restore the unsorted keys before each run */
  const ns = await timeGPU(sort, args, { before: () => device.queue.writeBuffer(args.keysInOut, 0, keys) });
  sort.destroy();
  for (const b of Object.values(args)) b.destroy();
  return (sortBytesPerElement({ datatype, type, keyBits }) * n) / ns;
}

/** One kernel: a random 4-byte gather (out[i] = data[perm[i]]) or scatter
 * (data[perm[i]] = i) over an n-element array. */
class RandomAccess extends BasePrimitive {
  constructor(args) {
    super(args);
    this.knownBuffers = ["perm", "data", "out"];
    this.workgroupSize = 256;
    this.datatype = "u32";
    this.params = new UniformStruct("RAParams", { n: "u32" });
  }
  compute() {
    const wgsl = () => /* wgsl */ `
${this.fnDeclarations.include("computeLinearizedGridParametersSplit")}
@compute @workgroup_size(256)
fn ${this.mode}(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  if (gid >= ra.n) { return; }
  ${this.mode === "gather" ? "out[gid] = data[perm[gid]];" : "data[perm[gid]] = gid;"}
}`;
    return [
      this.params.allocate("raParams", { n: this.n }),
      new Kernel({
        kernel: wgsl,
        entryPoint: this.mode,
        label: `random ${this.mode}`,
        buffers: [
          { name: "perm", access: "read", type: "array<u32>" },
          { name: "data", access: "read_write", type: "array<u32>" },
          { name: "out", access: "read_write", type: "array<u32>" },
          { name: "ra", buffer: "raParams", uniform: this.params },
        ],
        getDispatchGeometry: () => this.getSimpleDispatchGeometry({ workgroupCount: Math.ceil(this.n / 256) }),
      }),
    ];
  }
}

/** ns per random 4-byte gather and per scatter into an n-element array,
 * through a random permutation (each element touched once). The streamed
 * permutation and output are included, as they are in real kernels. */
export async function randomAccessCost(device, n, { adapter } = {}) {
  const perm = Uint32Array.from({ length: n }, (_, i) => i);
  const r = randomU32(n);
  for (let i = n - 1; i > 0; i--) {
    const j = r[i] % (i + 1);
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  const buffers = Object.fromEntries(["perm", "data", "out"].map((k) => [k, device.createBuffer({ size: 4 * n, usage: storage() })]));
  device.queue.writeBuffer(buffers.perm, 0, perm);
  const result = {};
  for (const mode of ["gather", "scatter"]) {
    const ra = new RandomAccess({ device, adapter, n, mode });
    result[mode] = (await timeGPU(ra, buffers)) / n;
    ra.destroy();
  }
  for (const b of Object.values(buffers)) b.destroy();
  return result;
}

/* ---- on its own: print the rates */
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const { createNodeDevice } = await import("./node_device.mjs");
  const logSizes = (process.argv[2] ?? "20,22,24").split(",").map(Number);
  const { adapter, device } = await createNodeDevice();
  const { vendor, architecture, description } = adapter.info;
  console.log(`adapter: ${vendor} ${architecture} ${description ?? ""}`);
  const copyBytes = Math.min(device.limits.maxBufferSize, 256 << 20);
  console.log(`copy bandwidth (read + write, ${copyBytes >> 20} MiB): ${(await copyBandwidth(device, copyBytes)).toFixed(1)} GB/s`);
  if (!device.features.has("timestamp-query")) {
    console.log("no GPU timestamps: sort and random-access rates need them");
    process.exit(0);
  }
  const sorts = [
    ["u32 keys", { datatype: "u32", type: "keysonly" }],
    ["u32 key-value", { datatype: "u32", type: "keyvalue" }],
    ["u64 key-value", { datatype: "u64", type: "keyvalue" }],
  ];
  for (const logN of logSizes) {
    const n = 2 ** logN;
    const parts = [];
    for (const [name, options] of sorts) parts.push(`${name} ${(await sortRate(device, n, { ...options, adapter })).toFixed(1)}`);
    const { gather, scatter } = await randomAccessCost(device, n, { adapter });
    console.log(
      `2^${logN}: sort GB/s: ${parts.join(", ")}; random 4-byte gather ${gather.toFixed(3)} ns, scatter ${scatter.toFixed(3)} ns`
    );
  }
  process.exit(0);
}
