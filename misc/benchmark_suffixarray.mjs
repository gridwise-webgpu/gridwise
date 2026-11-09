/* Benchmark SuffixArray in Node (Dawn) against its bandwidth model.
 *
 *   node misc/benchmark_suffixarray.mjs [corpora] [log2 sizes] [--file=path]
 *   node misc/benchmark_suffixarray.mjs random256,english 20,24
 *   node misc/benchmark_suffixarray.mjs file 24 --file=enwik8
 *
 * corpora: random256 (uniform bytes: almost every suffix is resolved by its
 * first 7 bytes), random4 (uniform over ACGT), english (a word salad with
 * long repeats), fibonacci (the Fibonacci word: repeats at every scale),
 * allA (the worst case: about log2(n / 7) rounds, all of n active), file
 * (the first n bytes of --file, repeated if shorter). Default
 * random256,random4,english,allA at 2^20 and 2^24.
 *
 * Prints the device rates the calibrated model uses at each size (64-bit
 * key-value sort rate; cost of a random 4-byte gather and scatter), then, per
 * run: rounds; the work ratio (active suffixes summed over the doubling
 * rounds, over n); the wall time of execute(), which includes a readback
 * after every round; the summed GPU kernel time; the overhead per
 * submission; the model at copy bandwidth and calibrated with the device
 * rates (suffixArrayTraffic); and, per phase, the GPU time, modeled bytes
 * and achieved bandwidth. Every result is
 * checked with checkSuffixArray. See docs/suffix-array.md, "Expected
 * performance". */
import { readFileSync } from "node:fs";
import { createNodeDevice } from "./node_device.mjs";
import { SuffixArray, suffixArrayTraffic, suffixArrayPhase, checkSuffixArray } from "../src/suffixarray.mjs";
import { OneSweepSort } from "../src/onesweep.mjs";
import { BasePrimitive, Kernel, UniformStruct } from "../src/primitive.mjs";

const positional = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const fileArg = process.argv.find((a) => a.startsWith("--file="))?.slice(7);
const corpora = (positional[0] ?? "random256,random4,english,allA").split(",");
const logSizes = (positional[1] ?? "20,24").split(",").map(Number);

const { adapter, device } = await createNodeDevice();
const { vendor, architecture, description } = adapter.info;
console.log(`adapter: ${vendor} ${architecture} ${description ?? ""}`);
const gpuTiming = device.features.has("timestamp-query");
console.log(`subgroups: ${device.features.has("subgroups")}, GPU timestamps: ${gpuTiming}`);

/* ---- copy bandwidth: the yardstick (each byte is read once and written once) */
async function copyBandwidth(bytes) {
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
  return (2 * bytes * 20) / (ms * 1e6); /* GB/s */
}
/* big enough to leave the caches (at least 16 bytes per character of the largest n) */
const copyBytes = Math.min(device.limits.maxBufferSize, Math.max(1 << 20, Math.min(256 << 20, 16 * 2 ** Math.max(...logSizes))));
const bandwidth = await copyBandwidth(copyBytes);
console.log(`copy bandwidth (read + write): ${bandwidth.toFixed(1)} GB/s`);

/* ---- device rates at size n, to calibrate the model: what this GPU achieves
 * on the sort SuffixArray uses, and on random 4-byte reads and writes into an
 * n-element array (both depend on n, through the caches) */
const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
async function gpuNS(primitive, args) {
  await primitive.execute({ ...args, enableGPUTiming: true });
  const { gpuTotalTimeNS } = await primitive.getTimingResult();
  return [gpuTotalTimeNS].flat().reduce((a, b) => a + b, 0);
}
const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
let seed = 0x2545f491;
const randomU32 = (k) => Uint32Array.from({ length: k }, () => (seed = (Math.imul(seed ^ (seed >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0));

/** GB/s of a 64-bit key-value OneSweep sort of n random keys with 63-bit keys
 * (the initial round's sort), counting 8 + 24 bytes per element per pass */
async function sortRate(n) {
  const keys = randomU32(2 * n);
  const keysInOut = device.createBuffer({ size: 8 * n, usage: storage });
  const keysTemp = device.createBuffer({ size: 8 * n, usage: storage });
  const payloadInOut = device.createBuffer({ size: 4 * n, usage: storage });
  const payloadTemp = device.createBuffer({ size: 4 * n, usage: storage });
  const sort = new OneSweepSort({ device, adapter, datatype: "u64", type: "keyvalue", direction: "ascending", KEY_BITS: 63 });
  const times = [];
  for (let i = 0; i < 4; i++) {
    device.queue.writeBuffer(keysInOut, 0, keys); /* unsorted every time */
    const ns = await gpuNS(sort, { keysInOut, keysTemp, payloadInOut, payloadTemp });
    if (i > 0) times.push(ns);
  }
  sort.destroy();
  for (const b of [keysInOut, keysTemp, payloadInOut, payloadTemp]) b.destroy();
  return ((8 + 24 * 8) * n) / median(times);
}

/** ns per random 4-byte gather (out[i] = data[perm[i]]) and scatter
 * (data[perm[i]] = i) into an n-element array, perm a random permutation */
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
async function randomAccessNS(n) {
  const perm = Uint32Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = randomU32(1)[0] % (i + 1);
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  const buffers = { perm: device.createBuffer({ size: 4 * n, usage: storage }), data: device.createBuffer({ size: 4 * n, usage: storage }), out: device.createBuffer({ size: 4 * n, usage: storage }) };
  device.queue.writeBuffer(buffers.perm, 0, perm);
  const result = {};
  for (const mode of ["gather", "scatter"]) {
    const ra = new RandomAccess({ device, adapter, n, mode });
    const times = [];
    for (let i = 0; i < 4; i++) {
      const ns = await gpuNS(ra, buffers);
      if (i > 0) times.push(ns);
    }
    result[mode] = median(times) / n;
    ra.destroy();
  }
  for (const b of Object.values(buffers)) b.destroy();
  return result;
}

const rates = {};
if (gpuTiming) {
  console.log("device rates at each size (for the calibrated model):");
  for (const logN of logSizes) {
    const n = 2 ** logN;
    const sort = await sortRate(n);
    const { gather, scatter } = await randomAccessNS(n);
    rates[logN] = { sort, gather, scatter };
    console.log(
      `  2^${logN}: 64-bit key-value sort ${sort.toFixed(1)} GB/s; random 4-byte gather ${gather.toFixed(3)} ns, scatter ${scatter.toFixed(3)} ns` +
        ` (= ${(4 / gather).toFixed(1)}, ${(4 / scatter).toFixed(1)} GB/s of useful data)`
    );
  }
}

/* ---- corpora */
function xorshift(seed) {
  let x = seed >>> 0 || 1;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return x >>> 0;
  };
}
const corpus = {
  random256: (n) => {
    const r = xorshift(n);
    return Uint8Array.from({ length: n }, () => r() & 0xff);
  },
  random4: (n) => {
    const r = xorshift(n ^ 4);
    return Uint8Array.from({ length: n }, () => "ACGT".charCodeAt(r() & 3));
  },
  english: (n) => {
    const words =
      "the of and to in a is that for it as was with be by on not he this are or his from at which but have an they you were her she there would their we him been has when who will more no if out so said what up its about into than them can only other new some could time these two may then do first any my now such like our over man me even most made after also did many before must through back years where much your way well down should because each just those people".split(
        " "
      );
    const r = xorshift(n ^ 0xe);
    const out = new Uint8Array(n);
    let k = 0;
    while (k < n) {
      /* Zipf-ish word choice; now and then repeat an earlier stretch */
      if (k > 1000 && (r() & 63) === 0) {
        const from = r() % (k - 500);
        const len = Math.min(n - k, 20 + (r() % 400));
        out.copyWithin(k, from, from + len);
        k += len;
        continue;
      }
      const w = words[Math.floor(words.length * (r() / 2 ** 32) ** 3)] + " ";
      for (let i = 0; i < w.length && k < n; i++) out[k++] = w.charCodeAt(i);
    }
    return out;
  },
  fibonacci: (n) => {
    const out = new Uint8Array(n);
    /* the Fibonacci word: character i is 'b' iff floor((i+2)/phi) - floor((i+1)/phi) == 0 */
    const phi = (1 + Math.sqrt(5)) / 2;
    for (let i = 0; i < n; i++) out[i] = Math.floor((i + 2) / phi) - Math.floor((i + 1) / phi) ? 97 : 98;
    return out;
  },
  allA: (n) => new Uint8Array(n).fill(65),
  file: (n) => {
    if (!fileArg) throw new Error("corpus 'file' needs --file=path");
    const data = readFileSync(fileArg);
    const out = new Uint8Array(n);
    for (let k = 0; k < n; k += data.length) out.set(data.subarray(0, Math.min(data.length, n - k)), k);
    return out;
  },
};

/* ---- runs */
const ms = (ns) => ns / 1e6;
const fmt = (x, digits = 2, width = 8) => x.toFixed(digits).padStart(width);
for (const name of corpora) {
  if (!corpus[name]) throw new Error(`unknown corpus ${name}; choose from ${Object.keys(corpus)}`);
  for (const logN of logSizes) {
    const n = 2 ** logN;
    const text = corpus[name](n);
    const padded = new Uint8Array(Math.ceil(n / 4) * 4);
    padded.set(text);
    const inputBuffer = device.createBuffer({ size: padded.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inputBuffer, 0, padded);
    const outputBuffer = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const sa = new SuffixArray({ device, adapter, n });

    /* warm up (compiles pipelines, allocates), then the median of 3 wall times */
    for (let i = 0; i < 2; i++) await sa.execute({ inputBuffer, outputBuffer });
    await device.queue.onSubmittedWorkDone();
    const walls = [];
    for (let i = 0; i < 3; i++) {
      const start = performance.now();
      await sa.execute({ inputBuffer, outputBuffer });
      await device.queue.onSubmittedWorkDone();
      walls.push(performance.now() - start);
    }
    const wall = walls.sort((a, b) => a - b)[1];

    /* per-phase GPU time from one timed run */
    const phases = {};
    let gpuTotal = 0;
    let kernels = 0;
    if (gpuTiming) {
      await sa.execute({ inputBuffer, outputBuffer, enableGPUTiming: true });
      for (const step of sa.timings) {
        for (const k of step.kernels) {
          const phase = suffixArrayPhase(k.label);
          phases[phase] = (phases[phase] ?? 0) + k.ns;
          gpuTotal += k.ns;
          kernels++;
        }
      }
    }

    /* check */
    const readback = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, n * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const errors = checkSuffixArray(text, new Uint32Array(readback.getMappedRange().slice(0)));
    readback.destroy();

    const traffic = suffixArrayTraffic({ n, activeCounts: sa.activeCounts });
    const work = sa.activeCounts.slice(1).reduce((a, b) => a + b, 0) / n; /* doubling rounds only */
    const modelMS = traffic.total / bandwidth / 1e6;
    /* calibrated: the sort at this GPU's sort rate, random accesses at its
     * gather and scatter costs, everything else at copy bandwidth */
    let calibratedMS = NaN;
    if (rates[logN]) {
      const { sort, gather, scatter } = rates[logN];
      const randomNS = (traffic.random.pack ?? 0) * gather + (traffic.totalRandom - (traffic.random.pack ?? 0)) * scatter;
      const streamBytes = traffic.total - traffic.bytes.sort - 4 * traffic.totalRandom;
      calibratedMS = ms(traffic.bytes.sort / sort + streamBytes / bandwidth + randomNS);
    }
    const perRound = (wall - ms(gpuTotal)) / (sa.rounds + 2);
    console.log(
      `\n${name} 2^${logN}: ${errors ? "WRONG" + errors : "correct"}; ${sa.rounds} doubling rounds, work ${work.toFixed(2)} n` +
        `\n  wall ${fmt(wall)} ms (${fmt(n / wall / 1e3, 1)} M char/s)   GPU ${fmt(ms(gpuTotal))} ms (${kernels} kernels)` +
        `   overhead ${fmt(perRound)} ms per submission` +
        `\n  model: ${fmt(modelMS)} ms at copy bandwidth (${(traffic.total / n).toFixed(0)} B/char); calibrated ${fmt(calibratedMS)} ms` +
        ` (GPU time / calibrated = ${(ms(gpuTotal) / calibratedMS).toFixed(2)})`
    );
    if (gpuTiming) {
      console.log("  phase           GPU ms   model MB   achieved GB/s");
      for (const phase of Object.keys(traffic.bytes)) {
        const t = phases[phase] ?? 0;
        const b = traffic.bytes[phase];
        console.log(`  ${phase.padEnd(14)} ${fmt(ms(t))} ${fmt(b / 1e6, 1, 10)} ${fmt(t ? b / t : 0, 1, 12)}`);
      }
      if (phases.other) console.log(`  other          ${fmt(ms(phases.other))}`);
    }
    sa.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
  }
}
process.exit(0);
