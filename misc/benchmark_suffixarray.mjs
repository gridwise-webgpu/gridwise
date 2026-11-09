/* Benchmark SuffixArray in Node (Dawn) against its bandwidth model.
 *
 *   node misc/benchmark_suffixarray.mjs [corpora] [log2 sizes] [--file=path]
 *   node misc/benchmark_suffixarray.mjs random256,english 20,24
 *   node misc/benchmark_suffixarray.mjs file 24 --file=enwik8
 *   node misc/benchmark_suffixarray.mjs english 24 --kpt=4 --no-speculate --no-local-scatter
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
import { copyBandwidth, sortRate, randomAccessCost } from "./device_rates.mjs";

const positional = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const fileArg = process.argv.find((a) => a.startsWith("--file="))?.slice(7);
/* --kpt=K: OneSweep's KEYS_PER_THREAD for SuffixArray's sorts; --no-speculate;
 * --local-scatter / --no-local-scatter (default: on from 2^23);
 * --local-bits=B: local scatter with 2^B blocks (default 6) */
const kptArg = process.argv.find((a) => a.startsWith("--kpt="))?.slice(6);
const localBitsArg = process.argv.find((a) => a.startsWith("--local-bits="))?.slice(13);
const saOptions = {
  ...(kptArg && { sortOptions: { KEYS_PER_THREAD: Number(kptArg) } }),
  ...(process.argv.includes("--no-speculate") && { speculate: false }),
  ...(process.argv.includes("--local-scatter") && { localScatter: true }),
  ...(process.argv.includes("--no-local-scatter") && { localScatter: false }),
  ...(localBitsArg && { localScatter: true, localScatterBits: Number(localBitsArg) }),
};
const corpora = (positional[0] ?? "random256,random4,english,allA").split(",");
const logSizes = (positional[1] ?? "20,24").split(",").map(Number);

const { adapter, device } = await createNodeDevice();
const { vendor, architecture, description } = adapter.info;
console.log(`adapter: ${vendor} ${architecture} ${description ?? ""}`);
const gpuTiming = device.features.has("timestamp-query");
console.log(`subgroups: ${device.features.has("subgroups")}, GPU timestamps: ${gpuTiming}`);

/* ---- device rates (misc/device_rates.mjs): copy bandwidth, big enough to
 * leave the caches; and, at each size, the rate of the sort SuffixArray uses
 * (64-bit key-value) and the cost of random 4-byte gathers and scatters */
const copyBytes = Math.min(device.limits.maxBufferSize, Math.max(1 << 20, Math.min(256 << 20, 16 * 2 ** Math.max(...logSizes))));
const bandwidth = await copyBandwidth(device, copyBytes);
console.log(`copy bandwidth (read + write): ${bandwidth.toFixed(1)} GB/s`);
const rates = {};
if (gpuTiming) {
  console.log("device rates at each size (for the calibrated model):");
  for (const logN of logSizes) {
    const n = 2 ** logN;
    const sort = await sortRate(device, n, { datatype: "u64", type: "keyvalue", keyBits: 63, adapter });
    const { gather, scatter } = await randomAccessCost(device, n, { adapter });
    rates[logN] = { sort, gather, scatter };
    console.log(
      `  2^${logN}: 64-bit key-value sort ${sort.toFixed(1)} GB/s; random 4-byte gather ${gather.toFixed(3)} ns, scatter ${scatter.toFixed(3)} ns`
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
    const sa = new SuffixArray({ device, adapter, n, ...saOptions });

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

    const traffic = suffixArrayTraffic({ n, activeCounts: sa.activeCounts, capacities: sa.capacities, localScatter: sa.localScatter });
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
      `\n${name} 2^${logN}: ${errors ? "WRONG" + errors : "correct"}; ${sa.rounds} doubling rounds (${sa.speculatedRounds} speculative), local scatter ${sa.localScatter ? `on (${sa.localScatterBits} bits)` : "off"}, work ${work.toFixed(2)} n` +
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
