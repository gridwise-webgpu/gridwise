/* Why is OneSweep slow on this GPU? Sweeps the tile size (KEYS_PER_THREAD)
 * and, with --spin, the lookback's spin limit (MAX_SPIN_COUNT), and reports
 * for each: the workgroup memory a pass uses, the time of one pass, the
 * bandwidth it achieves, and how often the lookback gave up and recomputed a
 * predecessor's histogram (the fallback).
 *
 *   node misc/onesweep_probe.mjs [log2 n] [u32|u64kv] [--spin] [--ablate] [--kpt=K,...]
 *   node misc/onesweep_probe.mjs 24 u32
 *
 * On an M1 Max (2^24): the spin limit made no difference (fallbacks were a
 * few percent of tiles at any limit). For u32 keys, 11 keys per thread made
 * a pass 1.4x faster than the default 15, likely because two workgroups then
 * share a core's 32 KB of workgroup memory; for u64 key-value, the default
 * 12 was already the fastest valid tile. Fewer than RADIX / subgroup size
 * (8 with 32-lane subgroups) sort wrongly: OneSweep now refuses them, and
 * the probe skips them and checks that every sort it times is correct.
 * Keys are random each run; each point is the median of 5 runs after a
 * warm-up. */
import { createNodeDevice } from "./node_device.mjs";
import { OneSweepSort } from "../src/onesweep.mjs";
import { copyBandwidth, sortBytesPerElement } from "./device_rates.mjs";

const logN = Number(process.argv[2] ?? 24);
const kind = process.argv[3] ?? "u32";
const spins = process.argv.includes("--spin") ? [4, 16, 64, 256, 4096] : [4];
/* --ablate: also time each pass with its global scatter and/or its lookback
 * removed (OneSweep's ABLATE; the output is then not sorted) */
const ablations = process.argv.includes("--ablate") ? ["", "scatter", "lookback", "scatter+lookback"] : [""];
const kptArg = process.argv.find((a) => a.startsWith("--kpt="))?.slice(6);
const n = 2 ** logN;
const options = kind === "u64kv" ? { datatype: "u64", type: "keyvalue" } : { datatype: "u32", type: "keysonly" };
const keyBytes = options.datatype === "u64" ? 8 : 4;

const { adapter, device } = await createNodeDevice();
console.log(`adapter: ${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description ?? ""}`);
if (!device.features.has("timestamp-query")) throw new Error("needs GPU timestamps");
const copyGBps = await copyBandwidth(device, Math.min(256 << 20, device.limits.maxBufferSize), { adapter });
console.log(`copy bandwidth (copy kernel): ${copyGBps.toFixed(1)} GB/s`);
console.log(
  `n = 2^${logN}, ${kind}; per pass: ${2 * (keyBytes + (options.type === "keyvalue" ? 4 : 0))} bytes per key;` +
    ` a pass at copy bandwidth would take ${((2 * (keyBytes + (options.type === "keyvalue" ? 4 : 0)) * 2 ** logN) / copyGBps / 1e6).toFixed(3)} ms\n`
);

let seed = 0x2545f491;
const keys = Uint32Array.from({ length: (keyBytes / 4) * n }, () => (seed = (Math.imul(seed ^ (seed >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0));
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const args = { keysInOut: device.createBuffer({ size: keyBytes * n, usage }), keysTemp: device.createBuffer({ size: keyBytes * n, usage }) };
if (options.type === "keyvalue") Object.assign(args, { payloadInOut: device.createBuffer({ size: 4 * n, usage }), payloadTemp: device.createBuffer({ size: 4 * n, usage }) });
const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
const passBytes = 2 * (keyBytes + (options.type === "keyvalue" ? 4 : 0)) * n;

console.log("keys/thread  spin  ablate             wg bytes   tiles   pass ms   pass GB/s   sort GB/s   fallbacks per pass (% of tiles)");
const kpts = kptArg
  ? kptArg.split(",").map(Number)
  : options.datatype === "u64"
    ? [4, 5, 6, 7, 8, 9, 10, 11, 12]
    : [4, 6, 8, 9, 10, 11, 12, 13, 14, 15];
/* the keys, sorted on the CPU, to check each configuration's output */
const sortedKeys =
  keyBytes === 8 ? new BigUint64Array(keys.buffer).slice().sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)) : keys.slice().sort();
async function sortedCorrectly(sort) {
  const bytes = keyBytes * n;
  const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(args[sort.outputKeysBufferName], 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const out = keyBytes === 8 ? new BigUint64Array(readback.getMappedRange()) : new Uint32Array(readback.getMappedRange());
  let ok = true;
  for (let i = 0; i < n && ok; i++) ok = out[i] === sortedKeys[i];
  readback.destroy();
  return ok;
}
for (const KEYS_PER_THREAD of kpts) {
  for (const MAX_SPIN_COUNT of spins) {
   for (const ABLATE of ablations) {
    let sort;
    const head = `${String(KEYS_PER_THREAD).padStart(11)} ${String(MAX_SPIN_COUNT).padStart(5)}  ${(ABLATE || "none").padEnd(16)}`;
    try {
      sort = new OneSweepSort({ device, adapter, ...options, direction: "ascending", KEYS_PER_THREAD, MAX_SPIN_COUNT, ABLATE, countFallbacks: true });
      device.queue.writeBuffer(args.keysInOut, 0, keys);
      await sort.warmUp(args, 150);
      const totals = [];
      const passes = [];
      let fallbacks;
      for (let r = 0; r < 5; r++) {
        device.queue.writeBuffer(args.keysInOut, 0, keys);
        await sort.execute({ ...args, enableGPUTiming: true });
        const { gpuTotalTimeNS } = await sort.getTimingResult();
        const ns = [gpuTotalTimeNS].flat();
        const labels = sort.kernelLabels();
        totals.push(ns.reduce((a, b) => a + b, 0));
        const passNS = ns.filter((_, i) => labels[i].includes("onesweep_pass"));
        passes.push(passNS.reduce((a, b) => a + b, 0) / passNS.length);
        if (r === 0) {
          fallbacks = await sort.fallbackCounts();
          if (!ABLATE && !(await sortedCorrectly(sort))) throw new Error("sorted WRONGLY");
        }
      }
      const pass = median(passes);
      const total = median(totals);
      const fb = fallbacks.perPass;
      const pct = ((100 * fb.reduce((a, b) => a + b, 0)) / fb.length / fallbacks.tiles).toFixed(1);
      console.log(
        `${head} ${String(sort.passWorkgroupBytes(KEYS_PER_THREAD)).padStart(9)} ${String(fallbacks.tiles).padStart(7)}` +
          ` ${(pass / 1e6).toFixed(3).padStart(9)} ${(passBytes / pass).toFixed(1).padStart(11)}` +
          ` ${((sortBytesPerElement(options) * n) / total).toFixed(1).padStart(11)}   ${fb.join(" ")} (${pct}%)`
      );
    } catch (e) {
      console.log(`${head}   ${e.message.split("\n")[0]}`);
    }
    sort?.destroy();
   }
  }
}
process.exit(0);
