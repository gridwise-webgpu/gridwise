/* Why is OneSweep slow on this GPU? Sweeps the tile size (KEYS_PER_THREAD)
 * and, with --spin, the lookback's spin limit (MAX_SPIN_COUNT), and reports
 * for each: the workgroup memory a pass uses, the time of one pass, the
 * bandwidth it achieves, and how often the lookback gave up and recomputed a
 * predecessor's histogram (the fallback).
 *
 *   node misc/onesweep_probe.mjs [log2 n] [u32|u64kv] [--spin]
 *   node misc/onesweep_probe.mjs 24 u32
 *
 * On an M1 Max (2^24): the spin limit made no difference (fallbacks were a
 * few percent of tiles at any limit); the tile size did, through occupancy.
 * A pass was fastest with the largest tile that still let two (u32 keys, 11
 * keys per thread, 1.4x the default's speed) or three (u64 key-value, 4 keys
 * per thread, 1.35x) workgroups share a core's 32 KB of workgroup memory.
 * Keys are random each run; each point is the median of 5 runs after a
 * warm-up. */
import { createNodeDevice } from "./node_device.mjs";
import { OneSweepSort } from "../src/onesweep.mjs";
import { copyBandwidth, sortBytesPerElement } from "./device_rates.mjs";

const logN = Number(process.argv[2] ?? 24);
const kind = process.argv[3] ?? "u32";
const spins = process.argv.includes("--spin") ? [4, 16, 64, 256, 4096] : [4];
const n = 2 ** logN;
const options = kind === "u64kv" ? { datatype: "u64", type: "keyvalue" } : { datatype: "u32", type: "keysonly" };
const keyBytes = options.datatype === "u64" ? 8 : 4;

const { adapter, device } = await createNodeDevice();
console.log(`adapter: ${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description ?? ""}`);
if (!device.features.has("timestamp-query")) throw new Error("needs GPU timestamps");
console.log(`copy bandwidth: ${(await copyBandwidth(device, Math.min(256 << 20, device.limits.maxBufferSize))).toFixed(1)} GB/s`);
console.log(`n = 2^${logN}, ${kind}; per pass: ${2 * (keyBytes + (options.type === "keyvalue" ? 4 : 0))} bytes per key\n`);

let seed = 0x2545f491;
const keys = Uint32Array.from({ length: (keyBytes / 4) * n }, () => (seed = (Math.imul(seed ^ (seed >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0));
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const args = { keysInOut: device.createBuffer({ size: keyBytes * n, usage }), keysTemp: device.createBuffer({ size: keyBytes * n, usage }) };
if (options.type === "keyvalue") Object.assign(args, { payloadInOut: device.createBuffer({ size: 4 * n, usage }), payloadTemp: device.createBuffer({ size: 4 * n, usage }) });
const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
const passBytes = 2 * (keyBytes + (options.type === "keyvalue" ? 4 : 0)) * n;

console.log("keys/thread  spin  wg bytes   tiles   pass ms   pass GB/s   sort GB/s   fallbacks per pass (% of tiles)");
const kpts = options.datatype === "u64" ? [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] : [3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15];
for (const KEYS_PER_THREAD of kpts) {
  for (const MAX_SPIN_COUNT of spins) {
    let sort;
    try {
      sort = new OneSweepSort({ device, adapter, ...options, direction: "ascending", KEYS_PER_THREAD, MAX_SPIN_COUNT, countFallbacks: true });
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
        if (r === 0) fallbacks = await sort.fallbackCounts();
      }
      const pass = median(passes);
      const total = median(totals);
      const fb = fallbacks.perPass;
      const pct = ((100 * fb.reduce((a, b) => a + b, 0)) / fb.length / fallbacks.tiles).toFixed(1);
      console.log(
        `${String(KEYS_PER_THREAD).padStart(11)} ${String(MAX_SPIN_COUNT).padStart(5)} ${String(sort.passWorkgroupBytes(KEYS_PER_THREAD)).padStart(9)} ${String(fallbacks.tiles).padStart(7)}` +
          ` ${(pass / 1e6).toFixed(3).padStart(9)} ${(passBytes / pass).toFixed(1).padStart(11)}` +
          ` ${((sortBytesPerElement(options) * n) / total).toFixed(1).padStart(11)}   ${fb.join(" ")} (${pct}%)`
      );
    } catch (e) {
      console.log(`${String(KEYS_PER_THREAD).padStart(11)} ${String(MAX_SPIN_COUNT).padStart(5)}   ${e.message.split("\n")[0]}`);
    }
    sort?.destroy();
  }
}
process.exit(0);
