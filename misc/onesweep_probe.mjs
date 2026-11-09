/* Why is OneSweep slow on this GPU? Sweeps the lookback's spin limit
 * (MAX_SPIN_COUNT) and the tile size (KEYS_PER_THREAD) and reports, for one
 * pass, the time, the bandwidth it achieves, and how often the lookback
 * gave up and fell back to recomputing a predecessor's histogram.
 *
 *   node misc/onesweep_probe.mjs [log2 n] [u32|u64kv]
 *   node misc/onesweep_probe.mjs 24 u32
 *
 * Reading the result:
 * - fallbacks near 0 at every spin limit, and time per pass falling with
 *   bigger tiles: the pass is bound by per-tile work (lookback latency,
 *   ranking), not by memory;
 * - many fallbacks that fall as the spin limit rises, with the time: the
 *   lookback gives up too soon on this GPU, and each fallback rereads a tile;
 * - time per pass flat across both: bound elsewhere (the scatter, or
 *   occupancy); compare with misc/device_rates.mjs.
 * Keys are random each run; each point is the median of 5 runs after a
 * warm-up. */
import { createNodeDevice } from "./node_device.mjs";
import { OneSweepSort } from "../src/onesweep.mjs";
import { copyBandwidth, sortBytesPerElement } from "./device_rates.mjs";

const logN = Number(process.argv[2] ?? 24);
const kind = process.argv[3] ?? "u32";
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

console.log("keys/thread  spin   tiles   pass ms   pass GB/s   sort GB/s   fallbacks per pass (% of tiles)");
const kpts = options.datatype === "u64" ? [4, 6, 8, 10, 12] : [5, 7, 9, 11, 13, 15];
for (const KEYS_PER_THREAD of kpts) {
  for (const MAX_SPIN_COUNT of [4, 16, 64, 256, 4096]) {
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
        `${String(KEYS_PER_THREAD).padStart(11)} ${String(MAX_SPIN_COUNT).padStart(5)} ${String(fallbacks.tiles).padStart(7)}` +
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
