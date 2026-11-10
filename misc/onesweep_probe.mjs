/* OneSweep's tile size on this GPU: for each KEYS_PER_THREAD, the time of
 * one pass, the bandwidth it reaches, and how that compares with a copy
 * kernel. Every timed sort is checked against a CPU sort.
 *
 *   node misc/onesweep_probe.mjs [log2 n] [u32|u64kv] [--kpt=K,...]
 *   node misc/onesweep_probe.mjs 24 u64kv
 *
 * Keys are random each run; each point is the median of 5 runs after a 1 s
 * warm-up (a V100 needed more than 150 ms to reach full clock). Fewer keys
 * per thread than RADIX / subgroup size (8 with 32-lane subgroups) sort
 * wrongly, and OneSweep refuses them; more than workgroup memory holds are
 * refused too. Findings so far: docs/sort.md, "Performance notes". */
import { createNodeDevice } from "./node_device.mjs";
import { OneSweepSort } from "../src/onesweep.mjs";
import { copyBandwidth, sortBytesPerElement } from "./device_rates.mjs";

const positional = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const logN = Number(positional[0] ?? 24);
const kind = positional[1] ?? "u32";
const kptArg = process.argv.find((a) => a.startsWith("--kpt="))?.slice(6);
const n = 2 ** logN;
const options = kind === "u64kv" ? { datatype: "u64", type: "keyvalue" } : { datatype: "u32", type: "keysonly" };
const keyBytes = options.datatype === "u64" ? 8 : 4;
const passBytes = 2 * (keyBytes + (options.type === "keyvalue" ? 4 : 0)) * n;

const { adapter, device } = await createNodeDevice();
console.log(`adapter: ${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description ?? ""}`);
if (!device.features.has("timestamp-query")) throw new Error("needs GPU timestamps");
const copyGBps = await copyBandwidth(device, Math.min(256 << 20, device.limits.maxBufferSize), { adapter });
console.log(`copy bandwidth (copy kernel): ${copyGBps.toFixed(1)} GB/s`);
console.log(`n = 2^${logN}, ${kind}: a pass moves ${passBytes / n} bytes per key; at copy bandwidth it would take ${(passBytes / copyGBps / 1e6).toFixed(3)} ms\n`);

let seed = 0x2545f491;
const keys = Uint32Array.from({ length: (keyBytes / 4) * n }, () => (seed = (Math.imul(seed ^ (seed >>> 15), 0x2c1b3c6d) + 0x9e3779b9) >>> 0));
const sortedKeys = keyBytes === 8 ? new BigUint64Array(keys.buffer).slice().sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)) : keys.slice().sort();
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const args = { keysInOut: device.createBuffer({ size: keyBytes * n, usage }), keysTemp: device.createBuffer({ size: keyBytes * n, usage }) };
if (options.type === "keyvalue") {
  Object.assign(args, { payloadInOut: device.createBuffer({ size: 4 * n, usage }), payloadTemp: device.createBuffer({ size: 4 * n, usage }) });
}
const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];

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

const kpts = kptArg ? kptArg.split(",").map(Number) : options.datatype === "u64" ? [8, 10, 12, 14, 16] : [8, 9, 10, 11, 12, 13, 14, 15];
console.log("keys/thread  wg bytes   tiles   pass ms   pass GB/s   % of copy   sort GB/s");
for (const KEYS_PER_THREAD of kpts) {
  let sort;
  try {
    sort = new OneSweepSort({ device, adapter, ...options, direction: "ascending", KEYS_PER_THREAD });
    device.queue.writeBuffer(args.keysInOut, 0, keys);
    await sort.warmUp(args, 1000);
    const totals = [];
    const passes = [];
    for (let r = 0; r < 5; r++) {
      device.queue.writeBuffer(args.keysInOut, 0, keys);
      await sort.execute({ ...args, enableGPUTiming: true });
      const { gpuTotalTimeNS } = await sort.getTimingResult();
      const ns = [gpuTotalTimeNS].flat();
      const labels = sort.kernelLabels();
      totals.push(ns.reduce((a, b) => a + b, 0));
      const passNS = ns.filter((_, i) => labels[i].includes("onesweep_pass"));
      passes.push(passNS.reduce((a, b) => a + b, 0) / passNS.length);
      if (r === 0 && !(await sortedCorrectly(sort))) throw new Error("sorted WRONGLY");
    }
    const pass = median(passes);
    console.log(
      `${String(KEYS_PER_THREAD).padStart(11)} ${String(sort.passWorkgroupBytes(KEYS_PER_THREAD)).padStart(9)} ${String(sort.passWorkgroupCount).padStart(7)}` +
        ` ${(pass / 1e6).toFixed(3).padStart(9)} ${(passBytes / pass).toFixed(1).padStart(11)} ${((100 * passBytes) / pass / copyGBps).toFixed(0).padStart(10)}%` +
        ` ${((sortBytesPerElement(options) * n) / median(totals)).toFixed(1).padStart(11)}`
    );
  } catch (e) {
    console.log(`${String(KEYS_PER_THREAD).padStart(11)}   ${e.message.split("\n")[0]}`);
  }
  sort?.destroy();
}
process.exit(0);
