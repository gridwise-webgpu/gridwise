/* Benchmark SpMV (SegmentedReduce) schedules in Node (Dawn): thread-mapped,
 * workgroup-mapped and merge-path, on balanced and unbalanced sparse
 * matrices.
 *
 *   node misc/benchmark_spmv.mjs [matrices] [log2 sizes] [--mtx=file.mtx,...] [--items=K,...]
 *   node misc/benchmark_spmv.mjs uniform,rmat,denserow 18,20,22
 *   node misc/benchmark_spmv.mjs --mtx=soc-LiveJournal1.mtx,road_usa.mtx
 *
 * Generated matrices, 2^s rows (and columns), about 16 nonzeros per row:
 *   uniform    every row 16 nonzeros, random columns: balanced
 *   rmat       an R-MAT graph (a, b, c, d = 0.57, 0.19, 0.19, 0.05; 16 2^s
 *              edges; vertex ids shuffled): scale-free, a few rows with
 *              tens of thousands of nonzeros, many empty
 *   denserow   8 nonzeros per row, plus one row holding half of all
 *              nonzeros: the worst case for one invocation per row
 * --mtx: Matrix Market files (e.g. from the SuiteSparse collection,
 * sparse.tamu.edu): coordinate real, integer or pattern (values 1),
 * general or symmetric. --items: merge-path steps per invocation to try
 * (default 7).
 *
 * Per matrix: rows, nonzeros, mean and longest row; then per schedule the
 * GPU time (median after a 1 s warm-up), GFLOP/s (2 per nonzero), and the
 * bandwidth on the bytes SpMV must move (values, column indices and one x
 * per nonzero; offsets and y per row: 12 nnz + 8 rows), against a copy
 * kernel's. Every result is checked against a CPU SpMV. See
 * docs/segmented-reduce.md. */
import { readFileSync } from "node:fs";
import { createNodeDevice } from "./node_device.mjs";
import { SpMV, spmvReference } from "../src/spmv.mjs";
import { copyBandwidth } from "./device_rates.mjs";

const positional = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const option = (name) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const mtxFiles = option("mtx")?.split(",") ?? [];
const itemsList = (option("items") ?? "7").split(",").map(Number);
const generated = mtxFiles.length && !positional.length ? [] : (positional[0] ?? "uniform,rmat,denserow").split(",");
const logSizes = (positional[1] ?? "18,20,22").split(",").map(Number);

const { adapter, device } = await createNodeDevice();
console.log(`adapter: ${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description ?? ""}`);
if (!device.features.has("timestamp-query")) throw new Error("needs GPU timestamps");
const copyGBps = await copyBandwidth(device, Math.min(256 << 20, device.limits.maxBufferSize), { adapter });
console.log(`copy bandwidth (copy kernel, read + write): ${copyGBps.toFixed(1)} GB/s\n`);

/* ---- matrices */
function xorshift(seed) {
  let x = seed >>> 0 || 1;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return x >>> 0;
  };
}

/** CSR from coordinate lists, by a counting sort on rows */
function csrFromCoo(rows, cols, rowOf, colOf, valueOf) {
  const nnz = rowOf.length;
  const offsets = new Uint32Array(rows + 1);
  for (let k = 0; k < nnz; k++) offsets[rowOf[k] + 1]++;
  for (let r = 0; r < rows; r++) offsets[r + 1] += offsets[r];
  const cursor = offsets.slice(0, rows);
  const indices = new Uint32Array(nnz);
  const values = new Float32Array(nnz);
  for (let k = 0; k < nnz; k++) {
    const p = cursor[rowOf[k]]++;
    indices[p] = colOf[k];
    values[p] = valueOf ? valueOf[k] : 1;
  }
  return { rows, cols, offsets, indices, values };
}

const randomValues = (nnz, rand) => Float32Array.from({ length: nnz }, () => rand() / 2 ** 31 - 1);

const generators = {
  uniform(s) {
    const rows = 2 ** s, d = 16, rand = xorshift(0x1234 + s);
    const offsets = Uint32Array.from({ length: rows + 1 }, (_, r) => r * d);
    const indices = Uint32Array.from({ length: rows * d }, () => rand() % rows);
    return { rows, cols: rows, offsets, indices, values: randomValues(rows * d, rand) };
  },
  rmat(s) {
    const n = 2 ** s, edges = 16 * n, rand = xorshift(0x5eed + s);
    /* quadrant thresholds for a, a + b, a + b + c, as 32-bit integers */
    const [ta, tb, tc] = [0.57, 0.76, 0.95].map((p) => Math.floor(p * 2 ** 32));
    const perm = Uint32Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = rand() % (i + 1);
      [perm[i], perm[j]] = [perm[j], perm[i]];
    }
    const rowOf = new Uint32Array(edges), colOf = new Uint32Array(edges);
    for (let e = 0; e < edges; e++) {
      let r = 0, c = 0;
      for (let bit = 0; bit < s; bit++) {
        const u = rand();
        r = (r << 1) | (u >= tb ? 1 : 0);
        c = (c << 1) | ((u >= ta && u < tb) || u >= tc ? 1 : 0);
      }
      rowOf[e] = perm[r >>> 0];
      colOf[e] = perm[c >>> 0];
    }
    return csrFromCoo(n, n, rowOf, colOf, randomValues(edges, rand));
  },
  denserow(s) {
    const rows = 2 ** s, d = 8, dense = d * rows, rand = xorshift(0xd00d + s);
    const offsets = new Uint32Array(rows + 1);
    const heavy = rows >> 1; /* the dense row, in the middle */
    for (let r = 0; r < rows; r++) offsets[r + 1] = offsets[r] + d + (r === heavy ? dense : 0);
    const nnz = offsets[rows];
    const indices = Uint32Array.from({ length: nnz }, () => rand() % rows);
    return { rows, cols: rows, offsets, indices, values: randomValues(nnz, rand) };
  },
};

/** A Matrix Market coordinate file, parsed from its bytes */
function readMatrixMarket(path) {
  const bytes = readFileSync(path);
  let p = 0;
  const line = () => {
    const start = p;
    while (p < bytes.length && bytes[p] !== 10) p++;
    return bytes.toString("latin1", start, p++);
  };
  const header = line().toLowerCase().split(/\s+/);
  if (header[0] !== "%%matrixmarket" || header[2] !== "coordinate") throw new Error(`${path}: not a Matrix Market coordinate file`);
  const field = header[3], symmetric = header[4] !== "general";
  if (!["real", "integer", "pattern"].includes(field)) throw new Error(`${path}: field ${field} not supported`);
  let size;
  do size = line().trim(); while (size.startsWith("%") || size === "");
  const [rows, cols, entries] = size.split(/\s+/).map(Number);
  const skip = () => { while (bytes[p] === 32 || bytes[p] === 9 || bytes[p] === 10 || bytes[p] === 13) p++; };
  const int = () => { skip(); let v = 0; while (bytes[p] >= 48 && bytes[p] <= 57) v = v * 10 + bytes[p++] - 48; return v; };
  const token = () => { skip(); const start = p; while (p < bytes.length && bytes[p] > 32) p++; return Number(bytes.toString("latin1", start, p)); };
  const capacity = symmetric ? 2 * entries : entries;
  const rowOf = new Uint32Array(capacity), colOf = new Uint32Array(capacity), valueOf = new Float32Array(capacity);
  let k = 0;
  for (let e = 0; e < entries; e++) {
    const r = int() - 1, c = int() - 1, v = field === "pattern" ? 1 : token();
    rowOf[k] = r; colOf[k] = c; valueOf[k++] = v;
    if (symmetric && r !== c) { rowOf[k] = c; colOf[k] = r; valueOf[k++] = v; }
  }
  return csrFromCoo(rows, cols, rowOf.subarray(0, k), colOf.subarray(0, k), valueOf.subarray(0, k));
}

/* ---- runs */
const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
function upload(array) {
  const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(array.byteLength / 4) * 4), usage });
  device.queue.writeBuffer(buffer, 0, array);
  return buffer;
}
async function download(buffer, length) {
  const mappable = device.createBuffer({ size: length * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, mappable, 0, length * 4);
  device.queue.submit([encoder.finish()]);
  await mappable.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(mappable.getMappedRange().slice());
  mappable.destroy();
  return data;
}

async function benchmark(name, m) {
  const { rows, cols, offsets, indices, values } = m;
  const nnz = offsets[rows];
  let longest = 0, empty = 0;
  for (let r = 0; r < rows; r++) {
    const len = offsets[r + 1] - offsets[r];
    longest = Math.max(longest, len);
    if (!len) empty++;
  }
  console.log(
    `${name}: ${rows} rows, ${cols} columns, ${nnz} nonzeros; ${(nnz / rows).toFixed(1)} per row, longest ${longest}, ${((100 * empty) / rows).toFixed(0)}% empty`
  );
  const rand = xorshift(0xabcdef);
  const x = randomValues(cols, rand);
  const buffers = { offsets: upload(offsets), indices: upload(indices), values: upload(values), x: upload(x), y: upload(new Float32Array(rows)) };
  const expected = spmvReference({ offsets, indices, values, x });
  const bytes = 12 * nnz + 8 * rows;
  const results = [];
  const runs = [
    ["thread", "thread", {}],
    ["workgroup", "workgroup", {}],
    ...itemsList.map((K) => [`mergepath${itemsList.length > 1 ? ` K=${K}` : ""}`, "mergepath", { itemsPerThread: K }]),
  ];
  for (const [label, schedule, extra] of runs) {
    const spmv = new SpMV({ device, adapter, rows, nnz, schedule, ...extra });
    device.queue.writeBuffer(buffers.y, 0, new Float32Array(rows));
    const { ns, kernels } = await spmv.benchmark({ ...buffers, warmMs: 1000 });
    const y = await download(buffers.y, rows);
    let bad = -1;
    for (let r = 0; r < rows && bad < 0; r++) {
      let magnitude = 0;
      for (let k = offsets[r]; k < offsets[r + 1]; k++) magnitude += Math.abs(values[k] * x[indices[k]]);
      if (!(Math.abs(y[r] - expected[r]) <= 1e-4 * magnitude + 1e-30)) bad = r;
    }
    spmv.destroy();
    const fixupNS = kernels.filter((k) => /fixup|heads/.test(k.label ?? "")).reduce((a, k) => a + k.ns, 0);
    results.push({ schedule, ns });
    console.log(
      `  ${label.padEnd(14)} ${(ns / 1e6).toFixed(3).padStart(9)} ms  ${((2 * nnz) / ns).toFixed(2).padStart(7)} GFLOP/s  ` +
        `${(bytes / ns).toFixed(1).padStart(7)} GB/s (${((100 * bytes) / ns / copyGBps).toFixed(0).padStart(3)}% of copy)` +
        (fixupNS ? `  fixup ${(fixupNS / 1e3).toFixed(1)} µs` : "") +
        (bad >= 0 ? `  WRONG: row ${bad} = ${y[bad]}, expected ${expected[bad]}` : "")
    );
  }
  const best = (s) => Math.min(...results.filter((r) => r.schedule === s).map((r) => r.ns));
  const mergepath = best("mergepath");
  console.log(`  merge-path is ${(best("thread") / mergepath).toFixed(2)}x thread-mapped's speed, ${(best("workgroup") / mergepath).toFixed(2)}x workgroup-mapped's\n`);
  for (const b of Object.values(buffers)) b.destroy();
}

for (const name of generated) {
  if (!generators[name]) throw new Error(`unknown matrix ${name}; choose from ${Object.keys(generators)} or --mtx=`);
  for (const s of logSizes) {
    const start = performance.now();
    const m = generators[name](s);
    console.log(`(generated in ${((performance.now() - start) / 1e3).toFixed(1)} s)`);
    await benchmark(`${name} 2^${s}`, m);
  }
}
for (const path of mtxFiles) {
  const start = performance.now();
  const m = readMatrixMarket(path);
  console.log(`(read in ${((performance.now() - start) / 1e3).toFixed(1)} s)`);
  await benchmark(path.split("/").pop(), m);
}
process.exit(0);
