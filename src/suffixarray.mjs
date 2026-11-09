/**
 * SuffixArray: the suffix array of a byte string, by prefix doubling over a
 * full radix sort of the suffixes that are still tied. Design and the
 * reasons for it: docs/suffix-array.md.
 *
 *   const sa = new SuffixArray({ device, n });
 *   await sa.execute({ inputBuffer, outputBuffer });
 *   // inputBuffer: the n bytes, packed 4 per u32 word (little-endian; e.g. a
 *   //   Uint8Array padded to a multiple of 4 bytes, written as is)
 *   // outputBuffer: n u32s, SA[0..n)
 *   // rankBuffer (optional): n u32s, rank[i] = 1 + position of suffix i
 *
 * Ranks: rank[i] = 1 + (start of suffix i's group in the final order); a
 * position at or past n has rank 0, below every suffix. Ranks take
 * b = ceil(log2(n + 1)) bits.
 *
 * Initial round: each suffix's first 7 symbols as one 63-bit key, 9 bits per
 * symbol (byte + 1; 0 = past the end, so byte 0 needs no sentinel), sorted
 * with OneSweepSort. Each later round, over the m active (still tied)
 * suffixes: pack (rank[i] << b | rank[i + h]) as a 2b-bit key; sort the
 * first m keys; flag group and subgroup heads; two inclusive max-scans give
 * each sorted position p its group start g and subgroup start q; the new
 * rank is rank[i] + (q - g); singletons retire; an exclusive scan compacts
 * the rest into the next active list. The host reads the active count after
 * each round (one 4-byte map) and stops at 0. Then SA[rank[i] - 1] = i.
 *
 * execute() runs several GPU submissions (one per round), so it cannot take
 * an encoder, and trials must be 1. After execute(): `rounds` (doubling
 * rounds run), `activeCounts` (active suffixes before each round), and, with
 * enableGPUTiming, `timings` ([{ stage, m, kernels: [{ label, ns }] }]).
 *
 * Limits: n <= 2^30 (OneSweep). Memory: about 44n bytes of working buffers
 * plus the input and output (docs/suffix-array.md).
 */
import {
  BasePrimitive,
  Kernel,
  AllocateBuffer,
  ExecutePrimitive,
  CopyBufferToBuffer,
  UniformStruct,
} from "./primitive.mjs";
import { OneSweepSort } from "./onesweep.mjs";
import { DLDFScan } from "./scandldf.mjs";
import { BinOpAdd, BinOpMax } from "./binop.mjs";

const WORKGROUP_SIZE = 256;
const INITIAL_SYMBOLS = 7; /* 7 x 9 bits = 63 bits */
const MAX_ROUNDS = 40;

export class SuffixArray extends BasePrimitive {
  constructor(args) {
    super(args);
    if (!(Number.isInteger(this.n) && this.n >= 1 && this.n <= 2 ** 30)) {
      throw new Error("SuffixArray: requires an integer n in [1, 2^30]");
    }
    this.knownBuffers = ["inputBuffer", "outputBuffer", "rankBuffer"];
    this.workgroupSize = WORKGROUP_SIZE;
    this.datatype = "u32";
    this.singleTrialOnly = true;
    this.rankBits = Math.ceil(Math.log2(this.n + 1));
    this.capacity4 = Math.ceil(this.n / 4) * 4; /* DLDFScan buffers: whole vec4s */
    this.params = new UniformStruct("SAParams", { n: "u32", m: "u32", h: "u32", b: "u32", initial: "u32" });
    const sortArgs = { device: this.device, adapter: this.adapter, datatype: "u64", type: "keyvalue", direction: "ascending" };
    this.initialSort = new OneSweepSort({ ...sortArgs, KEY_BITS: 9 * INITIAL_SYMBOLS });
    this.roundSort = new OneSweepSort({ ...sortArgs, KEY_BITS: 2 * this.rankBits });
    const scan = (type, Op) => new DLDFScan({ device: this.device, adapter: this.adapter, datatype: "u32", type, binop: new Op({ datatype: "u32" }) });
    this.groupScan = scan("inclusive", BinOpMax);
    this.subgroupScan = scan("inclusive", BinOpMax);
    this.keepScan = scan("exclusive", BinOpAdd);
    this.stage = "initial";
    if (!("label" in args)) this.label += this.makeParamString(["useSubgroups"]);
  }

  /* ---------------------------------------------------------------- WGSL */

  /** helpers shared by the kernels: key packing and the rank in a key */
  commonWGSL = () => /* wgsl */ `
${this.fnDeclarations.include("computeLinearizedGridParametersSplit")}
/* (r1 << b) | r2 as a 64-bit value, x = low word, y = high word (b < 32) */
fn packRanks(r1: u32, r2: u32) -> vec2u {
  return vec2u((r1 << sa.b) | r2, r1 >> (32u - sa.b));
}
/* the group rank a key carries: 1 in the initial round (one group), else r1 */
fn groupRank(k: vec2u) -> u32 {
  if (sa.initial == 1u) { return 1u; }
  return (k.x >> sa.b) | (k.y << (32u - sa.b));
}
fn sameKey(a: vec2u, b: vec2u) -> bool { return all(a == b); }
`;

  /** initial round: 7 symbols of 9 bits (byte + 1, 0 past the end) per suffix */
  initialWGSL = () => /* wgsl */ `
${this.commonWGSL()}
fn symbol(j: u32) -> u32 {
  if (j >= sa.n) { return 0u; }
  return ((text[j >> 2u] >> ((j & 3u) * 8u)) & 0xffu) + 1u;
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let i = gid;
  if (i >= sa.n) { return; }
  /* symbol k sits at bit 9 * (6 - k): bits 54, 45, 36 in y; 27 straddles; 18, 9, 0 in x */
  let s0 = symbol(i); let s1 = symbol(i + 1u); let s2 = symbol(i + 2u); let s3 = symbol(i + 3u);
  let s4 = symbol(i + 4u); let s5 = symbol(i + 5u); let s6 = symbol(i + 6u);
  let lo = (s3 << 27u) | (s4 << 18u) | (s5 << 9u) | s6;
  let hi = (s0 << 22u) | (s1 << 13u) | (s2 << 4u) | (s3 >> 5u);
  keys[i] = vec2u(lo, hi);
  payload[i] = i;
}`;

  /** later rounds: key = (rank[i], rank[i + h]) for the m active suffixes */
  packWGSL = () => /* wgsl */ `
${this.commonWGSL()}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let p = gid;
  if (p >= sa.m) { return; }
  let i = activeList[p];
  let j = i + sa.h;
  let r2 = select(0u, rank[j], j < sa.n);
  keys[p] = packRanks(rank[i], r2);
  payload[p] = i;
}`;

  /** heads: group starts (old rank changes) and subgroup starts (key changes) */
  headsWGSL = () => /* wgsl */ `
${this.commonWGSL()}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let p = gid;
  if (p >= sa.m) { return; }
  let k = keys[p];
  var groupHead = p == 0u;
  var subgroupHead = p == 0u;
  if (p > 0u) {
    let prev = keys[p - 1u];
    groupHead = groupRank(k) != groupRank(prev);
    subgroupHead = !sameKey(k, prev);
  }
  groupFlags[p] = select(0u, p, groupHead);
  subgroupFlags[p] = select(0u, p, subgroupHead);
}`;

  /** new ranks, and which suffixes stay active */
  updateWGSL = () => /* wgsl */ `
${this.commonWGSL()}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let p = gid;
  if (p >= sa.m) { return; }
  let k = keys[p];
  let q = subgroupStart[p];
  rank[payload[p]] = groupRank(k) + (q - groupStart[p]);
  /* a subgroup of one: p starts it and p + 1 does not continue it */
  let alone = q == p && (p + 1u == sa.m || !sameKey(keys[p + 1u], k));
  keep[p] = select(1u, 0u, alone);
}`;

  /** the next active list, and its length */
  compactWGSL = () => /* wgsl */ `
${this.commonWGSL()}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let p = gid;
  if (p >= sa.m) { return; }
  if (keep[p] == 1u) { nextActive[offsets[p]] = payload[p]; }
  if (p + 1u == sa.m) { count[0] = offsets[p] + keep[p]; }
}`;

  /** SA[rank[i] - 1] = i */
  outputWGSL = () => /* wgsl */ `
${this.commonWGSL()}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let i = gid;
  if (i >= sa.n) { return; }
  outputBuffer[rank[i] - 1u] = i;
}`;

  /* ---------------------------------------------------------------- actions */

  /** the buffers, allocated once for n */
  allocations() {
    const n = this.n;
    const scratch = (label, bytes) => new AllocateBuffer({ label, size: Math.max(16, bytes), clearBufferOnReuse: false });
    return [
      scratch("keys", 8 * n),
      scratch("keysTemp", 8 * n),
      scratch("listA", 4 * n),
      scratch("listB", 4 * n),
      ...(this.hasBuffer("rankBuffer") ? [] : [scratch("rankBuffer", 4 * n)]),
      scratch("groupFlags", 4 * this.capacity4),
      scratch("subgroupFlags", 4 * this.capacity4),
      scratch("groupStart", 4 * this.capacity4),
      scratch("subgroupStart", 4 * this.capacity4),
      new AllocateBuffer({ label: "count", size: 16, clearBufferOnReuse: false }),
      new AllocateBuffer({
        label: "countReadback",
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        clearBufferOnReuse: false,
      }),
    ];
  }

  kernel(wgsl, label, buffers, count) {
    return new Kernel({
      kernel: wgsl,
      entryPoint: "main",
      label: `SuffixArray ${label}`,
      buffers: [...buffers, { name: "sa", buffer: "saParams", uniform: this.params }],
      getDispatchGeometry: () => this.getSimpleDispatchGeometry({ workgroupCount: Math.ceil(count / WORKGROUP_SIZE) }),
    });
  }

  /** one round, initial or doubling, over this.m suffixes.
   * Payloads live in two buffers, "listA" and "listB". The round's active
   * list (from the last round's compaction) is in one, `this.activeList`;
   * pack writes the keys' payloads into the other, which the sort uses as
   * payloadInOut, with the active list's buffer as payloadTemp (pack has
   * consumed it). The compaction then writes the next active list into
   * whichever of the two the sort did not leave its output in. */
  roundActions() {
    const initial = this.stage === "initial";
    const m = initial ? this.n : this.m;
    const sort = initial ? this.initialSort : this.roundSort;
    sort.length = m;
    for (const scan of [this.groupScan, this.subgroupScan, this.keepScan]) scan.length = m;
    const other = (list) => (list === "listA" ? "listB" : "listA");
    const work = initial ? "listA" : other(this.activeList);
    const spare = other(work);
    const sortedKeys = sort.outputIsInTemp ? "keysTemp" : "keys";
    const sortedPayload = sort.outputIsInTemp ? spare : work;
    const nextActive = other(sortedPayload);
    this.nextActiveList = nextActive;
    const keys = (buffer, access = "read") => ({ name: "keys", buffer, access, type: "array<vec2u>" });
    return [
      this.params.allocate("saParams", { n: this.n, m, h: this.h, b: this.rankBits, initial: initial ? 1 : 0 }),
      initial
        ? this.kernel(this.initialWGSL, "initial keys", [u32s("text", "inputBuffer"), keys("keys", "read_write"), u32s("payload", work, "read_write")], m)
        : this.kernel(
            this.packWGSL,
            `pack (h = ${this.h})`,
            [u32s("activeList", this.activeList), u32s("rank", "rankBuffer"), keys("keys", "read_write"), u32s("payload", work, "read_write")],
            m
          ),
      new ExecutePrimitive({
        primitive: sort,
        buffers: { keysInOut: "keys", keysTemp: "keysTemp", payloadInOut: work, payloadTemp: spare },
      }),
      this.kernel(this.headsWGSL, "heads", [keys(sortedKeys), u32s("groupFlags", "groupFlags", "read_write"), u32s("subgroupFlags", "subgroupFlags", "read_write")], m),
      new ExecutePrimitive({ primitive: this.groupScan, buffers: { inputBuffer: "groupFlags", outputBuffer: "groupStart" } }),
      new ExecutePrimitive({ primitive: this.subgroupScan, buffers: { inputBuffer: "subgroupFlags", outputBuffer: "subgroupStart" } }),
      /* keep flags reuse groupFlags; their exclusive scan reuses subgroupFlags */
      this.kernel(
        this.updateWGSL,
        "update ranks",
        [
          keys(sortedKeys),
          u32s("payload", sortedPayload),
          u32s("groupStart", "groupStart"),
          u32s("subgroupStart", "subgroupStart"),
          u32s("rank", "rankBuffer", "read_write"),
          u32s("keep", "groupFlags", "read_write"),
        ],
        m
      ),
      new ExecutePrimitive({ primitive: this.keepScan, buffers: { inputBuffer: "groupFlags", outputBuffer: "subgroupFlags" } }),
      this.kernel(
        this.compactWGSL,
        "compact",
        [
          u32s("keep", "groupFlags"),
          u32s("offsets", "subgroupFlags"),
          u32s("payload", sortedPayload),
          u32s("nextActive", nextActive, "read_write"),
          u32s("count", "count", "read_write"),
        ],
        m
      ),
      new CopyBufferToBuffer({ source: "count", destination: "countReadback", size: 4 }),
    ];
  }

  compute() {
    const actions = [...this.allocations()];
    if (this.stage === "output") {
      const outputLength = this.getBuffer("outputBuffer")?.size / 4;
      if (!(outputLength >= this.n)) throw new Error(`SuffixArray: outputBuffer must hold n = ${this.n} u32s`);
      actions.push(
        this.params.allocate("saParams", { n: this.n, m: 0, h: this.h, b: this.rankBits, initial: 0 }),
        this.kernel(this.outputWGSL, "output", [u32s("rank", "rankBuffer"), u32s("outputBuffer", "outputBuffer", "read_write")], this.n)
      );
      return actions;
    }
    return [...actions, ...this.roundActions()];
  }

  /* ---------------------------------------------------------------- driver */

  async execute(args = {}) {
    if (args.encoder) throw new Error("SuffixArray: execute() submits one command buffer per round, so it cannot take an encoder");
    if ((args.trials ?? 1) > 1) throw new Error("SuffixArray: trials must be 1 (time repeated executes instead)");
    const inputBytes = (args.inputBuffer ?? this.getBuffer("inputBuffer"))?.size;
    if (!(inputBytes >= this.capacity4)) {
      throw new Error(`SuffixArray: inputBuffer must hold n = ${this.n} bytes, padded to a multiple of 4 (${this.capacity4})`);
    }
    const timing = !!args.enableGPUTiming;
    this.timings = timing ? [] : undefined;
    const step = async (stage, m) => {
      this.stage = stage;
      this.m = m;
      await super.execute(args);
      if (timing) {
        const { gpuTotalTimeNS } = await this.getTimingResult();
        const labels = this.kernelLabels();
        this.timings.push({ stage, m, h: this.h, kernels: [gpuTotalTimeNS].flat().map((ns, i) => ({ label: labels[i], ns })) });
      }
    };
    this.h = INITIAL_SYMBOLS;
    this.rounds = 0;
    this.activeCounts = [this.n];
    await step("initial", this.n);
    let m = await this.readCount();
    while (m > 0) {
      if (this.rounds >= MAX_ROUNDS) throw new Error(`SuffixArray: not done after ${MAX_ROUNDS} rounds`);
      this.activeCounts.push(m);
      this.activeList = this.nextActiveList;
      await step("round", m);
      this.rounds++;
      this.h *= 2;
      m = await this.readCount();
    }
    await step("output", 0);
  }

  async readCount() {
    const buffer = this.getBuffer("countReadback").buffer.buffer;
    await buffer.mapAsync(GPUMapMode.READ);
    const m = new Uint32Array(buffer.getMappedRange(0, 4))[0];
    buffer.unmap();
    return m;
  }

  destroy() {
    for (const p of [this.initialSort, this.roundSort, this.groupScan, this.subgroupScan, this.keepScan]) p.destroy();
    super.destroy();
  }
}

const u32s = (name, buffer, access = "read") => ({ name, buffer, access, type: "array<u32>" });

/* ------------------------------------------------------------ traffic */

/** Which phase of the algorithm a kernel label belongs to. */
export function suffixArrayPhase(label = "") {
  if (label.startsWith("OneSweep")) return "sort";
  if (/scan/i.test(label)) return "scans";
  const m = label.match(/^SuffixArray (initial keys|pack|heads|update ranks|compact|output)/);
  return m ? m[1] : "other";
}

/** The DRAM traffic the algorithm needs, by phase, in bytes, given the active
 * counts of a run (activeCounts[0] = n for the initial round, then one per
 * doubling round). Each element's bytes are counted once (neighbor reads and
 * the scans' and sort's small spine traffic are ignored), so this is a lower
 * bound. `random` counts the 4-byte gathers and scatters (rank reads in pack,
 * rank writes in update and output): each moves a whole memory sector (32
 * bytes on most GPUs) when the rank array does not fit in cache.
 *
 * Per element of a round over m suffixes:
 *   keys     initial: read 1 text byte, write key 8 + payload 4
 *            pack: read active 4 + rank 4 + rank 4 (random), write 12
 *   sort     global histogram reads keys (8), then each of P passes reads and
 *            writes key + payload (24): 8 + 24P, P = ceil(key bits / 8)
 *   heads    read key 8, write 2 flags 8
 *   scans    3 scans, each read 4 + write 4
 *   update   read key 8, payload 4, 2 starts 8; write rank 4 (random), keep 4
 *   compact  read keep, offset, payload 12; write active 4 (at most)
 * and, once, output: read rank 4n, write SA 4n (random). */
export function suffixArrayTraffic({ n, activeCounts, rankBits = Math.ceil(Math.log2(n + 1)) }) {
  const passes = (bits) => Math.ceil(bits / 8);
  const bytes = {};
  const random = {};
  const add = (phase, b, r = 0) => {
    bytes[phase] = (bytes[phase] ?? 0) + b;
    random[phase] = (random[phase] ?? 0) + r;
  };
  activeCounts.forEach((m, round) => {
    const initial = round === 0;
    if (initial) add("initial keys", 13 * m);
    else add("pack", 24 * m, 2 * m);
    add("sort", (8 + 24 * passes(initial ? 9 * INITIAL_SYMBOLS : 2 * rankBits)) * m);
    add("heads", 16 * m);
    add("scans", 24 * m);
    add("update ranks", 28 * m, m);
    add("compact", 16 * m);
  });
  add("output", 8 * n, n);
  const total = Object.values(bytes).reduce((a, b) => a + b, 0);
  const totalRandom = Object.values(random).reduce((a, b) => a + b, 0);
  return { bytes, random, total, totalRandom };
}

/* ------------------------------------------------------------------ CPU */

/** The suffix array of text (a Uint8Array) on the CPU, by prefix doubling
 * with counting sorts: O(n log n). For tests and checking. */
export function suffixArrayReference(text) {
  const n = text.length;
  let sa = new Uint32Array(n);
  let rank = new Uint32Array(n);
  let tmp = new Uint32Array(n);
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => text[a] - text[b] || a - b);
  sa = Uint32Array.from(order);
  for (let j = 0; j < n; j++) rank[sa[j]] = j > 0 && text[sa[j]] === text[sa[j - 1]] ? rank[sa[j - 1]] : j;
  const key2 = (i, h) => (i + h < n ? rank[i + h] + 1 : 0);
  for (let h = 1; h < n; h *= 2) {
    /* sort by (rank[i], key2) : counting sort on key2, then stable on rank */
    const count = new Uint32Array(n + 1);
    for (let i = 0; i < n; i++) count[key2(i, h)]++;
    for (let k = 1; k <= n; k++) count[k] += count[k - 1];
    const bySecond = new Uint32Array(n);
    for (let i = n - 1; i >= 0; i--) bySecond[--count[key2(i, h)]] = i;
    const count1 = new Uint32Array(n + 1);
    for (let i = 0; i < n; i++) count1[rank[i] + 1]++;
    for (let k = 1; k <= n; k++) count1[k] += count1[k - 1];
    for (const i of bySecond) sa[count1[rank[i]]++] = i;
    tmp[sa[0]] = 0;
    let distinct = 1;
    for (let j = 1; j < n; j++) {
      const a = sa[j - 1], b = sa[j];
      const same = rank[a] === rank[b] && key2(a, h) === key2(b, h);
      tmp[b] = same ? tmp[a] : j;
      if (!same) distinct++;
    }
    [rank, tmp] = [tmp, rank];
    if (distinct === n) break;
  }
  return sa;
}

/** "" if sa is the suffix array of text, else a description of the first
 * problems. Burkhardt and Kärkkäinen's linear check: sa is a permutation, and
 * each adjacent pair is ordered by (first byte, rank of the next suffix). */
export function checkSuffixArray(text, sa) {
  const n = text.length;
  if (sa.length < n) return `\nsuffix array has ${sa.length} entries, expected ${n}`;
  const rank = new Int32Array(n + 1).fill(-1);
  for (let j = 0; j < n; j++) {
    const i = sa[j];
    if (i >= n || rank[i] !== -1) return `\nnot a permutation: SA[${j}] = ${i}`;
    rank[i] = j;
  }
  rank[n] = -1; /* the empty suffix sorts first */
  const problems = [];
  for (let j = 1; j < n && problems.length < 5; j++) {
    const a = sa[j - 1], b = sa[j];
    const ok = text[a] < text[b] || (text[a] === text[b] && rank[a + 1] < rank[b + 1]);
    if (!ok) problems.push(`SA[${j - 1}] = ${a} and SA[${j}] = ${b} are out of order`);
  }
  return problems.length ? "\n" + problems.join("\n") : "";
}
