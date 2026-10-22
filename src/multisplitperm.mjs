/**
 * MultisplitPerm: a uniformly random permutation of 0..n-1, by an unstable
 * multisplit into groups followed by a shuffle of each group in workgroup
 * memory ("two-level" permutation).
 *
 *   const perm = new MultisplitPerm({ device, n, seed });
 *   await perm.execute({ outputBuffer });     // outputBuffer: n u32s
 *
 * Algorithm. key(i) = mix_seed(i); group(i) = top G bits of key(i), with
 * G ~ log2(n / groupSize). Elements are bucketed by group with one or more
 * multisplit passes of at most 8 bits each, then a local finish shuffles each
 * group in workgroup memory.
 *
 *   pass A   1 segment.  digit = top bitsA bits.  Keys are computed from the
 *            index, so the histogram kernel reads nothing.
 *   pass B   2^bitsA segments (pass A's buckets); digit = next bitsB bits.
 *   ...      (more passes when G > 16, or when a split asks for them)
 *   finish   one workgroup per group (localFinishWGSL)
 *
 * Each pass is three steps:
 *   hist     workgroup (segment, slot) walks tiles slot, slot+M, ... of its
 *            segment and writes hist[(segment * RADIX + digit) * M + slot]
 *   scan     DLDFScan, exclusive, over the whole hist array (an
 *            ExecutePrimitive action). Segments are contiguous and in order,
 *            so the scan gives each (segment, digit, slot) its output offset.
 *   scatter  the same tile walk; per tile, rank by digit with workgroup
 *            atomics, stage the tile in workgroup memory grouped by digit,
 *            and write each digit's elements to consecutive addresses.
 * Segment starts for the next pass, and group starts for the finish, are read
 * straight from the previous scanned histogram at stride M (slot 0).
 *
 * Why it is unbiased. Ranks come from workgroup atomics, so the order inside
 * a bucket depends on hardware arrival order. That order never reaches the
 * output: the local finish draws its own randomness and gives every
 * arrangement of a group the same probability whatever order it arrives in.
 * With reproducible: true, the finish also sorts each sub-bucket first, so
 * the output is bit-identical on every GPU and in every run (reference()
 * computes it on the CPU).
 *
 * Memory traffic per element: one pass (G <= 8): 4 B (A writes) + 8 B
 * (finish) = 12 B. Each further pass adds 12 B.
 *
 * Constructor options:
 *   n                 permutation length (required)
 *   seed              32-bit seed, default 0x2545f491
 *   groupSize         mean group size for the local finish, default 2048
 *                     (reproducible: 1024, or 2048 when 1024 would need 16
 *                     group bits); groupBits = ceil(log2(n / groupSize))
 *   split             how the group bits are divided into passes: "auto"
 *                     (default), "even", "wide-first", "narrow-first" (two
 *                     passes), "passes:<k>", or explicit widths "5+5+5" /
 *                     [5, 5, 5] (each 1..8 bits)
 *   carry             "keys" (default): elements are hashed keys, inverted
 *                     at the end; "indices": elements are indices, keys are
 *                     recomputed per pass; "pairs" (Philox only): elements
 *                     are (index, random word)
 *   reproducible      default false (see above)
 *   rng               null (keyed hash; see random.mjs) or
 *                     { key: [lo, hi], rounds = 10 } for Philox4x32
 *   tileK             elements per invocation per tile, default 16 (8 for pairs)
 *   targetWorkgroups  workgroups per multisplit pass, default 1024
 *   localCap          override the local finish's capacity (tests: forces
 *                     the overflow path)
 *   phaseLimit        run only the first phaseLimit steps of `phases`
 *                     (per-phase timing); default all
 *
 * Buffers: outputBuffer (n u32, written). Internal: keysTemp (and keysTemp2
 * for pairs), per-pass histograms, parameters.
 *
 * Origin: written for the random-permutations paper (Owens et al.); see
 * tuneMultisplitPerm below for choosing groupSize and split per device.
 */
import { BasePrimitive, Kernel, AllocateBuffer, ExecutePrimitive } from "./primitive.mjs";
import { DLDFScan } from "./scandldf.mjs";
import { BinOpAdd } from "./binop.mjs";
import { randomBitsWGSL, randomBitsJS, pmMixJS } from "./random.mjs";

const WORKGROUP_SIZE = 256; /* the finish's 256 sub-buckets assume it */

/** Split G group bits into passes of at most 8 bits:
 *  "auto"          narrow-first from 13 bits up, else even (best on M1 Max)
 *  "even"          as evenly as possible (14 -> 7 + 7)
 *  "wide-first"    8 bits first          (14 -> 8 + 6)
 *  "narrow-first"  8 bits last           (14 -> 6 + 8)
 *  "passes:<k>"    k passes of nearly equal width (narrower ones first)
 *  "5+5+5", [5, 5, 5]  explicit widths, each 1..8, summing to G */
export function multisplitPermPassBits(groupBits, split = "even") {
  if (Array.isArray(split) || /^\d+(\+\d+)*$/.test(String(split))) {
    const bits = (Array.isArray(split) ? split : String(split).split("+")).map(Number);
    if (bits.reduce((a, b) => a + b, 0) !== groupBits || bits.some((b) => b < 1 || b > 8)) {
      throw new Error(
        `multisplitperm: split ${bits.join("+")} must be 1..8-bit passes summing to ${groupBits} group bits`
      );
    }
    return bits;
  }
  const m = /^passes:(\d+)$/.exec(String(split));
  if (m) {
    const k = Math.max(Math.ceil(groupBits / 8), Math.min(+m[1], groupBits));
    return Array.from({ length: k }, (_, i) => Math.floor((groupBits + i) / k));
  }
  if (groupBits <= 8) return [groupBits];
  if (groupBits > 16) return multisplitPermPassBits(groupBits, `passes:${Math.ceil(groupBits / 8)}`);
  if (split === "auto") split = groupBits >= 13 ? "narrow-first" : "even";
  const a = { even: Math.ceil(groupBits / 2), "wide-first": 8, "narrow-first": groupBits - 8 }[split];
  if (a === undefined) throw new Error(`multisplitperm: unknown split ${split}`);
  return [a, groupBits - a];
}

/** Group-size cap for the local finish: the mean group size plus 8 standard
 * deviations (a larger group takes the slow exact path, which for random
 * bucketing essentially never happens), rounded up to a multiple of 256,
 * limited by the device's workgroup memory. */
export function localFinishCap(device, meanGroupSize = 4096, bytesPerElement = 4) {
  const deviceMax =
    Math.floor((device.limits.maxComputeWorkgroupStorageSize - 3328) / bytesPerElement / 256) * 256;
  const wanted = Math.ceil((meanGroupSize + 8 * Math.sqrt(meanGroupSize)) / 256) * 256;
  return Math.min(deviceMax, wanted);
}

/** WGSL: wgShuffle(bu, bn, m, key, seed) uniformly shuffles wgShuf[0..m),
 * m <= cap, in place. Every element gets an independent random sub-bucket
 * (0..255) from (value, key, seed); sub-buckets are placed with workgroup
 * atomics (arrival order inside a sub-bucket does not matter) and each is
 * then Fisher-Yates-shuffled by one invocation with its own random stream.
 * Values must be distinct within one call; key must be distinct between calls
 * that share a seed. Call from uniform control flow.
 *
 * canonical: sort each sub-bucket before Fisher-Yates, so the result depends
 * only on the set of values and the random streams (bit-reproducible on any
 * GPU). aux: each value comes with a random word in wgShufAux whose low byte
 * is its sub-bucket.
 *
 * env: a primitive (for fnDeclarations.wgScan, binop, workgroupSize). */
export function wgShuffleWGSL(env, cap, { canonical = false, aux = false } = {}) {
  const W = WORKGROUP_SIZE;
  const rankSubBuckets = /* wgsl */ `
  var dest: array<u32, SHUF_K>;
  for (var k = 0u; k < SHUF_K; k++) {
    let idx = lidx + k * ${W}u;
    if (idx < m) {
      let a = shufStart[d[k]];
      let e = shufStart[d[k] + 1u];
      var r = 0u;
      for (var q = a; q < e; q++) { r += select(0u, 1u, wgShuf[q] < v[k]); }
      dest[k] = a + r;
    }
  }
  workgroupBarrier();
  for (var k = 0u; k < SHUF_K; k++) {
    if (lidx + k * ${W}u < m) { wgShuf[dest[k]] = v[k]; }
  }
  workgroupBarrier();`;
  return /* wgsl */ `
const SHUF_CAP = ${cap}u;
const SHUF_K = ${cap / W}u;
var<workgroup> wgShuf: array<u32, SHUF_CAP>;${
    aux
      ? `
var<workgroup> wgShufAux: array<u32, SHUF_CAP>;   /* sub-bucket = low byte */`
      : ""
  }
var<workgroup> shufCnt: array<atomic<u32>, 256>;
var<workgroup> shufStart: array<u32, 257>;
fn wgShuffle(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform,
             m: u32, key: u32, seed: u32) {
  let lidx = builtinsNonuniform.lidx;
  atomicStore(&shufCnt[lidx], 0u);
  workgroupBarrier();
  var v: array<u32, SHUF_K>;
  var d: array<u32, SHUF_K>;
  for (var k = 0u; k < SHUF_K; k++) {
    let idx = lidx + k * ${W}u;
    if (idx < m) {
      v[k] = wgShuf[idx];
      d[k] = ${aux ? "wgShufAux[idx]" : "pm_rand(v[k], key, seed)"} & 255u;
      atomicAdd(&shufCnt[d[k]], 1u);
    }
  }
  workgroupBarrier();
  var total = 0u;
  let excl = wgExclusiveScan(atomicLoad(&shufCnt[lidx]), &total, builtinsUniform, builtinsNonuniform);
  shufStart[lidx] = excl;
  if (lidx == 0u) { shufStart[256] = total; }
  atomicStore(&shufCnt[lidx], excl);
  workgroupBarrier();
  for (var k = 0u; k < SHUF_K; k++) {
    let idx = lidx + k * ${W}u;
    if (idx < m) { wgShuf[atomicAdd(&shufCnt[d[k]], 1u)] = v[k]; }
  }
  workgroupBarrier();${canonical ? rankSubBuckets : ""}
  {
    let a = shufStart[lidx];
    var k = shufStart[lidx + 1u] - a;
    var ctr = 0u;
    loop {
      if (k <= 1u) { break; }
      let r = pm_bounded(k, key * 256u + lidx, seed ^ 0x6c8e9cf5u, &ctr);
      let t = wgShuf[a + k - 1u];
      wgShuf[a + k - 1u] = wgShuf[a + r];
      wgShuf[a + r] = t;
      k -= 1u;
    }
  }
  workgroupBarrier();
}
`;
}

/** WGSL module for the two-level finish (entry point "main"). keys[] holds
 * elements arranged by group: group g occupies [start(g), start(g+1)),
 * start(g) = groupStart[g * stride], start(numGroups) = n; the order inside a
 * group is arbitrary. One workgroup per group loads it into workgroup memory,
 * wgShuffles it, and writes it to outp[] (unmixed when unmix: keys are hashed
 * indices). Groups larger than cap take an exact single-invocation path in
 * global memory. canonical, rng, pairs: as in MultisplitPerm.
 *
 * Bindings: keys (storage), outp (storage), groupStart (read-only-storage),
 * params (uniform: n, numGroups, stride, seed, seed2). */
export function localFinishWGSL(env, cap, { unmix = true, canonical = false, rng = null, pairs = false } = {}) {
  const W = WORKGROUP_SIZE;
  const sortGroup = /* wgsl */ `
      for (var i = 1u; i < m; i++) {   /* insertion sort: slow, but this path essentially never runs */
        let x = keys[s + i];
        var j = i;
        loop {
          if (j == 0u) { break; }
          let y = keys[s + j - 1u];
          if (sortKeyOf(y) <= sortKeyOf(x)) { break; }
          keys[s + j] = y;
          j -= 1u;
        }
        keys[s + j] = x;
      }`;
  return /* wgsl */ `
${env.fnDeclarations.enableSubgroupsIfAppropriate()}
${env.fnDeclarations.commonDefinitions()}
${env.binop.wgslfn}
${env.fnDeclarations.subgroupEmulation()}
${env.fnDeclarations.subgroupInclusiveOpScan()}
${env.fnDeclarations.wgScan()}
${randomBitsWGSL(rng)}
${wgShuffleWGSL(env, cap, { canonical, aux: pairs })}
alias Elem = ${pairs ? "vec2u" : "u32"};   /* ${pairs ? "(index, random word)" : "index or hashed key"} */
struct LF { n: u32, numGroups: u32, stride: u32, seed: u32, seed2: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read_write> keys: array<Elem>;    /* arranged by group */
@group(0) @binding(1) var<storage, read_write> outp: array<u32>;     /* permutation (indices) */
@group(0) @binding(2) var<storage, read> groupStart: array<u32>;
@group(0) @binding(3) var<uniform> lf: LF;
var<workgroup> wS: u32;
var<workgroup> wE: u32;
fn startOf(g: u32) -> u32 { return select(groupStart[g * lf.stride], lf.n, g >= lf.numGroups); }
fn sortKeyOf(e: Elem) -> u32 { return ${pairs ? "e.x" : "e"}; }
fn outValue(e: Elem) -> u32 { return ${pairs ? "e.x" : unmix ? "pm_unmix(e, lf.seed)" : "e"}; }
@compute @workgroup_size(${W})
fn main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${env.fnDeclarations.computeLinearizedGridParametersSplit({ workgroupSize: W })}
  let g = wgid;
  let lidx = builtinsNonuniform.lidx;
  if (g >= lf.numGroups) { return; }
  if (lidx == 0u) { wS = startOf(g); wE = startOf(g + 1u); }
  let s = workgroupUniformLoad(&wS);
  let m = workgroupUniformLoad(&wE) - s;
  if (m == 0u) { return; }
  if (m > SHUF_CAP) {
    /* rare overflow: exact but slow; stays inside this group's range */
    if (lidx == 0u) {${canonical ? sortGroup : ""}
      var ctr = 0u;
      var k = m;
      loop {
        if (k <= 1u) { break; }
        let r = pm_bounded(k, g ^ 0x80000000u, lf.seed2 ^ 0x2f8e33a1u, &ctr);
        let t = keys[s + k - 1u];
        keys[s + k - 1u] = keys[s + r];
        keys[s + r] = t;
        k -= 1u;
      }
    }
    storageBarrier();
    workgroupBarrier();
    for (var idx = lidx; idx < m; idx += ${W}u) { outp[s + idx] = outValue(keys[s + idx]); }
    return;
  }
  for (var idx = lidx; idx < m; idx += ${W}u) {
    let e = keys[s + idx];
    wgShuf[idx] = sortKeyOf(e);${
      pairs
        ? `
    wgShufAux[idx] = e.y;`
        : ""
    }
  }
  workgroupBarrier();
  wgShuffle(builtinsUniform, builtinsNonuniform, m, g, lf.seed2);
  for (var idx = lidx; idx < m; idx += ${W}u) { outp[s + idx] = ${pairs ? "wgShuf[idx]" : "outValue(wgShuf[idx])"}; }
}`;
}

export class MultisplitPerm extends BasePrimitive {
  constructor(args) {
    super(args);
    const { device } = this;
    if (!(Number.isInteger(this.n) && this.n > 0)) {
      throw new Error("MultisplitPerm: requires a positive integer n");
    }
    this.knownBuffers = ["outputBuffer"];
    this.datatype = "u32";
    this.workgroupSize = WORKGROUP_SIZE;
    this.binop = new BinOpAdd({ datatype: "u32" });
    this.seed = (this.seed ?? 0x2545f491) >>> 0;
    this.split ??= "auto";
    this.carry ??= "keys";
    this.reproducible ??= false;
    this.rng ??= null;
    this.targetWorkgroups ??= 1024;
    if (this.rng && this.carry === "keys") this.carry = "indices"; /* Philox draws are not invertible */
    if (this.carry === "pairs" && !this.rng) {
      throw new Error('MultisplitPerm: carry "pairs" needs rng (Philox)');
    }
    const n = this.n;
    this.groupSize ??= this.reproducible && Math.ceil(Math.log2(n / 1024)) < 16 ? 1024 : 2048;
    this.tileK ??= this.carry === "pairs" ? 8 : 16; /* pairs double the workgroup memory per staged element */
    this.rngJS = randomBitsJS(this.rng);
    if (!("label" in args)) {
      this.label += this.makeParamString(["carry", "reproducible", "useSubgroups"]);
    }

    /* group bits. 9 bits would force a second pass for a single bit: use 8
     * (groups up to twice groupSize) instead; one pass was 1.3x faster at 2^20 */
    const bits = Math.max(1, Math.ceil(Math.log2(n / this.groupSize)));
    let groupBits = bits === 9 ? 8 : bits;
    /* the expected largest group (mean + 8 sd) must fit the local finish's
     * workgroup memory, or every group takes the slow overflow path */
    this.elementBytes = this.carry === "pairs" ? 8 : 4;
    const needed = (G) => {
      const mean = n / 2 ** G;
      return Math.ceil((mean + 8 * Math.sqrt(mean)) / 256) * 256;
    };
    while (
      this.localCap === undefined &&
      localFinishCap(device, n / 2 ** groupBits, this.elementBytes) < needed(groupBits)
    ) {
      groupBits++;
    }
    this.groupBits = groupBits;
    this.passBits = multisplitPermPassBits(groupBits, this.split);

    /* plan the passes: segments, slots and histogram size of each */
    const tile = this.tileK * WORKGROUP_SIZE;
    let numSegs = 1;
    let bitsDone = 0;
    this.plan = this.passBits.map((bits) => {
      const tiles = Math.ceil(n / numSegs / tile);
      const slots =
        Math.ceil(Math.max(1, Math.min(Math.ceil(this.targetWorkgroups / numSegs), tiles)) / 4) * 4;
      const p = {
        bits,
        radix: 1 << bits,
        shift: 32 - bitsDone - bits,
        numSegs,
        slots,
        histLength: numSegs * (1 << bits) * slots /* a multiple of 4, as DLDFScan's vec4 loads need */,
      };
      numSegs *= p.radix;
      bitsDone += bits;
      return p;
    });
    this.numGroups = numSegs;
    this.localCap ??= localFinishCap(device, n / numSegs, this.elementBytes);

    /* one scan primitive per pass, created once so that every compute()
     * returns the same objects */
    this.scans = this.plan.map(
      () =>
        new DLDFScan({
          device,
          adapter: this.adapter,
          datatype: "u32",
          type: "exclusive",
          binop: new BinOpAdd({ datatype: "u32" }),
          ...(this.disableSubgroups && { disableSubgroups: true }),
        })
    );

    this.phases = this.plan
      .flatMap((p, i) => ["hist", "scan", "scatter"].map((s) => `${"ABCDEFGH"[i]} ${s}`))
      .concat(["finish"]);
    this.config = {
      passBits: this.passBits,
      slots: this.plan.map((p) => p.slots),
      groupSize: this.groupSize,
      localCap: this.localCap,
      carry: this.carry,
      reproducible: this.reproducible,
      rng: this.rng ? `philox-${this.rng.rounds ?? 10}` : "hash32",
      tileK: this.tileK,
      targetWorkgroups: this.targetWorkgroups,
    };
  }

  /** hist_main and scatter_main for one multisplit pass */
  multisplitWGSL({ radixBits, shift, fromIndex }) {
    const W = WORKGROUP_SIZE;
    const K = this.tileK;
    const carry = this.carry;
    const rng = this.rng;
    const RADIX = 1 << radixBits;
    const pairs = carry === "pairs";
    const elemAt = !fromIndex
      ? "keysIn[idx]"
      : carry === "keys"
        ? "pm_mix(idx, ms.seed)"
        : pairs
          ? "vec2u(idx, pm_elem(idx))"
          : "idx";
    const keyOf = carry === "keys" ? "e" : pairs ? "e.y" : rng ? "pm_elem(e)" : "pm_mix(e, ms.seed)";
    /* digit of the element staged at j: a key's digit is cheap to recompute;
     * for an index, find the last digit whose run starts at or before j */
    const stagedDigit =
      carry === "keys"
        ? /* wgsl */ `
  return digitOf(stage[j]);`
        : /* wgsl */ `
  var lo = 0u;
  var hi = RADIX;
  loop {
    if (hi - lo <= 1u) { break; }
    let mid = (lo + hi) / 2u;
    if (tileStart[mid] <= j) { lo = mid; } else { hi = mid; }
  }
  return lo;`;
    return /* wgsl */ `
${this.fnDeclarations.enableSubgroupsIfAppropriate()}
${this.fnDeclarations.commonDefinitions()}
${this.binop.wgslfn}
${this.fnDeclarations.subgroupEmulation()}
${this.fnDeclarations.subgroupInclusiveOpScan()}
${this.fnDeclarations.wgScan()}
${randomBitsWGSL(rng)}
struct MS { n: u32, seed: u32, numSegs: u32, slots: u32, segStride: u32, p0: u32, p1: u32, p2: u32 };
alias Elem = ${pairs ? "vec2u" : "u32"};
@group(0) @binding(0) var<storage, read> keysIn: array<Elem>;        /* unused when keys come from indices */
@group(0) @binding(1) var<storage, read_write> keysOut: array<Elem>; /* scatter only */
@group(0) @binding(2) var<storage, read_write> hist: array<u32>;     /* [seg][digit][slot]: counts (hist), scanned offsets (scatter) */
@group(0) @binding(3) var<storage, read> segStart: array<u32>;       /* segment s begins at segStart[s * segStride] */
@group(0) @binding(4) var<uniform> ms: MS;
const RADIX = ${RADIX}u;
const SHIFT = ${shift}u;
const K = ${K}u;
const TILE = ${K * W}u;

fn elemAt(idx: u32) -> Elem { return ${elemAt}; }
fn keyOf(e: Elem) -> u32 { return ${keyOf}; }
fn digitOf(e: Elem) -> u32 { return (keyOf(e) >> SHIFT) & (RADIX - 1u); }
fn histIndex(seg: u32, d: u32, slot: u32) -> u32 { return (seg * RADIX + d) * ms.slots + slot; }

/* This workgroup's segment and slot, and the segment's range, broadcast so
 * that loop bounds are workgroup-uniform. */
var<workgroup> wBegin: u32;
var<workgroup> wEnd: u32;
struct Work { seg: u32, slot: u32, begin: u32, end: u32 };
fn myWork(wid: u32, lidx: u32) -> Work {
  let seg = wid / ms.slots;
  if (lidx == 0u) {
    wBegin = segStart[seg * ms.segStride];
    wEnd = select(segStart[(seg + 1u) * ms.segStride], ms.n, seg + 1u >= ms.numSegs);
  }
  return Work(seg, wid % ms.slots, workgroupUniformLoad(&wBegin), workgroupUniformLoad(&wEnd));
}

var<workgroup> cnt: array<atomic<u32>, RADIX>;

@compute @workgroup_size(${W})
fn hist_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let lidx = builtinsNonuniform.lidx;
  if (wgid >= ms.numSegs * ms.slots) { return; }
  let w = myWork(wgid, lidx);
  for (var d = lidx; d < RADIX; d += ${W}u) { atomicStore(&cnt[d], 0u); }
  workgroupBarrier();
  for (var base = w.begin + w.slot * TILE; base < w.end; base += ms.slots * TILE) {
    for (var k = 0u; k < K; k++) {
      let idx = base + lidx + k * ${W}u;
      if (idx < w.end) { atomicAdd(&cnt[digitOf(elemAt(idx))], 1u); }
    }
  }
  workgroupBarrier();
  for (var d = lidx; d < RADIX; d += ${W}u) { hist[histIndex(w.seg, d, w.slot)] = atomicLoad(&cnt[d]); }
}

var<workgroup> stage: array<Elem, TILE>;
var<workgroup> tileStart: array<u32, RADIX>;  /* exclusive scan of this tile's digit counts */
var<workgroup> cursor: array<u32, RADIX>;     /* next global output position for each digit */
fn stagedDigit(j: u32) -> u32 {${stagedDigit}
}

@compute @workgroup_size(${W})
fn scatter_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let lidx = builtinsNonuniform.lidx;
  if (wgid >= ms.numSegs * ms.slots) { return; }
  let w = myWork(wgid, lidx);
  for (var d = lidx; d < RADIX; d += ${W}u) { cursor[d] = hist[histIndex(w.seg, d, w.slot)]; }
  for (var base = w.begin + w.slot * TILE; base < w.end; base += ms.slots * TILE) {
    let tileLen = min(TILE, w.end - base);
    /* 1. load; rank within digit by workgroup atomics (any order is fine) */
    for (var d = lidx; d < RADIX; d += ${W}u) { atomicStore(&cnt[d], 0u); }
    workgroupBarrier();
    var elem: array<Elem, K>;
    var digit: array<u32, K>;
    var rank: array<u32, K>;
    for (var k = 0u; k < K; k++) {
      let j = lidx + k * ${W}u;
      if (j < tileLen) {
        elem[k] = elemAt(base + j);
        digit[k] = digitOf(elem[k]);
        rank[k] = atomicAdd(&cnt[digit[k]], 1u);
      }
    }
    workgroupBarrier();
    /* 2. tile offsets per digit */
    var count = 0u;
    if (lidx < RADIX) { count = atomicLoad(&cnt[lidx]); }
    var tileTotal = 0u;
    let start = wgExclusiveScan(count, &tileTotal, builtinsUniform, builtinsNonuniform);
    if (lidx < RADIX) { tileStart[lidx] = start; }
    workgroupBarrier();
    /* 3. stage the tile grouped by digit */
    for (var k = 0u; k < K; k++) {
      let j = lidx + k * ${W}u;
      if (j < tileLen) { stage[tileStart[digit[k]] + rank[k]] = elem[k]; }
    }
    workgroupBarrier();
    /* 4. write out: consecutive staged elements of a digit go to consecutive addresses */
    for (var k = 0u; k < K; k++) {
      let j = lidx + k * ${W}u;
      if (j < tileLen) {
        let d = stagedDigit(j);
        keysOut[cursor[d] + j - tileStart[d]] = stage[j];
      }
    }
    workgroupBarrier();
    if (lidx < RADIX) { cursor[lidx] += count; }
    workgroupBarrier();
  }
}`;
  }

  /** the buffer the element at pass `i` (0..P-1) is written to; the finish
   * (i = P) writes outputBuffer. Four-byte elements ping-pong between
   * keysTemp and outputBuffer; pairs need two eight-byte buffers. */
  keysBufferFor(i) {
    const P = this.plan.length;
    if (i === P) return "outputBuffer";
    if (this.carry === "pairs") return (P - 1 - i) % 2 === 0 ? "keysTemp" : "keysTemp2";
    return (P - i) % 2 === 0 ? "outputBuffer" : "keysTemp";
  }

  compute() {
    const outputLength = this.getBuffer("outputBuffer")?.size / 4;
    if (!(outputLength >= this.n)) {
      throw new Error(`MultisplitPerm: outputBuffer must hold n = ${this.n} u32s`);
    }
    const n = this.n;
    const P = this.plan.length;
    const uniform = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    const keysBytes = Math.max(16, n * this.elementBytes);
    const actions = [
      new AllocateBuffer({ label: "keysTemp", size: keysBytes, clearBufferOnReuse: false }),
      ...(this.carry === "pairs"
        ? [new AllocateBuffer({ label: "keysTemp2", size: keysBytes, clearBufferOnReuse: false })]
        : []),
      /* pass A's single segment starts at 0; also stands in for pass A's unused
       * keysIn. Never written (bound read-only), so no clear on reuse. */
      new AllocateBuffer({ label: "zero", size: 16, clearBufferOnReuse: false }),
    ];
    const bufferTypes = [["read-only-storage", "storage", "storage", "read-only-storage", "uniform"]];
    let step = 0;
    const enabled = () => step++ < (this.phaseLimit ?? Infinity);
    this.plan.forEach((p, i) => {
      const name = "ABCDEFGH"[i];
      const prev = this.plan[i - 1];
      const kernel = () => this.multisplitWGSL({ radixBits: p.bits, shift: p.shift, fromIndex: i === 0 });
      const keysIn = i === 0 ? "zero" : this.keysBufferFor(i - 1);
      const keysOut = this.keysBufferFor(i);
      const segStarts = prev ? `offsets${i - 1}` : "zero";
      const dispatch = () => this.getSimpleDispatchGeometry({ workgroupCount: p.numSegs * p.slots });
      actions.push(
        new AllocateBuffer({ label: `counts${i}`, size: p.histLength * 4, clearBufferOnReuse: false }),
        new AllocateBuffer({ label: `offsets${i}`, size: p.histLength * 4, clearBufferOnReuse: false }),
        new AllocateBuffer({
          label: `params${i}`,
          size: 32,
          usage: uniform,
          populateWith: new Uint32Array([n, this.seed, p.numSegs, p.slots, prev ? prev.slots : 0, 0, 0, 0]),
        }),
        new Kernel({
          kernel,
          entryPoint: "hist_main",
          label: `MultisplitPerm ${name} hist (${p.bits} bits)`,
          bufferTypes,
          bindings: [[keysIn, keysOut, `counts${i}`, segStarts, `params${i}`]],
          getDispatchGeometry: dispatch,
          enable: enabled(),
        }),
        new ExecutePrimitive({
          primitive: this.scans[i],
          buffers: { inputBuffer: `counts${i}`, outputBuffer: `offsets${i}` },
          label: `MultisplitPerm ${name} scan`,
          enable: enabled(),
        }),
        new Kernel({
          kernel,
          entryPoint: "scatter_main",
          label: `MultisplitPerm ${name} scatter (${p.bits} bits)`,
          bufferTypes,
          bindings: [[keysIn, keysOut, `offsets${i}`, segStarts, `params${i}`]],
          getDispatchGeometry: dispatch,
          enable: enabled(),
        })
      );
    });
    const last = this.plan[P - 1];
    actions.push(
      new AllocateBuffer({
        label: "finishParams",
        size: 32,
        usage: uniform,
        populateWith: new Uint32Array([n, this.numGroups, last.slots, this.seed, (this.seed ^ 0x5bd1e995) >>> 0, 0, 0, 0]),
      }),
      new Kernel({
        kernel: () =>
          localFinishWGSL(this, this.localCap, {
            unmix: this.carry === "keys",
            canonical: this.reproducible,
            rng: this.rng,
            pairs: this.carry === "pairs",
          }),
        entryPoint: "main",
        label: `MultisplitPerm finish (${this.numGroups} groups)`,
        bufferTypes: [["storage", "storage", "read-only-storage", "uniform"]],
        bindings: [[this.keysBufferFor(P - 1), "outputBuffer", `offsets${P - 1}`, "finishParams"]],
        getDispatchGeometry: () => this.getSimpleDispatchGeometry({ workgroupCount: this.numGroups }),
        enable: enabled(),
      })
    );
    return actions;
  }

  /** bytes read and written by the whole permutation (for bandwidth plots) */
  get bytesTransferred() {
    /* pass A writes n elements; each later pass reads and writes n; the
     * finish reads and writes n (its output is always 4 bytes per element) */
    const e = this.elementBytes;
    return this.n * (e + (this.plan.length - 1) * 2 * e + e + 4);
  }

  keyOf(i) {
    return this.rng ? this.rngJS.elem(i) : pmMixJS(i, this.seed);
  }
  groupOf(i) {
    return this.keyOf(i) >>> (32 - this.groupBits);
  }

  /** The permutation a reproducible MultisplitPerm writes, computed on the
   * CPU by mirroring the local finish: per group (in order), each element
   * gets a sub-bucket from (value, group, seed2); each sub-bucket is sorted by
   * value and Fisher-Yates-shuffled with its own stream; sub-buckets are
   * concatenated in order. */
  reference() {
    if (!this.reproducible) throw new Error("reference() needs reproducible: true");
    const { n, seed, groupBits } = this;
    const cap = this.localCap;
    const seed2 = (seed ^ 0x5bd1e995) >>> 0;
    const carryKeys = this.carry === "keys";
    const R = this.rngJS;
    const groups = Array.from({ length: this.numGroups }, () => []);
    for (let i = 0; i < n; i++) {
      const key = this.keyOf(i);
      groups[key >>> (32 - groupBits)].push(carryKeys ? key : i);
    }
    const toIndex = carryKeys ? new Map() : null;
    if (carryKeys) for (let i = 0; i < n; i++) toIndex.set(pmMixJS(i, seed), i);
    const out = new Uint32Array(n);
    const fisherYates = (a, key, s) => {
      const ctr = { value: 0 };
      for (let k = a.length; k > 1; k--) {
        const r = R.bounded(k, key >>> 0, s >>> 0, ctr);
        const t = a[k - 1];
        a[k - 1] = a[r];
        a[r] = t;
      }
    };
    let pos = 0;
    groups.forEach((vals, g) => {
      let shuffled;
      if (vals.length > cap) {
        /* overflow path: sort, then one Fisher-Yates over the group */
        shuffled = Uint32Array.from(vals).sort();
        fisherYates(shuffled, g ^ 0x80000000, seed2 ^ 0x2f8e33a1);
      } else {
        const sub = Array.from({ length: 256 }, () => []);
        for (const v of vals) sub[(this.carry === "pairs" ? R.elem(v) : R.rand(v, g, seed2)) & 255].push(v);
        shuffled = [];
        sub.forEach((b, d) => {
          const a = Uint32Array.from(b).sort();
          fisherYates(a, g * 256 + d, seed2 ^ 0x6c8e9cf5);
          shuffled.push(...a);
        });
      }
      for (const v of shuffled) out[pos++] = carryKeys ? toIndex.get(v) : v;
    });
    return out;
  }

  /** Checks that the output is a permutation of 0..n-1 whose positions are
   * in group order and, when reproducible, that it equals reference().
   * Returns "" on success, else a description of the first problems. */
  validate = (args = {}) => {
    const out = args.outputBuffer?.cpuBuffer ?? args.outputBuffer ?? this.getBuffer("outputBuffer").cpuBuffer;
    const n = this.n;
    const seen = new Uint8Array(n);
    const problems = [];
    let prevGroup = -1;
    for (let p = 0; p < n && problems.length < 5; p++) {
      const v = out[p];
      if (v >= n) problems.push(`position ${p}: ${v} is out of range`);
      else if (seen[v]) problems.push(`position ${p}: ${v} appears twice`);
      else {
        seen[v] = 1;
        const g = this.groupOf(v);
        if (g < prevGroup) problems.push(`position ${p}: group ${g} after group ${prevGroup}`);
        prevGroup = g;
      }
    }
    if (problems.length === 0 && this.reproducible) {
      const ref = this.reference();
      for (let p = 0; p < n && problems.length < 5; p++) {
        if (out[p] !== ref[p]) problems.push(`position ${p}: expected ${ref[p]} (reference), saw ${out[p]}`);
      }
    }
    return problems.length ? "\n" + problems.join("\n") : "";
  };

  destroy() {
    super.destroy();
    for (const scan of this.scans) scan.destroy();
  }
}

/**
 * On-device tuning: the fastest group size and pass split for one problem.
 *
 *   const best = await tuneMultisplitPerm({ device, n, reproducible, rng, carry });
 *   const perm = new MultisplitPerm({ device, n, seed, ...best.options });
 *
 * The best configuration depends on the GPU: an NVIDIA V100 wants at most
 * about 6 bits per pass at large n, while an Apple M1 Max and an NVIDIA A100
 * prefer two passes. Each candidate is timed for a few tens of milliseconds;
 * results are cached per device and problem.
 *
 *   candidates   [{ groupSize, split }, ...]; default: groups of 2,048 and
 *                4,096 (1,024 and 2,048 when reproducible) x splits "auto",
 *                "even", "wide-first", "passes:3", duplicates removed
 *   trialMs      time per measurement (median of three), default 30
 * Returns { options: { groupSize, split }, passBits, geps, tried: [...] }.
 *
 * Reproducible output depends on the group size (not the split). Tuning the
 * group size per device would break bit-identical results across devices, so
 * for reproducible runs pass candidates with a single groupSize.
 */
const TUNE_CACHE = new WeakMap();
export async function tuneMultisplitPerm({
  device,
  adapter,
  n,
  reproducible = false,
  rng = null,
  carry,
  candidates,
  trialMs = 30,
  disableSubgroups,
}) {
  const key = JSON.stringify({ n, reproducible, rng, carry, candidates, trialMs, disableSubgroups });
  let cache = TUNE_CACHE.get(device);
  if (!cache) TUNE_CACHE.set(device, (cache = new Map()));
  if (cache.has(key)) return cache.get(key);
  const groupSizes = reproducible ? [1024, 2048] : [2048, 4096];
  candidates ??= groupSizes.flatMap((groupSize) =>
    ["auto", "even", "wide-first", "passes:3"].map((split) => ({ groupSize, split }))
  );
  const outputBuffer = device.createBuffer({
    size: Math.max(16, n * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const tried = [];
  const seen = new Set();
  let best = null;
  for (const c of candidates) {
    let perm;
    try {
      perm = new MultisplitPerm({
        device, adapter, n, seed: 0x7e57, reproducible, rng, carry, disableSubgroups, ...c,
      });
    } catch {
      continue; /* e.g. a split that does not fit this n */
    }
    const sig = `${perm.passBits.join("+")}/${perm.groupSize}/${perm.numGroups}`;
    if (seen.has(sig)) continue;
    seen.add(sig);
    const ms = await timePrimitive(device, perm, { outputBuffer }, trialMs);
    const row = { ...c, passBits: perm.passBits.join("+"), ms: +ms.toFixed(4), geps: +(n / (ms * 1e6)).toFixed(3) };
    tried.push(row);
    if (!best || row.ms < best.ms) best = row;
    perm.destroy();
  }
  outputBuffer.destroy();
  const result = { options: { groupSize: best.groupSize, split: best.split }, passBits: best.passBits, geps: best.geps, tried };
  cache.set(key, result);
  return result;
}

/** Median of three trials, each of enough back-to-back executions in one
 * command buffer to take about trialMs; milliseconds per execution. */
async function timePrimitive(device, primitive, buffers, trialMs) {
  const run = async (reps) => {
    const encoder = device.createCommandEncoder();
    for (let r = 0; r < reps; r++) await primitive.execute({ ...buffers, encoder });
    const commandBuffer = encoder.finish();
    await device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    device.queue.submit([commandBuffer]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / reps;
  };
  await run(1);
  const reps = Math.max(2, Math.min(200, Math.ceil(trialMs / Math.max(await run(2), 1e-3))));
  const ts = [await run(reps), await run(reps), await run(reps)].sort((a, b) => a - b);
  return ts[1];
}
