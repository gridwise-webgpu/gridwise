/**
 * Multisplit: group elements by bucket. Each element's bucket comes from a
 * bin function (bins.mjs) of its key; the output holds bucket 0's elements,
 * then bucket 1's, and so on. Unstable: the order inside a bucket is not
 * the input order (it depends on the hardware). Up to 256 buckets per pass.
 *
 *   const ms = new Multisplit({ device, bins: new RadixBins({ shift: 24, bits: 8 }), type: "keyvalue" });
 *   await ms.execute({ keysIn, keysOut, valuesIn, valuesOut, bucketOffsets });
 *
 * Buffers (u32 words): keysIn, keysOut; with type "keyvalue", valuesIn and
 * valuesOut; with values: "index", valuesOut only, receiving each element's
 * input index (an argsort by bucket). bucketOffsets (optional, numBins + 1
 * u32s): where each bucket starts in the output, then the length.
 * Keys are u32 words read as the bins' datatype (u32, i32 or f32). Every key
 * must fall in a bin: an element whose bin is BIN_NONE is not supported.
 *
 * Options: bins (required), type ("keysonly", default, or "keyvalue"),
 * values ("index": see above), length (multisplit only the first `length`
 * elements; may change between executes), tileK (elements per invocation
 * per tile, default 16 keys only, 8 with values), targetWorkgroups (default
 * 1024).
 *
 * Algorithm (shared with MultisplitPerm, which runs it in several passes
 * over segments): workgroups each take a slot, a strided set of tiles.
 *   hist     each workgroup counts its tiles' buckets: hist[bucket][slot]
 *   scan     DLDFScan (exclusive) over hist gives every (bucket, slot) its
 *            output offset
 *   scatter  per tile: rank within bucket by workgroup atomics, stage the
 *            tile in workgroup memory grouped by bucket, then write each
 *            bucket's run to consecutive addresses
 * Memory traffic per element: read the key (hist), read and write the
 * element (scatter): 12 bytes keys only, 20 bytes key-value.
 *
 * A use: making random writes local. To apply writes A[dest[i]] = v[i]
 * whose destinations are scattered over a large array, first multisplit the
 * (dest, v) pairs by the destination's high bits, then apply them in output
 * order: each bucket's writes fall in a 1/numBins slice of A, so they stay
 * in cache and each cache line is written back once (Büren et al., HPDC
 * 2019, for suffix arrays).
 */
import { BasePrimitive, Kernel, AllocateBuffer, ExecutePrimitive, UniformStruct } from "./primitive.mjs";
import { DLDFScan } from "./scandldf.mjs";
import { BinOpAdd } from "./binop.mjs";

const WORKGROUP_SIZE = 256;

/** The WGSL of one multisplit pass: hist_main and scatter_main.
 *
 * env: the primitive (its fnDeclarations, and binop BinOpAdd for wgScan).
 * definitions: the caller's WGSL, placed after the helpers' directives
 * (enable subgroups must come first), providing:
 *   alias Elem             the element type staged in workgroup memory
 *   fn elemAt(idx) -> Elem the element at input index idx
 *   fn digitOf(e) -> u32   its bucket, in [0, radix)
 *   fn histDigitAt(idx)    the bucket of the element at idx, for the
 *                          histogram (may read less than elemAt)
 *   fn writeOut(pos, e)    write element e to output position pos
 * and bind hist (array<u32>), segStart (array<u32>) and a uniform `ms` with
 * n, numSegs, slots, segStride. Segment s (one per bucket of an earlier pass)
 * spans [segStart[s * segStride], next segment's start or n).
 *
 * radix: buckets (at most the workgroup size, 256); K: elements per
 * invocation per tile; stagedDigit: "recompute" (digitOf(stage[j]): cheap
 * when the bucket comes from the element) or "search" (binary search of the
 * tile's bucket starts: for elements whose bucket is expensive to compute). */
export function multisplitPassWGSL(env, { radix, K, stagedDigit = "recompute", definitions }) {
  const W = WORKGROUP_SIZE;
  const findStaged =
    stagedDigit === "recompute"
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
${env.fnDeclarations.include(
  "wgScan",
  ["wgHistogram", { name: "cnt", numBins: radix, workgroupSize: W }] /* the histogram, then each tile's ranks */,
  "computeLinearizedGridParametersSplit"
)}
${definitions}
const RADIX = ${radix}u;
const K = ${K}u;
const TILE = ${K * W}u;

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

@compute @workgroup_size(${W})
fn hist_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${env.fnDeclarations.computeLinearizedGridParametersSplit()}
  let lidx = builtinsNonuniform.lidx;
  if (wgid >= ms.numSegs * ms.slots) { return; }
  let w = myWork(wgid, lidx);
  cntClear(lidx);
  workgroupBarrier();
  for (var base = w.begin + w.slot * TILE; base < w.end; base += ms.slots * TILE) {
    for (var k = 0u; k < K; k++) {
      let idx = base + lidx + k * ${W}u;
      if (idx < w.end) { cntAdd(histDigitAt(idx), lidx); }
    }
  }
  workgroupBarrier();
  for (var d = lidx; d < RADIX; d += ${W}u) { hist[histIndex(w.seg, d, w.slot)] = cntCount(d); }
}

var<workgroup> stage: array<Elem, TILE>;
var<workgroup> tileStart: array<u32, RADIX>;  /* exclusive scan of this tile's digit counts */
var<workgroup> cursor: array<u32, RADIX>;     /* next global output position for each digit */
fn stagedDigit(j: u32) -> u32 {${findStaged}
}

@compute @workgroup_size(${W})
fn scatter_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${env.fnDeclarations.computeLinearizedGridParametersSplit()}
  let lidx = builtinsNonuniform.lidx;
  if (wgid >= ms.numSegs * ms.slots) { return; }
  let w = myWork(wgid, lidx);
  for (var d = lidx; d < RADIX; d += ${W}u) { cursor[d] = hist[histIndex(w.seg, d, w.slot)]; }
  for (var base = w.begin + w.slot * TILE; base < w.end; base += ms.slots * TILE) {
    let tileLen = min(TILE, w.end - base);
    /* 1. load; rank within digit by workgroup atomics (any order is fine) */
    cntClear(lidx);
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
    if (lidx < RADIX) { count = cntCount(lidx); }
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
        writeOut(cursor[d] + j - tileStart[d], stage[j]);
      }
    }
    workgroupBarrier();
    if (lidx < RADIX) { cursor[lidx] += count; }
    workgroupBarrier();
  }
}`;
}

/** Slots per segment for a pass over n elements in numSegs segments: about
 * targetWorkgroups workgroups in all, no more than there are tiles, rounded
 * up to a multiple of 4 (so the histogram is whole vec4s for DLDFScan). */
export function multisplitSlots(n, numSegs, tile, targetWorkgroups) {
  const tiles = Math.ceil(n / numSegs / tile);
  return Math.ceil(Math.max(1, Math.min(Math.ceil(targetWorkgroups / numSegs), tiles)) / 4) * 4;
}

export class Multisplit extends BasePrimitive {
  constructor(args) {
    super(args);
    if (!this.bins) throw new Error("Multisplit: requires bins (see bins.mjs)");
    if (!(this.bins.numBins >= 1 && this.bins.numBins <= WORKGROUP_SIZE)) {
      throw new Error(`Multisplit: at most ${WORKGROUP_SIZE} buckets per pass (bins have ${this.bins.numBins})`);
    }
    if (this.bins.needsLevels) throw new Error("Multisplit: RangeBins are not supported yet");
    this.type ??= "keysonly";
    if (!["keysonly", "keyvalue"].includes(this.type)) throw new Error(`Multisplit: unknown type ${this.type}`);
    if (this.values !== undefined && this.values !== "index") throw new Error('Multisplit: values must be "index" or omitted');
    this.pairs = this.type === "keyvalue" || this.values === "index";
    this.tileK ??= this.pairs ? 8 : 16;
    this.targetWorkgroups ??= 1024;
    this.knownBuffers = ["keysIn", "keysOut", "valuesIn", "valuesOut", "bucketOffsets"];
    this.datatype = "u32";
    this.workgroupSize = WORKGROUP_SIZE;
    this.binop = new BinOpAdd({ datatype: "u32" }); /* wgScan's */
    this.numBins = this.bins.numBins;
    this.params = new UniformStruct("MS", { n: "u32", numSegs: "u32", slots: "u32", segStride: "u32" });
    this.scan = new DLDFScan({
      device: this.device,
      adapter: this.adapter,
      datatype: "u32",
      type: "exclusive",
      binop: new BinOpAdd({ datatype: "u32" }),
      ...(this.disableSubgroups && { disableSubgroups: true }),
    });
    if (!("label" in args)) this.label += this.makeParamString(["type", "useSubgroups"]);
  }

  /** hist_main, scatter_main, and offsets_main (bucket starts) */
  multisplitWGSL = () => {
    const T = this.bins.datatype;
    const key = (word) => (T === "u32" ? word : `bitcast<${T}>(${word})`);
    const elemAt =
      this.values === "index" ? "vec2u(keysIn[idx], idx)" : this.type === "keyvalue" ? "vec2u(keysIn[idx], valuesIn[idx])" : "keysIn[idx]";
    const keyWord = this.pairs ? "e.x" : "e";
    const writeOut = this.pairs ? "keysOut[pos] = e.x; valuesOut[pos] = e.y;" : "keysOut[pos] = e;";
    const definitions = /* wgsl */ `
alias Elem = ${this.pairs ? "vec2u" : "u32"};
${this.bins.wgsl("binOf")}
fn elemAt(idx: u32) -> Elem { return ${elemAt}; }
fn digitOf(e: Elem) -> u32 { return binOf(${key(keyWord)}); }
fn histDigitAt(idx: u32) -> u32 { return binOf(${key("keysIn[idx]")}); }
fn writeOut(pos: u32, e: Elem) { ${writeOut} }`;
    return /* wgsl */ `
${multisplitPassWGSL(this, { radix: this.numBins, K: this.tileK, definitions })}
${this.hasBuffer("bucketOffsets") ? this.offsetsWGSL : ""}`;
  };

  offsetsWGSL = /* wgsl */ `
/* bucketOffsets[d] = start of bucket d (slot 0's offset); [numBins] = n */
@compute @workgroup_size(${WORKGROUP_SIZE})
fn offsets_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  let d = builtinsNonuniform.lidx;
  if (d < RADIX) { bucketOffsets[d] = hist[d * ms.slots]; }
  if (d == 0u) { bucketOffsets[RADIX] = ms.n; }
}`;

  compute() {
    const capacity = this.getBuffer("keysIn")?.size / 4;
    if (!(capacity >= 1)) throw new Error("Multisplit: needs a keysIn buffer");
    if (this.length !== undefined && !(Number.isInteger(this.length) && this.length >= 1 && this.length <= capacity)) {
      throw new Error(`Multisplit: length ${this.length} must be an integer in [1, ${capacity}] (keysIn's length)`);
    }
    const n = this.length ?? capacity;
    if (this.type === "keyvalue" && !this.hasBuffer("valuesIn")) throw new Error('Multisplit: type "keyvalue" needs valuesIn');
    if (this.pairs && !this.hasBuffer("valuesOut")) throw new Error("Multisplit: needs valuesOut");
    const tile = this.tileK * WORKGROUP_SIZE;
    /* the histogram is sized for the buffer, not the length, so a shorter
     * length does not reallocate it */
    const maxSlots = multisplitSlots(capacity, 1, tile, this.targetWorkgroups);
    const slots = multisplitSlots(n, 1, tile, this.targetWorkgroups);
    this.scan.length = this.numBins * slots;
    const histBytes = this.numBins * maxSlots * 4;
    const buffers = (hist) => [
      { name: "keysIn", access: "read", type: "array<u32>" },
      { name: "keysOut", access: "read_write", type: "array<u32>" },
      ...(this.type === "keyvalue" ? [{ name: "valuesIn", access: "read", type: "array<u32>" }] : []),
      ...(this.pairs ? [{ name: "valuesOut", access: "read_write", type: "array<u32>" }] : []),
      { name: "hist", buffer: hist, access: "read_write", type: "array<u32>" },
      { name: "segStart", buffer: "zero", access: "read", type: "array<u32>" },
      ...(this.hasBuffer("bucketOffsets") ? [{ name: "bucketOffsets", access: "read_write", type: "array<u32>" }] : []),
      { name: "ms", buffer: "msParams", uniform: this.params },
    ];
    const dispatch = () => this.getSimpleDispatchGeometry({ workgroupCount: slots });
    const label = `Multisplit (${this.numBins} buckets, ${this.type}${this.values ? ", values: " + this.values : ""})`;
    return [
      new AllocateBuffer({ label: "counts", size: histBytes, clearBufferOnReuse: false }),
      new AllocateBuffer({ label: "offsets", size: histBytes, clearBufferOnReuse: false }),
      new AllocateBuffer({ label: "zero", size: 16, clearBufferOnReuse: false }) /* one segment, starting at 0 */,
      this.params.allocate("msParams", { n, numSegs: 1, slots, segStride: 0 }),
      new Kernel({ kernel: this.multisplitWGSL, entryPoint: "hist_main", label: `${label} hist`, buffers: buffers("counts"), getDispatchGeometry: dispatch }),
      new ExecutePrimitive({ primitive: this.scan, buffers: { inputBuffer: "counts", outputBuffer: "offsets" } }),
      new Kernel({ kernel: this.multisplitWGSL, entryPoint: "scatter_main", label: `${label} scatter`, buffers: buffers("offsets"), getDispatchGeometry: dispatch }),
      ...(this.hasBuffer("bucketOffsets")
        ? [
            new Kernel({
              kernel: this.multisplitWGSL,
              entryPoint: "offsets_main",
              label: `${label} bucket offsets`,
              buffers: buffers("offsets"),
              getDispatchGeometry: () => [1],
            }),
          ]
        : []),
    ];
  }

  destroy() {
    this.scan.destroy();
    super.destroy();
  }

  /** "" if keysOut (and valuesOut) are a valid multisplit of the input:
   * buckets in order, the same multiset of (key, value) elements, and, if
   * given, correct bucket offsets. All arguments are typed arrays (u32). */
  validate({ keysIn, valuesIn, keysOut, valuesOut, bucketOffsets }) {
    const n = this.length ?? keysIn.length;
    const binOf = (word) => {
      const T = this.bins.datatype;
      const x = T === "f32" ? new Float32Array(new Uint32Array([word]).buffer)[0] : T === "i32" ? word | 0 : word;
      return this.bins.binOf(x);
    };
    const value = (i) => (this.values === "index" ? i : valuesIn?.[i]);
    const counts = new Array(this.numBins).fill(0);
    for (let i = 0; i < n; i++) counts[binOf(keysIn[i])]++;
    const starts = [0];
    for (let d = 0; d < this.numBins; d++) starts.push(starts[d] + counts[d]);
    const errors = [];
    if (bucketOffsets) {
      const d = starts.findIndex((s, k) => bucketOffsets[k] !== s);
      if (d >= 0) errors.push(`bucketOffsets[${d}] = ${bucketOffsets[d]}, expected ${starts[d]}`);
    }
    /* bucket of each output position, and the multiset of elements */
    const seen = new Map();
    const tag = (k, v) => (this.pairs ? `${k},${v}` : `${k}`);
    for (let i = 0; i < n; i++) seen.set(tag(keysIn[i], value(i)), (seen.get(tag(keysIn[i], value(i))) ?? 0) + 1);
    let d = 0;
    for (let p = 0; p < n && errors.length < 5; p++) {
      while (p >= starts[d + 1]) d++;
      if (binOf(keysOut[p]) !== d) errors.push(`output ${p} (key ${keysOut[p]}) is in bucket ${binOf(keysOut[p])}, expected ${d}`);
      const t = tag(keysOut[p], valuesOut?.[p]);
      const c = seen.get(t);
      if (!c) errors.push(`output ${p} (${t}) is not an input element, or appears too often`);
      else seen.set(t, c - 1);
    }
    return errors.length ? "\n" + errors.join("\n") : "";
  }
}
