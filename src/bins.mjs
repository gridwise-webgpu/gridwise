/** Bin functions: which histogram bin does a sample fall in?
 *
 * A bin function is to a histogram what a BinOp is to a scan: a small object
 * that supplies the WGSL (and a JavaScript mirror for validation) for one
 * part of a primitive. Each one declares
 *
 *   fn NAME(x: T) -> u32
 *
 * returning a bin in [0, numBins), or BIN_NONE (0xffffffff) for a sample that
 * falls in no bin. Samples outside the bins are not counted, as in CUB's
 * DeviceHistogram. Used by Histogram, MultisplitPerm and (through RadixBins)
 * anything that buckets keys by a bit field.
 *
 *   new EvenBins({ datatype, numBins, lower, upper })  equal-width bins over [lower, upper)
 *   new RangeBins({ datatype, levels })                bin i is [levels[i], levels[i+1])
 *   new RadixBins({ shift, bits })                     bits [shift, shift+bits) of a u32
 *
 * Members: datatype, numBins, wgsl(name) (the WGSL function), binOf(x) (the
 * JavaScript mirror; -1 for no bin), and, for RangeBins, levels (the typed
 * array the WGSL function reads from a storage buffer named "levels").
 *
 * Floating-point samples: WGSL lets implementations assume there are no
 * NaNs or infinities, so which bin (if any) they land in is unspecified.
 */

export const BIN_NONE = 0xffffffff;

/* a WGSL literal of datatype dt with exactly the value x */
function wgslLiteral(dt, x) {
  switch (dt) {
    case "f32": {
      const u = new Uint32Array(new Float32Array([x]).buffer)[0];
      return `bitcast<f32>(0x${u.toString(16)}u)`;
    }
    case "i32":
      return `i32(${x | 0})`;
    default:
      return `${x >>> 0}u`;
  }
}

export class Bins {
  constructor(args) {
    Object.assign(this, args);
  }
  toString() {
    return `${this.constructor.name}(${this.numBins} bins)`;
  }
}

/** numBins equal-width bins covering [lower, upper), like CUB's HistogramEven.
 * For u32 and i32 samples, (upper - lower) must be a multiple of numBins, so
 * that every bin has the same integer width (use RangeBins otherwise); upper
 * may be 2^32 for u32. For f32 samples, bin = trunc((x - lower) * scale) with
 * scale = f32(numBins / (upper - lower)), computed in f32 on both CPU and GPU. */
export class EvenBins extends Bins {
  constructor(args) {
    super(args);
    const { datatype, numBins, lower, upper } = this;
    if (!["u32", "i32", "f32"].includes(datatype)) {
      throw new Error(`EvenBins: datatype must be u32, i32 or f32, not ${datatype}`);
    }
    if (!(Number.isInteger(numBins) && numBins >= 1)) {
      throw new Error(`EvenBins: numBins must be a positive integer, not ${numBins}`);
    }
    if (!(upper > lower)) {
      throw new Error(`EvenBins: need lower < upper (got ${lower}, ${upper})`);
    }
    if (datatype === "f32") {
      this.lower = Math.fround(lower);
      this.upper = Math.fround(upper);
      this.scale = Math.fround(numBins / (this.upper - this.lower));
    } else {
      const range = upper - lower;
      if (!Number.isInteger(lower) || !Number.isInteger(upper) || range % numBins !== 0) {
        throw new Error(
          `EvenBins: for ${datatype} samples, upper - lower (${range}) must be an integer multiple of numBins (${numBins}); use RangeBins for uneven bins`
        );
      }
      if (range > 2 ** 32) {
        throw new Error("EvenBins: upper - lower must be at most 2^32");
      }
      this.width = range / numBins;
    }
  }
  wgsl(name = "binOf") {
    const { datatype: T, numBins: N } = this;
    if (T === "f32") {
      return /* wgsl */ `
fn ${name}(x: f32) -> u32 {
  if (!(x >= ${wgslLiteral("f32", this.lower)} && x < ${wgslLiteral("f32", this.upper)})) { return ${BIN_NONE}u; }
  /* rounding can reach numBins just below upper */
  return min(u32((x - ${wgslLiteral("f32", this.lower)}) * ${wgslLiteral("f32", this.scale)}), ${N - 1}u);
}`;
    }
    /* integers: the distance from lower, as u32, in units of width */
    return /* wgsl */ `
fn ${name}(x: ${T}) -> u32 {
  if (x < ${wgslLiteral(T, this.lower)}) { return ${BIN_NONE}u; }
  let b = bitcast<u32>(x - ${wgslLiteral(T, this.lower)}) / ${this.width}u;
  return select(${BIN_NONE}u, b, b < ${N}u);
}`;
  }
  binOf(x) {
    if (this.datatype === "f32") {
      const f = Math.fround;
      if (!(x >= this.lower && x < this.upper)) return -1;
      return Math.min(Math.trunc(f(f(x - this.lower) * this.scale)), this.numBins - 1);
    }
    if (x < this.lower) return -1;
    const b = Math.floor((x - this.lower) / this.width);
    return b < this.numBins ? b : -1;
  }
}

/** Bins bounded by levels, like CUB's HistogramRange: bin i is
 * [levels[i], levels[i+1]), so levels.length = numBins + 1, strictly
 * increasing. The WGSL function binary-searches a storage buffer named
 * "levels" (array<T>), which the primitive fills from this.levels. */
export class RangeBins extends Bins {
  constructor(args) {
    super(args);
    const { datatype } = this;
    const TA = { u32: Uint32Array, i32: Int32Array, f32: Float32Array }[datatype];
    if (!TA) throw new Error(`RangeBins: datatype must be u32, i32 or f32, not ${datatype}`);
    this.levels = TA.from(this.levels);
    if (this.levels.length < 2) throw new Error("RangeBins: need at least two levels");
    for (let i = 1; i < this.levels.length; i++) {
      if (!(this.levels[i] > this.levels[i - 1])) {
        throw new Error(`RangeBins: levels must be strictly increasing (levels[${i}] = ${this.levels[i]})`);
      }
    }
    this.numBins = this.levels.length - 1;
    this.needsLevels = true;
  }
  wgsl(name = "binOf") {
    const L = this.levels.length;
    return /* wgsl */ `
fn ${name}(x: ${this.datatype}) -> u32 {
  if (!(x >= levels[0] && x < levels[${L - 1}])) { return ${BIN_NONE}u; }
  /* invariant: levels[lo] <= x < levels[hi] */
  var lo = 0u;
  var hi = ${L - 1}u;
  while (hi - lo > 1u) {
    let mid = (lo + hi) / 2u;
    if (x >= levels[mid]) { lo = mid; } else { hi = mid; }
  }
  return lo;
}`;
  }
  binOf(x) {
    const lv = this.levels;
    if (!(x >= lv[0] && x < lv[lv.length - 1])) return -1;
    let lo = 0;
    let hi = lv.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (x >= lv[mid]) lo = mid;
      else hi = mid;
    }
    return lo;
  }
}

/** 2^bits bins by the bit field [shift, shift + bits) of a u32: a radix
 * digit, as in radix sort and MultisplitPerm. Every sample has a bin. */
export class RadixBins extends Bins {
  constructor(args) {
    super({ datatype: "u32", shift: 0, ...args });
    if (this.datatype !== "u32") throw new Error("RadixBins: datatype must be u32");
    if (!(this.bits >= 1 && this.bits <= 16 && this.shift >= 0 && this.shift + this.bits <= 32)) {
      throw new Error(`RadixBins: need 1 <= bits <= 16 and shift + bits <= 32 (got ${this.shift}, ${this.bits})`);
    }
    this.numBins = 2 ** this.bits;
  }
  wgsl(name = "binOf") {
    return /* wgsl */ `
fn ${name}(x: u32) -> u32 { return (x >> ${this.shift}u) & ${this.numBins - 1}u; }`;
  }
  binOf(x) {
    return (x >>> this.shift) & (this.numBins - 1);
  }
}
