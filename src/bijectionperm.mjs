/**
 * Random permutations from keyed bijections: perm[i] = f(i), where f is a
 * pseudorandom permutation of [0, n) chosen by the seed. No atomics, one
 * 4-byte write per element (or, with gather, out[i] = in[f(i)]).
 *
 *   const perm = new FeistelPerm({ device, n, seed });
 *   await perm.execute({ outputBuffer });                 // n u32s
 *   const perm = new MitchellPerm({ device, n, seed, gather: true });
 *   await perm.execute({ inputBuffer, outputBuffer });   // n 64-bit values each
 *
 * Three bijections:
 *
 *   FeistelPerm     balanced Feistel network on [0, 2^(2h)), 2^(2h) >= n, with a
 *                   triple32 or multiply-shift round function (cheap; a
 *                   heuristic pseudorandom permutation)
 *   SwapOrNotPerm   Hoang, Morris and Rogaway's swap-or-not cipher (CRYPTO
 *                   2012), directly on [0, n): no padding
 *   MitchellPerm    Mitchell, Stokes, Frank and Holmes, "Bandwidth-Optimal
 *                   Random Shuffling for GPUs" (ACM TOPC 2022), as NVIDIA
 *                   Thrust's thrust::shuffle implements it, bit for bit
 *
 * A bijection on a domain larger than n is restricted to [0, n) by cycle
 * walking (apply f until the value is below n; i's cycle contains i, so this
 * ends), in one kernel. MitchellPerm's default "scan" mode instead does what
 * Thrust does: evaluate f on the whole domain, keep the values below n, and
 * compact them in index order with a scan (count, DLDFScan, write).
 *
 * Fairness: a keyed bijection reaches at most 2^(key bits) of the n!
 * permutations, so none of these is uniform; their quality rests on f being
 * a good pseudorandom permutation. Being atomic-free, the output does not
 * depend on the hardware: reference() computes it on the CPU.
 *
 * Common options: n (required, 1..2^31), seed (default 0x2545f491), gather
 * (default false: write the permutation; true: out[i] = inputBuffer[f(i)]),
 * valueType (gathered values' WGSL type, default "vec2u"), phaseLimit (run
 * only the first phaseLimit of `phases`).
 *
 * Buffers: outputBuffer (n u32s, or n values with gather), inputBuffer
 * (gather only). Internal: round keys, parameters (and MitchellPerm's tile
 * counts and offsets).
 *
 * Origin: written for the random-permutations paper (Owens et al.), where the
 * round keys are derived from the seed exactly as here.
 */
import { BasePrimitive, Kernel, AllocateBuffer, ExecutePrimitive, UniformStruct } from "./primitive.mjs";
import { DLDFScan } from "./scandldf.mjs";
import { BinOpAdd } from "./binop.mjs";
import { randomBitsWGSL, MULHI_WGSL, pmT32JS, pmRandJS, pmBoundedJS, mulhiJS } from "./random.mjs";

const WORKGROUP_SIZE = 256;
const VALUE_WORDS = { u32: 1, i32: 1, f32: 1, vec2u: 2, vec2i: 2, vec2f: 2, vec4u: 4, vec4i: 4, vec4f: 4 };

/** Base class: a subclass supplies the bijection (WGSL fn bijection(x) and a
 * bit-exact JS bijectionJS(x)), its round keys and its parameters. The
 * parameters struct must start with n; WGSL sees it as `bp` and the keys as
 * `roundKeys: array<u32>`. */
export class BijectionPerm extends BasePrimitive {
  constructor(args) {
    super(args);
    if (!(Number.isInteger(this.n) && this.n >= 1 && this.n <= 2 ** 31)) {
      throw new Error(`${this.constructor.name}: requires an integer n in [1, 2^31]`);
    }
    this.seed = (this.seed ?? 0x2545f491) >>> 0;
    this.gather ??= false;
    this.valueType ??= "vec2u";
    if (this.gather && !(this.valueType in VALUE_WORDS)) {
      throw new Error(`${this.constructor.name}: unknown valueType ${this.valueType}`);
    }
    this.knownBuffers = this.gather ? ["inputBuffer", "outputBuffer"] : ["outputBuffer"];
    this.workgroupSize = WORKGROUP_SIZE;
    this.datatype = "u32";
    this.phases = ["bijection"];
  }

  /** WGSL type of an output element */
  get elementType() {
    return this.gather ? this.valueType : "u32";
  }
  get elementWords() {
    return this.gather ? VALUE_WORDS[this.valueType] : 1;
  }

  /* ---- what a subclass supplies ---- */
  /** WGSL: fn bijection(x: u32) -> u32, and anything it needs */
  bijectionWGSL() {
    throw new Error("BijectionPerm: subclass must define bijectionWGSL()");
  }
  /** the same function in JavaScript, bit for bit */
  bijectionJS(x) {
    throw new Error("BijectionPerm: subclass must define bijectionJS()");
  }
  /** the round keys (Uint32Array) */
  get roundKeys() {
    throw new Error("BijectionPerm: subclass must define roundKeys");
  }
  /** { params: UniformStruct (starts with n), values } */
  get parameters() {
    throw new Error("BijectionPerm: subclass must define parameters");
  }

  /* ---- shared ---- */
  /** f restricted to [0, n) by cycle walking */
  walkJS(i) {
    let x = this.bijectionJS(i);
    while (x >= this.n) x = this.bijectionJS(x);
    return x;
  }

  /** the indices execute() writes (with gather, the input positions it reads) */
  reference() {
    return Uint32Array.from({ length: this.n }, (_, i) => this.walkJS(i));
  }

  /** "" if the output is reference() (gathered, with gather), else the first differences */
  validate = (args = {}) => {
    const out = args.outputBuffer?.cpuBuffer ?? args.outputBuffer ?? this.getBuffer("outputBuffer").cpuBuffer;
    const inp = this.gather
      ? (args.inputBuffer?.cpuBuffer ?? args.inputBuffer ?? this.getBuffer("inputBuffer").cpuBuffer)
      : null;
    const ref = this.reference();
    const w = this.elementWords;
    const problems = [];
    for (let i = 0; i < this.n && problems.length < 5; i++) {
      for (let j = 0; j < w; j++) {
        const expected = this.gather ? inp[ref[i] * w + j] : ref[i];
        if (out[i * w + j] !== expected) {
          problems.push(`position ${i}${w > 1 ? ` word ${j}` : ""}: expected ${expected}, got ${out[i * w + j]}`);
          break;
        }
      }
    }
    return problems.length ? "\n" + problems.join("\n") : "";
  };

  /** buffer declarations every kernel shares */
  get commonBuffers() {
    const { params } = this.parameters;
    return [
      { name: "roundKeys", access: "read", type: "array<u32>" },
      { name: "bp", buffer: "bijectionParams", uniform: params },
    ];
  }
  get ioBuffers() {
    const T = this.elementType;
    return [
      { name: "outputBuffer", access: "read_write", type: `array<${T}>` },
      ...(this.gather ? [{ name: "inputBuffer", access: "read", type: `array<${T}>` }] : []),
    ];
  }
  /** output = f(x), or the gathered input value */
  emit(x) {
    return this.gather ? `inputBuffer[${x}]` : x;
  }

  walkWGSL = () => /* wgsl */ `
${this.fnDeclarations.include("computeLinearizedGridParametersSplit")}
${this.bijectionWGSL()}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  if (gid >= bp.n) { return; }
  var x = bijection(gid);
  loop {                       /* cycle walking */
    if (x < bp.n) { break; }
    x = bijection(x);
  }
  outputBuffer[gid] = ${this.emit("x")};
}`;

  /** actions every mode starts with */
  setupActions() {
    const outputLength = this.getBuffer("outputBuffer")?.size / (4 * this.elementWords);
    if (!(outputLength >= this.n)) {
      throw new Error(`${this.constructor.name}: outputBuffer must hold n = ${this.n} ${this.elementType}s`);
    }
    const { params, values } = this.parameters;
    const keys = this.roundKeys;
    return [
      params.allocate("bijectionParams", values),
      new AllocateBuffer({
        label: "roundKeys",
        size: Math.max(16, keys.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        populateWith: keys,
      }),
    ];
  }

  compute() {
    let step = 0;
    return [
      ...this.setupActions(),
      new Kernel({
        kernel: this.walkWGSL,
        entryPoint: "main",
        label: `${this.constructor.name} bijection`,
        buffers: [...this.ioBuffers, ...this.commonBuffers],
        getDispatchGeometry: () =>
          this.getSimpleDispatchGeometry({ workgroupCount: Math.ceil(this.n / WORKGROUP_SIZE) }),
        enable: step++ < (this.phaseLimit ?? Infinity),
      }),
    ];
  }

  get bytesTransferred() {
    return this.n * 4 * this.elementWords * (this.gather ? 2 : 1);
  }
}

/* ================================================================ Feistel */

const FEISTEL_ROUND_FUNCTIONS = {
  triple32: {
    wgsl: "pm_t32(right ^ roundKeys[r]) & mask",
    js: (right, key, h, mask) => pmT32JS((right ^ key) >>> 0) & mask,
  },
  mulshift: {
    wgsl: "((right ^ roundKeys[r]) * 0x9e3779b1u) >> (32u - bp.halfBits)",
    js: (right, key, h) => Math.imul((right ^ key) >>> 0, 0x9e3779b1) >>> (32 - h),
  },
};

/**
 * FeistelPerm: a balanced Feistel network on h-bit halves, h = ceil(ceil(log2 n) / 2),
 * cycle-walked to [0, n). For n a power of 4 the domain is exactly n.
 *
 * Round functions (right half and a 32-bit round key to h bits):
 *   "triple32" (default)  low h bits of triple32(right ^ key): 3 multiplies, 4 xorshifts
 *   "mulshift"            top h bits of (right ^ key) * 0x9e3779b1: one multiply
 *                         (multiply-shift hashing; needs more rounds)
 * Options: rounds (default 24), roundFunction. Round key r is
 * pmRandJS(seed, r, 0x6a09e667).
 */
export class FeistelPerm extends BijectionPerm {
  constructor(args) {
    super(args);
    this.rounds ??= 24;
    this.roundFunction ??= "triple32";
    if (!(this.roundFunction in FEISTEL_ROUND_FUNCTIONS)) {
      throw new Error(`FeistelPerm: unknown round function ${this.roundFunction}`);
    }
    this.halfBits = Math.max(1, Math.ceil(Math.ceil(Math.log2(this.n)) / 2));
    this.domainBits = 2 * this.halfBits;
    this.params = new UniformStruct("FeistelParams", { n: "u32", halfBits: "u32", rounds: "u32" });
    this.config = {
      roundFunction: this.roundFunction,
      rounds: this.rounds,
      gather: this.gather,
      domainBits: this.domainBits,
    };
    if (!("label" in args)) this.label += this.makeParamString(["roundFunction", "rounds", "gather"]);
  }
  get roundKeys() {
    return (this._keys ??= Uint32Array.from({ length: this.rounds }, (_, r) => pmRandJS(this.seed, r, 0x6a09e667)));
  }
  get parameters() {
    return { params: this.params, values: { n: this.n, halfBits: this.halfBits, rounds: this.rounds } };
  }
  bijectionWGSL() {
    return /* wgsl */ `
${randomBitsWGSL(null)}
fn bijection(x: u32) -> u32 {
  let mask = (1u << bp.halfBits) - 1u;
  var left = x >> bp.halfBits;
  var right = x & mask;
  for (var r = 0u; r < bp.rounds; r++) {
    let f = ${FEISTEL_ROUND_FUNCTIONS[this.roundFunction].wgsl};
    let t = right;
    right = left ^ f;
    left = t;
  }
  return (left << bp.halfBits) | right;
}`;
  }
  bijectionJS(x) {
    const h = this.halfBits,
      mask = (2 ** h - 1) >>> 0,
      F = FEISTEL_ROUND_FUNCTIONS[this.roundFunction].js;
    let left = x >>> h,
      right = (x & mask) >>> 0;
    for (const key of this.roundKeys) {
      const t = right;
      right = (left ^ F(right, key, h, mask)) >>> 0;
      left = t;
    }
    return ((left << h) | right) >>> 0;
  }
}

/* ================================================================ swap-or-not */

/**
 * SwapOrNotPerm: Hoang, Morris and Rogaway's swap-or-not cipher on [0, n).
 * Round r, with K_r uniform in [0, n) and a one-bit round function F_r:
 *
 *   partner = (K_r - x) mod n;  if F_r(max(x, partner)) = 1 then x = partner
 *
 * F_r is the low bit of triple32(max(x, partner) ^ k_r): a heuristic choice
 * (the security proof needs a real pseudorandom function). Options: rounds
 * (default 6 ceil(log2 n), a parameter rather than the paper's bound). Keys:
 * K_r = pmBoundedJS(n, r, seed), k_r = pmRandJS(r, 1, seed ^ 0xbb67ae85).
 */
export class SwapOrNotPerm extends BijectionPerm {
  constructor(args) {
    super(args);
    this.rounds ??= 6 * Math.ceil(Math.log2(this.n));
    this.params = new UniformStruct("SwapOrNotParams", { n: "u32", rounds: "u32" });
    this.config = { rounds: this.rounds, gather: this.gather };
    if (!("label" in args)) this.label += this.makeParamString(["rounds", "gather"]);
  }
  /** (K_r, k_r) interleaved */
  get roundKeys() {
    if (!this._keys) {
      this._keys = new Uint32Array(2 * this.rounds);
      for (let r = 0; r < this.rounds; r++) {
        this._keys[2 * r] = pmBoundedJS(this.n, r, this.seed);
        this._keys[2 * r + 1] = pmRandJS(r, 1, (this.seed ^ 0xbb67ae85) >>> 0);
      }
    }
    return this._keys;
  }
  get parameters() {
    return { params: this.params, values: { n: this.n, rounds: this.rounds } };
  }
  bijectionWGSL() {
    return /* wgsl */ `
${randomBitsWGSL(null)}
fn bijection(x0: u32) -> u32 {
  var x = x0;
  for (var r = 0u; r < bp.rounds; r++) {
    let K = roundKeys[2u * r];
    /* (K - x) mod n, without overflow for n <= 2^31 */
    let partner = select(K - x, K + (bp.n - x), K < x);
    if ((pm_t32(max(x, partner) ^ roundKeys[2u * r + 1u]) & 1u) == 1u) { x = partner; }
  }
  return x;
}`;
  }
  bijectionJS(x) {
    const n = this.n,
      keys = this.roundKeys;
    for (let r = 0; r < this.rounds; r++) {
      const K = keys[2 * r];
      const partner = K >= x ? K - x : K + (n - x);
      if (pmT32JS((Math.max(x, partner) ^ keys[2 * r + 1]) >>> 0) & 1) x = partner;
    }
    return x;
  }
}

/* ================================================================ Mitchell et al. (Thrust) */

const MITCHELL_PER_THREAD = 16;
const MITCHELL_TILE = WORKGROUP_SIZE * MITCHELL_PER_THREAD;
const bitWidth = (x) => (x === 0 ? 0 : 32 - Math.clz32(x));

/**
 * MitchellPerm: thrust::shuffle (thrust/detail/random_bijection.h,
 * feistel_bijection; thrust/system/detail/generic/shuffle.inl, shuffle_copy).
 *
 * The bijection ("VariablePhilox") on [0, 2^b), b = max(8, bit_width(n - 1)),
 * r_bits = ceil(b / 2), l_bits = floor(b / 2): L = x >> r_bits, R = x & r_mask,
 * then per round with key k:
 *   product = 0xD2B74407B1CE6E93 * L          (64-bit, mod 2^64)
 *   L' = ((product >> 32) ^ k ^ R) & l_mask
 *   R' = ((u32(product) << (r_bits - l_bits)) | (R >> l_bits)) & r_mask
 * result (L << r_bits) | R.
 *
 * Modes:
 *   "scan" (default)  Thrust's output: out[pos] = f(i) for the i in [0, 2^b)
 *                     with f(i) < n, pos counting those i in order. A count
 *                     kernel (valid values per 4096-index tile), DLDFScan of
 *                     the counts, and a write kernel that evaluates f again.
 *                     Thrust fuses this into one decoupled-lookback pass (one
 *                     evaluation per index); that needs forward progress
 *                     between workgroups, which WebGPU does not promise.
 *                     2^b < 2n, so at most half the evaluations are wasted.
 *   "walk"            one kernel, cycle walking. For n a power of two (>= 256)
 *                     the domain is n and this is exactly Thrust's output.
 * Options: mode, rounds (default 24, Thrust's). Round key r is
 * pmRandJS(seed, r, 0x510e527f). `scans` holds the nested scan primitive (a
 * caller may substitute one).
 */
export class MitchellPerm extends BijectionPerm {
  constructor(args) {
    super(args);
    this.mode ??= "scan";
    if (!["scan", "walk"].includes(this.mode)) throw new Error(`MitchellPerm: unknown mode ${this.mode}`);
    this.rounds ??= 24;
    const b = Math.max(8, bitWidth(this.n - 1));
    this.domainBits = b;
    this.rBits = Math.ceil(b / 2);
    this.lBits = Math.floor(b / 2);
    this.params = new UniformStruct("MitchellParams", {
      n: "u32",
      rBits: "u32",
      lBits: "u32",
      rounds: "u32",
      domain: "u32",
    });
    this.binop = new BinOpAdd({ datatype: "u32" }); /* the tile scans */
    if (this.mode === "scan") {
      this.tiles = Math.ceil(2 ** b / MITCHELL_TILE);
      this.scans = [
        new DLDFScan({
          device: this.device,
          adapter: this.adapter,
          datatype: "u32",
          type: "exclusive",
          binop: new BinOpAdd({ datatype: "u32" }),
          ...(this.disableSubgroups && { disableSubgroups: true }),
        }),
      ];
      this.phases = ["count valid keys", "scan", "write"];
    }
    this.config = { mode: this.mode, gather: this.gather, rounds: this.rounds, domainBits: b };
    if (!("label" in args)) this.label += this.makeParamString(["mode", "rounds", "gather"]);
  }
  get roundKeys() {
    return (this._keys ??= Uint32Array.from({ length: this.rounds }, (_, r) => pmRandJS(this.seed, r, 0x510e527f)));
  }
  get parameters() {
    const { n, rBits, lBits, rounds } = this;
    return { params: this.params, values: { n, rBits, lBits, rounds, domain: 2 ** this.domainBits } };
  }
  bijectionWGSL() {
    return /* wgsl */ `
${MULHI_WGSL}
/* Thrust's feistel_bijection::operator() */
fn bijection(x: u32) -> u32 {
  let rMask = (1u << bp.rBits) - 1u;
  let lMask = (1u << bp.lBits) - 1u;
  var L = x >> bp.rBits;
  var R = x & rMask;
  for (var i = 0u; i < bp.rounds; i++) {
    /* product = 0xD2B74407B1CE6E93 * L mod 2^64, as (hi, lo) 32-bit words */
    let lo = 0xB1CE6E93u * L;
    let hi = philox_mulhi(0xB1CE6E93u, L) + 0xD2B74407u * L;
    let lPrime = (hi ^ roundKeys[i]) ^ R;
    let rPrime = (lo << (bp.rBits - bp.lBits)) | (R >> bp.lBits);
    L = lPrime & lMask;
    R = rPrime & rMask;
  }
  return (L << bp.rBits) | R;
}`;
  }
  bijectionJS(x) {
    const rMask = (2 ** this.rBits - 1) >>> 0,
      lMask = (2 ** this.lBits - 1) >>> 0,
      shift = this.rBits - this.lBits;
    let L = x >>> this.rBits,
      R = (x & rMask) >>> 0;
    for (const key of this.roundKeys) {
      const lo = Math.imul(0xb1ce6e93, L) >>> 0;
      const hi = (mulhiJS(0xb1ce6e93, L) + Math.imul(0xd2b74407, L)) >>> 0;
      const lPrime = (hi ^ key ^ R) >>> 0;
      const rPrime = ((lo << shift) | (R >>> this.lBits)) >>> 0;
      L = (lPrime & lMask) >>> 0;
      R = (rPrime & rMask) >>> 0;
    }
    return ((L << this.rBits) | R) >>> 0;
  }
  /** Thrust's compaction order ("scan"), or cycle walking ("walk") */
  reference() {
    if (this.mode === "walk") return super.reference();
    const out = new Uint32Array(this.n);
    for (let i = 0, pos = 0; pos < this.n; i++) {
      const k = this.bijectionJS(i);
      if (k < this.n) out[pos++] = k;
    }
    return out;
  }

  /** "scan": count the valid values in each tile, or (write) recompute them
   * and write each at its offset */
  tileWGSL = (write) => () => /* wgsl */ `
${this.fnDeclarations.include("wgScan", "computeLinearizedGridParametersSplit")}
${this.bijectionWGSL()}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let tile = wgid;
  if (tile * ${MITCHELL_TILE}u >= bp.domain) { return; }  /* past the last tile (a split dispatch) */
  let base = tile * ${MITCHELL_TILE}u + builtinsNonuniform.lidx * ${MITCHELL_PER_THREAD}u;
  var keys: array<u32, ${MITCHELL_PER_THREAD}>;
  var c = 0u;
  for (var k = 0u; k < ${MITCHELL_PER_THREAD}u; k++) {
    let i = base + k;
    keys[k] = 0xffffffffu;
    if (i < bp.domain) {
      let key = bijection(i);
      if (key < bp.n) { keys[k] = key; c += 1u; }
    }
  }
  var total: u32;
  let prefix = wgExclusiveScan(c, &total, builtinsUniform, builtinsNonuniform);
${
  write
    ? /* wgsl */ `  var pos = offsets[tile] + prefix;
  for (var k = 0u; k < ${MITCHELL_PER_THREAD}u; k++) {
    if (keys[k] != 0xffffffffu) { outputBuffer[pos] = ${this.emit("keys[k]")}; pos += 1u; }
  }`
    : /* wgsl */ `  if (builtinsNonuniform.lidx == 0u) { counts[tile] = total; }`
}
}`;

  compute() {
    if (this.mode === "walk") return super.compute();
    let step = 0;
    const enabled = () => step++ < (this.phaseLimit ?? Infinity);
    const words = Math.ceil(this.tiles / 4) * 4;
    const dispatch = () => this.getSimpleDispatchGeometry({ workgroupCount: this.tiles });
    return [
      ...this.setupActions(),
      new AllocateBuffer({ label: "counts", size: words * 4, clearBufferOnReuse: false }),
      new AllocateBuffer({ label: "offsets", size: words * 4, clearBufferOnReuse: false }),
      new Kernel({
        kernel: this.tileWGSL(false),
        entryPoint: "main",
        label: "MitchellPerm count valid keys",
        buffers: [{ name: "counts", access: "read_write", type: "array<u32>" }, ...this.commonBuffers],
        getDispatchGeometry: dispatch,
        enable: enabled(),
      }),
      new ExecutePrimitive({
        primitive: this.scans[0],
        buffers: { inputBuffer: "counts", outputBuffer: "offsets" },
        label: "MitchellPerm scan",
        enable: enabled(),
      }),
      new Kernel({
        kernel: this.tileWGSL(true),
        entryPoint: "main",
        label: "MitchellPerm write",
        buffers: [{ name: "offsets", access: "read", type: "array<u32>" }, ...this.ioBuffers, ...this.commonBuffers],
        getDispatchGeometry: dispatch,
        enable: enabled(),
      }),
    ];
  }

  get bytesTransferred() {
    return super.bytesTransferred + (this.tiles ?? 0) * 4 * 3;
  }
}
