// Type definitions for gridwise
// Project: https://github.com/gridwise-js/gridwise
// Definitions by: Google DeepMind team

/** a WGSL helper for include(): its name, or [name, args] */
export type WgslHelper = string | [string, { [key: string]: any }];

/** The WGSL helpers available to a primitive's kernels (wgslFunctions.mjs) */
export interface WgslFunctions {
  /** module-scope WGSL for these helpers and everything they need, each once (put it first) */
  include(...helpers: WgslHelper[]): string;
  /** workgroup memory (bytes) declared by include(...helpers) */
  workgroupBytes(...helpers: WgslHelper[]): number;
  /** the individual helpers (wgScan, wgHistogram, subgroupDigitMatch, ...) */
  [helper: string]: any;
}

export class BasePrimitive {
  device: GPUDevice;
  label: string;
  useSubgroups: boolean;
  knownBuffers: string[];
  /** the WGSL helpers, for this primitive's device (with or without subgroups) */
  fnDeclarations: WgslFunctions;
  /** true when the kernels cannot be repeated in place: execute() then refuses trials > 1 */
  singleTrialOnly?: boolean;

  constructor(args: {
    device: GPUDevice;
    label?: string;
    disableSubgroups?: boolean;
    [key: string]: any;
  });

  execute(args?: { [key: string]: any }): Promise<void>;
  getTimingResult(): Promise<{
    gpuTotalTimeNS: number | number[];
    cpuTotalTimeNS: number;
  }>;
  getBuffer(name: string): any;
  registerBuffer(args: { label: string; buffer: any; device?: GPUDevice }): void;
  /** compute passes execute() records, including nested primitives' */
  countKernels(): number;
  /** labels of those kernels, in the order of getTimingResult()'s GPU times */
  kernelLabels(): string[];
  /** run back to back for warmMs (default 200) so the GPU is at full clock; returns the trials reached */
  warmUp(args?: { [key: string]: any }, warmMs?: number): Promise<number>;
  /** warm up, then time `repeats` executes of `trials` dispatches; medians per trial */
  benchmark(args?: { warmMs?: number; repeats?: number; trials?: number; [key: string]: any }): Promise<{
    ns: number;
    timing: "gpu" | "cpu";
    trials: number;
    repeats: number;
    kernels: { label: string; ns: number }[];
  }>;
  /** destroys GPU buffers the primitive allocated itself */
  destroy(): void;
}

/** One buffer a Kernel binds, declared once for WGSL and the bind group layout */
export interface KernelBuffer {
  /** the variable's name in WGSL */
  name: string;
  /** the primitive's buffer bound to it (default: name) */
  buffer?: string | { buffer: string; offset?: number; size?: number };
  /** "read" (read-only storage), "read_write" (storage) or "uniform" */
  access?: "read" | "read_write" | "uniform";
  /** its WGSL type, e.g. "array<f32>" */
  type?: string;
  /** a uniform parameter struct (sets access and type, and declares the struct) */
  uniform?: UniformStruct;
}

export class Kernel {
  label: string;
  kernel: string | ((args?: { [key: string]: any }) => string);

  constructor(
    args:
      | ((args?: { [key: string]: any }) => string)
      | {
          kernel: string | ((args?: { [key: string]: any }) => string);
          label?: string;
          /** buffers in binding order (or one list per bind group); replaces bufferTypes and bindings */
          buffers?: KernelBuffer[] | KernelBuffer[][];
          /** dispatch with the workgroup counts (x, y, z u32s) in this buffer, read on the GPU (INDIRECT usage) */
          indirect?: string | { buffer: string; offset?: number };
          [key: string]: any;
        }
  );

  enabled(): boolean;
  /** WGSL declarations of the declared buffers ("" without them) */
  readonly bufferDeclarations: string;
  /** the kernel's complete WGSL: its code, then its buffers' declarations */
  wgsl(): string;
}

/** WGSL declarations for a buffer list (for WGSL builders that return a whole module) */
export function declareBuffers(buffers: KernelBuffer[] | KernelBuffer[][]): string;
/** { bufferTypes, bindings } for a buffer list, for a Kernel whose WGSL declares its buffers itself */
export function bufferLayout(buffers: KernelBuffer[] | KernelBuffer[][]): { bufferTypes: string[][]; bindings: any[][] };

/** A uniform parameter struct declared once for WGSL and JavaScript */
export class UniformStruct {
  name: string;
  byteLength: number;
  constructor(name: string, fields: { [field: string]: "u32" | "i32" | "f32" });
  /** the WGSL struct declaration, padded to a multiple of 16 bytes */
  readonly wgsl: string;
  /** values packed in the struct's layout */
  pack(values: { [field: string]: number }): Uint8Array;
  /** an AllocateBuffer for a uniform buffer named label, holding values */
  allocate(label: string, values: { [field: string]: number }): AllocateBuffer;
}

export class AllocateBuffer {
  label: string;
  size: number;

  constructor(args: {
    label: string;
    size: number;
    usage?: number;
    [key: string]: any;
  });
}

/** Run another primitive inside this one's compute(), on the same encoder. */
export class ExecutePrimitive {
  primitive: BasePrimitive;
  buffers?: { [childBufferName: string]: string };

  constructor(args: {
    primitive: BasePrimitive;
    /** child buffer name -> this primitive's buffer name */
    buffers?: { [childBufferName: string]: string };
    label?: string;
    enable?: boolean;
  });

  enabled(): boolean;
}

/** Upload CPU data into a known buffer at a byte offset. A queue write: it lands
 * before any of this execute()'s GPU work runs, not in action order. */
export class WriteGPUBuffer {
  label: string;
  cpuSource: ArrayBufferView | ArrayBuffer;
  offset?: number;

  constructor(args: { label: string; cpuSource: ArrayBufferView | ArrayBuffer; offset?: number; [key: string]: any });
}

export class Buffer {
  label: string;
  device: GPUDevice;
  size: number;
  buffer: GPUBuffer;

  constructor(args: {
    label: string;
    device: GPUDevice;
    size: number;
    usage?: number;
  });

  destroy(): void;
}

export class BinOp {
  datatype?: string;
  identity: any;
  op: (a: any, b: any) => any;
  wgslfn: string;
  wgslatomic?: string;
  subgroupReduceOp?: string;
  subgroupInclusiveScanOp?: string;
  subgroupExclusiveScanOp?: string;

  constructor(args?: { datatype?: string; [key: string]: any });
}

export class BinOpNop extends BinOp {}
export const BinOpNopU32: BinOpNop;

export class BinOpAdd extends BinOp {}
export const BinOpAddU32: BinOpAdd;
export const BinOpAddF32: BinOpAdd;
export const BinOpAddI32: BinOpAdd;

export class BinOpMin extends BinOp {}
export const BinOpMinU32: BinOpMin;
export const BinOpMinF32: BinOpMin;
export const BinOpMinI32: BinOpMin;

export class BinOpMax extends BinOp {}
export const BinOpMaxU32: BinOpMax;
export const BinOpMaxF32: BinOpMax;
export const BinOpMaxI32: BinOpMax;

export class BinOpMultiply extends BinOp {}
export const BinOpMultiplyU32: BinOpMultiply;
export const BinOpMultiplyF32: BinOpMultiply;
export const BinOpMultiplyI32: BinOpMultiply;

export function makeBinOp(args: { op: string; datatype: string }): BinOp;

export class DLDFScan extends BasePrimitive {
  type: "exclusive" | "inclusive" | "reduce";
  datatype: string;
  binop: BinOp;

  constructor(args: {
    device: GPUDevice;
    binop: BinOp;
    type: "exclusive" | "inclusive" | "reduce";
    datatype: string;
    label?: string;
    disableSubgroups?: boolean;
  });
}

/** reduce-then-scan; same arguments as DLDFScan, any input length */
export class HierarchicalScan extends BasePrimitive {
  type: "exclusive" | "inclusive" | "reduce";
  datatype: string;
  binop: BinOp;

  constructor(args: {
    device: GPUDevice;
    binop: BinOp;
    type: "exclusive" | "inclusive" | "reduce";
    datatype: string;
    label?: string;
    disableSubgroups?: boolean;
    workgroupSize?: number;
    itemsPerThread?: number;
  });
}

/** zero the buffer named label (needs COPY_DST usage), in encoder order */
export class ClearBuffer {
  constructor(args: { label: string; enable?: boolean });
}

/** A bin function: which histogram bin a sample falls in (see bins.mjs) */
export class Bins {
  datatype: "u32" | "i32" | "f32";
  numBins: number;
  /** WGSL declaring fn name(x: datatype) -> u32 (BIN_NONE for no bin) */
  wgsl(name?: string): string;
  /** the JavaScript mirror; -1 for no bin */
  binOf(x: number): number;
}
export const BIN_NONE: number;
/** numBins equal-width bins over [lower, upper); integer types need (upper - lower) % numBins == 0 */
export class EvenBins extends Bins {
  constructor(args: { datatype: "u32" | "i32" | "f32"; numBins: number; lower: number; upper: number });
}
/** bin i is [levels[i], levels[i+1]); levels strictly increasing */
export class RangeBins extends Bins {
  levels: Uint32Array | Int32Array | Float32Array;
  constructor(args: { datatype: "u32" | "i32" | "f32"; levels: ArrayLike<number> });
}
/** 2^bits bins by bits [shift, shift + bits) of a u32 */
export class RadixBins extends Bins {
  constructor(args: { bits: number; shift?: number });
}

/** counts of samples per bin, like CUB's DeviceHistogram; samples in no bin are not counted */
export class Histogram extends BasePrimitive {
  bins: Bins;
  numBins: number;
  /** whether the bins fit in workgroup memory (otherwise counted in the output directly) */
  readonly workgroupBins: boolean;
  constructor(args: {
    device: GPUDevice;
    bins: Bins;
    datatype?: string;
    workgroupSize?: number;
    itemsPerThread?: number;
    copies?: number;
    maxWorkgroups?: number;
    label?: string;
  });
  /** the counts computed on the CPU */
  reference(input: ArrayLike<number>): Uint32Array;
}

export class OneSweepSort extends BasePrimitive {
  datatype: string;
  direction: "ascending" | "descending";
  type: "keysonly" | "keyvalue";
  inputLength: number;
  copyOutputToTemp: boolean;

  constructor(args: {
    device: GPUDevice;
    datatype: string;
    direction?: "ascending" | "descending";
    type?: "keysonly" | "keyvalue";
    inputLength: number;
    copyOutputToTemp?: boolean;
    label?: string;
    disableSubgroups?: boolean;
  });
}

export type PhiloxRng = { key: [number, number]; rounds?: number };

export function randomBitsWGSL(rng?: PhiloxRng | null): string;
export function randomBitsJS(rng?: PhiloxRng | null): {
  rand: (a: number, b: number, seed: number) => number;
  bounded: (k: number, key: number, seed: number, ctr?: { value: number }) => number;
  elem: ((i: number) => number) | null;
};
export const PHILOX_WGSL: string;
export function pmMixJS(i: number, seed: number): number;
export function pmRandJS(a: number, b: number, seed: number): number;
export function pmBoundedJS(k: number, key: number, seed: number, ctr?: { value: number }, rand?: (a: number, b: number, s: number) => number): number;
export function philox4x32JS(counter: number[], key: number[], rounds?: number): number[];

export interface MultisplitPermOptions {
  device: GPUDevice;
  n: number;
  seed?: number;
  groupSize?: number;
  split?: string | number[];
  carry?: "keys" | "indices" | "pairs";
  reproducible?: boolean;
  rng?: PhiloxRng | null;
  tileK?: number;
  targetWorkgroups?: number;
  localCap?: number;
  phaseLimit?: number;
  label?: string;
  disableSubgroups?: boolean;
  adapter?: GPUAdapter;
}

export class MultisplitPerm extends BasePrimitive {
  n: number;
  seed: number;
  groupBits: number;
  passBits: number[];
  numGroups: number;
  phases: string[];
  config: { [key: string]: any };

  constructor(args: MultisplitPermOptions);
  execute(args: { outputBuffer: GPUBuffer; [key: string]: any }): Promise<void>;
  /** the permutation a reproducible MultisplitPerm writes, computed on the CPU */
  reference(): Uint32Array;
  keyOf(i: number): number;
  groupOf(i: number): number;
  /** "" when valid */
  validate(args?: { outputBuffer?: Uint32Array }): string;
}

/** Parallel Fisher-Yates (Shun et al., SODA 2015): bit-identical to sequential
 * Fisher-Yates with the same swap targets; rounds sized by indirect dispatch */
export class ShunPerm extends BasePrimitive {
  n: number;
  seed: number;
  maxRounds: number;
  constructor(args: {
    device: GPUDevice;
    n: number;
    seed?: number;
    rng?: null | { key: [number, number]; rounds?: number };
    maxRounds?: number;
    [key: string]: any;
  });
  execute(args: { outputBuffer: GPUBuffer; [key: string]: any }): Promise<void>;
  /** rounds the last execute() needed (Infinity if maxRounds ran out) */
  roundsUsed(): Promise<number>;
  /** sequential Fisher-Yates with the same swap targets, on the CPU */
  reference(): Uint32Array;
  /** "" when the output equals reference() */
  validate(args?: { outputBuffer?: Uint32Array }): string;
}

/** The suffix array of a byte string: prefix doubling over OneSweepSort of
 * the still-tied suffixes (docs/suffix-array.md) */
/** Group elements by bucket (a bin function of the key, up to 256 buckets);
 * unstable. docs/multisplit.md */
export class Multisplit extends BasePrimitive {
  bins: Bins;
  type: "keysonly" | "keyvalue";
  values?: "index";
  length?: number;
  constructor(args: {
    device: GPUDevice;
    bins: Bins;
    type?: "keysonly" | "keyvalue";
    values?: "index";
    length?: number;
    tileK?: number;
    targetWorkgroups?: number;
    [key: string]: any;
  });
  execute(args: {
    keysIn: GPUBuffer;
    keysOut: GPUBuffer;
    valuesIn?: GPUBuffer;
    valuesOut?: GPUBuffer;
    bucketOffsets?: GPUBuffer;
    [key: string]: any;
  }): Promise<void>;
  /** "" if the outputs are a multisplit of the inputs (typed arrays) */
  validate(args: {
    keysIn: Uint32Array;
    valuesIn?: Uint32Array;
    keysOut: Uint32Array;
    valuesOut?: Uint32Array;
    bucketOffsets?: Uint32Array;
  }): string;
}
/** WGSL of one multisplit pass (hist_main, scatter_main), shared with MultisplitPerm */
export function multisplitPassWGSL(
  env: BasePrimitive,
  options: { radix: number; K: number; stagedDigit?: "recompute" | "search"; definitions: string }
): string;

/** The suffix array of a byte string: prefix doubling over OneSweepSort of
 * the still-tied suffixes (docs/suffix-array.md) */
export class SuffixArray extends BasePrimitive {
  n: number;
  rankBits: number;
  /** doubling rounds the last execute() ran */
  rounds: number;
  /** active suffixes before each round (activeCounts[0] = n, the initial round) */
  activeCounts: number[];
  /** each round's capacity (sort length); above its active count when the
   * round was submitted before the last count was read back */
  capacities: number[];
  /** rounds submitted before the last count was read back */
  speculatedRounds: number;
  /** with enableGPUTiming: kernel times, per round */
  timings?: { stage: string; m: number; h: number; kernels: { label: string; ns: number }[] }[];
  constructor(args: {
    device: GPUDevice;
    n: number;
    adapter?: GPUAdapter;
    /** submit a round before the last count is read back when it pays (default true) */
    speculate?: boolean;
    /** options for both OneSweepSorts (e.g. KEYS_PER_THREAD) */
    sortOptions?: { [key: string]: any };
    /** write ranks and the output through a Multisplit by destination block
     * (default: n >= 2^23) */
    localScatter?: boolean;
    [key: string]: any;
  });
  /** inputBuffer: the n bytes, 4 per u32 word, padded to a multiple of 4 bytes;
   * outputBuffer: n u32s; rankBuffer (optional): n u32s, rank[i] = 1 + position of suffix i */
  execute(args: {
    inputBuffer: GPUBuffer;
    outputBuffer: GPUBuffer;
    rankBuffer?: GPUBuffer;
    enableGPUTiming?: boolean;
    [key: string]: any;
  }): Promise<void>;
}
/** the suffix array on the CPU (prefix doubling, O(n log n)) */
export function suffixArrayReference(text: Uint8Array): Uint32Array;
/** "" if sa is text's suffix array (Burkhardt and Kärkkäinen's linear check) */
export function checkSuffixArray(text: Uint8Array, sa: Uint32Array): string;
/** modeled DRAM bytes by phase for a run's active counts (a lower bound) */
export function suffixArrayTraffic(args: {
  n: number;
  activeCounts: number[];
  capacities?: number[];
  localScatter?: boolean;
  rankBits?: number;
}): {
  bytes: { [phase: string]: number };
  random: { [phase: string]: number };
  total: number;
  totalRandom: number;
};
/** the phase a SuffixArray kernel label belongs to */
export function suffixArrayPhase(label: string): string;

/** For each tile (segment) of a CSR-style offsets array, the binop over its
 * atoms' values, with a load-balancing schedule separate from the work
 * (Osama et al., PPoPP 2023). docs/segmented-reduce.md */
export interface SegmentedReduceOptions {
  device: GPUDevice;
  binop: BinOp;
  /** number of tiles; offsets holds tiles + 1 entries (may change between executes) */
  tiles: number;
  /** offsets[tiles] (may change between executes) */
  atoms: number;
  /** WGSL expression of atom `atom` (in tile `tile`)'s value, given the names of the WGSL variables */
  atom: (atom: string, tile: string) => string;
  /** WGSL statement that consumes tile `tile`'s result `value` (default: outputBuffer[tile] = value;) */
  emit?: (tile: string, value: string) => string;
  /** the buffers atom and emit use, bound by name in execute() (access "read" unless given) */
  buffers?: { name: string; type: string; access?: "read" | "read_write" }[];
  /** "mergepath" (default) or "thread" */
  schedule?: "mergepath" | "thread";
  /** mergepath: path steps per invocation (default 7) */
  itemsPerThread?: number;
  adapter?: GPUAdapter;
  [key: string]: any;
}
export class SegmentedReduce extends BasePrimitive {
  tiles: number;
  atoms: number;
  schedule: "mergepath" | "thread";
  constructor(args: SegmentedReduceOptions);
  /** offsets (tiles + 1 u32s), the buffers named in options.buffers, and
   * outputBuffer when there is no emit */
  execute(args: { offsets: GPUBuffer; [buffer: string]: any }): Promise<void>;
}
/** the reduction on the CPU, for checking */
export function segmentedReduceReference(args: {
  offsets: Uint32Array;
  tiles: number;
  binop: BinOp;
  atomValue: (atom: number, tile: number) => number;
}): number[];

/** y = A x, A in CSR form (f32): a SegmentedReduce over A's rows. docs/segmented-reduce.md */
export class SpMV extends SegmentedReduce {
  rows: number;
  nnz: number;
  constructor(args: {
    device: GPUDevice;
    rows: number;
    nnz: number;
    schedule?: "mergepath" | "thread";
    itemsPerThread?: number;
    adapter?: GPUAdapter;
    [key: string]: any;
  });
  /** offsets: rows + 1 u32s; indices: nnz u32 columns; values: nnz f32s; x: f32 per column; y: rows f32s (written) */
  execute(args: {
    offsets: GPUBuffer;
    indices: GPUBuffer;
    values: GPUBuffer;
    x: GPUBuffer;
    y: GPUBuffer;
    enableGPUTiming?: boolean;
    [key: string]: any;
  }): Promise<void>;
}
/** y = A x on the CPU (f64 sums) */
export function spmvReference(args: { offsets: Uint32Array; indices: Uint32Array; values: Float32Array; x: Float32Array }): Float64Array;

export function tuneMultisplitPerm(args: {
  device: GPUDevice;
  n: number;
  reproducible?: boolean;
  rng?: PhiloxRng | null;
  carry?: "keys" | "indices" | "pairs";
  candidates?: { groupSize: number; split: string | number[] }[];
  trialMs?: number;
  disableSubgroups?: boolean;
  adapter?: GPUAdapter;
}): Promise<{
  options: { groupSize: number; split: string | number[] };
  passBits: string;
  geps: number;
  tried: { [key: string]: any }[];
}>;

export function multisplitPermPassBits(groupBits: number, split?: string | number[]): number[];
export function localFinishCap(device: GPUDevice, meanGroupSize?: number, bytesPerElement?: number, reservedBytes?: number): number;
/** workgroup memory the local finish needs besides its elements (for localFinishCap) */
export function localFinishReservedBytes(env: BasePrimitive): number;
export function localFinishWGSL(env: BasePrimitive, cap: number, options?: { [key: string]: any }): string;
/** the local finish's uniform parameters (n, numGroups, stride, seed, seed2) */
export const localFinishParams: UniformStruct;
/** the local finish's buffers (keys, outp, groupStart, lf), bound to the given buffer names */
export function localFinishBuffers(names?: { keys?: string; outp?: string; groupStart?: string; lf?: string }): KernelBuffer[];
export function wgShuffleWGSL(env: BasePrimitive, cap: number, options?: { canonical?: boolean; aux?: boolean }): string;
