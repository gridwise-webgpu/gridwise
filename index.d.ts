// Type definitions for gridwise
// Project: https://github.com/gridwise-js/gridwise
// Definitions by: Google DeepMind team

export class BasePrimitive {
  device: GPUDevice;
  label: string;
  useSubgroups: boolean;
  knownBuffers: string[];

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

export class WriteGPUBuffer {
  label: string;
  cpuSource: any;

  constructor(args: { label: string; cpuSource: any; [key: string]: any });
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
export function localFinishCap(device: GPUDevice, meanGroupSize?: number, bytesPerElement?: number): number;
export function localFinishWGSL(env: BasePrimitive, cap: number, options?: { [key: string]: any }): string;
/** the local finish's uniform parameters (n, numGroups, stride, seed, seed2) */
export const localFinishParams: UniformStruct;
/** the local finish's buffers (keys, outp, groupStart, lf), bound to the given buffer names */
export function localFinishBuffers(names?: { keys?: string; outp?: string; groupStart?: string; lf?: string }): KernelBuffer[];
export function wgShuffleWGSL(env: BasePrimitive, cap: number, options?: { canonical?: boolean; aux?: boolean }): string;
