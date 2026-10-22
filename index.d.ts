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
  /** destroys GPU buffers the primitive allocated itself */
  destroy(): void;
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
          [key: string]: any;
        }
  );

  enabled(): boolean;
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
export function wgShuffleWGSL(env: BasePrimitive, cap: number, options?: { canonical?: boolean; aux?: boolean }): string;
