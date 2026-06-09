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
