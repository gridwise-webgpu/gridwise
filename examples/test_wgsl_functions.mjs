import { BasePrimitive, Kernel, AllocateBuffer, WriteGPUBuffer } from "../src/primitive.mjs";
import { BinOpAdd, BinOpMax } from "../src/binop.mjs";
import { datatypeToTypedArray } from "../src/util.mjs";

class TestWGSLFunctionsPrimitive extends BasePrimitive {
  constructor(args) {
    super(args);
    this.binop = args.binop ?? new BinOpAdd({ datatype: args.datatype });
    this.workgroupSize = args.workgroupSize ?? 256;
    this.numThreadsPerWorkgroup = this.workgroupSize;
    this.inputLength = args.inputLength ?? this.workgroupSize;
    this.kernelBody = args.kernelBody;
    this.workgroupCount = 1;
    this.getDispatchGeometry = this.getSimpleDispatchGeometry;
    this.knownBuffers = ["inputBuffer", "outputBuffer"];
  }

  compute() {
    const shaderCode = /* wgsl */ `
      ${this.fnDeclarations.enableSubgroupsIfAppropriate()}
      @group(0) @binding(0) var<storage, read> inputBuffer: array<${this.datatype}>;
      @group(0) @binding(1) var<storage, read_write> outputBuffer: array<${this.datatype}>;

      ${this.fnDeclarations.commonDefinitions()}
      ${this.fnDeclarations.subgroupEmulation()}
      ${this.binop.wgslfn}
      ${this.fnDeclarations.roundUpDivU32()}

      ${this.fnDeclarations.vec4InclusiveScan()}
      ${this.fnDeclarations.vec4ExclusiveScan()}
      ${this.fnDeclarations.vec4InclusiveToExclusive()}
      ${this.fnDeclarations.vec4Reduce()}
      ${this.fnDeclarations.vec4ScalarBinopV4()}
      ${this.fnDeclarations.subgroupZero()}
      ${this.fnDeclarations.subgroupShuffle()}
      ${this.fnDeclarations.subgroupBallot()}
      ${this.fnDeclarations.subgroupReduce()}
      ${this.fnDeclarations.subgroupInclusiveOpScan()}
      ${this.fnDeclarations.wgScan()}
      ${this.fnDeclarations.wgScan({ scanType: "inclusive" })}
      ${this.fnDeclarations.wgHistogram({ name: "hist", numBins: 16, copies: 2 })}

      @compute @workgroup_size(${this.workgroupSize}) fn main(
        builtinsNonuniform: BuiltinsNonuniform,
        builtinsUniform: BuiltinsUniform
      ) {
        ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
        ${this.fnDeclarations.initializeSubgroupVars()}
        ${this.kernelBody}
      }
    `;

    return [
      new AllocateBuffer({
        label: "inputBuffer",
        size: this.inputLength * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      new AllocateBuffer({
        label: "outputBuffer",
        size: this.inputLength * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
      new WriteGPUBuffer({
        label: "inputBuffer",
        cpuSource: new Uint32Array(this.inputLength),
      }),
      new Kernel({
        label: "test_kernel",
        kernel: shaderCode,
        bufferTypes: [["read-only-storage", "storage"]],
        bindings: [["inputBuffer", "outputBuffer"]],
      })
    ];
  }
}

export async function runSuite(device, resultsCallback) {
  const tests = [
    {
      name: "vec4InclusiveScan",
      kernelBody: "outputBuffer[gid] = vec4InclusiveScan(vec4<u32>(1u, 2u, 3u, 4u)).z;",
      expected: (lidx) => 6,
    },
    {
      name: "vec4ExclusiveScan",
      kernelBody: "outputBuffer[gid] = vec4ExclusiveScan(vec4<u32>(1u, 2u, 3u, 4u)).z;",
      expected: (lidx) => 3,
    },
    {
      name: "vec4InclusiveToExclusive",
      kernelBody: "let inc = vec4InclusiveScan(vec4<u32>(1u, 2u, 3u, 4u)); outputBuffer[gid] = vec4InclusiveToExclusive(inc).z;",
      expected: (lidx) => 3,
    },
    {
      name: "vec4Reduce",
      kernelBody: "outputBuffer[gid] = vec4Reduce(vec4<u32>(1u, 2u, 3u, 4u));",
      expected: (lidx) => 10,
    },
    {
      name: "vec4ScalarBinopV4",
      kernelBody: "let v = vec4ScalarBinopV4(2u, vec4<u32>(1u, 2u, 3u, 4u)); outputBuffer[gid] = v.y;",
      expected: (lidx) => 4,
    },
    {
      name: "isSubgroupZero",
      kernelBody: "outputBuffer[gid] = select(0u, 1u, isSubgroupZero(builtinsNonuniform.lidx, sgsz));",
      expected: (lidx, sgsz) => (lidx < sgsz ? 1 : 0),
    },
    {
      name: "subgroupShuffle",
      kernelBody: "outputBuffer[gid] = subgroupShuffle(builtinsNonuniform.lidx, sgid ^ 1u);",
      expected: (lidx, sgsz) => {
        const base = lidx - (lidx % sgsz);
        return base + ((lidx % sgsz) ^ 1);
      },
    },
    {
      name: "subgroupBallot",
      kernelBody: "let pred = (builtinsNonuniform.lidx % 2u) == 1u; let ballot = subgroupBallot(pred); outputBuffer[gid] = ballot.x;",
      expected: (lidx, sgsz) => {
        if (sgsz === 32) return 0xaaaaaaaa >>> 0;
        if (sgsz === 64) return 0xaaaaaaaa >>> 0;
        return 0xaaaaaaaa >>> 0;
      },
    },
    {
      name: "subgroupReduce",
      kernelBody: "outputBuffer[gid] = subgroupReduce(1u);",
      expected: (lidx, sgsz) => sgsz,
    },
    {
      name: "subgroupAdd",
      kernelBody: "outputBuffer[gid] = subgroupAdd(1u);",
      expected: (lidx, sgsz) => sgsz,
    },
    {
      name: "subgroupInclusiveOpScan",
      kernelBody: "let val = builtinsNonuniform.lidx % sgsz; outputBuffer[gid] = subgroupInclusiveOpScan(val, sgid, sgsz);",
      expected: (lidx, sgsz) => {
        const sgid = lidx % sgsz;
        return (sgid * (sgid + 1)) / 2;
      },
    },
    {
      name: "subgroupExclusiveAdd",
      kernelBody: "outputBuffer[gid] = subgroupExclusiveAdd(1u);",
      expected: (lidx, sgsz) => lidx % sgsz,
    },
    /* wgScan on v(i) = (7i + 3) mod 11 across the workgroup */
    {
      name: "wgScan exclusive (add)",
      kernelBody:
        "var total = 0u; let v = (builtinsNonuniform.lidx * 7u + 3u) % 11u; outputBuffer[gid] = wgExclusiveScan(v, &total, builtinsUniform, builtinsNonuniform);",
      expected: (lidx) => {
        let s = 0;
        for (let i = 0; i < lidx; i++) s += (i * 7 + 3) % 11;
        return s;
      },
    },
    {
      name: "wgScan inclusive (add)",
      kernelBody:
        "var total = 0u; let v = (builtinsNonuniform.lidx * 7u + 3u) % 11u; outputBuffer[gid] = wgInclusiveScan(v, &total, builtinsUniform, builtinsNonuniform);",
      expected: (lidx) => {
        let s = 0;
        for (let i = 0; i <= lidx; i++) s += (i * 7 + 3) % 11;
        return s;
      },
    },
    {
      /* the totals of two back-to-back calls (the scan's memory can be reused at once) */
      name: "wgScan totals, back-to-back calls",
      kernelBody:
        "var total = 0u; let v = (builtinsNonuniform.lidx * 7u + 3u) % 11u; let a = wgExclusiveScan(v, &total, builtinsUniform, builtinsNonuniform); var total2 = 0u; let b = wgInclusiveScan(1u, &total2, builtinsUniform, builtinsNonuniform); outputBuffer[gid] = total * 1000u + total2 + (b - builtinsNonuniform.lidx - 1u);",
      expected: () => {
        let s = 0;
        for (let i = 0; i < 256; i++) s += (i * 7 + 3) % 11;
        return s * 1000 + 256;
      },
    },
    {
      /* a binop with no native subgroup scan: max of v(i) = (37i) mod 101 */
      name: "wgScan exclusive (max)",
      binop: new BinOpMax({ datatype: "u32" }),
      kernelBody:
        "var total = 0u; let v = (builtinsNonuniform.lidx * 37u) % 101u; outputBuffer[gid] = wgExclusiveScan(v, &total, builtinsUniform, builtinsNonuniform);",
      expected: (lidx) => {
        let m = 0;
        for (let i = 0; i < lidx; i++) m = Math.max(m, (i * 37) % 101);
        return m;
      },
    },
    {
      name: "wgScan inclusive (max), total",
      binop: new BinOpMax({ datatype: "u32" }),
      kernelBody:
        "var total = 0u; let v = (builtinsNonuniform.lidx * 37u) % 101u; let p = wgInclusiveScan(v, &total, builtinsUniform, builtinsNonuniform); outputBuffer[gid] = p * 1000u + total;",
      expected: (lidx) => {
        let m = 0;
        for (let i = 0; i <= lidx; i++) m = Math.max(m, (i * 37) % 101);
        return m * 1000 + 100;
      },
    },
  ];

  /* wgHistogram: bin lidx % 7 for every invocation, plus one sample in no
   * bin; bins 0..6 count 37, 37, 37, 37, 36, 36, 36 of 256 */
  tests.push({
    name: "wgHistogram (2 copies, BIN_NONE ignored)",
    kernelBody:
      "let l = builtinsNonuniform.lidx; histClear(l); workgroupBarrier(); histAdd(l % 7u, l); histAdd(0xffffffffu, l); workgroupBarrier(); outputBuffer[gid] = select(0u, histCount(l), l < 16u);",
    expected: (lidx) => (lidx >= 7 ? 0 : 36 + (lidx < 256 % 7 ? 1 : 0)),
  });

  const hasSubgroups = device.features.has("subgroups");
  const modes = [
    { name: "hardware", disableSubgroups: false, enabled: hasSubgroups },
    { name: "emulated", disableSubgroups: true, enabled: true },
  ];

  for (const mode of modes) {
    if (!mode.enabled) continue;

    resultsCallback({ type: "mode_start", mode: mode.name });

    for (const test of tests) {
      try {
        const primitive = new TestWGSLFunctionsPrimitive({
          device,
          datatype: "u32",
          workgroupSize: 256,
          disableSubgroups: mode.disableSubgroups,
          kernelBody: test.kernelBody,
          ...(test.binop && { binop: test.binop }),
        });

        await primitive.execute();
        const outputBuffer = primitive.getBuffer("outputBuffer");
        await outputBuffer.copyGPUToCPU();
        const results = outputBuffer.cpuBuffer;

        const sgsz = mode.disableSubgroups ? 32 : (device.adapterInfo?.subgroupMinSize ?? 32);

        let passed = true;
        let errorMsg = "";
        for (let i = 0; i < results.length; i++) {
          const expectedVal = test.expected(i, sgsz);
          if (results[i] !== expectedVal) {
            passed = false;
            errorMsg = `At index ${i}: expected ${expectedVal}, got ${results[i]}`;
            break;
          }
        }

        resultsCallback({
          type: "test_result",
          mode: mode.name,
          name: test.name,
          passed,
          errorMsg,
        });
      } catch (err) {
        resultsCallback({
          type: "test_result",
          mode: mode.name,
          name: test.name,
          passed: false,
          errorMsg: err.stack || err.message,
        });
      }
    }
  }

  resultsCallback({ type: "suite_end" });
}
