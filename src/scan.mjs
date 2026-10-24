import { range } from "./util.mjs";
import {
  BasePrimitive,
  Kernel,
  AllocateBuffer,
  ExecutePrimitive,
  UniformStruct,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddU32 } from "./binop.mjs";
import { datatypeToTypedArray, datatypeToBytes } from "./util.mjs";

// exports: TestSuites, Primitives

export class BaseScan extends BasePrimitive {
  constructor(args) {
    super(args);

    // default scan is exclusive
    this.type = args.type ?? "exclusive";
    if (
      this.type != "inclusive" &&
      this.type != "exclusive" &&
      this.type != "reduce"
    ) {
      throw new Error(
        `${this.constructor.name}: scan type (currently ${this.type}) must be {inclusive, exclusive, reduce}.`
      );
    }

    for (const required of ["datatype", "binop"]) {
      if (!this[required]) {
        throw new Error(
          `${this.constructor.name}: ${required} is a required parameter.`
        );
      }
    }

    if (this.datatype != this.binop.datatype) {
      throw new Error(
        `${this.constructor.name}: datatype (${this.datatype}) is incompatible with binop datatype (${this.binop.datatype}).`
      );
    }

    this.knownBuffers = ["inputBuffer", "outputBuffer"];

    for (const knownBuffer of this.knownBuffers) {
      /* we passed an existing buffer into the constructor */
      if (knownBuffer in args) {
        this.registerBuffer({ label: knownBuffer, buffer: args[knownBuffer] });
        delete this[knownBuffer]; // let's make sure it's in one place only
      }
    }

    /* by default, delegate to simple call from BasePrimitive */
    this.getDispatchGeometry = this.getSimpleDispatchGeometry;
  }

  get bytesTransferred() {
    return (
      this.getBuffer("inputBuffer").size + this.getBuffer("outputBuffer").size
    );
  }

  validate = (args = {}) => {
    /** if we pass in buffers, use them, otherwise use the named buffers
     * that are stored in the primitive */
    /* assumes that cpuBuffers are populated with useful data */
    const memsrc = args.inputBuffer ?? this.getBuffer("inputBuffer").cpuBuffer;
    const memdest =
      args.outputBuffer ?? this.getBuffer("outputBuffer").cpuBuffer;
    let referenceOutput;
    try {
      referenceOutput = new (datatypeToTypedArray(this.datatype))(
        memdest.length
      );
    } catch (error) {
      console.error(error, "Tried to allocate array of length", memdest.length);
    }
    for (let i = 0; i < memsrc.length; i++) {
      switch (this.type) {
        case "exclusive":
          referenceOutput[i] =
            i == 0
              ? this.binop.identity
              : this.binop.op(referenceOutput[i - 1], memsrc[i - 1]);
          break;
        case "inclusive":
          referenceOutput[i] = this.binop.op(
            i == 0 ? this.binop.identity : referenceOutput[i - 1],
            memsrc[i]
          );
          break;
        case "reduce":
          referenceOutput[0] = this.binop.op(
            i == 0 ? this.binop.identity : referenceOutput[0],
            memsrc[i]
          );
          break;
      }
    }
    function validates(args) {
      return args.cpu == args.gpu;
    }
    let returnString = "";
    let allowedErrors = 5;
    for (let i = 0; i < memdest.length; i++) {
      if (allowedErrors == 0) {
        break;
      }
      if (
        !validates({
          cpu: referenceOutput[i],
          gpu: memdest[i],
          datatype: this.datatype,
        })
      ) {
        returnString += `\nElement ${i}: expected ${
          referenceOutput[i]
        }, instead saw ${memdest[i]} (diff: ${Math.abs(
          (referenceOutput[i] - memdest[i]) / referenceOutput[i]
        )}).`;
        if (this.getBuffer("debugBuffer")) {
          returnString += ` debug[${i}] = ${
            this.getBuffer("debugBuffer").cpuBuffer[i]
          }.`;
        }
        if (this.getBuffer("debug2Buffer")) {
          returnString += ` debug2[${i}] = ${
            this.getBuffer("debug2Buffer").cpuBuffer[i]
          }.`;
        }
        allowedErrors--;
      }
    }
    if (returnString !== "") {
      console.log(
        this.label,
        this.type,
        "with input",
        memsrc,
        "should validate to",
        referenceOutput,
        "and actually validates to",
        memdest,
        this.getBuffer("debugBuffer") ? "\ndebugBuffer" : "",
        this.getBuffer("debugBuffer")
          ? this.getBuffer("debugBuffer").cpuBuffer
          : "",
        this.getBuffer("debug2Buffer") ? "\ndebug2Buffer" : "",
        this.getBuffer("debug2Buffer")
          ? this.getBuffer("debug2Buffer").cpuBuffer
          : "",
        this.binop.constructor.name,
        this.binop.datatype,
        "identity is",
        this.binop.identity,
        "length is",
        memsrc.length,
        "memsrc[",
        memsrc.length - 1,
        "] is",
        memsrc[memsrc.length - 1]
      );
    }
    return returnString;
  };
}

export const scanWGSizePlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  fx: { field: "timing" },
  stroke: { field: "workgroupSize" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

export const scanWGCountPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  fx: { field: "timing" },
  stroke: { field: "workgroupCount" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup count",
};

/* needs to be a function if we do string interpolation */
// eslint-disable-next-line no-unused-vars
function scanWGCountFnPlot() {
  return {
    x: { field: "inputBytes", label: "Input array size (B)" },
    /* y.field is just showing off that we can have an embedded function */
    /* { field: "bandwidth", ...} would do the same */
    y: { field: (d) => d.bandwidth, label: "Achieved bandwidth (GB/s)" },
    stroke: { field: "workgroupCount" },
    text_br: (d) => `${d.gpuinfo.description}`,
    caption: `${this.category} | ${this.testSuite} | Lines are workgroup count`,
  };
}

// eslint-disable-next-line no-unused-vars
const scanWGSizeBinOpPlot = {
  x: { field: "inputBytes", label: "Input array size (B)" },
  y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
  fy: { field: "binop" },
  stroke: { field: "workgroupSize" },
  test_br: "gpuinfo.description",
  caption: "Lines are workgroup size",
};

/**
 * HierarchicalScan: a scan (exclusive, inclusive or reduce) built in the
 * classic reduce-then-scan way, from one workgroup function (wgScan) and
 * composition (ExecutePrimitive). Not chained like DLDFScan, so it makes no
 * forward-progress assumptions and needs no subgroups: without them it uses
 * only workgroup barriers.
 *
 * The input is cut into tiles of workgroupSize * itemsPerThread elements;
 * invocation i of a tile owns itemsPerThread consecutive elements.
 *
 *   one tile:   tile_scan          scan the tile (or reduce it) directly
 *   more:       tile_reduce        each tile's reduction -> partials
 *               (HierarchicalScan) exclusive scan of partials -> carries,
 *                                  itself a HierarchicalScan, run with
 *                                  ExecutePrimitive: the recursion ends when
 *                                  the partials fit in one tile
 *               tile_scan          scan each tile, starting from its carry
 *   reduce:     tile_reduce, then a HierarchicalScan reduce of the partials
 *               (one tile: tile_reduce writes the result directly)
 *
 * Memory traffic: one read for the reductions, then one read and one write
 * for the scan (about 3n elements, against about 2n for DLDFScan).
 *
 * Options (besides BaseScan's datatype, binop, type): workgroupSize (default
 * 256), itemsPerThread (default 4).
 */
export class HierarchicalScan extends BaseScan {
  constructor(args) {
    super(args);
    this.workgroupSize = this.workgroupSize ?? 256;
    this.itemsPerThread = this.itemsPerThread ?? 4;
    if (!("label" in args)) {
      this.label += this.makeParamString(["type", "datatype", "binop", "useSubgroups"]);
    }
    this.params = new UniformStruct("ScanParameters", { n: "u32", numTiles: "u32" });
  }

  finalizeRuntimeParameters() {
    this.inputLength = this.getBuffer("inputBuffer").size / datatypeToBytes(this.datatype);
    this.tileSize = this.workgroupSize * this.itemsPerThread;
    this.numTiles = Math.max(1, Math.ceil(this.inputLength / this.tileSize));
    this.workgroupCount = this.numTiles;
  }

  /** The scan of the per-tile partials: another HierarchicalScan (exclusive,
   * or reduce for a reduce), created once and reused. */
  get partialsScan() {
    const type = this.type === "reduce" ? "reduce" : "exclusive";
    if (this.__partialsScan?.type !== type) {
      this.__partialsScan = new HierarchicalScan({
        device: this.device,
        datatype: this.datatype,
        binop: this.binop,
        type,
        workgroupSize: this.workgroupSize,
        itemsPerThread: this.itemsPerThread,
        ...(this.disableSubgroups && { disableSubgroups: true }),
        ...(this.adapter && { adapter: this.adapter }),
      });
    }
    return this.__partialsScan;
  }

  /** WGSL for one tile kernel. mode: "reduce" writes each tile's reduction
   * to out[tile]; "scan" writes the scan of each tile, starting from
   * carries[tile] when withCarries. */
  tileKernel = ({ mode, withCarries }) => {
    const T = this.datatype;
    const K = this.itemsPerThread;
    const W = this.workgroupSize;
    const inclusive = this.type === "inclusive";
    return /* wgsl */ `
${this.fnDeclarations.include("wgScan", "computeLinearizedGridParametersSplit")}

const K = ${K}u;

@compute @workgroup_size(${W})
fn main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  if (wgid >= params.numTiles) { return; }
  let lidx = builtinsNonuniform.lidx;
  /* this invocation's K consecutive elements: load them and scan them in order */
  let first = wgid * ${W * K}u + lidx * K;
  var items: array<${T}, K>;
  var threadTotal: ${T} = ${this.binop.identity};
  for (var k = 0u; k < K; k++) {
    let i = first + k;
    let x = select(${this.binop.identity}, inputBuffer[min(i, params.n - 1u)], i < params.n);
    ${mode === "scan" ? (inclusive ? "threadTotal = binop(threadTotal, x);\n    items[k] = threadTotal;" : "items[k] = threadTotal;\n    threadTotal = binop(threadTotal, x);") : "threadTotal = binop(threadTotal, x);"}
  }
  /* combine the invocations' totals across the workgroup */
  var tileTotal: ${T};
  let prefix = wgExclusiveScan(threadTotal, &tileTotal, builtinsUniform, builtinsNonuniform);
  ${
    mode === "reduce"
      ? `if (lidx == 0u) { out[wgid] = tileTotal; }`
      : `let carry = ${withCarries ? "binop(carries[wgid], prefix)" : "prefix"};
  for (var k = 0u; k < K; k++) {
    let i = first + k;
    if (i < params.n) { out[i] = binop(carry, items[k]); }
  }`
  }
}`;
  };

  compute() {
    this.finalizeRuntimeParameters();
    const T = this.datatype;
    const actions = [this.params.allocate("scanParameters", { n: this.inputLength, numTiles: this.numTiles })];
    const dispatch = () => this.getSimpleDispatchGeometry({ workgroupCount: this.numTiles });
    const tile = (mode, withCarries, outName) =>
      new Kernel({
        kernel: () => this.tileKernel({ mode, withCarries }),
        entryPoint: "main",
        buffers: [
          { name: "inputBuffer", access: "read", type: `array<${T}>` },
          { name: "out", buffer: outName, access: "read_write", type: `array<${T}>` },
          { name: "params", buffer: "scanParameters", uniform: this.params },
          ...(withCarries ? [{ name: "carries", access: "read", type: `array<${T}>` }] : []),
        ],
        label: `HierarchicalScan tile_${mode} (${this.numTiles} tiles)`,
        getDispatchGeometry: dispatch,
      });
    if (this.numTiles === 1) {
      /* one tile: scan or reduce it directly */
      actions.push(tile(this.type === "reduce" ? "reduce" : "scan", false, "outputBuffer"));
      return actions;
    }
    /* exactly numTiles elements: the nested scan sizes itself from this buffer,
     * and padding would add elements (wrong for min, max, multiply) */
    const partialsSize = this.numTiles * datatypeToBytes(T);
    actions.push(
      new AllocateBuffer({ label: "partials", size: partialsSize, clearBufferOnReuse: false }),
      tile("reduce", false, "partials")
    );
    if (this.type === "reduce") {
      actions.push(
        new ExecutePrimitive({
          primitive: this.partialsScan,
          buffers: { inputBuffer: "partials", outputBuffer: "outputBuffer" },
          label: "reduce of the tile reductions",
        })
      );
      return actions;
    }
    actions.push(
      new AllocateBuffer({ label: "carries", size: partialsSize, clearBufferOnReuse: false }),
      new ExecutePrimitive({
        primitive: this.partialsScan,
        buffers: { inputBuffer: "partials", outputBuffer: "carries" },
        label: "exclusive scan of the tile reductions",
      }),
      tile("scan", true, "outputBuffer")
    );
    return actions;
  }

  destroy() {
    super.destroy();
    this.__partialsScan?.destroy();
  }
}

const ScanParams = {
  inputLength: range(8, 24).map((i) => 2 ** i),
  workgroupSize: range(5, 8).map((i) => 2 ** i),
};

// eslint-disable-next-line no-unused-vars
const ScanParamsSingleton = {
  inputLength: [2 ** 7],
  maxGSLWorkgroupCount: [2 ** 7],
  workgroupSize: [2 ** 7],
};

export const HierarchicalScanTestSuite = new BaseTestSuite({
  category: "scan",
  testSuite: "hierarchical scan",
  trials: 10,
  params: ScanParams,
  uniqueRuns: ["inputLength", "workgroupSize"],
  primitive: HierarchicalScan,
  primitiveArgs: {
    datatype: "u32",
    binop: BinOpAddU32,
  },
  plots: [
    { ...scanWGSizePlot, ...{ fy: { field: "workgroupSize" } } },
    { ...scanWGCountPlot, ...{ fy: { field: "workgroupCount" } } },
  ],
});
