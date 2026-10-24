/**
 * Histogram: count how many input samples fall in each bin, like CUB's
 * DeviceHistogram (HistogramEven, HistogramRange).
 *
 *   const h = new Histogram({ device, bins: new EvenBins({ datatype: "f32", numBins: 256, lower: 0, upper: 1 }) });
 *   await h.execute({ inputBuffer, outputBuffer });  // outputBuffer: numBins u32 counts
 *
 * bins is a bin function from bins.mjs (EvenBins, RangeBins, RadixBins); its
 * datatype is the samples' datatype. Samples outside every bin are not
 * counted. The output is zeroed first (it needs COPY_DST usage), so it holds
 * this execute()'s counts, which are exact: integer atomics give the same
 * result in any order.
 *
 * How: each workgroup counts a grid-strided share of the input into a
 * histogram in workgroup memory (wgHistogram, shared with MultisplitPerm),
 * then adds its nonzero counts into the output with atomics. When the bins do
 * not fit in workgroup memory, it counts into the output directly, as CUB
 * does.
 *
 * Options: workgroupSize (256), itemsPerThread (16; sets the workgroup count),
 * copies (copies of the bins per workgroup, against atomic contention;
 * default 4 for <= 256 bins, 2 for <= 1024, else 1), maxWorkgroups (8192).
 */
import { BasePrimitive, Kernel, AllocateBuffer, ClearBuffer } from "./primitive.mjs";
import { datatypeToBytes } from "./util.mjs";

export class Histogram extends BasePrimitive {
  constructor(args) {
    super(args);
    if (!this.bins) throw new Error("Histogram: bins (EvenBins, RangeBins or RadixBins) is required");
    this.datatype ??= this.bins.datatype;
    if (this.datatype !== this.bins.datatype) {
      throw new Error(`Histogram: datatype ${this.datatype} does not match the bins' datatype ${this.bins.datatype}`);
    }
    this.numBins = this.bins.numBins;
    this.workgroupSize ??= 256;
    this.itemsPerThread ??= 16;
    this.maxWorkgroups ??= 8192;
    this.copies ??= this.numBins <= 256 ? 4 : this.numBins <= 1024 ? 2 : 1;
    this.knownBuffers = ["inputBuffer", "outputBuffer"];
    for (const knownBuffer of this.knownBuffers) {
      if (knownBuffer in args) {
        this.registerBuffer({ label: knownBuffer, buffer: args[knownBuffer] });
        delete this[knownBuffer];
      }
    }
    this.getDispatchGeometry = () => [this.workgroupCount];
  }

  /** do the bins (all copies) fit in workgroup memory? */
  get workgroupBins() {
    const limit = this.device.limits.maxComputeWorkgroupStorageSize;
    return this.numBins * this.copies * 4 <= limit;
  }

  get bytesTransferred() {
    return this.getBuffer("inputBuffer").size + this.numBins * 4;
  }

  finalizeRuntimeParameters() {
    this.inputLength = this.getBuffer("inputBuffer").size / datatypeToBytes(this.datatype);
    this.workgroupCount = Math.max(
      1,
      Math.min(Math.ceil(this.inputLength / (this.workgroupSize * this.itemsPerThread)), this.maxWorkgroups)
    );
    this.histogramParameters = new Uint32Array([this.inputLength, 0, 0, 0]);
  }

  histogramWGSL = () => {
    const T = this.datatype;
    const W = this.workgroupSize;
    const N = this.numBins;
    const ranged = this.bins.needsLevels;
    const wg = this.workgroupBins;
    return /* wgsl */ `
struct HistogramParameters { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> inputBuffer: array<${T}>;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: HistogramParameters;
${ranged ? `@group(0) @binding(3) var<storage, read> levels: array<${T}>;` : ""}
${this.bins.wgsl("binOf")}
${wg ? this.fnDeclarations.wgHistogram({ name: "wgBins", numBins: N, copies: this.copies, workgroupSize: W }) : ""}

@compute @workgroup_size(${W})
fn histogram(@builtin(local_invocation_index) lidx: u32,
             @builtin(workgroup_id) wid: vec3u,
             @builtin(num_workgroups) nwg: vec3u) {
  let stride = nwg.x * ${W}u;
${
  wg
    ? /* wgsl */ `  wgBinsClear(lidx);
  workgroupBarrier();
  for (var i = wid.x * ${W}u + lidx; i < params.n; i += stride) {
    wgBinsAdd(binOf(inputBuffer[i]), lidx);
  }
  workgroupBarrier();
  for (var b = lidx; b < ${N}u; b += ${W}u) {
    let c = wgBinsCount(b);
    if (c != 0u) { atomicAdd(&outputBuffer[b], c); }
  }`
    : /* wgsl */ `  /* bins too many for workgroup memory: count in the output directly */
  for (var i = wid.x * ${W}u + lidx; i < params.n; i += stride) {
    let b = binOf(inputBuffer[i]);
    if (b < ${N}u) { atomicAdd(&outputBuffer[b], 1u); }
  }`
}
}`;
  };

  compute() {
    this.finalizeRuntimeParameters();
    const ranged = this.bins.needsLevels;
    return [
      new AllocateBuffer({
        label: "histogramParameters",
        size: this.histogramParameters.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        populateWith: this.histogramParameters,
      }),
      ...(ranged
        ? [
            new AllocateBuffer({
              label: "levels",
              size: this.bins.levels.byteLength,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
              populateWith: this.bins.levels,
              clearBufferOnReuse: false,
            }),
          ]
        : []),
      new ClearBuffer({ label: "outputBuffer" }),
      new Kernel({
        kernel: this.histogramWGSL,
        entryPoint: "histogram",
        bufferTypes: [["read-only-storage", "storage", "uniform", ...(ranged ? ["read-only-storage"] : [])]],
        bindings: [["inputBuffer", "outputBuffer", "histogramParameters", ...(ranged ? ["levels"] : [])]],
        label: `histogram (${this.bins})`,
        /* with trials > 1, each dispatch must start from zero counts */
        resetBuffersForBenchmarkingOnly: ["outputBuffer"],
      }),
    ];
  }

  /** the counts on the CPU, from the bins' JavaScript mirror */
  reference(input) {
    const counts = new Uint32Array(this.numBins);
    for (let i = 0; i < input.length; i++) {
      const b = this.bins.binOf(input[i]);
      if (b >= 0) counts[b]++;
    }
    return counts;
  }

  validate = (args = {}) => {
    const input = args.inputBuffer ?? this.getBuffer("inputBuffer").cpuBuffer;
    const output = args.outputBuffer ?? this.getBuffer("outputBuffer").cpuBuffer;
    const expected = this.reference(input);
    let errors = "";
    let shown = 0;
    for (let b = 0; b < this.numBins && shown < 5; b++) {
      if (output[b] !== expected[b]) {
        errors += `\nbin ${b}: expected ${expected[b]}, got ${output[b]}`;
        shown++;
      }
    }
    return errors;
  };
}
