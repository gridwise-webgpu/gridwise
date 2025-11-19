import { range, arrayProd, datatypeToTypedArray, datatypeToBytes } from "./util.mjs";
import {
    BasePrimitive,
    Kernel,
    AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddU32, BinOpAddF32 } from "./binop.mjs";
import { arithmeticBinCPU, lookupBinCPU, buildHistogramStrategy } from "./histogramStrategies.mjs";
import { generateHistogramInitKernel, generateHistogramSweepKernel } from "./histogramKernels.mjs";


class BaseHistogram extends BasePrimitive {

    constructor(args) {
        super(args);

        // Required parameters 
        for (const required of ["datatype"]) {
            if (!this[required]) {
                throw new Error(`${this.constructor.name}: ${required} is required`);
            }
        }
        if (args.bins) {
            if (args.bins.type === 'even') {
                this.binType = 'even';
                this.numBins = args.bins.numBins;
                if (!this.numBins) {
                    throw new Error(`${this.constructor.name}: bins must have 'numBins'`);
                }
                this.minValue = args.bins.min ?? 0.0;
                this.maxValue = args.bins.max ?? 1.0;
            } else if (args.bins.type === 'custom') {
                this.binType = 'custom';
                this.binEdges = args.bins.bin_edges;

                this.numBins = this.binEdges.length - 1;
                this.minValue = this.binEdges[0];
                this.maxValue = this.binEdges[this.binEdges.length - 1];
            } else {
                throw new Error(`${this.constructor.name}: unknown bins type ${args.bins.type}`);
            }

        } else {
            if (!args.numBins) {
                throw new Error(`${this.constructor.name}: numBins is required`);
            }
            this.binType = 'even';
            this.numBins = args.numBins;
            this.minValue = args.minValue ?? 0.0;
            this.maxValue = args.maxValue ?? 1.0;
        }


        // Optional: binop parameter (defaults to BinOpAddU32 for counting)--can be sum, min, max(for further extensions)
        this.binop = args.binop ?? BinOpAddU32;

        if (!this.binop) {
            throw new Error(`${this.constructor.name}: binop is required`);
        }

        this.knownBuffers = [
            "inputBuffer",
            "outputBuffer",
            "uniforms"
        ];

        for (const knownBuffer of this.knownBuffers) {
            if (knownBuffer in args) {
                this.registerBuffer({
                    label: knownBuffer,
                    buffer: args[knownBuffer],
                    device: this.device,
                });
                delete this[knownBuffer];
            }
        }

        this.outputDatatype = this.binop.datatype;

    }
    //if bandwidth is low check this 
    get bytesTransferred() {
        // Histogram memory traffic:
        // 1. Init kernel: Write entire output buffer (numBins * 4 bytes)
        // 2. Sweep kernel:
        //    - Read entire input buffer (inputLength * datatype_size)
        //    - Atomic RMW on output buffer: Each input does read+write (2 * 4 bytes * inputLength)
        //    Conservative estimate: read input once + write output once
        const inputBytes = this.getBuffer("inputBuffer").size;
        const outputBytes = this.getBuffer("outputBuffer").size;

        // For histogram: input read once + output init + output atomic writes
        return inputBytes + (2 * outputBytes); // Read input, init output, update output
    }

    finalizeRuntimeParameters() {
        const inputBuffer = this.getBuffer("inputBuffer");
        const inputSize = inputBuffer.size / datatypeToBytes(this.datatype);

        //rebuild config to ensure correct min/max values- buildStrategy.mjs

        this.config = buildHistogramStrategy({
            bins: this.binType === 'custom'
                ? { type: 'custom', bin_edges: this.binEdges, min: this.minValue, max: this.maxValue }
                : { type: 'even', numBins: this.numBins, min: this.minValue, max: this.maxValue },
            datatype: this.datatype,
            inputSize: inputSize,
            binOp: this.binop
        });
        this.config.initKernel = generateHistogramInitKernel(this.config.numBins);
        this.config.sweepKernel = generateHistogramSweepKernel(this.config, this.datatype);
    }



    compute() {
        this.finalizeRuntimeParameters();
        const operations = [];

        operations.push(
            new AllocateBuffer({
                label: "outputBuffer",
                size: this.config.buffers.output.size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            })
        );


        // Unified uniform buffer for both init and sweep kernels
        operations.push(
            new AllocateBuffer({
                label: "uniforms",
                size: this.config.buffers.uniforms.size,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                populateWith: new Uint32Array(this.config.buffers.uniforms.data)
            })
        );

        // Bin edges buffer (if custom bins)
        if (this.config.buffers.bin_edges) {
            operations.push(
                new AllocateBuffer({
                    label: "bin_edges",
                    size: this.config.buffers.bin_edges.size,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                    populateWith: this.config.buffers.bin_edges.data
                })
            );
        }

        operations.push(
            new Kernel({
                kernel: () => this.config.initKernel,
                entryPoint: "histogramInitKernel",
                bufferTypes: [["storage", "uniform"]],
                bindings: [["outputBuffer", "uniforms"]],
                label: "histogram init",
                getDispatchGeometry: () => this.config.dispatchDimensions
            })

        );

        // Sweep kernel
        operations.push(
            new Kernel({
                kernel: () => this.config.sweepKernel,
                entryPoint: "histogramKernel",
                bufferTypes: this.config.buffers.bin_edges
                    ? [["read-only-storage", "storage", "uniform", "read-only-storage"]]
                    : [["read-only-storage", "storage", "uniform"]],
                bindings: this.config.buffers.bin_edges
                    ? [["inputBuffer", "outputBuffer", "uniforms", "bin_edges"]]
                    : [["inputBuffer", "outputBuffer", "uniforms"]],
                label: "histogram sweep",
                getDispatchGeometry: () => this.config.dispatchDimensions
            })
        );

        return operations;
    }

    validate = (args = {}) => {
        const memsrc = args.inputBuffer ?? this.getBuffer("inputBuffer")?.cpuBuffer;
        const memdest = args.outputBuffer ?? this.getBuffer("outputBuffer")?.cpuBuffer;

        if (!memsrc || !memdest) {
            return "";
        }

        const OutputArrayType = {
            'u32': Uint32Array,
            'i32': Int32Array,
            'f32': Float32Array
        }[this.outputDatatype] ?? Uint32Array;

        let referenceOutput = new OutputArrayType(this.numBins);

        // Initialize based on binop
        for (let bin = 0; bin < this.numBins; bin++) {
            referenceOutput[bin] = this.binop.identity ?? 0;
        }

        // Process each sample
        for (let i = 0; i < memsrc.length; i++) {
            const value = memsrc[i];
            let binIndex = -1;
            //binary serach for custom bins
            if (this.binType === 'custom') {
                binIndex = lookupBinCPU(value, this.binEdges);
            } else {
                // Arithmetic binning 
                binIndex = arithmeticBinCPU(value, this.minValue, this.maxValue, this.numBins);
            }

            // Ignore out-of-bounds inputs
            if (binIndex >= 0 && binIndex < this.numBins) {
                // For histogram: increment count by 1 (not by the value!)
                referenceOutput[binIndex] = this.binop.op(
                    referenceOutput[binIndex],
                    1
                );
            }
        }

        // Compare results
        let returnString = "";
        let allowedErrors = 5;

        for (let bin = 0; bin < this.numBins; bin++) {
            if (allowedErrors == 0) break;
            if (referenceOutput[bin] != memdest[bin]) {
                returnString += `\nBin ${bin}: expected ${referenceOutput[bin]}, got ${memdest[bin]}.`;
                allowedErrors--;
            }
        }

        return returnString;
    };


    getBinMetadata() {
        if (this.binType === 'even') {
            const binWidth = (this.maxValue - this.minValue) / this.numBins;
            return Array.from({ length: this.numBins }, (_, i) => ({
                index: i,
                range: [
                    this.minValue + i * binWidth,
                    this.minValue + (i + 1) * binWidth
                ],
                label: `[${(this.minValue + i * binWidth).toFixed(3)}, ${(this.minValue + (i + 1) * binWidth).toFixed(3)})`
            }));
        } else {
            // Custom bins
            return Array.from({ length: this.numBins }, (_, i) => ({
                index: i,
                range: [this.binEdges[i], this.binEdges[i + 1]],
                label: `[${this.binEdges[i]}, ${this.binEdges[i + 1]})`
            }));
        }
    }


    getHistogramResult(customData = null) {
        const counts = customData || this.getBuffer("outputBuffer").cpuBuffer;

        if (!counts) {
            console.warn('getHistogramResult: No histogram data available');
            return [];
        }

        const metadata = this.getBinMetadata();

        return metadata.map((bin, i) => ({
            bin: i,
            range: bin.range,
            label: bin.label,
            count: counts[i]
        }));
    }
    /**
 */
    getBinIndex(value) {
        if (this.binType === 'even') {
            // Arithmetic binning (same as GPU kernel)
            const scale = this.numBins / (this.maxValue - this.minValue);
            const normalized = (value - this.minValue) * scale;
            const bin = Math.floor(normalized);

            // Clamp to valid range
            if (bin < 0 || bin >= this.numBins) {
                return -1;
            }
            return bin;
        } else {
            // Custom bins - binary search (same as GPU kernel)
            if (value < this.binEdges[0] || value >= this.binEdges[this.binEdges.length - 1]) {
                return -1;  // Out of bounds
            }

            // Binary search to find bin
            let left = 0;
            let right = this.numBins;

            while (left < right) {
                const mid = Math.floor((left + right) / 2);
                if (value < this.binEdges[mid]) {
                    right = mid;
                } else if (value >= this.binEdges[mid + 1]) {
                    left = mid + 1;
                } else {
                    return mid;
                }
            }

            return -1;  // Shouldn't reach here
        }
    }

    /**/
    getBinIndices(values) {
        const indices = new Uint32Array(values.length);

        for (let i = 0; i < values.length; i++) {
            const binIndex = this.getBinIndex(values[i]);
            indices[i] = binIndex === -1 ? 0xFFFFFFFF : binIndex;  // Use max uint32 for out-of-bounds
        }

        return indices;
    }
}


// Test Suite Configuration

export const histogramBandwidthPlot = {
    x: { field: "inputBytes", label: "Input array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    fx: { field: "timing" },
    stroke: { field: "inputLength" },
    test_br: "gpuinfo.description",
    caption: "Lines are input length",
};

export const histogramTimePlot = {
    x: { field: "inputBytes", label: "Input array size (B)" },
    y: { field: "cpuTotalTimeMS", label: "Time (ms)" },
    fx: { field: "timing" },
    stroke: { field: "inputLength" },
    test_br: "gpuinfo.description",
    caption: "Lines are input length",
};

export const histogramBinTypePlot = {
    x: { field: "inputBytes", label: "Input array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    fx: { field: "timing" },
    stroke: { field: "binType" },
    test_br: "gpuinfo.description",
    caption: "Lines are bin type (even/custom)",
};

export const HistogramEvenBinsBWPlot = {
    x: { field: "inputBytes", label: "Input array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    stroke: { field: "timing" },
    test_br: "gpuinfo.description",
    caption: "CPU timing (performance.now), GPU timing (timestamps)",
};

export const HistogramCustomBinsBWPlot = {
    x: { field: "inputBytes", label: "Input array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    stroke: { field: "timing" },
    test_br: "gpuinfo.description",
    caption: "CPU timing (performance.now), GPU timing (timestamps)",
};

const HistogramParams = {
    inputLength: range(8, 27).map((i) => 2 ** i),
    numBins: [64],
};

// eslint-disable-next-line no-unused-vars
const HistogramParamsSingleton = {
    inputLength: [2 ** 27],
    numBins: [64],
};

export const EvenBinsHistogramTestSuite = new BaseTestSuite({
    category: "histogram",
    testSuite: "even bins histogram",
    trials: 20,
    params: HistogramParams,
    uniqueRuns: ["inputLength", "numBins"],
    primitive: BaseHistogram,
    primitiveArgs: {
        datatype: "f32",
        binop: BinOpAddU32,
        bins: { type: 'even', numBins: 64, min: 0.0, max: 100.0 },
        gputimestamps: true,
    },
    plots: [
        HistogramEvenBinsBWPlot,
    ],
});

export const CustomBinsHistogramTestSuite = new BaseTestSuite({
    category: "histogram",
    testSuite: "custom bins histogram",
    trials: 10,
    params: HistogramParams,
    uniqueRuns: ["inputLength", "numBins"],
    primitive: BaseHistogram,
    primitiveArgs: {
        datatype: "f32",
        binop: BinOpAddU32,
        bins: {
            type: 'custom',
            bin_edges: [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]
        },
        gputimestamps: true,
    },
    plots: [
        HistogramCustomBinsBWPlot,
    ],
});

