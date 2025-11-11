import { range, arrayProd, datatypeToTypedArray, datatypeToBytes, createUniformBuffer } from "./util.mjs";
import {
    BasePrimitive,
    Kernel,
    AllocateBuffer,
} from "./primitive.mjs";
import { BaseTestSuite } from "./testsuite.mjs";
import { BinOpAddU32, BinOpAddF32 } from "./binop.mjs";



export class BaseHistogram extends BasePrimitive {
    constructor(args) {
        super(args);

        // Required parameters 
        for (const required of ["datatype", "numBins"]) {
            if (!this[required]) {
                throw new Error(`${this.constructor.name}: ${required} is required`);
            }
        }

        // Histogram always outputs u32 
        this.binop = args.binop ?? new BinOpAddU32();

        if (this.binop.datatype !== "u32") {
            throw new Error(
                `${this.constructor.name}: binop datatype must be u32 (histogram output is always u32), but got ${this.binop.datatype}.`
            );
        }
        this.minValue = args.minValue ?? 0.0;
        this.maxValue = args.maxValue ?? 1.0;


        this.knownBuffers = ["inputBuffer", "outputBuffer"];

        for (const knownBuffer of this.knownBuffers) {
            if (knownBuffer in args) {
                this.registerBuffer({ label: knownBuffer, buffer: args[knownBuffer] });
                delete this[knownBuffer];
            }
        }
        this.getDispatchGeometry = this.getSimpleDispatchGeometry;
    }

    get bytesTransferred() {
        return (
            this.getBuffer("inputBuffer").size + this.getBuffer("outputBuffer").size
        );
    }

    validate = (args = {}) => {
        const memsrc = args.inputBuffer ?? this.getBuffer("inputBuffer").cpuBuffer;
        const memdest = args.outputBuffer ?? this.getBuffer("outputBuffer").cpuBuffer;
        let referenceOutput;
        try {
            referenceOutput = new Uint32Array(this.numBins);
        } catch (error) {
            console.error(error, "Tried to allocate array of length", this.numBins);
        }
        for (let bin = 0; bin < this.numBins; bin++) {
            referenceOutput[bin] = 0;
        }
        for (let i = 0; i < memsrc.length; i++) {
            const value = memsrc[i];

            const normalized = (value - this.minValue) / (this.maxValue - this.minValue);
            let binIndex = Math.floor(normalized * this.numBins);
            binIndex = Math.max(0, Math.min(binIndex, this.numBins - 1));
            referenceOutput[binIndex] = referenceOutput[binIndex] + 1;
        }
        function validates(args) {
            return args.cpu == args.gpu;
        }

        let returnString = "";
        let allowedErrors = 5;

        for (let bin = 0; bin < memdest.length; bin++) {
            if (allowedErrors == 0) {
                break;
            }
            if (
                !validates({
                    cpu: referenceOutput[bin],
                    gpu: memdest[bin],
                    datatype: this.datatype,
                })
            ) {
                const ref = referenceOutput[bin];
                const gpu = memdest[bin];
                const diff = ref === 0 ? Math.abs(gpu - ref) : Math.abs((ref - gpu) / ref);
                returnString += `\nBin ${bin}: expected ${ref}, instead saw ${gpu} (diff: ${diff}).`;
                if (this.getBuffer("debugBuffer")) {
                    returnString += ` debug[${bin}] = ${this.getBuffer("debugBuffer").cpuBuffer[bin]}.`;
                }
                if (this.getBuffer("debug2Buffer")) {
                    returnString += ` debug2[${bin}] = ${this.getBuffer("debug2Buffer").cpuBuffer[bin]}.`;
                }
                allowedErrors--;
            }
        }

        console.log(
            this.label,
            "histogram",
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
            "bins:",
            this.numBins,
            "range:",
            this.minValue,
            "to",
            this.maxValue,
            "input length:",
            memsrc.length
        );

        return returnString;
    };
}

export const histogramBandwidthPlot = {
    x: { field: "inputBytes", label: "Input array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    stroke: { field: "timing" },  // Lines colored by GPU vs CPU
    text_br: "gpuinfo.description",
    caption: "Histogram Bandwidth (GPU vs CPU)",
};

function histogramWGCountFnPlot() {
    return {
        x: { field: "inputBytes", label: "Input array size (B)" },
        y: { field: (d) => d.bandwidth, label: "Achieved bandwidth (GB/s)" },
        stroke: { field: "workgroupCount" },
        text_br: (d) => `${d.gpuinfo.description}`,
        caption: `${this.category} | ${this.testSuite} | Lines are workgroup count`,
    };
}

const histogramWGSizeBinOpPlot = {
    x: { field: "inputBytes", label: "Input array size (B)" },
    y: { field: "bandwidth", label: "Achieved bandwidth (GB/s)" },
    fy: { field: "binop" },
    stroke: { field: "workgroupSize" },
    text_br: "gpuinfo.description",
    caption: "Lines are workgroup size",
};

//https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/

/*Step 1: clear private bins (like setting to 0).
Step 2: each thread counts elements into its workgroup’s private histogram. 
Step 3: merge your private histogram into the final global result. */

export class WGHistogram extends BaseHistogram {
    constructor(args) {
        super(args);
    }
    finalizeRuntimeParameters() {
        this.workgroupSize = this.workgroupSize ?? 256;
        this.maxGSLWorkgroupCount = this.maxGSLWorkgroupCount ?? 256;

        this.workgroupCount = Math.min(Math.ceil(this.getBuffer("inputBuffer").size / this.workgroupSize), this.maxGSLWorkgroupCount);
        this.numPartials = this.workgroupCount;

        this.inputLength = this.getBuffer("inputBuffer").size / datatypeToBytes(this.datatype);

        this.histogramUniformsBuffer = createUniformBuffer([
            { type: 'u32', value: this.inputLength },
            { type: 'u32', value: this.numBins },
            { type: 'u32', value: this.workgroupCount },
            { type: 'f32', value: this.minValue },
            { type: 'f32', value: this.maxValue },
        ]);

    }
    histogramKernelDefinition = () => {
        return /*wgsl*/ `
    @group(0) @binding(0) var<storage, read> inputBuffer: array<${this.datatype}>;
    @group(0) @binding(1) var<storage,read_write>outputBuffer:array<atomic<u32>>;
    
    struct HistogramUniforms{
        inputLength:u32,
        numBins:u32,
        numWorkgroups:u32,
        minValue:f32,
        maxValue:f32,
    }
    @group(0) @binding(2) var<uniform> uniforms:HistogramUniforms;

    var<workgroup>privateHistogram:array<atomic<u32>,${this.numBins}>;

    @compute @workgroup_size(${this.workgroupSize})
    fn main(
        @builtin(global_invocation_id)globalId:vec3<u32>,
        @builtin(local_invocation_id)localId:vec3<u32>,
        @builtin(workgroup_id)wg_id:vec3<u32>
    ) {
        let gwIndex:u32 =globalId.x;
        let localIndex:u32=localId.x;
        let wgIndex:u32=wg_id.x;

        var i:u32=localIndex;
        let WGS: u32=${this.workgroupSize}u;
        let NB:u32=uniforms.numBins;

        loop{
            if(i>=NB){break;}
            atomicExchange(&privateHistogram[i],0u);
            i=i+WGS;
        }
        workgroupBarrier();

        var idx: u32 = gwIndex;
        let inputLen: u32 = uniforms.inputLength;
        
        while (idx < inputLen) {
            let value: ${this.datatype} = inputBuffer[idx];
            
            let range: f32 = uniforms.maxValue - uniforms.minValue;
            let normalized: f32 = (f32(value) - uniforms.minValue) / range;
            var binIndex: i32 = i32(floor(normalized * f32(NB)));
            binIndex = clamp(binIndex, 0, i32(NB) - 1);
            
            atomicAdd(&privateHistogram[u32(binIndex)], 1u);

            idx = idx + uniforms.numWorkgroups * WGS; 
        }

        workgroupBarrier();

        var b: u32 = localIndex;
        while (b < NB) {
            let partialCount: u32 = atomicLoad(&privateHistogram[b]);
            if (partialCount > 0u) {
                atomicAdd(&outputBuffer[b], partialCount);
            }
            b = b + WGS;
        }
    }
    `;
    }

    compute() {
        this.finalizeRuntimeParameters();

        return [
            new AllocateBuffer({
                label: "histogramUniforms",
                size: this.histogramUniformsBuffer.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                populateWith: this.histogramUniformsBuffer,
            }),

            new Kernel({
                kernel: this.histogramKernelDefinition,
                bufferTypes: [["read-only-storage", "storage", "uniform"]],
                bindings: [["inputBuffer", "outputBuffer", "histogramUniforms"]],
                label: "histogram kernel",
                getDispatchGeometry: () => {
                    return [this.workgroupCount];
                },
            }),
        ];
    }
}

/*https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/ */

export class HierarchicalHistogram extends BaseHistogram {
    constructor(args) {
        super(args);
    }
    finalizeRuntimeParameters() {
        this.workgroupSize = this.workgroupSize ?? 256;
        this.maxGSLWorkgroupCount = this.maxGSLWorkgroupCount ?? 512;
        this.numThreadsPerWorkgroup = arrayProd(this.workgroupSize);

        this.inputLength = this.getBuffer("inputBuffer").size / datatypeToBytes(this.datatype);
        const idealWorkgroupCount = Math.ceil(this.inputLength / this.numThreadsPerWorkgroup);
        this.workgroupCount = Math.min(idealWorkgroupCount, this.maxGSLWorkgroupCount);
        this.numPartials = this.workgroupCount;

        this.histogramUniformsBuffer = createUniformBuffer([
            { type: 'u32', value: this.inputLength },
            { type: 'u32', value: this.numBins },
            { type: 'u32', value: this.workgroupCount },
            { type: 'f32', value: this.minValue },
            { type: 'f32', value: this.maxValue },
        ]);

        this.clearUniformsBuffer = createUniformBuffer([
            { type: 'u32', value: this.numBins }
        ]);

        this.accumulateDispatchCount = this.numBins;
        this.accumulateUniformsBuffer = createUniformBuffer([
            { type: 'u32', value: this.numBins },
            { type: 'u32', value: this.workgroupCount }
        ]);
    }
    // Kernel 1: Each workgroup builds its own local histogram using workgroup atomics
    histogramPerWorkgroupKernel = () => {
        return /*wgsl*/`
        @group(0) @binding(0) var<storage, read>inputBuffer:array<${this.datatype}>;
        @group(0) @binding(1) var<storage,read_write>partials:array<u32>;

        struct HistogramUniforms{
            inputLength:u32,
            numBins:u32,
            numWorkgroups:u32,
            minValue:f32,
            maxValue:f32,
        }
        @group(0) @binding(2) var<uniform>uniforms:HistogramUniforms;

        var<workgroup> privateHistogram: array<atomic<u32>, ${this.numBins}>;
        
        @compute @workgroup_size(${this.workgroupSize})
        fn histogramPerWorkgroupKernel(
            @builtin(global_invocation_id) globalId: vec3<u32>,
            @builtin(local_invocation_id) localId: vec3<u32>,
            @builtin(workgroup_id) wgId: vec3<u32>
        ) {
            let gwIndex: u32 = globalId.x;
            let localIndex: u32 = localId.x;
            let wgIndex: u32 = wgId.x;
            
            let WGS: u32 = ${this.workgroupSize}u;
            let NB: u32 = uniforms.numBins;
            
            var i: u32 = localIndex;
            loop {
                if (i >= NB) { break; }
                atomicStore(&privateHistogram[i], 0u);
                i = i + WGS;
            }
            workgroupBarrier();
            
            let inputLen: u32 = uniforms.inputLength;
            
            var idx: u32 = gwIndex;
            
            while (idx < inputLen) {
                let value: ${this.datatype} = inputBuffer[idx];
                
                let range: f32 = uniforms.maxValue - uniforms.minValue;
                let normalized: f32 = (f32(value) - uniforms.minValue) / range;
                var binIndex: i32 = i32(floor(normalized * f32(NB)));
                binIndex = clamp(binIndex, 0, i32(NB) - 1);
                
                atomicAdd(&privateHistogram[u32(binIndex)], 1u);
                
                idx = idx + uniforms.numWorkgroups * WGS;
            }
            
            workgroupBarrier();
            
            var b: u32 = localIndex;
            while (b < NB) {
                let partialCount: u32 = atomicLoad(&privateHistogram[b]);
                let partialIndex: u32 = b * uniforms.numWorkgroups + wgIndex;
                partials[partialIndex] = partialCount;
                b = b + WGS;
            }
        }`;
    };
    // kernel 2: for accumulated partial
    accumulateHistogramsKernel = () => {
        return /* wgsl */ `
        @group(0) @binding(0) var<storage, read> partials: array<u32>;
        @group(0) @binding(1) var<storage, read_write> outputBuffer: array<atomic<u32>>;

        struct AccumulateUniforms {
            numBins: u32,
            numWorkgroups: u32,
        }
        @group(0) @binding(2) var<uniform> uniforms: AccumulateUniforms;
        
        var<workgroup> localSum: atomic<u32>;
        
        @compute @workgroup_size(${this.workgroupSize})
        fn accumulateHistogramsKernel(
            @builtin(global_invocation_id) globalId: vec3<u32>,
            @builtin(local_invocation_id) localId: vec3<u32>,
            @builtin(workgroup_id) workgroupId: vec3<u32>
        ) {
            let localIdx: u32 = localId.x;
            let binIdx: u32 = workgroupId.x;
            let NB: u32 = uniforms.numBins;
            let numWG: u32 = uniforms.numWorkgroups;
            let WGS: u32 = ${this.workgroupSize}u;
            
            if (binIdx >= NB) {
                return;
            }
            
            if (localIdx == 0u) {
                atomicStore(&localSum, 0u);
            }
            workgroupBarrier();
            
            var wgIdx: u32 = localIdx;
            while (wgIdx < numWG) {
                let partialIndex: u32 = binIdx * numWG + wgIdx;
                let value: u32 = partials[partialIndex];
                atomicAdd(&localSum, value);
                wgIdx = wgIdx + WGS;
            }
            
            workgroupBarrier();
            
            if (localIdx == 0u) {
                let total: u32 = atomicLoad(&localSum);
                atomicStore(&outputBuffer[binIdx], total);
            }
        }`;
    };

    // Kernel to clear the output buffer
    clearOutputBufferKernel = () => {
        return /* wgsl */ `
        @group(0) @binding(0) var<storage, read_write> outputBuffer: array<atomic<u32>>;
        
        struct ClearUniforms {
            numBins: u32,
        }
        @group(0) @binding(1) var<uniform> uniforms: ClearUniforms;
        
        @compute @workgroup_size(${this.workgroupSize})
        fn clearOutputBufferKernel(
            @builtin(global_invocation_id) globalId: vec3<u32>
        ) {
            let idx: u32 = globalId.x;
            if (idx < uniforms.numBins) {
                atomicStore(&outputBuffer[idx], 0u);
            }
        }`;
    };

    compute() {
        this.finalizeRuntimeParameters();

        return [
            // Allocate partials buffer: transposed layout [numBins][numWorkgroups]
            new AllocateBuffer({
                label: "partials",
                size: this.numBins * this.workgroupCount * 4, // 4 bytes per u32
            }),

            // Allocate and populate histogram uniforms
            new AllocateBuffer({
                label: "histogramUniforms",
                size: this.histogramUniformsBuffer.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                populateWith: this.histogramUniformsBuffer,
            }),

            // Allocate and populate accumulate uniforms
            new AllocateBuffer({
                label: "accumulateUniforms",
                size: this.accumulateUniformsBuffer.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                populateWith: this.accumulateUniformsBuffer,
            }),

            // Kernel 1: Each workgroup builds local histogram from contiguous chunks
            new Kernel({
                kernel: this.histogramPerWorkgroupKernel,
                bufferTypes: [["read-only-storage", "storage", "uniform"]],
                bindings: [["inputBuffer", "partials", "histogramUniforms"]],
                label: "histogram per workgroup",
                logKernelCodeToConsole: false,
                getDispatchGeometry: () => {
                    return [this.workgroupCount];
                },
            }),

            // Kernel 2: Accumulate all per-workgroup histograms (NO ATOMICS!)
            new Kernel({
                kernel: this.accumulateHistogramsKernel,
                bufferTypes: [["read-only-storage", "storage", "uniform"]],
                bindings: [["partials", "outputBuffer", "accumulateUniforms"]],
                label: "accumulate histograms",
                logKernelCodeToConsole: false,
                getDispatchGeometry: () => {
                    // One workgroup per bin = numBins workgroups
                    return [this.numBins];
                },
            }),
        ];
    }

}

const HistogramParams = {
    inputLength: [2 ** 20, 2 ** 22, 2 ** 24, 2 ** 26, 2 ** 27],
    numBins: [64, 256],
    workgroupSize: [256],
    maxGSLWorkgroupCount: [256, 512, 1024],
    minValue: [-1024.0],
    maxValue: [1024.0],
};

const HistogramParamsSingleton = {
    inputLength: [2 ** 10],
    numBins: [64],
    workgroupSize: [256],
    maxGSLWorkgroupCount: [64],
    minValue: [-1024.0],
    maxValue: [1024.0],
};

export const WGHistogramTestSuite = new BaseTestSuite({
    category: "histogram",
    testSuite: "workgroup histogram",
    trials: 10,
    params: HistogramParams,
    uniqueRuns: ["inputLength", "numBins", "workgroupSize"],
    primitive: WGHistogram,
    primitiveArgs: {
        datatype: "f32",
        binop: BinOpAddU32,
        gputimestamps: true,
    },
    plots: [
        histogramBandwidthPlot
    ],
});

export const HierarchicalHistogramTestSuite = new BaseTestSuite({
    category: "histogram",
    testSuite: "hierarchical histogram",
    trials: 10,
    params: HistogramParams,
    uniqueRuns: ["inputLength", "numBins", "maxGSLWorkgroupCount"],
    primitive: HierarchicalHistogram,
    primitiveArgs: {
        datatype: "f32",
        binop: BinOpAddU32,
        gputimestamps: true,
    },
    plots: [
        histogramBandwidthPlot
    ],
});
