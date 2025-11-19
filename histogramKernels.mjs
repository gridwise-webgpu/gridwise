/* */


export function generateHistogramInitKernel(numBins) {
    return `
@group(0) @binding(0) var<storage, read_write> output: array<atomic<u32>>;  // ← Add @ symbol!

struct Uniforms {
    inputSize: u32,
    numBins: u32,
    minValue: f32,
    binScale: f32,
}

@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn histogramInitKernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx < uniforms.numBins) {
        atomicStore(&output[idx], 0u);
    }
}
`;
}

/**
 * Generate WGSL for histogram sweep kernel (KERNEL 2)
 * Purpose: Compute histogram using 3-phase:
 *   Phase 1: Initialize workgroup-private histogram (shared memory)
 *   Phase 2: Process input with grid-stride loop
 *   Phase 3: Merge workgroup results to global histogram

 */
export function generateHistogramSweepKernel(config, datatype) {
    const { strategyType, numBins, useSharedMemory, workgroupSize } = config;
    const hasBinEdges = strategyType === 'lookup';

    // Get binning function from strategy(It's a function that generates the binning logic code where it tells the GPU HOW to decide which bin a value belongs to.)
    const binningFunctionWGSL = config.generateBinningWGSL();

    const sharedMemoryDecl = useSharedMemory ? `
var<workgroup> privateHistogram: array<atomic<u32>, ${numBins}>;
` : '';

    const bindings = `
@group(0) @binding(0) var<storage, read> input: array<${datatype}>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
${hasBinEdges ? '@group(0) @binding(3) var<storage, read> bin_edges: array<f32>;' : ''}
`;


    const uniformsStruct = `
struct Uniforms {
    inputSize: u32,
    numBins: u32,
    minValue: f32,
    binScale: f32,
}
`;

    const mainKernel = `
@compute @workgroup_size(${workgroupSize})
fn histogramKernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) numWorkgroups: vec3<u32>
) {
    let localIdx = lid.x;
    let globalIdx = gid.x;
    
    ${useSharedMemory ? `
    // Each thread initializes multiple bins using stride loop
    var b = localIdx;
    while (b < uniforms.numBins) {
        atomicStore(&privateHistogram[b], 0u);
        b += ${workgroupSize}u;
    }
    workgroupBarrier();
    ` : ''}
    
    // PHASE 2: Process samples (grid-stride loop)
    var idx = globalIdx;
    let stride = numWorkgroups.x * ${workgroupSize}u;
    
    while (idx < uniforms.inputSize) {
        let sample = input[idx];
        
        ${strategyType === 'lookup'
            ? 'let bin = bin_values(sample, &bin_edges);'
            : 'let bin = bin_value(sample);'
        }
        
    
        if (bin != 0xFFFFFFFFu && bin < uniforms.numBins) {
            ${useSharedMemory
            ? 'atomicAdd(&privateHistogram[bin], 1u);'
            : 'atomicAdd(&output[bin], 1u);'}
        }
        
        idx += stride;
    }

    // 
    // Each thread merges multiple bins using stride loop.
    ${useSharedMemory ? `
    workgroupBarrier();
    
    var bin = localIdx;
    while (bin < uniforms.numBins) {
        let count = atomicLoad(&privateHistogram[bin]);
        if (count > 0u) {
            atomicAdd(&output[bin], count);
        }
        bin += ${workgroupSize}u;
    }
    ` : ''}
}
`;

    const generatedCode = `
${bindings}
${uniformsStruct}
${sharedMemoryDecl}
${binningFunctionWGSL}
${mainKernel}
`;


    return generatedCode;
}
