/*Histogram Strategy Builder

/*Maximum bins that fit in workgroup shared memory */
export const MAX_SHARED_MEMORY_BINS = 256;

/*Default workgroup size*/
export const DEFAULT_WORKGROUP_SIZE = 256;

/*Values processed per thread (coalesced memory access)*/
export const VAL_PER_THREAD = 1;

/*Target GPU occupancy multiplier */
const OCCUPANCY_MULTIPLIER = 4; // GPU will handle grid-stride efficiently

/*Maximum workgroups - WebGPU limit per dimension */
const MAX_WORKGROUPS = 65535;


/*Datatype helper class*/

function createUnifiedUniformsBuffer(inputSize, numBins, minValue, binScale) {
    const data = new ArrayBuffer(16);
    const view = new DataView(data);
    view.setUint32(0, inputSize, true);
    view.setUint32(4, numBins, true);
    view.setFloat32(8, minValue, true);
    view.setFloat32(12, binScale, true);
    return data;
}



export function arithmeticBinCPU(inputValue, min, max, numBins) {
    // Accept all values, clamp to range (matches GPU behavior)
    const scale = numBins / (max - min);
    const normalized = (inputValue - min) * scale;
    let bin = Math.floor(normalized);

    // Clamp for out-of-bounds and floating-point edge cases
    bin = Math.max(0, Math.min(bin, numBins - 1));

    return bin;
}

export function lookupBinCPU(inputValue, bin_edges) {
    if (inputValue < bin_edges[0] || inputValue >= bin_edges[bin_edges.length - 1]) {
        return -1;
    }


    // Binary search for upper bound
    let left = 0;
    let right = bin_edges.length - 1;


    while (left < right - 1) {
        const mid = Math.floor((left + right) / 2);
        if (inputValue < bin_edges[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }

    // Verify inputValue is in valid range [left, right)
    if (inputValue >= bin_edges[left] && inputValue < bin_edges[right]) {
        return left;
    }

    return -1; // Out of bounds
}


// WGSL generation


/* Generate WGSL code for arithmetic binning (even bins)*/

function generateArithmeticWGSL(min, max, numBins, datatype) {
    return `
fn bin_value(value: ${datatype}) -> u32 {
    let min_value: f32 = ${min};
    let bin_scale: f32 = ${numBins} / (${max} - ${min});
    
    let normalized: f32 = (f32(value) - min_value) * bin_scale;
    var bin: i32 = i32(floor(normalized));
    
    bin = clamp(bin, 0, i32(uniforms.numBins) - 1);
    
    return u32(bin);
}
`;
}
/*fn bin_value(value: ${datatype}) -> u32 {
  // Read min and scale from uniforms (allows dynamic configuration)
  let min_value: f32 = uniforms.minValue;
  let bin_scale: f32 = uniforms.binScale;
  
  // Convert value to normalized bin index
  let normalized: f32 = (f32(value) - min_value) * bin_scale;
  var bin: i32 = i32(floor(normalized));
  
  // Clamp to valid range [0, numBins)
  bin = clamp(bin, 0, i32(uniforms.numBins) - 1);
  
  return u32(bin);
}

/* Generate WGSL code for lookup binning (binary search)*/

function generateLookupWGSL(bin_edges, datatype) {
    return `
fn bin_values(value: ${datatype}, bin_edges: ptr<storage, array<f32>>) -> u32 {
    let num_edges: u32 = ${bin_edges.length}u;
    let value_f32: f32 = f32(value);
    if(value_f32<bin_edges[0] || value_f32>=bin_edges[num_edges-1u]){
        return 0xFFFFFFFFu;
    }

    var left:u32=0u;
    var right:u32=num_edges-1u;
    while(left<right-1u){
        let mid:u32=(left+right)/2u;
        if(value_f32<bin_edges[mid]){
            right=mid;
        }else{
            left=mid;
        }
    }
    if(value_f32>=bin_edges[left] && value_f32<bin_edges[right]){
        return left;
    }
    return 0xFFFFFFFFu;
}
`;

}


/*Validate even bins configuration */
function validateEvenBins(bins) {

    //positive bins
    if (typeof bins.numBins !== 'number' || bins.numBins <= 0) {
        throw new Error(
            `HistogramStrategy: bins.numBins must be positive integer, got ${bins.numBins}`
        );
    }
    //finite min
    if (!Number.isFinite(bins.min)) {
        throw new Error(
            `HistogramStrategy: bins.min must be finite number, got ${bins.min}`
        );
    }
    //finite max
    if (!Number.isFinite(bins.max)) {
        throw new Error(
            `HistogramStrategy: bins.max must be finite number, got ${bins.max}`
        );
    }

    if (bins.min >= bins.max) {
        throw new Error(
            `HistogramStrategy: bins.min (${bins.min}) must be < bins.max (${bins.max})`
        );
    }

    // Check for arithmetic overflow in scale computation(not very small ranges or not very large ranges to avoid infinities)
    const range = bins.max - bins.min;
    const scale = bins.numBins / range;

    if (!Number.isFinite(scale)) {
        throw new Error(
            `HistogramStrategy: Bin scale computation overflow. ` +
            `Range [${bins.min}, ${bins.max}] with ${bins.numBins} bins is too large.`
        );
    }

    // Warn if too many bins-const (MAX_SHARED_MEMORY_BINS = 256);
    if (bins.numBins > MAX_SHARED_MEMORY_BINS) {
        console.warn(
            `HistogramStrategy: ${bins.numBins} bins exceeds shared memory limit (${MAX_SHARED_MEMORY_BINS}). ` +
            `Will use slower global atomics. Consider reducing bin count.`
        );
    }
}

/*Validate custom bins configuration*/

function validateCustomBins(bins) {

    if (!Array.isArray(bins.bin_edges)) {
        throw new Error('HistogramStrategy: bins.bin_edges must be an array');
    }

    if (bins.bin_edges.length < 2) {
        throw new Error(
            `HistogramStrategy: bins.bin_edges must have at least 2 boundaries, got ${bins.bin_edges.length}`
        );
    }
    for (let i = 1; i < bins.bin_edges.length; i++) {
        if (bins.bin_edges[i] <= bins.bin_edges[i - 1]) {
            throw new Error(`HistogramStrategy: bin_edges must be monotonically increasing (found ${bins.bin_edges[i - 1]} >= ${bins.bin_edges[i]} at index ${i})`);
        }
    }


    // Check all bin_edges are finite
    for (let i = 0; i < bins.bin_edges.length; i++) {
        if (!Number.isFinite(bins.bin_edges[i])) {
            throw new Error(
                `HistogramStrategy: bins.bin_edges[${i}] = ${bins.bin_edges[i]} is not finite`
            );
        }
    }

    // Check strictly ascending order
    for (let i = 1; i < bins.bin_edges.length; i++) {
        if (bins.bin_edges[i] <= bins.bin_edges[i - 1]) {
            throw new Error(
                `HistogramStrategy: bins.bin_edges must be strictly ascending. ` +
                `Found bin_edges[${i - 1}] = ${bins.bin_edges[i - 1]} >= bin_edges[${i}] = ${bins.bin_edges[i]}`
            );
        }
    }

    // Warn if many bin_edges (slow binary search)
    if (bins.bin_edges.length > 1000) {
        console.warn(
            `HistogramStrategy: ${bins.bin_edges.length} bin edges may cause slow binary search. ` +
            `Consider reducing number of bins.`
        );
    }
}

/*Calculate workgroup dimensions */

function calculateDimensions(inputSize, workgroupSize = DEFAULT_WORKGROUP_SIZE) {

    //workgroupSize=256
    const valuesPerWorkgroup = workgroupSize * VAL_PER_THREAD;

    // Calculate tiles needed to cover all input values
    const tilesNeeded = Math.ceil(inputSize / valuesPerWorkgroup);
    // Let small inputs use fewer workgroups naturally
    let workgroupCount = Math.min(tilesNeeded, MAX_WORKGROUPS);

    // Estimate occupancy
    const estimatedSMCount = 100;
    const targetOccupancy = estimatedSMCount * OCCUPANCY_MULTIPLIER;
    const estimatedOccupancy = Math.min(workgroupCount / targetOccupancy, 1.0);

    return {
        workgroupSize,
        workgroupCount,
        valuesPerThread: VAL_PER_THREAD,
        valuesPerWorkgroup,
        dispatchDimensions: [workgroupCount, 1, 1],
        estimatedOccupancy
    };
}




/*Build histogram execution configuration*/
function planBuffers(numBins, inputSize, bins) {
    const buffers = {
        output: {
            size: numBins * 4, // u32 = 4 bytes per bin
            usage: 'STORAGE | COPY_SRC'
        }
    };

    if (bins.type === 'custom') {
        const edgesArray = new Float32Array(bins.bin_edges);
        buffers.bin_edges = {
            size: edgesArray.byteLength,
            usage: 'STORAGE | COPY_DST',
            data: edgesArray
        };
    }

    return buffers;
}

export function buildHistogramStrategy(userConfig) {
    const { bins, datatype, inputSize } = userConfig;

    let strategyType;
    let numBins;
    let binningFunction;
    let generateWGSL;

    if (bins.type === 'even') {
        validateEvenBins(bins);
        strategyType = 'arithmetic';
        numBins = bins.numBins;
        binningFunction = (value) => arithmeticBinCPU(value, bins.min, bins.max, numBins);
        generateWGSL = () => generateArithmeticWGSL(bins.min, bins.max, numBins, datatype);

    } else if (bins.type === 'custom') {
        validateCustomBins(bins);
        strategyType = 'lookup';
        numBins = bins.bin_edges.length - 1;
        binningFunction = (value) => lookupBinCPU(value, bins.bin_edges);
        generateWGSL = () => generateLookupWGSL(bins.bin_edges, datatype);

    } else {
        throw new Error(
            `HistogramStrategy: Unknown bins.type '${bins.type}'. Expected 'even' or 'custom'.`
        );
    }

    const useSharedMemory = (numBins <= MAX_SHARED_MEMORY_BINS);
    const dimensions = calculateDimensions(inputSize);

    // Plan buffers
    const buffers = planBuffers(numBins, inputSize, bins);

    // Create unified uniform buffer
    if (bins.type === 'even') {
        const binScale = numBins / (bins.max - bins.min);
        buffers.uniforms = {
            size: 16,
            usage: 'UNIFORM | COPY_DST',
            data: createUnifiedUniformsBuffer(inputSize, numBins, bins.min, binScale)
        };
    } else {
        buffers.uniforms = {
            size: 16,
            usage: 'UNIFORM | COPY_DST',
            data: createUnifiedUniformsBuffer(inputSize, numBins, 0.0, 0.0)
        };
    }

    return {
        strategyType,
        numBins,
        useSharedMemory,
        binningFunction,
        generateBinningWGSL: generateWGSL,
        ...dimensions,
        buffers,
        metadata: {
            strategyType,
            numBins,
            useSharedMemory,
            workgroupSize: dimensions.workgroupSize,
            workgroupCount: dimensions.workgroupCount,
            estimatedOccupancy: dimensions.estimatedOccupancy,
        }
    };
}




