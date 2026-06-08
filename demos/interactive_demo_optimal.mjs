import { OneSweepSort } from "../onesweep.mjs";
import { DLDFScan } from "../scandldf.mjs";
import { BinOpAdd } from "../binop.mjs";

console.clear();

try {
if (!navigator.gpu) {
  showError("WebGPU is not available in this browser.");
  throw new Error("WebGPU not supported");
}

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredLimits: {
    maxComputeWorkgroupStorageSize: 32768,
  },
  requiredFeatures: adapter.features.has("subgroups") ? ["subgroups"] : [],
});

if (!device) {
  showError("WebGPU device creation failed.");
  throw new Error("WebGPU device creation failed");
}

const canvas = document.getElementById("canvas");
const context = canvas.getContext("webgpu");
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

context.configure({
  device,
  format: presentationFormat,
  alphaMode: "premultiplied",
});

let renderUniformBuffer = null;

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  if (device && renderUniformBuffer) {
    device.queue.writeBuffer(renderUniformBuffer, 0, new Float32Array([canvas.width, canvas.height, 0, 0]));
  }
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

// Global State
let particleCount = 10000;
let isOperating = false;
let currentOperation = null;
let mouseX = -1000;
let mouseY = -1000;
let mouseDown = false;
let attractMode = true;

const starSlider = document.getElementById("starSlider");
const starCountDisplay = document.getElementById("starCount");
const sortBtn = document.getElementById("sortBtn");
const scanBtn = document.getElementById("scanBtn");
const reduceBtn = document.getElementById("reduceBtn");

// Colors (HSL to RGB conversion for shader)
const COLORS = [
  [0.98, 0.22, 0.35], // Redish
  [0.95, 0.55, 0.15], // Orange
  [0.15, 0.85, 0.85], // Cyan
  [0.75, 0.25, 0.90], // Purple
  [0.20, 0.75, 0.95], // Blue
  [0.90, 0.90, 0.20], // Yellow
  [0.20, 0.85, 0.25], // Green
  [0.25, 0.25, 0.95], // Deep Blue
];

// GPU Buffers & Pipelines
let particlesBuffer = null;
let simulationUniformBuffer = null;
let colorBuffer = null;

let sorter = null;
let sortKeysBuffer = null;
let sortTempKeysBuffer = null;
let sortPayloadBuffer = null;
let sortTempPayloadBuffer = null;

let scanner = null;
let scanInputBuffer = null;
let scanOutputBuffer = null;
let scanMappableBuffer = null;

let reducer = null;
let reduceInputBuffer = null;
let reduceOutputBuffer = null;
let reduceMappableBuffer = null;

let computePipeline = null;
let initTargetsPipeline = null;
let renderPipeline = null;
let extractKeysPipeline = null;
let applySortedPipeline = null;
let applyScanPipeline = null;
let applyReducePipeline = null;
let applyScanBindGroup = null;
let applyReduceBindGroup = null;

// Initialize GPU Resources
function initGPUResources(count) {
  if (particlesBuffer) particlesBuffer.destroy();
  if (simulationUniformBuffer) simulationUniformBuffer.destroy();
  if (colorBuffer) colorBuffer.destroy();
  
  if (sortKeysBuffer) sortKeysBuffer.destroy();
  if (sortTempKeysBuffer) sortTempKeysBuffer.destroy();
  if (sortPayloadBuffer) sortPayloadBuffer.destroy();
  if (sortTempPayloadBuffer) sortTempPayloadBuffer.destroy();
  
  if (scanInputBuffer) scanInputBuffer.destroy();
  if (scanOutputBuffer) scanOutputBuffer.destroy();
  if (scanMappableBuffer) scanMappableBuffer.destroy();
  
  if (reduceInputBuffer) reduceInputBuffer.destroy();
  if (reduceOutputBuffer) reduceOutputBuffer.destroy();
  if (reduceMappableBuffer) reduceMappableBuffer.destroy();

  // Particle structure:
  // float x, y;
  // float originalX, originalY;
  // float vx, vy;
  // float targetX, targetY;
  // uint colorIndex;
  // float size;
  // Total size = 10 floats = 40 bytes per particle
  const particleBytes = count * 40;
  particlesBuffer = device.createBuffer({
    size: particleBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Populate initial particle data on CPU and write to GPU
  const initialParticles = new Float32Array(count * 10);
  for (let i = 0; i < count; i++) {
    const x = Math.random() * canvas.width;
    const y = Math.random() * canvas.height;
    const colorIndex = Math.floor(Math.random() * COLORS.length);
    const size = 0.4 + Math.random() * 1.2;

    const offset = i * 10;
    initialParticles[offset + 0] = x; // x
    initialParticles[offset + 1] = y; // y
    initialParticles[offset + 2] = x; // originalX
    initialParticles[offset + 3] = y; // originalY
    initialParticles[offset + 4] = (Math.random() - 0.5) * 0.5; // vx
    initialParticles[offset + 5] = (Math.random() - 0.5) * 0.5; // vy
    initialParticles[offset + 6] = x; // targetX
    initialParticles[offset + 7] = y; // targetY
    
    // Use Uint32 view to write integer colorIndex
    const view = new Uint32Array(initialParticles.buffer);
    view[offset + 8] = colorIndex; // colorIndex
    
    initialParticles[offset + 9] = size; // size
  }
  device.queue.writeBuffer(particlesBuffer, 0, initialParticles);

  // Colors buffer
  const colorData = new Float32Array(COLORS.flat());
  colorBuffer = device.createBuffer({
    size: colorData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(colorBuffer, 0, colorData);

  // Simulation Uniform Buffer
  simulationUniformBuffer = device.createBuffer({
    size: 32, // mousePos (vec2f), canvasSize (vec2f), mouseDown (u32), attractMode (u32), isOperating (u32), padding (f32) = 32 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Sort resources
  sorter = new OneSweepSort({
    device,
    datatype: "u32",
    type: "keyvalue",
    direction: "ascending",
    inputLength: count,
    copyOutputToTemp: true,
  });

  sortKeysBuffer = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  sortTempKeysBuffer = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  sortPayloadBuffer = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  sortTempPayloadBuffer = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Scan resources
  scanner = new DLDFScan({
    device,
    binop: new BinOpAdd({ datatype: "u32" }),
    type: "inclusive",
    datatype: "u32",
  });
  scanInputBuffer = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  scanOutputBuffer = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  scanMappableBuffer = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Reduce resources
  reducer = new DLDFScan({
    device,
    binop: new BinOpAdd({ datatype: "u32" }),
    type: "reduce",
    datatype: "u32",
  });
  reduceInputBuffer = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  reduceOutputBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  reduceMappableBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Re-build pipeline bindings
  buildBindGroups();
}

// Shader Declarations
const simulationWGSL = `
  struct Particle {
    pos: vec2f,
    origPos: vec2f,
    vel: vec2f,
    destination: vec2f,
    colorIndex: u32,
    size: f32,
  }

  struct Params {
    mouseX: f32,
    mouseY: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    mouseDown: u32,
    attractMode: u32,
    isOperating: u32,
    padding: f32,
  }

  @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
  @group(0) @binding(1) var<uniform> params: Params;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&particles)) { return; }

    var p = particles[index];

    if (params.isOperating == 0u) {
      let dx = params.mouseX - p.pos.x;
      let dy = params.mouseY - p.pos.y;
      let dist = sqrt(dx * dx + dy * dy);

      if (params.mouseDown == 1u && dist < 150.0 && dist > 0.1) {
        let force = ((150.0 - dist) / 150.0) * select(-0.4, 0.4, params.attractMode == 1u);
        p.vel.x += (dx / dist) * force;
        p.vel.y += (dy / dist) * force;
      }

      p.vel.x *= 0.99;
      p.vel.y *= 0.99;

      p.pos.x += p.vel.x;
      p.pos.y += p.vel.y;

      if (p.pos.x < 0.0 || p.pos.x > params.canvasWidth) { p.vel.x *= -0.5; }
      if (p.pos.y < 0.0 || p.pos.y > params.canvasHeight) { p.vel.y *= -0.5; }
      p.pos.x = max(0.0, min(params.canvasWidth, p.pos.x));
      p.pos.y = max(0.0, min(params.canvasHeight, p.pos.y));
    } else {
      p.pos.x += (p.destination.x - p.pos.x) * 0.08;
      p.pos.y += (p.destination.y - p.pos.y) * 0.08;
    }

    particles[index] = p;
  }
`;

const extractKeysWGSL = `
  struct Particle {
    pos: vec2f,
    origPos: vec2f,
    vel: vec2f,
    destination: vec2f,
    colorIndex: u32,
    size: f32,
  }

  @group(0) @binding(0) var<storage, read> particles: array<Particle>;
  @group(0) @binding(1) var<storage, read_write> keys: array<u32>;
  @group(0) @binding(2) var<storage, read_write> payloads: array<u32>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&particles)) { return; }

    keys[index] = u32(particles[index].size * 1000.0);
    payloads[index] = index;
  }
`;

const applySortedWGSL = `
  struct Particle {
    pos: vec2f,
    origPos: vec2f,
    vel: vec2f,
    destination: vec2f,
    colorIndex: u32,
    size: f32,
  }

  struct Params {
    mouseX: f32,
    mouseY: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    mouseDown: u32,
    attractMode: u32,
    isOperating: u32,
    padding: f32,
  }

  @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
  @group(0) @binding(1) var<storage, read> sortedIndices: array<u32>;
  @group(0) @binding(2) var<uniform> params: Params;

  // Simple pseudo-random helper
  fn hash(x: u32) -> f32 {
    var a = x;
    a = (a ^ 61u) ^ (a >> 16u);
    a = a + (a << 3u);
    a = a ^ (a >> 4u);
    a = a * 0x27d4eb2du;
    a = a ^ (a >> 15u);
    return f32(a) / 4294967296.0;
  }

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    let rank = id.x;
    let count = arrayLength(&particles);
    if (rank >= count) { return; }

    let origIdx = sortedIndices[rank];
    let progress = f32(rank) / f32(count);
    
    let usableWidth = params.canvasWidth - params.padding * 2.0;
    let usableHeight = params.canvasHeight - params.padding * 2.0;

    particles[origIdx].destination.x = params.padding + progress * usableWidth;
    particles[origIdx].destination.y = params.padding + (hash(rank) * 0.5 + 0.25) * usableHeight;
    particles[origIdx].vel = vec2f(0.0, 0.0);
  }
`;

const resetTargetsWGSL = `
  struct Particle {
    pos: vec2f,
    origPos: vec2f,
    vel: vec2f,
    destination: vec2f,
    colorIndex: u32,
    size: f32,
  }
  @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&particles)) { return; }
    particles[index].destination = particles[index].origPos;
  }
`;

const applyScanWGSL = `
  struct Particle {
    pos: vec2f,
    origPos: vec2f,
    vel: vec2f,
    destination: vec2f,
    colorIndex: u32,
    size: f32,
  }

  struct Params {
    mouseX: f32,
    mouseY: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    mouseDown: u32,
    attractMode: u32,
    isOperating: u32,
    padding: f32,
  }

  @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
  @group(0) @binding(1) var<storage, read> scannedValues: array<u32>;
  @group(0) @binding(2) var<uniform> params: Params;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    let count = arrayLength(&particles);
    if (idx >= count) { return; }

    var maxValue = scannedValues[count - 1u];
    if (maxValue == 0u) {
      maxValue = 1u;
    }

    let normalized = f32(scannedValues[idx]) / f32(maxValue);
    let wavePhase = normalized * 3.14159265 * 12.0;
    
    let usableWidth = params.canvasWidth - params.padding * 2.0;
    
    particles[idx].destination.x = params.padding + normalized * usableWidth;
    particles[idx].destination.y = params.canvasHeight / 2.0 + sin(wavePhase) * (params.canvasHeight * 0.3);
    particles[idx].vel = vec2f(0.0, 0.0);
  }
`;

const applyReduceWGSL = `
  struct Particle {
    pos: vec2f,
    origPos: vec2f,
    vel: vec2f,
    destination: vec2f,
    colorIndex: u32,
    size: f32,
  }

  struct Params {
    mouseX: f32,
    mouseY: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    mouseDown: u32,
    attractMode: u32,
    isOperating: u32,
    padding: f32,
  }

  @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
  @group(0) @binding(1) var<storage, read> reduceResult: array<u32>;
  @group(0) @binding(2) var<uniform> params: Params;

  fn hash(x: u32) -> f32 {
    var a = x;
    a = (a ^ 61u) ^ (a >> 16u);
    a = a + (a << 3u);
    a = a ^ (a >> 4u);
    a = a * 0x27d4eb2du;
    a = a ^ (a >> 15u);
    return f32(a) / 4294967296.0;
  }

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    let count = arrayLength(&particles);
    if (idx >= count) { return; }

    let totalSize = reduceResult[0];
    let meanSize = f32(totalSize) / f32(count);
    
    let maxRadius = min(params.canvasWidth, params.canvasHeight) * 0.45;
    let scaledRadius = maxRadius * min(meanSize / 700.0, 1.5);
    
    let centerX = params.canvasWidth / 2.0;
    let centerY = params.canvasHeight / 2.0;
    
    let progress = f32(idx) / f32(count);
    let angle = (f32(idx) * 137.5) * (3.14159265 / 180.0);
    let radius = sqrt(progress) * scaledRadius;
    
    particles[idx].destination.x = centerX + cos(angle) * radius;
    particles[idx].destination.y = centerY + sin(angle) * radius;
    particles[idx].vel = vec2f(0.0, 0.0);
  }
`;

const renderWGSL = `
  struct Particle {
    pos: vec2f,
    origPos: vec2f,
    vel: vec2f,
    destination: vec2f,
    colorIndex: u32,
    size: f32,
  }

  struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) color: vec3f,
    @location(1) uv: vec2f,
    @location(2) size: f32,
  }

  struct RenderParams {
    canvasWidth: f32,
    canvasHeight: f32,
    padding1: f32,
    padding2: f32,
  }

  @group(0) @binding(0) var<storage, read> particles: array<Particle>;
  @group(0) @binding(1) var<storage, read> colors: array<vec3f>;
  @group(0) @binding(2) var<uniform> renderParams: RenderParams;

  @vertex
  fn vs(
    @builtin(instance_index) instanceIdx: u32,
    @builtin(vertex_index) vertexIdx: u32
  ) -> VertexOutput {
    let p = particles[instanceIdx];
    let size = p.size * 2.5;

    // Define standard quad coordinates
    var quadOffset = vec2f(0.0, 0.0);
    var uv = vec2f(0.0, 0.0);
    if (vertexIdx == 0u) {
      quadOffset = vec2f(-1.0, -1.0);
      uv = vec2f(-1.0, -1.0);
    } else if (vertexIdx == 1u) {
      quadOffset = vec2f(1.0, -1.0);
      uv = vec2f(1.0, -1.0);
    } else if (vertexIdx == 2u) {
      quadOffset = vec2f(-1.0, 1.0);
      uv = vec2f(-1.0, 1.0);
    } else {
      quadOffset = vec2f(1.0, 1.0);
      uv = vec2f(1.0, 1.0);
    }

    // Convert pixel position to NDC [-1, 1]
    let canvasSize = vec2f(renderParams.canvasWidth, renderParams.canvasHeight);
    let pixelPos = p.pos + quadOffset * size;
    let ndcPos = (pixelPos / canvasSize) * 2.0 - 1.0;
    
    var out: VertexOutput;
    // Flip Y for WebGPU coordinates
    out.pos = vec4f(ndcPos.x, -ndcPos.y, 0.0, 1.0);
    out.color = colors[p.colorIndex];
    out.uv = uv;
    out.size = p.size;
    return out;
  }

  @fragment
  fn fs(in: VertexOutput) -> @location(0) vec4f {
    let dist = dot(in.uv, in.uv);
    if (dist > 1.0) { discard; }

    // Glow effect
    let intensity = smoothstep(1.0, 0.0, dist);
    let finalColor = in.color * intensity;
    return vec4f(finalColor, intensity * 0.9);
  }
`;

// Shader creation helper with log checking
async function createShaderModuleWithLogging(device, code, label) {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  let hasErrors = false;
  for (const msg of info.messages) {
    if (msg.type === "error") {
      console.error(`[Shader Error] ${label}: line ${msg.lineNum}:${msg.linePos} - ${msg.message}`);
      hasErrors = true;
    } else {
      console.warn(`[Shader Warning] ${label}: line ${msg.lineNum}:${msg.linePos} - ${msg.message}`);
    }
  }
  if (hasErrors) {
    throw new Error(`Shader compilation failed for ${label}`);
  }
  return module;
}

const simModule = await createShaderModuleWithLogging(device, simulationWGSL, "simulation");
const extractModule = await createShaderModuleWithLogging(device, extractKeysWGSL, "extractKeys");
const applySortedModule = await createShaderModuleWithLogging(device, applySortedWGSL, "applySorted");
const resetTargetsModule = await createShaderModuleWithLogging(device, resetTargetsWGSL, "resetTargets");
const applyScanModule = await createShaderModuleWithLogging(device, applyScanWGSL, "applyScan");
const applyReduceModule = await createShaderModuleWithLogging(device, applyReduceWGSL, "applyReduce");
const renderModule = await createShaderModuleWithLogging(device, renderWGSL, "render");

computePipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: simModule, entryPoint: "main" },
});

extractKeysPipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: extractModule, entryPoint: "main" },
});

applySortedPipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: applySortedModule, entryPoint: "main" },
});

initTargetsPipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: resetTargetsModule, entryPoint: "main" },
});

applyScanPipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: applyScanModule, entryPoint: "main" },
});

applyReducePipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module: applyReduceModule, entryPoint: "main" },
});

renderPipeline = device.createRenderPipeline({
  layout: "auto",
  vertex: { module: renderModule, entryPoint: "vs" },
  fragment: {
    module: renderModule,
    entryPoint: "fs",
    targets: [{
      format: presentationFormat,
      blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
      }
    }],
  },
  primitive: { topology: "triangle-strip" },
});

// Bind Groups State
let computeBindGroup = null;
let extractBindGroup = null;
let applySortedBindGroup = null;
let initTargetsBindGroup = null;
let renderBindGroup = null;

function buildBindGroups() {
  computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer } },
      { binding: 1, resource: { buffer: simulationUniformBuffer } },
    ],
  });

  extractBindGroup = device.createBindGroup({
    layout: extractKeysPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer } },
      { binding: 1, resource: { buffer: sortKeysBuffer } },
      { binding: 2, resource: { buffer: sortPayloadBuffer } },
    ],
  });

  applySortedBindGroup = device.createBindGroup({
    layout: applySortedPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer } },
      { binding: 1, resource: { buffer: sortTempPayloadBuffer } },
      { binding: 2, resource: { buffer: simulationUniformBuffer } },
    ],
  });

  initTargetsBindGroup = device.createBindGroup({
    layout: initTargetsPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: particlesBuffer } }],
  });

  // Render Uniform canvas size
  renderUniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(renderUniformBuffer, 0, new Float32Array([canvas.width, canvas.height, 0, 0]));

  renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer } },
      { binding: 1, resource: { buffer: colorBuffer } },
      { binding: 2, resource: { buffer: renderUniformBuffer } },
    ],
  });

  applyScanBindGroup = device.createBindGroup({
    layout: applyScanPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer } },
      { binding: 1, resource: { buffer: scanOutputBuffer } },
      { binding: 2, resource: { buffer: simulationUniformBuffer } },
    ],
  });

  applyReduceBindGroup = device.createBindGroup({
    layout: applyReducePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer } },
      { binding: 1, resource: { buffer: reduceOutputBuffer } },
      { binding: 2, resource: { buffer: simulationUniformBuffer } },
    ],
  });
}

function updateUniforms() {
  const uniformData = new ArrayBuffer(32);
  const f32 = new Float32Array(uniformData);
  const u32 = new Uint32Array(uniformData);
  
  f32[0] = mouseX;
  f32[1] = mouseY;
  f32[2] = canvas.width;
  f32[3] = canvas.height;
  u32[4] = mouseDown ? 1 : 0;
  u32[5] = attractMode ? 1 : 0;
  u32[6] = isOperating ? 1 : 0;
  f32[7] = 100.0; // padding
  
  document.title = `GridWise Optimal | Canvas: ${canvas.width}x${canvas.height} | Mouse: ${Math.round(mouseX)},${Math.round(mouseY)}`;
  
  device.queue.writeBuffer(simulationUniformBuffer, 0, uniformData);
}

// Perform Operations entirely on GPU
async function performSort() {
  if (currentOperation) {
    clearTimeout(currentOperation);
  }
  isOperating = true;

  try {
    const commandEncoder = device.createCommandEncoder();
    
    // 1. Extract keys (particle sizes) and payloads (indices)
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(extractKeysPipeline);
    pass.setBindGroup(0, extractBindGroup);
    pass.dispatchWorkgroups(Math.ceil(particleCount / 256));
    pass.end();
    
    device.queue.submit([commandEncoder.finish()]);

    // 2. Sort on the GPU
    await sorter.execute({
      keysInOut: sortKeysBuffer,
      keysTemp: sortTempKeysBuffer,
      payloadInOut: sortPayloadBuffer,
      payloadTemp: sortTempPayloadBuffer,
    });

    // 3. Apply sorted positions to particles
    updateUniforms();
    const commandEncoder2 = device.createCommandEncoder();
    const pass2 = commandEncoder2.beginComputePass();
    pass2.setPipeline(applySortedPipeline);
    pass2.setBindGroup(0, applySortedBindGroup);
    pass2.dispatchWorkgroups(Math.ceil(particleCount / 256));
    pass2.end();
    device.queue.submit([commandEncoder2.finish()]);

    currentOperation = setTimeout(() => {
      // Re-trigger targets reset on GPU
      const encoderReset = device.createCommandEncoder();
      const passReset = encoderReset.beginComputePass();
      passReset.setPipeline(initTargetsPipeline);
      passReset.setBindGroup(0, initTargetsBindGroup);
      passReset.dispatchWorkgroups(Math.ceil(particleCount / 256));
      passReset.end();
      device.queue.submit([encoderReset.finish()]);

      currentOperation = setTimeout(() => {
        isOperating = false;
        currentOperation = null;
      }, 8000);
    }, 8000);

  } catch (error) {
    console.error("Sort error:", error);
    showError("GPU sort failed: " + error.message);
    isOperating = false;
    currentOperation = null;
  }
}

async function performScan() {
  if (currentOperation) clearTimeout(currentOperation);
  isOperating = true;
  
  try {
    const commandEncoder = device.createCommandEncoder();
    
    // 1. Extract particle sizes
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(extractKeysPipeline);
    pass.setBindGroup(0, extractBindGroup);
    pass.dispatchWorkgroups(Math.ceil(particleCount / 256));
    pass.end();
    
    device.queue.submit([commandEncoder.finish()]);

    // 2. Scan
    await scanner.execute({
      inputBuffer: sortKeysBuffer,
      outputBuffer: scanOutputBuffer,
    });

    // 3. Apply scan targets
    updateUniforms();
    const commandEncoder2 = device.createCommandEncoder();
    const pass2 = commandEncoder2.beginComputePass();
    pass2.setPipeline(applyScanPipeline);
    pass2.setBindGroup(0, applyScanBindGroup);
    pass2.dispatchWorkgroups(Math.ceil(particleCount / 256));
    pass2.end();
    device.queue.submit([commandEncoder2.finish()]);

    // 4. Setup resets
    currentOperation = setTimeout(() => {
      const encoderReset = device.createCommandEncoder();
      const passReset = encoderReset.beginComputePass();
      passReset.setPipeline(initTargetsPipeline);
      passReset.setBindGroup(0, initTargetsBindGroup);
      passReset.dispatchWorkgroups(Math.ceil(particleCount / 256));
      passReset.end();
      device.queue.submit([encoderReset.finish()]);

      currentOperation = setTimeout(() => {
        isOperating = false;
        currentOperation = null;
      }, 8000);
    }, 8000);
  } catch (error) {
    console.error("Scan error:", error);
    isOperating = false;
  }
}

async function performReduce() {
  if (currentOperation) clearTimeout(currentOperation);
  isOperating = true;
  
  try {
    const commandEncoder = device.createCommandEncoder();
    
    // 1. Extract particle sizes
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(extractKeysPipeline);
    pass.setBindGroup(0, extractBindGroup);
    pass.dispatchWorkgroups(Math.ceil(particleCount / 256));
    pass.end();
    
    device.queue.submit([commandEncoder.finish()]);

    // 2. Reduce
    await reducer.execute({
      inputBuffer: sortKeysBuffer,
      outputBuffer: reduceOutputBuffer,
    });

    // 3. Apply reduce targets
    updateUniforms();
    const commandEncoder2 = device.createCommandEncoder();
    const pass2 = commandEncoder2.beginComputePass();
    pass2.setPipeline(applyReducePipeline);
    pass2.setBindGroup(0, applyReduceBindGroup);
    pass2.dispatchWorkgroups(Math.ceil(particleCount / 256));
    pass2.end();
    device.queue.submit([commandEncoder2.finish()]);

    // 4. Setup resets
    currentOperation = setTimeout(() => {
      const encoderReset = device.createCommandEncoder();
      const passReset = encoderReset.beginComputePass();
      passReset.setPipeline(initTargetsPipeline);
      passReset.setBindGroup(0, initTargetsBindGroup);
      passReset.dispatchWorkgroups(Math.ceil(particleCount / 256));
      passReset.end();
      device.queue.submit([encoderReset.finish()]);

      currentOperation = setTimeout(() => {
        isOperating = false;
        currentOperation = null;
      }, 8000);
    }, 8000);
  } catch (error) {
    console.error("Reduce error:", error);
    isOperating = false;
  }
}

// Frame loop
let lastFpsUpdate = performance.now();
let frames = 0;

function render() {
  updateUniforms();

  const commandEncoder = device.createCommandEncoder();

  // 1. Run simulation pass
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, computeBindGroup);
  computePass.dispatchWorkgroups(Math.ceil(particleCount / 256));
  computePass.end();

  // 2. Render pass directly from buffer
  const textureView = context.getCurrentTexture().createView();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [{
      view: textureView,
      clearValue: { r: 0.04, g: 0.04, b: 0.06, a: 1.0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  renderPass.setPipeline(renderPipeline);
  renderPass.setBindGroup(0, renderBindGroup);
  renderPass.draw(4, particleCount); // 4 vertices per instance (quad)
  renderPass.end();

  device.queue.submit([commandEncoder.finish()]);

  // FPS metric logging
  const now = performance.now();
  frames++;
  if (now - lastFpsUpdate >= 1000) {
    const fps = Math.round((frames * 1000) / (now - lastFpsUpdate));
    console.log(`FPS: ${fps} (Particles: ${particleCount}, isOperating: ${isOperating})`);
    frames = 0;
    lastFpsUpdate = now;
  }

  requestAnimationFrame(render);
}

// Event Listeners
canvas.addEventListener("mousemove", (e) => {
  mouseX = e.clientX;
  mouseY = e.clientY;
});
canvas.addEventListener("mouseleave", () => {
  mouseX = -1000;
  mouseY = -1000;
});
starSlider.addEventListener("input", (e) => {
  particleCount = parseInt(e.target.value);
  starCountDisplay.textContent = formatStarCount(particleCount);
});
starSlider.addEventListener("change", (e) => {
  particleCount = parseInt(e.target.value);
  if (currentOperation) {
    clearTimeout(currentOperation);
    currentOperation = null;
  }
  isOperating = false;
  setTimeout(() => {
    initGPUResources(particleCount);
  }, 100);
});
canvas.addEventListener("mousedown", (e) => {
  mouseDown = true;
  attractMode = !e.shiftKey;
});
canvas.addEventListener("mouseup", () => {
  mouseDown = false;
});

function formatStarCount(count) {
  return count >= 1000 ? (count / 1000).toFixed(0) + "K" : count;
}
function showError(msg) {
  const el = document.getElementById("errorDisplay");
  el.textContent = msg;
  el.style.display = "block";
  setTimeout(() => { el.style.display = "none"; }, 6000);
}

sortBtn.addEventListener("click", performSort);
scanBtn.addEventListener("click", performScan);
reduceBtn.addEventListener("click", performReduce);

// Start
initGPUResources(particleCount);
render();

// Automated benchmark runner for Antigravity agent
async function runAutomatedBenchmark() {
  try {
    console.log("=== STARTING AUTOMATED BENCHMARK ===");
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    
    await sleep(3000);
    console.log("--- 10K Baseline Complete ---");
    console.log("Triggering 10K Sort...");
    await performSort();
    await sleep(9000);
    console.log("--- 10K Sort Complete ---");
    
    console.log("Changing to 50K particles...");
    particleCount = 50000;
    starSlider.value = 50000;
    starCountDisplay.textContent = formatStarCount(50000);
    initGPUResources(50000);
    await sleep(3000);
    console.log("--- 50K Baseline Complete ---");
    console.log("Triggering 50K Sort...");
    await performSort();
    await sleep(9000);
    console.log("--- 50K Sort Complete ---");
    
    console.log("Changing to 100K particles...");
    particleCount = 100000;
    starSlider.value = 100000;
    starCountDisplay.textContent = formatStarCount(100000);
    initGPUResources(100000);
    await sleep(3000);
    console.log("--- 100K Baseline Complete ---");
    console.log("Triggering 100K Sort...");
    await performSort();
    await sleep(9000);
    console.log("--- 100K Sort Complete ---");
    
    console.log("=== AUTOMATED BENCHMARK FINISHED ===");
  } catch (err) {
    console.error("BENCHMARK RUNNER CRASHED:", err);
    console.error(err.stack);
  }
}
setTimeout(runAutomatedBenchmark, 1000);

} catch (error) {
  console.error("CRITICAL RUNTIME ERROR IN OPTIMAL PATH SCRIPT:", error);
  console.error(error.stack);
}
