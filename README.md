# Gridwise

High-performance WebGPU parallel compute primitives in JavaScript.

Gridwise provides high-throughput GPU compute primitives — **scan** (prefix sum), **reduce**, and **radix sort** — built on WebGPU. It runs directly in modern browsers and in Node.js (via [`webgpu`](https://www.npmjs.com/package/webgpu) / Dawn) with zero native build dependencies.

---

## ⚡ Key Features

* **High Performance**: Employs state-of-the-art Decoupled Lookback / Fallback (DLDF) for scan/reduce and OneSweep for single-pass radix sorting.
* **Hardware Acceleration & Subgroup Emulation**: Leverages WebGPU subgroup instructions where supported, with automatic fallback emulation for devices without subgroup extensions.
* **Flexible Data Types & Operators**: Supports `u32`, `i32`, and `f32` data types with configurable binary operators (`BinOpAdd`, `BinOpMin`, `BinOpMax`, `BinOpMultiply`).
* **Zero Build Step**: Shaders are assembled as runtime WGSL template literals — no offline compilation step required.
* **Full TypeScript Support**: Bundled type definitions (`index.d.ts`).

---

## 📦 Installation

```bash
npm install gridwise
```

---

## 🚀 Quickstart Example

### Prefix Scan (`DLDFScan`)

```javascript
import { DLDFScan, BinOpAdd } from "gridwise";

// 1. Initialize WebGPU
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredFeatures: adapter.features.has("subgroups") ? ["subgroups"] : [],
});

// 2. Create input and output GPU buffers
const count = 1000000;
const inputData = new Uint32Array(count).fill(1);

const inputBuffer = device.createBuffer({
  size: inputData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true,
});
new Uint32Array(inputBuffer.getMappedRange()).set(inputData);
inputBuffer.unmap();

const outputBuffer = device.createBuffer({
  size: inputData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

// 3. Instantiate and run DLDFScan
const scan = new DLDFScan({
  device,
  type: "exclusive",
  datatype: "u32",
  binop: BinOpAdd,
});

scan.registerBuffer({ label: "inputBuffer", buffer: inputBuffer });
scan.registerBuffer({ label: "outputBuffer", buffer: outputBuffer });

await scan.execute({ count });
```

### Radix Sort (`OneSweepSort`)

```javascript
import { OneSweepSort } from "gridwise";

const sort = new OneSweepSort({
  device,
  type: "keysonly",
  datatype: "u32",
  direction: "ascending",
});

sort.registerBuffer({ label: "keysInOut", buffer: keysBuffer });
await sort.execute({ count });
```

---

## 📖 Documentation

* [Documentation & API Reference](https://gridwise-webgpu.github.io/gridwise/docs/)
* [Architecture & Primitive Design](https://gridwise-webgpu.github.io/gridwise/docs/architecture.html)
* [Subgroup Strategy & Emulation](https://gridwise-webgpu.github.io/gridwise/docs/subgroup-strategy.html)

---

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE.txt).
