---
layout: page
title: "Gridwise Architecture"
category: gridwise
permalink: /architecture/
excerpt: "Overview of Gridwise's core architecture, including Primitives and Buffers, and how they provide best-in-class WebGPU compute performance."
order: 1
---

The primary goal of Gridwise is to deliver best-in-class performance on WebGPU compute primitives while minimizing the amount of code that must be written by the library user. Ideally, a Gridwise user will declare and then execute a primitive and Gridwise will handle all low-level details of setting up and calling the necessary WebGPU primitives.

## Gridwise Abstraction

### Primitive

The primary abstraction in Gridwise is a `Primitive`. Primitives are instances of a primitive-specific subclass of the JavaScript class `BasePrimitive`. They have an `execute` member function, and the typical usage is to instantiate a primitive using `new()` and then call `execute()` on that primitive. Both instantiation and execution have numerous options. As an example, let's look at a scan primitive, which is an instance of the `DLDFScan` class ("decoupled-lookback, decoupled-fallback scan"):

```js
const datatype = "u32";
const dldfscanPrimitive = new DLDFScan({
  device,
  binop: new BinOpAdd({ datatype }),
  type: "exclusive", // "exclusive" is the default
  datatype,
});

await dldfscanPrimitive.execute({
  inputBuffer,
  outputBuffer,
});
```

This particular primitive is parameterized by its datatype (in this case, "u32"), by the binary operation ("binop") performed by the scan (in this case, addition on u32 data), and by the scan operation (exclusive, inclusive, or reduce).

When the scan is actually executed, its arguments are buffers that store its input and output. This particular primitive has named arguments of an input buffer named `inputBuffer` and an output buffer named `outputBuffer`. These buffers can be WebGPU buffers of type `GPUBuffer` but can also be `Buffer`s, described next.

The primitive performs all necessary WebGPU operations, including (optionally) setting up an encoder, building up and setting WebGPU layouts and pipelines, running the pipeline, and optionally recording GPU-side or CPU-side timing. It also caches WebGPU layouts and pipelines to avoid the expense of recreating them if they have already been created. Internally, a primitive describes its work as a list of actions returned by its `compute()` method (kernels to dispatch, internal buffers to allocate, copies), and `execute()` carries them out in order. One kind of action runs another primitive, so a primitive can use an existing one, such as a scan, as one of its own steps (see [Composing primitives](#composing-primitives) below).

### Declaring a kernel's buffers and parameters

A kernel's buffers are declared once, in its `Kernel` action, in binding order. Gridwise derives the bind group layout, the bind groups and the WGSL declarations from that list, so the kernel's WGSL uses the buffers without declaring them. Parameters that go in a uniform buffer are declared once as a `UniformStruct`, which generates the WGSL struct (padded to 16 bytes) and packs the values. From `Histogram`:

```js
// constructor
this.params = new UniformStruct("HistogramParameters", { n: "u32" });

// compute()
return [
  this.params.allocate("histogramParameters", { n: this.inputLength }),
  new Kernel({
    kernel: this.histogramWGSL, // uses inputBuffer, outputBuffer, params.n, levels
    entryPoint: "histogram",
    buffers: [
      { name: "inputBuffer", access: "read", type: `array<${T}>` },
      { name: "outputBuffer", access: "read_write", type: "array<atomic<u32>>" },
      { name: "params", buffer: "histogramParameters", uniform: this.params },
      ...(ranged ? [{ name: "levels", access: "read", type: `array<${T}>` }] : []),
    ],
  }),
];
```

Each entry has `name` (the WGSL variable), `buffer` (the primitive's buffer bound to it, default `name`), `access` (`"read"`, `"read_write"` or `"uniform"`) and `type`, or `uniform` (a `UniformStruct`) in place of `access` and `type`. A list of lists gives one list per bind group. `UniformStruct.allocate()` returns an `AllocateBuffer` whose upload is skipped when the values have not changed. When several kernels each need their own values (OneSweep's passes), `allocateSlots(label, valuesList, alignment)` packs them into one buffer at offsets aligned to `device.limits.minUniformBufferOffsetAlignment`, and `slot(label, i, alignment)` is the `buffer` entry that binds set `i`. The older form, with `bufferTypes` and `bindings` given as parallel lists and the declarations written in the WGSL, still works. A function that returns a whole WGSL module for others to bind (such as `localFinishWGSL`) can declare its buffers with `declareBuffers(list)`, and its own `Kernel` then uses `...bufferLayout(list)` in place of `buffers`, so the list is still written once.

### WGSL helpers

Gridwise's WGSL helpers (`wgScan`, `wgReduce`, `wgHistogram`, subgroup operations and their emulation, and others in `wgslFunctions.mjs`) are listed once at the top of a kernel, and `include()` adds everything they need (enabling subgroups, emulated subgroups when the device has none, the binop, the builtins structs), each once, in a valid order:

```js
${this.fnDeclarations.include(
  "wgScan",
  ["wgHistogram", { name: "cnt", numBins: RADIX, workgroupSize: W }],
  "computeLinearizedGridParametersSplit",
)}
```

`this.fnDeclarations.workgroupBytes(...)`, with the same list, is the workgroup memory those helpers declare, for sizing a kernel's own workgroup arrays against `device.limits.maxComputeWorkgroupStorageSize`.

### Indirect dispatch

A `Kernel` normally dispatches the workgroup count its `getDispatchGeometry()` returns, fixed when `execute()` records it. With `indirect: "args"` (or `{ buffer, offset }`), it instead reads three u32s (x, y, z) from that buffer when the GPU runs it, so an earlier kernel can size it from data on the GPU, with no trip to the CPU. The buffer needs `GPUBufferUsage.INDIRECT`. Write the counts with the `indirectDispatch` WGSL helper, which splits a count across x and y to stay within `maxComputeWorkgroupsPerDimension`; the dispatched kernel linearizes its workgroup id (`computeLinearizedGridParametersSplit`) and skips ids at or past the count:

```js
new AllocateBuffer({ label: "args", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT, clearBufferOnReuse: false }),
new Kernel({ kernel: sizeIt /* include("indirectDispatch"); setDispatchSize(count) */, buffers: [...], getDispatchGeometry: () => [1] }),
new Kernel({ kernel: work, buffers: [...], indirect: "args" }),
```

`ShunPerm` sizes each of its rounds this way. A primitive whose kernels cannot be repeated in place (one that advances state or appends to a list, as `ShunPerm`'s rounds do) sets `this.singleTrialOnly = true`, and `execute()` then refuses `trials` > 1.

### Actions, and when they take effect

`compute()` returns a list of actions: `Kernel`, `AllocateBuffer`, `WriteGPUBuffer`, `ClearBuffer`, `CopyBufferToBuffer` and `ExecutePrimitive`. `execute()` records the GPU work into one command encoder and submits it at the end, so the encoder commands (`Kernel`, `ClearBuffer`, `CopyBufferToBuffer`, and the actions of an `ExecutePrimitive`) run on the GPU in list order.

Uploads of CPU data are different. `AllocateBuffer`'s `populateWith` and `WriteGPUBuffer` are queue writes (`queue.writeBuffer`), performed while `execute()` is still recording, so they land before any of that `execute()`'s GPU work, wherever they appear in the list:

```js
[ new Kernel(A), new WriteGPUBuffer({ label: "x", cpuSource }), new Kernel(B) ]
// runs as: write x, then A, then B
```

To change a buffer between two kernels, use an encoder command or a kernel. `populateWith` fills a buffer the primitive allocates; `WriteGPUBuffer` writes into any buffer the primitive knows (including one the caller passed in), optionally at a byte offset. Both skip uploading unchanged data into a uniform buffer.

### Composing primitives

A primitive's work is a list of actions returned by its `compute()` method: `Kernel`s to dispatch, `AllocateBuffer`s for internal buffers, copies and writes. One of those actions can be another primitive. `ExecutePrimitive` runs a child primitive at that point in the list, on the same encoder, with the child's buffers bound to the parent's by name, so a new primitive can reuse an existing one (for example, a scan) as one of its steps and still be a single primitive to execute, time and test. [Composing Primitives](../composing-primitives/) describes why this exists and how it works.

### Buffer

One of the challenges of writing a primitive library is handling data, which may be stored on the CPU (in a JavaScript typed array) or on the GPU (as a WebGPU GPUBuffer). Gridwise's `Buffer` class attempts to abstract away the details of separately managing CPU and GPU buffer data structures with one unified data structure that stores, and moves data between, both. This data structure has grown organically to handle many use cases and deserves more focus by future developers as a principled data structure in WebGPU programming.
