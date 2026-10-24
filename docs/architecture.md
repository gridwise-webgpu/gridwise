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

### Composing primitives

A primitive's work is a list of actions returned by its `compute()` method: `Kernel`s to dispatch, `AllocateBuffer`s for internal buffers, copies and writes. One of those actions can be another primitive. `ExecutePrimitive` runs a child primitive at that point in the list, on the same encoder, with the child's buffers bound to the parent's by name, so a new primitive can reuse an existing one (for example, a scan) as one of its steps and still be a single primitive to execute, time and test. [Composing Primitives](../composing-primitives/) describes why this exists and how it works.

### Buffer

One of the challenges of writing a primitive library is handling data, which may be stored on the CPU (in a JavaScript typed array) or on the GPU (as a WebGPU GPUBuffer). Gridwise's `Buffer` class attempts to abstract away the details of separately managing CPU and GPU buffer data structures with one unified data structure that stores, and moves data between, both. This data structure has grown organically to handle many use cases and deserves more focus by future developers as a principled data structure in WebGPU programming.
