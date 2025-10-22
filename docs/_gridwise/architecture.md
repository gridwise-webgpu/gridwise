---
layout: post
title: "Gridwise Architecture"
date: 2025-08-12
excerpt: "Overview of Gridwise's core architecture, including Primitives and Buffers, and how they provide best-in-class WebGPU compute performance."
---

The primary goal of Gridwise is to deliver best-in-class performance on WebGPU compute primitives while minimizing the amount of code that must be written by the library user. Ideally, a Gridwise user will declare and then execute a primitive and Gridwise will handle all low-level details of setting up and calling the necessary WebGPU primitives.

## Gridwise Abstraction

### Primitive

The primary abstraction in Gridwise is a `Primitive`. Primitives are instances of a primitive-specific subclass of a JavaScript `Primitive` class. They have an `execute` member function, and the typical usage is to instantiate a primitive using `new()` and then call `execute()` on that primitive. Both instantiation and execution have numerous options. As an example, let's look at a scan primitive, which is an instance of the `DLDFScan` class ("decoupled-lookback, decoupled-fallback scan"):

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

This particular primitive is parameterized by its datatype (in this case, "u32"), by the binary operation ("binop") performed by the scan (in this case, addition on u32 data), and by the scan operation (exclusive or inclusive).

When the scan is actually executed, its arguments are buffers that store its input and output. This particular primitive has named arguments of an input buffer named `inputBuffer` and an output buffer named `outputBuffer`. These buffers can be WebGPU buffers of type `GPUBuffer` but can also be `Buffer`s, described next.

The primitive performs all necessary WebGPU operations, including (optionally) setting up an encoder, building up and setting WebGPU layouts and pipelines, running the pipeline, and optionally recording GPU-side or CPU-side timing. It also caches WebGPU layouts and pipelines to avoid the expense of recreating them if they have already been created.

### Buffer

One of the challenges of writing a primitive library is handling data, which may be stored on the CPU (in a JavaScript typed array) or on the GPU (as a WebGPU GPUBuffer). Gridwise's `Buffer` class attempts to abstract away the details of separately managing CPU and GPU buffer data structures with one unified data structure that stores, and moves data between, both. This data structure has grown organically to handle many use cases and deserves more focus by future developers as a principled data structure in WebGPU programming.
