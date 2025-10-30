---
layout: post
title: "Gridwise WebGPU Timing Strategy"
date: 2025-08-12
---

Building a high-performance WebGPU primitive library requires careful timing measurements to ensure that primitives deliver high performance to their users. In developing Gridwise, we designed both CPU and GPU timing methodologies and describe them here.

## GPU Timing

Gridwise's GPU timing uses WebGPU's GPU timestamp queries. Gregg Tavares's "[WebGPU Timing Performance
](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html)" is both an excellent overview of using timestamp queries as well as the basis for our implementation. We began with his [timing helper](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html#a-timing-helper) and augmented it to support calling multiple kernels (which is useful for benchmarking purposes).

In our implementation, the `Primitive` class has a static `__timingHelper` member that is initialized to understand. The timing helper is enabled by passing `enableGPUTiming: true` as a member of the argument object to `primitive.execute()` (i.e., `primitive.execute{ enableGPUTiming: true, ...}`).

When GPU timing is enabled in this way, `primitive.execute` checks for a defined `__timingHelper` and initializes one if not, counting the number of kernels in the current primitive for the timing helper's initialization. As explained by the [timing helper](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html#a-timing-helper)'s documentation, the timing helper replaces `beginComputePass` with its own `beginComputePass` that initializes the timing helper.

We record a separate timing value for each kernel in the primitive and thus return a list of kernel time durations, one list element per kernel.

`primitive.execute`'s argument object also takes a `trials` argument that defaults to 1. If `trials` is _n_, then every kernel dispatch within the primitive is replaced with _n_ consecutive dispatches. Running more trials is helpful for more reliable timing measurements.

## CPU timing

CPU timing is only possible if the primitive creates its own encoder to ensure that the resulting command buffer only reflects the work within the primitive. It is enabled in a similar way to GPU timing (`primitive.execute{ enableCPUTiming: true, ...}`). In our internal discussions, we have settled on the following way to best measure CPU timing, which is what is implemented within Gridwise, but we are open to better suggestions here. Our strategy is to clear the device's queue, record the current time as `cpuStartTime`, submit the command buffer and wait for the queue to clear, then record the current time as `gpuStartTime`. The elapsed time is the time between the two CPU time stamps.

```js
const commandBuffer = encoder.finish();
if (args?.enableCPUTiming) {
  await this.device.queue.onSubmittedWorkDone();
  this.cpuStartTime = performance.now();
}
this.device.queue.submit([commandBuffer]);
if (args?.enableCPUTiming) {
  await this.device.queue.onSubmittedWorkDone();
  this.cpuEndTime = performance.now();
}
```

## Returning timing information

The primitive object has a `async getTimingResult` member function that returns the CPU and GPU timing result as `const { gpuTotalTimeNS, cpuTotalTimeNS } = await primitive.getTimingResult();`. For GPU timing, this call returns a list of total elapsed times per kernel in ns. For CPU timing, this call returns an elapsed time for the entire primitive in ns. Neither accounts for `trials`, so the caller of `getTimingResult` should divide any returned values by the number of trials.
