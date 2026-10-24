---
layout: page
title: "Gridwise WebGPU Timing Strategy"
category: gridwise
permalink: /timing-strategy/
excerpt: "CPU and GPU timing methodologies for measuring and optimizing WebGPU primitive performance in Gridwise."
order: 10
---

Building a high-performance WebGPU primitive library requires careful timing measurements to ensure that primitives deliver high performance to their users. In developing Gridwise, we designed both CPU and GPU timing methodologies and describe them here.

## GPU Timing

Gridwise's GPU timing uses WebGPU's GPU timestamp queries. Gregg Tavares's "[WebGPU Timing Performance
](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html)" is both an excellent overview of using timestamp queries as well as the basis for our implementation. We began with his [timing helper](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html#a-timing-helper) and augmented it to support calling multiple kernels (which is useful for benchmarking purposes).

In our implementation, the `BasePrimitive` class has a static `__timingHelper` member that is initialized to `undefined`. The timing helper is enabled by passing `enableGPUTiming: true` as a member of the argument object to `primitive.execute()` (i.e., `primitive.execute({ enableGPUTiming: true, ... })`).

When GPU timing is enabled in this way, `primitive.execute` checks for a defined `__timingHelper` and initializes one if not, counting the number of kernels in the current primitive for the timing helper's initialization. As explained by the [timing helper](https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html#a-timing-helper)'s documentation, the timing helper replaces `beginComputePass` with its own `beginComputePass` that initializes the timing helper.

We record a separate timing value for each kernel in the primitive and thus return a list of kernel time durations, one list element per kernel. If the primitive runs other primitives with `ExecutePrimitive` actions, their kernels are counted too (`countKernels()` recurses into them), they record into the same timing helper, and the list includes their kernels in execution order (see [Composing Primitives](../composing-primitives/)).

`primitive.execute`'s argument object also takes a `trials` argument that defaults to 1. If `trials` is _n_, then every kernel dispatch within the primitive is replaced with _n_ consecutive dispatches. Running more trials is helpful for more reliable timing measurements.

### Timestamp resolution

Dawn, Chrome's WebGPU implementation, rounds GPU timestamps down to multiples of 65.536 µs (2^16 ns) by default, to limit timing side channels. This happens on every GPU. Each kernel's time is the difference of two rounded timestamps, so a kernel faster than about 0.1 ms reads as 0, 65.536 µs or 131.072 µs. To time short kernels:

- **Node (`webgpu` package):** create the GPU with `create(["disable-dawn-features=timestamp_quantization"])`. `examples/regression_node.mjs` does this.
- **Chrome (and other Chromium browsers):** launch it with `--disable-dawn-features=timestamp_quantization`, the same Dawn toggle.
- **Otherwise:** use `trials` so that each kernel's timed span is much longer than 65.536 µs. One kernel's `trials` dispatches are timed together, so the rounding error is spread over all of them.

## Benchmarking a primitive

`primitive.benchmark()` does the steps a reliable measurement needs:

```js
const r = await primitive.benchmark({ inputBuffer, outputBuffer }); // execute()'s arguments
// r.ns: median time per execute; r.kernels: [{ label, ns }, ...] per kernel (GPU timing)
```

1. **Warm up** (`primitive.warmUp()`): run the primitive back to back for 200 ms (`warmMs`), doubling `trials` until one execute keeps the GPU busy for at least 20 ms. GPUs lower their clocks when they look idle, and executes separated by CPU work look idle. On a V100 the SM clock fell from 1,297 to 180 MHz ("GPU Idle" the only reason) during such a run, and every kernel slowed by up to 10×.
2. **Time** `repeats` (5) executes with `trials` dispatches each (by default, the number the warm-up reached), with GPU timestamps if the device has them (per kernel, labeled by `kernelLabels()`), else CPU timing.
3. **Report** medians per trial.

For measurements that must be comparable between runs, also lock the GPU's clocks where you can, and use full-resolution timestamps (above). In Node, `misc/node_device.mjs` (`createNodeDevice()`) sets up a device that way, and `misc/benchmark_node.mjs` benchmarks the main primitives with `benchmark()`.

## CPU timing

CPU timing is only possible if the primitive creates its own encoder to ensure that the resulting command buffer only reflects the work within the primitive. It is enabled in a similar way to GPU timing (`primitive.execute({ enableCPUTiming: true, ... })`). In our internal discussions, we have settled on the following way to best measure CPU timing, which is what is implemented within Gridwise, but we are open to better suggestions here. Our strategy is to clear the device's queue, record the current time as `cpuStartTime`, submit the command buffer and wait for the queue to clear, then record the current time as `cpuEndTime`. The elapsed time is the time between the two CPU time stamps.

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
