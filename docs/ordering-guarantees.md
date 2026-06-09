---
layout: page
title: "Ordering and Synchronization Guarantees"
category: gridwise
permalink: /ordering-guarantees/
order: 11
---

When integrating Gridwise parallel primitives into real-time graphics engines (e.g., Three.js, Babylon.js), physics simulations, or complex compute pipelines, understanding WebGPU's execution and synchronization model is critical. 

A common concern is whether `async/await` is required between dispatches, which would introduce CPU-GPU round-trip latency and stall the pipeline. **Gridwise is designed to avoid these stalls completely.**

---

## 1. WebGPU's Queue Execution Model

WebGPU commands are recorded on the CPU (using `GPUCommandEncoder` and `GPUComputePassEncoder`) and submitted to the GPU queue using:

```js
device.queue.submit([commandBuffer1, commandBuffer2, ...]);
```

- **Submission Order**: Submissions to the same queue are guaranteed to begin execution in the order they are submitted.
- **Overlapping Execution**: The GPU may overlap execution of independent commands for optimization, but it must preserve the appearance of sequential execution with respect to resource hazards.

---

## 2. Resource Hazards & Automatic Synchronization

If a compute pass writes to a buffer (e.g., `keysTemp`) and a subsequent compute pass reads from that same buffer, this is a **Read-After-Write (RAW) hazard**.

In WebGPU, you **do not** need to manually insert pipeline barriers or execution barriers (unlike Vulkan or WebGL/OpenGL memory barriers). 
- **Between Compute Passes**: WebGPU automatically inserts the necessary memory and pipeline barriers between separate compute passes to ensure all writes in a previous pass are visible to the reads in the next pass.
- **Within a Single Compute Pass**: WebGPU does *not* automatically synchronize multiple dispatches within the *same* compute pass. For this reason, Gridwise primitives separate dependent phases into distinct compute passes (e.g., `global_hist` and `onesweep_pass` are executed as separate compute passes).

---

## 3. Chaining Primitives on the GPU (No CPU await required)

If you want to run multiple Gridwise primitives sequentially (e.g., Sort a buffer, then run a Scan on it, then run a custom simulation shader):

1. **Do not** await each `execute()` call on the CPU.
2. **Do** pass a shared `GPUCommandEncoder` to each primitive's `execute()` method.

Because `execute()` accepts a pre-existing `encoder`, it will record its compute passes to your command buffer without submitting it to the queue.

### Example: Zero-Stall Pipeline Chaining

```js
const encoder = device.createCommandEncoder({ label: "simulation_step_encoder" });

// 1. Run Sort (records passes into the encoder)
await sortPrimitive.execute({
  keysInOut: keysBuffer,
  keysTemp: tempBuffer,
  encoder: encoder
});

// 2. Run Scan on the sorted keys (records passes into the same encoder)
await scanPrimitive.execute({
  inputBuffer: keysBuffer,
  outputBuffer: scannedBuffer,
  encoder: encoder
});

// 3. Record your own custom simulation / graphics render passes using the same buffers
const passEncoder = encoder.beginComputePass();
passEncoder.setPipeline(mySimulationPipeline);
passEncoder.setBindGroup(0, bindGroupContainingScannedBuffer);
passEncoder.dispatchWorkgroups(workgroups);
passEncoder.end();

// 4. Submit the entire chain to the GPU queue in a single call
const commandBuffer = encoder.finish();
device.queue.submit([commandBuffer]);

// The GPU executes the entire sequence (Sort -> Scan -> Simulation) in order.
// No CPU-GPU round-trips or main thread stalls occur!
```

> [!NOTE]
> Although `execute` is an `async` function, when an external `encoder` is passed, it only resolves when recording is finished. It does *not* wait for GPU execution. The promise resolves immediately so you can proceed to record the next step.

---

## 4. CPU-GPU Synchronization (When is await necessary?)

The only time you must use `await` or CPU-side synchronization is when the CPU needs to read back the results from the GPU.

In Gridwise, this is done using:
- **`mappableBuffer.mapAsync(GPUMapMode.READ)`**: Initiates buffer mapping. Awaiting this promise guarantees that the GPU has finished all preceding commands writing to that buffer, and that the data is now readable in host memory.
- **`device.queue.onSubmittedWorkDone()`**: Returns a promise that resolves when all work submitted to the queue up to that point has fully completed execution on the GPU. This is useful for CPU-side timing measurements.
