---
layout: page
title: "Ordering and Synchronization Guarantees"
category: gridwise
permalink: /ordering-guarantees/
order: 11
---

When integrating Gridwise parallel primitives into real-time graphics engines (e.g., Three.js, Babylon.js), physics simulations, or complex compute pipelines, understanding WebGPU's execution and synchronization model is critical. 

A common concern is whether the CPU has to wait for the GPU between dispatches, which would introduce CPU-GPU round-trip latency and stall the pipeline. **Gridwise is designed to avoid these stalls completely.**

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
- **Within a Single Compute Pass**: WebGPU also orders consecutive dispatches within the *same* compute pass: each dispatch is its own synchronization scope, so a dispatch sees the storage writes of the dispatches before it. (What WebGPU does *not* synchronize is invocations *within* one dispatch; that is what workgroup barriers and atomics are for.) Gridwise records each kernel in its own compute pass anyway (e.g., `global_hist` and `onesweep_pass`), which lets GPU timing time each kernel separately.

---

## 3. Chaining Primitives on the GPU (No CPU await required)

If you want to run multiple Gridwise primitives sequentially (e.g., Sort a buffer, then run a Scan on it, then run a custom simulation shader):

1. **Do** pass a shared `GPUCommandEncoder` to each primitive's `execute()` method.
2. **Do not** wait for the GPU between primitives: no submit per primitive, and no `device.queue.onSubmittedWorkDone()` or `mapAsync()` in between.
3. **Do** `await` each `execute()` call. With an `encoder`, `execute()` only *records* the primitive's compute passes into your encoder, and its promise resolves as soon as recording is done; it does not submit anything or wait for the GPU. Awaiting it costs no GPU round trip, and it makes sure every pass has been recorded before you call `encoder.finish()`.

Because `execute()` accepts a pre-existing `encoder`, it records its compute passes into your command buffer without submitting it to the queue.

### Example: Zero-Stall Pipeline Chaining

```js
const encoder = device.createCommandEncoder({ label: "simulation_step_encoder" });

// 1. Run Sort (records passes into the encoder)
await sortPrimitive.execute({
  keysInOut: keysBuffer,
  keysTemp: tempBuffer,
  encoder: encoder
});

// 2. Run Scan on the sorted keys (records passes into the same encoder).
//    Sort reports which of its two buffers holds the sorted keys.
await scanPrimitive.execute({
  inputBuffer: sortPrimitive.outputIsInTemp ? tempBuffer : keysBuffer,
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
> `execute` is an `async` function. When an external `encoder` is passed, its promise resolves when recording is finished, not when the GPU has run the work, so awaiting it does not stall on the GPU. Do await it before recording the next step or calling `encoder.finish()`: recording can involve asynchronous steps, and a pass that has not been recorded yet would be missing from the command buffer.

---

## 4. CPU-GPU Synchronization (When is await necessary?)

The only time you must use `await` or CPU-side synchronization is when the CPU needs to read back the results from the GPU.

In Gridwise, this is done using:
- **`mappableBuffer.mapAsync(GPUMapMode.READ)`**: Initiates buffer mapping. Awaiting this promise guarantees that the GPU has finished all preceding commands writing to that buffer, and that the data is now readable in host memory.
- **`device.queue.onSubmittedWorkDone()`**: Returns a promise that resolves when all work submitted to the queue up to that point has fully completed execution on the GPU. This is useful for CPU-side timing measurements.
