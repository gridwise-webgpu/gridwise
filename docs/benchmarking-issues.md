---
layout: page
title: "Benchmarking Issues & Idempotence"
category: gridwise
permalink: /benchmarking-issues/
order: 12
---

# Benchmarking Issues and Idempotence in WebGPU Compute Primitives

When evaluating the performance of highly optimized WebGPU compute primitives (like `OneSweepSort` or `DLDFScan`), running multiple consecutive trials is standard practice to amortize dispatch overhead and warm up pipeline caches. However, WebGPU's execution model and the specific mechanics of decoupled lookback introduce unique challenges for benchmark idempotence.

---

## 1. The GPU-Side Benchmarking Loop & Atomics

To avoid CPU-GPU round-trip latency, Gridwise records a loop of $N$ dispatches directly inside a single command buffer when `trials > 1` (configured via `args.trials`). 

### The Problem
Any buffer allocation and initialization commands (such as `AllocateBuffer` with `clearBufferOnReuse: true`) are recorded and executed **on the CPU before the command buffer is submitted**.
* Between consecutive dispatches on the GPU, **no CPU-driven clearing commands run**.
* If a buffer accumulates state across dispatches (e.g. via `atomicAdd`), trial 1 will accumulate into the buffer, and trial 2 will start with the populated buffer, yielding incorrect results or timing.

---

## 2. Decoupled Lookback Spine Corruption

Both `OneSweepSort` (`passHist` buffer) and `DLDFScan` (`spine` buffer) use lookback spines containing workgroup status flags (`FLAG_READY`, `FLAG_INCLUSIVE`) and intermediate values to synchronize scanning progress across workgroups.

Even though each workgroup in a new trial will eventually overwrite its slot in the spine (using `atomicStore`), the lookback synchronization loop relies on **spin-waiting for preceding slots to change from `0` (`FLAG_NOT_READY`) to a ready state.**

If the lookback spine is not reset between trials:
1. **Trial 0 completes**: All slots in the spine are left at `FLAG_INCLUSIVE`.
2. **Trial 1 starts**:
   * Workgroup 1 computes its local sum and looks back at `spine[0]` to see if Workgroup 0 is finished.
   * Instead of finding `0` (not ready), it immediately reads `FLAG_INCLUSIVE` left over from Trial 0.
   * Workgroup 1 assumes Workgroup 0 has already finished, reads the stale prefix sum from Trial 0, and completes.
3. **The Consequences**:
   * **Timing skew**: The workgroups skip the lookback spin-waiting loop entirely, leading to artificially fast and inaccurate benchmarking times.
   * **Correctness corruption**: If the input dataset changes, the primitive will return corrupted results because it reads stale prefix values from the previous execution.

---

## 3. Buffer Reset Strategies & Overhead

To resolve the lookback spine corruption without skewing performance timings, we must carefully evaluate the cost of buffer resetting:

### Self-Cleaning Buffers
Some atomic buffers, such as `sortBump` and `scanBump`, do not need external resets because the compute shader itself resets the atomic counter back to `0` using `atomicStore` on the final workgroup dispatch.

### Dynamic Reset of Lookback Spines
Because the lookback spines (`passHist` and `spine`) are small (e.g., 15.6 KB for a 1,000,000 element scan), they can be cleared on the GPU between dispatches using a lightweight GPU shader (`clear_buffer`) with zero measurable overhead.
* **Sort Solution**: Include `"passHist"` in `resetBuffersForBenchmarkingOnly` inside [onesweep.mjs](file:///Users/jdowens/Documents/working/gridwise/onesweep.mjs).
* **Scan Solution**: Set `clearBufferOnReuse: true` and add `resetBuffersForBenchmarkingOnly: ["spine"]` inside [scandldf.mjs](file:///Users/jdowens/Documents/working/gridwise/scandldf.mjs).

### Restoring Input Data
For in-place algorithms like sorting, the input buffer is modified by the sort itself. Restoring the input data via full memory copies (GPU-to-GPU) would double the memory traffic and artificially deflate performance for bandwidth-bound primitives.
* **Radix Sort Data Independence**: Radix sort execution time is data-independent (always performing the same number of bit-level scatters). Sorting already-sorted data has negligible impact on performance.
* **Dynamic Offsets**: If different random inputs are required on each trial without memory copy overhead, we can allocate a single large input buffer containing multiple random datasets and update the dynamic binding offset (`setBindGroup`) between trials.
