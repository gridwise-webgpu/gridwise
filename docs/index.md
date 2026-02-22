---
layout: splash
title: Gridwise
permalink: /
---

# Gridwise

Gridwise is a JavaScript library of high-performance GPU compute primitives built on WebGPU. It
provides three core operations — **scan** (prefix sum), **reduce**, and **sort** — that run
directly in the browser or in Node.js via node-webgpu, with no native dependencies. The API is
intentionally minimal: instantiate a primitive configured with a WebGPU device and parameters,
then call `execute` with GPU buffers. Gridwise supports `u32`, `i32`, and `f32` data types;
configurable binary operations (add, min, max, multiply); and ascending and descending radix sort
with optional key-value payload support.

Gridwise targets primitives with the highest performance. Achieving this goal is complex because
WebGPU has modest forward-progress guarantees. Gridwise's sort and scan primitives leverage a
decoupled lookback and fallback (DLDF) strategy to achieve top performance. Interested readers
can find ample detail on this approach in [_Decoupled Fallback: A Portable Single-Pass
GPU Scan_](https://doi.org/10.1145/3694906.3743326).

Gridwise is written in WebGPU and WGSL, the WebGPU shader language. All shader code is assembled
as WGSL template literals at runtime — there is no offline compilation step, making the library
straightforward to audit and extend.

<div class="splash-actions">
  <a class="button" href="{{ "/documentation/" | relative_url }}">Documentation</a>
  <a class="button" href="{{ "/examples/" | relative_url }}">Examples</a>
</div>
