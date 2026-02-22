---
layout: home
title: Examples
permalink: /examples/
sidebar: examples
---

Working examples demonstrating Gridwise primitives in the browser. All examples run WebGPU compute
on the GPU and display results in the page. Open any example in a WebGPU-capable browser (Chrome or
Edge 113+, or Safari 18+).

## Standalone Examples

These examples run with fixed parameters and require no configuration. They demonstrate a single
primitive end-to-end: device setup, buffer allocation, GPU execution, and result validation.

### Scan / Reduce

Runs exclusive scan, inclusive scan, and reduce over random input arrays using the add, min, and
max binary operations. Output is plotted against a CPU-computed reference.

<a href="{{ "/examples/scan_example.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/scan_example.mjs" target="_blank" class="doc-btn">Source</a>

### Sort

Sorts an array of random keys on the GPU using the OneSweep radix sort. Demonstrates keysonly and
key-value sorting in both ascending and descending directions across u32, i32, and f32 types.
Results are validated against a CPU reference.

<a href="{{ "/examples/sort_example.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/sort_example.mjs" target="_blank" class="doc-btn">Source</a>

### Reduce

Computes a sum-reduction over 2<sup>24</sup> i32 values, producing a single output. The page
walks through the essential API calls with inline code excerpts and validates the GPU result against
a CPU reference.

<a href="{{ "/examples/reduce_example.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/reduce_example.mjs" target="_blank" class="doc-btn">Source</a>

## Configurable Examples

These examples expose a configuration pane so you can interactively choose the primitive,
datatype, binary operation, and input length before running.

### Scan / Reduce with Configuration Pane

Interactive scan and reduce demonstration. Set the scan type (exclusive, inclusive, or reduce),
datatype, binary operation, and input length in the pane, then click Start to run on the GPU and
validate the output.

<a href="{{ "/examples/scan_pane_example.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/scan_pane_example.mjs" target="_blank" class="doc-btn">Source</a>

## Performance Testing

These examples benchmark Gridwise primitives over a range of input sizes (logarithmically spaced)
and plot throughput in GB/s versus input length. Both CPU and GPU timing are reported. See the
[timing strategy documentation]({{ "/timing-strategy/" | relative_url }}) for how
primitives are timed and how to interpret results.

### Scan and Sort Performance

Benchmarks scan, reduce, and sort over a configurable range of input sizes. Choose the primitive,
datatype, and sort direction; results are plotted after each run. Note that sort overwrites its
input, so repeated trials measure a partially-sorted array.

<a href="{{ "/examples/scan_sort_perf.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/scan_sort_perf.mjs" target="_blank" class="doc-btn">Source</a>

### Reduce Performance

Benchmarks the reduce primitive over a configurable range of input sizes. Explains the
warmup-then-trials timing strategy with inline code excerpts showing how to call
`getTimingResult` after execution.

<a href="{{ "/examples/reduce_perf.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/reduce_perf.mjs" target="_blank" class="doc-btn">Source</a>

## Regression Testing

### Gridwise Regression Tests

Runs a correctness suite covering scan, reduce, and sort across multiple datatypes, binary
operations, input sizes, and sort directions. Each test reports pass or fail individually. A second
suite of large tests (1M--3M elements) can be triggered separately via a button once the main suite
completes.

<a href="{{ "/examples/regression.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/regression.mjs" target="_blank" class="doc-btn">Source</a>

---
