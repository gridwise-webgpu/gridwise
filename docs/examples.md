---
layout: page
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

Runs an exclusive add-scan over 2<sup>24</sup> i32 values and validates it against a CPU
reference. The scan type, datatype and binary operation are set at the top of the source and
are easy to change.

<a href="{{ "/examples/scan_example.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/scan_example.mjs" target="_blank" class="doc-btn">Source</a>

### Sort

Sorts 2<sup>24</sup> random i32 keys (keys only, ascending) on the GPU using the OneSweep radix
sort. The page walks through the essential API calls, including finding which buffer holds the
sorted output, and validates the result against a CPU reference.

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

## Regression Testing

### Gridwise Regression Tests

Runs a correctness suite covering scan and reduce (`DLDFScan` and `HierarchicalScan`), sort,
random permutations, the random-number generators, and primitive composition, across multiple
datatypes, binary operations, input sizes, and sort directions. Each test reports pass or fail
individually. A second suite of large tests (10<sup>6</sup> to 2<sup>24</sup> elements) can be
triggered separately via a button once the main suite completes.

<a href="{{ "/examples/regression.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/regression.mjs" target="_blank" class="doc-btn">Source</a>

---
