---
layout: page
title: Performance
permalink: /performance/
sidebar: performance
---

Benchmarks for Gridwise primitives over a range of input sizes (logarithmically spaced),
plotting throughput in GB/s versus input length. Both CPU and GPU timing are reported. See the
[timing strategy documentation]({{ "/timing-strategy/" | relative_url }}) for how
primitives are timed and how to interpret results.

## Scan and Sort Performance

Benchmarks scan, reduce, and sort over a configurable range of input sizes. Choose the primitive,
datatype, and sort direction; results are plotted after each run. Note that sort overwrites its
input, so repeated trials measure a partially-sorted array.

<a href="{{ "/examples/scan_sort_perf.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/scan_sort_perf.mjs" target="_blank" class="doc-btn">Source</a>

## Reduce Performance

Benchmarks the reduce primitive over a configurable range of input sizes. Explains the
warmup-then-trials timing strategy with inline code excerpts showing how to call
`getTimingResult` after execution.

<a href="{{ "/examples/reduce_perf.html" | relative_url }}" class="doc-btn">Open Example</a>
<a href="https://github.com/gridwise-webgpu/gridwise/blob/main/examples/reduce_perf.mjs" target="_blank" class="doc-btn">Source</a>

---
