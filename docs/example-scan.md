---
layout: page
title: Scan Example
category: examples
permalink: /examples/scan/
---

# Scan Example

This example demonstrates how to use the `scan` primitive for parallel prefix sum operations.

## Overview

The scan primitive computes prefix sums (also known as cumulative sums) on an array. This example shows both exclusive and inclusive scan variants.

## Scan Types

- **Exclusive Scan**: Output[i] = sum of all elements before position i (output[0] is identity)
- **Inclusive Scan**: Output[i] = sum of all elements up to and including position i

## Live Demo

<a href="/gridwise/examples/scan_example.html" class="nav-button" style="display: inline-flex;">
  Open Live Example
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
  </svg>
</a>

## Interactive Demo

Try the interactive pane example where you can adjust parameters:

<a href="/gridwise/examples/scan_pane_example.html" class="nav-button" style="display: inline-flex;">
  Open Interactive Demo
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
  </svg>
</a>

## In-Depth Analysis

Explore a comprehensive guide with detailed explanations, performance graphs, and real-world use cases:

<a href="/gridwise/examples/scan_indepth.html" class="nav-button" style="display: inline-flex; background: #10b981; color: white;"> 
  📊 View In-Depth Analysis
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
  </svg>
</a>

## Code Example

```javascript
const scanPrimitive = new DLDFScan({
  device,
  binop: new BinOpAdd({ datatype: "i32" }),
  type: "exclusive-scan"
});
```

## Source Code

View the complete source code on [GitHub](https://github.com/gridwise-webgpu/gridwise/blob/main/examples/scan_example.mjs).
