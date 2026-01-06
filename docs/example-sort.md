---
layout: page
title: Sort Example
category: examples
permalink: /examples/sort/
---

# Sort Example

This example demonstrates how to use the `sort` primitive for parallel sorting operations using WebGPU.

## Overview

The sort primitive performs efficient parallel sorting on GPU using the OneSweep algorithm. This example shows how to sort large arrays of values.

## Features

- **Parallel Sorting**: GPU-accelerated sorting using OneSweep
- **Multiple Data Types**: Support for various numeric types
- **Large Arrays**: Efficiently handles millions of elements

## Live Demo

<a href="/gridwise/examples/sort_example.html" class="nav-button" style="display: inline-flex;">
  Open Live Example
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
  </svg>
</a>

## Performance Comparison

Compare scan and sort performance:

<a href="/gridwise/examples/scan_sort_perf.html" class="nav-button" style="display: inline-flex;">
  View Performance Comparison
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
  </svg>
</a>

## Code Example

```javascript
const sortPrimitive = new OneSweep({
  device,
  datatype: "u32",
  order: "ascending"
});
```

## Source Code

View the complete source code on [GitHub](https://github.com/gridwise-webgpu/gridwise/blob/main/examples/sort_example.mjs).
