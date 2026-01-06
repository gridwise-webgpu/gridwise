---
layout: page
title: Reduce Example
category: examples
permalink: /examples/reduce/
---

# Reduce Example

This example demonstrates how to use the `reduce` primitive to compute a sum-reduction on an input array.

## Overview

The reduce primitive performs a parallel reduction operation on an array of values, producing a single output value. This example computes a sum-reduction on 2<sup>24</sup> i32 values.

## Key Concepts

- **Binary Operation**: Sum-reduction using `BinOpAdd`
- **Data Type**: `i32`
- **Input Size**: 16,777,216 elements

## Live Demo

<a href="/gridwise/examples/reduce_example.html" class="nav-button" style="display: inline-flex;">
  Open Live Example
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
  </svg>
</a>

## Code Example

```javascript
const datatype = "i32";
const binop = new BinOpAdd({ datatype });
const reducePrimitive = new DLDFScan({
  device,
  binop,
  type: "reduce"
});
```

## Performance

<a href="/gridwise/examples/reduce_perf.html" class="nav-button" style="display: inline-flex;">
  View Performance Test
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
  </svg>
</a>

## Source Code

View the complete source code on [GitHub](https://github.com/gridwise-webgpu/gridwise/blob/main/examples/reduce_example.mjs).
