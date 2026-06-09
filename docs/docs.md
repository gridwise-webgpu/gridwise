---
layout: page
title: Documentation
permalink: /documentation/
---

# Using Gridwise (Guides & Primitives)

Everything you need to know to use Gridwise in your application: how to configure, run, and time the primitives, manage buffers, and define custom binary operators.

### [Scan and Reduce]({{ "/scan-and-reduce/" | relative_url }})
Comprehensive guide to scan (prefix sum) and reduce operations in Gridwise. Explains the difference between exclusive scan, inclusive scan, and reduce. Covers binary operations (Add, Min, Max), data types (u32, i32, f32), API usage, and performance guidelines.

### [Sort]({{ "/sort/" | relative_url }})
Complete documentation for Gridwise's OneSweepSort implementation. Covers key-only and key-value pair sorting, configuring sort direction (ascending/descending), buffer management, in-place vs. temporary buffer approaches, and performance characteristics.

### [Binary Operations]({{ "/binop/" | relative_url }})
Guide to binary operations (monoids) used in Gridwise's scan and reduce. Documents standard operators (Add, Min, Max, Multiply) and shows how to specialize them by datatype or implement custom binary operators by extending the `BinOp` class.

### [Buffer Management]({{ "/buffer/" | relative_url }})
Best practices for allocating, managing, and optimizing GPU buffers in Gridwise. Covers buffer creation, storage usage flags, memory optimization, copy strategies, and handling edge cases such as non-aligned input lengths.

### [Timing Strategy]({{ "/timing-strategy/" | relative_url }})
Detailed explanation of timing mechanisms in Gridwise for accurate performance profiling. Covers CPU timing (`performance.now`) and GPU timestamp query approaches, their accuracy tradeoffs, warmup strategies, and trial averaging.

### [Ordering and Synchronization Guarantees]({{ "/ordering-guarantees/" | relative_url }})
Detailed explanation of WebGPU's execution and queue model, resource hazard synchronization, and how to chain multiple Gridwise primitives sequentially on the GPU with zero CPU stalls.

---

# How Gridwise is Built (Internals & Design)

Deep-dives into the architecture, implementation choices, WGSL code-generation strategies, and hardware-level optimizations of the Gridwise library.

### [Architecture]({{ "/architecture/" | relative_url }})
Overview of Gridwise's system design, module structure, and how primitives are organized as modular, reusable classes for clean separation of concerns and extensibility.

### [Primitive Design]({{ "/primitive-design/" | relative_url }})
Deep dive into the design principles behind Gridwise primitives, focusing on single-pass chained algorithms for scan, sort, and reduce. Explores decoupled lookback/fallback optimization techniques and memory bandwidth bounds.

### [Subgroup Strategy]({{ "/subgroup-strategy/" | relative_url }})
Detailed guide to GPU subgroups (and subgroup size variation across vendors) and Gridwise's strategy for subgroup detection, optional subgroup acceleration, and software subgroup emulation.

### [Built-ins Strategy]({{ "/builtins-strategy/" | relative_url }})
Exploration of WebGPU WGSL built-in functions and how Gridwise strategically selects and optimizes their use for reduction operations and aggregation patterns on different hardware.

### [WebGPU Object Caching Strategy]({{ "/webgpu-object-caching-strategy/" | relative_url }})
Detailed documentation of how Gridwise caches and reuses WebGPU objects (compute pipelines, bind groups, shader modules) to eliminate startup overhead and improve throughput.

### [Writing a WebGPU WGSL Workgroup Reduce Function]({{ "/writing-a-webgpu-wgsl-workgroup-reduce-function/" | relative_url }})
In-depth educational tutorial on implementing custom workgroup-level reduce functions in WGSL, covering reduction patterns, workgroup barriers, synchronization, and subgroup utilization.
