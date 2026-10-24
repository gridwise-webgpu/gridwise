---
layout: page
title: Documentation
permalink: /documentation/
---

# Using Gridwise (Guides & Primitives)

Everything you need to know to use Gridwise in your application: how to configure, run, and time the primitives, manage buffers, and define custom binary operators.

### [Scan and Reduce]({{ "/scan-and-reduce/" | relative_url }})
Comprehensive guide to scan (prefix sum) and reduce operations in Gridwise: the chained `DLDFScan` and the reduce-then-scan `HierarchicalScan`. Explains the difference between exclusive scan, inclusive scan, and reduce. Covers binary operations (Add, Min, Max), data types (u32, i32, f32), API usage, and performance guidelines.

### [Sort]({{ "/sort/" | relative_url }})
Complete documentation for Gridwise's OneSweepSort implementation. Covers key-only and key-value pair sorting, configuring sort direction (ascending/descending), which buffer holds the sorted output, sorting fewer key bits, and performance characteristics.

### [Histogram]({{ "/histogram/" | relative_url }})
Counts of samples per bin, modeled on CUB's DeviceHistogram: equal-width or level-bounded bins (bin functions `EvenBins`, `RangeBins`, `RadixBins`), the workgroup-histogram helper shared with permutations, and options.

### [Binary Operations]({{ "/binop/" | relative_url }})
Guide to binary operations (monoids) used in Gridwise's scan and reduce. Documents standard operators (Add, Min, Max, Multiply) and shows how to specialize them by datatype or implement custom binary operators by extending the `BinOp` class.

### [Random Permutations]({{ "/random-permutations/" | relative_url }})
`MultisplitPerm`, an unbiased GPU random permutation that can be bit-reproducible across GPUs: how it works, its options, and tuning.

### [Random Numbers]({{ "/random-numbers/" | relative_url }})
Random bits for WGSL kernels: the fast keyed hash and Philox4x32, their bit-exact JavaScript mirrors, how to declare and call them, and when to use each.

### [Buffer Management]({{ "/buffer/" | relative_url }})
Gridwise's `Buffer` class, which pairs a CPU typed array with a `GPUBuffer`: the use cases it serves, its constructor options (including test-data initialization), and its methods for copying data between CPU and GPU.

### [Timing Strategy]({{ "/timing-strategy/" | relative_url }})
How Gridwise times primitives: GPU timestamp queries (one time per kernel, nested primitives included), CPU timing with `performance.now`, the `trials` argument, and `getTimingResult`.

### [Ordering and Synchronization Guarantees]({{ "/ordering-guarantees/" | relative_url }})
Detailed explanation of WebGPU's execution and queue model, resource hazard synchronization, and how to chain multiple Gridwise primitives sequentially on the GPU with zero CPU stalls.

---

# How Gridwise is Built (Internals & Design)

Deep-dives into the architecture, implementation choices, WGSL code-generation strategies, and hardware-level optimizations of the Gridwise library.

### [Architecture]({{ "/architecture/" | relative_url }})
Overview of Gridwise's system design, module structure, and how primitives are organized as modular, reusable classes for clean separation of concerns and extensibility.

### [Composing Primitives]({{ "/composing-primitives/" | relative_url }})
How a primitive can run another primitive as one of its steps with the `ExecutePrimitive` action: the motivation (reusing a scan inside a new primitive), usage, and how ordering, buffer binding and GPU timing work for nested primitives.

### [Primitive Design]({{ "/primitive-design/" | relative_url }})
Deep dive into the design principles behind Gridwise primitives, focusing on single-pass chained algorithms for scan, sort, and reduce. Explores decoupled lookback/fallback optimization techniques and memory bandwidth bounds.

### [Subgroup Strategy]({{ "/subgroup-strategy/" | relative_url }})
Detailed guide to GPU subgroups (and subgroup size variation across vendors) and Gridwise's strategy for subgroup detection, optional subgroup acceleration, and software subgroup emulation.

### [Built-ins Strategy]({{ "/builtins-strategy/" | relative_url }})
Exploration of WebGPU WGSL built-in functions and how Gridwise strategically selects and optimizes their use for reduction operations and aggregation patterns on different hardware.

### [Benchmarking Issues & Idempotence]({{ "/benchmarking-issues/" | relative_url }})
Why repeated trials of a primitive need care: state left in atomic counters and lookback spines between dispatches, how Gridwise resets them, and restoring sort's input.

### [WebGPU Object Caching Strategy]({{ "/webgpu-object-caching-strategy/" | relative_url }})
How Gridwise caches and reuses WebGPU objects (shader modules, bind group layouts, pipeline layouts, compute pipelines) to avoid recreating them, and why bind groups are not cached.

### [Package Development & Publishing Guide]({{ "/package-development/" | relative_url }})
Developer guide covering npm package structure, development/verification scripts, CI workflows, and workflows for modifying or publishing Gridwise.

### [npm Publishing Instructions]({{ "/npm-instructions/" | relative_url }})
Step-by-step guide for first-time and subsequent npm package releases, account login, dry runs, and semantic versioning.

### [Writing a WebGPU WGSL Workgroup Reduce Function]({{ "/writing-a-webgpu-wgsl-workgroup-reduce-function/" | relative_url }})
A walkthrough of writing a library workgroup-reduce function in WGSL and the language's pain points along the way: explicit built-ins, module-scoped workgroup memory, no overloading or generics, and adapter-dependent subgroup sizes.
