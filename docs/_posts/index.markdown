---
layout: home
permalink: /
---

<style>
    .doc-btn {
        display: inline-block;
        padding: 8px 14px;
        background: #39bda7;
        color: white;
        text-decoration: none;
        border-radius: 4px;
        margin-right: 8px;
        margin-bottom: 10px;
        font-size: 14px;
        transition: background 0.2s;
    }

    .doc-btn:hover {
        background: #2a9a87;
        text-decoration: none;
        color: white;
    }

    .code-block {
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 12px;
        overflow-x: auto;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        margin: 10px 0;
    }
</style>

Comprehensive guides and technical documentation for Gridwise WebGPU primitives.

## Architecture

Overview of Gridwise's system design, module structure, and how primitives are organized for extensibility and performance. Learn about the high-level organization of Gridwise components, including how different primitives (scan, reduce, sort) are implemented as modular, reusable units. Understand the architectural decisions that enable performance optimization while maintaining clean separation of concerns and ease of extension.

[Read](https://gridwise-webgpu.github.io/gridwise/architecture/){:target="_blank" class="doc-btn"}

## Primitive Design

Deep dive into the design principles behind Gridwise primitives with focus on single-pass chained algorithms for sort, scan, and reduce. Explores the tradeoffs between using subgroup instructions for maximum performance versus software emulation for broader compatibility. Covers memory bandwidth considerations, the lookback and fallback optimization techniques, and how to choose between chained algorithms and hybrid approaches for different use cases.

[Read](https://gridwise-webgpu.github.io/gridwise/primitive-design/){:target="_blank" class="doc-btn"}

## Scan and Reduce

Comprehensive guide to scan (prefix sum) and reduce operations in Gridwise. Explains the difference between exclusive scan (first element is identity), inclusive scan (each element includes itself), and reduce (single output value). Covers binary operations (Add, Min, Max), data type support (u32, i32, f32), API usage patterns with code examples, and when to use each variant for optimal performance.

[Read](https://gridwise-webgpu.github.io/gridwise/scan-and-reduce/){:target="_blank" class="doc-btn"}

## Sort

Complete documentation for Gridwise's OneSweepSort implementation. Covers both key-only sorting and key-value pair sorting with full payload support. Explains configurable sort direction (ascending/descending), supported data types, buffer management strategies, and in-place versus temporary buffer approaches. Includes detailed API documentation and performance characteristics across different input sizes and configurations.

[Read](https://gridwise-webgpu.github.io/gridwise/sort/){:target="_blank" class="doc-btn"}

## Binary Operations

Guide to binary operations used in Gridwise's scan and reduce primitives. Documents available operations (Add, Min, Max, Multiply) and their properties. Explains how to implement custom binary operations by extending the binop interface, including implementation requirements, data type constraints, and validation patterns. Critical for users who need domain-specific aggregation operations.

[Read](https://gridwise-webgpu.github.io/gridwise/binop/){:target="_blank" class="doc-btn"}

## Buffer Management

Best practices for allocating, managing, and optimizing GPU buffers in Gridwise applications. Covers buffer creation strategies, memory usage patterns, and how to minimize memory allocation overhead. Explains the relationship between buffer sizes and performance, copy strategies for input/output, and how to handle edge cases with non-aligned input lengths. Essential for building efficient Gridwise applications.

[Read](https://gridwise-webgpu.github.io/gridwise/buffer/){:target="_blank" class="doc-btn"}

## Timing Strategy

Detailed explanation of timing mechanisms in Gridwise for accurate performance measurement and benchmarking. Covers both CPU timing (performance.now) and GPU timing (timestamp queries) approaches, their accuracy tradeoffs, and when to use each. Explains warmup strategies, trial averaging, and how to interpret results across different hardware configurations for reliable performance comparisons.

[Read](https://gridwise-webgpu.github.io/gridwise/timing-strategy/){:target="_blank" class="doc-btn"}

## Subgroup Strategy

Detailed guide to GPU subgroups and their critical role in Gridwise primitive performance. Explains what subgroups are, how different GPU architectures have different subgroup sizes, and the performance benefits of subgroup operations. Covers Gridwise's approach to subgroup detection, optional subgroup acceleration, and fallback strategies for hardware without subgroup support to maintain broad device compatibility.

[Read](https://gridwise-webgpu.github.io/gridwise/subgroup-strategy/){:target="_blank" class="doc-btn"}

## Built-ins Strategy

Exploration of WebGPU WGSL built-in functions and how Gridwise strategically selects and optimizes their use in primitive implementations. Explains which built-ins provide the best performance for reduction operations, aggregation patterns, and data movement. Covers vendor-specific optimizations and how to identify when built-in usage versus hand-tuned WGSL code provides the best performance on different hardware.

[Read](https://gridwise-webgpu.github.io/gridwise/builtins-strategy/){:target="_blank" class="doc-btn"}

## WebGPU Object Caching Strategy

Comprehensive guide to Gridwise's approach for caching and reusing WebGPU objects (compute pipelines, bind groups, buffer layouts) across multiple invocations. Explains how object caching reduces GPU state setup overhead and improves throughput for repeated operations. Covers caching strategies for different primitive configurations, memory management of cached objects, and invalidation patterns for long-running applications.

[Read](https://gridwise-webgpu.github.io/gridwise/webgpu-object-caching-strategy/){:target="_blank" class="doc-btn"}

## Writing a WebGPU WGSL Workgroup Reduce Function

In-depth tutorial on implementing custom workgroup-level reduce functions in WGSL for integration with Gridwise primitives. Covers reduction patterns, memory synchronization with workgroup barriers, handling of non-power-of-2 workgroup sizes, and optimization techniques using subgroups where available. Includes complete code examples and validation strategies for custom reduce operations.

[Read](https://gridwise-webgpu.github.io/gridwise/writing-a-webgpu-wgsl-workgroup-reduce-function/){:target="_blank" class="doc-btn"}
