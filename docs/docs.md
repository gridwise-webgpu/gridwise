---
layout: home
title: Documentation
permalink: /documentation/
---

Comprehensive guides and technical documentation for Gridwise WebGPU primitives.

## Architecture

Overview of Gridwise's system design, module structure, and how primitives are organized for extensibility and
performance. Learn about the high-level organization of Gridwise components, including how different primitives
(scan, reduce, sort) are implemented as modular, reusable units. Understand the architectural decisions that enable
performance optimization while maintaining clean separation of concerns and ease of extension.

<a href="{{ "/architecture/" | relative_url }}" class="doc-btn">Read</a>

## Primitive Design

Deep dive into the design principles behind Gridwise primitives with focus on single-pass chained algorithms for
sort, scan, and reduce. Explores the tradeoffs between using subgroup instructions for maximum performance versus
software emulation for broader compatibility. Covers memory bandwidth considerations, the lookback and fallback
optimization techniques, and how to choose between chained algorithms and hybrid approaches for different use cases.

<a href="{{ "/primitive-design/" | relative_url }}" class="doc-btn">Read</a>

## Scan and Reduce

Comprehensive guide to scan (prefix sum) and reduce operations in Gridwise. Explains the difference between exclusive
scan (first element is identity), inclusive scan (each element includes itself), and reduce (single output value).
Covers binary operations (Add, Min, Max), data type support (u32, i32, f32), API usage patterns with code examples,
and when to use each variant for optimal performance.

<a href="{{ "/scan-and-reduce/" | relative_url }}" class="doc-btn">Read</a>

## Sort

Complete documentation for Gridwise's OneSweepSort implementation. Covers both key-only sorting and key-value pair
sorting with full payload support. Explains configurable sort direction (ascending/descending), supported data
types, buffer management strategies, and in-place versus temporary buffer approaches. Includes detailed API
documentation and performance characteristics across different input sizes and configurations.

<a href="{{ "/sort/" | relative_url }}" class="doc-btn">Read</a>

## Binary Operations

Guide to binary operations used in Gridwise's scan and reduce primitives. Documents available operations (Add, Min,
Max, Multiply) and their properties. Explains how to implement custom binary operations by extending the binop
interface, including implementation requirements, data type constraints, and validation patterns. Critical for users
who need domain-specific aggregation operations.

<a href="{{ "/binop/" | relative_url }}" class="doc-btn">Read</a>

## Buffer Management

Best practices for allocating, managing, and optimizing GPU buffers in Gridwise applications. Covers buffer creation
strategies, memory usage patterns, and how to minimize memory allocation overhead. Explains the relationship between
buffer sizes and performance, copy strategies for input/output, and how to handle edge cases with non-aligned input
lengths. Essential for building efficient Gridwise applications.

<a href="{{ "/buffer/" | relative_url }}" class="doc-btn">Read</a>

## Timing Strategy

Detailed explanation of timing mechanisms in Gridwise for accurate performance measurement and benchmarking. Covers
both CPU timing (performance.now) and GPU timing (timestamp queries) approaches, their accuracy tradeoffs, and when
to use each. Explains warmup strategies, trial averaging, and how to interpret results across different hardware
configurations for reliable performance comparisons.

<a href="{{ "/timing-strategy/" | relative_url }}" class="doc-btn">Read</a>

## Subgroup Strategy

Detailed guide to GPU subgroups and their critical role in Gridwise primitive performance. Explains what subgroups
are, how different GPU architectures have different subgroup sizes, and the performance benefits of subgroup
operations. Covers Gridwise's approach to subgroup detection, optional subgroup acceleration, and fallback
strategies for hardware without subgroup support to maintain broad device compatibility.

<a href="{{ "/subgroup-strategy/" | relative_url }}" class="doc-btn">Read</a>

## Built-ins Strategy

Exploration of WebGPU WGSL built-in functions and how Gridwise strategically selects and optimizes their use in
primitive implementations. Explains which built-ins provide the best performance for reduction operations,
aggregation patterns, and data movement. Covers vendor-specific optimizations and how to identify when built-in
usage versus hand-tuned WGSL code provides the best performance on different hardware.

<a href="{{ "/builtins-strategy/" | relative_url }}" class="doc-btn">Read</a>

## WebGPU Object Caching Strategy

Comprehensive guide to Gridwise's approach for caching and reusing WebGPU objects (compute pipelines, bind groups,
buffer layouts) across multiple invocations. Explains how object caching reduces GPU state setup overhead and
improves throughput for repeated operations. Covers caching strategies for different primitive configurations,
memory management of cached objects, and invalidation patterns for long-running applications.

<a href="{{ "/webgpu-object-caching-strategy/" | relative_url }}" class="doc-btn">Read</a>

## Writing a WebGPU WGSL Workgroup Reduce Function

In-depth tutorial on implementing custom workgroup-level reduce functions in WGSL for integration with Gridwise
primitives. Covers reduction patterns, memory synchronization with workgroup barriers, handling of non-power-of-2
workgroup sizes, and optimization techniques using subgroups where available. Includes complete code examples and
validation strategies for custom reduce operations.

<a href="{{ "/writing-a-webgpu-wgsl-workgroup-reduce-function/" | relative_url }}" class="doc-btn">Read</a>
