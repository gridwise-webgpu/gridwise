---
layout: post
title: "Gridwise WebGPU Primitive Strategy wrt Subgroups"
date: 2025-08-21
---

We wish to implement WebGPU sort, scan, and reduce. The fastest known GPU techniques for these operations are single-pass (chained) algorithms that minimize overall memory bandwidth. However, these techniques have historically been written using warp/subgroup support, and that subgroup support appears to be critical for their high performance. This document looks at the different considerations for different primitive implementations.

## Brief overview of single-pass (chained) algorithm

Most GPU algorithms are limited by memory bandwidth. This is also true of sort, scan, and reduce. The fastest algorithms are those that require the least memory traffic. This description is high-level and elides many details.

The single-pass (chained) algorithms work as follows:

- The input is divided into tiles. We launch one workgroup per tile. Each workgroup can process its entire tile in parallel and in its own local workgroup memory.
- Each tile maintains a globally visible region of memory where it posts its results. Those results can be in one of three states: “invalid”, “local reduction” (reflecting a result that is only the function of the current tile’s input), and “global reduction” (reflecting a result that is a function of the current tile’s input AND all previous tiles’ inputs).
- Each tile follows these steps:
  - Consume its input tile and post a globally visible “local reduction” of that input.
  - Compute a globally visible “global reduction”. The global reduction is the result of ALL tiles up to and including the current tile. In practice, this is computed by fetching the previous tile’s global reduction result and combining it with the current tile’s local reduction result, resulting in a global reduction result for my tile.
  - Post that globally visible global reduction result to globally visible memory.
- This approach computes tiles in parallel, but then serializes the reduction of the results of each of those tiles. Serializing this reduction seems like a performance bottleneck. However, the reduction operation itself is cheap, and all inputs and outputs for this reduction operation are likely in cache. In practice, this reduction is not a bottleneck on modern GPUs.
- Refinements to this process include:
  - If your workgroup is waiting on the previous tile’s global reduction result, look farther back in the serialization chain to aggressively accumulate enough results for your tile to reconstruct the global summary your workgroup needs. (“Lookback”)
  - If your workgroup is waiting on the previous tile’s local reduction to post, just (redundantly) recompute that local reduction in your own workgroup. (“Fallback”)

The benefit to a single-pass (chained) algorithm is that it requires a minimum of global memory traffic: each input element is only read once, each output element is only written once, and intermediate memory traffic is negligible compared to reads/writes of input/output elements.

### How can we make chained algorithms fast?

The most important implementation focus in these algorithms is to ensure that the entire implementation is memory-bound. The computation per tile typically requires reading a tile of data from memory, computing results based on that input tile data, and writing results back to memory. For maximum overall throughput, the computation throughput must be faster than the memory throughput. In practice, on today’s GPUs, this requires careful kernel design with respect to computation; simple kernels are likely to become performance bottlenecks. Kernels with less memory demand (specifically, reduction kernels) are especially performance-sensitive because there is less memory traffic to cover the cost of the computation.

Specifically, it appears likely that kernels must take advantage of subgroup instructions to achieve sufficient throughput. Without these primitives, kernels require numerous workgroup barriers that inhibit performance. Subgroup instructions are particularly challenging because different hardware has different subgroup sizes; writing subgroup-size-agnostic kernels is complex.

## Design choice: Use subgroups \+ emulation vs. no-subgroups

### Use subgroup instructions everywhere, emulate subgroup instructions in software where not available

- \+: One code base (easier maintenance)
- –: Emulated code is not likely to be performance-competitive
- –: Current subgroup support is fragile

### Never use subgroups anywhere

- \+: Most portable
- –: Unlikely to deliver top performance

## Design choice: Use chained algorithms vs. hybrid vs. not

### Always use chained algorithms

- \+: Likely to be highest-performance option
- \+: Maintain only one code base
- –: Most complex implementation
- –: Unlikely to deliver good performance without subgroups, which present definite fragility challenges in the chained context

### Hybrid approach: sometimes use chained algorithms, sometimes don’t

- \+: Allows most flexible performance tradeoffs between performance and capabilities
- –: Must maintain two different code bases (little overlap)
- –: Little ability to specialize beyond “has subgroups vs. no subgroups”

### Never use chained algorithms

- \+: Well-known and \-tested implementation strategy
- \+: Maintain one code base
- \+: Simplest code
- –: Will not achieve top performance (theoretically, ⅔ the performance of chained approaches)
