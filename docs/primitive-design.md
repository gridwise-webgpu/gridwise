---
layout: page
title: "Primitive Strategy with Subgroups"
category: gridwise
permalink: /primitive-design/
excerpt: "How Gridwise implements single-pass chained algorithms for sort, scan, and reduce with and without subgroup support."
order: 7
---

<div style="max-width: 100%; padding: 0;">

<!-- Header Section -->
<header style="max-width: 100%; margin-bottom: 3rem;">
  <h1 style="font-size: 2.5rem; font-weight: 800; color: #111827; margin-bottom: 1.5rem; line-height: 1.2;">Primitive Strategy with Subgroups</h1>
  <p style="font-size: 1.25rem; color: #6b7280; line-height: 1.75;">
    We wish to implement WebGPU sort, scan, and reduce. The fastest known GPU techniques for these operations are single-pass (chained) algorithms that minimize overall memory bandwidth. However, these techniques have historically been written using warp/subgroup support, and that subgroup support appears to be critical for their high performance.
  </p>
</header>

<!-- Section 1: Algorithm Overview -->
<section style="max-width: 100%; margin-bottom: 4rem;">
  <h2 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Single-Pass (Chained) Algorithm</h2>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    Most GPU algorithms are limited by memory bandwidth. This is also true of sort, scan, and reduce. The fastest algorithms are those that require the least memory traffic.
  </p>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    <strong>How it works:</strong>
  </p>
  <ol style="color: #4b5563; line-height: 1.75; margin-left: 1.5rem; margin-bottom: 1rem;">
    <li style="margin-bottom: 0.75rem;">The input is divided into tiles. We launch one workgroup per tile. Each workgroup can process its entire tile in parallel and in its own local workgroup memory.</li>
    <li style="margin-bottom: 0.75rem;">Each tile maintains a globally visible region of memory where it posts its results. Those results can be in one of three states:
      <ul style="margin-top: 0.5rem; margin-left: 1rem;">
        <li style="margin-bottom: 0.25rem;"><strong>"invalid"</strong> — no result yet</li>
        <li style="margin-bottom: 0.25rem;"><strong>"local reduction"</strong> — result is only a function of the current tile's input</li>
        <li style="margin-bottom: 0.25rem;"><strong>"global reduction"</strong> — result is a function of the current tile's input AND all previous tiles' inputs</li>
      </ul>
    </li>
    <li style="margin-bottom: 0.5rem;">Each tile follows these steps:
      <ul style="margin-top: 0.5rem; margin-left: 1rem;">
        <li style="margin-bottom: 0.25rem;">Consume its input tile and post a globally visible "local reduction" of that input</li>
        <li style="margin-bottom: 0.25rem;">Compute a globally visible "global reduction" by fetching the previous tile's global reduction result and combining it with the current tile's local reduction result</li>
        <li style="margin-bottom: 0.25rem;">Post that globally visible global reduction result to globally visible memory</li>
      </ul>
    </li>
  </ol>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    This approach computes tiles in parallel, but then serializes the reduction of the results of each of those tiles. The reduction operation itself is cheap, and all inputs and outputs are likely in cache.
  </p>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    <strong>Key refinements:</strong>
  </p>
  <ul style="color: #4b5563; line-height: 1.75; margin-left: 1.5rem; margin-bottom: 1.5rem;">
    <li style="margin-bottom: 0.5rem;"><strong>Lookback:</strong> If waiting on the previous tile's global reduction result, look farther back in the serialization chain to aggressively accumulate enough results</li>
    <li style="margin-bottom: 0.5rem;"><strong>Fallback:</strong> If waiting on the previous tile's local reduction to post, redundantly recompute that local reduction in your own workgroup</li>
  </ul>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1rem;">
    <h4 style="font-size: 1rem; font-weight: 700; color: #065f46; margin-bottom: 1rem;">Memory Efficiency Benefits</h4>
    <p style="color: #065f46; margin-bottom: 0.5rem;"><strong>Minimum global memory traffic:</strong></p>
    <ul style="color: #065f46; line-height: 1.5; margin-left: 1.5rem;">
      <li style="margin-bottom: 0.5rem;">Each input element read once</li>
      <li style="margin-bottom: 0.5rem;">Each output element written once</li>
      <li style="margin-bottom: 0.5rem;">Negligible intermediate traffic</li>
    </ul>
  </div>
  
  <div style="padding: 1.5rem; border: 2px solid #ef4444; border-radius: 0.5rem; background: #fef2f2;">
    <h4 style="font-size: 1rem; font-weight: 700; color: #991b1b; margin-bottom: 0.5rem;">⚠️ Performance Critical</h4>
    <p style="color: #991b1b; margin: 0;">Subgroup instructions are essential for sufficient throughput. Without them, workgroup barriers inhibit performance.</p>
  </div>
</section>

<!-- Section 2: Making Chained Algorithms Fast -->
<section style="max-width: 100%; margin-bottom: 4rem;">
  <h2 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">How Can We Make Chained Algorithms Fast?</h2>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    The most important implementation focus is to ensure that the entire implementation is <strong>memory-bound</strong>.
  </p>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    The computation per tile typically requires:
  </p>
  <ul style="color: #4b5563; line-height: 1.75; margin-left: 1.5rem; margin-bottom: 1rem;">
    <li style="margin-bottom: 0.5rem;">Reading a tile of data from memory</li>
    <li style="margin-bottom: 0.5rem;">Computing results based on that input tile data</li>
    <li style="margin-bottom: 0.5rem;">Writing results back to memory</li>
  </ul>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    For maximum overall throughput, the <strong>computation throughput must be faster than the memory throughput</strong>. In practice, on today's GPUs, this requires careful kernel design with respect to computation; simple kernels are likely to become performance bottlenecks.
  </p>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    Kernels with less memory demand (specifically, <strong>reduction kernels</strong>) are especially performance-sensitive because there is less memory traffic to cover the cost of the computation.
  </p>
  
  <div style="background: #fef2f2; border-left: 4px solid #ef4444; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;">
    <p style="color: #991b1b; font-weight: 600; margin: 0;">It appears likely that kernels must take advantage of subgroup instructions to achieve sufficient throughput. Without these primitives, kernels require numerous workgroup barriers that inhibit performance.</p>
  </div>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem;">
    <h4 style="font-size: 1rem; font-weight: 700; color: #065f46; margin-bottom: 1rem;">Subgroup Challenge</h4>
    <p style="color: #065f46; margin-bottom: 1rem;">
      Subgroup instructions are particularly challenging because different hardware has different subgroup sizes.
    </p>
    <p style="color: #065f46; margin-bottom: 0.5rem;"><strong>Common Subgroup Sizes:</strong></p>
    <ul style="color: #065f46; line-height: 1.5; margin-left: 1.5rem; margin-bottom: 1rem;">
      <li style="margin-bottom: 0.25rem;">NVIDIA: 32 threads (warp)</li>
      <li style="margin-bottom: 0.25rem;">AMD: 64 threads (wavefront)</li>
      <li style="margin-bottom: 0.25rem;">Intel: 8-32 threads (SIMD)</li>
    </ul>
    <p style="color: #065f46; margin: 0;">
      Writing <strong>subgroup-size-agnostic kernels</strong> is complex and requires careful abstraction.
    </p>
  </div>
</section>

<!-- Section 3: Design Choices -->
<section style="max-width: 100%; padding-top: 2rem; border-top: 1px solid #e5e7eb; margin-bottom: 3rem;">
  <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 2rem;">Design Choices</h2>

  <h3 style="font-size: 1.25rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Subgroup Support Strategy</h3>
  
  <p style="margin-bottom: 0.5rem;"><strong>Use subgroups everywhere + emulate:</strong></p>
  <ul style="margin-left: 1.5rem; margin-bottom: 1rem;">
    <li style="color: #10b981; margin-bottom: 0.25rem;">✓ One code base (easier maintenance)</li>
    <li style="color: #ef4444; margin-bottom: 0.25rem;">✗ Emulated code is not likely to be performance-competitive</li>
    <li style="color: #ef4444; margin-bottom: 0.25rem;">✗ Current subgroup support is fragile</li>
  </ul>

  <p style="margin-bottom: 0.5rem;"><strong>Never use subgroups:</strong></p>
  <ul style="margin-left: 1.5rem; margin-bottom: 2rem;">
    <li style="color: #10b981; margin-bottom: 0.25rem;">✓ Most portable</li>
    <li style="color: #ef4444; margin-bottom: 0.25rem;">✗ Unlikely to deliver top performance</li>
  </ul>

  <h3 style="font-size: 1.25rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Algorithm Strategy</h3>
  
  <p style="margin-bottom: 0.5rem;"><strong>Always use chained:</strong></p>
  <ul style="margin-left: 1.5rem; margin-bottom: 1rem;">
    <li style="color: #10b981; margin-bottom: 0.25rem;">✓ Likely to be highest-performance option</li>
    <li style="color: #10b981; margin-bottom: 0.25rem;">✓ Maintain only one code base</li>
    <li style="color: #ef4444; margin-bottom: 0.25rem;">✗ Most complex implementation</li>
    <li style="color: #ef4444; margin-bottom: 0.25rem;">✗ Unlikely to deliver good performance without subgroups, which present definite fragility challenges in the chained context</li>
  </ul>

  <p style="margin-bottom: 0.5rem;"><strong>Hybrid approach:</strong></p>
  <ul style="margin-left: 1.5rem; margin-bottom: 1rem;">
    <li style="color: #10b981; margin-bottom: 0.25rem;">✓ Allows most flexible performance tradeoffs between performance and capabilities</li>
    <li style="color: #ef4444; margin-bottom: 0.25rem;">✗ Must maintain two different code bases (little overlap)</li>
    <li style="color: #ef4444; margin-bottom: 0.25rem;">✗ Little ability to specialize beyond "has subgroups vs. no subgroups"</li>
  </ul>

  <p style="margin-bottom: 0.5rem;"><strong>Never use chained:</strong></p>
  <ul style="margin-left: 1.5rem; margin-bottom: 1.5rem;">
    <li style="color: #10b981; margin-bottom: 0.25rem;">✓ Well-known and -tested implementation strategy</li>
    <li style="color: #10b981; margin-bottom: 0.25rem;">✓ Maintain one code base</li>
    <li style="color: #10b981; margin-bottom: 0.25rem;">✓ Simplest code</li>
    <li style="color: #ef4444; margin-bottom: 0.25rem;">✗ Will not achieve top performance (theoretically, ⅔ the performance of chained approaches)</li>
  </ul>
</section>

</div>

<style>
@media (max-width: 768px) {
  section[style*="grid-template-columns: 1fr 1fr"] {
    grid-template-columns: 1fr !important;
  }
  div[style*="position: sticky"] {
    position: relative !important;
  }
}
</style>
