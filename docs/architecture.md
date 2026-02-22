---
layout: gridwise-home
title: "Gridwise Architecture"
category: gridwise
permalink: /architecture/
excerpt: "Overview of Gridwise's core architecture, including Primitives and Buffers, and how they provide best-in-class WebGPU compute performance."
order: 1
---

<div style="max-width: 100%; padding: 0;">

<!-- Header Section -->
<header style="max-width: 800px; margin-bottom: 3rem;">
  <div style="display: flex; align-items: center; gap: 0.5rem; color: #4f46e5; font-weight: 600; font-size: 0.875rem; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em;">
    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
    </svg>
    <span>Core Concepts</span>
  </div>
  <h1 style="font-size: 2.5rem; font-weight: 800; color: #111827; margin-bottom: 1.5rem; line-height: 1.2;">Architecture Overview</h1>
  <p style="font-size: 1.25rem; color: #6b7280; line-height: 1.75;">
    The primary goal of Gridwise is to deliver best-in-class performance on WebGPU compute primitives while minimizing the amount of code that must be written by the library user. Ideally, a Gridwise user will declare and then execute a primitive and Gridwise will handle all low-level details of setting up and calling the necessary WebGPU primitives.
  </p>
</header>

<!-- Gridwise Abstraction Title -->
<h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin: 3rem 0 2rem 0; padding-top: 2rem; border-top: 1px solid #e5e7eb;">Gridwise Abstraction</h2>

<!-- Section 1: Primitives (Split View) -->
<section style="display: grid; grid-template-columns: 1fr 1fr; gap: 3rem; margin-bottom: 4rem; align-items: start;">
  <div>
    <h3 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Primitive</h3>
    <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
      The primary abstraction in Gridwise is a <strong>Primitive</strong>. Primitives are instances of a primitive-specific subclass of a JavaScript <code>Primitive</code> class. They have an <code>execute</code> member function, and the typical usage is to instantiate a primitive using <code>new()</code> and then call <code>execute()</code> on that primitive. Both instantiation and execution have numerous options.
    </p>
    <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
      As an example, let's look at a scan primitive, which is an instance of the <code>DLDFScan</code> class ("decoupled-lookback, decoupled-fallback scan"):
    </p>
  </div>

  <div style="position: sticky; top: 6rem; background: #1f2937; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);">
    <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: #374151; border-bottom: 1px solid #4b5563;">
      <div style="display: flex; gap: 0.375rem;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
      </div>
      <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.75rem; color: #9ca3af; margin-left: 1rem;">scan-primitive.js</span>
    </div>
    <pre style="padding: 1.5rem; margin: 0; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.7; overflow-x: auto;"><code><span style="color: #a5b4fc">const</span> <span style="color: #e2e8f0">datatype = </span><span style="color: #34d399">"u32"</span>;
<span style="color: #a5b4fc">const</span> <span style="color: #e2e8f0">dldfscanPrimitive = </span><span style="color: #a5b4fc">new</span> <span style="color: #fbbf24">DLDFScan</span>({
  <span style="color: #94a3b8">device,</span>
  <span style="color: #e2e8f0">binop: </span><span style="color: #a5b4fc">new</span> <span style="color: #fbbf24">BinOpAdd</span>({ datatype }),
  <span style="color: #e2e8f0">type: </span><span style="color: #34d399">"exclusive"</span>, <span style="color: #6b7280">// "exclusive" is the default</span>
  <span style="color: #94a3b8">datatype,</span>
});

<span style="color: #a5b4fc">await</span> <span style="color: #e2e8f0">dldfscanPrimitive.</span><span style="color: #38bdf8">execute</span>({
  <span style="color: #94a3b8">inputBuffer,</span>
  <span style="color: #94a3b8">outputBuffer,</span>
});</code></pre>
  </div>
</section>

<!-- Primitive Details -->
<div style="max-width: 800px; margin-bottom: 3rem;">
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    This particular primitive is parameterized by its datatype (in this case, <code>"u32"</code>), by the binary operation ("binop") performed by the scan (in this case, addition on u32 data), and by the scan operation (exclusive or inclusive).
  </p>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    When the scan is actually executed, its arguments are buffers that store its input and output. This particular primitive has named arguments of an input buffer named <code>inputBuffer</code> and an output buffer named <code>outputBuffer</code>. These buffers can be WebGPU buffers of type <code>GPUBuffer</code> but can also be <code>Buffer</code>s, described next.
  </p>
  <p style="color: #4b5563; line-height: 1.75;">
    The primitive performs all necessary WebGPU operations, including (optionally) setting up an encoder, building up and setting WebGPU layouts and pipelines, running the pipeline, and optionally recording GPU-side or CPU-side timing. It also caches WebGPU layouts and pipelines to avoid the expense of recreating them if they have already been created.
  </p>
</div>

<!-- Section 2: Buffers -->
<section style="display: grid; grid-template-columns: 1fr 1fr; gap: 3rem; padding-top: 2rem; border-top: 1px solid #e5e7eb;">
  <div>
    <h3 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Buffer</h3>
    <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
      One of the challenges of writing a primitive library is handling data, which may be stored on the CPU (in a JavaScript typed array) or on the GPU (as a WebGPU <code>GPUBuffer</code>). Gridwise's <code>Buffer</code> class attempts to abstract away the details of separately managing CPU and GPU buffer data structures with one unified data structure that stores, and moves data between, both.
    </p>
    <p style="color: #4b5563; line-height: 1.75;">
      This data structure has grown organically to handle many use cases and deserves more focus by future developers as a principled data structure in WebGPU programming.
    </p>
  </div>

  <div style="display: flex; flex-direction: column; gap: 1rem;">
    <div style="padding: 1.5rem; border: 1px solid #e5e7eb; border-radius: 0.75rem; transition: all 0.3s;">
      <svg style="width: 2rem; height: 2rem; color: #4f46e5; margin-bottom: 1rem;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
      </svg>
      <h4 style="font-size: 1.125rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem;">State Management</h4>
      <p style="font-size: 0.875rem; color: #6b7280; margin: 0; line-height: 1.5;">Automatically tracks if data is "dirty" on the CPU or GPU and syncs only when necessary.</p>
    </div>
    <div style="padding: 1.5rem; border: 1px solid #e5e7eb; border-radius: 0.75rem; transition: all 0.3s;">
      <svg style="width: 2rem; height: 2rem; color: #4f46e5; margin-bottom: 1rem;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
      <h4 style="font-size: 1.125rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem;">Type Safety</h4>
      <p style="font-size: 0.875rem; color: #6b7280; margin: 0; line-height: 1.5;">Maps JavaScript TypedArrays directly to GPU-side memory layouts.</p>
    </div>
  </div>
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
