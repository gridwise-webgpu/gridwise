---
layout: page
title: "Object Caching Strategy"
category: gridwise
permalink: /webgpu-object-caching-strategy/
excerpt: "How Gridwise caches WebGPU objects to improve performance by avoiding expensive object recreation."
order: 9
---

<div style="margin-bottom: 3rem;">
  <div style="display: inline-flex; align-items: center; gap: 0.75rem; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 0.5rem 1.25rem; border-radius: 0.5rem; font-weight: 600; font-size: 0.875rem; margin-bottom: 1.5rem;">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
      <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
      <line x1="12" y1="22.08" x2="12" y2="12"/>
    </svg>
    Object Caching
  </div>
  <h1 style="font-size: 2.5rem; font-weight: 800; color: #1e293b; margin: 0 0 1rem 0; line-height: 1.2;">WebGPU Object Caching Strategy</h1>
  <p style="font-size: 1.25rem; color: #64748b; margin: 0; line-height: 1.6;">How Gridwise caches WebGPU objects to improve performance by avoiding expensive object recreation.</p>
</div>

<div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;">
  <p style="margin: 0 0 1rem 0; color: #064e3b; line-height: 1.6;">
    This document outlines the caching strategy used for WebGPU objects within Gridwise. Creating WebGPU objects is not free and is potentially expensive. Caching created objects so that they can be reused potentially helps performance.
  </p>
  <p style="margin: 0; color: #064e3b; line-height: 1.6;">
    The downsides of caching are that caching itself is not free and that the WebGPU back end may do its own caching. In Gridwise, <strong>caching is enabled by default</strong> but can be disabled (by instantiating a primitive with the argument <code style="background: #dcfce7; color: #166534; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">webgpucache = "disabled"</code>).
  </p>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #1e293b; margin: 3rem 0 1.5rem 0; padding-bottom: 0.75rem; border-bottom: 3px solid #10b981;">Cacheable WebGPU Objects</h2>

<div style="background: white; border: 2px solid #d1fae5; border-radius: 0.75rem; padding: 2rem; margin-bottom: 2rem;">
  <p style="margin: 0 0 1.5rem 0; color: #475569; line-height: 1.6;">
    The following WebGPU objects are currently cached by our library:
  </p>

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
    <div style="background: #f0fdf4; border: 2px solid #bbf7d0; border-radius: 0.5rem; padding: 1rem;">
      <code style="color: #059669; font-weight: 700; font-size: 1rem;">GPUShaderModule</code>
    </div>
    <div style="background: #f0fdf4; border: 2px solid #bbf7d0; border-radius: 0.5rem; padding: 1rem;">
      <code style="color: #059669; font-weight: 700; font-size: 1rem;">GPUPipelineLayout</code>
    </div>
    <div style="background: #f0fdf4; border: 2px solid #bbf7d0; border-radius: 0.5rem; padding: 1rem;">
      <code style="color: #059669; font-weight: 700; font-size: 1rem;">GPUBindGroupLayout</code>
    </div>
    <div style="background: #f0fdf4; border: 2px solid #bbf7d0; border-radius: 0.5rem; padding: 1rem;">
      <code style="color: #059669; font-weight: 700; font-size: 1rem;">GPUComputePipeline</code>
    </div>
  </div>

  <div style="background: #ecfdf5; border: 1px solid #a7f3d0; padding: 1.25rem; border-radius: 0.5rem;">
    <p style="margin: 0; color: #065f46; line-height: 1.6; font-size: 0.9375rem;">
      Of these, <code style="background: #d1fae5; color: #065f46; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">GPUShaderModule</code> is potentially independent of the <code style="background: #d1fae5; color: #065f46; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">GPUDevice</code>, while the others are dependent on the specific <code style="background: #d1fae5; color: #065f46; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">GPUDevice</code> instance.
    </p>
  </div>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #1e293b; margin: 3rem 0 1.5rem 0; padding-bottom: 0.75rem; border-bottom: 3px solid #10b981;">Cache Implementation</h2>

<div style="background: white; border: 2px solid #d1fae5; border-radius: 0.75rem; padding: 2rem; margin-bottom: 2rem;">
  <p style="margin: 0 0 1.5rem 0; color: #475569; line-height: 1.6;">
    Every primitive shares a <code style="background: #f1f5f9; color: #059669; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">__deviceToWebGPUObjectCache</code>, which is a <code style="background: #f1f5f9; color: #059669; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">WeakMap</code> that maps a <code style="background: #f1f5f9; color: #059669; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">GPUDevice</code> to its corresponding cache. Each device's cache contains several individual caches for different object types. These are regular <code style="background: #f1f5f9; color: #059669; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">Map</code> objects that map a generated key to the WebGPU object.
  </p>

  <p style="margin: 0 0 1rem 0; color: #475569; line-height: 1.6; font-weight: 600;">
    The available caches are:
  </p>

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
    <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1rem; border-radius: 0.5rem;">
      <code style="color: #059669; font-weight: 700; font-size: 0.9375rem;">pipelineLayouts</code>
    </div>
    <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1rem; border-radius: 0.5rem;">
      <code style="color: #059669; font-weight: 700; font-size: 0.9375rem;">bindGroupLayouts</code>
    </div>
    <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1rem; border-radius: 0.5rem;">
      <code style="color: #059669; font-weight: 700; font-size: 0.9375rem;">computeModules</code>
    </div>
    <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1rem; border-radius: 0.5rem;">
      <code style="color: #059669; font-weight: 700; font-size: 0.9375rem;">computePipelines</code>
    </div>
  </div>

  <div style="background: #ecfdf5; border: 1px solid #a7f3d0; padding: 1.25rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
    <p style="margin: 0; color: #065f46; line-height: 1.6; font-size: 0.9375rem;">
      Each of these caches can be <strong>individually enabled or disabled</strong> when a primitive is created.
    </p>
  </div>

  <p style="margin: 0 0 1rem 0; color: #475569; line-height: 1.6;">
    Here is a simplified code representation of the cache structure:
  </p>

  <div style="background: #1e293b; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
    <div style="background: linear-gradient(135deg, #374151 0%, #1f2937 100%); padding: 0.75rem 1.25rem; border-bottom: 1px solid #374151; display: flex; align-items: center; justify-content: space-between;">
      <div style="display: flex; gap: 0.5rem; align-items: center;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
      </div>
      <span style="color: #9ca3af; font-size: 0.875rem; font-weight: 500;">cache-structure.js</span>
    </div>
    <div style="padding: 1.5rem; overflow-x: auto;">
      <pre style="margin: 0; color: #e5e7eb; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.9375rem; line-height: 1.6;"><span style="color: #818cf8;">export class</span> <span style="color: #10b981;">BasePrimitive</span> {
  <span style="color: #818cf8;">static</span> __deviceToWebGPUObjectCache <span style="color: #818cf8;">=</span> <span style="color: #818cf8;">new</span> <span style="color: #10b981;">WeakMap</span>();
  <span style="color: #64748b;">// ... inside a method ...</span>
  BasePrimitive.__deviceToWebGPUObjectCache.<span style="color: #10b981;">set</span>(
    <span style="color: #818cf8;">this</span>.device,
    <span style="color: #818cf8;">new</span> <span style="color: #10b981;">WebGPUObjectCache</span>()
  );
}

<span style="color: #818cf8;">class</span> <span style="color: #10b981;">WebGPUObjectCache</span> {
  <span style="color: #10b981;">constructor</span>() {
    <span style="color: #818cf8;">this</span>.caches <span style="color: #818cf8;">=</span> [
      <span style="color: #a5f3fc;">"pipelineLayouts"</span>,
      <span style="color: #a5f3fc;">"bindGroupLayouts"</span>,
      <span style="color: #a5f3fc;">"computeModules"</span>,
      <span style="color: #a5f3fc;">"computePipelines"</span>,
    ];

    <span style="color: #818cf8;">for</span> (<span style="color: #818cf8;">const</span> cache <span style="color: #818cf8;">of</span> <span style="color: #818cf8;">this</span>.caches) {
      <span style="color: #818cf8;">this</span>[cache] <span style="color: #818cf8;">=</span> <span style="color: #818cf8;">new</span> <span style="color: #10b981;">CountingMap</span>({ <span style="color: #64748b;">// wrapper over a Map</span>
        enabled: <span style="color: #818cf8;">this</span>.initiallyEnabled.<span style="color: #10b981;">includes</span>(cache),
      });
    }
  }
}</pre>
    </div>
  </div>
</div>

<div style="background: white; border: 2px solid #d1fae5; border-radius: 0.75rem; padding: 2rem; margin-bottom: 2rem;">
  <h3 style="margin: 0 0 1.5rem 0; color: #1e293b; font-size: 1.25rem; font-weight: 700; display: flex; align-items: center; gap: 0.75rem;">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="15" y1="9" x2="9" y2="15"/>
    </svg>
    Bind Group Caching
  </h3>
  
  <div style="background: #ecfdf5; border: 1px solid #a7f3d0; padding: 1.25rem; border-radius: 0.5rem;">
    <p style="margin: 0; color: #065f46; line-height: 1.6;">
      <strong>Bind groups are not cached.</strong> This decision was made because bind groups depend on <code style="background: #d1fae5; color: #065f46; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">GPUBuffer</code> objects. Reliably creating a cache key from a <code style="background: #d1fae5; color: #065f46; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">GPUBuffer</code> is problematic due to its dynamic state.
    </p>
  </div>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #1e293b; margin: 3rem 0 1.5rem 0; padding-bottom: 0.75rem; border-bottom: 3px solid #10b981;">Cache Key Generation</h2>

<div style="background: white; border: 2px solid #d1fae5; border-radius: 0.75rem; padding: 2rem; margin-bottom: 2rem;">
  <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;">
    <p style="margin: 0; color: #064e3b; line-height: 1.6;">
      To use objects as keys in a Map, we need a consistent and unique representation. Since Maps use Same-Value-Zero equality, two different objects with the same properties will not be treated as the same key. To solve this, we <code style="background: #dcfce7; color: #166534; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">JSON.stringify</code> a simplified representation of the object to create a string-based cache key.
    </p>
  </div>

  <p style="margin: 0 0 1.5rem 0; color: #475569; line-height: 1.6; font-weight: 600;">
    Here's how keys are generated for different object types:
  </p>

  <div style="display: flex; flex-direction: column; gap: 2rem;">
    <!-- Pipeline Layout -->
    <div style="background: #f8fafc; border-radius: 0.75rem; padding: 1.5rem; border-left: 4px solid #10b981;">
      <h4 style="margin: 0 0 1rem 0; color: #1e293b; font-size: 1.125rem; font-weight: 700;">Pipeline Layout</h4>
      <p style="margin: 0 0 1rem 0; color: #475569; line-height: 1.6; font-size: 0.9375rem;">
        The cache key for a <code style="background: #e2e8f0; color: #059669; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem;">GPUPipelineLayout</code> is an array of strings representing the buffer types for that layout.
      </p>
      <div style="background: #1e293b; border-radius: 0.5rem; overflow: hidden;">
        <div style="background: #374151; padding: 0.5rem 1rem; border-bottom: 1px solid #4b5563;">
          <span style="color: #9ca3af; font-size: 0.875rem; font-weight: 500;">Example</span>
        </div>
        <div style="padding: 1rem; overflow-x: auto;">
          <pre style="margin: 0; color: #e5e7eb; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.875rem; line-height: 1.6;">[<span style="color: #a5f3fc;">"read-only-storage"</span>, <span style="color: #a5f3fc;">"storage"</span>, <span style="color: #a5f3fc;">"uniform"</span>, <span style="color: #a5f3fc;">"storage"</span>, <span style="color: #a5f3fc;">"storage"</span>, <span style="color: #a5f3fc;">"storage"</span>];</pre>
        </div>
      </div>
    </div>

    <!-- Bind Group Layout -->
    <div style="background: #f8fafc; border-radius: 0.75rem; padding: 1.5rem; border-left: 4px solid #10b981;">
      <h4 style="margin: 0 0 1rem 0; color: #1e293b; font-size: 1.125rem; font-weight: 700;">Bind Group Layout</h4>
      <p style="margin: 0 0 1rem 0; color: #475569; line-height: 1.6; font-size: 0.9375rem;">
        The cache key for a <code style="background: #e2e8f0; color: #059669; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem;">GPUBindGroupLayout</code> is the set of entries for that layout.
      </p>
      <div style="background: #1e293b; border-radius: 0.5rem; overflow: hidden;">
        <div style="background: #374151; padding: 0.5rem 1rem; border-bottom: 1px solid #4b5563;">
          <span style="color: #9ca3af; font-size: 0.875rem; font-weight: 500;">Example</span>
        </div>
        <div style="padding: 1rem; overflow-x: auto;">
          <pre style="margin: 0; color: #e5e7eb; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.875rem; line-height: 1.6;">[
  { binding: <span style="color: #fbbf24;">0</span>, visibility: <span style="color: #fbbf24;">4</span>, buffer: { type: <span style="color: #a5f3fc;">"read-only-storage"</span> } },
  {
    binding: <span style="color: #fbbf24;">1</span>,
    visibility: <span style="color: #fbbf24;">4</span>,
    buffer: {
      <span style="color: #64748b;">/*...*/</span>
    },
  },
  <span style="color: #64748b;">// ... and so on for all entries</span>
];</pre>
        </div>
      </div>
    </div>

    <!-- Compute Module -->
    <div style="background: #f8fafc; border-radius: 0.75rem; padding: 1.5rem; border-left: 4px solid #10b981;">
      <h4 style="margin: 0 0 1rem 0; color: #1e293b; font-size: 1.125rem; font-weight: 700;">Compute Module</h4>
      <p style="margin: 0; color: #475569; line-height: 1.6; font-size: 0.9375rem;">
        The cache key for a <code style="background: #e2e8f0; color: #059669; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem;">GPUShaderModule</code> (referred to as a compute module in the context of compute shaders) is the <strong>entire kernel string</strong>. While underlying WebGPU engines like Dawn might have their own caching mechanisms, we implement a library-level cache for them as well.
      </p>
    </div>

    <!-- Compute Pipelines -->
    <div style="background: #f8fafc; border-radius: 0.75rem; padding: 1.5rem; border-left: 4px solid #10b981;">
      <h4 style="margin: 0 0 1rem 0; color: #1e293b; font-size: 1.125rem; font-weight: 700;">Compute Pipelines</h4>
      <p style="margin: 0 0 1rem 0; color: #475569; line-height: 1.6; font-size: 0.9375rem;">
        The cache key for a <code style="background: #e2e8f0; color: #059669; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem;">GPUComputePipeline</code> is derived from its descriptor object. Since the <code style="background: #e2e8f0; color: #059669; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem;">GPUPipelineLayout</code> and <code style="background: #e2e8f0; color: #059669; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem;">GPUShaderModule</code> are cached separately, we can reuse their cache keys to optimize the key generation for the pipeline itself.
      </p>
      <div style="background: #ecfdf5; border: 1px solid #a7f3d0; padding: 1rem; border-radius: 0.5rem;">
        <p style="margin: 0; color: #065f46; line-height: 1.6; font-size: 0.875rem;">
          A <code style="background: #d1fae5; color: #065f46; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.8125rem; font-weight: 600;">__cacheKey</code> property is stored on cacheable objects, and this key is used during stringification to avoid deep, recursive serialization.
        </p>
      </div>
    </div>
  </div>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #1e293b; margin: 3rem 0 1.5rem 0; padding-bottom: 0.75rem; border-bottom: 3px solid #10b981;">Cache Statistics</h2>

<div style="background: white; border: 2px solid #d1fae5; border-radius: 0.75rem; padding: 2rem; margin-bottom: 2rem;">
  <p style="margin: 0 0 1.5rem 0; color: #475569; line-height: 1.6;">
    The caches collect <strong>hit and miss statistics</strong> to help understand their effectiveness.
  </p>

  <p style="margin: 0 0 1rem 0; color: #475569; line-height: 1.6; font-size: 0.9375rem;">
    Example output from statistics collection:
  </p>

  <div style="background: #1e293b; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
    <div style="background: linear-gradient(135deg, #374151 0%, #1f2937 100%); padding: 0.75rem 1.25rem; border-bottom: 1px solid #374151; display: flex; align-items: center; justify-content: space-between;">
      <div style="display: flex; gap: 0.5rem; align-items: center;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
      </div>
      <span style="color: #9ca3af; font-size: 0.875rem; font-weight: 500;">cache-stats.txt</span>
    </div>
    <div style="padding: 1.5rem; overflow-x: auto;">
      <pre style="margin: 0; color: #e5e7eb; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.9375rem; line-height: 1.6;"><span style="color: #10b981;">Cache hits/misses:</span>
<span style="color: #e5e7eb;">Pipeline layouts:</span>     <span style="color: #a5f3fc;">7</span><span style="color: #64748b;">/</span><span style="color: #fbbf24;">1</span>
<span style="color: #e5e7eb;">Bind group layouts:</span>   <span style="color: #a5f3fc;">0</span><span style="color: #64748b;">/</span><span style="color: #fbbf24;">1</span>
<span style="color: #e5e7eb;">Compute modules:</span>      <span style="color: #a5f3fc;">4</span><span style="color: #64748b;">/</span><span style="color: #fbbf24;">4</span>
<span style="color: #e5e7eb;">Compute pipelines:</span>    <span style="color: #a5f3fc;">4</span><span style="color: #64748b;">/</span><span style="color: #fbbf24;">4</span></pre>
    </div>
  </div>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #1e293b; margin: 3rem 0 1.5rem 0; padding-bottom: 0.75rem; border-bottom: 3px solid #10b981;">Measuring Performance with CPU Timing</h2>

<div style="background: white; border: 2px solid #d1fae5; border-radius: 0.75rem; padding: 2rem; margin-bottom: 2rem;">
  <p style="margin: 0 0 1.5rem 0; color: #475569; line-height: 1.6;">
    To measure the performance impact of caching, enable CPU timing. This will wait for the GPU to finish its work and then record the CPU time taken.
  </p>

  <div style="background: #fef3c7; border: 1px solid #fbbf24; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
    <p style="margin: 0; color: #78350f; line-height: 1.6; font-size: 0.875rem; font-style: italic;">
      <strong>TODO:</strong> move this into the timing article
    </p>
  </div>

  <div style="background: #1e293b; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
    <div style="background: linear-gradient(135deg, #374151 0%, #1f2937 100%); padding: 0.75rem 1.25rem; border-bottom: 1px solid #374151; display: flex; align-items: center; justify-content: space-between;">
      <div style="display: flex; gap: 0.5rem; align-items: center;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
      </div>
      <span style="color: #9ca3af; font-size: 0.875rem; font-weight: 500;">cpu-timing-example.js</span>
    </div>
    <div style="padding: 1.5rem; overflow-x: auto;">
      <pre style="margin: 0; color: #e5e7eb; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.9375rem; line-height: 1.6;"><span style="color: #818cf8;">const</span> commandBuffer <span style="color: #818cf8;">=</span> encoder.<span style="color: #10b981;">finish</span>();

<span style="color: #818cf8;">if</span> (args?.<span style="color: #10b981;">enableCPUTiming</span>) {
  <span style="color: #818cf8;">await</span> <span style="color: #818cf8;">this</span>.device.queue.<span style="color: #10b981;">onSubmittedWorkDone</span>();
  <span style="color: #818cf8;">this</span>.cpuStartTime <span style="color: #818cf8;">=</span> performance.<span style="color: #10b981;">now</span>();
}

<span style="color: #818cf8;">this</span>.device.queue.<span style="color: #10b981;">submit</span>([commandBuffer]);

<span style="color: #818cf8;">if</span> (args?.<span style="color: #10b981;">enableCPUTiming</span>) {
  <span style="color: #818cf8;">await</span> <span style="color: #818cf8;">this</span>.device.queue.<span style="color: #10b981;">onSubmittedWorkDone</span>();
  <span style="color: #818cf8;">this</span>.cpuEndTime <span style="color: #818cf8;">=</span> performance.<span style="color: #10b981;">now</span>();
}</pre>
    </div>
  </div>
</div>
