---
layout: page
title: "Timing Strategy"
category: gridwise
permalink: /timing-strategy/
excerpt: "CPU and GPU timing methodologies for measuring and optimizing WebGPU primitive performance in Gridwise."
order: 10
---

<div style="max-width: 100%; padding: 0;">

<header style="max-width: 800px; margin-bottom: 3rem;">
  <h1 style="font-size: 2.5rem; font-weight: 800; color: #111827; margin-bottom: 1.5rem; line-height: 1.2;">Timing Strategy</h1>
  <p style="font-size: 1.25rem; color: #6b7280; line-height: 1.75;">
    CPU and GPU timing methodologies for measuring and optimizing WebGPU primitive performance in Gridwise.
  </p>
</header>

<div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;">
  <p style="margin: 0; color: #065f46; line-height: 1.75;">
    Building a high-performance WebGPU primitive library requires careful timing measurements to ensure that primitives deliver high performance to their users. In developing Gridwise, we designed both <strong>CPU</strong> and <strong>GPU timing methodologies</strong> and describe them here.
  </p>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin: 3rem 0 2rem 0; padding-top: 2rem; border-top: 1px solid #e5e7eb;">GPU Timing</h2>

<div style="max-width: 800px; margin-bottom: 3rem;">
  <h3 style="font-size: 1.25rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">WebGPU Timestamp Queries</h3>
  
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    Gridwise's GPU timing uses WebGPU's GPU timestamp queries. Gregg Tavares's <a href="https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html" style="color: #10b981; text-decoration: none; font-weight: 600;">"WebGPU Timing Performance"</a> is both an excellent overview of using timestamp queries as well as the basis for our implementation. We began with his <a href="https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html#a-timing-helper" style="color: #10b981; text-decoration: none; font-weight: 600;">timing helper</a> and augmented it to support calling multiple kernels (which is useful for benchmarking purposes).
  </p>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
    <p style="margin: 0; color: #065f46; line-height: 1.75;">
      In our implementation, the <code>Primitive</code> class has a static <code>__timingHelper</code> member that is initialized to understand. The timing helper is enabled by passing <code>enableGPUTiming: true</code> as a member of the argument object to <code>primitive.execute()</code> (i.e., <code>primitive.execute({ enableGPUTiming: true, ...})</code>).
    </p>
  </div>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    <strong>Initialization:</strong> When GPU timing is enabled, <code>primitive.execute</code> checks for a defined <code>__timingHelper</code> and initializes one if not, counting the number of kernels in the current primitive for the timing helper's initialization.
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    <strong>Per-Kernel Timing:</strong> We record a separate timing value for each kernel in the primitive and thus return a <strong>list of kernel time durations</strong>, one list element per kernel.
  </p>

  <p style="color: #4b5563; line-height: 1.75;">
    <strong>Multiple Trials:</strong> <code>primitive.execute</code>'s argument object also takes a <code>trials</code> argument that defaults to 1. If <code>trials</code> is <em>n</em>, then every kernel dispatch within the primitive is replaced with <em>n</em> consecutive dispatches. Running more trials is helpful for more reliable timing measurements.
  </p>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin: 3rem 0 2rem 0; padding-top: 2rem; border-top: 1px solid #e5e7eb;">CPU Timing</h2>

<div style="max-width: 800px; margin-bottom: 3rem;">
  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
    <p style="margin: 0; color: #065f46; line-height: 1.75;">
      CPU timing is only possible if the primitive creates its own encoder to ensure that the resulting command buffer only reflects the work within the primitive. It is enabled in a similar way to GPU timing (<code>primitive.execute({ enableCPUTiming: true, ...})</code>).
    </p>
  </div>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    In our internal discussions, we have settled on the following way to best measure CPU timing, which is what is implemented within Gridwise, but we are open to better suggestions here. Our strategy is to clear the device's queue, record the current time as <code>cpuStartTime</code>, submit the command buffer and wait for the queue to clear, then record the current time as <code>cpuEndTime</code>. The elapsed time is the time between the two CPU time stamps.
  </p>

  <div style="background: #1f2937; border-radius: 0.75rem; overflow: hidden; margin-bottom: 2rem;">
    <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: #374151; border-bottom: 1px solid #4b5563;">
      <div style="display: flex; gap: 0.375rem;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
      </div>
      <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.75rem; color: #9ca3af; margin-left: 1rem;">cpu-timing.js</span>
    </div>
    <pre style="padding: 1.5rem; margin: 0; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.7; overflow-x: auto;"><code><span style="color: #a5b4fc">const</span> <span style="color: #e2e8f0">commandBuffer = encoder.</span><span style="color: #34d399">finish</span>();
<span style="color: #a5b4fc">if</span> (<span style="color: #e2e8f0">args?.</span><span style="color: #94a3b8">enableCPUTiming</span>) {
  <span style="color: #a5b4fc">await</span> <span style="color: #a5b4fc">this</span><span style="color: #e2e8f0">.device.queue.</span><span style="color: #34d399">onSubmittedWorkDone</span>();
  <span style="color: #a5b4fc">this</span><span style="color: #e2e8f0">.cpuStartTime = performance.</span><span style="color: #34d399">now</span>();
}
<span style="color: #a5b4fc">this</span><span style="color: #e2e8f0">.device.queue.</span><span style="color: #34d399">submit</span>([<span style="color: #94a3b8">commandBuffer</span>]);
<span style="color: #a5b4fc">if</span> (<span style="color: #e2e8f0">args?.</span><span style="color: #94a3b8">enableCPUTiming</span>) {
  <span style="color: #a5b4fc">await</span> <span style="color: #a5b4fc">this</span><span style="color: #e2e8f0">.device.queue.</span><span style="color: #34d399">onSubmittedWorkDone</span>();
  <span style="color: #a5b4fc">this</span><span style="color: #e2e8f0">.cpuEndTime = performance.</span><span style="color: #34d399">now</span>();
}</code></pre>
  </div>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin: 3rem 0 2rem 0; padding-top: 2rem; border-top: 1px solid #e5e7eb;">Returning Timing Information</h2>

<div style="max-width: 800px; margin-bottom: 3rem;">
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    The primitive object has an <code>async getTimingResult</code> member function that returns the CPU and GPU timing result:
  </p>

  <div style="background: #1f2937; border-radius: 0.75rem; overflow: hidden; margin-bottom: 1.5rem;">
    <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: #374151; border-bottom: 1px solid #4b5563;">
      <div style="display: flex; gap: 0.375rem;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
      </div>
      <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.75rem; color: #9ca3af; margin-left: 1rem;">get-timing.js</span>
    </div>
    <pre style="padding: 1.5rem; margin: 0; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.7; overflow-x: auto;"><code><span style="color: #a5b4fc">const</span> { <span style="color: #94a3b8">gpuTotalTimeNS</span>, <span style="color: #94a3b8">cpuTotalTimeNS</span> } = <span style="color: #a5b4fc">await</span> <span style="color: #e2e8f0">primitive.</span><span style="color: #34d399">getTimingResult</span>();</code></pre>
  </div>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    <strong>GPU Timing:</strong> Returns a <strong>list of total elapsed times per kernel</strong> in nanoseconds (ns).
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    <strong>CPU Timing:</strong> Returns an <strong>elapsed time for the entire primitive</strong> in nanoseconds (ns).
  </p>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem;">
    <p style="margin: 0; color: #065f46; line-height: 1.75;">
      <strong>Important:</strong> Neither timing method accounts for <code>trials</code>, so the caller of <code>getTimingResult</code> should <strong>divide any returned values by the number of trials</strong>.
    </p>
  </div>
</div>

</div>
