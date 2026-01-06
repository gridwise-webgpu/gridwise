---
layout: page
title: "Scan and Reduce"
category: gridwise
permalink: /scan-and-reduce/
excerpt: "Implementation details of scan (parallel prefix) and reduce operations in Gridwise, including our SPAA 2025 paper findings."
order: 4
---

<div style="max-width: 100%; padding: 0;">

<header style="max-width: 100%; margin-bottom: 3rem;">
  <h1 style="font-size: 2.5rem; font-weight: 800; color: #111827; margin-bottom: 1.5rem; line-height: 1.2;">Scan and Reduce</h1>
  <p style="font-size: 1.25rem; color: #6b7280; line-height: 1.75;">
    <a href="https://en.wikipedia.org/wiki/Prefix_sum" style="color: #10b981; text-decoration: underline;">Scan</a> (parallel prefix) is a fundamental parallel compute primitive useful in both other primitives as well as a wide range of application domains. Here we describe how our implementation works and how it is used in Gridwise. We published a <a href="https://dl.acm.org/doi/10.1145/3694906.3743326" style="color: #10b981; text-decoration: underline;">paper on our scan implementation</a> at SPAA 2025.
  </p>
</header>

<!-- Section 1: Terminology -->
<section style="max-width: 100%; margin-bottom: 3rem;">
  <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 1.5rem;">Terminology</h2>
  
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    <strong>Scan:</strong> Scan inputs an array of <em>n</em> data elements and outputs an array of the same size. Output element <em>i</em> is the "sum" of the input elements up to element <em>i</em>. More generally, that "sum" operator can be any <a href="https://en.wikipedia.org/wiki/Monoid" style="color: #10b981; text-decoration: underline;">monoid</a> (a binary operation with an identity element). 
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    If the operator is addition, scan is often called prefix-sum, but we can also compute a prefix-multiplication, prefix-max or -min, or many other operators. For simplicity, we will use "sum" and addition in this article. <strong>(Gridwise supports any user-specified monoid.)</strong>
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    <strong>Reduce:</strong> A reduction of a set of inputs is the "sum" of all of those elements.
  </p>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem;">
    <h3 style="font-size: 1.125rem; font-weight: 700; color: #065f46; margin-bottom: 1rem;">Scan Variants</h3>
    <p style="color: #065f46; line-height: 1.75; margin-bottom: 0.75rem;">
      <strong>Exclusive Scan:</strong> Each output element is the sum of all previous items in the input, <strong>not including</strong> the current item.<br/>
      <code>exclusive_out[i] = sum(in[0:i-1])</code>
    </p>
    <p style="color: #065f46; line-height: 1.75; margin: 0;">
      <strong>Inclusive Scan:</strong> Each output element is the sum of all previous items in the input, <strong>up to and including</strong> the current item.<br/>
      <code>inclusive_out[i] = sum(in[0:i])</code>
    </p>
  </div>
</section>

<!-- Section 2: GPU Scan Background -->
<section style="max-width: 100%; padding-top: 2rem; border-top: 1px solid #e5e7eb; margin-bottom: 3rem;">
  <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 2rem;">GPU Scan Background</h2>
  
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    On GPUs, scan performance is bound by memory bandwidth. The best GPU scan implementations fully saturate the GPU's memory system. Thus the best GPU scan implementations are those that use algorithms that require the fewest accesses to memory.
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    <strong>Reduce-Then-Scan (Classic):</strong> Divide the input into tiles, compute the reduction of all elements in each tile, compute the prefix sum of those per-tile reductions, then use the values of that per-tile prefix sum as inputs into a blockwise scan of each individual tile. For an <em>n</em>-element scan, this incurs <strong>3n memory accesses</strong> (<em>n</em> for the per-block reduction and 2<em>n</em> to read the input/write the output in the blockwise scan).
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    <strong>StreamScan / Chained Scan (2015):</strong> In 2015, Yan et al. introduced <a href="https://dl.acm.org/doi/10.1145/2442516.2442539" style="color: #10b981; text-decoration: underline;">StreamScan</a>, which required only <strong>2n memory accesses</strong> (the theoretical minimum). This approach <em>serializes</em> the scan of the per-tile values across tile processors. The key data structure is the <strong>carry chain</strong>, which stores the inclusive scan of the tile reductions. It only requires reading and writing each element once.
  </p>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;">
    <p style="color: #065f46; line-height: 1.75; margin: 0;">
      <strong>Decoupled Lookback (2016):</strong> In 2016, <a href="https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back" style="color: #065f46; text-decoration: underline; font-weight: 600;">Merrill and Garland</a> addressed performance loss on NVIDIA hardware by enabling stalled tiles to "look back" into the carry chain to fetch the necessary values. This ensured that a single stalled tile did not halt the entire scan computation, allowing <a href="https://docs.nvidia.com/cuda/cub/index.html" style="color: #065f46; text-decoration: underline;">their implementation</a> to run at maximum throughput. <strong>CUB scan is as fast as a memory copy.</strong>
    </p>
  </div>

  <p style="color: #6b7280; font-size: 0.875rem; font-style: italic; margin: 0;">
    (All of the above is covered in great detail in our <a href="https://dl.acm.org/doi/10.1145/3694906.3743326" style="color: #10b981; text-decoration: underline;">SPAA 2025 paper</a>.)
  </p>
</section>

<!-- Section 3: Forward Progress Guarantees -->
<section style="max-width: 100%; padding-top: 2rem; border-top: 1px solid #e5e7eb; margin-bottom: 3rem;">
  <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 2rem;">Decoupled Lookback and Forward Progress Guarantees</h2>
  
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    <strong>The Problem:</strong> NVIDIA GPUs make a <em>forward progress guarantee</em>: once a processor begins to process a tile, it is guaranteed to make progress on that tile. This guarantee is necessary for the correctness of Merrill and Garland's implementation of decoupled lookback.
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    Unfortunately, not all GPUs provide this forward progress guarantee. <a href="https://dl.acm.org/doi/10.1145/3485508" style="color: #10b981; text-decoration: underline;">Apple and ARM GPUs do not.</a>
  </p>

  <div style="background: #fef2f2; border-left: 4px solid #ef4444; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
    <p style="color: #991b1b; line-height: 1.75; margin: 0;">
      <strong>⚠️ Critical Issue:</strong> Without the forward-progress guarantee, computing tiles may be fully blocked by waiting tiles and never make progress, leading to a completely stalled computation. <strong>On Apple hardware, as we found during the development of our scan primitive, this locks up the entire machine.</strong>
    </p>
  </div>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem;">
    <p style="color: #065f46; line-height: 1.75; margin: 0; font-weight: 600;">
      Deploying a WebGPU implementation that depends on a forward-progress guarantee is thus not a viable option.
    </p>
  </div>
</section>

<!-- Section 4: Gridwise's Implementation -->
<section style="max-width: 100%; padding-top: 2rem; border-top: 1px solid #e5e7eb; margin-bottom: 3rem;">
  <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 2rem;">Gridwise's Scan and Reduce Implementations</h2>
  
  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
    <h3 style="font-size: 1.25rem; font-weight: 700; color: #065f46; margin-bottom: 0.75rem;">Gridwise's Scan</h3>
    <p style="color: #065f46; line-height: 1.75; margin-bottom: 0.75rem;">
      Gridwise implements a chained scan that <strong>does not require forward-progress guarantees</strong>. It does so in a high-performance way that allows scan to run at full memory bandwidth, even on hardware without forward-progress guarantees.
    </p>
    <p style="color: #065f46; line-height: 1.75; margin: 0;">
      Merrill and Garland's <strong>lookback strategy</strong> allows a stalled tile processor to look back into the carry chain; we add <strong>fallback capability</strong> to allow a stalled tile processor to redundantly compute per-tile reductions, and to do so using the full parallelism of the stalled tile processor. <em>Lookback and fallback were challenging to implement correctly.</em>
    </p>
  </div>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    <strong>Gridwise's Reduce:</strong> The carry chain in a chained scan stores the inclusive scan of the reduction of each tile. The last element in that carry chain is the sum of all input tiles and is thus the reduction of the entire input. A reduce can thus be implemented using the existing scan machinery (and run at full memory bandwidth). When configured as a reduction primitive, our scan implementation leaves out any scan-specific computation.
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 0.5rem;">
    <strong>Leveraging Subgroups:</strong> WebGPU's optional <a href="https://www.w3.org/TR/webgpu/#subgroups" style="color: #10b981; text-decoration: underline;">subgroups</a> feature enables WebGPU programs to use SIMD instructions within a workgroup. These can deliver significant performance gains. Gridwise has some, but incompletely deployed, <a href="subgroup-strategy.html" style="color: #10b981; text-decoration: underline;">support for emulating SIMD instructions</a>.
  </p>

  <p style="color: #4b5563; line-height: 1.75; font-size: 0.875rem; font-style: italic;">
    Our initial performance testing indicated that scan was 2.5x slower using subgroup emulation vs. using subgroup hardware.
  </p>
</section>

</div>

<!-- Section 5: Configuration and Usage -->
<section style="max-width: 100%; padding-top: 2rem; border-top: 1px solid #e5e7eb; margin-bottom: 3rem;">
  <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 2rem;">Configuring and Calling Gridwise Scan and Reduce</h2>
    
    <!-- Defining the Primitive -->
    <div style="margin-bottom: 3rem;">
      <h3 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1.5rem;">Defining the Primitive</h3>
      <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
        Declare the scan or reduce primitive as an instance of the <code>DLDFScan</code> class. An example scan declaration:
      </p>

      <div style="background: #1f2937; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: #374151; border-bottom: 1px solid #4b5563;">
          <div style="display: flex; gap: 0.375rem;">
            <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
            <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
            <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
          </div>
          <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.75rem; color: #9ca3af; margin-left: 1rem;">define-scan.js</span>
        </div>
        <pre style="padding: 1.5rem; margin: 0; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.7; overflow-x: auto;"><code><span style="color: #a5b4fc">const</span> <span style="color: #e2e8f0">datatype = </span><span style="color: #34d399">"u32"</span>; <span style="color: #6b7280">// or "i32" or "f32"</span>
<span style="color: #a5b4fc">const</span> <span style="color: #e2e8f0">dldfscanPrimitive = </span><span style="color: #a5b4fc">new</span> <span style="color: #fbbf24">DLDFScan</span>({
  <span style="color: #94a3b8">device,</span>
  <span style="color: #e2e8f0">binop: </span><span style="color: #a5b4fc">new</span> <span style="color: #fbbf24">BinOpAdd</span>({ datatype }),
  <span style="color: #e2e8f0">type: </span><span style="color: #34d399">"exclusive"</span>,
  <span style="color: #94a3b8">datatype,</span> <span style="color: #6b7280">// use the "datatype" string defined above</span>
  <span style="color: #e2e8f0">gputimestamps: </span><span style="color: #fbbf24">true</span>,
});</code></pre>
      </div>

      <h4 style="font-size: 1.125rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Constructor Arguments</h4>
      
      <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1.5rem;">
        <p style="color: #065f46; margin-bottom: 1rem;"><strong>Required parameters:</strong></p>
        <p style="margin: 0 0 0.75rem 0;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">device</code> – The GPU device on which this primitive will run.</p>
        <p style="margin: 0 0 0.75rem 0;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">datatype</code> – Currently we support the (WGSL) types <code>"u32"</code>, <code>"i32"</code>, and <code>"f32"</code>, specified as strings. It is definitely possible to support more complex datatypes (e.g., structs), but this would take non-trivial engineering work.</p>
        <p style="margin: 0;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">binop</code> – The "binary operation" aka a <a href="https://en.wikipedia.org/wiki/Monoid" style="color: #065f46; text-decoration: underline;">monoid</a>, specified as a combination of a datatype and a binary operator that operates on that datatype. The core scan implementation is agnostic as to the binary operation; the <code>binop</code> supplies that operation. The binop class is described in more detail <a href="binop.html" style="color: #065f46; text-decoration: underline;">here</a>.</p>
      </div>
      
      <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1.5rem;">
        <p style="color: #065f46; margin-bottom: 1rem;"><strong>Optional parameters:</strong></p>
        <p style="margin: 0 0 0.75rem 0;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">type</code> – Any of <code>"exclusive"</code>, <code>"inclusive"</code>, or <code>"reduce"</code>. Default is <code>"exclusive"</code>.</p>
        <p style="margin: 0;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">gputimestamps</code> – Enable GPU timestamps for the primitive's kernel calls.</p>
      </div>
    </div>

    <!-- Configuring the Primitive -->
    <div style="margin-bottom: 3rem;">
      <h3 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1.5rem;">Configuring the Primitive</h3>
      <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
        Once the primitive is <em>defined</em>, it must then be <em>configured</em>. The primitive knows that it requires an input and output buffer, named <code>inputBuffer</code> and <code>outputBuffer</code>. (We use our <a href="buffer.html" style="color: #4f46e5; text-decoration: underline;"><code>Buffer</code> class</a> for this.) We configure the primitive by registering data buffers with the primitive.
      </p>
      <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
        This can be done either with a <code>primitive.registerBuffer()</code> call or as an argument to the <code>execute</code> call. (The former is preferred if we need to register the buffer(s) once and then call <code>execute</code> many times.)
      </p>

      <div style="background: #1f2937; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: #374151; border-bottom: 1px solid #4b5563;">
          <div style="display: flex; gap: 0.375rem;">
            <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
            <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
            <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
          </div>
          <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.75rem; color: #9ca3af; margin-left: 1rem;">configure-scan.js</span>
        </div>
        <pre style="padding: 1.5rem; margin: 0; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.7; overflow-x: auto;"><code><span style="color: #a5b4fc">const</span> <span style="color: #e2e8f0">inputLength = </span><span style="color: #fbbf24">2</span> ** <span style="color: #fbbf24">20</span>;
<span style="color: #e2e8f0">testInputBuffer = </span><span style="color: #a5b4fc">new</span> <span style="color: #fbbf24">Buffer</span>({
  <span style="color: #94a3b8">device,</span>
  <span style="color: #e2e8f0">datatype: </span><span style="color: #34d399">"f32"</span>,
  <span style="color: #94a3b8">inputLength,</span>
  <span style="color: #e2e8f0">label: </span><span style="color: #34d399">"inputBuffer"</span>,
  <span style="color: #e2e8f0">createCPUBuffer: </span><span style="color: #fbbf24">true</span>,
  <span style="color: #e2e8f0">initializeCPUBuffer: </span><span style="color: #fbbf24">true</span> <span style="color: #6b7280">/* fill with default data */</span>,
  <span style="color: #e2e8f0">createGPUBuffer: </span><span style="color: #fbbf24">true</span>,
  <span style="color: #e2e8f0">initializeGPUBuffer: </span><span style="color: #fbbf24">true</span> <span style="color: #6b7280">/* with CPU data */</span>,
  <span style="color: #e2e8f0">createMappableGPUBuffer: </span><span style="color: #fbbf24">false</span>, <span style="color: #6b7280">/* never reading this back */</span>
});
dldfscanPrimitive.<span style="color: #38bdf8">registerBuffer</span>(testInputBuffer);</code></pre>
      </div>
    </div>

    <!-- Calling the Primitive -->
    <div style="margin-bottom: 3rem;">
      <h3 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1.5rem;">Calling Scan or Reduce</h3>
      <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
        Once the primitive is defined and configured, simply call its <code>execute()</code> method.
      </p>
      <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
        If you have not yet registered buffers, you can specify them in the argument object as <code>inputBuffer</code> and <code>outputBuffer</code>.
      </p>

      <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1.5rem;">
        <h4 style="font-size: 1rem; font-weight: 700; color: #065f46; margin-bottom: 0.75rem;">Optional Timing Arguments</h4>
        <p style="color: #065f46; margin-bottom: 0.75rem;">
          These are timing-specific and thus which you are unlikely to use unless you are benchmarking:
        </p>
        <ul style="color: #065f46; line-height: 1.6; margin-left: 1.5rem;">
          <li style="margin-bottom: 0.25rem;"><code>trials</code> with an integer argument. This will run the kernel(s) that number of times. Default: 1.</li>
          <li style="margin-bottom: 0.25rem;"><code>enableGPUTiming</code> with either true or false. If true, please ensure that the device has a set of required features that include <code>timestamp-query</code>. Default: false.</li>
          <li><code>enableCPUTiming</code> with either true or false. Default: false.</li>
        </ul>
        <p style="font-weight: 600; margin: 0.75rem 0 0 0;">
          Note that <code>execute()</code> is declared <code>async</code>.
        </p>
      </div>

      <div style="background: #1f2937; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: #374151; border-bottom: 1px solid #4b5563;">
          <div style="display: flex; gap: 0.375rem;">
            <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
            <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
            <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
          </div>
          <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.75rem; color: #9ca3af; margin-left: 1rem;">execute-scan.js</span>
        </div>
        <pre style="padding: 1.5rem; margin: 0; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.7; overflow-x: auto;"><code><span style="color: #a5b4fc">await</span> dldfscanPrimitive.<span style="color: #38bdf8">execute</span>();

<span style="color: #6b7280">// or, if we want to specify buffers only when execute is called</span>
<span style="color: #a5b4fc">await</span> dldfscanPrimitive.<span style="color: #38bdf8">execute</span>({
  <span style="color: #e2e8f0">inputBuffer: </span>mySrcBuffer,
  <span style="color: #e2e8f0">outputBuffer: </span>myDestBuffer,
});

<span style="color: #6b7280">// or (maybe if you're benchmarking)</span>
<span style="color: #a5b4fc">await</span> dldfscanPrimitive.<span style="color: #38bdf8">execute</span>({
  <span style="color: #e2e8f0">trials: </span><span style="color: #fbbf24">1</span>,
  <span style="color: #e2e8f0">enableGPUTiming: </span><span style="color: #fbbf24">false</span>,
  <span style="color: #e2e8f0">enableCPUTiming: </span><span style="color: #fbbf24">true</span>,
});</code></pre>
      </div>
    </div>
  </div>
</section>

<!-- Section 6: Usage Notes -->
<section style="max-width: 100%; padding-top: 2rem; border-top: 1px solid #e5e7eb; margin-bottom: 3rem;">
  <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 2rem;">Usage and Performance Notes</h2>
  
  <div style="padding: 1.5rem; border: 2px solid #ef4444; border-radius: 0.5rem; background: #fef2f2; margin-bottom: 2rem;">
    <h3 style="font-size: 1.125rem; font-weight: 700; color: #991b1b; margin-bottom: 0.75rem;">⚠️ Input Length Requirement</h3>
    <p style="color: #991b1b; line-height: 1.6; margin: 0;">
      Input lengths <strong>must be</strong> a multiple of 4. Pad the end of your input array with enough identity elements to make this work. (This is because internally, we use <code>vec4</code>s for computation.)
    </p>
  </div>

  <p style="margin-bottom: 1rem;"><strong>Performance Testing:</strong> Scan has had extensive performance testing and the defaults are fairly stable across different GPUs. The workgroup size, for instance, is set to 256. This particular iteration of the scan kernel has barely been tested with other workgroup sizes and they are unlikely to work out of the box.</p>

  <p style="margin-bottom: 2rem;"><strong>Future Work:</strong> If we extended scan to larger datatypes (beyond 32 bits), we expect that workgroup memory consumption would become an issue. We expect we would have to reduce workgroup size accordingly.</p>
</section>

</div>