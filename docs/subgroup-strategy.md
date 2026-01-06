---
layout: page
title: "Gridwise WebGPU Subgroup Emulation Strategy"
category: gridwise
permalink: /subgroup-strategy/
excerpt: "Strategies for writing high-performance code that uses WGSL subgroups when available and falls back to emulation when not."
order: 8
---

<div style="max-width: 100%; padding: 0;">

<header style="max-width: 100%; margin-bottom: 3rem;">
  <h1 style="font-size: 2.5rem; font-weight: 800; color: #111827; margin-bottom: 1.5rem; line-height: 1.2;">Subgroup Emulation Strategy</h1>
  <p style="font-size: 1.25rem; color: #6b7280; line-height: 1.75;">
    WGSL's subgroup built-in functions have the potential to significantly improve performance on subgroup-capable hardware. A large (and growing) fraction of WebGPU devices support subgroups. However, writing high-performance code that supports both using subgroups when available and falls back to code with no subgroups is a challenge. This document describes my experience attempting to do exactly that.
  </p>
</header>

<div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;">
  <p style="margin: 0; color: #065f46; line-height: 1.75;">
    In the below discussion, we use the term <strong>"hw"</strong> to indicate "the development experience that targets hardware-supported subgroups" and <strong>"emu"</strong> for the development experience that targets non-hardware-supported subgroups, which are not supported by WGSL and thus must be emulated".
  </p>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin: 3rem 0 2rem 0; padding-top: 2rem; border-top: 1px solid #e5e7eb;">Goals of this Effort</h2>

<div style="max-width: 100%; margin-bottom: 3rem;">
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    In an ideal world, this effort would have resulted in the following outcomes:
  </p>
  
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    <strong>For users of primitives:</strong> No code changes. The same code runs on both subgroup-capable (hw) and non-subgroup-capable (emu) hardware. However, my effort did not prioritize performance for emu hardware. To prioritize performance here, it is likely that we'd need separate primitive formulations for hw/emu scenarios.
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    <strong>For developers of primitives:</strong> Minimal additional complexity. This was not entirely achieved.
  </p>

  <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 1.5rem; border-radius: 0.5rem;">
    <p style="color: #78350f; line-height: 1.75; margin: 0;">
      <strong>Note:</strong> The checked-in code does not work out of the box for emu. It would not be excessively difficult to make it work, but it would take several hours of grungy effort. I priorized hw development, not emu development, during our scan implementation; scan is already quite complicated and keeping it working for both during active development was just not a priority.
    </p>
  </div>
</div>

<!-- Initial Goal Section -->
<h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin: 3rem 0 2rem 0; padding-top: 2rem; border-top: 1px solid #e5e7eb;">Initial goal: subgroupShuffle</h2>

<div style="max-width: 100%; margin-bottom: 3rem;">
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    Let's begin by making one call, <code>subgroupShuffle</code>, work in both hw and emu contexts. This call looks like:
  </p>

  <div class="code-box-simple">
    <div class="code-box-header">WGSL Call</div>
    <div class="code-box-content">
      <pre><code><span class="property">y</span> = <span class="function">subgroupShuffle</span>(<span class="property">x</span>, <span class="property">id</span>);</code></pre>
    </div>
  </div>

  <p style="color: #4b5563; line-height: 1.75; margin: 1.5rem 0;">
    The first roadblock is that <code>subgroupShuffle</code> is not defined as an emu function. Fortunately (and intelligently), WGSL allows the programmer to directly override it ("shadow") with a user-specified function. So let's do that (for the emu context only). If we lack subgroup hardware, we have to communicate shuffled values through workgroup memory, so we have to declare a region of workgroup memory with one element per thread.
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    Here, <code>source</code> is a thread index within the subgroup, and <code>sgid</code> is the <code>@builtin(subgroup_invocation_id)</code>. The code is straightforward:
  </p>

  <div class="code-box-simple">
    <div class="code-box-header">Emulation Implementation</div>
    <div class="code-box-content">
      <pre><code><span class="keyword">var</span>&lt;<span class="builtin">workgroup</span>&gt; <span class="property">wg_sw_subgroups</span>: <span class="type">array</span>&lt;<span class="type">u32</span>, <span class="number">256</span>&gt;;

<span class="keyword">fn</span> <span class="function">subgroupShuffle</span>(<span class="property">x</span>: <span class="type">u32</span>, <span class="property">source</span>: <span class="type">u32</span>) -&gt; <span class="type">u32</span> {
  <span class="property">wg_sw_subgroups</span>[<span class="property">sgid</span>] = <span class="property">x</span>;
  <span class="function">workgroupBarrier</span>();
  <span class="keyword">return</span> <span class="property">wg_sw_subgroups</span>[<span class="property">source</span>];
}</code></pre>
    </div>
  </div>

  <p style="color: #4b5563; line-height: 1.75; margin-top: 1.5rem;">
    Well, not so much. There are many challenges ahead of us.
  </p>
</div>

<!-- Challenges Section -->
<h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin: 3rem 0 2rem 0; padding-top: 2rem; border-top: 1px solid #e5e7eb;">Implementation Challenges</h2>

<!-- Challenge 1: @builtin sgid -->
<section style="max-width: 100%; margin-bottom: 3rem;">
  <h3 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Challenge 1: the @builtin sgid</h3>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    Our first challenge is using <code>@builtin(subgroup_invocation_id) sgid</code>. In emu, this <code>@builtin</code> is not defined. We can pass it in as an argument, however.
  </p>
  <p style="color: #4b5563; line-height: 1.75;">
    Thus one possible solution is to define <code>fn subgroupShuffleWrapper(x, source, sgid)</code> and then use <code>subgroupShuffleWrapper</code> everywhere. We began our development using this strategy, but it is undesirable; it's not reasonable to ask every possible developer within this library to use a set of different functions with different APIs than those in the spec, and it significantly complicated development. We needed a better way, which we address as part of the next challenge.
  </p>
</section>

<!-- Challenge 2: Supporting both hw and emu -->
<section style="max-width: 100%; margin-bottom: 3rem;">
  <h3 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Challenge 2: supporting both hw and emu with minimal impact on the programmer</h3>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    Our second challenge is ensuring that our <code>fn subgroupShuffle</code> definition is only visible when the kernel is compiled in emu, but not in hw. How can we do this?
  </p>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    First, the WebGPU call <code>device.features.has("subgroups")</code> tells us if subgroups are supported. We can use the result of this call to declare one of two sets of functions: one that assumes subgroups are available (hw) and one that does not (emu). In our implementation, this set of functions is called <code>fnDeclarations</code>. Our syntax is not important here; what is important is what happens in hw and what happens in emu.
  </p>

  <h4 style="font-size: 1.125rem; font-weight: 700; color: #111827; margin: 2rem 0 1rem 0;">Step 1: Enable Subgroups</h4>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    At the top of the kernel, we require <code>${this.fnDeclarations.enableSubgroupsIfAppropriate}</code>. If we are hw, this emits <code>enable subgroups;</code>; if we are emu, this emits nothing.
  </p>

  <h4 style="font-size: 1.125rem; font-weight: 700; color: #111827; margin: 2rem 0 1rem 0;">Step 2: Declare Subgroup Emulation Variables</h4>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    With our next declaration, we partially solve the problem we identified above, where the subgroup size and subgroup id builtins are available in hw but not in emu. If the kernel is using any subgroup calls, we require <code>${this.fnDeclarations.subgroupEmulation}</code>. In hw, this emits nothing. In emu, this declares workgroup memory (for performing the subgroup operations) and subgroup variables (subgroup size and subgroup id), all at module scope:
  </p>

  <div class="code-box-simple">
    <div class="code-box-content">
      <pre><code><span class="keyword">var</span>&lt;<span class="builtin">workgroup</span>&gt; <span class="property">wg_sw_subgroups</span>: <span class="type">array</span>&lt;${<span class="property">env.datatype</span>}, ${<span class="property">env.workgroupSize</span>}&gt;;
<span class="keyword">const</span> <span class="property">sgsz</span>: <span class="type">u32</span> = ${<span class="property">env.workgroupSize</span>};
<span class="keyword">var</span>&lt;<span class="builtin">private</span>&gt; <span class="property">sgid</span>: <span class="type">u32</span>;</code></pre>
    </div>
  </div>

  <p style="color: #4b5563; line-height: 1.75; margin: 1rem 0 1.5rem 0;">
    However, it does not actually assign values to <code>sgsz</code> and <code>sgid</code>.
  </p>

  <h4 style="font-size: 1.125rem; font-weight: 700; color: #111827; margin: 2rem 0 1rem 0;">Step 3: Declare Subgroup Functions</h4>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    Next, for each subgroup call we want to make, we "declare" the call. For subgroup shuffle, in hw, we emit nothing, because <code>subgroupShuffle</code> is builtin. In emu (note we use the <code>sgid</code> variable we declared at module scope above):
  </p>

  <div class="code-box-simple">
    <div class="code-box-content">
      <pre><code><span class="keyword">fn</span> <span class="function">subgroupShuffle</span>(<span class="property">x</span>: <span class="type">u32</span>, <span class="property">source</span>: <span class="type">u32</span>) -&gt; <span class="type">u32</span> {
  <span class="comment">/* subgroup emulation must pass through wg_sw_subgroups */</span>
  <span class="comment">/* write my value to workgroup memory */</span>
  <span class="property">wg_sw_subgroups</span>[<span class="property">sgid</span>] = <span class="function">bitcast</span>&lt;${<span class="property">env.datatype</span>}&gt;(<span class="property">x</span>);
  <span class="function">workgroupBarrier</span>();
  <span class="keyword">var</span> <span class="property">shuffled</span>: <span class="type">u32</span> = <span class="function">bitcast</span>&lt;<span class="type">u32</span>&gt;(<span class="property">wg_sw_subgroups</span>[<span class="property">source</span>]);
  <span class="function">workgroupBarrier</span>();
  <span class="keyword">return</span> <span class="property">shuffled</span>;
}</code></pre>
    </div>
  </div>

  <p style="color: #4b5563; line-height: 1.75; margin: 1rem 0 1.5rem 0;">
    In hw, each supported subgroup call emits nothing, but we also define other useful functions that are not already defined and emit different implementations for hw and emu. (Example: WGSL supports both inclusive (<code>subgroupInclusiveAdd</code>) and exclusive subgroup (<code>subgroupExclusiveAdd</code>) scans, but only if the scan operator is addition. Our function library has support for non-addition inclusive and exclusive subgroup scans for both hw and emu.)
  </p>

  <h4 style="font-size: 1.125rem; font-weight: 700; color: #111827; margin: 2rem 0 1rem 0;">Step 4: Initialize Subgroup Variables</h4>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    Finally, we need to assign values to <code>sgsz</code> and <code>sgid</code> to functions wehre they are used. Here we use a declaration within each function, <code>${this.fnDeclarations.initializeSubgroupVars()}</code>. For hw, this does nothing. For emu, this emits <code>let sgsz: u32 = builtinsUniform.sgsz;\nlet sgid: u32 = builtinsNonuniform.sgid;</code>.
  </p>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
    <p style="color: #065f46; line-height: 1.75; margin: 0;">
      The burden on the programmer is to (1) declare necessary functions at the top of the module and (2) initialize subgroup variables at the top of each function that uses subgroups, but not to change kernel code.
    </p>
  </div>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    For a hypothetical module/kernel whose only subgroup operation is <code>subgroupShuffle</code>, that code looks like:
  </p>

  <div class="code-box-simple">
    <div class="code-box-header">complete-example.wgsl</div>
    <div class="code-box-content">
      <pre><code><span class="comment">// must be first line of kernel</span>
${<span class="property">this.fnDeclarations.enableSubgroupsIfAppropriate</span>}
${<span class="property">this.fnDeclarations.subgroupEmulation</span>}
${<span class="property">this.fnDeclarations.subgroupShuffle</span>}

<span class="keyword">fn</span> <span class="function">kernel</span>() {
  ${<span class="property">this.fnDeclarations.initializeSubgroupVars</span>()}
  <span class="comment">// ...</span>
}</code></pre>
    </div>
  </div>
</section>

<!-- Challenge 3: Emulated Subgroup Size -->
<section style="max-width: 100%; margin-bottom: 3rem;">
  <h3 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Challenge 3: choosing an emulated subgroup size</h3>
  
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    Finally, we will have to write each emu subgroup operation. Our third challenge is to choose a subgroup size to emulate.
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    First, we know that using hw subgroup operations will deliver better performance than emu, for several reasons:
  </p>
  <ol style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem; padding-left: 1.5rem;">
    <li style="margin-bottom: 0.5rem;">Hardware-supported subgroup instructions will run faster than the sequences of instructions we need to emulate them</li>
    <li style="margin-bottom: 0.5rem;">Because emulated subgroups don't run in lockstep, we will require more workgroup barriers to emulate subgroups (workgroup barriers will have the largest impact in latency-sensitive and/or large-workgroup kernels)</li>
    <li style="margin-bottom: 0.5rem;">Emulated subgroup instructions need to run through workgroup memory, which is slower than registers</li>
    <li>Allocating additional workgroup memory (at least one word per thread of the workgroup) might decrease the number of subgroups that can fit on a processor, hurting occupancy</li>
  </ol>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    Recall that WebGPU does not specify a subgroup size (in hw), although it does specify a minimum and maximum subgroup size. (In fact, some WebGPU-capable hardware may use different subgroup sizes across different kernels in the same application.) WebGPU developers must thus write their code assuming any subgroup size between the minimum and the maximum. Since our kernels already have to handle a range of subgroup sizes, we have some flexibility to choose a subgroup size in emu. We have 3 main choices:
  </p>

  <ol style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem; padding-left: 1.5rem;">
    <li style="margin-bottom: 0.75rem;">Assume that very small subgroup sizes run in lockstep, and use those subgroup sizes. Then we can potentially avoid some barriers and gain some efficiency. <em>But:</em> We can't assume that. WebGPU does not report that information. (If it did, we could take advantage of it.)</li>
    <li style="margin-bottom: 0.75rem;">Assume a comfortable subgroup size (e.g., 32), and add appropriate barriers.</li>
    <li style="font-weight: 600;">Since we're going to have to put barriers everywhere anyway, assume subgroup size == workgroup size.</li>
  </ol>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    Let's take a step back and think about how subgroups are used. Consider a reduction across a workgroup (each thread has one item and the workgroup adds up all items). The typical pattern for a workgroup reduction leveraging subgroup support is to (a) use a subgroup reduction on each subgroup then (b) reduce across the results from each subgroup. This pattern is typical: parallelize across subgroups, then combine the results.
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    Now, if we choose alternatives 1 or 2, then it is highly likely that each workgroup will contain several subgroups. Many primitives will thus have two stages: per-subgroup, then per-workgroup. If we choose alternative 3 (subgroup size == workgroup size), then our algorithms may be simpler, because we don't have to combine results from multiple subgroups within a workgroup. This also simplifies the code that emulates subgroups.
  </p>

  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    We do see one clear structural issue, though: some subgroup operations have a maximum size (e.g., <code>subgroupBallot</code> has a maximum subgroup size of 128, because it returns exactly 128 bits).
  </p>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem;">
    <p style="color: #065f46; line-height: 1.75; margin: 0;">
      Nonetheless for simplicity, we currently choose to always emulate subgroups that are the size of the workgroup, recognizing that this is not a fully generalizable solution.
    </p>
  </div>
</section>

<!-- Summary Section -->
<h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin: 3rem 0 2rem 0; padding-top: 2rem; border-top: 1px solid #e5e7eb;">Summary</h2>

<div style="max-width: 100%; margin-bottom: 3rem;">
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    On an Apple M3 with a high-performance scan kernel, the performance difference between hw and emu with the same kernel is ~2.5x.
  </p>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem;">
    <h3 style="font-size: 1.125rem; font-weight: 700; color: #065f46; margin-bottom: 0.5rem;">Open Question</h3>
    <p style="color: #065f46; line-height: 1.75; margin: 0;">
      An open question is whether it is better to write different <em>kernels</em> for hw and emu as opposed to what we did: writing different versions of subgroup functions and keeping the same kernel. The answer probably depends on the nature of the kernels. We did not explore the latter alternative at all.
    </p>
  </div>
</div>

</div>

<!-- Include the simple code box component -->
{% include code-box-simple.html %}
