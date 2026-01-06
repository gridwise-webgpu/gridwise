---
layout: page
title: "Binary Operator Class"
category: gridwise
permalink: /binop/
excerpt: "Understanding the BinOp class that represents monoids with binary operations, datatypes, and identity elements for use in primitives."
order: 3
---

<div style="max-width: 100%; padding: 0;">

<!-- Header Section -->
<header style="max-width: 800px; margin-bottom: 3rem;">
  <div style="display: flex; align-items: center; gap: 0.5rem; color: #4f46e5; font-weight: 600; font-size: 0.875rem; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em;">
    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
    </svg>
    <span>Core Concepts</span>
  </div>
  <h1 style="font-size: 2.5rem; font-weight: 800; color: #111827; margin-bottom: 1.5rem; line-height: 1.2;">Binary Operator Class</h1>
  <p style="font-size: 1.25rem; color: #6b7280; line-height: 1.75;">
    Gridwise's binary operator class represents a <a href="https://en.wikipedia.org/wiki/Monoid" style="color: #4f46e5; text-decoration: underline;">monoid</a>, which has as its constituent parts a binary operation, a datatype for the data on which the operator is applied, and an identity element. In Gridwise, we package these elements into an instance of a JS class, <code>BinOp</code>.
  </p>
</header>

<!-- Section 1: Understanding BinOp (Split View) -->
<section style="display: grid; grid-template-columns: 1fr 1fr; gap: 3rem; margin-bottom: 4rem; align-items: start;">
  <div>
    <h2 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Understanding BinOp</h2>
    <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
      The <code>BinOp</code> class represents a monoid with:
    </p>
    <ul style="color: #4b5563; line-height: 1.75; margin-left: 1.5rem; margin-bottom: 1rem;">
      <li style="margin-bottom: 0.5rem;"><strong>Binary operation</strong> — the operator itself</li>
      <li style="margin-bottom: 0.5rem;"><strong>Datatype</strong> — the type of data the operator works on</li>
      <li style="margin-bottom: 0.5rem;"><strong>Identity element</strong> — where <code>x = I op x</code></li>
    </ul>
    <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
      For instance, addition's identity is zero (since <code>x + 0 = x</code>), and multiplication's identity is one (since <code>x × 1 = x</code>).
    </p>
    <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
      <code>BinOp</code> is implemented in the source file <code>binop.mjs</code>. We specialize <code>BinOp</code> to particular operations (e.g., <code>Add</code>) and then further specialize it with a datatype. This class defines objects used in WGSL code generation and CPU correctness checking.
    </p>
  </div>

  <div style="position: sticky; top: 6rem;">
    <div style="background: #1f2937; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); margin-bottom: 1rem;">
      <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: #374151; border-bottom: 1px solid #4b5563;">
        <div style="display: flex; gap: 0.375rem;">
          <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
          <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
          <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
        </div>
        <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.75rem; color: #9ca3af; margin-left: 1rem;">using-binop.js</span>
      </div>
      <pre style="padding: 1.5rem; margin: 0; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.7; overflow-x: auto;"><code><span style="color: #6b7280">// Predefined binop</span>
<span style="color: #a5b4fc">const</span> <span style="color: #e2e8f0">myPrimitive = </span><span style="color: #a5b4fc">new</span> <span style="color: #fbbf24">gridwisePrimitive</span>({
  <span style="color: #94a3b8">device,</span>
  <span style="color: #e2e8f0">binop: </span><span style="color: #e2e8f0">BinOpAddU32</span>,
  <span style="color: #6b7280">...</span>
});</code></pre>
    </div>
    <div style="background: #1f2937; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);">
      <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: #374151; border-bottom: 1px solid #4b5563;">
        <div style="display: flex; gap: 0.375rem;">
          <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
          <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
          <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
        </div>
        <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.75rem; color: #9ca3af; margin-left: 1rem;">dynamic-binop.js</span>
      </div>
      <pre style="padding: 1.5rem; margin: 0; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.7; overflow-x: auto;"><code><span style="color: #6b7280">// Instantiate on the fly</span>
<span style="color: #a5b4fc">const</span> <span style="color: #e2e8f0">datatype = </span><span style="color: #34d399">"f32"</span>;
<span style="color: #a5b4fc">const</span> <span style="color: #e2e8f0">myPrimitive = </span><span style="color: #a5b4fc">new</span> <span style="color: #fbbf24">gridwisePrimitive</span>({
  <span style="color: #94a3b8">device,</span>
  <span style="color: #e2e8f0">binop: </span><span style="color: #a5b4fc">new</span> <span style="color: #fbbf24">BinOpAdd</span>({ datatype }),
  <span style="color: #6b7280">...</span>
});</code></pre>
    </div>
  </div>
</section>

<!-- Section 2: BinOp Requirements -->
<section style="padding-top: 2rem; border-top: 1px solid #e5e7eb;">
  <div style="max-width: 800px;">
    <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 2rem;">What does a BinOp provide?</h2>
    <p style="color: #4b5563; line-height: 1.75; margin-bottom: 2rem;">
      We want to write primitives that work for any monoid. Other languages have more structured ways to write such code, but WGSL development in JavaScript commonly uses string-pasting to construct runtime-generated kernels. A <code>BinOp</code> provides all the text and member functions that are specific to the particular monoid we are using.
    </p>
    
    <div style="display: grid; gap: 1.5rem;">
      <!-- Required Properties -->
      <div style="padding: 1.5rem; border: 1px solid #e5e7eb; border-radius: 0.75rem; background: #f9fafb;">
        <h3 style="font-size: 1.25rem; font-weight: 700; color: #111827; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
          <span style="display: inline-block; width: 1.5rem; height: 1.5rem; border-radius: 50%; background: #4f46e5; color: white; text-align: center; line-height: 1.5rem; font-size: 0.75rem;">✓</span>
          Required Properties
        </h3>
        
        <div style="display: grid; gap: 1rem;">
          <div>
            <h4 style="font-size: 1rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;"><code>identity</code></h4>
            <p style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.5rem;">An identity element. For addition, this is 0 (independent of datatype). For multiplication, it is 1; for minimum, the largest representable value for that datatype; for maximum, the smallest representable value.</p>
            <code style="background: #e5e7eb; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem;">this.identity = 0;</code>
          </div>
          
          <div>
            <h4 style="font-size: 1rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;"><code>op</code></h4>
            <p style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.5rem;">A CPU-side function. This JavaScript function takes two arguments <code>a</code> and <code>b</code> and returns <code>a op b</code>. Because JavaScript's internal datatypes are limited, this sometimes requires judicious use of JavaScript typed arrays.</p>
            <code style="background: #e5e7eb; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem;">this.op = (a, b) => a + b;</code>
          </div>
          
          <div>
            <h4 style="font-size: 1rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;"><code>wgslfn</code></h4>
            <p style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.5rem;">A GPU-side function declaration. This must define a WGSL function named <code>binop</code>. Like <code>op</code>, this function takes two arguments <code>a</code> and <code>b</code> and returns <code>a op b</code>. It can use string interpolation as appropriate and will probably have to use a datatype.</p>
            <code style="background: #e5e7eb; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; display: block;">this.wgslfn = `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return a+b;}`;</code>
          </div>
        </div>
      </div>
      
      <!-- Optional Properties -->
      <div style="padding: 1.5rem; border: 1px solid #e5e7eb; border-radius: 0.75rem; background: #fffbeb;">
        <h3 style="font-size: 1.25rem; font-weight: 700; color: #111827; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
          <span style="display: inline-block; width: 1.5rem; height: 1.5rem; border-radius: 50%; background: #f59e0b; color: white; text-align: center; line-height: 1.5rem; font-size: 0.75rem;">?</span>
          Optional WGSL Function Names
        </h3>
        <p style="font-size: 0.875rem; color: #78350f; margin-bottom: 1rem;">These are "optional" in the sense that they are not a core part of <code>BinOp</code>, so most primitives will probably work if they are not specified.</p>
        
        <div style="display: grid; gap: 1rem;">
          <div>
            <h4 style="font-size: 1rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;"><code>wgslatomic</code></h4>
            <p style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.5rem;">A string that is a function name. This names the WGSL atomic function that is the atomic variant of <code>wgslfn</code>. Any of <a href="https://www.w3.org/TR/WGSL/#atomic-rmw" style="color: #4f46e5; text-decoration: underline;">these functions</a> are appropriate, but note that (at the time of writing) WGSL atomics are only available for <code>i32</code> and <code>u32</code> datatypes.</p>
            <code style="background: #e5e7eb; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem;">this.wgslatomic = "atomicAdd";</code>
          </div>
          
          <div>
            <h4 style="font-size: 1rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;">Subgroup Functions</h4>
            <p style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.75rem;">Note these functions apply to only a subset of operations and datatypes. Supporting anything outside of this subset requires <a href="subgroup-strategy.html" style="color: #4f46e5; text-decoration: underline;">emulation</a>.</p>
            
            <ul style="font-size: 0.875rem; color: #6b7280; list-style: none; padding: 0; display: grid; gap: 0.75rem;">
              <li style="padding: 0.75rem; background: white; border-radius: 0.5rem; border: 1px solid #e5e7eb;">
                <strong style="color: #111827;"><code>subgroupReduceOp</code></strong> — Reduces the values in a subgroup using this operation. Supported WGSL functions: <code>subgroup{Add,And,Max,Min,Mul,Or,Xor}</code>
              </li>
              <li style="padding: 0.75rem; background: white; border-radius: 0.5rem; border: 1px solid #e5e7eb;">
                <strong style="color: #111827;"><code>subgroupInclusiveScanOp</code></strong> — Computes an inclusive scan of the values in a subgroup using this operation. Supported WGSL functions: <code>subgroupInclusive{Add,Mul}</code>
              </li>
              <li style="padding: 0.75rem; background: white; border-radius: 0.5rem; border: 1px solid #e5e7eb;">
                <strong style="color: #111827;"><code>subgroupExclusiveScanOp</code></strong> — Computes an exclusive scan of the values in a subgroup using this operation. Supported WGSL functions: <code>subgroupExclusive{Add,Mul}</code>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- Section 3: Example Implementation -->
<section style="padding-top: 3rem; margin-top: 3rem; border-top: 1px solid #e5e7eb;">
  <div style="max-width: 800px;">
    <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 1.5rem;">Example Implementation</h2>
    <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
      Below is an example implementation, <code>BinOpAdd</code>, which takes an argument of <code>{ datatype = "..." }</code> that is used to specialize it.
    </p>
  </div>

  <div style="background: #1f2937; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);">
    <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: #374151; border-bottom: 1px solid #4b5563;">
      <div style="display: flex; gap: 0.375rem;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
      </div>
      <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.75rem; color: #9ca3af; margin-left: 1rem;">binop.mjs</span>
    </div>
    <pre style="padding: 1.5rem; margin: 0; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.7; overflow-x: auto;"><code><span style="color: #a5b4fc">export class</span> <span style="color: #fbbf24">BinOpAdd</span> <span style="color: #a5b4fc">extends</span> <span style="color: #fbbf24">BinOp</span> {
  <span style="color: #34d399">constructor</span>(<span style="color: #e2e8f0">args</span>) {
    <span style="color: #a5b4fc">super</span>(args);
    <span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">identity</span> = <span style="color: #fbbf24">0</span>;
    <span style="color: #a5b4fc">if</span> (args.<span style="color: #e2e8f0">datatype</span> == <span style="color: #34d399">"f32"</span>) {
      <span style="color: #a5b4fc">const</span> <span style="color: #e2e8f0">f32array</span> = <span style="color: #a5b4fc">new</span> <span style="color: #fbbf24">Float32Array</span>(<span style="color: #fbbf24">3</span>);
      <span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">op</span> = (<span style="color: #e2e8f0">a</span>, <span style="color: #e2e8f0">b</span>) => {
        f32array[<span style="color: #fbbf24">1</span>] = a;
        f32array[<span style="color: #fbbf24">2</span>] = b;
        f32array[<span style="color: #fbbf24">0</span>] = f32array[<span style="color: #fbbf24">1</span>] + f32array[<span style="color: #fbbf24">2</span>];
        <span style="color: #a5b4fc">return</span> f32array[<span style="color: #fbbf24">0</span>];
      };
    } <span style="color: #a5b4fc">else</span> {
      <span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">op</span> = (<span style="color: #e2e8f0">a</span>, <span style="color: #e2e8f0">b</span>) => a + b;
    }
    <span style="color: #a5b4fc">switch</span> (<span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">datatype</span>) {
      <span style="color: #a5b4fc">case</span> <span style="color: #34d399">"f32"</span>:
        <span style="color: #a5b4fc">break</span>;
      <span style="color: #a5b4fc">case</span> <span style="color: #34d399">"i32"</span>:
        <span style="color: #a5b4fc">break</span>;
      <span style="color: #a5b4fc">case</span> <span style="color: #34d399">"u32"</span>: <span style="color: #6b7280">// fall-through OK</span>
      <span style="color: #a5b4fc">default</span>:
        <span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">wgslatomic</span> = <span style="color: #34d399">"atomicAdd"</span>; <span style="color: #6b7280">// u32 only</span>
        <span style="color: #a5b4fc">break</span>;
    }
    <span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">wgslfn</span> = <span style="color: #34d399">`fn binop(a : ${<span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">datatype</span>}, b : ${<span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">datatype</span>}) -> ${<span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">datatype</span>} {return a+b;}`</span>;
    <span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">subgroupReduceOp</span> = <span style="color: #34d399">"subgroupAdd"</span>;
    <span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">subgroupInclusiveScanOp</span> = <span style="color: #34d399">"subgroupInclusiveAdd"</span>;
    <span style="color: #a5b4fc">this</span>.<span style="color: #e2e8f0">subgroupExclusiveScanOp</span> = <span style="color: #34d399">"subgroupExclusiveAdd"</span>;
  }
}</code></pre>
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
