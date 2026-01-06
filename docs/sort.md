---
layout: page
title: "Sort"
category: gridwise
permalink: /sort/
excerpt: "GPU radix sort implementation using the OneSweep architecture with chained scan and forward-progress guarantees."
order: 5
---

<div style="margin-bottom: 3rem;">
  <div style="display: inline-flex; align-items: center; gap: 0.75rem; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 0.5rem 1.25rem; border-radius: 0.5rem; font-weight: 600; font-size: 0.875rem; margin-bottom: 1.5rem;">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M7 12h10M3 18h18"/>
    </svg>
    Radix Sort
  </div>
  <h1 style="font-size: 2.5rem; font-weight: 800; color: #1e293b; margin: 0 0 1rem 0; line-height: 1.2;">OneSweep Sort</h1>
  <p style="font-size: 1.25rem; color: #64748b; margin: 0; line-height: 1.6;">GPU radix sort implementation using the OneSweep architecture with chained scan and forward-progress guarantees.</p>
</div>

<div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;">
  <p style="margin: 0; color: #064e3b; line-height: 1.6;">
    The dominant approach to GPU sorting is a <a href="https://en.wikipedia.org/wiki/Radix_sort" style="color: #059669; text-decoration: none; font-weight: 600;">radix sort</a> over input keys. In general, radix sorts deliver high performance on GPUs because they require <em>O(n)</em> work for inputs of <em>n</em> elements, because their constituent memory accesses are generally fairly coalesced and thus deliver good memory performance, and because the underlying compute primitives that compose to make the sort are good matches for GPUs.
  </p>
</div>

<div style="background: white; border: 2px solid #d1fae5; border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 2rem;">
  <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2">
      <circle cx="12" cy="12" r="10"/>
      <path d="M12 16v-4M12 8h.01"/>
    </svg>
    <h3 style="margin: 0; color: #1e293b; font-size: 1.125rem; font-weight: 700;">Architecture Choice</h3>
  </div>
  <p style="margin: 0; color: #475569; line-height: 1.6;">
    The specific sort architecture we choose is <a href="https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus" style="color: #059669; text-decoration: none; font-weight: 600;">OneSweep</a>, developed by Andrey Adinets and Duane Merrill of NVIDIA. Internally, OneSweep uses a chained scan, as does our implementation. The challenges we outlined in our <a href="scan-and-reduce.html#decoupled-lookback-and-forward-progress-guarantees" style="color: #059669; text-decoration: none;">scan description</a> with respect to forward-progress guarantees are the same. Our sort implementation employs both <strong>lookback</strong> and <strong>fallback</strong> to ensure that it will work on GPUs that lack forward-progress guarantees.
  </p>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #1e293b; margin: 3rem 0 1.5rem 0; padding-bottom: 0.75rem; border-bottom: 3px solid #10b981;">Our Sort Implementation</h2>

<div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem;">
  <p style="margin: 0; color: #064e3b; line-height: 1.6;">
    <strong>Core Concept:</strong> At its heart, radix sort computes a permutation of its input values and then performs the permutation. Computing the entire permutation would be intractable (the size of the intermediate data structures would be enormous), so typically a radix sort makes several passes over the input, each time computing a permutation for a subset of input bits (this subset is called a <strong>"digit"</strong>).
  </p>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
  <div style="background: #ffffff; border: 1px solid #d1e7dd; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 1px 3px rgba(45, 92, 79, 0.08);">
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
      <div style="background: linear-gradient(135deg, #2d5c4f 0%, #1e4338 100%); color: #e8f5f1; width: 2.5rem; height: 2.5rem; border-radius: 0.375rem; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 1.125rem; box-shadow: 0 2px 4px rgba(45, 92, 79, 0.2);">8</div>
      <h3 style="margin: 0; color: #1a1f2e; font-size: 1.125rem; font-weight: 600;">Bits Per Pass</h3>
    </div>
    <p style="margin: 0; color: #5a6c7d; line-height: 1.6; font-size: 0.9375rem;">
      Each digit can take on 2<sup>8</sup> = <strong>256</strong> possible values
    </p>
  </div>

  <div style="background: #ffffff; border: 1px solid #d1e7dd; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 1px 3px rgba(45, 92, 79, 0.08);">
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
      <div style="background: linear-gradient(135deg, #3a6d5d 0%, #2d5c4f 100%); color: #e8f5f1; width: 2.5rem; height: 2.5rem; border-radius: 0.375rem; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 1.125rem; box-shadow: 0 2px 4px rgba(45, 92, 79, 0.2);">4</div>
      <h3 style="margin: 0; color: #1a1f2e; font-size: 1.125rem; font-weight: 600;">Passes for 32-bit</h3>
    </div>
    <p style="margin: 0; color: #5a6c7d; line-height: 1.6; font-size: 0.9375rem;">
      Starting with least significant bits, classifying into <strong>256 buckets</strong> per pass
    </p>
  </div>

  <div style="background: #ffffff; border: 1px solid #d1e7dd; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 1px 3px rgba(45, 92, 79, 0.08);">
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#2d5c4f" stroke-width="2">
        <path d="M4 7h16M4 12h16M4 17h16"/>
      </svg>
      <h3 style="margin: 0; color: #1a1f2e; font-size: 1.125rem; font-weight: 600;">Tile-Based Processing</h3>
    </div>
    <p style="margin: 0; color: #5a6c7d; line-height: 1.6; font-size: 0.9375rem;">
      Input divided into equal-sized tiles, one workgroup per tile
    </p>
  </div>
</div>

<div style="background: #ffffff; border: 1px solid #d1e7dd; border-radius: 0.5rem; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(45, 92, 79, 0.08);">
  <h3 style="margin: 0 0 1.5rem 0; color: #1a1f2e; font-size: 1.25rem; font-weight: 600;">Computing Destination Addresses</h3>
  <p style="margin: 0 0 1.5rem 0; color: #3a4a5a; line-height: 1.6;">
    Computing this permutation means computing a <strong>"destination address"</strong> for each key—to where in the output will this key be written? That address is the sum of:
  </p>
  
  <div style="display: flex; flex-direction: column; gap: 1rem;">
    <div style="display: flex; align-items: start; gap: 1rem; background: #f5faf8; padding: 1.25rem; border-radius: 0.375rem; border-left: 3px solid #2d5c4f;">
      <div style="background: linear-gradient(135deg, #2d5c4f 0%, #1e4338 100%); color: #e8f5f1; width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0; box-shadow: 0 2px 4px rgba(45, 92, 79, 0.2);">1</div>
      <div>
        <p style="margin: 0; color: #1a1f2e; font-weight: 600;">Global Histogram</p>
        <p style="margin: 0.5rem 0 0 0; color: #5a6c7d; font-size: 0.9375rem; line-height: 1.6;">
          The number of keys that fall into a bucket before my bucket
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #5a6c7d; font-size: 0.875rem; font-style: italic;">
          Computed via: <code style="background: #e8f5f1; color: #1e4338; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem;">global_hist</code> kernel + <code style="background: #e8f5f1; color: #1e4338; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem;">onesweep_scan</code> (exclusive-sum-scan on histograms)
        </p>
      </div>
    </div>

    <div style="display: flex; align-items: start; gap: 1rem; background: #f5faf8; padding: 1.25rem; border-radius: 0.375rem; border-left: 3px solid #3a6d5d;">
      <div style="background: linear-gradient(135deg, #3a6d5d 0%, #2d5c4f 100%); color: #e8f5f1; width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0; box-shadow: 0 2px 4px rgba(45, 92, 79, 0.2);">2</div>
      <div>
        <p style="margin: 0; color: #1a1f2e; font-weight: 600;">Chained Scan</p>
        <p style="margin: 0.5rem 0 0 0; color: #5a6c7d; font-size: 0.9375rem; line-height: 1.6;">
          The number of keys that are in my bucket and are processed in a previous tile
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #5a6c7d; font-size: 0.875rem; font-style: italic;">
          Computed via: <code style="background: #e8f5f1; color: #1e4338; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem;">onesweep_pass</code> kernel with chained sum scan (lookback + fallback)
        </p>
      </div>
    </div>

    <div style="display: flex; align-items: start; gap: 1rem; background: #f5faf8; padding: 1.25rem; border-radius: 0.375rem; border-left: 3px solid #2d5c4f;">
      <div style="background: linear-gradient(135deg, #2d5c4f 0%, #1e4338 100%); color: #e8f5f1; width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0; box-shadow: 0 2px 4px rgba(45, 92, 79, 0.2);">3</div>
      <div>
        <p style="margin: 0; color: #1a1f2e; font-weight: 600;">Workgroup-Local Count</p>
        <p style="margin: 0.5rem 0 0 0; color: #5a6c7d; font-size: 0.9375rem; line-height: 1.6;">
          The number of keys that are in my bucket and are processed earlier than my key within my tile
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #5a6c7d; font-size: 0.875rem; font-style: italic;">
          Computed via: <code style="background: #e8f5f1; color: #1e4338; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem;">onesweep_pass</code> kernel (workgroup-local only, doesn't participate in chained scan)
        </p>
      </div>
    </div>
  </div>
</div>

<div style="background: linear-gradient(135deg, #e8f5f1 0%, #d1e7dd 100%); border-left: 4px solid #2d5c4f; padding: 1.5rem; border-radius: 0.375rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(45, 92, 79, 0.1);">
  <p style="margin: 0 0 0.75rem 0; font-weight: 600; font-size: 1.125rem; color: #1a3a2e;">Memory Coalescing Optimization</p>
  <p style="margin: 0; line-height: 1.6; color: #1a3a2e;">
    Given this computed address per key, we could directly scatter each key to its location in global memory. However, to improve memory coalescing, we <strong>first write keys into workgroup memory</strong> and scatter from there. This puts neighboring keys next to each other in workgroup memory and significantly improves the throughput of the global scatter.
  </p>
</div>

<div style="background: #ffffff; border: 1px solid #d1e7dd; border-radius: 0.5rem; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(45, 92, 79, 0.08);">
  <h3 style="margin: 0 0 1rem 0; color: #1a1f2e; font-size: 1.25rem; font-weight: 600;">256-Entry Histogram Lookback</h3>
  <p style="margin: 0 0 1rem 0; color: #3a4a5a; line-height: 1.6;">
    Note that the chained scan in onesweep is a chained scan over an entire 256-entry histogram. Lookback on such a large data structure is more complicated than lookback on a single data value, as we see in <a href="scan-and-reduce.html" style="color: #2d5c4f; text-decoration: none; font-weight: 600; border-bottom: 1px solid #2d5c4f;">our scan implementation</a>, because we use an entire workgroup to look back.
  </p>
  <div style="background: #f5faf8; border: 1px solid #c5ddd2; padding: 1.25rem; border-radius: 0.375rem;">
    <p style="margin: 0; color: #1a3a2e; line-height: 1.6;">
      <strong>Additional Complexity:</strong> <em>Any</em> thread in the lookback may fail to find a ready value, and if this is the case, the entire workgroup must drop into fallback. Thus we have to keep track of both <strong>per-thread lookback success</strong> as well as <strong>per-subgroup lookback success</strong>, and only determine lookback is successful if all subgroups report success.
    </p>
  </div>
</div>

<div style="background: linear-gradient(135deg, #e8f5f1 0%, #d1e7dd 100%); border-left: 4px solid #2d5c4f; padding: 1.5rem; border-radius: 0.375rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(45, 92, 79, 0.1);">
  <p style="margin: 0 0 0.75rem 0; font-weight: 600; font-size: 1.125rem; color: #1a3a2e;">Ping-Pong Array Behavior</p>
  <p style="margin: 0; line-height: 1.6; color: #1a3a2e;">
    Radix sort implementations (including ours) typically use a ping-pong pair of arrays: on each pass, one array is the input and one array is the output, and on each pass, their roles switch. Because we are sorting 32- or 64-bit keys at 8 bits per pass, this means <strong>the input will be overwritten by the output</strong> and the primitive's output will be produced in the same buffer as its original input. Overwriting the input is not ideal behavior but is probably preferable to approaches that hide it from the user (by, say, preemptively copying the input into a temporary buffer and copying the temporary input and output at the end of the computation).
  </p>
</div>

<h2 style="font-size: 1.875rem; font-weight: 600; color: #1a1f2e; margin: 3rem 0 1.5rem 0; padding-bottom: 0.75rem; border-bottom: 2px solid #2d5c4f;">Configuring and Calling Gridwise Sort</h2>

<div style="background: #ffffff; border: 1px solid #d1e7dd; border-radius: 0.5rem; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(45, 92, 79, 0.08);">
  <h3 style="margin: 0 0 1.5rem 0; color: #1a1f2e; font-size: 1.25rem; font-weight: 600; display: flex; align-items: center; gap: 0.75rem;">
    <span style="background: linear-gradient(135deg, #e8f5f1 0%, #d1e7dd 100%); color: #2d5c4f; width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 1rem; border: 2px solid #2d5c4f;">1</span>
    Defining the Primitive
  </h3>
  
  <p style="margin: 0 0 1.5rem 0; color: #3a4a5a; line-height: 1.6;">
    Declare the scan or reduce primitive as an instance of the <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">OneSweepSort</code> class. An example scan declaration:
  </p>

  <div style="background: #1a1f2e; border-radius: 0.5rem; overflow: hidden; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
    <div style="background: linear-gradient(135deg, #2a3142 0%, #1f2533 100%); padding: 0.75rem 1.25rem; border-bottom: 1px solid #2a3142; display: flex; align-items: center; justify-content: space-between;">
      <div style="display: flex; gap: 0.5rem; align-items: center;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #2d5c4f;"></div>
      </div>
      <span style="color: #9ca3af; font-size: 0.875rem; font-weight: 500;">define-sort.js</span>
    </div>
    <div style="padding: 1.5rem; overflow-x: auto;">
      <pre style="margin: 0; color: #e5e7eb; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.9375rem; line-height: 1.6;"><span style="color: #7dd3c0;">const</span> datatype <span style="color: #7dd3c0;">=</span> <span style="color: #a5f3fc;">"u32"</span><span style="color: #64748b;">; // or "i32" or "f32"</span>
<span style="color: #7dd3c0;">const</span> oneSweepSortPrimitive <span style="color: #7dd3c0;">=</span> <span style="color: #7dd3c0;">new</span> <span style="color: #5dd4a8;">OneSweepSort</span>({
  device,
  datatype, <span style="color: #64748b;">// use the "datatype" string defined above</span>
  type: <span style="color: #a5f3fc;">"keysonly"</span>,
  direction: <span style="color: #a5f3fc;">"ascending"</span>,
});</pre>
    </div>
  </div>

  <p style="margin: 0 0 1rem 0; color: #3a4a5a; line-height: 1.6; font-weight: 600;">
    Gridwise OneSweep supports all combinations of:
  </p>

  <div style="display: flex; flex-direction: column; gap: 1rem;">
    <div style="background: #f5faf8; padding: 1.25rem; border-radius: 0.375rem; border-left: 3px solid #2d5c4f;">
      <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.5rem;">
        <code style="background: #e8f5f1; color: #1e4338; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">datatype</code>
        <span style="color: #5a6c7d; font-size: 0.875rem;">Required</span>
      </div>
      <p style="margin: 0; color: #3a4a5a; line-height: 1.6; font-size: 0.9375rem;">
        <code style="color: #2d5c4f; font-weight: 600;">u32</code>, <code style="color: #2d5c4f; font-weight: 600;">i32</code>, <code style="color: #2d5c4f; font-weight: 600;">f32</code>, <code style="color: #2d5c4f; font-weight: 600;">u64</code>. Internally, OneSweep converts non-unsigned-int keys into unsigned-int keys that respect the original order, sorts as if the keys were unsigned ints, and then reverses the conversion when writing the keys into the output.
      </p>
    </div>

    <div style="background: #f5faf8; padding: 1.25rem; border-radius: 0.375rem; border-left: 3px solid #3a6d5d;">
      <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.5rem;">
        <code style="background: #e8f5f1; color: #1e4338; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">type</code>
        <span style="color: #5a6c7d; font-size: 0.875rem;">Optional (default: <code style="color: #2d5c4f;">keysonly</code>)</span>
      </div>
      <p style="margin: 0; color: #3a4a5a; line-height: 1.6; font-size: 0.9375rem;">
        <code style="color: #3a6d5d; font-weight: 600;">keysonly</code>, <code style="color: #3a6d5d; font-weight: 600;">keyvalue</code>. A key-value sort has an array of keys and also an array of values where each value is associated with its corresponding key in the keys array.
      </p>
    </div>

    <div style="background: #f5faf8; padding: 1.25rem; border-radius: 0.375rem; border-left: 3px solid #478572;">
      <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.5rem;">
        <code style="background: #e8f5f1; color: #1e4338; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">direction</code>
        <span style="color: #5a6c7d; font-size: 0.875rem;">Optional (default: <code style="color: #2d5c4f;">ascending</code>)</span>
      </div>
      <p style="margin: 0; color: #3a4a5a; line-height: 1.6; font-size: 0.9375rem;">
        <code style="color: #478572; font-weight: 600;">ascending</code> (sort low to high), <code style="color: #478572; font-weight: 600;">descending</code> (sort high to low).
      </p>
    </div>
  </div>
</div>

<div style="background: #ffffff; border: 1px solid #d1e7dd; border-radius: 0.5rem; padding: 2rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(45, 92, 79, 0.08);">
  <h3 style="margin: 0 0 1.5rem 0; color: #1a1f2e; font-size: 1.25rem; font-weight: 600; display: flex; align-items: center; gap: 0.75rem;">
    <span style="background: linear-gradient(135deg, #e8f5f1 0%, #d1e7dd 100%); color: #2d5c4f; width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 1rem; border: 2px solid #2d5c4f;">2</span>
    Configuring the Primitive
  </h3>
  
  <p style="margin: 0 0 1.5rem 0; color: #3a4a5a; line-height: 1.6;">
    Once the primitive is <em>defined</em>, it must then be <em>configured</em>. The primitive knows that it requires an input/output and temporary buffer, labeled <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">keysInOut</code> and <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">keysTemp</code>. (We use our <a href="buffer.html" style="color: #2d5c4f; text-decoration: none; font-weight: 600; border-bottom: 1px solid #2d5c4f;"><code>Buffer</code> class</a> for this.) If we are doing a key-value sort, we also require <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">payloadInOut</code> and <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">payloadTemp</code> buffers, which store the values.) We configure the primitive by registering data buffers with the primitive. This can be done either with a <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">primitive.registerBuffer()</code> call or as an argument to the <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">execute</code> call. (The former is preferred if we need to register the buffer(s) once and then call <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">execute</code> many times.)
  </p>

  <p style="margin: 0 0 1.5rem 0; color: #3a4a5a; line-height: 1.6;">
    To register a buffer, simply call <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">primitive.registerBuffer(buffer)</code>, where <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">buffer.label</code> is one of the buffers above. The below code creates a <code style="background: #f5faf8; color: #2d5c4f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600; border: 1px solid #d1e7dd;">Buffer</code> then registers it.
  </p>

  <div style="background: #1a1f2e; border-radius: 0.5rem; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
    <div style="background: linear-gradient(135deg, #2a3142 0%, #1f2533 100%); padding: 0.75rem 1.25rem; border-bottom: 1px solid #2a3142; display: flex; align-items: center; justify-content: space-between;">
      <div style="display: flex; gap: 0.5rem; align-items: center;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #2d5c4f;"></div>
      </div>
      <span style="color: #9ca3af; font-size: 0.875rem; font-weight: 500;">configure-sort.js</span>
    </div>
    <div style="padding: 1.5rem; overflow-x: auto;">
      <pre style="margin: 0; color: #e5e7eb; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.9375rem; line-height: 1.6;"><span style="color: #7dd3c0;">const</span> inputLength <span style="color: #7dd3c0;">=</span> <span style="color: #5dd4a8;">2</span> <span style="color: #7dd3c0;">**</span> <span style="color: #5dd4a8;">20</span>;
testKeysBuffer <span style="color: #7dd3c0;">=</span> <span style="color: #7dd3c0;">new</span> <span style="color: #5dd4a8;">Buffer</span>({
  device,
  datatype: <span style="color: #a5f3fc;">"f32"</span>,
  length: inputLength,
  label: <span style="color: #a5f3fc;">"keysInOut"</span>,
  createCPUBuffer: <span style="color: #5dd4a8;">true</span>,
  initializeCPUBuffer: <span style="color: #5dd4a8;">true</span> <span style="color: #64748b;">/* fill with default data */</span>,
  storeCPUBackup: <span style="color: #5dd4a8;">true</span>, <span style="color: #64748b;">/* because readback will overwrite the CPU data */</span>
  createGPUBuffer: <span style="color: #5dd4a8;">true</span>,
  initializeGPUBuffer: <span style="color: #5dd4a8;">true</span> <span style="color: #64748b;">/* with CPU data */</span>,
  createMappableGPUBuffer: <span style="color: #5dd4a8;">true</span>, <span style="color: #64748b;">/* we read this back to test correctness */</span>
});
oneSweepSortPrimitive.<span style="color: #5dd4a8;">registerBuffer</span>(testKeysBuffer);</pre>
    </div>
  </div>
</div>

<div style="background: white; border: 2px solid #e2e8f0; border-radius: 0.75rem; padding: 2rem; margin-bottom: 2rem;">
  <h3 style="margin: 0 0 1.5rem 0; color: #1e293b; font-size: 1.25rem; font-weight: 700; display: flex; align-items: center; gap: 0.75rem;">
    <span style="background: #ddd6fe; color: #6366f1; width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1rem;">3</span>
    Calling Scan or Reduce
  </h3>
  
  <p style="margin: 0 0 1.5rem 0; color: #475569; line-height: 1.6;">
    Once the primitive is defined and configured, simply call its <code style="background: #f1f5f9; color: #6366f1; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">execute()</code> method.
  </p>

  <div style="background: #fef3c7; border: 1px solid #fbbf24; padding: 1.25rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
    <p style="margin: 0 0 0.75rem 0; color: #78350f; line-height: 1.6;">
      If you have not yet registered buffers, you can specify them in the argument object as <code style="background: #fde68a; color: #78350f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">keysInOut</code>, <code style="background: #fde68a; color: #78350f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">keysTemp</code>, etc.
    </p>
    <p style="margin: 0; color: #78350f; line-height: 1.6; font-size: 0.9375rem;">
      Other possible arguments (which are timing-specific and thus which you are unlikely to use unless you are benchmarking):
    </p>
    <ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem; color: #78350f; line-height: 1.6; font-size: 0.9375rem;">
      <li><code style="background: #fde68a; color: #78350f; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem;">trials</code> with an integer argument. This will run the kernel(s) that number of times. Default: 1.</li>
      <li><code style="background: #fde68a; color: #78350f; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem;">enableGPUTiming</code> with either true or false. If true, please ensure that the device has a set of required features that include <code>timestamp-query</code>. Default: false.</li>
      <li><code style="background: #fde68a; color: #78350f; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875rem;">enableCPUTiming</code> with either true or false. Default: false.</li>
    </ul>
  </div>

  <p style="margin: 0 0 1.5rem 0; color: #475569; line-height: 1.6;">
    Note that <code style="background: #f1f5f9; color: #6366f1; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">execute()</code> is declared <code style="background: #f1f5f9; color: #6366f1; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9375rem; font-weight: 600;">async</code>.
  </p>

  <div style="background: #1e293b; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
    <div style="background: linear-gradient(135deg, #374151 0%, #1f2937 100%); padding: 0.75rem 1.25rem; border-bottom: 1px solid #374151; display: flex; align-items: center; justify-content: space-between;">
      <div style="display: flex; gap: 0.5rem; align-items: center;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #ef4444;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #f59e0b;"></div>
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background: #10b981;"></div>
      </div>
      <span style="color: #9ca3af; font-size: 0.875rem; font-weight: 500;">execute-sort.js</span>
    </div>
    <div style="padding: 1.5rem; overflow-x: auto;">
      <pre style="margin: 0; color: #e5e7eb; font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace; font-size: 0.9375rem; line-height: 1.6;"><span style="color: #818cf8;">await</span> oneSweepSortPrimitive.<span style="color: #fbbf24;">execute</span>();
<span style="color: #64748b;">// or, if we want to specify buffers only when execute is called</span>
<span style="color: #818cf8;">await</span> oneSweepSortPrimitive.<span style="color: #fbbf24;">execute</span>({
  keysInOut: testKeysBuffer,
  keysTemp: testKeysTempBuffer,
});
<span style="color: #64748b;">// or (maybe if you're benchmarking)</span>
<span style="color: #818cf8;">await</span> oneSweepSortPrimitive.<span style="color: #fbbf24;">execute</span>({
  trials: <span style="color: #fbbf24;">1</span>,
  enableGPUTiming: <span style="color: #fbbf24;">false</span>,
  enableCPUTiming: <span style="color: #fbbf24;">true</span>,
});</pre>
    </div>
  </div>
</div>

<h2 style="font-size: 1.875rem; font-weight: 700; color: #1e293b; margin: 3rem 0 1.5rem 0; padding-bottom: 0.75rem; border-bottom: 3px solid #8b5cf6;">Usage and Performance Notes</h2>

<div style="display: flex; flex-direction: column; gap: 1.5rem; margin-bottom: 2rem;">
  <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-left: 4px solid #ef4444; padding: 1.5rem; border-radius: 0.5rem;">
    <div style="display: flex; align-items: start; gap: 1rem;">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" style="flex-shrink: 0; margin-top: 0.125rem;">
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      <div style="color: #7f1d1d;">
        <p style="margin: 0 0 0.5rem 0; font-weight: 600; font-size: 1.125rem;">Input Size Limitation</p>
        <p style="margin: 0; line-height: 1.6;">
          The number of items to sort must be no greater than <strong>2<sup>30</sup></strong>. (CUB does the same thing.) We use the two most-significant bits as status bits. It would take a large engineering effort to remove this limitation.
        </p>
      </div>
    </div>
  </div>

  <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-left: 4px solid #ef4444; padding: 1.5rem; border-radius: 0.5rem;">
    <div style="display: flex; align-items: start; gap: 1rem;">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" style="flex-shrink: 0; margin-top: 0.125rem;">
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      <div style="color: #7f1d1d;">
        <p style="margin: 0 0 0.5rem 0; font-weight: 600; font-size: 1.125rem;">Input Length Alignment</p>
        <p style="margin: 0; line-height: 1.6;">
          Just as with scan, input lengths <em>must be</em> a multiple of 4. Pad the end of your input array with enough largest-key-value elements to make this work. (This is because internally, we use <code style="background: #fde68a; color: #78350f; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">vec4</code>s for computation.)
        </p>
      </div>
    </div>
  </div>

  <div style="background: white; border: 2px solid #e2e8f0; border-radius: 0.75rem; padding: 1.5rem;">
    <div style="display: flex; align-items: start; gap: 1rem;">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" style="flex-shrink: 0; margin-top: 0.125rem;">
        <circle cx="12" cy="12" r="10"/>
        <path d="M12 16v-4M12 8h.01"/>
      </svg>
      <div style="color: #064e3b;">
        <p style="margin: 0 0 0.5rem 0; font-weight: 600; font-size: 1.125rem; color: #1e293b;">Performance Characteristics</p>
        <p style="margin: 0; line-height: 1.6; color: #475569;">
          During its development, sort had extensive performance testing and the defaults are fairly stable across different GPUs. We sort <strong>8 bits per pass</strong> and this particular implementation has never been tested with a different number of bits per pass. This could be remedied with engineering effort.
        </p>
      </div>
    </div>
  </div>
</div>