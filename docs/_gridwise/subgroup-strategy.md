---
layout: post
title: "Gridwise WebGPU Subgroup Emulation Strategy"
date: 2025-08-18
excerpt: "Strategies for writing high-performance code that uses WGSL subgroups when available and falls back to emulation when not."
---

WGSL's [subgroup built-in functions](https://gpuweb.github.io/gpuweb/wgsl/#subgroup-builtin-functions) have the potential to significantly improve performance on subgroup-capable hardware. A [large (and growing) fraction of WebGPU devices](https://web3dsurvey.com/webgpu) support subgroups. However, writing high-performance code that supports both using subgroups when available _and_ falls back to code with no subgroups is a challenge. This document describes my experience attempting to do exactly that.

In the below discussion, we use the term "hw" to indicate "the development experience that targets hardware-supported subgroups" and "emu" for the development experience that targets non-hardware-supported subgroups, which are not supported by WGSL and thus must be emulated".

## Goals of this effort

In an ideal world, this effort would have resulted in the following outcomes:

- For users of primitives:
  - No code changes. The same code runs on both subgroup-capable (hw) and non-subgroup-capable (emu) hardware
  - However, my effort did not prioritize performance for emu hardware
  - To prioritize performance here, it is likely that we'd need separate primitive formulations for hw/emu scenarios
- For developers of primitives:
  - Minimal additional complexity. This was not entirely achieved.

The checked-in code does not work out of the box for emu. It would not be excessively difficult to make it work, but it would take several hours of grungy effort. I priorized hw development, not emu development, during our scan implementation; scan is already quite complicated and keeping it working for both during active development was just not a priority.

## Initial goal: subgroupShuffle

Let's begin by making one call, [`subgroupShuffle`](https://www.w3.org/TR/WGSL/#subgroupshuffle-builtin), work in both hw and emu contexts. This call looks like:

```wgsl
y = subgroupShuffle(x, id);
```

The first roadblock is that `subgroupShuffle` is not defined as an emu function. Fortunately (and intelligently), WGSL allows the programmer to directly override it ("shadow") with a user-specified function. So let's do that (for the emu context only). If we lack subgroup hardware, we have to communicate shuffled values through workgroup memory, so we have to declare a region of workgroup memory with one element per thread. Here, `source` is a thread index within the subgroup, and `sgid` is the `@builtin(subgroup_invocation_id)`. The code is straightforward:

```wgsl
var<workgroup> wg_sw_subgroups: array<u32, 256>;
fn subgroupShuffle(x: u32, source: u32) -> u32 {
  wg_sw_subgroups[sgid] = x;
  workgroupBarrier();
  return wg_sw_subgroups[source];
}
```

Great! We're done!

Well, not so much. There are many challenges ahead of us.

### Challenge: the @builtin sgid

Our first challenge is using `@builtin(subgroup_invocation_id) sgid`. In emu, this `@builtin` is not defined. We can pass it in as an argument, however.

Thus one possible solution is to define `fn subgroupShuffleWrapper(x, source, sgid)` and then use `subgroupShuffleWrapper` everywhere. We began our development using this strategy, but it is undesirable; it's not reasonable to ask every possible developer within this library to use a set of different functions with different APIs than those in the spec, and it significantly complicated development. We needed a better way, which we address as part of the next challenge.

### Challenge: supporting both hw and emu with minimal impact on the programmer

Our second challenge is ensuring that our `fn subgroupShuffle` definition is only visible when the kernel is compiled in emu, but not in hw. How can we do this?

First, the WebGPU call `device.features.has("subgroups")` tells us if subgroups are supported. We can use the result of this call to declare one of two sets of functions: one that assumes subgroups are available (hw) and one that does not (emu). In our implementation, this set of functions is called `fnDeclarations`. Our syntax is not important here; what is important is what happens in hw and what happens in emu.

At the top of the kernel, we require `${this.fnDeclarations.enableSubgroupsIfAppropriate}`. If we are hw, this emits `enable subgroups;`; if we are emu, this emits nothing.

With our next declaration, we partially solve the problem we identified above, where the subgroup size and subgroup id builtins are available in hw but not in emu. If the kernel is using any subgroup calls, we require `${this.fnDeclarations.subgroupEmulation}`. In hw, this emits nothing. In emu, this declares workgroup memory (for performing the subgroup operations) and subgroup variables (subgroup size and subgroup id), all at module scope:

```wgsl
var<workgroup> wg_sw_subgroups: array<${env.datatype}, ${env.workgroupSize}>;
const sgsz: u32 = ${env.workgroupSize};
var<private> sgid: u32;
```

However, it does not actually assign values to `sgsz` and `sgid`.

Next, for each subgroup call we want to make, we "declare" the call. For subgroup shuffle, in hw, we emit nothing, because `subgroupShuffle` is builtin. In emu (note we use the `sgid` variable we declared at module scope above):

```wgsl
fn subgroupShuffle(x: u32, source: u32) -> u32 {
  /* subgroup emulation must pass through wg_sw_subgroups */
  /* write my value to workgroup memory */
  wg_sw_subgroups[sgid] = bitcast<${env.datatype}>(x);
  workgroupBarrier();
  var shuffled: u32 = bitcast<u32>(wg_sw_subgroups[source]);
  workgroupBarrier();
  return shuffled;
}
```

In hw, each supported subgroup call emits nothing, but we also define other useful functions that are not already defined and emit different implementations for hw and emu. (Example: WGSL supports both inclusive (`subgroupInclusiveAdd`) and exclusive subgroup (`subgroupExclusiveAdd`) scans, but only if the scan operator is addition. Our function library has support for non-addition inclusive and exclusive subgroup scans for both hw and emu.)

Finally, we need to assign values to `sgsz` and `sgid` to functions wehre they are used. Here we use a declaration within each function, `      ${this.fnDeclarations.initializeSubgroupVars()}`. For hw, this does nothing. For emu, this emits `let sgsz: u32 = builtinsUniform.sgsz;\nlet sgid: u32 = builtinsNonuniform.sgid;`.

The burden on the programmer is to (1) declare necessary functions at the top of the module and (2) initialize subgroup variables at the top of each function that uses subgroups, but not to change kernel code. For a hypothetical module/kernel whose only subgroup operation is `subgroupShuffle`, that code looks like:

```wgsl
${this.fnDeclarations.enableSubgroupsIfAppropriate} // must be first line of kernel
${this.fnDeclarations.subgroupEmulation}
${this.fnDeclarations.subgroupShuffle}

fn kernel() {
  ${this.fnDeclarations.initializeSubgroupVars()}
  // ...
}
```

### Challenge: choosing an emulated subgroup size

Finally, we will have to write each emu subgroup operation. Our third challenge is to choose a subgroup size to emulate.

First, we know that using hw subgroup operations will deliver better performance than emu, for several reasons.

1. Hardware-supported subgroup instructions will run faster than the sequences of instructions we need to emulate them
2. Because emulated subgroups don't run in lockstep, we will require more workgroup barriers to emulate subgroups
   - Workgroup barriers will have the largest impact in latency-sensitive and/or large-workgroup kernels
3. Emulated subgroup instructions need to run through workgroup memory, which is slower than registers
4. Allocating additional workgroup memory (at least one word per thread of the workgroup) might decrease the number of subgroups that can fit on a processor, hurting occupancy

Recall that WebGPU does not specify a subgroup size (in hw), although it does specify a minimum and maximum subgroup size. (In fact, some WebGPU-capable hardware may use different subgroup sizes across different kernels in the same application.) WebGPU developers must thus write their code assuming any subgroup size between the minimum and the maximum. Since our kernels already have to handle a range of subgroup sizes, we have some flexibility to choose a subgroup size in emu. We have 3 main choices:

1. Assume that very small subgroup sizes run in lockstep, and use those subgroup sizes. Then we can potentially avoid some barriers and gain some efficiency.
   - But: We can't assume that. WebGPU does not report that information. (If it did, we could take advantage of it.)
2. Assume a comfortable subgroup size (e.g., 32), and add appropriate barriers.
3. Since we're going to have to put barriers everywhere anyway, assume subgroup size == workgroup size.

Let's take a step back and think about how subgroups are used. Consider a reduction across a workgroup (each thread has one item and the workgroup adds up all items). The typical pattern for a workgroup reduction leveraging subgroup support is to (a) use a subgroup reduction on each subgroup then (b) reduce across the results from each subgroup. This pattern is typical: parallelize across subgroups, then combine the results.

Now, if we choose alternatives 1 or 2, then it is highly likely that each workgroup will contain several subgroups. Many primitives will thus have two stages: per-subgroup, then per-workgroup. If we choose alternative 3 (subgroup size == workgroup size), then our algorithms may be simpler, because we don't have to combine results from multiple subgroups within a workgroup. This also simplifies the code that emulates subgroups.

We do see one clear structural issue, though: some subgroup operations have a maximum size (e.g., `subgroupBallot` has a maximum subgroup size of 128, because it returns exactly 128 bits).

Nonetheless for simplicity, we currently choose to always emulate subgroups that are the size of the workgroup, recognizing that this is not a fully generalizable solution.

## Summary

On an Apple M3 with a high-performance scan kernel, the performance difference between hw and emu with the same kernel is ~2.5x.

An open question is whether it is better to write different _kernels_ for hw and emu as opposed to what we did: writing different versions of subgroup functions and keeping the same kernel. The answer probably depends on the nature of the kernels. We did not explore the latter alternative at all.
