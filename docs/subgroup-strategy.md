---
layout: page
title: "Gridwise WebGPU Subgroup Emulation Strategy"
category: gridwise
permalink: /subgroup-strategy/
excerpt: "Strategies for writing high-performance code that uses WGSL subgroups when available and falls back to emulation when not."
order: 8
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

Finally, we will have to actually write each emu subgroup operation. Our third challenge is to choose a subgroup size to emulate.

First, we know that using hw subgroup operations will deliver better performance than emu, for several reasons.

1. Hardware-supported subgroup instructions will run faster than the sequences of instructions we need to emulate them
2. Because emulated subgroups lack hardware subgroup communication primitives, we must coordinate them manually using workgroup memory and barriers
   - Workgroup barriers will have the largest impact in latency-sensitive and/or large-workgroup kernels
3. Emulated subgroup instructions need to run through workgroup memory, which is slower than registers
4. Allocating additional workgroup memory (at least one word per thread of the workgroup) might decrease the number of subgroups that can fit on a processor, hurting occupancy

Recall that WebGPU does not specify a subgroup size (in hw), although it does specify a minimum and maximum subgroup size. (In fact, some WebGPU-capable hardware may use different subgroup sizes across different kernels in the same application.) WebGPU developers must thus write their code assuming any subgroup size between the minimum and the maximum. Since our kernels already have to handle a range of subgroup sizes, we have some flexibility to choose a subgroup size in emu.

We choose to emulate virtual subgroups of size **32** (partitioning the workgroup's flat array and tracking thread subgroup IDs via `lidx % 32u`). 

### Why 32-thread virtual subgroups? (VS Workgroup-sized subgroups)

When choosing an emulated subgroup size, there are two primary designs:
1. **Workgroup-Sized Subgroups** (where the subgroup size matches the entire workgroup size, e.g. 128 or 256 threads).
2. **Fixed 32-Thread Virtual Subgroups** (where each workgroup is partitioned into independent 32-thread lanes, matching hardware subgroup widths).

Below is a comparison of the tradeoffs between these two alternatives:

#### Alternative A: Workgroup-Sized Subgroups
*   **Pros**:
    *   **Simpler Indexing**: There is a 1-to-1 mapping between the workgroup and the subgroup. Thread lane IDs (`sgid`) are simply equal to local thread indices (`lidx`).
    *   **Lower Shared Memory Pressure**: Emulation does not need to partition shared memory into multiple steps/subgroups, conserving valuable Local Data Store (LDS) space.
    *   **Simpler Reductions**: Full workgroup scans/reductions can be implemented with classic, straightforward tree-based reductions.
    *   **Single-Stage Algorithms**: Many parallel primitives normally require a two-stage formulation (e.g. reduce/scan across subgroups, then combine subgroup results at the workgroup level). Equating subgroup size to workgroup size simplifies the primary algorithms by eliminating the need to combine results across subgroups.
*   **Cons**:
    *   **Incompatible with Advanced Algorithms**: Primitives like Radix Sort's warp-level multi-split (`WLMS`) or high-performance prefix scans are designed around the hardware assumption of a 32-lane subgroup. They pack voting outcomes and predicate masks into a single 32-bit `u32` word (e.g. via `subgroupBallot`). If the subgroup size is 256, a 32-bit mask overflows, requiring complex multi-word array allocations that break compatibility with native GPU code.
    *   **Coarse-Grained Synchronization**: Operations on a 256-thread subgroup require workgroup barriers that stall all 256 threads, eliminating the independent, lock-free lane progress that subgroups are meant to achieve.

#### Alternative B: Fixed 32-Thread Virtual Subgroups (Chosen Strategy)
*   **Pros**:
    *   **Strict Compatibility**: Matches the physical subgroup width (32 lanes) of major desktop and mobile GPU architectures (NVIDIA, AMD, Apple, etc.). This ensures that bit-packing operations, mask logic, and trailing-zero counting behave identically in both native and emulation modes.
    *   **Fine-Grained Atomic Synchronization**: Because subgroups are limited to 32 threads, we can use local atomic flag step counters to synchronize communication groups of 32 threads. This avoids stalling the entire workgroup with `workgroupBarrier()` and more closely emulates the independent forward progress semantics of native subgroup execution.
*   **Cons**:
    *   **Increased Shared Memory Overhead**: To emulate shuffles and ballot steps without full-workgroup barriers, we must allocate multi-step atomic arrays (e.g., `wg_sw_subgroups_u32_steps` of size `[6][workgroupSize]`), which increases LDS usage and can slightly limit occupancy on some GPUs.
    *   **Complex Subgroup Mapping**: Requires dynamic lane indexing (`lidx % 32u`) and local subgroup boundaries (`lidx & ~31u`).

**Recommendation**: We advocate and implement **32-thread virtual subgroups** because correctness and compatibility with standard subgroup algorithm interfaces (like 32-bit ballot masks and lane shuffles) are paramount. This allows developers to write single-codebase GPU algorithms that function correctly on all devices, with the emulation layer handling subgroup lane partitioning transparently.

### Challenge: Uniform Control Flow and Emulated Barriers
WebGPU/WGSL provides no guarantee of lockstep execution and requires explicit synchronization for cross-thread memory visibility. Emulating subgroup operations inside non-uniform control flow (such as subgroup-conditional branches like `if (sg0)`) poses a major dilemma:
* We need barriers (like `workgroupBarrier()`) to synchronize emulated subgroup operations.
* However, WGSL restricts `workgroupBarrier()` to strictly **uniform control flow** across the entire workgroup.

We resolve this dilemma using two different strategies depending on the code pattern:
1. **Barrier-Free Emulation**: For simple subgroup shuffles/ballots within uniform code paths, we omit internal `workgroupBarrier()` calls. This relies on the assumption that the 32 threads in the virtual subgroup are scheduled such that their memory writes are visible without explicit synchronization. Note that this is a spec violation and technically a data race under the WebGPU specification, as WebGPU makes no guarantees about lockstep execution or implicit memory visibility across threads. We do this purely as a pragmatic workaround to avoid uniformity compilation errors on current compilers, but it is not spec-compliant.
2. **Workgroup-Uniform Coordination**: For loop-based primitives (like Decoupled Lookback in `DLDFScan`), we restructure the loop to run workgroup-uniformly. All subgroups enter the loop together and participate in the barriers. Subgroup 0 computes the lookback conditions and broadcasts the results (e.g., loop exit states) to the rest of the workgroup using shared workgroup memory. All threads then read the shared variables to exit the loop in unison, maintaining 100% uniform control flow and spec-compliance.

## Summary

On an Apple M4 GPU, the performance difference for large inputs (1M to 3M elements) is significant:
- **Prefix Scans (`DLDFScan`)**: Emulation (EMU) is **11x to 28x slower** than native subgroups (HW).
- **Radix Sort (`OneSweepSort`)**: Emulation (EMU) is **9x to 37x slower** than native subgroups (HW).

An open question is whether it is better to write different _kernels_ for hw and emu as opposed to what we did: writing different versions of subgroup functions and keeping the same kernel. The answer probably depends on the nature of the kernels. We did not explore the latter alternative at all.

## Further Discussion

### The Impossibility of Statically Spec-Compliant Emulation with a Single Codebase

If we mandate that the emulated (emu) code and hardware-subgroup (hw) code share the exact same WGSL kernel codebase (without conditional compiler templating or code-generation splits), it is impossible to emulate subgroup operations robustly and remain spec-compliant if subgroup operations are called inside conditional branches.

Under the WebGPU/WGSL specification:
* **Memory Visibility Requirement**: Any data exchange between threads via workgroup shared memory (`var<workgroup>`) requires a `workgroupBarrier()` to guarantee memory visibility and avoid data races.
* **Uniformity Restriction**: `workgroupBarrier()` can only be called in control flow that is statically uniform across the entire workgroup.

Under native GPU execution, a subgroup of size \(S_g\) executes instruction-synchronously. Communication via subgroup registers (e.g., shuffling) does not require barriers because the hardware guarantees that all lanes progress in lockstep. In lookback-based chained scans, a thread \(i\) can spin-wait on a memory location to be updated by a preceding tile \(j\):
$$\text{SpinWait}(T_i) \implies \text{Read}(\text{spine}[j])$$

If we attempt to run the exact same logic under emulation (where subgroups are simulated by sharing workgroup memory across the thread group), the execution model shifts from concurrent SIMD lanes to scheduled workgroup threads. If a thread \(i\) enters a spinning loop:
$$\text{while} \ (\text{spine}[j] == \text{Pending}) \ \{ \ \dots \ \}$$
and the thread \(k\) (within the same workgroup) responsible for executing fallback reductions to resolve the stall is scheduled on the same hardware compute unit, the spin-loop of \(i\) will starve \(k\) of execution cycles. Because WebGPU does not guarantee preemption or fair scheduling between threads of a workgroup, the spin-lock becomes permanent, resulting in a GPU hang:
$$\text{Starve}(T_k) \implies \text{Deadlock}$$

To prevent this under emulation, we must introduce workgroup-uniform synchronization primitives (such as `workgroupBarrier()`) inside the lookback loop. However, adding these barriers to a unified codebase forces native subgroup threads to execute workgroup barriers on every iteration, destroying their register-based, barrier-free performance. Thus, a robust, unified, single-codebase execution path for both hardware subgroups and emulation is mathematically and architecturally impossible without sacrificing either performance or correctness.

Thus, to achieve 100% spec compliance, we are forced to separate the WGSL codebases, or place strict design constraints on the developer.

### SwiftShader and CPU Spin-Loop Starvation

A major source of hangs on platforms like GitHub Actions CI is the execution environment itself. Standard cloud CI runners lack physical GPUs and run WebGPU shaders via **SwiftShader** (a CPU-based Vulkan software rasterizer).

Under native GPU execution, even if emulated subgroup helpers (like `subgroupShuffle` or `subgroupBallot` in `wgslFunctions.mjs`) use atomic-based spin-loops:
$$\text{while} \ (\text{atomicLoad}(\text{flag}) < \text{shuffle\_count}) \ \{ \ \dots \ \}$$
the GPU's hardware warp scheduler ensures that threads make progress and eventually update their flags, allowing the loop to terminate.

However, on a CPU-based software renderer like SwiftShader:
1. Workgroups are mapped to standard CPU/OS threads.
2. The CPU has a very limited number of physical cores/hardware execution threads.
3. If a thread \(i\) enters an atomic spin loop waiting for another thread \(j\) in the same workgroup to write its data, thread \(i\) will consume $100\%$ CPU time on its core.
4. Because WebGPU makes no execution progress guarantees and has no concept of thread yielding or OS-level preemption for compute shaders, the spinning thread \(i\) starves thread \(j\) on the CPU core. Thread \(j\) is never scheduled to run, so it can never update the flag, leading to a permanent CPU deadlock/hang.

To prevent this deadlock on CPU-based emulators, any emulation layer relying on shared-memory synchronization must use explicit `workgroupBarrier()` calls rather than atomic spin-loops. However, introducing `workgroupBarrier()` into the emulation functions forces those functions to be called strictly in statically uniform control flow, which conflicts with standard hardware-subgroup usage where subgroup instructions can run inside divergent conditional branches.

### Host-Side JS Templating as a Preprocessor

In our codebase, we utilize Javascript string interpolation (e.g., `${this.useSubgroups ? '...' : '...'}`) to dynamically generate the WGSL shader source before passing it to the WebGPU device. This acts as a host-side preprocessor (similar to `#ifdef` in C/C++).

This templating allows us to keep a single **host-side codebase** (a single Javascript class/file per primitive) while branching the actual **device-side WGSL codebase** during runtime compilation. It enables us to implement loop restructures (like Alternative 2) or generate entirely different shader blocks (like Alternative 3) without forcing the user to load different files or manage separate execution classes.

However, while this solves the usability goal for the library user, it still requires the library developer to design, write, and maintain two separate WGSL code flows within the same template string.

### Alternatives for Spec-Compliant Subgroup Emulation

Below are three potential design strategies to achieve full spec compliance, along with their pros and cons.

#### Alternative 1: Out-of-Branch Execution (Hoisting)
In this approach, the programmer manually restructures the WGSL kernel to hoist all subgroup operations out of conditional control flow. Data is stored in shared memory, processed unconditionally (uniformly) across the entire workgroup, and the result is referenced later inside the branch.

* **Pros**:
  * Statically spec-compliant and compiles on all standard WGSL compilers.
  * Allows a single unified kernel codebase.
* **Cons**:
  * Significantly increases code complexity and programmer overhead.
  * Forces the hardware subgroup path to also use temporary variables and barriers, introducing unnecessary register usage and performance penalties.

#### Alternative 2: Workgroup-Uniform Coordination
Instead of hoisting, we keep the subgroup operations inside the conditional branches but make the branch itself workgroup-uniform. One subgroup computes the branch conditions, writes them to shared memory, and broadcasts them. All subgroups then read the state and enter the branch together.

* **Pros**:
  * Keeps the subgroup operations logically situated inside their loops or conditional blocks.
  * Statically spec-compliant.
* **Cons**:
  * Introduces overhead to compute, write, and broadcast the branch decisions across the entire workgroup.
  * Requires writing separate templated code paths (e.g., using JS string interpolation) to split the hardware and emulated loop structures.

#### Alternative 3: Separate Kernels (Emu vs. HW)
Instead of trying to emulate subgroup built-ins line-by-line in a shared kernel, we write and maintain two completely different shader implementations. The hardware kernel uses native subgroup built-ins, while the emulated kernel uses standard workgroup-level parallel algorithms (such as Kogge-Stone or Hillis-Steele scans) designed natively for uniform workgroup barriers.

* **Pros**:
  * 100% spec-compliant and extremely robust.
  * Delivers maximum possible performance for both hardware subgroup paths (no register/barrier overhead) and emulated paths (optimized workgroup-shared algorithms).
* **Cons**:
  * Increases codebase maintenance overhead as developers must write, test, and maintain two versions of every primitive.

## Case Study: Control Flow Divergence in subgroupAny (OneSweep Sort)

In the implementation of the OneSweep Radix Sort lookback loop, a deadlock was discovered when running under software subgroup emulation. The original code compiled and ran correctly on native hardware subgroup platforms but hung indefinitely under emulation.

### The Problem: Divergent Control Flow

The lookback loop spins waiting for preceding tiles to publish their histograms. The original kernel structure executed `subgroupAny` inside a thread-divergent conditional block:

```wgsl
if (!sgLookbackComplete) {
  if (!lookbackComplete) { // Thread-divergent branch (per-lane status)
    while (spinCount < MAX_SPIN_COUNT) {
      flagPayload = atomicLoad(&passHist[...]);
      if ((flagPayload & FLAG_MASK) > FLAG_NOT_READY) { break; }
      spinCount++;
    }
    // subgroupAny is called ONLY by threads that have NOT completed lookback
    if (subgroupAny(spinCount == MAX_SPIN_COUNT) && (sgid == 0)) {
      wg_incomplete = 1;
    }
  }
}
```

* **On Hardware**: Native hardware subgroups use execution masks to dynamically disable inactive lanes. Threads that have already completed lookback (`lookbackComplete == true`) simply bypass the branch, and the hardware evaluates `subgroupAny` correctly using only the active participating lanes.
* **On Emulation**: Software emulation simulates subgroup barriers using workgroup shared memory barriers and transaction counters. Every thread in the virtual subgroup must execute the helper function uniformly. Because lanes that completed early bypassed the `if (!lookbackComplete)` block, they never reached the barrier inside the emulated `subgroupAny`, causing the participating threads to deadlock waiting for them.

### The Fix: Uniform Execution

To make the kernel emulation-friendly, the divergent `subgroupAny` call was hoisted out of the thread-divergent block while keeping it within the subgroup-uniform block. A subgroup-uniform variable `didSpinTimeout` is initialized and updated inside the branch, then passed to `subgroupAny` uniformly:

```wgsl
if (!sgLookbackComplete) { // Subgroup-uniform branch
  var didSpinTimeout = false;
  
  if (!lookbackComplete) { // Thread-divergent branch
    while (spinCount < MAX_SPIN_COUNT) { ... }
    didSpinTimeout = (spinCount == MAX_SPIN_COUNT);
  }
  
  // Hoisted: Every thread in the subgroup now executes subgroupAny uniformly
  if (subgroupAny(didSpinTimeout) && (sgid == 0)) {
    wg_incomplete = 1;
  }
}
```

Now, all threads in the subgroup execute `subgroupAny` in lockstep. Completed threads participate by passing `didSpinTimeout = false`, and active threads pass their actual spin status. This prevents barrier misalignment and resolves the emulation deadlock.

### Principles for Writing Emulation-Friendly Subgroup Kernels

If you are writing WebGPU kernels intended to be portable to devices without hardware subgroup support (running via a software emulation layer), follow these guidelines:

1. **Treat Subgroup Operations as Collective Barriers**: Always write code under the assumption that subgroup built-ins act as collective operations requiring participation from all threads in the virtual subgroup.
2. **Never Call Subgroup Built-ins Inside Divergent Branches**: Do not place `subgroupShuffle`, `subgroupReduce`, `subgroupAny`, `subgroupAll`, or `subgroupBallot` inside control flow paths that are evaluated per-lane (e.g., `if (!threadCompleted)` or `if (laneId < activeCount)`).
3. **Compute Locally, Participate Uniformly**: If a subgroup operation must evaluate a condition computed inside a divergent branch, compute the condition locally, store it in an initialized variable outside the branch, and call the subgroup operation uniformly.

