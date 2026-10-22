---
layout: page
title: "Composing Primitives"
category: gridwise
permalink: /composing-primitives/
excerpt: "ExecutePrimitive: building a primitive that runs another primitive as one of its steps, with shared buffers, ordering and GPU timing."
order: 13
---

Some primitives need another primitive as one of their steps. A primitive's `compute()` can include an `ExecutePrimitive` action, which runs another primitive at that point, on the same command encoder, with the other primitive's buffers bound to this primitive's buffers by name.

## Motivation

The first primitive that needed this was `MultisplitPerm`, a random permutation built from multisplit passes (it is in a separate pull request). Each pass is three steps:

1. a histogram kernel counts how many elements go to each bucket;
2. an exclusive scan of those counts gives every bucket its output offset;
3. a scatter kernel moves each element to its bucket.

Steps 1 and 3 are short kernels that belong to the new primitive. Step 2 is a scan, and Gridwise already has the hardest-won primitive it could want for that: `DLDFScan`, with its decoupled lookback and fallback, its handling of different subgroup sizes, and its emulation path. The new primitive should use that scan, not a copy of it.

Before `ExecutePrimitive`, there were three ways to do that, and none was satisfactory:

- **Let the caller chain the primitives.** As [Ordering and Synchronization Guarantees](../ordering-guarantees/) describes, an application can record several primitives into one encoder. That works for an application, but the result is not a primitive. Every caller has to know the composite's internal steps and allocate its internal buffers (histograms, offsets), and the composite cannot be executed, timed, tested or benchmarked as one unit.
- **Copy the scan's kernels into the new primitive.** This duplicates the most delicate code in Gridwise, and the copy would not get later fixes to the scan.
- **Override `execute()` and call the child's `execute({ encoder })` from it.** This records the right passes, but it bypasses `compute()`, so the rest of `BasePrimitive` cannot see the child. GPU timing breaks: the timing helper is sized from the parent's kernels only, and the child's `execute()` resets the shared helper. The child also cannot use buffers the parent allocates with `AllocateBuffer`, because those do not exist until execution starts.

`ExecutePrimitive` makes the child one of the parent's actions, so the parent stays a single primitive with the usual `execute()`, timing and buffer handling.

## Usage

```js
import { BasePrimitive, Kernel, AllocateBuffer, ExecutePrimitive, DLDFScan, BinOpAdd } from "gridwise";

class HistogramThenScan extends BasePrimitive {
  constructor(args) {
    super(args);
    this.knownBuffers = ["inputBuffer", "offsets"];
    /* create the child once, here, not in compute() */
    this.scan = new DLDFScan({
      device: this.device,
      binop: new BinOpAdd({ datatype: "u32" }),
      type: "exclusive",
      datatype: "u32",
    });
  }
  compute() {
    return [
      new AllocateBuffer({ label: "counts", size: this.countsBytes }),
      new Kernel({ /* histogram: inputBuffer -> counts */ }),
      new ExecutePrimitive({
        primitive: this.scan,
        /* the child's buffer name: this primitive's buffer name */
        buffers: { inputBuffer: "counts", outputBuffer: "offsets" },
        label: "scan of counts",
      }),
    ];
  }
  destroy() {
    super.destroy();
    this.scan.destroy();
  }
}
```

- **Create the child once**, typically in the constructor, and return the same object from every `compute()` call. Kernel counting and execution must see the same child.
- **`buffers`** maps each of the child's buffer names to one of this primitive's buffer names: one it was given by its caller, or one it allocates with `AllocateBuffer`.
- **`enable`** (optional, default `true`) works as it does for `Kernel`: a disabled action is skipped and not counted.
- **`destroy()`** of the parent does not destroy its children. A primitive that owns children should destroy them in its own `destroy()`, as above.

## How it works

- **Order.** The child's actions run at the position of the `ExecutePrimitive` action in the parent's list, on the parent's encoder, each kernel in its own compute pass as usual. WebGPU therefore orders the child's reads and writes after the parent's earlier passes and before its later ones, exactly as for kernels the parent records itself.
- **Buffers.** Before the child runs, each mapped buffer is registered in the child under the child's name. If the parent has not allocated a mapped buffer yet, it allocates it at that moment from its own `AllocateBuffer` action, so that the child can size itself from it. This matters when kernels are counted for GPU timing, which happens before any action runs.
- **GPU timing.** `countKernels()` counts the enabled `Kernel` actions and adds each enabled child's `countKernels()`, recursively. `execute()` uses it to size the timing helper. The child then records into the parent's timing helper instead of setting up its own, so `getTimingResult()` on the parent returns one time per kernel, the child's included, in execution order. See [Timing Strategy](../timing-strategy/).
- **Trials.** The parent's `trials` argument is passed to the child, so every kernel, the child's included, is dispatched that many times.
- **`compute()` is called once per `execute()`**, and the same action list is used for counting kernels and for execution.

Nesting is recursive: a child may itself contain `ExecutePrimitive` actions. Only one level of nesting is tested so far (the `ScanPlusOne` regression test in `examples/regression.mjs`).
