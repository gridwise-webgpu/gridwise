---
layout: page
title: "Gridwise Random Permutations"
category: gridwise
permalink: /random-permutations/
excerpt: "MultisplitPerm: an unbiased, optionally bit-reproducible GPU random permutation built from a multisplit and a per-group shuffle."
order: 14
---

`MultisplitPerm` writes a uniformly random permutation of 0..n-1 into an output buffer:

```js
import { MultisplitPerm } from "gridwise";

const perm = new MultisplitPerm({ device, n, seed });
await perm.execute({ outputBuffer }); // outputBuffer holds n u32s
```

## How it works

Each index gets a random key. The top G bits of the key choose a group, with G about log2(n / 2048), so groups average about 2,048 elements. One or more multisplit passes of at most 8 bits each move every element into its group. A final kernel then shuffles each group in workgroup memory, one workgroup per group.

Each multisplit pass is a histogram kernel, an exclusive scan of the histogram (a `DLDFScan`, run with the `ExecutePrimitive` action described below), and a scatter kernel. The scatter ranks elements within a digit using workgroup atomics, so the order inside a bucket depends on the hardware's atomic arrival order. That order never reaches the output: the final shuffle draws its own randomness, and every arrangement of a group is equally likely whatever order the group arrives in.

With `reproducible: true`, the final shuffle sorts each of its sub-buckets before shuffling them. The output then depends only on which elements are in each group, so it is bit-identical on every GPU and in every run. `perm.reference()` computes the same permutation on the CPU, and `validate()` compares against it.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `n` | (required) | permutation length |
| `seed` | `0x2545f491` | 32-bit seed |
| `groupSize` | 2048 (1024 when reproducible) | mean group size for the final shuffle |
| `split` | `"auto"` | how the group bits divide into passes: `"even"`, `"wide-first"`, `"narrow-first"`, `"passes:3"`, or explicit widths such as `"5+5+5"` |
| `carry` | `"keys"` | what the passes move: hashed keys (inverted at the end), `"indices"`, or `"pairs"` (index plus random word; Philox only) |
| `reproducible` | `false` | bit-identical output on every GPU |
| `rng` | `null` | `null` for the keyed hash, or `{ key: [lo, hi], rounds }` for Philox4x32 |
| `phaseLimit` | all | run only the first steps of `perm.phases` (for per-phase timing) |

The fastest `groupSize` and `split` depend on the GPU. `tuneMultisplitPerm({ device, n })` times the candidates on the device and returns the best options. Reproducible output depends on the group size, so for reproducible runs, keep the group size fixed across devices and tune only the split.

## Pieces other primitives can use

Writing this primitive added several general-purpose pieces to Gridwise:

- **`ExecutePrimitive` action** (see [Composing Primitives](../composing-primitives/)). A primitive's `compute()` can run another primitive on the same encoder, binding the child's named buffers to its own: `new ExecutePrimitive({ primitive: this.scan, buffers: { inputBuffer: "counts", outputBuffer: "offsets" } })`. Create the child once (for example, in the constructor), not in `compute()`. GPU timing counts and records the child's kernels together with the parent's.
- **`wgExclusiveScan` in `wgslFunctions`.** An exclusive scan of one value per invocation across a workgroup, the scan counterpart of `wgReduce`. The subgroup version handles any subgroup size from 4 up. The version for devices without subgroups uses only workgroup barriers, not emulated subgroups, so it does not depend on forward progress.
- **`random.mjs`.** WGSL random bits (a keyed invertible hash and Philox4x32) with bit-exact JavaScript mirrors for validation.
- **`destroy()` frees internal buffers.** `BasePrimitive.destroy()` now destroys the GPU buffers the primitive allocated itself with `AllocateBuffer`. Buffers the caller passed in are left alone.
