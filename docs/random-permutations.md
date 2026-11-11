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
| `carry` | `"keys"` (`"indices"` with Philox) | what the passes move: hashed keys (inverted at the end; keyed hash only), `"indices"`, or `"pairs"` (index plus random word; Philox only) |
| `reproducible` | `false` | bit-identical output on every GPU |
| `rng` | `null` | `null` for the keyed hash, or `{ key: [lo, hi], rounds }` for Philox4x32 |
| `phaseLimit` | all | run only the first steps of `perm.phases` (for per-phase timing) |

The fastest `groupSize` and `split` depend on the GPU. `tuneMultisplitPerm({ device, n })` times the candidates on the device and returns the best options. Reproducible output depends on the group size, so for reproducible runs, keep the group size fixed across devices and tune only the split.

## ShunPerm: parallel Fisher-Yates

`ShunPerm` is Shun, Gu, Blelloch, Fineman and Gibbons's parallel Fisher-Yates ("Sequential Random Permutation, List Contraction and Tree Contraction are Highly Parallel", SODA 2015). Its output is bit-identical to sequential Fisher-Yates with the same random swap targets, on any GPU and in any run; `reference()` computes it on the CPU.

```js
import { ShunPerm } from "gridwise";
const perm = new ShunPerm({ device, n, seed });
await perm.execute({ outputBuffer });   // n u32s
await perm.roundsUsed();                // rounds needed; Infinity if maxRounds ran out
```

The iterations run in rounds (reserve both locations of each pending swap with `atomicMax`, then commit the swaps that own both). The number of rounds is about 3 log2 n with high probability and only known on the GPU, so `ShunPerm` records `maxRounds` rounds (default `ceil(3 log2 n) + 16`) and sizes each round with an indirect dispatch: a one-invocation kernel writes the round's workgroup count from the number of iterations still pending, with no trip to the CPU, and rounds after the last dispatch no workgroups. Its kernels cannot be repeated in place, so `trials` must be 1. Options: `n`, `seed`, `rng` (Philox, as for `MultisplitPerm`), `maxRounds`.

## Keyed bijections: FeistelPerm, SwapOrNotPerm, MitchellPerm

These write `perm[i] = f(i)` for a pseudorandom bijection `f` of `[0, n)` chosen by the seed. There are no atomics and no data movement besides one 4-byte write per element. With `gather: true` they write `out[i] = in[f(i)]` instead (64-bit values by default; `valueType` sets the WGSL type). The output depends only on `n` and the seed, and `reference()` computes it on the CPU.

```js
import { FeistelPerm, SwapOrNotPerm, MitchellPerm } from "gridwise";
const perm = new FeistelPerm({ device, n, seed, rounds: 8 });
await perm.execute({ outputBuffer });                       // n u32s
const thrust = new MitchellPerm({ device, n, seed, gather: true });
await thrust.execute({ inputBuffer, outputBuffer });       // n vec2u values each
```

- **`FeistelPerm`**: a balanced Feistel network on two `h`-bit halves, with a `triple32` (default) or one-multiply `mulshift` round function. Options: `rounds` (default 24) and `roundFunction`.
- **`SwapOrNotPerm`**: Hoang, Morris and Rogaway's swap-or-not cipher ("An Enciphering Scheme Based on a Card Shuffle", CRYPTO 2012). It works directly on `[0, n)`. Option: `rounds` (default `6 ceil(log2 n)`).
- **`MitchellPerm`**: Mitchell, Stokes, Frank and Holmes ("Bandwidth-Optimal Random Shuffling for GPUs", ACM TOPC 2022), ported from NVIDIA Thrust's `thrust::shuffle`. It gives the same output as Thrust, bit for bit, for the same round keys.
  - `mode: "scan"` (default) does what Thrust does: evaluate `f` on its power-of-two domain and compact the values below `n` in index order. Thrust does this in one decoupled-lookback pass. That pass needs forward progress between workgroups, which WebGPU does not promise. So this version uses three steps: count the valid values per tile, run `DLDFScan` on the counts, then evaluate `f` again and write.
  - `mode: "walk"` uses cycle walking in one kernel instead.

When `f`'s domain is larger than `n`, `FeistelPerm` and `MitchellPerm`'s walk mode restrict it to `[0, n)` by cycle walking: apply `f` until the value is below `n`. This always ends, because `i`'s cycle contains `i`.

None of these is uniform: a keyed bijection reaches at most 2^(key bits) of the n! permutations. Their quality rests on `f` being a good pseudorandom permutation. They are the fastest way to get a permutation that looks random.

A new keyed bijection is a subclass of `BijectionPerm` that supplies four things:

- `bijectionWGSL()`: a WGSL `fn bijection(x: u32) -> u32`;
- `bijectionJS(x)`: its bit-exact JavaScript mirror;
- `roundKeys`: a `Uint32Array`;
- `parameters`: a `UniformStruct` that starts with `n`, and its values.

`BijectionPerm` provides the buffers, the cycle-walking kernel, gather, `reference()` and `validate()`.

## Pieces other primitives can use

Writing this primitive added several general-purpose pieces to Gridwise:

- **`ExecutePrimitive` action** (see [Composing Primitives](../composing-primitives/)). A primitive's `compute()` can run another primitive on the same encoder, binding the child's named buffers to its own: `new ExecutePrimitive({ primitive: this.scan, buffers: { inputBuffer: "counts", outputBuffer: "offsets" } })`. Create the child once (for example, in the constructor), not in `compute()`. GPU timing counts and records the child's kernels together with the parent's.
- **`wgScan` in `wgslFunctions`.** An exclusive or inclusive scan of one value per invocation across a workgroup, for any binop: the scan counterpart of `wgReduce`. The subgroup version handles any subgroup size from 4 up. The version for devices without subgroups uses only workgroup barriers, not emulated subgroups, so it does not depend on forward progress.
- **`wgHistogram` in `wgslFunctions`, and bin functions in `bins.mjs`.** The histogram pass counts each element's digit (`RadixBins`) into a workgroup histogram; the [`Histogram`](../histogram/) primitive is built from the same two pieces.
- **`random.mjs`.** WGSL random bits (a keyed invertible hash and Philox4x32) with bit-exact JavaScript mirrors for validation. See [Random Numbers](../random-numbers/).
- **Indirect dispatch.** A `Kernel` with `indirect: "args"` takes its workgroup counts from a GPU buffer, written by an earlier kernel with the `indirectDispatch` WGSL helper (which keeps each dimension within the device's limit). `ShunPerm` sizes each round this way.
- **`destroy()` frees internal buffers.** `BasePrimitive.destroy()` now destroys the GPU buffers the primitive allocated itself with `AllocateBuffer`. Buffers the caller passed in are left alone.
