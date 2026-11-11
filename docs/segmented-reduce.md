---
layout: page
title: "Gridwise Segmented Reduce and SpMV"
category: gridwise
permalink: /segmented-reduce/
excerpt: "SegmentedReduce: a load-balanced reduction of each segment of a CSR-style array, with the schedule (thread-mapped or merge-path) separate from the work, which the caller writes as JS functions that return WGSL; SpMV is a SegmentedReduce."
order: 20
---

`SegmentedReduce` reduces each segment of a CSR-style array, with the load-balancing schedule kept separate from the work. `SpMV` (y = Ax, CSR, f32) is a `SegmentedReduce` that supplies the work.

It follows the programming model of Osama, Porumbescu and Owens, [A Programming Model for GPU Load Balancing](https://escholarship.org/uc/item/9nq090zg) (PPoPP 2023):

- **atoms** are units of work of equal cost, such as a matrix's nonzeros;
- **tiles** are groups of atoms of any size, such as its rows, given by an offsets array: tile t holds atoms `[offsets[t], offsets[t + 1])`;
- a **schedule** decides which invocation handles which atoms, independent of what the work is.

For every tile, `SegmentedReduce` combines its atoms' values with a binop and emits the result.

```js
import { SegmentedReduce, BinOpAdd } from "gridwise";

const spmv = new SegmentedReduce({
  device,
  schedule: "mergepath",
  binop: new BinOpAdd({ datatype: "f32" }),
  tiles: rows,
  atoms: nnz,
  buffers: [
    { name: "indices", type: "array<u32>" },
    { name: "values", type: "array<f32>" },
    { name: "x", type: "array<f32>" },
    { name: "y", type: "array<f32>", access: "read_write" },
  ],
  atom: (a) => `values[${a}] * x[indices[${a}]]`,
  emit: (t, v) => `y[${t}] = ${v};`,
});
await spmv.execute({ offsets, indices, values, x, y });
```

That is all of `SpMV` (`src/spmv.mjs`), which also offers `rows` and `nnz` as names for `tiles` and `atoms`.

## The work as JS functions that return WGSL

The paper's C++ interface hands the schedule the work as lambdas, which the compiler inlines. WGSL has no lambdas, templates or function pointers, so Gridwise uses the nearest thing JavaScript offers: functions that return WGSL.

- **`atom: (atom, tile) => expression`** gives an atom's value.
- **`emit: (tile, value) => statement`** consumes a tile's result. Without it, results go to `outputBuffer`.

The schedule calls each function with the names of its own WGSL variables and places the result in its kernel. A schedule never knows what the work is, and the work never knows the schedule. Changing schedules is one option, as in the paper.

This is code generation by string, and it is deliberately so: JavaScript composes it freely today, and the patterns it needs (functions as parameters, generic element types) are what a future WGSL might support directly. Generation happens once per pipeline, so it costs nothing per execute.

**Buffers** the functions use are declared in `buffers` (`{ name, type, access }`, access `"read"` unless given) and passed to `execute()` by those names, with `offsets` (tiles + 1 u32s).

**Options:**

- `binop` (required): any `BinOp`; it may be non-commutative, since atoms are combined in order;
- `tiles`, `atoms` (required): the number of tiles, and `offsets[tiles]`; both may change between executes;
- `atom` (required), `emit`, `buffers`: above;
- `schedule`: `"mergepath"` (default) or `"thread"`;
- `itemsPerThread`: merge-path steps per invocation (default 7).

## Schedules

**`thread`** gives each invocation one tile at a time, striding over the tiles. It is balanced only when tiles are about the same size. One long tile holds up the whole kernel, because a single invocation walks all of it.

**`mergepath`** (Merrill and Garland, SC 2016) walks the merge of two lists: the tiles' ends and the atom indices. Each step of the path either consumes an atom or ends a tile, so there are tiles + atoms steps, and every invocation takes `itemsPerThread` of them, whatever the tile sizes. A binary search over the offsets finds where each invocation's steps start.

Tiles split between invocations are put back together in two stages:

1. **Within a workgroup.** Each invocation emits the tiles it both starts and ends, and carries a partial value for the tile it is in at the end. A segmented inclusive scan of the carries, keyed by tile, gives each invocation what earlier invocations carried into the tile it started in.
2. **Between workgroups.** A tile that began in an earlier workgroup is not emitted by the workgroup that ends it. That workgroup records its value instead, and a second kernel (`fixup`) adds what the earlier workgroups carried in that tile, walking back over them, and emits it.

So every tile is emitted exactly once, with its whole value, and `emit` can be any WGSL: it is never asked to combine partial results.

### Fit to WebGPU

- **Barriers in uniform control flow.** WGSL requires `workgroupBarrier()` where control flow is provably uniform across the workgroup. Merge-path's per-invocation loops have no barriers; its only barriers are in the workgroup scan, whose loop bounds are constants. A block-mapped schedule (a workgroup per tile, scanning atom counts per tile) would need its loop bounds made uniform with `workgroupUniformLoad`.
- **No float atomics.** CUDA merge-path SpMV can finish split rows with `atomicAdd` on floats. WebGPU atomics are 32-bit integers only, so the fixup kernel combines split tiles instead, which also keeps the result deterministic.
- **No subgroups needed.** Both schedules use only workgroup memory and barriers, so they run the same on every device, including those without subgroups.

## Testing and measuring

The regression tests (`segred`) check `SegmentedReduce` (u32 add, i32 max) and `SpMV` against the CPU on several shapes: uniform tiles, a power law, a few tiles spanning many workgroups, one tile, all tiles empty. They cover both schedules and 1, 7 and 16 items per invocation, and change `tiles` and `atoms` between executes.

`misc/benchmark_spmv.mjs` times both schedules on generated matrices (`uniform`, `rmat` scale-free graphs, `denserow` with one row holding half the nonzeros) and on Matrix Market files (`--mtx=`, e.g. from the [SuiteSparse collection](https://sparse.tamu.edu)), checks every result, and reports GFLOP/s and bandwidth against a copy kernel's.

## Possible improvements not made

- **Coalesced loads.** Each invocation reads its atoms consecutively, so a subgroup's loads are strided by `itemsPerThread`. CUB stages a tile's atoms through workgroup memory first.
- **One search per workgroup.** Each invocation searches the global offsets. CUB searches once per workgroup, then within a tile of row ends in workgroup memory.
- **A parallel fixup.** A tile spanning w workgroups makes one invocation walk back over w carries. A segmented scan of the carries would bound that for very long tiles.
- **Atoms from the GPU.** The caller passes `atoms` (offsets[tiles]) to size the grid. An indirect dispatch could read it on the GPU instead, for offsets produced by an earlier kernel.
- **More schedules.** Workgroup-mapped and subgroup-mapped schedules, as in the paper.
