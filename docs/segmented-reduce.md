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
- `tileStart: (t) => expression`: tile t's first atom, for t from 0 to `tiles`, in place of an `offsets` buffer (the paper's iterators over a sparse format);
- `wgsl`: more WGSL declarations, such as functions that `atom`, `emit` or `tileStart` call;
- `schedule`: `"mergepath"` (default) or `"thread"`;
- `itemsPerThread`: merge-path steps per invocation (default 4, the fastest on an M1 Max, a V100 and an A100).

## Schedules

**`thread`** gives each invocation one tile at a time, striding over the tiles. It is balanced only when tiles are about the same size. One long tile holds up the whole kernel, because a single invocation walks all of it.

**`mergepath`** (Merrill and Garland, SC 2016) walks the merge of two lists: the tiles' ends and the atom indices. Each step of the path either consumes an atom or ends a tile, so there are tiles + atoms steps, and every invocation takes `itemsPerThread` of them, whatever the tile sizes. A binary search over the offsets finds where each invocation's steps start.

### How merge-path puts split tiles back together

Merge-path balances the work by cutting the path into equal pieces, wherever the cuts fall. So a tile can be cut between invocations, and a long tile between many workgroups. Each piece of a tile computes a partial value, and the partial values have to be combined, in order, into one value per tile before it is emitted. That happens in two stages: within a workgroup, then between workgroups.

**Within a workgroup.** Each invocation walks its `itemsPerThread` steps:

- a tile it both starts and ends is whole: it emits it right away;
- the tile it ends first may have started in an earlier invocation, so it holds that partial value back;
- the tile it is in when its steps run out is unfinished: its partial value is the invocation's *carry*.

Tiles only increase along the path, so the invocations that carry the same tile sit next to each other. A segmented inclusive scan of the carries in workgroup memory, keyed by tile, gives each invocation the sum of what the invocations before it carried in its tile. The invocation that ends that tile adds it in front of its held-back value and emits the tile. The scan's loop bounds are constants, so its barriers satisfy WGSL's uniformity rules.

**Between workgroups.** The first tile a workgroup ends may have started in an earlier workgroup, whose partial values this workgroup cannot see: workgroups don't wait for each other (WebGPU makes no promise that they run in any order, or at the same time). So the workgroup doesn't emit that tile. It records it as its *head* (a flag and the value so far), and records what it carries out of its last tile. (Workgroup 0 has nothing before it, so it emits its first tile.) What remains is to add, in front of each head, the carries of the earlier workgroups in that tile.

Here is an example with five workgroups and tiles A to E. Tile B is long: it starts in workgroup 0 and ends in workgroup 3. In SpMV the tiles are rows, so b0 is the part of row B's dot product computed in workgroup 0.

| Workgroup | Starts in | Emits itself | Head (tile it ends first, value so far) | Carries out |
|---|---|---|---|---|
| 0 | A | A | (none: workgroup 0 emits A) | B: b0 |
| 1 | B | | (none: ends nothing) | B: b1 |
| 2 | B | | (none: ends nothing) | B: b2 |
| 3 | B | C | B: b3 | D: d3 |
| 4 | D | E | D: d4 | (past the end): identity |

B's value is b0 + b1 + b2 + b3, and D's is d3 + d4. Each head needs the *run* of carries in its tile that ends just before it: workgroup 3's head needs carries 0–2, and workgroup 4's needs carry 3.

**The fixup is a segmented reduce.** That last sentence describes a segmented reduce, so `SegmentedReduce` does it with another `SegmentedReduce`:

- **Atoms:** the workgroups' carries, here b0, b1, b2, d3 and the identity.
- **Tiles:** one per workgroup. Tile w holds the run of carries in the tile that workgroup w-1 carried out. That run ends at carry w-1, and it starts where the carried tiles first equal carry w-1's tile. The carried tiles are nondecreasing, so a binary search finds that start. Where workgroup w carries the same tile as w-1, the run hasn't ended, and tile w is empty.
- **Its emit:** for each workgroup w that has a head, it adds the run's sum *in front of* w's head value, in place.

In the example, the fixup's tile starts (its `tileStart`) are 0, 0, 0, 0, 3, 5, so:

| Fixup tile (workgroup) | Its atoms (carries) | Its emit |
|---|---|---|
| 0, 1, 2 | none | (no heads) |
| 3 | 0–2: b0, b1, b2 | head 3 = (b0 + b1 + b2) + b3 = B |
| 4 | 3–4: d3, identity | head 4 = (d3 + identity) + d4 = D |

The last tile also takes the last workgroup's carry, which is past the end of the tiles and holds the identity, so it changes nothing. Then one more kernel emits each head through the caller's `emit`: B from workgroup 3, D from workgroup 4.

**Why this is the right structure:**

- **It is load-balanced at every level.** The fixup's atoms and tiles go through the merge-path schedule too, so a tile spanning 30,000 workgroups costs the same as 30,000 short tiles. The first version instead had each head walk back over its run with one invocation. On a matrix with one row holding half of 2^26 nonzeros, that walk crossed 33,000 workgroups and took half the time (7.4 of 14.9 ms on an M1 Max).
- **It recurses.** The fixup splits its own runs across its workgroups and has its own fixup, until one workgroup suffices. A level of W workgroups has W tiles and W atoms, so its path has 2W steps, and with 256 invocations of 4 steps each the next level has W / 512 workgroups. A path of 2^28 steps (tiles plus atoms) takes two levels of fixup.
- **It reuses the abstraction.** The fixup is written exactly as a caller would write any `SegmentedReduce`: `atom` reads a carry, `tileStart` is the binary search, and `emit` writes into the head. Only the buffers differ.
- **It keeps order.** Each run's carries are combined in order, and in front of the head value, so the binop need not be commutative.
- **It binds few buffers.** A fixup level reads and writes only its parent's carry buffers, never the caller's, so the number of buffers a kernel binds doesn't grow with depth. (WebGPU guarantees only 8 storage buffers per shader stage.)
- **Every kernel can be repeated.** The fixup's `emit` writes each head's whole value to a slot of its own, never back into the head value it reads, so running any kernel twice gives the same result. (`benchmark()` repeats kernels in place, and a first version that added into the head in place got wrong answers there.)
- **Every tile is emitted exactly once, with its whole value.** So `emit` can be any WGSL: it is never asked to combine partial results, and no atomics are needed.

### Fit to WebGPU

- **Barriers in uniform control flow.** WGSL requires `workgroupBarrier()` where control flow is provably uniform across the workgroup. Merge-path's per-invocation loops have no barriers; its only barriers are in the workgroup scan, whose loop bounds are constants. A block-mapped schedule (a workgroup per tile, scanning atom counts per tile) would need its loop bounds made uniform with `workgroupUniformLoad`.
- **No float atomics.** CUDA merge-path SpMV can finish split rows with `atomicAdd` on floats. WebGPU atomics are 32-bit integers only, so the fixup combines split tiles instead, which also keeps the result deterministic.
- **No subgroups needed.** Both schedules use only workgroup memory and barriers, so they run the same on every device, including those without subgroups.

## Testing and measuring

The regression tests (`segred`) check `SegmentedReduce` (u32 add, i32 max) and `SpMV` against the CPU on several shapes: uniform tiles, a power law, a few tiles spanning many workgroups, one tile, all tiles empty. They cover both schedules, 1, 7 and 16 items per invocation, three levels of fixup (one tile of 5 million atoms, one item per invocation), and change `tiles` and `atoms` between executes.

`misc/benchmark_spmv.mjs` times both schedules on generated matrices (`uniform`, `rmat` scale-free graphs, `denserow` with one row holding half the nonzeros) and on Matrix Market files (`--mtx=`, e.g. from the [SuiteSparse collection](https://sparse.tamu.edu)), checks every result, and reports GFLOP/s and bandwidth against a copy kernel's.

## Measured performance

`node misc/benchmark_spmv.mjs uniform,rmat,denserow 18,20,22 --items=2,4,7`, on an Apple M1 Max (355 GB/s copy bandwidth), an NVIDIA V100 (809 GB/s) and an A100 (PCIe, 1,351 GB/s), merge-path with 4 steps per invocation. Every result was checked against a CPU SpMV.

GFLOP/s (2 per nonzero) at 2^22 rows and 67M nonzeros, thread-mapped → merge-path:

| Matrix | Longest row | M1 Max | V100 | A100 |
|---|---|---|---|---|
| uniform (16 per row) | 16 | 8.1 → 17.5 | 8.8 → 17.5 | 18.6 → 38.8 |
| R-MAT (scale-free, 52% of rows empty) | 160,302 | 1.6 → 19.2 | 2.3 → 18.4 | 2.6 → 39.9 |
| one row with half the nonzeros | 33,554,440 | 0.01 → 17.7 | 0.01 → 17.5 | 0.01 → 39.0 |

That is 2.0–2.2× faster on the uniform matrix, 8–15× on R-MAT, and 1,500–2,800× with one dense row (thread-mapped takes 9.6–15.4 s there; merge-path 3.4–7.7 ms).

- **Merge-path's speed does not depend on the matrix's shape.** On each GPU it runs within 10% of the same GFLOP/s on all three matrices, at every size from 2^18 to 2^22 rows. Thread-mapped runs at the speed of its longest row.
- **Merge-path also wins on the balanced matrix** on all three GPUs at 2^22. Neighboring invocations' loads are `itemsPerThread` elements apart, against a row (16 elements) apart for thread-mapped. Of 2, 4 and 7 steps, 4 was fastest or within 2% of it everywhere. (At 2^18 and 2^20 on the A100, the uniform matrix runs the same either way.)
- **The recursive fixup removed the cost of long rows.** With the first version's walk-back fixup, the dense-row matrix took 9.2 ms on the A100 (5.8 ms of it fixup) and 14.9 ms on the M1 Max (7.4 ms); now 3.4 and 7.6 ms, the same as the uniform matrix.
- **The fixup costs a few small kernels: about 30 µs.** Each level is a short merge-path kernel and a heads kernel. At 2^18 that is 7–25% of a 0.12–0.43 ms SpMV, where the walk-back fixup cost 3–8 µs on matrices without long rows; at 2^22 it is under 1%.
- **Large matrices are bound by the random reads of x.** Each nonzero reads x at a random column. The random-read cost that `misc/device_rates.mjs` measures at 2^22 (V100 0.107 ns, A100 0.052 ns per read) times 67M nonzeros is 7.2 and 3.5 ms, against 7.7 and 3.5 ms measured for merge-path on the uniform matrix. So at this size, merge-path SpMV runs as fast as its reads of x allow, and loading the atoms better would not help.

## Possible improvements not made

- **Coalesced loads.** Each invocation reads its atoms consecutively, so a subgroup's loads are strided by `itemsPerThread`. CUB stages a tile's atoms through workgroup memory first. This could help only where x fits in cache (small matrices): large ones are bound by the reads of x.
- **A cheaper fixup for small problems.** The recursive fixup's few kernels cost about 30 µs whatever the matrix. A single-kernel fixup when no tile spans many workgroups would save most of that on small matrices.
- **One search per workgroup.** Each invocation searches the global offsets. CUB searches once per workgroup, then within a tile of row ends in workgroup memory.
- **Atoms from the GPU.** The caller passes `atoms` (offsets[tiles]) to size the grid. An indirect dispatch could read it on the GPU instead, for offsets produced by an earlier kernel.
- **More schedules.** Workgroup-mapped and subgroup-mapped schedules, as in the paper.
