---
layout: page
title: "Gridwise Multisplit"
category: gridwise
permalink: /multisplit/
excerpt: "Multisplit: group elements by bucket (up to 256 per pass), keys only, key-value, or with each element's index; shared with MultisplitPerm and used by SuffixArray to make random writes local."
order: 19
---

`Multisplit` groups elements by bucket: the output holds bucket 0's elements, then bucket 1's, and so on. Each element's bucket is a bin function of its key, the same bin functions `Histogram` uses (`RadixBins`, `EvenBins`; see `bins.mjs`). It is a radix sort's single pass, with any bucket function, and it is not stable: the order within a bucket depends on the hardware.

```js
import { Multisplit, RadixBins } from "gridwise";

const ms = new Multisplit({ device, bins: new RadixBins({ shift: 24, bits: 8 }), type: "keyvalue" });
await ms.execute({ keysIn, keysOut, valuesIn, valuesOut, bucketOffsets });
```

**Buffers** are u32 words:

- `keysIn` and `keysOut`;
- with `type: "keyvalue"`, also `valuesIn` and `valuesOut`;
- with `values: "index"`, only `valuesOut`, which receives each element's input index (an argsort by bucket);
- `bucketOffsets` (optional, numBins + 1 u32s): where each bucket starts in the output, then the length.

Keys are read as the bins' datatype (u32, i32 or f32). Every key must fall in a bin; `RangeBins` (which read their levels from a buffer) and bins that leave samples out are not supported yet.

**Options:**

- `bins` (required): at most 256 buckets, the workgroup size;
- `type`: `"keysonly"` (default) or `"keyvalue"`;
- `values: "index"`;
- `length`: multisplit only the first `length` elements; it may change between executes without reallocating;
- `tileK` (elements per invocation per tile: 16 keys only, 8 with values) and `targetWorkgroups` (default 1024).

## How it works

Each workgroup takes a *slot*: a strided set of tiles (tile `s`, `s + M`, `s + 2M`, … for M slots).

1. **Histogram.** Each workgroup counts its tiles' buckets into `hist[bucket][slot]`.
2. **Scan.** `DLDFScan` (exclusive) over `hist` gives each (bucket, slot) its output offset.
3. **Scatter.** Per tile: rank each element within its bucket with workgroup atomics; stage the tile in workgroup memory grouped by bucket; then write each bucket's run to consecutive addresses, so the writes coalesce.

Memory traffic per element: 12 bytes keys only (read the key twice, write it once), 20 bytes key-value.

The pass is `multisplitPassWGSL()`, which `MultisplitPerm` also uses, several passes deep over segments, for random permutations. `Multisplit` is that pass with one segment, input from buffers and output to buffers.

## Making random writes local

A kernel that scatters, `A[dest[i]] = v[i]`, with destinations spread over an array much larger than the cache, pays for a whole cache line read and written back per 4-byte write: on an M1 Max, 0.8 ns per write at 2^24 elements, about 290 bytes of traffic each.

Multisplitting the (dest, v) pairs by the destination's top bits first, then applying them in output order, makes the writes local: each bucket's writes fall in a 1/256 slice of A, written while it is in cache. The cost is about 32 streamed bytes per write instead. Büren et al. (HPDC 2019) do this for suffix arrays. `SuffixArray` tried it for its rank updates and output: 6–16% faster at 2^24 on random and English-like text, but no faster or slower on other inputs, so it was left out for simplicity (see the suffix array page's future directions).
