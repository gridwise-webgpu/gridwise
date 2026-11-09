---
layout: page
title: "Gridwise Suffix Array: Design"
category: gridwise
permalink: /suffix-array-design/
excerpt: "Design of SuffixArray: prefix doubling over a full radix sort of the still-tied suffixes, built from OneSweep and DLDFScan, and why it was chosen over the alternatives."
order: 18
---

_Status: design. This page records the decisions behind `SuffixArray` before it is written; it will become the primitive's documentation._

The suffix array of a text T of length n lists the starting positions of T's suffixes in lexicographic order. Its inverse, the rank array (ISA), gives each suffix's position in that order. Together they underlie full-text search (FM-index, pattern counting), the Burrows–Wheeler transform, compression, deduplication and n-gram statistics.

```js
const sa = new SuffixArray({ device, n });              // n symbols
await sa.execute({ inputBuffer, outputBuffer });        // text in, SA[0..n) out (u32)
```

## Summary of the design

`SuffixArray` uses **prefix doubling**. Round by round, it sorts the suffixes by their first h symbols, doubling h each round. Each round has three steps:

- a **full radix sort of only the suffixes still tied with another**, keyed by the pair (rank of i, rank of i + h) packed into one 64-bit key;
- **two scans** that turn the sorted keys into new ranks;
- a **compaction** that drops every suffix whose rank is now unique.

Almost all the work is done by two existing Gridwise primitives: `OneSweepSort` (u64 keys, `KEY_BITS`, u32 payload) and `DLDFScan` (max and add). The new code is four small kernels and a host loop.

The design follows four priorities, in order: reuse of Gridwise primitives, simplicity, fit to WebGPU, then performance. The sections below take each decision in turn.

## The algorithm

**Ranks.** A suffix's rank is 1 + the position in the final suffix array where its group of tied suffixes starts. Positions past the end of the text have rank 0, which sorts before every real rank. A rank therefore needs b = ⌈log₂(n + 1)⌉ bits. Because ranks are positions in the final order, sorting by them never moves a group out of its final range. That is why a full sort is correct and a segmented sort is not needed.

**Initial round.** Each suffix's first 7 symbols are packed into one 64-bit key, 9 bits per symbol. A byte c is stored as c + 1, and 0 means "past the end", so the end of the text sorts below every byte, including byte 0. The keys are sorted with the suffix index as the payload (`KEY_BITS` 63, 8 passes). This resolves the suffixes to h = 7.

**Each later round**, over the m active (still tied) suffixes:

1. **Pack** (new kernel). For active suffix i, the key is `(rank[i] << b) | rank[i + h]` (2b bits) and the payload is i. This gathers `rank[i + h]` at a scattered position.
2. **Sort** (`OneSweepSort`, `datatype: "u64"`, `KEY_BITS: 2b`, key-value) the first m elements. For n = 2^24 that is 48 bits, or 6 passes.
3. **Find heads** (new kernel). At each sorted position p, record p if its key's high part (`rank[i]`) differs from position p − 1, giving the start of the old group. Separately, record p if its whole key differs, giving the start of the new subgroup.
4. **Scan** (`DLDFScan`, `BinOpMax`, inclusive), twice. These give, for every p, its group start g and subgroup start q in the sorted list.
5. **Update** (new kernel). The new rank is `rank[i] + (q − g)`, scattered to `rank[i]`. A suffix is retired if its subgroup holds one element: p is a head and p + 1 is a head or the end.
6. **Compact** (`DLDFScan` add, plus a scatter kernel). The suffixes not retired become the next round's active list, and their count is copied to a small readback buffer.
7. **Host:** read the count. If it is 0, stop; otherwise set h to 2h and go to step 1.

**Output.** `SA[rank[i] − 1] = i`, one scatter.

**Correctness.** Every member of a non-singleton group is active, so a group occupies consecutive positions in the sorted active list, and its final range starts at its rank. That makes `rank[i] + (q − g)` the subgroup's start, which is the standard prefix-doubling invariant in final-position coordinates. Burkhardt and Kärkkäinen's linear-time check verifies the result in tests: SA is a permutation, and adjacent suffixes compare correctly by (first symbol, rank of the next suffix).

## Decisions

### 1. Formulation: prefix doubling, not skew, induced sorting or the 2016 hybrid

| Formulation | What it would need in Gridwise | Verdict |
|---|---|---|
| Prefix doubling, full sort of the active suffixes (chosen) | OneSweep, DLDFScan, 4 small kernels | Reuses what exists |
| Wang, Baxter & Owens 2016 hybrid (skew first step + prefix doubling with segmented sort + skew merge) | A merge-based segmented sort and a merge primitive, both new | Fastest published single-GPU design (CUDA, 2016), but adds two large primitives; revisit later |
| Osipov 2012 (Manber–Myers with a 32-bit key) | A 64-bit payload, which OneSweep lacks; re-sorts all n suffixes each round | Narrower key, but more work and a payload Gridwise does not have |
| Skew / DCX (Deo & Keely 2013; Haag et al. 2025) | Wide-tuple sorts, a merge, recursion with host round trips | Work does not depend on the input's repeats, but needs the most new machinery |
| Induced sorting (SA-IS, divsufsort) | — | Its inducing scan is sequential: the fastest CPU method, unsuited to GPUs |
| Partitioned prefix doubling without a rank array (Cascade, 2026) | Partitioning, rank messages, host streaming | For inputs beyond device memory; a later extension of this design |

The chosen design is, in effect, one partition of Cascade's design, or a single-device version of Flick and Aluru's distributed prefix doubling.

### 2. Reuse: one sort, two scans

- **`OneSweepSort` with 64-bit keys** does the heavy lifting. Its `KEY_BITS` option sorts exactly the 2b bits a rank pair needs. The pair is packed as one 64-bit value (see [64-bit keys](../sort/#64-bit-keys)), so a full sort orders by `rank[i]` and then by `rank[i + h]`. The u32 payload carries the suffix index.
  - The 64-bit path has regression tests: full keys, key-value, `KEY_BITS` crossing the word boundary, ties, and emulated subgroups.
- **`DLDFScan`** with `BinOpMax` turns head flags into group and subgroup starts. With `BinOpAdd` it turns keep flags into compaction offsets.
- **The new kernels** (pack, heads, update, compact-scatter) are elementwise or a single gather or scatter, written with the usual Gridwise kernel conventions: declared buffers, `UniformStruct` parameters and `computeLinearizedGridParametersSplit`.
- **Not used:** a segmented sort, a merge and a multisplit. Gridwise has no segmented sort or merge, and the design avoids needing them. Multisplit appears only as an optional optimization (decision 5).

### 3. Simplicity

- **No segmented sort.** Ranks are final-order positions, so sorting the active suffixes globally keeps each group inside its range. Segment boundaries come out of the head flags afterwards.
- **No recursion and no skew frame.** That means no merge and no change of coordinate system between levels. The skew frame from the 2016 hybrid saves a third of the work, but needs a merge. It is listed as future work, to try only if measurements justify it.
- **The same fixed sequence every round:** pack, sort, heads, two scans, update, compact. Only m and h change.

### 4. Fit to WebGPU

- **32-bit WGSL only.** WGSL has no 64-bit integers. The 64-bit keys are OneSweep's existing `vec2u` type, and the new kernels pack and unpack the high and low words with shifts.
- **One readback per round.** WebGPU has no device-side kernel launch, and OneSweep's length is set on the host. So the host reads the active count after each round: 4 bytes and one `mapAsync`, typically 10–25 times per input.
  - Over-provisioning rounds with indirect dispatch, as `ShunPerm` does, could remove the readback for the small kernels but not for the sort. It is not worth the complexity until a profile shows readback time matters.
- **Sorting a prefix of a buffer.** Each round sorts the first m elements of buffers sized for n. OneSweep currently sorts its whole buffer, so this needs a small OneSweep change: a length shorter than the buffer (prerequisite P1 in the plan).
- **Workgroup memory.** OneSweep sizes its tile to `maxComputeWorkgroupStorageSize`. The new kernels use little or no workgroup memory.
- **Buffer limits.** The largest buffers hold the 8n-byte keys.
  - Under WebGPU's default `maxStorageBufferBindingSize` of 128 MiB, n is limited to 2^24 = 16M symbols. `SuffixArray` therefore requests the adapter's maximum, as Gridwise's Node device setup does.
  - OneSweep itself is limited to 2^30 elements.
- **Subgroups.** OneSweep needs subgroups of 16–32 lanes or emulated subgroups.
  - Narrower hardware subgroups, as on lavapipe, are not supported yet. That is a known OneSweep gap.
  - The emulated path depends on spin-waits that some schedulers do not support ([#89](https://github.com/gridwise-webgpu/gridwise/issues/89)).
  - `SuffixArray` inherits both limitations, and adds no subgroup code of its own.
- **No forward-progress assumptions** beyond those of OneSweep's and DLDFScan's lookback, which already have fallbacks.

### 5. Performance

Per round, the cost is roughly proportional to m, the number of active suffixes:

- a gather of `rank[i + h]` (random reads);
- a ⌈2b/8⌉-pass sort, moving about 24 bytes per element per pass;
- three scans over m;
- a scatter of new ranks (random writes);
- the compaction.

**Retirement** makes m shrink quickly on typical text. The 2016 work found that dropping finished suffixes matters more than any other choice in prefix doubling.

**Known costs, to measure and then attack in this order:**

1. **Random rank gathers and scatters.** Büren et al. (2019) measured these at about 60% of prefix-doubling time. Their remedy fits Gridwise: partially sort the writes by destination block (a multisplit), then write each block contiguously from workgroup memory.
2. **Key width.** Early rounds sort 2b bits. Replacing `rank[i]` in the key with a dense id for its active group shrinks the key once few groups remain. The initial symbol count (7) is also a tunable.
3. **Repetitive inputs**, such as long runs, periodic strings or duplicated documents, need about log₂ of the longest repeat in rounds, and keep m large. This is prefix doubling's known weak spot; difference-cover methods avoid it. It will be measured explicitly, starting with an all-"A" input.

**Memory.** Without aliasing, the working set is about 41n bytes:

- the text (n);
- ranks (4n);
- keys and their sort buffer (16n);
- payloads and theirs (8n);
- scan space (8n);
- the output (4n).

Reusing the sort's temporary buffers for the scans and the output brings it to about 30n. That is in line with published GPU designs: 2016 hybrid about 46n, Osipov 32n, libcubwt 20.5n.

### 6. Input and alphabet

- **First version:** byte text, packed four bytes per u32 word (WGSL has no `u8`), with n given explicitly.
- **Symbol encoding:** 9-bit symbols with 0 meaning "past the end", so byte 0 is an ordinary symbol and no sentinel is required.
- **Planned extension:** integer alphabets, such as u32 tokens with a given alphabet size σ. Symbols then take ⌈log₂(σ + 1)⌉ bits, and the initial key packs as many as fit in 64 bits.

### 7. Outputs

- **Default output:** the suffix array.
- **Optional extra output:** the rank array (ISA), which falls out of the construction.
- **Later:** a BWT output and an LCP array, either from Kasai's algorithm parallelized as in Deo and Keely (2013), or from the doubling rounds themselves.

## Verification

- **CPU reference:** a straightforward prefix-doubling implementation in JavaScript, fine up to about 10^6 symbols.
- **Linear-time checker:** for larger inputs and for benchmark runs.
- **Test inputs:**
  - random text over 2, 4 and 256 symbols;
  - English text, source code and DNA;
  - n = 1, 2 and 7;
  - a non-multiple of 4;
  - all "A";
  - periodic strings ("ab" repeated, "abc" repeated);
  - text containing byte 0;
  - the Manzini and Silesia corpora (benchmarks).
- **Devices:** Apple M-series, NVIDIA, the AMD simulator, and the emulated-subgroup path.

## References

- U. Manber and G. Myers. Suffix arrays: a new method for on-line string searches. SODA 1990.
- N. J. Larsson and K. Sadakane. Faster suffix sorting. Theoretical Computer Science 387(3), 2007.
- V. Osipov. Parallel suffix array construction for shared memory architectures. SPIRE 2012.
- M. Deo and S. Keely. Parallel suffix array and least common prefix for the GPU. PPoPP 2013.
- L. Wang, S. Baxter and J. D. Owens. Fast parallel skew and prefix-doubling suffix array construction on the GPU. Concurrency and Computation: Practice and Experience 28(12), 2016.
- P. Flick and S. Aluru. Parallel distributed memory construction of suffix and longest common prefix arrays. SC 2015.
- F. Büren, D. Jünger, R. Kobus, C. Hundt and B. Schmidt. Suffix array construction on multi-GPU systems. HPDC 2019.
- M. Haag, F. Kurpicz, P. Sanders and M. Schimek. Fast and lightweight distributed suffix array construction. ESA 2025.
- J. Khan and P. Pandey. Cascade: GPU-accelerated out-of-core suffix array construction for LLM corpora. arXiv:2610.15254, 2026.
- S. Burkhardt and J. Kärkkäinen. Fast lightweight suffix array construction and checking. CPM 2003.
