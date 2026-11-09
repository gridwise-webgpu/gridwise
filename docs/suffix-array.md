---
layout: page
title: "Gridwise Suffix Array: Design"
category: gridwise
permalink: /suffix-array-design/
excerpt: "Design of SuffixArray: prefix doubling over a full radix sort of the still-tied suffixes, built from OneSweep and DLDFScan, and why it was chosen over the alternatives."
order: 18
---

_Status: first version implemented (`src/suffixarray.mjs`). This page records the decisions behind `SuffixArray` and the performance it should reach; measured results will be added as they come in._

The suffix array of a text T of length n lists the starting positions of T's suffixes in lexicographic order. Its inverse, the rank array (ISA), gives each suffix's position in that order. Together they underlie full-text search (FM-index, pattern counting), the Burrows–Wheeler transform, compression, deduplication and n-gram statistics.

```js
const sa = new SuffixArray({ device, n });              // n bytes of text
await sa.execute({ inputBuffer, outputBuffer });        // text in, SA[0..n) out (u32)
// optional: rankBuffer (n u32s) receives the rank array, rank[i] = 1 + position of suffix i
```

The input holds the n bytes packed four to a u32 word, little-endian, so a `Uint8Array` padded with anything to a multiple of 4 bytes can be written as is. Byte 0 is an ordinary symbol; no sentinel is needed. `execute()` runs one GPU submission per round, so it does not take an encoder, and `trials` must be 1. Afterwards, `sa.rounds` is the number of doubling rounds, `sa.activeCounts` the number of active suffixes before each round, and, with `enableGPUTiming`, `sa.timings` the per-round kernel times.

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
- **One readback per round, overlapped when it pays.** WebGPU has no device-side kernel launch, and OneSweep's length is set on the host. So the host reads the active count after each round: 4 bytes and one `mapAsync`. On an M1 Max each submission and readback cost 2–4 ms beyond its kernels.
  - **Speculation.** Each round's kernels read the actual active count from the GPU (`count[k − 1]`), and the host gives the round a *capacity*: the sort length and dispatch size. Counts only shrink, so the last count the host has read is a valid capacity even if a newer round's count is still on its way. Positions past the actual count get keys that sort after every real key, and the other kernels ignore them.
  - So the host can submit round k + 1 before reading round k's count, hiding the readback and the next round's encoding behind the GPU's work. The price is sorting the spare positions. The host speculates only when the count shrank by less than a quarter in the last round, or is below 2^16; at most one round is in flight beyond the last count read. `speculate: false` turns it off.
- **Sorting a prefix of a buffer.** Each round sorts the first m elements of buffers sized for n, with OneSweep's `length` option; the scans likewise use DLDFScan's `length`. Buffers are allocated once, for n.
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

**Memory.** As written, the working set is about 53n bytes:

- the text (n);
- ranks (4n);
- keys and their sort buffer (16n);
- payloads and theirs (8n), which also hold the active list;
- the active suffixes' ranks (4n);
- two flag and two scan buffers (16n);
- the output (4n).

Reusing the sort's temporary buffers for the flags and scans, and the key buffer for the output, would bring it to about 37n. That is in line with published GPU designs: 2016 hybrid about 46n, Osipov 32n, libcubwt 20.5n.

### Expected performance: a bandwidth model

Every phase is a streaming pass, a sort, or a gather or scatter, so the design should be **bound by memory traffic**: its time should be the bytes it must move at the rate the device moves them, plus a fixed cost per round. `suffixArrayTraffic()` in `src/suffixarray.mjs` computes those bytes from a run's active counts; `misc/benchmark_suffixarray.mjs` compares them with measured kernel times.

**Bytes per element**, counting each element's reads and writes once:

| Phase | Initial round (m = n) | Doubling round (per active suffix) |
|---|---|---|
| keys | read 1 text byte, write key 8 + payload 4: **13** | pack: read active 4, its rank 4, rank[i + h] 4 (random); write 12: **24** |
| sort | 8 + 24 × 8 passes (63-bit keys): **200** | 8 + 24P, P = ⌈2b/8⌉: **152** at n = 2^20, **176** at 2^24 |
| heads | read key 8, write 2 flags: **16** | **16** |
| 3 scans | read 4 + write 4 each: **24** | **24** |
| update | read key, payload, 2 starts (20); write rank (random), keep flag, new rank: **32** | **32** |
| compact | read 16; write active and its rank, ≤ 8: **24** | **24** |
| **total** | **309** | **128 + 24P: 272 at 2^20, 296 at 2^24** |

Plus 8 bytes per symbol for the output: a random scatter, `SA[rank[i] − 1] = i`, or, when the initial round resolves every suffix, a streaming copy of the sorted payload.

Random accesses: one rank gather (pack) and one rank scatter (update) per active suffix per round, plus one scatter per symbol in the output after doubling rounds.

**The model:**

> T ≈ sort bytes / R<sub>sort</sub> + other bytes / BW + gathers · t<sub>gather</sub> + scatters · t<sub>scatter</sub> + (rounds + 2) · t<sub>submit</sub>

with four device rates, measured at each n by `misc/device_rates.mjs` (which the benchmark calls; run on its own, it prints the rates for any GPU):

- R<sub>sort</sub>: what OneSweep achieves on 64-bit key-value data, counting 8 + 24 bytes per element per pass;
- BW: copy bandwidth (read plus write);
- t<sub>gather</sub>, t<sub>scatter</sub>: the cost of one random 4-byte access into an n-element array (it jumps once the array outgrows the caches);
- t<sub>submit</sub>: the fixed cost of a submission and its 4-byte readback (`mapAsync`), plus about P + 9 kernel launches.

The input decides how many suffixes stay active, through the **work ratio** Σ m<sub>r</sub> / n over the doubling rounds:

| Input | Rounds | Work ratio | Bytes per symbol, n = 2^24 |
|---|---|---|---|
| Uniform random bytes | 0 (7 symbols already distinguish almost every suffix) | 0 | 317 |
| Random over 4 symbols | 2 | 1.1 | 630 |
| Natural-language text | 4–8 | 2–4 | 900–1,500 |
| Fibonacci word, all "A" (worst cases) | ⌈log₂(n / 7)⌉ ≈ 21–22 | about 20 | about 6,300 |

A full 64-bit key-value sort moves 200 bytes per element, so typical text costs about as much as 5–7 such sorts, and the worst case about 30.

### Measured: the first run (Apple M1 Max, 365 GB/s copy bandwidth)

The first version, before the two changes below, on a MacBook Pro with an M1 Max, through Dawn and Metal:

| Input, n = 2^24 | Rounds | Work ratio | Wall | GPU | Copy-bandwidth model |
|---|---|---|---|---|---|
| Random bytes | 0 | 0 | 72 ms | 67 ms | 14 ms |
| Random over 4 symbols | 2 | 1.06 | 127 ms | 117 ms | 28 ms |
| English-like text | 6 | 3.50 | 297 ms | 270 ms | 60 ms |
| Fibonacci word | 21 | 19.8 | 1,193 ms | 1,110 ms | 272 ms |
| All "A" | 22 | 20.3 | 784 ms | 704 ms | 278 ms |

The work ratios are as predicted. The times are 3–5 times the copy-bandwidth model, for three reasons, each now a measured rate in the model:

- **The sort runs at about 107 GB/s**, 29% of copy bandwidth, at every size from 2^22 up and on every input. It is about half of the GPU time.
- **Random access is far more expensive than its bytes.** Once the 64 MB rank array no longer fits in the cache, each 4-byte scatter costs about what a whole 128-byte line read and written back would: the output scatter took 13.3 ms at 2^24 (16.8M × 256 bytes at 365 GB/s is 12 ms). At 2^22 the same kernels run 4–6 times faster. All "A" escapes this, because its ranks arrive in nearly sequential order. On English-like text at 2^24, rank gathers and scatters took 47% of GPU time, as much as the sort.
- **Each submission costs about 2–4 ms** beyond its kernels (wall minus GPU time). With 20 rounds this is half the wall time at n = 2^20.

With the measured sort rate and per-access costs in place of copy bandwidth, the model accounts for about 90% of the GPU time (random bytes, 2^24: 31 ms sort, 3.5 ms streaming, 24 ms random access, against 67 ms measured).

**Changes made in response:**

1. **The active list carries each suffix's rank.** The compaction writes the new rank next to the suffix index, so pack reads `rank[i]` as a stream and gathers only `rank[i + h]`. One random access per active suffix per round becomes 8 streamed bytes.
2. **When the initial round resolves every suffix, the output is a copy** of the sorted payload instead of a scatter by rank.
3. **Unchanged ranks are not rewritten.** The first subgroup of each group keeps its rank, so update skips its scatter: on English-like text that is 57% of the rank writes (20% on the Fibonacci word, 12% on random ACGT, measured on the CPU at 2^17).
4. **Speculative rounds** (decision 4) hide the per-round readback when the active count shrinks slowly.

**Next, in order of expected gain:**

1. **Sort rate.** Find what limits OneSweep on 64-bit key-value data on this GPU (tile size under 32 KB of workgroup memory, the lookback, the scatter); the sort is half the time on every input.
2. **Fewer key bits.** Later rounds sort 2b bits; replacing `rank[i]` with a dense id for its group (and dropping the sorted-by-construction high part) would save passes once few groups remain.
3. **Local rank scatters.** Group the update's scatters by destination block before writing (Büren et al.), so each cache line is written once.
4. **Fewer readbacks.** Let OneSweep take its length from a GPU buffer, so several rounds can be submitted together; this matters for repetitive inputs and small n.

**How to measure:** `node misc/benchmark_suffixarray.mjs random256,random4,english,fibonacci,allA 20,22,24` prints the device rates at each size, then per input the rounds, work ratio, wall and GPU time, the overhead per submission, both models (copy bandwidth only, and calibrated with the device rates), and per phase the GPU time, modeled bytes and achieved bandwidth.

### 6. Input and alphabet

- **First version:** byte text, packed four bytes per u32 word (WGSL has no `u8`), with n given explicitly.
- **Symbol encoding:** 9-bit symbols with 0 meaning "past the end", so byte 0 is an ordinary symbol and no sentinel is required.
- **Planned extension:** integer alphabets, such as u32 tokens with a given alphabet size σ. Symbols then take ⌈log₂(σ + 1)⌉ bits, and the initial key packs as many as fit in 64 bits.

### 7. Outputs

- **Default output:** the suffix array.
- **Optional extra output:** the rank array (ISA), which falls out of the construction.
- **Later:** a BWT output and an LCP array, either from Kasai's algorithm parallelized as in Deo and Keely (2013), or from the doubling rounds themselves.

## Verification

- **CPU reference:** `suffixArrayReference()`, prefix doubling with counting sorts in JavaScript (O(n log n)), used up to 2^20 symbols.
- **Linear-time checker:** `checkSuffixArray()` (Burkhardt and Kärkkäinen), on every test and benchmark run.
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
