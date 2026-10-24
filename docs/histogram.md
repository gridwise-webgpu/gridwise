---
layout: page
title: "Gridwise Histogram"
category: gridwise
permalink: /histogram/
excerpt: "Histogram: counts of samples per bin, modeled on CUB's DeviceHistogram, built from bin functions and a workgroup histogram that permutations (and, in spirit, sort) share."
order: 17
---

`Histogram` counts how many samples of an input array fall in each bin. It is modeled on CUB's [`DeviceHistogram`](https://github.com/dmlc/cub/blob/master/cub/device/device_histogram.cuh): equal-width bins (`HistogramEven`) or bins bounded by arbitrary levels (`HistogramRange`), with samples outside every bin left uncounted.

```js
import { Histogram, EvenBins } from "gridwise";

const histogram = new Histogram({
  device,
  bins: new EvenBins({ datatype: "f32", numBins: 256, lower: 0, upper: 1 }),
});
await histogram.execute({ inputBuffer, outputBuffer }); // outputBuffer: 256 u32 counts
```

## Bin functions

The bins are a _bin function_: an object that decides which bin a sample falls in. It plays the same role for a histogram that a [`BinOp`](../binop/) plays for a scan. Each bin function supplies a WGSL function `fn binOf(x: T) -> u32` that returns a bin, or `BIN_NONE` (`0xffffffff`) for a sample in no bin. It also supplies a JavaScript mirror, `binOf(x)`, which returns `-1` for no bin and is used for validation. Its `datatype` is the samples' datatype.

| Bin function | Bins | Notes |
|---|---|---|
| `new EvenBins({ datatype, numBins, lower, upper })` | `numBins` equal-width bins covering [`lower`, `upper`) | For `u32` and `i32`, `upper - lower` must be a multiple of `numBins` (`upper` may be 2^32 for `u32`); use `RangeBins` for uneven integer bins. For `f32`, the bin is `trunc((x - lower) * scale)` with `scale = numBins / (upper - lower)` rounded to f32, computed identically in WGSL and JavaScript. |
| `new RangeBins({ datatype, levels })` | bin _i_ is [`levels[i]`, `levels[i+1]`) | `levels` must be strictly increasing; `numBins = levels.length - 1`. The primitive uploads the levels to a storage buffer, and the WGSL binary-searches it. |
| `new RadixBins({ shift, bits })` | 2^`bits` bins by bits [`shift`, `shift + bits`) of a `u32` | A radix digit, as in radix sort and `MultisplitPerm`; every sample has a bin. |

For floating-point samples, WGSL allows implementations to assume there are no NaNs or infinities, so which bin (if any) such a sample lands in is unspecified.

## Options and buffers

- **`bins`** (required): a bin function. `datatype` defaults to the bins' datatype and must match it.
- **`inputBuffer`**: the samples. Its length is its size divided by the datatype's size; any length works.
- **`outputBuffer`**: `numBins` `u32` counts. Each `execute()` zeroes it first, so it needs `COPY_DST` usage, and it holds that execution's counts. The counts are exact: integer atomics give the same total in any order.
- `workgroupSize` (256) and `itemsPerThread` (16; sets how many workgroups share the input, up to `maxWorkgroups`, 8192).
- `copies`: how many copies of the bins each workgroup keeps, to reduce atomic contention when there are few bins. The default is 4 for up to 256 bins, 2 for up to 1024, and 1 otherwise.

## How it works

Each workgroup counts a grid-strided share of the input into a histogram in workgroup memory, then adds its nonzero counts into the output with atomics, as CUB does when the bins fit in shared memory. When they don't fit (the bins times `copies` times 4 bytes exceed the device's `maxComputeWorkgroupStorageSize`), each sample is counted directly in the output instead; `histogram.workgroupBins` says which path a device and bin count use.

## Ingredients shared with other primitives

The histogram is assembled from two pieces that other primitives use too:

- **Bin functions** (`src/bins.mjs`). `MultisplitPerm` finds each element's bucket with `RadixBins`.
- **`wgHistogram`**, a WGSL helper in `wgslFunctions` and the histogram counterpart of [`wgScan`](../scan-and-reduce/). It declares the workgroup-memory bins and functions to clear them, count a sample (ignoring `BIN_NONE`) and read a bin's total, with optional copies. Callers flush the totals however they need. `Histogram` adds them into a global histogram. `MultisplitPerm` stores each workgroup's counts at its own position, so that one scan turns them into scatter offsets.

OneSweep's `global_hist` kernel computes the same kind of histogram (four radix-digit histograms per read). For now it keeps its own hand-tuned code; moving it onto these pieces is worth doing only if performance is unchanged.

The output is cleared with the `ClearBuffer` action (`new ClearBuffer({ label: "outputBuffer" })`), which zeroes a buffer in order with the rest of a primitive's commands. Other primitives can use it too.
