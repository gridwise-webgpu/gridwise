---
layout: page
title: "Gridwise Sort"
category: gridwise
permalink: /sort/
excerpt: "GPU radix sort implementation using the OneSweep architecture with chained scan and forward-progress guarantees."
order: 5
---

The dominant approach to GPU sorting is a [radix sort](https://en.wikipedia.org/wiki/Radix_sort) over input keys. In general, radix sorts deliver high performance on GPUs because they require _O(n)_ work for inputs of _n_ elements, because their constituent memory accesses are generally fairly coalesced and thus deliver good memory performance, and because the underlying compute primitives that compose to make the sort are good matches for GPUs.

The specific sort architecture we choose is [OneSweep](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus), developed by Andrey Adinets and Duane Merrill of NVIDIA. Internally, OneSweep uses a chained scan, as does our implementation. The challenges we outlined in our [scan description](../scan-and-reduce/#decoupled-lookback-and-forward-progress-guarantees) with respect to forward-progress guarantees are the same. Our sort implementation employs both lookback and fallback to ensure that it will work on GPUs that lack forward-progress guarantees.

## Our Sort Implementation

At its heart, radix sort computes a permutation of its input values and then performs the permutation. Computing the entire permutation would be intractable (the size of the intermediate data structures would be enormous), so typically a radix sort makes several passes over the input, each time computing a permutation for a subset of input bits (this subset is called a "digit"). OneSweep begins with the least significant bits, as do we. Our implementation considers 8 bits per pass, meaning each digit can take on 2^8 = 256 possible values. We thus require 4 passes to sort 32-bit keys. On each pass, we classify the key into one of 256 "buckets" based on its digit value. The permutation we want to compute will always place keys with lower digit values before keys with higher digit values.

As with scan, we divide the input into equal-sized "tiles" and assign one workgroup to each tile.

Computing this permutation means computing a "destination address" for each key---to where in the output will this key be written? That address is the sum of:

1. The number of keys that fall into a bucket before my bucket
2. The number of keys that are in my bucket and are processed in a previous tile
3. The number of keys that are in my bucket and are processed earlier than
   my key within my tile

We compute (1) by constructing a global histogram of all bucket sizes (kernel: `global_hist`, which constructs a separate histogram for each digit) and then running exclusive-sum-scan on these histograms (kernel: `onesweep_scan`).

We compute (2) by first computing the number of keys per bucket within my histogram and then using a chained sum scan to retrieve the sum of the sizes of the previous workgroup's buckets (kernel: `onesweep_pass`). We add in (1) at the start of this chained scan, so we actually chained-scan (1) + (2). The chained scan requires lookback to look at the results of this scan, and if the hardware does not offer forward-progress guarantees, the chained scan also requires fallback to redundantly compute this value. We compute (3) in `onesweep_pass` as well, but it is workgroup-local only; it does not participate in the chained-scan.

Given this computed address per key, we could directly scatter each key to its location in global memory. However, to improve memory coalescing, we first write keys into workgroup memory and scatter from there. This puts neighboring keys next to each other in workgroup memory and significantly improves the throughput of the global scatter.

Note that the chained scan in onesweep is a chained scan over an entire 256-entry histogram. Lookback on such a large data structure is more complicated than lookback on a single data value, as we see in [our scan implementation](../scan-and-reduce/), because we use an entire workgroup to look back. The additional complexity is that _any_ thread in the lookback may fail to find a ready value, and if this is the case, the entire workgroup must drop into fallback. Thus we have to keep track of both per-thread lookback success as well as per-subgroup lookback success, and only determine lookback is successful if all subgroups report success.

Radix sort implementations (including ours) typically use a ping-pong pair of arrays: on each pass, one array is the input and one array is the output, and on each pass, their roles switch. Which array holds the sorted result therefore depends on the number of passes. With 8 bits per pass, sorting full 32- or 64-bit keys takes an even number of passes (4 or 8), so the output lands back in the input buffer, overwriting it. Sorting fewer key bits (see `KEY_BITS` below) can take an odd number of passes, and then the output lands in the temporary buffer. Gridwise follows [CUB's `DoubleBuffer`](https://nvidia.github.io/cccl/cub/api/structcub_1_1DoubleBuffer.html) here: rather than always copying the result to a fixed place, the primitive tells you where it is (see [Finding the sorted output](#finding-the-sorted-output)). Either buffer may be overwritten.

## Configuring and Calling Gridwise Sort

### Defining the primitive

Declare the sort primitive as an instance of the `OneSweepSort` class. An example sort declaration:

```js
const datatype = "u32"; // or "i32" or "f32"
const oneSweepSortPrimitive = new OneSweepSort({
  device,
  datatype, // use the "datatype" string defined above
  type: "keysonly",
  direction: "ascending",
});
```

Gridwise OneSweep supports all combinations of:

- `datatype`: `u32`, `i32`, `f32`, `u64`. Internally, OneSweep converts non-unsigned-int keys into unsigned-int keys that respect the original order, sorts as if the keys were unsigned ints, and then reverses the conversion when writing the keys into the output.
- `type`: `keysonly`, `keyvalue`. A key-value sort has an array of keys and also an array of values where each value is associated with its corresponding key in the keys array. Default: `keysonly`.
- `direction`: `ascending`, `descending`. The default is `ascending` (sort low to high), but we support sorting in the other direction as well.

### Configuring the primitive

Once the primitive is _defined_, it must then be _configured_. The primitive knows that it requires an input/output and temporary buffer, labeled `keysInOut` and `keysTemp`. (We use our [`Buffer` class](../buffer/) for this.) If we are doing a key-value sort, we also require `payloadInOut` and `payloadTemp` buffers, which store the values. We configure the primitive by registering data buffers with the primitive. This can be done either with a `primitive.registerBuffer()` call or as an argument to the `execute` call. (The former is preferred if we need to register the buffer(s) once and then call `execute` many times.)

To register a buffer, simply call `primitive.registerBuffer(buffer)`, where `buffer.label` is one of the buffers above. The below code creates a `Buffer` then registers it.

```js
const inputLength = 2 ** 20;
const testKeysBuffer = new Buffer({
  device,
  datatype, // must match the primitive's datatype
  length: inputLength,
  label: "keysInOut",
  createCPUBuffer: true,
  initializeCPUBuffer: true /* fill with default data */,
  storeCPUBackup: true /* because readback will overwrite the CPU data */,
  createGPUBuffer: true,
  initializeGPUBuffer: true /* with CPU data */,
  createMappableGPUBuffer: true /* we read this back to test correctness */,
});
oneSweepSortPrimitive.registerBuffer(testKeysBuffer);
```

### Finding the sorted output

After `execute()`, the sorted keys are in one of the two key buffers, and the sorted payload (for a key-value sort) is in the matching payload buffer. Ask the primitive which:

- `primitive.outputKeysBufferName` is `"keysInOut"` or `"keysTemp"`;
- `primitive.outputPayloadBufferName` is `"payloadInOut"` or `"payloadTemp"`;
- `primitive.outputIsInTemp` is `true` when the output is in the temporary buffers.

These are valid as soon as the primitive is constructed, since they depend only on the number of passes. Read the result with, for example, `primitive.getBuffer(primitive.outputKeysBufferName)`.

If you would rather have the output in a fixed place, construct the primitive with `copyOutputToTemp: true`. The output is then always in `keysTemp` (and `payloadTemp`), so sort behaves like any other primitive with separate input and output buffers. That costs a buffer-to-buffer copy when the pass count is even. When it is odd, the output is already in the temporary buffers and nothing is copied. The default is `false`: no copy, and you read the buffer the primitive names.

```js
const sort = new OneSweepSort({ device, datatype: "u32", type: "keyvalue" });
await sort.execute({ keysInOut, keysTemp, payloadInOut, payloadTemp });
const sortedKeys = sort.getBuffer(sort.outputKeysBufferName);       // keysInOut here: 4 passes
const sortedValues = sort.getBuffer(sort.outputPayloadBufferName);  // payloadInOut
```

### Sorting fewer key bits

If only the low bits of `u32` keys vary (for instance, keys known to be below 2^20), pass `KEY_BITS` to the constructor to sort only those bits. That saves passes: `KEY_BITS: 20` takes 3 passes instead of 4. Because the pass count can then be odd, use `outputKeysBufferName` (or `copyOutputToTemp: true`) to find the result.

### Sorting a prefix of the buffers

Set `length` (in the constructor, or on the primitive between executes) to sort only the first `length` keys and payloads of the buffers; the rest is neither read nor sorted. The pipelines do not depend on `length`, so one primitive can sort a shrinking working set inside buffers allocated once:

```js
const sort = new OneSweepSort({ device, datatype: "u64", type: "keyvalue", KEY_BITS: 48, copyOutputToTemp: true });
for (const m of workingSetSizes) {
  sort.length = m;            // m <= the buffers' length
  await sort.execute({ keysInOut, keysTemp, payloadInOut, payloadTemp });
}
```

### 64-bit keys

With `datatype: "u64"`, keys are WGSL `vec2u` values, 8 bytes each: element `[0]` (`x`) holds the low 32 bits and element `[1]` (`y`) the high 32 bits. A full key takes 8 passes, low word first. In JavaScript, fill and read the buffers as `BigUint64Array`s, which use the same layout. Payloads in a key-value sort are still `u32`.

`KEY_BITS` counts the low bits of the whole 64-bit value, so `KEY_BITS: 48` sorts all 32 bits of `x` and the low 16 bits of `y`, in 6 passes. To sort on two fields, pack them into one 64-bit value with the more significant field in the higher bits. For example, to sort by a pair of 24-bit numbers (a, b), first by a and then by b, store `(a << 24) | b` and use `KEY_BITS: 48`: that is `x = (a << 24) | b` (keeping the low 32 bits) and `y = a >> 8`. Putting `a` in `y` and `b` in `x` instead would need `KEY_BITS: 64`.

A 64-bit pass moves 8-byte keys, so it costs more than a 32-bit pass, and each workgroup handles a smaller tile (12 rather than 15 keys per thread, to fit workgroup memory).

### Calling sort

Once the primitive is defined and configured, simply call its `execute()` method.

If you have not yet registered buffers, you can specify them in the argument object as `keysInOut`, `keysTemp`, etc.

Other possible arguments (which are timing-specific and thus which you are unlikely to use unless you are benchmarking) are:

- `trials` with an integer argument. This will run the kernel(s) that number of times. Default: 1.
- `enableGPUTiming` with either true or false. If true, please ensure that the device has a set of required features that include `timestamp-query`. Default: false.
- `enableCPUTiming` with either true or false. Default: false.

Note that `execute()` is declared `async`.

```js
await oneSweepSortPrimitive.execute();
// or, if we want to specify buffers only when execute is called
await oneSweepSortPrimitive.execute({
  keysInOut: testKeysBuffer,
  keysTemp: testKeysTempBuffer,
});
// or (maybe if you're benchmarking)
await oneSweepSortPrimitive.execute({
  trials: 1,
  enableGPUTiming: false,
  enableCPUTiming: true,
});
```

## Usage and performance notes

The number of items to sort must be no greater than 2^30. (CUB's OneSweep implementation has the same restriction.) We use the two most-significant bits as status bits. It would take a large engineering effort to remove this limitation.

Just as with `DLDFScan`, input lengths _must be_ a multiple of 4. Pad the end of your input array with enough elements to make this work, using the largest key value for an ascending sort or the smallest for a descending sort, so the padding sorts to the end. (This is because internally, we use `vec4`s for computation.)

During its development, sort had extensive performance testing and the defaults are fairly stable across different GPUs. We sort 8 bits per pass and this particular implementation has never been tested with a different number of bits per pass. This could be remedied with engineering effort.

Sort works with subgroups of any width up to 128 lanes. Its warp-level multisplit (`WLMS`) tracks which lanes of a subgroup share a digit with one bit per lane. On devices whose subgroups are at most 32 wide (the adapter's `subgroupMaxSize`; Apple and NVIDIA, for instance), that mask is one `u32`. Only on devices whose subgroups can be wider (AMD RDNA, 32 to 64) is the wider version generated, which uses a `vec4<u32>` mask and a few more registers. The choice is made when the shader is generated, so 32-wide devices run exactly the 32-bit code.

Ascending key-value sort is stable. Elements with equal keys preserve their original order from the input.

Descending key-value sort's ordering is more complicated to specify. Gridwise does not guarantee a particular tie-breaking order, only that keys will be properly ordered. Gridwise's validation has no dependence on any descending tie-breaking convention.

### Keys per thread (the tile size)

Each pass works on tiles of `KEYS_PER_THREAD` × 256 keys. The default is 15 for 32-bit keys and 12 for 64-bit keys, reduced if the tile and the pass's other workgroup memory would not fit the device's limit. `KEYS_PER_THREAD` can also be given explicitly. Each subgroup's digit histogram is staged in the tile, so the tile must hold (256 / subgroup width) × 256 entries. That means at least 8 keys per thread with 32-lane subgroups, or with emulated subgroups; with fewer, the sort is silently wrong, so OneSweep refuses them. A tile that does not fit workgroup memory is refused too.

### Measured performance and where a pass's time goes

These measurements come from `misc/onesweep_probe.mjs`, which sweeps `KEYS_PER_THREAD`, checks every sort, and compares a pass with a copy kernel. They were taken at 2^24 keys, through Dawn in Node.

| GPU | Copy kernel | 32-bit keys, per pass | 64-bit key-value, per pass |
|---|---|---|---|
| Apple M1 Max | 360–370 GB/s | 15 keys/thread (default): 2.6 ms; 11: 1.8 ms (1.4× faster) | 12 (default) is the fastest valid tile, about 100 GB/s (29% of copy) |
| NVIDIA V100 | 808 GB/s | 15 (default) is the fastest: 0.585 ms, 230 GB/s (28%) | 8–12 all about 1.31 ms, 308 GB/s (38%); 14 and 16 are slower |

So the best tile size depends on the GPU, and the defaults are right for the V100 but not for 32-bit keys on the M1 Max.

To see where a 64-bit key-value pass's 1.31 ms goes on the V100 (against 0.50 ms at copy bandwidth), an experimental build removed parts of the pass and timed it. These builds give wrong output and are kept only on the `onesweep-instruments` branch:

| Part of the pass | Time | Share |
|---|---|---|
| Moving the bytes at copy bandwidth | 0.50 ms | 38% |
| Work inside each tile: ranking keys by digit, per-subgroup histograms, staging keys and then payloads through workgroup memory | about 0.5 ms | 38% |
| The scattered global writes | about 0.26 ms | 20% |
| The decoupled lookback (fallbacks were 4–6% of tiles; raising the spin limit did not help) | about 0.05 ms | ≤5% |

For 32-bit keys the lookback mattered more (about 22% of a pass) and the scatter much less (3%).

### Possible improvements not made

These could make a pass faster, but each adds complexity to code that is hard to get right across hardware, and the gain may not carry over between GPUs. They are recorded here rather than implemented.

- **Stage keys and payloads in non-atomic workgroup memory.** The tile is staged through `wg_warpHist`, an `array<atomic<u32>>`, because the same memory also holds the per-subgroup histograms, which need atomics. So every staged key and payload costs atomic loads and stores: two each for a 64-bit key. WGSL cannot alias one workgroup variable as both atomic and non-atomic.
  - The `buffer_view` language extension (WGSL; [proposal](https://github.com/gpuweb/gpuweb/blob/main/proposals/buffer-view.md), [issue 5338](https://github.com/gpuweb/gpuweb/issues/5338)) adds `var<workgroup> scratch: buffer<N>` and `bufferView`/`bufferArrayView`. One workgroup allocation can then be viewed as different non-atomic types, for example `array<u32>` and `array<vec2u>`. Dawn supports it, and the `webgpu` package for Node lists it in `wgslLanguageFeatures`.
  - But a view may not be or contain an atomic type, because Metal cannot cast non-atomic memory to atomic. We confirmed this: Dawn rejects `bufferArrayView<array<atomic<u32>>>` with "return type of bufferArrayView cannot contain an atomic type".
  - Using it would therefore mean building the histograms without workgroup atomics (each subgroup owning its histogram, written by one lane per digit after the subgroup match), or giving the histograms their own atomic array. Either way, a staging buffer can then be non-atomic and shared between keys and payloads.
- **One staging round for key and payload together**, instead of keys, a barrier, then payloads: one fewer trip through workgroup memory and one fewer barrier for key-value sorts.
- **Per-vendor tile sizes:** for example 11 keys per thread for 32-bit keys on Apple GPUs. This is simple, but needs measurements on more GPUs before a default changes, and must not slow the others.
