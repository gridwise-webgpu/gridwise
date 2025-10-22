---
layout: post
title: "Gridwise Sort"
date: 2025-09-16
excerpt: "GPU radix sort implementation using the OneSweep architecture with chained scan and forward-progress guarantees."
---

The dominant approach to GPU sorting is a [radix sort](https://en.wikipedia.org/wiki/Radix_sort) over input keys. In general, radix sorts deliver high performance on GPUs because they require _O(n)_ work for inputs of _n_ elements, because their constituent memory accesses are generally fairly coalesced and thus deliver good memory performance, and because the underlying compute primitives that compose to make the sort are good matches for GPUs.

The specific sort architecture we choose is [OneSweep](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus), developed by Andrey Adinets and Duane Merrill of NVIDIA. Internally, OneSweep uses a chained scan, as does our implementation. The challenges we outlined in our [scan description](scan-and-reduce.html#decoupled-lookback-and-forward-progress-guarantees) with respect to forward-progress guarantees are the same. Our sort implementation employs both lookback and fallback to ensure that it will work on GPUs that lack forward-progress guarantees.

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

Note that the chained scan in onesweep is a chained scan over an entire 256-entry histogram. Lookback on such a large data structure is more complicated than lookback on a single data value, as we see in [our scan implementation](scan-and-reduce.html), because we use an entire workgroup to look back. The additional complexity is that _any_ thread in the lookback may fail to find a ready value, and if this is the case, the entire workgroup must drop into fallback. Thus we have to keep track of both per-thread lookback success as well as per-subgroup lookback success, and only determine lookback is successful if all subgroups report success.

Radix sort implementations (including ours) typically use a ping-pong pair of arrays: on each pass, one array is the input and one array is the output, and on each pass, their roles switch. Because we are sorting 32- or 64-bit keys at 8 bits per pass, this means the input will be overwritten by the output and the primitive's output will be produced in the same buffer as its original input. Overwriting the input is not ideal behavior but is probably preferable to approaches that hide it from the user (by, say, preemptively copying the input into a temporary buffer and copying the temporary input and output at the end of the computation).

## Configuring and Calling Gridwise Sort

### Defining the primitive

Declare the scan or reduce primitive as an instance of the `OneSweepSort` class.  An example scan declaration:

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

Once the primitive is _defined_, it must then be _configured_. The primitive knows that it requires an input/output and temporary buffer, labeled `keysInOut` and `keysTemp`. (We use our [`Buffer` class](buffer.html) for this.) If we are doing a key-value sort, we also require `payloadInOut` and `payloadTemp` buffers, which store the values.) We configure the primitive by registering data buffers with the primitive. This can be done either with a `primitive.registerBuffer()` call or as an argument to the `execute` call. (The former is preferred if we need to register the buffer(s) once and then call `execute` many times.)

To register a buffer, simply call `primitive.registerBuffer(buffer)`, where `buffer.label` is one of the buffers above. The below code creates a `Buffer` then registers it.

```js
const inputLength = 2 ** 20;
testKeysBuffer = new Buffer({
  device,
  datatype: "f32",
  length: inputLength,
  label: "keysInOut",
  createCPUBuffer: true,
  initializeCPUBuffer: true /* fill with default data */,
  storeCPUBackup: true, /* because readback will overwrite the CPU data */
  createGPUBuffer: true,
  initializeGPUBuffer: true /* with CPU data */,
  createMappableGPUBuffer: true, /* we read this back to test correctness */
});
oneSweepSortPrimitive.registerBuffer(testKeysBuffer);
```

### Calling scan or reduce

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

The number of items to sort must be no greater than 2^30. (CUB does the same thing.) We use the two most-significant bits as status bits. It would take a large engineering effort to remove this limitation.

Just as with scan, input lengths _must be_ a multiple of 4. Pad the end of your input array with enough largest-key-value elements to make this work. (This is because internally, we use `vec4`s for computation.)

During its development, sort had extensive performance testing and the defaults are fairly stable across different GPUs. We sort 8 bits per pass and this particular implementation has never been tested with a different number of bits per pass. This could be remedied with engineering effort.