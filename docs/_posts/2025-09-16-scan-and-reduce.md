---
layout: post
title: "Gridwise Scan and Reduce"
date: 2025-09-16
excerpt: "Understand scan (parallel prefix) and reduce operations - fundamental parallel compute primitives for high-performance applications."
---

[Scan](https://en.wikipedia.org/wiki/Prefix_sum) (parallel prefix) is a fundamental parallel compute primitive useful in both other primitives as well as a wide range of application domains. (The [Wikipedia page](https://en.wikipedia.org/wiki/Prefix_sum#Applications) describes many possible uses of scan.) Here we describe how our implementation works and how it is used in Gridwise. We published a [paper on our scan implementation](https://dl.acm.org/doi/10.1145/3694906.3743326) at SPAA 2025. Here we instead aim for a higher-level, more informal description of our implementation.

## Terminology

Scan inputs an array of *n* data elements and outputs an array of the same size. Output element *i* is the "sum" of the input elements up to element *i*. More generally, that "sum" operator can be any [monoid](https://en.wikipedia.org/wiki/Monoid) (a binary operation with an identity element). If the operator is addition, scan is often called prefix-sum, but we can also compute a prefix-multiplication, prefix-max or -min, or many other operators. For simplicity, we will use "sum" and addition in this article. (Gridwise supports any user-specified monoid.)

We also use the term "reduce", where a reduction of a set of inputs is the "sum" of all of those elements.

Finally, scans have two variants, exclusive and inclusive. Each output element of an *exclusive* scan is the sum of all previous items in the input, not including the current item. (`exclusive_out[i] = sum(in[0:i-1]).`) Each output element of an *inclusive* scan is the sum of all previous items in the input, up to and including the current item. (`inclusive_out[i] = sum(in[0:i]).`)

## GPU Scan Background

On GPUs, scan performance is bound by memory bandwidth. The best GPU scan implementations fully saturate the GPU's memory system. Thus the best GPU scan implementations are those that use algorithms that require the fewest accesses to memory. The "classic" GPU way to compute scan is to divide the input into tiles, compute the reduction of all elements in each tile, compute the prefix sum of those per-tile reductions, then use the values of that per-tile prefix sum as inputs into a blockwise scan of each individual tile. This strategy is called *reduce-then-scan*. The details aren't important; what is important is that for an *n*-element scan, we incur 3*n* memory accesses (*n* for the per-block reduction and 2*n* to read the input/write the output in the blockwise scan).

In 2015, Yan et al. introduced an alternate GPU-scan implementation, [StreamScan](https://dl.acm.org/doi/10.1145/2442516.2442539), which required only 2*n* memory accesses. Like reduce-then-scan, StreamScan computes a reduction for each input tile, but unlike reduce-then-scan, StreamScan then *serializes* the scan of the per-tile values across tile processors. This approach is called a *chained scan*. Each tile processor must wait for its predecessor to compute the reduction of all elements up to and including the predecessor's tile. Then the tile adds its tile reduction to the global reduction and passes it to the next tile's processor, then uses the partial sum to complete the scan of its tile. The key data structure here is the carry chain, which stores the inclusive scan of the tile reductions. The most important aspect of this implementation is that it only requires reading and writing each element once and thus incurs only 2*n* memory references, the theoretical minimum.

(All of the above is covered in great detail in our [SPAA 2025 paper](https://dl.acm.org/doi/10.1145/3694906.3743326).)

In an ideal world, all tile processors could run in lockstep and have equal access to memory bandwidth. However, tiles contend with each other for access to memory, and tile processors may be working on multiple tiles simultaneously. Tiles thus don't run in lockstep; some later tiles may finish before earlier tiles, and thus have to wait for those tiles to finish. This waiting causes performance loss. In 2016, [Merrill and Garland](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back) addressed this performance loss on NVIDIA hardware by enabling stalled tiles to "look back" into the carry chain to fetch the necessary values to compute their carry-chain element. This change ensured that a single stalled tile did not halt the entire scan computation, allowing [their implementation](https://docs.nvidia.com/cuda/cub/index.html) to run at maximum throughput. CUB scan is as fast as a memory copy.

## Decoupled Lookback and Forward Progress Guarantees

NVIDIA GPUs make a *forward progress guarantee*: once a processor begins to process a tile, it is guaranteed to make progress on that tile; the GPU scheduler ensures the processor will be allocated compute time to move forward on the processing of its tile. This guarantee is necessary for the correctness of Merrill and Garland's implementation of decoupled lookback.

Unfortunately, not all GPUs provide this forward progress guarantee. [Apple and ARM GPUs do not.](https://dl.acm.org/doi/10.1145/3485508) At any point in the scan computation, some tiles are computing a result and other tiles are waiting for a previous tile to finish its computation and write it into the carry chain. Without the forward-progress guarantee, the computing tiles may be fully blocked by waiting tiles and never make progress, leading to a completely stalled computation. On Apple hardware, as we found during the development of our scan primitive, this locks up the entire machine.

Deploying a WebGPU implementation that depends on a forward-progress guarantee is thus not a viable option.

## Gridwise's Scan and Reduce Implementations

### Gridwise's Scan

Gridwise implements a chained scan that does not require forward-progress guarantees. It does so in a high-performance way that allows scan to run at full memory bandwidth, even on hardware without forward-progress guarantees. Merrill and Garland's lookback strategy allows a stalled tile processor to look back into the carry chain; we add fallback capability to allow a stalled tile processor to redundantly compute per-tile reductions, and to do so using the full parallelism of the stalled tile processor.

Lookback and fallback were challenging to implement correctly.

### Gridwise's Reduce

The carry chain in a chained scan stores the inclusive scan of the reduction of each tile. The last element in that carry chain is the sum of all input tiles and is thus the reduction of the entire input. A reduce can thus be implemented using the existing scan machinery (and run at full memory bandwidth). A traditional reduce implementation (computing tile reductions in parallel then reducing the tile reductions into a single value) would likely run just as fast, but would require a different implementation, so we have chosen to use the chained-scan implementation as Gridwise's reduce. When configured as a reduction primitive, our scan implementation leaves out any scan-specific computation (e.g., we don't have to compute the final per-tile scan).

### Leveraging Subgroups

WebGPU's optional [subgroups](https://www.w3.org/TR/webgpu/#subgroups) feature enables WebGPU programs to use SIMD instructions within a workgroup. These can deliver significant performance gains for several reasons: they leverage custom hardware for the computation itself; they do not have to route data through workgroup memory; they require one hardware instruction to do what would take many hardware instructions in emulation; and they require fewer barriers. Gridwise has some, but incompletely deployed, [support for emulating SIMD instructions](subgroup-strategy.html). Our initial performance testing indicated that scan was 2.5x slower using subgroup emulation vs. using subgroup hardware.

It is possible that the fastest scan and reduce implementations in the absence of subgroup hardware are not chained ones. We have not investigated this at all. In general, in Gridwise, we expect that subgroup operations are available, and we use them to reduce across subgroups and workgroups, to broadcast information from one thread to others, and to compute local subgroup-sized scans as part of workgroup scans.

## Configuring and Calling Gridwise Scan and Reduce

### Defining the primitive

Declare the scan or reduce primitive as an instance of the `DLDFScan` class.  An example scan declaration:

```js
const datatype = "u32"; // or "i32" or "f32"
const dldfscanPrimitive = new DLDFScan({
  device,
  binop: new BinOpAdd({ datatype }),
  type: "exclusive",
  datatype, // use the "datatype" string defined above
  gputimestamps: true,
});
```

The argument is a JS object and its members may include:

- **device** (required): the GPU device on which this primitive will run.
- **datatype** (required): Scans need to specify the datatype that can be scanned. Currently we support the (WGSL) types "u32", "i32", and "f32", specified as strings. It is definitely possible to support more complex datatypes (e.g., structs), but this would take non-trivial engineering work.
- **binop** (required): the "binary operation" aka a [monoid](https://en.wikipedia.org/wiki/Monoid), specified as a combination of a datatype and a binary operator that operates on that datatype. The core scan implementation is agnostic as to the binary operation; the `binop` supplies that operation. The binop class is described in more detail [here](binop.html).
- **type**: any of `"exclusive"`, `"inclusive"`, or `"reduce"`. Default is `"exclusive"`.
- **datatype** (required): currently, any WGSL scalar primitive type (`"u32"`, `"i32"`, or `"f32"`). Scan and reduce could be extended to other datatypes with some engineering effort.
- **gputimestamps**: enable GPU timestamps for the primitive's kernel calls.

### Configuring the primitive

Once the primitive is _defined_, it must then be _configured_. The primitive knows that it requires an input and output buffer, named `inputBuffer` and `outputBuffer`. (We use our [`Buffer` class](buffer.html) for this.) We configure the primitive by registering data buffers with the primitive. This can be done either with a `primitive.registerBuffer()` call or as an argument to the `execute` call. (The former is preferred if we need to register the buffer(s) once and then call `execute` many times.)

To register a buffer, simply call `primitive.registerBuffer(buffer)`, where `buffer.label` is either `inputBuffer` or `outputBuffer`. The below code creates a `Buffer` then registers it.

```js
const inputLength = 2 ** 20;
testInputBuffer = new Buffer({
  device,
  datatype: "f32",
  length: inputLength,
  label: "inputBuffer",
  createCPUBuffer: true,
  initializeCPUBuffer: true /* fill with default data */,
  createGPUBuffer: true,
  initializeGPUBuffer: true /* with CPU data */,
  createMappableGPUBuffer: false, /* never reading this back */
});
primitdldfscanPrimitiveive.registerBuffer(testInputBuffer);
```

### Calling scan or reduce

Once the primitive is defined and configured, simply call its `execute()` method. 

If you have not yet registered buffers, you can specify them in the argument object as `inputBuffer` and `outputBuffer`.

Other possible arguments (which are timing-specific and thus which you are unlikely to use unless you are benchmarking) are:
- `trials` with an integer argument. This will run the kernel(s) that number of times. Default: 1.
- `enableGPUTiming` with either true or false. If true, please ensure that the device has a set of required features that include `timestamp-query`. Default: false.
- `enableCPUTiming` with either true or false. Default: false.

Note that `execute()` is declared `async`.

```js
await dldfscanPrimitive.execute();
// or, if we want to specify buffers only when execute is called
await dldfscanPrimitive.execute({
  inputBuffer: mySrcBuffer,
  outputBuffer: myDestBuffer,
});
// or (maybe if you're benchmarking)
await dldfscanPrimitive.execute({
  trials: 1,
  enableGPUTiming: false,
  enableCPUTiming: true,
});
```

## Usage and performance notes

Input lengths _must be_ a multiple of 4. Pad the end of your input array with enough identity elements to make this work. (This is because internally, we use `vec4`s for computation.)

Scan has had extensive performance testing and the defaults are fairly stable across different GPUs. The workgroup size, for instance, is set to 256. This particular iteration of the scan kernel has barely been tested with other workgroup sizes and they are unlikely to work out of the box.

If we extended scan to larger datatypes (beyond 32 bits), we expect that workgroup memory consumption would become an issue. We expect we would have to reduce workgroup size accordingly.