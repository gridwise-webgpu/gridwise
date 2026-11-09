---
layout: page
title: "Gridwise WebGPU Object Caching Strategy"
category: gridwise
permalink: /webgpu-object-caching-strategy/
excerpt: "How Gridwise caches WebGPU objects to improve performance by avoiding expensive object recreation."
order: 9
---

## Gridwise WebGPU Object Caching Strategy

This document outlines the caching strategy used for WebGPU objects within Gridwise. Creating WebGPU objects is not free and is potentially expensive. Caching created objects so that they can be reused potentially helps performance. The downsides of caching are that caching itself is not free and that the WebGPU back end may do its own caching. In Gridwise, caching is enabled by default but can be disabled (by instantiating a primitive with the argument `webgpucache: "disable"`; `webgpucache: "enable"` turns it back on). The setting applies to the device's whole cache, so it affects every primitive on that device.

### Cacheable WebGPU Objects

The following WebGPU objects are currently cached by our library:

- `GPUShaderModule`
- `GPUPipelineLayout`
- `GPUBindGroupLayout`
- `GPUComputePipeline`
- `GPUBindGroup`

Of these, `GPUShaderModule` is potentially independent of the `GPUDevice`, while the others are dependent on the specific `GPUDevice` instance.

## Cache Implementation

Every primitive shares a `__deviceToWebGPUObjectCache`, which is a `WeakMap` that maps a `GPUDevice` to its corresponding cache. Each device's cache contains several individual caches for different object types. These are regular `Map` objects that map a generated key to the WebGPU object.

The available caches are:

- `pipelineLayouts`
- `bindGroupLayouts`
- `computeModules`
- `computePipelines`
- `bindGroups`

Internally, each of these caches can be individually enabled or disabled (`new WebGPUObjectCache({ enabled: [...] })`); the primitive's `webgpucache` argument enables or disables all of them.

Here is a simplified code representation of the cache structure:

```js
export class BasePrimitive {
  static __deviceToWebGPUObjectCache = new WeakMap();
  // ... inside a method ...
  BasePrimitive.__deviceToWebGPUObjectCache.set(
    this.device,
    new WebGPUObjectCache()
  );
}

class WebGPUObjectCache {
  constructor() {
    this.caches = [
      "pipelineLayouts",
      "bindGroupLayouts",
      "computeModules",
      "computePipelines",
      "bindGroups",
    ];

    for (const cache of this.caches) {
      this[cache] = new CountingMap({ // wrapper over a Map
        enabled: this.initiallyEnabled.includes(cache),
      });
    }
  }
}
```

### Bind Group Caching

A bind group is fully determined by its layout and, for each binding, the `GPUBuffer` and the range (offset and size) bound. Creating bind groups afresh on every `execute()` was about a third of the CPU time of a small execute (a 7-pass sort took 4.0 ms of CPU per execute in Node before the bind-group cache and the short module keys below; 1.2 ms after).

**A canonical name for a buffer.** Gridwise's buffer names are not canonical. One `GPUBuffer` can have several names: a child primitive's `keysInOut` may be its parent's `keys`. One name can mean different buffers over time: `SuffixArray`'s `listA` and `listB` swap roles every round, and `AllocateBuffer` replaces a buffer whose size changes. So the cache does not use names. It uses the `GPUBuffer` object itself: a JavaScript object's identity is canonical for as long as the object exists.

- `gpuObjectId(buffer)` gives each `GPUBuffer` a small integer, the first time it is seen, from a counter that only goes up. A module-level `WeakMap` holds the ids, so it does not keep buffers alive, and ids are never reused.
- The key of a bind group is `pipeline key | group index | id:offset:size, …` (one entry per binding). The pipeline key pins the bind group layout.
- A buffer's contents can change freely (`writeBuffer`, kernels) without invalidating its bind groups: they refer to the buffer, not its contents.

**Eviction.** A cached bind group refers to its buffers, so entries for destroyed buffers must not accumulate.

- When `AllocateBuffer` replaces a buffer, and when `destroy()` frees a primitive's own buffers, the cache drops every bind group that binds them (`forgetBuffer`, through an index from buffer id to keys).
- Buffers that a caller passes in and destroys later are invisible to Gridwise. For those, the cache starts over once it holds 4096 bind groups.

`WebGPUObjectCache.bindGroup(pipeline, index, entries, create)` does the lookup; `webgpucache: "disable"` turns it off with the others.

### Cache Key Generation

To use objects as keys in a Map, we need a consistent and unique representation. Since Maps use Same-Value-Zero equality, two different objects with the same properties will not be treated as the same key. To solve this, we `JSON.stringify` a simplified representation of the object to create a string-based cache key.

Here's how keys are generated for different object types:

#### Pipeline Layout

The cache key for a `GPUPipelineLayout` is an array of strings representing the buffer types for that layout.

Example:

```js
["read-only-storage", "storage", "uniform", "storage", "storage", "storage"];
```

#### Bind Group Layout

The cache key for a `GPUBindGroupLayout` is the set of entries for that layout.

Example:

```js
[
  { binding: 0, visibility: 4, buffer: { type: "read-only-storage" } },
  {
    binding: 1,
    visibility: 4,
    buffer: {
      /*...*/
    },
  },
  // ... and so on for all entries
];
```

#### Compute Module

The cache key for a `GPUShaderModule` (referred to as a compute module in the context of compute shaders) is the entire kernel string. While underlying WebGPU engines like Dawn might have their own caching mechanisms, we implement a library-level cache for them as well.

The module's own `__cacheKey`, which the pipeline's key embeds, is a short id (`module3`), not the source: stringifying a sort's whole WGSL into every pipeline lookup cost about 1 ms per execute.

#### Compute Pipelines

The cache key for a `GPUComputePipeline` is derived from its descriptor object. Since the `GPUPipelineLayout` and `GPUShaderModule` are cached separately, we can reuse their cache keys to optimize the key generation for the pipeline itself.

A `__cacheKey` property is stored on cacheable objects, and this key is used during stringification to avoid deep, recursive serialization.

### Cache Statistics

The caches collect hit and miss statistics to help understand their effectiveness.

Example output from statistics collection:

```
Cache hits/misses:
Pipeline layouts: 7/1
Bind group layouts: 0/1
Compute modules: 4/4
Compute pipelines: 4/4
Bind groups: 3/3 (3 cached)
```

### Measuring Performance with CPU Timing

To measure the performance impact of caching, enable CPU timing. This will wait for the GPU to finish its work and then record the CPU time taken.

(TODO: move this into the timing article)

```js
const commandBuffer = encoder.finish();

if (args?.enableCPUTiming) {
  await this.device.queue.onSubmittedWorkDone();
  this.cpuStartTime = performance.now();
}

this.device.queue.submit([commandBuffer]);

if (args?.enableCPUTiming) {
  await this.device.queue.onSubmittedWorkDone();
  this.cpuEndTime = performance.now();
}
```
