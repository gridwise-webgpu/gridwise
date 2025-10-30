---
layout: post
title: "Gridwise WebGPU Object Caching Strategy"
date: 2025-08-12
excerpt: "Learn about Gridwise's caching strategy for WebGPU objects to improve performance while balancing overhead costs."
---

## Gridwise WebGPU Object Caching Strategy

This document outlines the caching strategy used for WebGPU objects within Gridwise. Creating WebGPU objects is not free and is potentially expensive. Caching created objects so that they can be reused potentially helps performance. The downsides of caching are that caching itself is not free and that the WebGPU back end may do its own caching. In Gridwise, caching is enabled by default but can be disabled (by instantiating a primitive with the argument `webgpucache = "disabled"`).

### Cacheable WebGPU Objects

The following WebGPU objects are currently cached by our library:

- `GPUShaderModule`
- `GPUPipelineLayout`
- `GPUBindGroupLayout`
- `GPUComputePipeline`

Of these, `GPUShaderModule` is potentially independent of the `GPUDevice`, while the others are dependent on the specific `GPUDevice` instance.

## Cache Implementation

Every primitive shares a `\_\_deviceToWebGPUObjectCache`, which is a `WeakMap` that maps a `GPUDevice` to its corresponding cache. Each device's cache contains several individual caches for different object types. These are regular `Map` objects that map a generated key to the WebGPU object.

The available caches are:

- `pipelineLayouts`
- `bindGroupLayouts`
- `computeModules`
- `computePipelines`

Each of these caches can be individually enabled or disabled when a primitive is created.

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

Bind groups are not cached. This decision was made because bind groups depend on `GPUBuffer` objects. Reliably creating a cache key from a
`GPUBuffer` is problematic due to its dynamic state.

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
