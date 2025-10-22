---
layout: post
title: "Gridwise's Buffer Class"
date: 2025-09-16
excerpt: "The Buffer class encapsulates data that spans both CPU (typed arrays) and GPU (GPUBuffer), providing a unified interface for data management."
---

During Gridwise's development, we found a need to encapsulate the concept of a single wad of data that spans CPU and GPU. We call this class `Buffer`. It contains both a CPU-side JS typed array and a GPU-side `GPUBuffer`. The abstraction is that these two objects are (roughly) consistent with each other (they are not meant to store two logically different objects).

We believe this is an object whose design could be revisited and improved, because it is generally useful in WebGPU primitive development and more generally across WebGPU development. We welcome a redesign. For that purpose, we list our use cases:

- We want to couple CPU and GPU buffers that store the same logical data as one logical entity (here, a `Buffer` class). The class should be able to copy data between them easily.
  - On the CPU side, we want to use a JavaScript typed array.
  - On the GPU side, we want to use a WebGPU `GPUBuffer`. We note a `GPUBuffer` can be a subset of a GPU-side allocation.
- We want to allow initializing this `Buffer` in multiple flexible ways. For instance, we should be able to initialize a `Buffer` using an existing `GPUBuffer`, or alternatively ask the `Buffer` constructor to allocate one.
- We need to generate data on the CPU for testing, storing it in the (CPU) buffer. We wish to support different methods for data generation (for instance, random numbers within a range, or a dataset where `buffer[i] = i`).
- We want to hide WebGPU details if possible (for instance, the call to create a `GPUBuffer`, the call to copy data between CPU and GPU).
- The `Buffer` needs a label so it can be associated with Gridwise primitives.
- At least one of our primitives (sort) writes its GPU output on top of its GPU input. A `Buffer` class must support this and additionally store the original (CPU) input for correctness validation.
- We want to support querying a `Buffer` for both:
  - `size` (the number of bytes in the buffer)
  - `length` (the number of elements in the buffer)

## `Buffer` Class documentation

## Constructor

### `constructor(args)`

This creates a new `Buffer` instance, using the properties of the **`args`** object to configure the buffer.

**Arguments:**

The constructor takes a single **`args`** object with these optional properties:

* **`label`**: `string` - A descriptive name for the buffer.
* **`device`**: `GPUDevice` - **(Required)** The WebGPU device used to create the buffer.
* **`datatype`**: `string` - **(Required)** The data type of elements in the buffer (e.g., `'f32'`, `'u32'`, `'i32'`).
* **`size`**: `number` - The buffer's total size in bytes. You must specify either **`size`** or **`length`**, but not both.
* **`length`**: `number` - The number of elements in the buffer.
* **`usage`**: `GPUBufferUsageFlags` - Specifies how the GPU buffer will be used. It defaults to `GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST`.
* **`buffer`**: `GPUBuffer` or `GPUBufferBinding` - An existing buffer to wrap. If you provide this, **`createGPUBuffer`** should be `false`.
* **`createCPUBuffer`**: `boolean` - If `true`, a CPU-side `TypedArray` is created.
* **`initializeCPUBuffer`**: `string` - A keyword specifying how to fill the CPU buffer with initial data.
    * For floats (`'f32'`): `'randomizeMinusOneToOne'`, `'randomizeAbsUnder1024'`, `'fisher-yates'`.
    * For integers (`'u32'`, `'i32'`, `'u64'`): `'xor-beef'`, `'randomizeAbsUnder1024'`, `'constant'`, `'bitreverse'`, `'randomBytes'`, `'fisher-yates'`.
* **`storeCPUBackup`**: `boolean` - If `true`, a backup of the initialized CPU buffer is saved for later (this is useful if the primitive overwrites the buffer contents but we still need the original contents, for instance if we are validating the output).
* **`createGPUBuffer`**: `boolean` - If `true`, the corresponding `GPUBuffer` is created on the GPU.
* **`initializeGPUBuffer`**: `boolean` - If `true`, data from the CPU buffer is immediately copied to the GPU buffer. Requires `createCPUBuffer` to be `true`.
* **`createMappableGPUBuffer`**: `boolean` - If `true`, a secondary, mappable GPU buffer is also created to help read data back from the GPU.

***

## Methods

### `createCPUBuffer(args)`

Creates and optionally initializes the CPU-side `TypedArray` buffer. You can call this after construction if the buffer wasn't created in the constructor.

* **`args`**: `object` - An optional object to override the instance's **`length`** and **`datatype`** or provide initialization options.

### `createGPUBuffer(args)`

Creates the `GPUBuffer` on the device. You can call this after construction if the GPU buffer wasn't created in the constructor.

* **`args`**: `object` - An optional object to override the instance's **`device`**, **`size`**, **`datatype`**, or **`usage`**.

### `createMappableGPUBuffer(size)`

Creates a separate GPU buffer that can be mapped by the CPU. This buffer is used as a temporary staging area for transferring data from the main GPU buffer back to the CPU.

* **`size`**: `number` - The size in bytes for the mappable buffer.

### `copyCPUToGPU()`

Uploads data from the CPU-side buffer (`cpuBuffer`) to the main GPU buffer.

### `async copyGPUToCPU()`

Asynchronously copies data from the GPU buffer back to the CPU-side buffer. It works by copying the data to a temporary mappable buffer, reading it back to the CPU, and updating the `cpuBuffer` property.

### `copyCPUBackupToCPU()`

Restores the `cpuBuffer` from the backup that was created during initialization.

### `destroy()`

Destroys the associated GPU buffers to free up GPU memory.

***

## Properties (Getters & Setters)

### `buffer`

* **`get`**: Returns the GPU buffer as a `GPUBufferBinding` object (e.g., `{ buffer: GPUBuffer }`).
* **`set`**: Sets the internal GPU buffer. Accepts a raw `GPUBuffer` or a `GPUBufferBinding` object.

### `cpuBuffer`

* **`get`**: Returns the CPU-side `TypedArray` (e.g., `Float32Array`, `Uint32Array`).

### `cpuBufferBackup`

* **`get`**: Returns the backup `TypedArray` if one was created.

### `size`

* **`get`**: Returns the size of the buffer in **bytes**.

### `length`

* **`get`**: Returns the number of **elements** in the buffer.

### `device`

* **`get`**: Returns the associated `GPUDevice`.