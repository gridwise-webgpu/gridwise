---
layout: page
title: "Buffer Class"
category: gridwise
permalink: /buffer/
excerpt: "The Buffer class encapsulates data that spans both CPU (typed arrays) and GPU (GPUBuffer), providing a unified interface for data management."
order: 2
---

<div style="max-width: 100%; padding: 0;">

<!-- Header Section -->
<header style="max-width: 100%; margin-bottom: 3rem;">
  <h1 style="font-size: 2.5rem; font-weight: 800; color: #111827; margin-bottom: 1.5rem; line-height: 1.2;">Buffer Class</h1>
  <p style="font-size: 1.25rem; color: #6b7280; line-height: 1.75;">
    During Gridwise's development, we found a need to encapsulate the concept of a single wad of data that spans CPU and GPU. The <code>Buffer</code> class contains both a CPU-side JS typed array and a GPU-side <code>GPUBuffer</code>, providing a unified interface for data management across both domains.
  </p>
</header>

<!-- Section 1: Purpose & Design Philosophy -->
<section style="max-width: 100%; margin-bottom: 4rem;">
  <h2 style="font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Design Philosophy</h2>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1rem;">
    The abstraction is that the CPU and GPU buffers are (roughly) consistent with each other—they are not meant to store two logically different objects. The <code>Buffer</code> class manages synchronization and lifecycle for both representations.
  </p>
  <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
    We believe this is an object whose design could be revisited and improved, because it is generally useful in WebGPU primitive development and more generally across WebGPU development. We welcome a redesign.
  </p>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem;">
    <p style="margin: 0 0 0.75rem 0;"><strong style="color: #065f46;">Unified Interface:</strong> Single data structure that abstracts away the complexity of managing separate CPU and GPU buffers.</p>
    <p style="margin: 0;"><strong style="color: #065f46;">Automatic Sync:</strong> Handles data transfers between CPU and GPU with explicit control when needed.</p>
  </div>
</section>

<!-- Section 2: Use Cases -->
<section style="max-width: 100%; padding-top: 2rem; border-top: 1px solid #e5e7eb; margin-bottom: 3rem;">
  <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 2rem;">Use Cases</h2>
  
  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1.5rem;">
    <p style="margin: 0 0 0.75rem 0;"><strong style="color: #065f46;">Unified Data Management:</strong> Couple CPU and GPU buffers that store the same logical data as one logical entity. The class can copy data between them easily.</p>
    <ul style="color: #065f46; margin: 0.5rem 0 0 1.5rem;">
      <li>CPU side: JavaScript typed array</li>
      <li>GPU side: WebGPU <code>GPUBuffer</code> (can be a subset of a GPU-side allocation)</li>
    </ul>
  </div>
  
  <p style="margin-bottom: 0.5rem;"><strong>Flexible Initialization:</strong> Initialize a <code>Buffer</code> in multiple ways—using an existing <code>GPUBuffer</code>, or asking the <code>Buffer</code> constructor to allocate one.</p>
  
  <p style="margin-bottom: 0.5rem;"><strong>Test Data Generation:</strong> Generate data on the CPU for testing, storing it in the CPU buffer. Support different methods for data generation (random numbers within a range, sequential data where <code>buffer[i] = i</code>, etc.).</p>
  
  <p style="margin-bottom: 0.5rem;"><strong>WebGPU Abstraction:</strong> Hide WebGPU details when possible (calls to create a <code>GPUBuffer</code>, transfers between CPU and GPU).</p>
  
  <p style="margin-bottom: 0.5rem;"><strong>Primitive Integration:</strong> The <code>Buffer</code> needs a label so it can be associated with Gridwise primitives.</p>
  
  <p style="margin-bottom: 0.5rem;"><strong>In-Place Operations:</strong> Some primitives (like sort) write their GPU output on top of their GPU input. A <code>Buffer</code> class must support this and additionally store the original (CPU) input for correctness validation.</p>
  
  <p style="margin-bottom: 1.5rem;"><strong>Size Queries:</strong> Query a <code>Buffer</code> for both <code>size</code> (number of bytes) and <code>length</code> (number of elements).</p>
</section>

<!-- Section 3: Constructor -->
<section style="padding-top: 2rem; border-top: 1px solid #e5e7eb; margin-bottom: 3rem;">
  <div style="max-width: 100%;">
    <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 1rem;">Constructor</h2>
    
    <div style="background: #1f2937; border-radius: 0.75rem; overflow: hidden; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;">
      <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: #374151; border-bottom: 1px solid #4b5563;">
        <span style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.75rem; color: #9ca3af;">constructor(args)</span>
      </div>
      <div style="padding: 1.5rem;">
        <p style="color: #e5e7eb; font-size: 0.875rem; margin: 0;">
          Creates a new <code style="color: #fbbf24;">Buffer</code> instance, using the properties of the <code style="color: #fbbf24;">args</code> object to configure the buffer.
        </p>
      </div>
    </div>

    <h3 style="font-size: 1.25rem; font-weight: 700; color: #111827; margin-bottom: 1.5rem;">Arguments</h3>
    <p style="color: #4b5563; line-height: 1.75; margin-bottom: 1.5rem;">
      The constructor takes a single <code>args</code> object with these properties:
    </p>

    <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1.5rem;">
      <h4 style="font-size: 1.125rem; font-weight: 700; color: #065f46; margin-bottom: 1rem;">Required Properties</h4>
      
      <p style="margin-bottom: 0.5rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">device</code> (GPUDevice) – The WebGPU device used to create the buffer.</p>
      
      <p style="margin-bottom: 1rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">datatype</code> (string) – The data type of elements in the buffer (e.g., <code>'f32'</code>, <code>'u32'</code>, <code>'i32'</code>).</p>
      
      <h4 style="font-size: 1.125rem; font-weight: 700; color: #065f46; margin-bottom: 1rem; margin-top: 1rem;">Size Configuration <span style="font-size: 0.875rem; font-weight: 400;">(specify one)</span></h4>
      
      <p style="margin-bottom: 0.5rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">size</code> (number) – The buffer's total size in bytes. You must specify either <code>size</code> or <code>length</code>, but not both.</p>
      
      <p style="margin-bottom: 0;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">length</code> (number) – The number of elements in the buffer.</p>
    </div>
    
    <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1.5rem;">
      <h4 style="font-size: 1.125rem; font-weight: 700; color: #065f46; margin-bottom: 1rem;">Optional Properties</h4>
        
        <p style="margin-bottom: 0.5rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">label</code> (string) – A descriptive name for the buffer.</p>
        
        <p style="margin-bottom: 0.5rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">usage</code> (GPUBufferUsageFlags) – Specifies how the GPU buffer will be used. Defaults to <code>GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST</code>.</p>
        
        <p style="margin-bottom: 0.5rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">buffer</code> (GPUBuffer | GPUBufferBinding) – An existing buffer to wrap. If you provide this, <code>createGPUBuffer</code> should be <code>false</code>.</p>
        
        <p style="margin-bottom: 0.5rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">createCPUBuffer</code> (boolean) – If <code>true</code>, a CPU-side <code>TypedArray</code> is created.</p>
        
        <p style="margin-bottom: 0.5rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">initializeCPUBuffer</code> (string) – A keyword specifying how to fill the CPU buffer with initial data:</p>
        <ul style="margin: 0.5rem 0 0.5rem 1.5rem; line-height: 1.4;">
          <li>For floats (<code>'f32'</code>): <code>'randomizeMinusOneToOne'</code>, <code>'randomizeAbsUnder1024'</code>, <code>'fisher-yates'</code></li>
          <li>For integers (<code>'u32'</code>, <code>'i32'</code>, <code>'u64'</code>): <code>'xor-beef'</code>, <code>'randomizeAbsUnder1024'</code>, <code>'constant'</code>, <code>'bitreverse'</code>, <code>'randomBytes'</code>, <code>'fisher-yates'</code></li>
        </ul>
        
        <p style="margin-bottom: 0.5rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">storeCPUBackup</code> (boolean) – If <code>true</code>, a backup of the initialized CPU buffer is saved for later (useful if the primitive overwrites the buffer contents but we still need the original contents for validation).</p>
        
        <p style="margin-bottom: 0.5rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">createGPUBuffer</code> (boolean) – If <code>true</code>, the corresponding <code>GPUBuffer</code> is created on the GPU.</p>
        
        <p style="margin-bottom: 0.5rem;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">initializeGPUBuffer</code> (boolean) – If <code>true</code>, data from the CPU buffer is immediately copied to the GPU buffer. Requires <code>createCPUBuffer</code> to be <code>true</code>.</p>
        
        <p style="margin: 0;"><code style="background: #dcfce7; color: #065f46; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-weight: 600;">createMappableGPUBuffer</code> (boolean) – If <code>true</code>, a secondary, mappable GPU buffer is also created to help read data back from the GPU.</p>
      </div>
  </div>
</section>

<!-- Section 4: Methods -->
<section style="max-width: 100%; padding-top: 2rem; border-top: 1px solid #e5e7eb; margin-bottom: 3rem;">
  <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 2rem;">Methods</h2>
  
  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1.5rem;">
    <h3 style="font-size: 1.125rem; font-weight: 700; color: #065f46; margin-bottom: 0.75rem;">
      <code>createCPUBuffer(args)</code>
    </h3>
    <p style="color: #065f46; margin: 0 0 0.5rem 0;">
      Creates and optionally initializes the CPU-side <code>TypedArray</code> buffer. You can call this after construction if the buffer wasn't created in the constructor.
    </p>
    <p style="color: #065f46; margin: 0;"><strong>Parameters:</strong> <code>args</code> (object) - An optional object to override the instance's <code>length</code> and <code>datatype</code> or provide initialization options.</p>
  </div>

  <p style="margin-bottom: 0.5rem;"><strong><code>createGPUBuffer(args)</code></strong> – Creates the <code>GPUBuffer</code> on the device. You can call this after construction if the GPU buffer wasn't created in the constructor.</p>
  <p style="margin-bottom: 1rem; padding-left: 1rem;">Parameters: <code>args</code> (object) - An optional object to override the instance's <code>device</code>, <code>size</code>, <code>datatype</code>, or <code>usage</code>.</p>

  <p style="margin-bottom: 0.5rem;"><strong><code>createMappableGPUBuffer(size)</code></strong> – Creates a separate GPU buffer that can be mapped by the CPU. This buffer is used as a temporary staging area for transferring data from the main GPU buffer back to the CPU.</p>
  <p style="margin-bottom: 1rem; padding-left: 1rem;">Parameters: <code>size</code> (number) - The size in bytes for the mappable buffer.</p>

  <p style="margin-bottom: 0.5rem;"><strong><code>copyCPUToGPU()</code></strong> – Uploads data from the CPU-side buffer (<code>cpuBuffer</code>) to the main GPU buffer.</p>

  <p style="margin-bottom: 0.5rem;"><strong><code>async copyGPUToCPU()</code></strong> – Asynchronously copies data from the GPU buffer back to the CPU-side buffer. It works by copying the data to a temporary mappable buffer, reading it back to the CPU, and updating the <code>cpuBuffer</code> property.</p>

  <p style="margin-bottom: 0.5rem;"><strong><code>copyCPUBackupToCPU()</code></strong> – Restores the <code>cpuBuffer</code> from the backup that was created during initialization.</p>

  <div style="padding: 1.5rem; border: 2px solid #ef4444; border-radius: 0.5rem; background: #fef2f2; margin-top: 1rem;">
    <p style="color: #991b1b; font-weight: 600; margin: 0;">
      <strong><code>destroy()</code></strong> – Destroys the associated GPU buffers to free up GPU memory.
    </p>
  </div>
</section>

<!-- Section 5: Properties -->
<section style="max-width: 100%; padding-top: 2rem; border-top: 1px solid #e5e7eb; margin-bottom: 3rem;">
  <h2 style="font-size: 1.875rem; font-weight: 700; color: #111827; margin-bottom: 2rem;">Properties (Getters & Setters)</h2>
  
  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1rem;">
    <p style="margin: 0 0 0.5rem 0;"><strong style="color: #065f46;"><code>buffer</code></strong></p>
    <p style="margin: 0 0 0.25rem 0;"><strong>get:</strong> Returns the GPU buffer as a <code>GPUBufferBinding</code> object (e.g., <code>{ buffer: GPUBuffer }</code>).</p>
    <p style="margin: 0;"><strong>set:</strong> Sets the internal GPU buffer. Accepts a raw <code>GPUBuffer</code> or a <code>GPUBufferBinding</code> object.</p>
  </div>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1rem;">
    <p style="margin: 0 0 0.5rem 0;"><strong style="color: #065f46;"><code>cpuBuffer</code></strong></p>
    <p style="margin: 0;"><strong>get:</strong> Returns the CPU-side <code>TypedArray</code> (e.g., <code>Float32Array</code>, <code>Uint32Array</code>).</p>
  </div>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1rem;">
    <p style="margin: 0 0 0.5rem 0;"><strong style="color: #065f46;"><code>cpuBufferBackup</code></strong></p>
    <p style="margin: 0;"><strong>get:</strong> Returns the backup <code>TypedArray</code> if one was created.</p>
  </div>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1rem;">
    <p style="margin: 0 0 0.5rem 0;"><strong style="color: #065f46;"><code>size</code></strong></p>
    <p style="margin: 0;"><strong>get:</strong> Returns the size of the buffer in <strong>bytes</strong>.</p>
  </div>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem; margin-bottom: 1rem;">
    <p style="margin: 0 0 0.5rem 0;"><strong style="color: #065f46;"><code>length</code></strong></p>
    <p style="margin: 0;"><strong>get:</strong> Returns the number of <strong>elements</strong> in the buffer.</p>
  </div>

  <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 1.5rem;">
    <p style="margin: 0 0 0.5rem 0;"><strong style="color: #065f46;"><code>device</code></strong></p>
    <p style="margin: 0;"><strong>get:</strong> Returns the associated <code>GPUDevice</code>.</p>
  </div>
</section>

</div>

<style>
@media (max-width: 768px) {
  section[style*="grid-template-columns: 1fr 1fr"] {
    grid-template-columns: 1fr !important;
  }
  div[style*="position: sticky"] {
    position: relative !important;
  }
}
</style>