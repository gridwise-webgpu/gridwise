# Release Notes

## Known WebGPU Compatibility Issues

This release contains the following known compatibility limitations across WebGPU browser implementations and software renderers:

### 1. Firefox WGSL Function Pointer Parameter Restriction (Naga Limitation)
* **Description**: Under Firefox, shader validation/compilation will fail when trying to create shader modules that pass `workgroup` or `storage` memory pointers to user-defined helper functions (e.g., `wgReduce` and `workgroupScan`).
* **Technical Details**: While the WebGPU WGSL specification explicitly allows passing pointers in the `workgroup` address space to helper functions, Firefox's WGSL translator (**Naga**) does not currently support compiling or lowering these pointer parameters to its backend shading languages. Chrome (which uses the **Tint** compiler) handles this correctly by automatically inlining the helper functions or translating the accesses to reference the global variables.
* **Impact**: Gridwise primitives like `DLDFScan` which leverage helper functions with workgroup pointer parameters will fail to compile and run in Firefox.

### 2. CPU Spin-Loop Starvation on SwiftShader (Emulated Subgroups)
* **Description**: Subgroup emulation will deadlock or hang when executed on CPU-based Vulkan software rasterizers like **SwiftShader** (which is the default on headless CI environments like GitHub Actions).
* **Technical Details**: The emulated subgroup helpers (like `subgroupShuffle` and `subgroupBallot`) rely on atomic-based spin-loops to pass data between threads within a workgroup:
  $$\text{while} \ (\text{atomicLoad}(\text{flag}) < \text{shuffle\_count}) \ \{ \ \dots \ \}$$
  On a physical GPU, the hardware scheduler ensures forward progress for all threads. However, on CPU-based renderers, workgroups are mapped to a limited pool of CPU threads. If one CPU thread enters a spin-loop waiting for another thread in the same workgroup to update the flag, it consumes $100\%$ CPU time. Because WebGPU compute shaders do not guarantee preemption or thread yielding, the spinning thread starves the other thread, preventing it from ever being scheduled to update the flag, causing a permanent CPU deadlock.
* **Impact**: Gridwise tests running in emulated subgroup mode will hang indefinitely when run in headless CPU-only CI environments using SwiftShader.
