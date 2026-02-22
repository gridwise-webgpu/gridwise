---
layout: page
title: "Gridwise WebGPU @builtins Strategy"
category: gridwise
permalink: /builtins-strategy/
excerpt: "Exploring the use of WGSL @builtin arguments in Gridwise for optimal WebGPU primitive performance."
order: 6
---

Builtins used by WGSL functions must be passed in through the function signature, and if different function implementations use a different set of builtins, their function signatures will differ. This is problematic for the developer, who would like to switch between different function implementations during code development.

Rather than pass every single builtin into every function, in Gridwise, we addressed this challenge by bundling all builtins into a struct (wgslFunctions.mjs). Then when we must pass builtins into a function, we simply pass the entire struct.

```wgsl
  struct Builtins {
    @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
    @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
    @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    @builtin(local_invocation_id) lid: vec3u /* 3D thread index within workgroup */,
    @builtin(subgroup_size) sgsz: u32, /* 32 on Apple GPUs */
    @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */
  }
```

Some functions may require passing in only uniform builtins (to allow workgroup or subgroup uniformity). Thus we also define, and use, structs that include only the uniform and nonuniform subset of builtins:

```wgsl
  struct BuiltinsNonuniform {
    @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
    @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    @builtin(local_invocation_id) lid: vec3u /* 3D thread index within workgroup */,
    @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */
  }
  struct BuiltinsUniform {
    @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
    @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    @builtin(subgroup_size) sgsz: u32 /* 32 on Apple GPUs */
  }
```

[Abstraction Challenges in Writing a WebGPU/WGSL Workgroup Reduce Function](../writing-a-webgpu-wgsl-workgroup-reduce-function/) has more detail on this design decision.
