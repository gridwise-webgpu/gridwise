---
layout: post
title: "Abstraction Challenges in Writing a WebGPU/WGSL Workgroup Reduce Function"
date: 2025-04-08
excerpt: "A practical walkthrough of implementing a workgroupReduce function in WGSL, exploring language challenges and abstraction pain points."
---

In this post, we'll walk through the process of writing a `workgroupReduce` function in WGSL. We write this function as if we were a library writer who would implement `workgroupReduce` with the intent that other unknown developers will call our function. Our goal is to explore the WGSL issues facing the library writer. Through this practical example of creating a `reduceWorkgroup` function, we will identify and discuss several pain points in the language.

---

## Summary of Pain Points

When writing a generic library function in WGSL, we identified several key difficulties:

- **Explicit Built-in Declarations:** WGSL's requirement to explicitly declare `@builtins` complicates APIs.
- **Inconsistent Built-in Variants:** Not all built-ins have both 3D and 1D variants.
- **Module-Scoped Workgroup Memory:** Declaring workgroup memory only at the module scope harms modularity.
- **Lack of Generics:** The absence of function overloading or templating makes it difficult to write generic APIs.

These limitations lead to a choice between two library design patterns: a few complex kernels using template-literal string pasting, or many simple kernels using metaprogramming.

---

## Scenario: A reduceWorkgroup Function

We will focus on creating a `reduceWorkgroup` function that will be incorporated into a primitive library. We expect external users will face a need to input a linear array of data and have each workgroup compute the sum of a sub-section of that data. Our implementation, as part of a library, will allow them to do so without concern for how the reduction is implemented.

The semantics of the function are as follows: consider a workgroup of N invocations (e.g., N = 128); workgroup W will compute the sum of `input[W*N:W*(N+1))`, and each invocation in the workgroup will receive this sum as the return value of a function call.

### Desired Function Call: Caller (WGSL Definition)

Ideally, making the function call from within a WGSL kernel would be simple, with the minimal call looking something like

```wgsl
...
val = reduceWorkgroup(ptr);
...
```

Here we assume that `ptr` is an address pointing to an item within a linear input array, and that the private variable `val` and the deferenced value at `ptr` have the same datatype. If the workgroup has `W` invocations, then `val`, for every invocation in workgroup `W`, will assume the value `reduce(ptr[W*N:W*(N+1)))`.

This interface does not specify the reduction operation (for instance, addition, max, or min), nor the datatype. Possible options to address the former include:

- A separate reduction function for each operation (`reduceAddWorkgroup`, `reduceMinWorkgroup`, etc.)
- A default operation (e.g., `add`)
- Separately defining an operation (a "binop" == a monoid) that must be explicitly set by the caller and implicitly used by the function

However, we do not further discuss this issue in this document.

### Desired Function Call: Callee

We would also like the callee function/kernel to be as simple as possible. The minimum kernel that calls our function, and places the output into an output array, would look something like this:

```wgsl
@compute @workgroup_size(128) fn reduceKernel(
  @builtin(local_invocation_index) lidx: u32,
  @builtin(workgroup_id) wgid: vec3u
) {
  let r = reduceWorkgroup(&in);
  if (lidx == 0) {
    out[wgid.x] = r;
  }
}
```

### Issues with the above kernel

#### Builtins must be explicitly declared

The above kernel has 5 lines of code. 2 of them are declaring built-ins. WGSL requires these declarations, but other languages have made different choices. Why are variables like `local_invocation_index` and `workgroup_id` not available within a kernel without the need for these declarations?

Possible ways to remove the need for these declarations include

- Making a builtin variable available within kernels (e.g., CUDA's `threadIdx`)
- Making a function call that retrieves a builtin variable available within kernels (e.g., SYCL's `get_id()`)

#### Some 3D builtins lack a 1D variant

The use of `wgid` above reflects a common implementation pattern: we logically have a 1D data space (`in`, `out`) and thus want our workgroups to also be organized as a 1D data space. However, the only access we have to the workgroup id is through a 3D builtin. Above, we just assume that such a builtin only uses the `.x` part of the `workgroup_id` builtin, but this assumption is checked nowhere.

It would be convenient to have a 1D builtin equivalent for every 3D builtin (specifically, WGSL lacks `workgroup_index` and `global_invocation_index`).

### Implementing reduceWorkgroup

Now let's turn to actually implementing the `reduceWorkgroup` function.

An initial, simple implementation of `reduceWorkgroup` might allocate an workgroup-memory variable and use `atomicAdd` to accumulate inputs into it. It might look like this:

```wgsl
fn reduceWorkgroup(input: ptr<storage, array<u32>, read>) -> u32 {
  atomicAdd(&wg_temp[0], input[?]);
  workgroupBarrier();
  return atomicLoad(&wg_temp[0]);
}
```

#### Leaky abstraction: `wg_temp`

The above code uses storage `wg_temp` located in workgroup memory. WGSL requires that variables in the workgroup address space be declared at the module scope, which is a significant limitation on modularity. Who declares this storage? How big is it? Why can't I declare it within the `reduceWorkgroup` function?

We could declare the workgroup memory `wg_temp_reduceWorkgroup` at the module scope as follows:

```wgsl
var<workgroup> wg_temp_reduceWorkgroup: array<atomic<u32>, 1>;
fn reduceWorkgroup(input: ptr<storage, array<u32>, read>,
                   gid: vec3u,
                  ) -> u32 {
  atomicAdd(&wg_temp_reduceWorkgroup[0], input[?]);
  workgroupBarrier();
  return atomicLoad(&wg_temp_reduceWorkgroup[0]);
}
```

However, such a declaration by the writer of a library might have a name conflict for the workgroup memory `wg_temp_reduceWorkgroup`, either within the library or with a different allocation defined by the user of the library. Note that if we define many different `reduceWorkgroup` functions, for instance ones that are specialized to datatype and reduction operation, we also must declare a different workgroup memory variable for each of them. Then a compiler would have to identify which specialized functions are actually used and only instantiate their workgroup memory regions. Finally, while it is likely that a complex workload that used many different reduce functions from this library could potentially share the same workgroup memory region used for the accumulation, the above code would not allow this sharing.

We might consider a language change that allows workgroup memory to be declared within a function instead of at module scope. This potentially eliminates name conflicts. But the programmer still must account for the use of workgroup memory.

#### Specifying the input region

The above code has the line `atomicAdd(&wg_temp[0], input[?])`. How do we fill in the `[?]`?

What we want is for each individual instance (thread) to fetch the input value that corresponds to its global instance id. Instance 0 fetches `input[0]`, instance 128 fetches `input[128]`, etc. This information is only available through a builtin, which requires that the enclosing kernel must input that builtin and that the function add an additional input:

```wgsl
@compute @workgroup_size(128) fn reduceKernel(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_index) lidx: u32,
  @builtin(workgroup_id) wgid: vec3u) {
  let r = reduceWorkgroup(&in, gid.x);
  if (lidx == 0) {
    out[wgid.x] = r;
  }
}
```

While this second argument to `reduceWorkgroup` adds flexibility, it would be nice to not require the kernel author to plumb a builtin through the kernel if that builtin was used in the default way.

(Look at the use of `wgid.x` in the above kernel and recall the comment above that "It would be useful to have a 1D builtin equivalent for every 3D builtin (specifically, `workgroup_index` and `global_invocation_index`)"---here is a use for `workgroup_index`.)

## A Working Implementation

Kernel (WGSL) code:

```wgsl
@compute @workgroup_size(128) fn reduceKernel(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_index) lidx: u32,
  @builtin(workgroup_id) wgid: vec3u) {
  let r = reduceWorkgroup(&in, gid.x);
  if (lidx == 0) {
    out[wgid.x] = r;
  }
}
```

Implementation of `reduceWorkgroup`:

```wgsl
var<workgroup> wg_temp: array<atomic<u32>, 1>;

fn reduceWorkgroup(input: ptr<storage, array<u32>, read>,
                   gid: vec3u) -> u32 {
  atomicAdd(&wg_temp[0], input[gid.x]);
  workgroupBarrier();
  return atomicLoad(&wg_temp[0]);
}
```

Note that nothing in the actual kernel itself is specific to datatype. (Or workgroup size, for that matter.)

### Generalizing our implementation ...

#### ... across datatypes

So, as an experiment, let's consider adding another `reduceWorkgroup` function that reduces `f32` values rather than the `u32` values above. The callee code does not change, but the library function implementation does:

```wgsl
var<workgroup> wg_temp: array<atomic<f32>, 1>;

fn reduceWorkgroup(input: ptr<storage, array<f32>, read>,
                   gid: vec3u) -> f32 {
  atomicAdd(&wg_temp[0], input[gid.x]);
  workgroupBarrier();
  return atomicLoad(&wg_temp[0]);
}
```

This addition of an `f32 reduceWorkgroup` function uncovers two new issues:

- The `wg_temp` workgroup variable now has a name collision between the `u32` variant and the `f32` variant.
- We now have two different functions named `reduceWorkgroup`, which is not permitted in WGSL. We can't overload a function name. Type-specific dispatch would be useful here.

The careful WGSL developer notices a third issue: there is no `atomicAdd` on `f32` variables.

The above issues motivate a library design that has different functions for each datatype, specifically here `reduceWorkgroupU32` and `reduceWorkgroupF32`.

#### ... across storage locations

Recall that our `reduceWorkgroup` signature shows that the input data is located in global (storage) memory:

```wgsl
fn reduceWorkgroup(input: ptr<storage, array<u32>, read>,
                   gid: vec3u) -> u32 {
  atomicAdd(&wg_temp[0], input[gid.x]);
  workgroupBarrier();
  return atomicLoad(&wg_temp[0]);
}
```

Let's consider a `reduceWorkgroup` that instead stored its input in workgroup memory. It would be ideal if we could support inputs in either global (storage) or workgroup memory without any changes to the code.

```wgsl
fn reduceWorkgroup(input: ptr<workgroup, array<u32>, SIZE>,
                   gid: vec3u) -> u32 {
  atomicAdd(&wg_temp[0], input[gid.x]);
  workgroupBarrier();
  return atomicLoad(&wg_temp[0]);
}
```

Note that the _bodies_ of these two functions are identical; only the argument type is different. Nonetheless WGSL requires we implement both functions separately because they have different argument types. This argues for a library designer separately implementing `reduceWorkgroupWGU32` and `reduceWorkgroupStorageU32`.

## A More Realistic Workgroup Reduce Function

The above functions are simple but not high-performance, because all instances must serialize on the atomic reduction variable. Higher-performance reduce implementations instead exploit parallelism across instances within a workgroup. As well, the highest-performance implementations will likely make use of WebGPU subgroups.

One of the challenges of subgroups is that their size is not constant within WebGPU. In fact, different kernels running within the same application on some WebGPU-capable hardware may not even share the same subgroup size. If we wish to write a subgroup-size-agnostic kernel, we must support all possible subgroup sizes within that kernel. The challenge with such an implementation is that any allocation that depends on the subgroup size must perform a worst-case allocation that works with any subgroup size.

WebGPU provides two adapter properties---`MIN_SUBGROUP_SIZE` and `MAX_SUBGROUP_SIZE`---that we can use to perform allocations. Below is the start of a high-performance workgroup-reduce kernel that requires a workgroup-memory allocation for partial reductions. The size of the workgroup-memory allocation is directly proportional to the workgroup size (variable: `{$workgroupSize}`) and inversely proportional to the subgroup size (variable: `${MIN_SUBGROUP_SIZE}`).

```wgsl
const BLOCK_DIM: u32 = ${workgroupSize};
const TEMP_WG_MEM_SIZE = BLOCK_DIM / ${MIN_SUBGROUP_SIZE};
var<workgroup> wg_temp: array<u32, TEMP_WG_MEM_SIZE>;

fn reduceWorkgroup(input: ptr<storage, array<u32>, read>,
                   gid: vec3u, lidx: u32, sgid: u32, sgsz: u32
                   ) -> u32 {
  let sid = lidx / sgsz;
  let lane_log = u32(countTrailingZeros(sgsz)); /* log_2(sgsz) */
  let local_spine: u32 = BLOCK_DIM >> lane_log;
  /* BLOCK_DIM / subgroup size; how many partial reductions in this tile? */
  // ...
}
```

The value of `MIN_SUBGROUP_SIZE` (and more broadly, any value that is adapter-dependent and thus not determinable until runtime) has two impacts on this code:

- First, the allocation is inversely proportional to `MIN_SUBGROUP_SIZE`. We would like to only allocate the memory we need. It is not clear in WGSL what the right way to perform this specialization might be: static (compile-time) specialization, such as what a C++ template provides, or instead runtime compilation that incorporates the adapter property.
- Second, the algorithm can be made simpler if the subgroup sizes are large enough (specifically, if `MIN_SUBGROUP_SIZE * MIN_SUBGROUP_SIZE >= workgroupSize`). It would be desirable to have the maximum possible compile-time specialization for such a decision rather than making all aspects of the decision at runtime.

In summary, the presence of an adapter-specific property, only discoverable at runtime, mandates making runtime decisions that developers would prefer to do at compile time.

## Multiple implementations are interesting, but builtins complicate function signatures

It would be desirable for a library user to write code that could call `reduceWorkgroup` but not have to choose which implementation of `reduceWorkgroup`; instead, that choice could be made by the underlying library in some static or runtime-dependent way. This ability would also be useful for the library developer, who might want to try different `reduceWorkgroup` implementations in real workloads to determine the fastest one without having to change the function call within the workload. However, if each `reduceWorkgroup` implementation had a different function signature, this would be impossible.

Unfortunately, the function signatures of different implementations of `reduceWorkgroup` (above) are different.

```wgsl
fn reduceWorkgroup(input: ptr<storage, array<u32>, read>,
                   gid: vec3u) -> u32
```

```wgsl
fn reduceWorkgroup(input: ptr<storage, array<u32>, read>,
                   gid: vec3u, lidx: u32, sgid: u32, sgsz: u32
                   ) -> u32
```

The difference is which builtins are used. Builtins must be passed in through the function signature, and if different implementations use a different set of builtins, their function signatures will differ.

One remedy is to pass every builtin into every function. This is verbose and quite kludgey. Instead we have addressed this problem by ...

### Packing Built-ins into Structs

To simplify the function signatures, we pack built-ins into structs:

Code snippet

```wgsl
struct Builtins {
 @builtin(global_invocation_id) gid: vec3u,
 @builtin(num_workgroups) nwg: vec3u,
 @builtin(workgroup_id) wgid: vec3u,
 @builtin(local_invocation_index) lidx: u32,
 @builtin(local_invocation_id) lid: vec3u,
 @builtin(subgroup_size) sgsz: u32,
 @builtin(subgroup_invocation_id) sgid: u32
}
```

We also divide those builtins into two different structs, differentiated by whether the member is uniform or non-uniform. The use of `BuiltinsUniform` is sometimes necessary to ensure workgroup or subgroup uniformity.

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
  @builtin(subgroup_size) sgsz: u32 /* subgroup size */
}
```

This allows for a cleaner function signature, but mandates that the library user must use the library's naming conventions.

## Conclusion

We conclude by reiterating the main pain points in WGSL that make writing generic compute libraries challenging:

- Explicit declaration of @builtins.
- Lack of 1D variants for all built-ins.
- Module-scoped workgroup memory.
- Lack of function overloading and templating.

These limitations force library designers to choose between metaprogramming to generate many specialized functions or using template literals to create a few complex kernels.
