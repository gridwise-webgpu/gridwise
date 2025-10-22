---
layout: post
title: "Gridwise's Binary Operator Class"
date: 2025-09-16
excerpt: "Understanding the BinOp class that represents monoids with binary operations, datatypes, and identity elements for use in primitives."
---

Gridwise's binary operator class is called `binop`. This class represents a [monoid](https://en.wikipedia.org/wiki/Monoid), which has as its constituent parts a binary operation, a datatype for the data on which the operator is applied, and an identity element. (If we call the identity element `I` and the operator `op`, then `x = I op x`. For instance, addition's identity is zero, and multiplication's is one.) In Gridwise, we package these elements into an instance of a JS class, `BinOp`. This class then defines a number of objects that are used in WGSL code generation and CPU correctness checking.

`BinOp` is implemented in the source file binop.md. We specialize `BinOp` to particular operations (e.g., `Add`) and then further specialize it with a datatype. Many Gridwise primitives require a `BinOp` argument and the common use will be something like:

```js
const myPrimitive = new gridwisePrimitive({
  device,
  binop: BinOpAddU32,   // predefined in binop.js
  ...
```

or

```js
const datatype = "f32";
const myPrimitive = new gridwisePrimitive({
  device,
  binop: new BinOpAdd({ datatype }),  // instantiate on the fly, e.g.,
                                      // if datatype is generated at runtime
  ...
```

## What does a BinOp provide / what must a BinOp define?

We want to write primitives that work for any monoid. Other languages have more structured ways to write such code, but WGSL development in JavaScript commonly uses string-pasting to construct runtime-generated kernels. A `BinOp` provides all the text and member functions that are specific to the particular monoid we are using. These are:

- An identity element `identity`. For addition, this is 0 (independent of datatype). For multiplication, it is 1; for minimum, the largest representable value for that datatype; for maximum, the smallest representable value. Example: `this.identity = 0;`
- A CPU-side function `op`. This JavaScript function takes two arguments `a` and `b` and returns `a op b`. Because JavaScript's internal datatypes are limited, this sometimes requires judicious use of JavaScript typed arrays. (We are happy to take suggestions on how we can do this more efficiently.) Example: `this.op = (a, b) => a + b;`
- A GPU-side function declaration `wgslop`. This must define a WGSL function named `binop`. Like `op`, this function takes two arguments `a` and `b` and returns `a op b`. It can use string interpolation as appropriate and will probably have to use a datatype. Example: `this.wgslfn = fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return a+b;};`
- Four optional WGSL function names. These are "optional" in the sense that they are not a core part of `BinOp`, so most primitives will probably work if they are not specified. These are:
  - An atomic function `wgslatomic`. This should be a string that is a function name. This names the WGSL atomic function that is the atomic variant of `wgslop`. Any of [these functions](https://www.w3.org/TR/WGSL/#atomic-rmw) are appropriate, but note that (at the time of writing) WGSL atomics are only available for `i32` and `u32` datatypes. Example: `this.wgslatomic = "atomicAdd";`
  - Three subgroup functions. Note these functions apply to only a subset of operations and datatypes. Supporting anything outside of this subset requires [emulation](subgroup-strategy.html).
    - `subgroupReduceOp`, which reduces the values in a subgroup using this operation. At the time of writing, supported WGSL functions of this type are `subgroup{Add,And,Max,Min,Mul,Or,Xor}`. Example: `this.subgroupReduceOp = "subgroupAdd";`
    - `subgroupInclusiveScanOp`, which computes an inclusive scan of the values in a subgroup using this operation. At the time of writing, supported WGSL functions of this type are `subgroupInclusive{Add,Mul}`. Example: `this.subgroupInclusiveScanOp = "subgroupInclusiveAdd";`
    - `subgroupExclusiveScanOp`, which computes an exclusive scan of the values in a subgroup using this operation. At the time of writing, supported WGSL functions of this type are `subgroupExclusive{Add,Mul}`. Example: `this.subgroupExclusiveScanOp = "subgroupExclusiveAdd";`


Below is an example implementation, `BinOpAdd`, which takes an argument of `{ datatype = "..." }` that is used to specialize it.

```js
export class BinOpAdd extends BinOp {
  constructor(args) {
    super(args);
    this.identity = 0;
    if (args.datatype == "f32") {
      const f32array = new Float32Array(3);
      this.op = (a, b) => {
        f32array[1] = a;
        f32array[2] = b;
        f32array[0] = f32array[1] + f32array[2];
        return f32array[0];
      };
    } else {
      this.op = (a, b) => a + b;
    }
    switch (this.datatype) {
      case "f32":
        break;
      case "i32":
        break;
      case "u32": // fall-through OK
      default:
        this.wgslatomic = "atomicAdd"; // u32 only
        break;
    }
    this.wgslfn = `fn binop(a : ${this.datatype}, b : ${this.datatype}) -> ${this.datatype} {return a+b;}`;
    this.subgroupReduceOp = "subgroupAdd";
    this.subgroupInclusiveScanOp = "subgroupInclusiveAdd";
    this.subgroupExclusiveScanOp = "subgroupExclusiveAdd";
  }
}
```
