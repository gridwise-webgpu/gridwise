---
layout: page
title: "Random Numbers"
category: gridwise
permalink: /random-numbers/
excerpt: "Random bits for WGSL kernels: a fast keyed hash and Philox4x32, both with bit-exact JavaScript mirrors. What each is, how to use them, and when to use which."
order: 15
---

Gridwise provides random bits for WGSL kernels in `src/random.mjs`. One call, `randomBitsWGSL(rng)`, declares a small set of WGSL functions backed by one of two generators:

- **a keyed hash** (`rng = null`, the default): fast, with 32-bit seeds;
- **Philox4x32** (`rng = { key: [lo, hi], rounds }`): a counter-based generator with a published statistical pedigree, slower.

Both are pure integer arithmetic, so they give the same bits on every GPU. Each has a JavaScript mirror, `randomBitsJS(rng)`, that computes exactly the same values on the CPU, which makes GPU results checkable bit for bit.

## What you get

`randomBitsWGSL(rng)` declares these functions (prefixed `pm_`, from their origin in the permutation primitives):

| Function | Returns |
|---|---|
| `pm_rand(a, b, seed)` | 32 random bits: draw `b` of the stream identified by `a` and `seed` |
| `pm_bounded(k, a, seed, &ctr)` | a uniformly distributed integer in [0, `k`), `k` ≥ 1; reads the stream (`a`, `seed`) starting at draw `ctr` and advances `ctr` |
| `pm_mix(i, seed)` | a keyed bijection on `u32`: distinct `i` give distinct results; `pm_unmix(h, seed)` inverts it |
| `pm_t32(x)` | Chris Wellons's *triple32* integer hash, a bijection on `u32` (the building block of the hash generator and of `pm_mix`) |
| `pm_elem(i)` | Philox only: one random word per index `i`, from a counter space separate from `pm_rand`'s |

`pm_bounded` uses rejection sampling, not `% k`, so it has no modulo bias. Each attempt is accepted with probability above 1/2, and after 64 rejections it returns 0, so its result is exactly uniform except with probability below 2^-64.

### The keyed hash (`rng = null`)

`pm_rand(a, b, seed)` is `pm_t32(pm_t32(a ^ seed) + b * 0x85ebca6b + 0x27d4eb2f)`: two rounds of triple32, keyed by the stream and the seed.

- **Speed:** a handful of multiplies and shifts per draw.
- **Seed:** a runtime value, typically from a uniform, so changing it needs no new shader.
- **Streams:** a stream is identified by `a ^ seed`, so (`a` = 0, `seed` = 1) is the same stream as (`a` = 1, `seed` = 0). Keep one seed per run and use `a` to separate streams.
- **Quality:** triple32 is a carefully searched, low-bias integer hash, but this keyed construction has no published statistical analysis. In Gridwise it has been tested only through the uniformity tests of the permutation primitives, which it passes.
- **State space:** with a 32-bit seed, a computation has at most 2^32 distinct random outcomes.

### Philox4x32 (`rng = { key: [lo, hi], rounds = 10 }`)

Philox4x32 (Salmon, Moraes, Dror and Shaw, SC 2011, as in Random123) maps a 128-bit counter and a 64-bit key to 128 random bits. `pm_rand(a, b, seed)` returns word `b & 3` of the block at counter (`a`, `b >> 2`, `seed`, 0).

- **Quality:** Philox4x32-10 passes the TestU01 BigCrush battery, and Salmon et al. report that 7 rounds already pass it. 10 rounds is their recommended default, with a safety margin. Gridwise's implementation is checked against Random123's known-answer vectors.
- **Streams:** `a`, the draw index and `seed` are separate counter words, so distinct (`a`, `seed`) pairs never share a stream.
- **Key:** 64 bits, *compiled into the shader*. A new key means a new shader module and pipeline, so choose the key once (per application or per experiment) and vary the counters (`a`, `seed`) between runs instead.
- **Cost:** each call is `rounds` rounds of two 32 × 32 → 64-bit multiplies. WGSL has no 64-bit integers, so the high halves are assembled from 16-bit pieces, which makes Philox several times more expensive per word than the hash. To soften that, `pm_rand` keeps the last block per invocation: reading a stream with consecutive `b`, as `pm_bounded` does, costs one Philox call per four draws.

## How to use them

Declare the functions in a kernel and call them:

```js
import { randomBitsWGSL, randomBitsJS } from "gridwise";

const rng = null;  // or { key: [0x243f6a88, 0x85a308d3], rounds: 10 }
const kernel = /* wgsl */ `
${randomBitsWGSL(rng)}
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> seed: u32;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u) {
  var ctr = 0u;
  out[id.x] = pm_bounded(6u, id.x, seed, &ctr) + 1u;   /* a fair die roll per invocation */
}`;
```

Check the results on the CPU with the mirror, which computes the same values:

```js
const R = randomBitsJS(rng);
for (let i = 0; i < n; i++) {
  if (gpuOut[i] !== R.bounded(6, i, seed, { value: 0 }) + 1) { /* mismatch */ }
}
```

The mirror returns `{ rand(a, b, seed), bounded(k, a, seed, ctr), elem(i) }` (`elem` is `null` for the hash), matching `pm_rand`, `pm_bounded` and `pm_elem`. `pmMixJS` and `philox4x32JS` are also exported.

**Streams and counters.** Give each independent consumer its own stream `a` (an element index, a workgroup index, a group number) and read it with increasing draw indices `b`. `pm_bounded` does that for you through `ctr`. Never reuse the same (`a`, `b`, `seed`) for two purposes that should be independent.

## When to use which

| You need | Use |
|---|---|
| speed, where "looks random" is enough: testing, randomized algorithms whose correctness does not depend on the quality of the bits, visual effects, a fresh seed every run | the **keyed hash** (`rng = null`) |
| bits whose statistical quality you will claim or rely on: Monte Carlo estimates, simulations, results reported as "uniformly random" | **Philox**, 10 rounds |
| Philox's quality at lower cost, accepting a smaller safety margin | Philox with `rounds: 7` |
| more than 2^32 possible outcomes for one computation | **Philox** (64-bit key plus counters) |
| a random *bijection* on `u32`, such as distinct random keys, or a value you can map back to where it came from | **`pm_mix`** / `pm_unmix` (with either generator) |
| a uniform integer in a range | **`pm_bounded`**, never `pm_rand(...) % k` |

For example, `MultisplitPerm` uses the hash by default and Philox when given `rng`. On an Apple M1 Max it ran at about 3.6–3.95 G elements/s with the hash, about 1.7–1.9 with 7-round Philox, and about 1.3–1.4 with 10-round Philox. That's the price of Philox in a kernel that draws several random words per element.
