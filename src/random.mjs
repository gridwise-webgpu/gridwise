/**
 * Random bits for WGSL kernels, with bit-exact JavaScript mirrors for
 * validation on the CPU.
 *
 * WGSL (randomBitsWGSL(rng) declares all of these):
 *   pm_t32(x)             triple32 hash (Chris Wellons), a bijection on u32
 *   pm_mix(i, seed)       keyed invertible hash of i; pm_unmix is its inverse
 *   pm_rand(a, b, seed)   32 random bits for stream (a, seed), draw b
 *   pm_bounded(k, key, seed, &ctr)
 *                         uniform integer in [0, k), exact (rejection
 *                         sampling); ctr advances the stream
 *   pm_elem(i)            one random word per index (Philox only)
 *
 *   rng = null                         keyed hash: fast; 32-bit seeds; no
 *                                      statistical pedigree beyond testing
 *   rng = { key: [lo, hi], rounds }    Philox4x32-<rounds> (default 10;
 *                                      Salmon et al., SC 2011) under a
 *                                      64-bit key compiled into the shader
 *
 * JS: randomBitsJS(rng) returns { rand, bounded, elem } computing exactly what
 * the WGSL computes; pmMixJS, pmRandJS, pmBoundedJS, philox4x32JS.
 *
 * Origin: written for the random-permutation primitives (multisplitperm.mjs).
 */

/** triple32 (Chris Wellons) and its exact inverse, and the keyed invertible
 * hash pm_mix built from them. The inverse undoes each xorshift with a fixed
 * number of shifts (x ^ x>>s ^ x>>2s ...), with no data-dependent loop. */
const MIX_CORE_WGSL = /* wgsl */ `
fn pm_t32(x0: u32) -> u32 {
  var x = x0;
  x ^= x >> 17u; x *= 0xed5ad4bbu;
  x ^= x >> 11u; x *= 0xac4c1b51u;
  x ^= x >> 15u; x *= 0x31848babu;
  x ^= x >> 14u;
  return x;
}
fn pm_it32(x0: u32) -> u32 {
  var x = x0 ^ (x0 >> 14u) ^ (x0 >> 28u); x *= 0x32b21703u;
  x = x ^ (x >> 15u) ^ (x >> 30u);        x *= 0x469e0db1u;
  x = x ^ (x >> 11u) ^ (x >> 22u);        x *= 0x79a85073u;
  return x ^ (x >> 17u);
}
fn pm_mix(i: u32, seed: u32) -> u32 {
  return pm_t32(pm_t32(i ^ seed) + (seed * 0x9e3779b9u + 0x7f4a7c15u));
}
fn pm_unmix(h: u32, seed: u32) -> u32 {
  return pm_it32(pm_it32(h) - (seed * 0x9e3779b9u + 0x7f4a7c15u)) ^ seed;
}
`;

/** pm_bounded(k, key, seed, &ctr): uniform integer in [0, k), exact by
 * rejection on pm_rand; ctr advances the stream (key, seed). */
const BOUNDED_WGSL = /* wgsl */ `
fn pm_bounded(k: u32, key: u32, seed: u32, ctr: ptr<function, u32>) -> u32 {
  var m = k - 1u;
  m |= m >> 1u; m |= m >> 2u; m |= m >> 4u; m |= m >> 8u; m |= m >> 16u;
  for (var a = 0u; a < 64u; a++) {
    let r = pm_rand(key, *ctr, seed) & m;
    *ctr += 1u;
    if (r < k) { return r; }
  }
  return 0u;
}
`;

/** Philox4x32-10 (Salmon et al., SC 2011), checked against Random123's
 * known-answer vectors. WGSL has no 64-bit integers, so the high half of
 * each 32 x 32 product is assembled from 16-bit pieces. */
export const PHILOX_WGSL = /* wgsl */ `
fn philox_mulhi(a: u32, b: u32) -> u32 {
  let al = a & 0xffffu; let ah = a >> 16u; let bl = b & 0xffffu; let bh = b >> 16u;
  let lh = al * bh; let hl = ah * bl;
  let mid = ((al * bl) >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
  return ah * bh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}
fn philox4x32(c: vec4u, k: vec2u) -> vec4u {
  var x = c;
  var key = k;
  for (var r = 0u; r < PHILOX_ROUNDS; r++) {
    let h0 = philox_mulhi(0xD2511F53u, x.x); let l0 = 0xD2511F53u * x.x;
    let h1 = philox_mulhi(0xCD9E8D57u, x.z); let l1 = 0xCD9E8D57u * x.z;
    x = vec4u(h1 ^ x.y ^ key.x, l1, h0 ^ x.w ^ key.y, l0);
    key += vec2u(0x9E3779B9u, 0xBB67AE85u);
  }
  return x;
}
`;

/** WGSL for the random bits every kernel draws from:
 *   rng = null      pm_rand is a keyed hash (two triple32 rounds), keyed by
 *                   32-bit stream values
 *   rng = { key: [lo, hi], rounds = 10 }
 *                   Philox4x32-<rounds> under the 64-bit key (compiled into the
 *                   shader). pm_rand(a, b, s) is word (b & 3) of the block at
 *                   counter (a, b >> 2, s, 0); the last block is kept per
 *                   invocation, so a stream read with consecutive b (as
 *                   pm_bounded does) costs one Philox call per four draws.
 *                   pm_elem(i) = word 0 of the block at (i, 0, 0, 1).
 * Both also define pm_t32, pm_mix, pm_unmix and pm_bounded. */
export function randomBitsWGSL(rng = null) {
  if (!rng) return MIX_CORE_WGSL + /* wgsl */ `
fn pm_rand(a: u32, b: u32, seed: u32) -> u32 { return pm_t32(pm_t32(a ^ seed) + b * 0x85ebca6bu + 0x27d4eb2fu); }
` + BOUNDED_WGSL;
  const [lo, hi] = rng.key.map((v) => (v >>> 0) + "u");
  return MIX_CORE_WGSL + `const PHILOX_ROUNDS = ${rng.rounds ?? 10}u;` + PHILOX_WGSL + /* wgsl */ `
const PM_KEY = vec2u(${lo}, ${hi});
var<private> pm_blockTag: vec3u;
var<private> pm_block: vec4u;
var<private> pm_haveBlock: bool = false;
fn pm_rand(a: u32, b: u32, seed: u32) -> u32 {
  let tag = vec3u(a, b >> 2u, seed);
  if (!pm_haveBlock || any(tag != pm_blockTag)) {
    pm_block = philox4x32(vec4u(tag, 0u), PM_KEY);
    pm_blockTag = tag;
    pm_haveBlock = true;
  }
  return pm_block[b & 3u];
}
fn pm_elem(i: u32) -> u32 { return philox4x32(vec4u(i, 0u, 0u, 1u), PM_KEY).x; }
` + BOUNDED_WGSL;
}

/** The default random bits (keyed hash, 32-bit streams). */
export const PERMUTE_MIX_WGSL = randomBitsWGSL(null);

/* ================================================================ JS mirrors (bit-exact) */

export const pmT32JS = (x) => {
  x ^= x >>> 17; x = Math.imul(x, 0xed5ad4bb); x ^= x >>> 11; x = Math.imul(x, 0xac4c1b51);
  x ^= x >>> 15; x = Math.imul(x, 0x31848bab); x ^= x >>> 14; return x >>> 0;
};
export const pmMixJS = (i, s) => pmT32JS((pmT32JS((i ^ s) >>> 0) + ((Math.imul(s, 0x9e3779b9) + 0x7f4a7c15) >>> 0)) >>> 0);
export const pmRandJS = (a, b, s) => pmT32JS((pmT32JS((a ^ s) >>> 0) + Math.imul(b, 0x85ebca6b) + 0x27d4eb2f) >>> 0);
/** pm_bounded: uniform in [0, k); ctr = { value } is advanced like the WGSL counter */
export function pmBoundedJS(k, key, seed, ctr = { value: 0 }, rand = pmRandJS) {
  let m = k - 1; m |= m >>> 1; m |= m >>> 2; m |= m >>> 4; m |= m >>> 8; m |= m >>> 16;
  for (let a = 0; a < 64; a++) {
    const r = (rand(key, ctr.value, seed) & m) >>> 0;
    ctr.value++;
    if (r < k) return r;
  }
  return 0;
}
function mulhiJS(a, b) {
  const al = a & 0xffff, ah = a >>> 16, bl = b & 0xffff, bh = b >>> 16;
  const lh = al * bh, hl = ah * bl, mid = ((al * bl) >>> 16) + (lh & 0xffff) + (hl & 0xffff);
  return (ah * bh + (lh >>> 16) + (hl >>> 16) + (mid >>> 16)) >>> 0;
}
export function philox4x32JS(c, k, rounds = 10) {
  let [x0, x1, x2, x3] = c.map((v) => v >>> 0), [k0, k1] = k.map((v) => v >>> 0);
  for (let r = 0; r < rounds; r++) {
    const h0 = mulhiJS(0xD2511F53, x0), l0 = Math.imul(0xD2511F53, x0) >>> 0;
    const h1 = mulhiJS(0xCD9E8D57, x2), l1 = Math.imul(0xCD9E8D57, x2) >>> 0;
    [x0, x1, x2, x3] = [(h1 ^ x1 ^ k0) >>> 0, l1, (h0 ^ x3 ^ k1) >>> 0, l0];
    k0 = (k0 + 0x9E3779B9) >>> 0; k1 = (k1 + 0xBB67AE85) >>> 0;
  }
  return [x0, x1, x2, x3];
}
/** JS mirror of randomBitsWGSL(rng): { rand(a, b, s), bounded(k, key, s, ctr), elem(i) } */
export function randomBitsJS(rng = null) {
  if (!rng) return { rand: pmRandJS, bounded: (k, key, s, ctr) => pmBoundedJS(k, key, s, ctr), elem: null };
  const key = rng.key, rounds = rng.rounds ?? 10;
  const rand = (a, b, s) => philox4x32JS([a, b >>> 2, s, 0], key, rounds)[b & 3];
  return { rand, bounded: (k, kk, s, ctr) => pmBoundedJS(k, kk, s, ctr, rand), elem: (i) => philox4x32JS([i, 0, 0, 1], key, rounds)[0] };
}
