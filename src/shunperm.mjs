/**
 * ShunPerm: a uniformly random permutation of 0..n-1 by parallel Fisher-Yates
 * (deterministic reservations), from Shun, Gu, Blelloch, Fineman and Gibbons,
 * "Sequential Random Permutation, List Contraction and Tree Contraction are
 * Highly Parallel" (SODA 2015).
 *
 *   const perm = new ShunPerm({ device, n, seed });
 *   await perm.execute({ outputBuffer });   // outputBuffer: n u32s
 *   await perm.roundsUsed();                // rounds needed (Infinity if maxRounds ran out)
 *
 * Sequential Fisher-Yates: A = identity; for i = n-1 down to 1:
 * swap(A[i], A[H(i)]) with H(i) uniform in [0, i]. Iteration i must come
 * after every earlier iteration (larger index) that touches location i or
 * H(i). Shun et al. show that this dependence structure has depth O(log n)
 * with high probability, so the iterations can run in rounds:
 *
 *   reserve  every pending iteration i does atomicMax(R[i], i) and
 *            atomicMax(R[H(i)], i)
 *   commit   iteration i owns both locations (R[i] == R[H(i)] == i), so every
 *            earlier iteration touching them has committed: swap, release
 *            both reservations. Otherwise append i to the next round's list.
 *
 * The output is bit-identical to sequential Fisher-Yates with the same H
 * (reference()), on any GPU, in any run: the only atomics are atomicMax
 * (order-independent) and the list append (the list's order does not
 * matter). Work is about 3.5 n iteration attempts; the rounds are
 * random-access bound.
 *
 * GPU structure: the number of rounds is only known on the GPU, so all
 * maxRounds rounds are recorded up front. Before each round a one-invocation
 * kernel sizes the round's indirect dispatch from the number of iterations
 * still pending; rounds after the last dispatch no workgroups. No CPU
 * round trips. The kernels cannot be repeated in place (a round appends to
 * a list), so trials must be 1.
 *
 * Options: n (required), seed (default 0x2545f491), rng (null: keyed hash;
 * { key: [lo, hi], rounds } for Philox4x32, see random.mjs), maxRounds
 * (default ceil(3 log2 n) + 16).
 */
import { BasePrimitive, Kernel, AllocateBuffer, ClearBuffer, UniformStruct } from "./primitive.mjs";
import { randomBitsWGSL, randomBitsJS } from "./random.mjs";

const WORKGROUP_SIZE = 256;

export class ShunPerm extends BasePrimitive {
  constructor(args) {
    super(args);
    if (!(Number.isInteger(this.n) && this.n >= 2)) {
      throw new Error("ShunPerm: requires an integer n >= 2");
    }
    this.knownBuffers = ["outputBuffer"];
    this.workgroupSize = WORKGROUP_SIZE;
    this.seed = (this.seed ?? 0x2545f491) >>> 0;
    this.rng ??= null;
    this.maxRounds ??= Math.ceil(3 * Math.log2(this.n)) + 16;
    this.singleTrialOnly = true; /* each round appends to a list: not repeatable in place */
    this.params = new UniformStruct("SH", { n: "u32", seed: "u32" });
    if (!("label" in args)) this.label += ` [ n: ${this.n} ]`;
  }

  /** the module with init, reserve and commit */
  shunWGSL = () => /* wgsl */ `
${this.fnDeclarations.include("computeLinearizedGridParametersSplit")}
${randomBitsWGSL(this.rng)}

fn swapTarget(i: u32) -> u32 { var ctr = 0u; return pm_bounded(i + 1u, i, sh.seed, &ctr); }
/* the idx-th pending iteration of round r, or 0 (never pending) past the end;
 * round 1's list is implicit: iterations 1..n-1 */
fn pending(idx: u32, r: u32) -> u32 {
  if (idx >= atomicLoad(&counts[r - 1u])) { return 0u; }
  if (r == 1u) { return idx + 1u; }
  return listIn[idx];
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn init(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  if (gid < sh.n) { A[gid] = gid; }
  if (gid == 0u) { atomicStore(&counts[0], sh.n - 1u); state[0] = 0u; }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn reserve(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let i = pending(gid, state[0]);
  if (i == 0u) { return; }
  atomicMax(&R[i], i);
  atomicMax(&R[swapTarget(i)], i);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn commit(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let r = state[0];
  let i = pending(gid, r);
  if (i == 0u) { return; }
  let h = swapTarget(i);
  if (atomicLoad(&R[i]) == i && atomicLoad(&R[h]) == i) {
    let t = A[i]; A[i] = A[h]; A[h] = t;
    atomicStore(&R[i], 0u);
    atomicStore(&R[h], 0u);
  } else {
    listOut[atomicAdd(&counts[r], 1u)] = i;
  }
}`;

  /** one invocation: advance the round and size its indirect dispatch */
  roundWGSL = () => /* wgsl */ `
${this.fnDeclarations.include(["indirectDispatch", { name: "setRoundSize", buffer: "args" }])}
@compute @workgroup_size(1)
fn main() {
  let r = state[0] + 1u;
  state[0] = r;
  setRoundSize((counts[r - 1u] + ${WORKGROUP_SIZE - 1}u) / ${WORKGROUP_SIZE}u);
}`;

  compute() {
    const outputLength = this.getBuffer("outputBuffer")?.size / 4;
    if (!(outputLength >= this.n)) {
      throw new Error(`ShunPerm: outputBuffer must hold n = ${this.n} u32s`);
    }
    const n = this.n;
    const words = (k) => Math.max(16, k * 4);
    /* round r reads lists[(r + 1) % 2] and appends to lists[r % 2] */
    const lists = ["listA", "listB"];
    const shunBuffers = (r) => [
      { name: "A", buffer: "outputBuffer", access: "read_write", type: "array<u32>" },
      { name: "R", buffer: "reservations", access: "read_write", type: "array<atomic<u32>>" } /* 0 = free */,
      { name: "listIn", buffer: lists[(r + 1) % 2], access: "read_write", type: "array<u32>" },
      { name: "listOut", buffer: lists[r % 2], access: "read_write", type: "array<u32>" },
      { name: "counts", access: "read_write", type: "array<atomic<u32>>" } /* counts[r]: pending after round r */,
      { name: "state", access: "read_write", type: "array<u32>" } /* state[0]: the current round */,
      { name: "sh", buffer: "shunParams", uniform: this.params },
    ];
    const actions = [
      this.params.allocate("shunParams", { n, seed: this.seed }),
      new AllocateBuffer({ label: "reservations", size: words(n), clearBufferOnReuse: false }),
      new AllocateBuffer({ label: "listA", size: words(n), clearBufferOnReuse: false }),
      new AllocateBuffer({ label: "listB", size: words(n), clearBufferOnReuse: false }),
      new AllocateBuffer({ label: "counts", size: words(this.maxRounds + 1), clearBufferOnReuse: false }),
      new AllocateBuffer({ label: "state", size: 16, clearBufferOnReuse: false }),
      new AllocateBuffer({
        label: "args",
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC,
        clearBufferOnReuse: false,
      }),
      new ClearBuffer({ label: "reservations" }),
      new ClearBuffer({ label: "counts" }),
      new Kernel({
        kernel: this.shunWGSL,
        entryPoint: "init",
        label: "ShunPerm init",
        buffers: shunBuffers(0),
        getDispatchGeometry: () => this.getSimpleDispatchGeometry({ workgroupCount: Math.ceil(n / WORKGROUP_SIZE) }),
      }),
    ];
    for (let r = 1; r <= this.maxRounds; r++) {
      actions.push(
        new Kernel({
          kernel: this.roundWGSL,
          label: `ShunPerm round ${r} size`,
          buffers: [
            { name: "counts", access: "read_write", type: "array<u32>" },
            { name: "state", access: "read_write", type: "array<u32>" },
            { name: "args", access: "read_write", type: "array<u32>" },
          ],
          getDispatchGeometry: () => [1],
        }),
        ...["reserve", "commit"].map(
          (entryPoint) =>
            new Kernel({
              kernel: this.shunWGSL,
              entryPoint,
              label: `ShunPerm round ${r} ${entryPoint}`,
              buffers: shunBuffers(r),
              indirect: "args",
            })
        )
      );
    }
    return actions;
  }

  /** Rounds the last execute() needed (Infinity if iterations were still
   * pending after maxRounds: the output is then not a finished shuffle). */
  async roundsUsed() {
    const counts = this.getBuffer("counts").buffer.buffer;
    const bytes = (this.maxRounds + 1) * 4;
    const readback = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(counts, 0, readback, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const c = new Uint32Array(readback.getMappedRange().slice(0));
    readback.destroy();
    for (let r = 0; r <= this.maxRounds; r++) if (c[r] === 0) return r;
    return Infinity;
  }

  /** sequential Fisher-Yates with the same swap targets: what execute() writes */
  reference() {
    const A = Uint32Array.from({ length: this.n }, (_, i) => i);
    const R = randomBitsJS(this.rng);
    for (let i = this.n - 1; i >= 1; i--) {
      const h = R.bounded(i + 1, i, this.seed, { value: 0 });
      const t = A[i];
      A[i] = A[h];
      A[h] = t;
    }
    return A;
  }

  /** "" if the output equals reference(), else the first differences */
  validate = (args = {}) => {
    const out = args.outputBuffer?.cpuBuffer ?? args.outputBuffer ?? this.getBuffer("outputBuffer").cpuBuffer;
    const ref = this.reference();
    const problems = [];
    for (let i = 0; i < this.n && problems.length < 5; i++) {
      if (out[i] !== ref[i]) problems.push(`position ${i}: expected ${ref[i]}, got ${out[i]}`);
    }
    return problems.length ? "\n" + problems.join("\n") : "";
  };
}
