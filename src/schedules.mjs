/**
 * Load-balancing schedules for SegmentedReduce: which invocation handles
 * which atoms, separate from what the work is (Osama, Porumbescu and Owens,
 * PPoPP 2023). A schedule knows nothing about the work: it writes the loops,
 * and calls the work through WGSL that SegmentedReduce defines in the same
 * module:
 *
 *   binop(a, b), IDENTITY      the reduction
 *   tileStartOf(t) -> u32      tile t's first atom, t in [0, sr.tiles]
 *   atomValue(atom, tile)      an atom's value
 *   emitTile(tile, value)      consume a tile's result (once per tile)
 *   sr.tiles, sr.atoms, sr.workgroups   (uniform)
 *   WORKGROUP_SIZE invocations per workgroup; computeLinearizedGridParametersSplit
 *
 * A schedule is a class with:
 *
 *   workgroups({ tiles, atoms })   workgroups of its main kernel
 *   wgsl(sr)                       WGSL of its kernels (entry points)
 *   buffers(sr)                    buffers its kernels bind beyond the
 *                                  caller's (optional)
 *   actions(sr, { workgroups })    the actions of one execute; sr.kernel()
 *                                  makes a Kernel of one entry point
 *
 * To add one, write the class and pass an instance (schedule: new Mine())
 * or register a name (SegmentedReduce.schedules.mine = Mine).
 */

export const WORKGROUP_SIZE = 256;

export class Schedule {
  constructor(options = {}) {
    Object.assign(this, options);
  }
  toString() {
    return this.name;
  }
  buffers() {
    return [];
  }
}

/** One invocation per tile, striding over the tiles. Balanced only when
 * tiles are about the same size: one long tile holds up the kernel. */
export class ThreadMapped extends Schedule {
  name = "thread";
  workgroups({ tiles }) {
    return Math.ceil(tiles / WORKGROUP_SIZE);
  }
  wgsl(sr) {
    return /* wgsl */ `
@compute @workgroup_size(${WORKGROUP_SIZE})
fn thread_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${sr.fnDeclarations.computeLinearizedGridParametersSplit()}
  for (var tile = gid; tile < sr.tiles; tile += totalThreadCount) {
    var acc = IDENTITY;
    let end = tileStartOf(tile + 1u);
    for (var atom = tileStartOf(tile); atom < end; atom++) { acc = binop(acc, atomValue(atom, tile)); }
    emitTile(tile, acc);
  }
}`;
  }
  actions(sr, { workgroups }) {
    return [sr.kernel({ entryPoint: "thread_main", workgroups })];
  }
}

/** One workgroup per tile, striding over the tiles; its invocations stride
 * over the tile's atoms, then reduce in workgroup memory. Balanced for long
 * tiles of about the same size; a short tile leaves most of a workgroup
 * idle. (The paper's block-mapped schedule.) */
export class WorkgroupMapped extends Schedule {
  name = "workgroup";
  workgroups({ tiles }) {
    return Math.min(tiles, 65536);
  }
  wgsl(sr) {
    const W = WORKGROUP_SIZE;
    return /* wgsl */ `
var<workgroup> partial: array<${sr.datatype}, ${W}>;
@compute @workgroup_size(${W})
fn workgroup_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${sr.fnDeclarations.computeLinearizedGridParametersSplit()}
  let lidx = builtinsNonuniform.lidx;
  /* wgid, workgroupCount and sr.tiles are uniform, so the barriers below
   * are in uniform control flow */
  for (var tile = wgid; tile < sr.tiles; tile += workgroupCount) {
    var acc = IDENTITY;
    let end = tileStartOf(tile + 1u);
    for (var atom = tileStartOf(tile) + lidx; atom < end; atom += ${W}u) { acc = binop(acc, atomValue(atom, tile)); }
    partial[lidx] = acc;
    workgroupBarrier();
    /* tree reduction, keeping order: each step combines neighbors */
    for (var stride = 1u; stride < ${W}u; stride <<= 1u) {
      var combined = IDENTITY;
      let mine = lidx % (2u * stride) == 0u;
      if (mine) { combined = binop(partial[lidx], partial[lidx + stride]); }
      workgroupBarrier();
      if (mine) { partial[lidx] = combined; }
      workgroupBarrier();
    }
    if (lidx == 0u) { emitTile(tile, partial[0]); }
    workgroupBarrier();
  }
}`;
  }
  actions(sr, { workgroups }) {
    return [sr.kernel({ entryPoint: "workgroup_main", workgroups })];
  }
}

/** The merge path of the tile ends and the atom indices (Merrill and
 * Garland, SC 2016): tiles + atoms steps, split evenly, so every invocation
 * does itemsPerThread steps (default 4) whatever the tile sizes.
 *
 * A tile split between invocations is combined by a segmented scan in the
 * workgroup. Between workgroups, combining is itself a segmented reduce: each
 * workgroup's carry (its partial value of the tile unfinished at its end) is
 * an atom, and each workgroup is a tile, holding the run of carries in the
 * tile it ends. So a second SegmentedReduce (merge-path too; recursively,
 * until one workgroup suffices) combines them into the finishing workgroups'
 * head values, however many workgroups a tile spans, and a last kernel emits
 * the heads. Each tile is emitted exactly once, with its whole value, so emit
 * may be any WGSL (it is never asked to combine). See
 * docs/segmented-reduce.md for a worked example. */
export class MergePath extends Schedule {
  name = "mergepath";
  constructor(options = {}) {
    super({ itemsPerThread: 4, ...options });
    if (!(Number.isInteger(this.itemsPerThread) && this.itemsPerThread >= 1)) {
      throw new Error("MergePath: itemsPerThread must be a positive integer");
    }
  }
  workgroups({ tiles, atoms }) {
    return Math.ceil((tiles + atoms) / (WORKGROUP_SIZE * this.itemsPerThread));
  }
  /* per-level names, so that each level of fixup binds its parent's carries
   * under different names from its own */
  carryNames(sr) {
    return { C: `wgTile${sr.level}`, V: `wgValue${sr.level}` };
  }
  buffers(sr) {
    const { C, V } = this.carryNames(sr);
    return [
      { name: C, access: "read_write", type: "array<u32>" },
      { name: V, access: "read_write", type: `array<${sr.datatype}>` },
    ];
  }

  /* Step d of the path either consumes an atom (atom < the current tile's
   * end) or ends the current tile. Invocation i takes steps [i K, (i + 1) K);
   * a binary search finds where its path starts. */
  wgsl(sr) {
    const T = sr.datatype;
    const W = WORKGROUP_SIZE;
    const { C, V } = this.carryNames(sr);
    return /* wgsl */ `
const K = ${this.itemsPerThread}u;

/* the carried (unfinished) tile and value of each invocation, then their
 * segmented inclusive scan */
var<workgroup> scanTile: array<u32, ${W}>;
var<workgroup> scanValue: array<${T}, ${W}>;
var<workgroup> wgStart: u32;   /* the tile the workgroup's path starts in */
var<workgroup> headDone: u32;  /* did the workgroup end it? */
var<workgroup> headValue: ${T};

@compute @workgroup_size(${W})
fn mergepath_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${sr.fnDeclarations.computeLinearizedGridParametersSplit()}
  if (wgid >= sr.workgroups) { return; }
  let lidx = builtinsNonuniform.lidx;
  let total = sr.tiles + sr.atoms;
  let diag = min(gid * K, total);
  /* where this invocation's path starts: the most tiles ended (tile) such
   * that tile ends come no later than the atoms before them */
  var lo = select(0u, diag - sr.atoms, diag > sr.atoms);
  var hi = min(diag, sr.tiles);
  while (lo < hi) {
    let mid = (lo + hi) / 2u;
    if (tileStartOf(mid + 1u) <= diag - mid - 1u) { lo = mid + 1u; } else { hi = mid; }
  }
  var tile = lo;
  var atom = diag - lo;
  var tileEnd = select(sr.atoms, tileStartOf(tile + 1u), tile < sr.tiles);
  let stepEnd = min(diag + K, total);
  var acc = IDENTITY;
  var firstDone = false;   /* ended the tile it started in? */
  var firstValue = IDENTITY;
  for (var d = diag; d < stepEnd; d++) {
    if (atom < tileEnd) {
      acc = binop(acc, atomValue(atom, tile));
      atom++;
    } else {
      /* end of tile: the first tile ended may have started in an earlier
       * invocation, so it waits for the scan; later ones are whole */
      if (!firstDone) { firstDone = true; firstValue = acc; } else { emitTile(tile, acc); }
      acc = IDENTITY;
      tile++;
      tileEnd = select(sr.atoms, tileStartOf(tile + 1u), tile < sr.tiles);
    }
  }
  if (lidx == 0u) {
    wgStart = lo;
    headDone = 0u;
  }
  /* segmented inclusive scan of the carried values, keyed by tile: tiles are
   * nondecreasing across invocations, so a tile's carries are contiguous */
  scanTile[lidx] = tile;
  scanValue[lidx] = acc;
  workgroupBarrier();
  for (var off = 1u; off < ${W}u; off <<= 1u) {
    let joins = lidx >= off && scanTile[lidx - off] == scanTile[lidx];
    var earlier = IDENTITY;
    if (joins) { earlier = scanValue[lidx - off]; }
    workgroupBarrier();
    if (joins) { scanValue[lidx] = binop(earlier, scanValue[lidx]); }
    workgroupBarrier();
  }
  if (firstDone) {
    /* the tile this invocation started in: add what earlier invocations of
     * this workgroup carried in it */
    var value = firstValue;
    if (lidx > 0u && scanTile[lidx - 1u] == lo) { value = binop(scanValue[lidx - 1u], value); }
    if (wgid > 0u && lo == wgStart) {
      /* the workgroup's first tile (one invocation ends it), possibly begun
       * in earlier workgroups: the fixup adds theirs and emits it */
      headDone = 1u;
      headValue = value;
    } else {
      emitTile(lo, value);
    }
  }
  workgroupBarrier();
  if (lidx == ${W - 1}u) {
    /* what this workgroup carries out (to the tile unfinished at its end),
     * and its first tile's value if it ended here */
    ${C}[wgid] = scanTile[${W - 1}u];
    ${V}[wgid] = scanValue[${W - 1}u];
    ${C}[sr.workgroups + wgid] = headDone;
    ${V}[sr.workgroups + wgid] = headValue;
  }
}

/* after the fixup has written each head's whole value: emit the heads */
@compute @workgroup_size(${W})
fn heads_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${sr.fnDeclarations.computeLinearizedGridParametersSplit()}
  let w = gid;
  if (w == 0u || w >= sr.workgroups) { return; }
  if (${C}[sr.workgroups + w] == 1u) {
    emitTile(${C}[w - 1u], ${V}[2u * sr.workgroups + w]);
  }
}`;
  }

  actions(sr, { workgroups }) {
    const { C, V } = this.carryNames(sr);
    const actions = [
      /* per workgroup: carried tile, then head flag; carried value, then
       * head value, then the head's whole value (written by the fixup, not in
       * place, so that every kernel can be repeated: trials) */
      sr.allocate({ label: C, size: Math.max(16, 8 * workgroups) }),
      sr.allocate({ label: V, size: Math.max(16, 12 * workgroups) }),
      sr.kernel({ entryPoint: "mergepath_main", workgroups, label: "path" }),
    ];
    /* one workgroup emits every tile itself; more need the fixup */
    if (workgroups > 1) {
      const fixup = this.fixup(sr);
      Object.assign(fixup, { tiles: workgroups, atoms: workgroups });
      actions.push(
        sr.run(fixup, [C, V]),
        sr.kernel({ entryPoint: "heads_main", workgroups: Math.ceil(workgroups / WORKGROUP_SIZE), label: "heads" })
      );
    }
    return actions;
  }

  /** The fixup: a SegmentedReduce over sr's W workgroup carries, whose
   * tiles are sr's workgroups. Tile w's atoms are the carries of the run (the
   * same carried tile) that ends at workgroup w - 1, if w ends that tile (w's
   * head); runs start where the carried tiles, which are nondecreasing, first
   * equal w - 1's. Its emit adds the run's carries before w's head value,
   * into a third slot (never in place: a repeated kernel would add them
   * twice). (Tile W - 1 also takes the last workgroup's carry, past the last
   * tile: the identity.) It binds only sr's carry buffers, so every level
   * binds the same few. */
  fixup(sr) {
    const { C, V } = this.carryNames(sr);
    const head = (w) => `${V}[sr.tiles + ${w}]`;
    return sr.child("fixup", {
      label: `${sr.label} fixup`,
      schedule: new MergePath({ itemsPerThread: this.itemsPerThread }),
      level: sr.level + 1,
      tiles: 1,
      atoms: 0,
      buffers: [
        { name: C, type: "array<u32>" },
        { name: V, type: `array<${sr.datatype}>`, access: "read_write" },
      ],
      wgsl: /* wgsl */ `
fn runStart(t: u32) -> u32 {
  if (t == 0u) { return 0u; }
  if (t >= sr.tiles) { return sr.tiles; }
  let key = ${C}[t - 1u];
  var lo = 0u;
  var hi = t - 1u;
  while (lo < hi) {
    let mid = (lo + hi) / 2u;
    if (${C}[mid] < key) { lo = mid + 1u; } else { hi = mid; }
  }
  return lo;
}`,
      tileStart: (t) => `runStart(${t})`,
      atom: (a) => `${V}[${a}]`,
      emit: (w, v) => `if (${w} > 0u && ${C}[sr.tiles + ${w}] == 1u) { ${V}[2u * sr.tiles + ${w}] = binop(${v}, ${head(w)}); }`,
    });
  }
}
