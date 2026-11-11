/**
 * SegmentedReduce: a load-balanced reduction of each segment of a CSR-style
 * array, with the load-balancing schedule separate from the work.
 *
 * The work is described as in Osama, Porumbescu and Owens, "A Programming
 * Model for GPU Load Balancing" (PPoPP 2023): atoms (units of work of equal
 * cost, e.g. a matrix's nonzeros) grouped into tiles (segments of any length,
 * e.g. its rows), given by an offsets array: tile t holds atoms
 * [offsets[t], offsets[t + 1]). For every tile, SegmentedReduce combines the
 * values of its atoms with a binop and emits the result:
 *
 *   result[t] = binop over a in tile t of atom(a, t)   (identity if t is empty)
 *
 * The caller says what an atom's value is and what to do with a tile's result
 * as JS functions that return WGSL, called with the names of the WGSL
 * variables that hold the atom and tile index: Gridwise's stand-in for the C++
 * lambdas the paper uses. A sparse matrix-vector product (y = Ax, CSR):
 *
 *   const spmv = new SegmentedReduce({
 *     device, schedule: "mergepath", binop: new BinOpAdd({ datatype: "f32" }),
 *     tiles: rows, atoms: nnz,
 *     buffers: [{ name: "indices", type: "array<u32>" }, { name: "values", type: "array<f32>" },
 *               { name: "x", type: "array<f32>" }, { name: "y", type: "array<f32>", access: "read_write" }],
 *     atom: (a) => `values[${a}] * x[indices[${a}]]`,
 *     emit: (t, v) => `y[${t}] = ${v};`,
 *   });
 *   await spmv.execute({ offsets, indices, values, x, y });
 *
 * Without emit, results go to outputBuffer (array of the binop's datatype).
 * Buffers named in `buffers` are bound by those names (access "read" unless
 * given). The binop may be non-commutative: atoms are combined in order.
 *
 * Options: binop (required); tiles and atoms (required: the number of tiles,
 * and offsets[tiles]; both may change between executes); atom (required),
 * emit, buffers; tileStart ((t) => WGSL expression of tile t's first atom,
 * for t in [0, tiles], instead of an offsets buffer); wgsl (more WGSL
 * declarations, e.g. functions that atom, emit or tileStart call);
 * schedule ("mergepath", default, or "thread"); itemsPerThread (mergepath:
 * path steps per invocation, default 4).
 *
 * Schedules (the paper's names):
 *   thread     one invocation per tile, striding over the tiles. Balanced
 *              only when tiles are about the same size: one long tile
 *              holds up the kernel.
 *   mergepath  the merge path of the tile ends and the atom indices (Merrill
 *              and Garland, SC 2016): tiles + atoms steps, split evenly, so
 *              every invocation does itemsPerThread steps whatever the tile
 *              sizes. A tile split between invocations is combined by a
 *              segmented scan in the workgroup. Between workgroups, combining
 *              is itself a segmented reduce: each workgroup's carry is an
 *              atom, and the tiles are the runs of carries in the same tile,
 *              each ending at the workgroup that finishes it. So a second
 *              SegmentedReduce (merge-path, on the carries; recursively, until
 *              one workgroup suffices) combines them into the finishing
 *              workgroups' head values, however many workgroups a tile spans,
 *              and a last kernel emits the heads. Each tile is emitted exactly
 *              once, with its whole value, so emit may be any WGSL (it is
 *              never asked to combine).
 */
import { BasePrimitive, Kernel, AllocateBuffer, ExecutePrimitive, UniformStruct } from "./primitive.mjs";

const WORKGROUP_SIZE = 256;
const SCHEDULES = ["mergepath", "thread"];

export class SegmentedReduce extends BasePrimitive {
  constructor(args) {
    super(args);
    if (!this.binop) throw new Error("SegmentedReduce: requires a binop (e.g. new BinOpAdd({ datatype: \"f32\" }))");
    if (typeof this.atom !== "function") throw new Error("SegmentedReduce: requires atom: (atom, tile) => WGSL expression");
    this.schedule ??= "mergepath";
    if (!SCHEDULES.includes(this.schedule)) throw new Error(`SegmentedReduce: schedule must be one of ${SCHEDULES.join(", ")}`);
    this.itemsPerThread ??= 4;
    if (!(Number.isInteger(this.itemsPerThread) && this.itemsPerThread >= 1)) throw new Error("SegmentedReduce: itemsPerThread must be a positive integer");
    this.datatype = this.binop.datatype;
    this.userBuffers = (this.buffers ?? []).map((b) => ({ access: "read", ...b }));
    if (!this.emit) {
      this.emit = (t, v) => `outputBuffer[${t}] = ${v};`;
      this.userBuffers.push({ name: "outputBuffer", access: "read_write", type: `array<${this.datatype}>` });
    }
    this.level ??= 0; /* 0 for the caller's; n for the nth level of fixup */
    this.wgTileName = `wgTile${this.level}`;
    this.wgValueName = `wgValue${this.level}`;
    this.knownBuffers = [...(this.tileStart ? [] : ["offsets"]), ...this.userBuffers.map((b) => b.name)];
    this.tileStart ??= (t) => `offsets[${t}]`;
    this.workgroupSize = WORKGROUP_SIZE;
    this.params = new UniformStruct("SRParams", { tiles: "u32", atoms: "u32", workgroups: "u32" });
    if (!("label" in args)) this.label += this.makeParamString(["schedule"]);
  }

  /** workgroups of the main kernel */
  workgroupCount() {
    return this.schedule === "thread"
      ? Math.ceil(this.tiles / WORKGROUP_SIZE)
      : Math.ceil((this.tiles + this.atoms) / (WORKGROUP_SIZE * this.itemsPerThread));
  }

  /* the caller's pieces, as WGSL functions the schedules call */
  definitions() {
    const T = this.datatype;
    return /* wgsl */ `
${this.binop.wgslfn}
const IDENTITY: ${T} = ${T}(${this.binop.identity});
${this.wgsl ?? ""}
fn tileStartOf(t: u32) -> u32 { return ${this.tileStart("t")}; }
fn atomValue(atom: u32, tile: u32) -> ${T} { return ${this.atom("atom", "tile")}; }
fn emitTile(tile: u32, value: ${T}) { ${this.emit("tile", "value")} }`;
  }

  threadWGSL = () => /* wgsl */ `
${this.fnDeclarations.include("computeLinearizedGridParametersSplit")}
${this.definitions()}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn thread_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  for (var tile = gid; tile < sr.tiles; tile += totalThreadCount) {
    var acc = IDENTITY;
    let end = tileStartOf(tile + 1u);
    for (var atom = tileStartOf(tile); atom < end; atom++) { acc = binop(acc, atomValue(atom, tile)); }
    emitTile(tile, acc);
  }
}`;

  /* Merge path: step d of the path either consumes an atom (atom < the
   * current tile's end) or ends the current tile. Invocation i takes steps
   * [i K, (i + 1) K); a binary search finds where its path starts. */
  mergepathWGSL = () => {
    const T = this.datatype;
    const W = WORKGROUP_SIZE;
    return /* wgsl */ `
${this.fnDeclarations.include("computeLinearizedGridParametersSplit")}
${this.definitions()}
const K = ${this.itemsPerThread}u;
const NONE = 0xffffffffu;

/* the carried (unfinished) tile and value of each invocation, then their
 * segmented inclusive scan */
var<workgroup> scanTile: array<u32, ${W}>;
var<workgroup> scanValue: array<${T}, ${W}>;
var<workgroup> wgStart: u32;   /* the tile the workgroup's path starts in */
var<workgroup> headDone: u32;  /* did the workgroup end it? */
var<workgroup> headValue: ${T};

@compute @workgroup_size(${W})
fn mergepath_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
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
    ${this.wgTileName}[wgid] = scanTile[${W - 1}u];
    ${this.wgValueName}[wgid] = scanValue[${W - 1}u];
    ${this.wgTileName}[sr.workgroups + wgid] = headDone;
    ${this.wgValueName}[sr.workgroups + wgid] = headValue;
  }
}

/* after the fixup has added the earlier workgroups' carries into each head
 * value: emit the heads */
@compute @workgroup_size(${W})
fn heads_main(builtinsUniform: BuiltinsUniform, builtinsNonuniform: BuiltinsNonuniform) {
  ${this.fnDeclarations.computeLinearizedGridParametersSplit()}
  let w = gid;
  if (w == 0u || w >= sr.workgroups) { return; }
  if (${this.wgTileName}[sr.workgroups + w] == 1u) {
    emitTile(${this.wgTileName}[w - 1u], ${this.wgValueName}[2u * sr.workgroups + w]);
  }
}`;
  };

  /** The fixup: a SegmentedReduce over this one's W workgroup carries,
   * whose tiles are this one's workgroups. Tile w's atoms are the carries of
   * the run (the same carried tile) that ends at workgroup w - 1, if w ends
   * that tile (w's head); runs start where the carried tiles, which are
   * nondecreasing, first equal w - 1's. Its emit adds the run's carries
   * before w's head value, into a third slot (never in place: a repeated
   * kernel would add them twice). (Tile W - 1 also takes the last
   * workgroup's carry, past the last tile: the identity.) It reads and writes
   * only this level's carry buffers, so every level binds the same few. */
  fixupPrimitive() {
    const C = this.wgTileName;
    const V = this.wgValueName;
    const head = (w) => `${V}[sr.tiles + ${w}]`;
    this.fixup ??= new SegmentedReduce({
      device: this.device,
      adapter: this.adapter,
      label: `${this.label} fixup`,
      binop: this.binop,
      schedule: "mergepath",
      itemsPerThread: this.itemsPerThread,
      level: this.level + 1,
      tiles: 1,
      atoms: 0,
      buffers: [
        { name: C, type: "array<u32>" },
        { name: V, type: `array<${this.datatype}>`, access: "read_write" },
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
    return this.fixup;
  }

  compute() {
    if (!(Number.isInteger(this.tiles) && this.tiles >= 1)) throw new Error(`SegmentedReduce: tiles (${this.tiles}) must be a positive integer`);
    if (!(Number.isInteger(this.atoms) && this.atoms >= 0)) throw new Error(`SegmentedReduce: atoms (${this.atoms}) must be a nonnegative integer`);
    if (this.tiles + this.atoms >= 2 ** 32 - WORKGROUP_SIZE * this.itemsPerThread) throw new Error("SegmentedReduce: tiles + atoms must fit in 32 bits");
    const workgroups = this.workgroupCount();
    const T = this.datatype;
    const buffers = [
      ...(this.knownBuffers.includes("offsets") ? [{ name: "offsets", access: "read", type: "array<u32>" }] : []),
      ...this.userBuffers,
      ...(this.schedule === "mergepath"
        ? [
            { name: this.wgTileName, access: "read_write", type: "array<u32>" },
            { name: this.wgValueName, access: "read_write", type: `array<${T}>` },
          ]
        : []),
      { name: "sr", buffer: "srParams", uniform: this.params },
    ];
    const dispatch = (count) => () => this.getSimpleDispatchGeometry({ workgroupCount: count });
    const label = `SegmentedReduce (${this.schedule})`;
    const actions = [this.params.allocate("srParams", { tiles: this.tiles, atoms: this.atoms, workgroups })];
    if (this.schedule === "thread") {
      actions.push(
        new Kernel({ kernel: this.threadWGSL, entryPoint: "thread_main", label, buffers, getDispatchGeometry: dispatch(workgroups) })
      );
    } else {
      actions.push(
        /* per workgroup: carried tile, then head flag; carried value, then
         * head value, then the head's whole value (written by the fixup, not
         * in place, so that every kernel can be repeated: trials) */
        new AllocateBuffer({ label: this.wgTileName, size: Math.max(16, 8 * workgroups), clearBufferOnReuse: false }),
        new AllocateBuffer({ label: this.wgValueName, size: Math.max(16, 12 * workgroups), clearBufferOnReuse: false }),
        new Kernel({ kernel: this.mergepathWGSL, entryPoint: "mergepath_main", label: `${label} path`, buffers, getDispatchGeometry: dispatch(workgroups) })
      );
      /* one workgroup emits every tile itself; more need the fixup */
      if (workgroups > 1) {
        const fixup = this.fixupPrimitive();
        Object.assign(fixup, { tiles: workgroups, atoms: workgroups });
        const names = [this.wgTileName, this.wgValueName];
        actions.push(
          new ExecutePrimitive({ primitive: fixup, buffers: Object.fromEntries(names.map((n) => [n, n])) }),
          new Kernel({
            kernel: this.mergepathWGSL,
            entryPoint: "heads_main",
            label: `${label} heads`,
            buffers,
            getDispatchGeometry: dispatch(Math.ceil(workgroups / WORKGROUP_SIZE)),
          })
        );
      }
    }
    return actions;
  }

  destroy() {
    this.fixup?.destroy();
    super.destroy();
  }
}

/** The reduction on the CPU: result[t] = binop over tile t's atoms of
 * atomValue(a, t), for checking. offsets: tiles + 1 entries. */
export function segmentedReduceReference({ offsets, tiles, binop, atomValue }) {
  const result = new Array(tiles);
  for (let t = 0; t < tiles; t++) {
    let acc = binop.identity;
    for (let a = offsets[t]; a < offsets[t + 1]; a++) acc = binop.op(acc, atomValue(a, t));
    result[t] = acc;
  }
  return result;
}
