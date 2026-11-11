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
 * schedule (a Schedule, or the name of one: "mergepath", default,
 * "thread" or "workgroup"; see schedules.mjs); itemsPerThread (passed to a
 * schedule given by name, e.g. merge-path's steps per invocation).
 *
 * SegmentedReduce holds the work; the schedule writes the kernels and says
 * what runs. It knows no schedule by more than its name.
 */
import { BasePrimitive, Kernel, AllocateBuffer, ExecutePrimitive, UniformStruct } from "./primitive.mjs";
import { WORKGROUP_SIZE, Schedule, ThreadMapped, WorkgroupMapped, MergePath } from "./schedules.mjs";

export class SegmentedReduce extends BasePrimitive {
  /** schedules by name; add yours here, or pass an instance */
  static schedules = { thread: ThreadMapped, workgroup: WorkgroupMapped, mergepath: MergePath };

  constructor(args) {
    super(args);
    if (!this.binop) throw new Error('SegmentedReduce: requires a binop (e.g. new BinOpAdd({ datatype: "f32" }))');
    if (typeof this.atom !== "function") throw new Error("SegmentedReduce: requires atom: (atom, tile) => WGSL expression");
    this.schedule ??= "mergepath";
    if (typeof this.schedule === "string") {
      const Named = SegmentedReduce.schedules[this.schedule];
      if (!Named) throw new Error(`SegmentedReduce: unknown schedule ${this.schedule}; choose from ${Object.keys(SegmentedReduce.schedules).join(", ")}`);
      this.schedule = new Named(this.itemsPerThread !== undefined ? { itemsPerThread: this.itemsPerThread } : {});
    }
    if (!(this.schedule instanceof Schedule)) throw new Error("SegmentedReduce: schedule must be a Schedule or a schedule's name");
    this.datatype = this.binop.datatype;
    this.userBuffers = (this.buffers ?? []).map((b) => ({ access: "read", ...b }));
    if (!this.emit) {
      this.emit = (t, v) => `outputBuffer[${t}] = ${v};`;
      this.userBuffers.push({ name: "outputBuffer", access: "read_write", type: `array<${this.datatype}>` });
    }
    /* nesting depth: 0 for the caller's, n for a schedule's nth level of
     * child primitives (e.g. merge-path's fixup), which name buffers by it */
    this.level ??= 0;
    this.knownBuffers = [...(this.tileStart ? [] : ["offsets"]), ...this.userBuffers.map((b) => b.name)];
    this.tileStart ??= (t) => `offsets[${t}]`;
    this.workgroupSize = WORKGROUP_SIZE;
    this.children = {};
    this.params = new UniformStruct("SRParams", { tiles: "u32", atoms: "u32", workgroups: "u32" });
    if (!("label" in args)) this.label += ` [ schedule: ${this.schedule.name} ]`;
  }

  /** The work, as WGSL functions the schedules call, then the schedule's
   * kernels: one module for all of a schedule's entry points. */
  moduleWGSL = () => {
    const T = this.datatype;
    return /* wgsl */ `
${this.fnDeclarations.include("computeLinearizedGridParametersSplit")}
${this.binop.wgslfn}
const IDENTITY: ${T} = ${T}(${this.binop.identity});
${this.wgsl ?? ""}
fn tileStartOf(t: u32) -> u32 { return ${this.tileStart("t")}; }
fn atomValue(atom: u32, tile: u32) -> ${T} { return ${this.atom("atom", "tile")}; }
fn emitTile(tile: u32, value: ${T}) { ${this.emit("tile", "value")} }
${this.schedule.wgsl(this)}`;
  };

  /* ---- for schedules' actions() */

  /** a Kernel running one of the schedule's entry points, binding the
   * caller's buffers, the schedule's and the parameters */
  kernel({ entryPoint, workgroups, label }) {
    return new Kernel({
      kernel: this.moduleWGSL,
      entryPoint,
      label: [this.level ? `SegmentedReduce fixup level ${this.level}` : `SegmentedReduce (${this.schedule.name})`, label].filter(Boolean).join(" "),
      buffers: [
        ...(this.knownBuffers.includes("offsets") ? [{ name: "offsets", access: "read", type: "array<u32>" }] : []),
        ...this.userBuffers,
        ...this.schedule.buffers(this),
        { name: "sr", buffer: "srParams", uniform: this.params },
      ],
      getDispatchGeometry: () => this.getSimpleDispatchGeometry({ workgroupCount: workgroups }),
    });
  }
  /** a scratch buffer, not cleared between executes */
  allocate({ label, size }) {
    return new AllocateBuffer({ label, size, clearBufferOnReuse: false });
  }
  /** a SegmentedReduce the schedule runs (made once, kept under key, and
   * destroyed with this one): same device and binop */
  child(key, options) {
    this.children[key] ??= new SegmentedReduce({ device: this.device, adapter: this.adapter, binop: this.binop, ...options });
    return this.children[key];
  }
  /** run a child primitive here, binding these of our buffers by the same names */
  run(primitive, names) {
    return new ExecutePrimitive({ primitive, buffers: Object.fromEntries(names.map((n) => [n, n])) });
  }

  compute() {
    if (!(Number.isInteger(this.tiles) && this.tiles >= 1)) throw new Error(`SegmentedReduce: tiles (${this.tiles}) must be a positive integer`);
    if (!(Number.isInteger(this.atoms) && this.atoms >= 0)) throw new Error(`SegmentedReduce: atoms (${this.atoms}) must be a nonnegative integer`);
    if (this.tiles + this.atoms >= 2 ** 32 - 2 ** 20) throw new Error("SegmentedReduce: tiles + atoms must fit in 32 bits");
    const workgroups = this.schedule.workgroups({ tiles: this.tiles, atoms: this.atoms });
    return [
      this.params.allocate("srParams", { tiles: this.tiles, atoms: this.atoms, workgroups }),
      ...this.schedule.actions(this, { workgroups }),
    ];
  }

  destroy() {
    for (const child of Object.values(this.children)) child.destroy();
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
