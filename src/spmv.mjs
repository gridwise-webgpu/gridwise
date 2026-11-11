/**
 * SpMV: y = A x for a sparse matrix A in CSR form (f32), a SegmentedReduce
 * whose tiles are A's rows and whose atoms are its nonzeros.
 *
 *   const spmv = new SpMV({ device, rows, nnz, schedule: "mergepath" });
 *   await spmv.execute({ offsets, indices, values, x, y });
 *
 * Buffers: offsets (rows + 1 u32: row r's nonzeros are
 * [offsets[r], offsets[r + 1])), indices (nnz u32 column indices), values
 * (nnz f32), x (f32, one per column), y (rows f32, written).
 * Options: rows, nnz (both may change between executes), schedule
 * ("mergepath", default, or "thread"), itemsPerThread; see SegmentedReduce.
 */
import { SegmentedReduce } from "./segmentedreduce.mjs";
import { BinOpAdd } from "./binop.mjs";

export class SpMV extends SegmentedReduce {
  constructor(args) {
    super({
      binop: new BinOpAdd({ datatype: "f32" }),
      buffers: [
        { name: "indices", type: "array<u32>" },
        { name: "values", type: "array<f32>" },
        { name: "x", type: "array<f32>" },
        { name: "y", type: "array<f32>", access: "read_write" },
      ],
      atom: (a) => `values[${a}] * x[indices[${a}]]`,
      emit: (r, v) => `y[${r}] = ${v};`,
      ...args,
    });
  }
  get rows() {
    return this.tiles;
  }
  set rows(r) {
    this.tiles = r;
  }
  get nnz() {
    return this.atoms;
  }
  set nnz(n) {
    this.atoms = n;
  }
}

/** y = A x on the CPU (f64 sums), for checking. */
export function spmvReference({ offsets, indices, values, x }) {
  const rows = offsets.length - 1;
  const y = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let k = offsets[r]; k < offsets[r + 1]; k++) sum += values[k] * x[indices[k]];
    y[r] = sum;
  }
  return y;
}
