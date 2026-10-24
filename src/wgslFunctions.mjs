/**
 * wgslFunctions provides a set of prewritten functions for use in compute
 * kernels. The intent is that the same functions are available on both
 * subgroup-supporting and subgroup-nonsupporting hardware.
 *
 * Usage: The Primitive class automatically incorporates these functions
 * by instantiating an instance of the wgslFunctions or
 * wgslFunctionsWithoutSubgroupSupport class in its constructor:
 *
 * `this.fnDeclarations = new wgslFunctions(this);`
 *
 * If subgroups are not supported, instead:
 *
 * `this.fnDeclarations = new wgslFunctionsWithoutSubgroupSupport(this);`
 *
 * Then, within a kernel, the kernel programmer can include individual
 * functions, such as:
 *
 * `${this.fnDeclarations.vec4InclusiveScan}`
 *
 * which declares the function `vec4InclusiveScan`.
 *
 * More detailed usage:
 *
 * 1. At the top of the kernel, place the following:
 *    `${this.fnDeclarations.enableSubgroupsIfAppropriate}`
 *    WGSL requires declaring subgroups must be done at the top of the kernel.
 *    If subgroups are not enabled, this declaration will do nothing.
 * 2. Next, if the kernel is using any subgroup calls, place the following:
 *    `${this.fnDeclarations.subgroupEmulation}`
 *    If we're making subgroup calls and we don't have subgroup hardware,
 *    this sets up necessary declarations (workgroup memory and subgroup
 *    variables). If subgroups are supported, this declaration does nothing.
 * 3. If the primitive is parameterized by a datatype and a binary operation,
 *    it may be useful to define a monoid that combines the two. If "monoid"
 *    means nothing to you, instead consider: it may be useful to have a WGSL
 *    operation called "binop" that is a binary operation that combines two
 *    inputs of a specific datatype. (For instance, + on f32 values.)
 *    Declare that binop with:
 *    `${this.binop.wgslfn}`
 *    (This definition is in binop.mjs, not here.)
 * 4. In this file are a large number of functions to call, each of which
 *    outputs a WGSL function definition. These functions output WGSL code
 *    that may be parameterized by objects in the Primitive's `this`.
 *    Override any parameter by passing it in the input object to the
 *    function call.
 *    - Example: vec4InclusiveScan is parameterized by `datatype`. If
 *      Primitive's this.datatype is "f32", and we call
 *      `${this.fnDeclarations.vec4InclusiveScan()}` in our kernel,
 *      the signature of the declared function will be
 *      `fn vec4InclusiveScan(in: vec4<f32>) -> vec4<f32>`.
 *      But if we want to override it to u32 when we instantiate the
 *      function, we can instead call
 *      `${this.fnDeclarations.vec4InclusiveScan({datatype: "u32"})}`.
 *    The names of the parameters that are necessary for various function
 *    declarations are:
 *    - binop (the monoid containing a binary operator and datatype, see
 *      binop.mjs)
 *    - datatype (a string, one of {"f32", "u32", "i32"})
 *    - workgroupSize (a number)
 *
 * Notes for authors of functions:
 * - If a function doesn't require subgroups, put it in wgslFunctions
 * - If it does, write the subgroup version in wgslFunctions
 *   and the non-subgroup version in wgslFunctionsWithoutSubgroupSupport;
 *   the latter will override the former (this is done in the Primitive class)
 * Your functions can be parameterized by any element in the Primitive object
 *   and by any element in the argument to the function. The recommended usage
 *   places both of these in an `env` object, where any element of args
 *   overrides any element in the Primitive object, and then the WGSL function
 *   definition can use elements in this `env` object. (For instance,
 *   `env.datatype`.)
 *   - Implementation note. Originally these functions were getters and did
 *     not have to be called as functions. However, during the development
 *     of more complex functions, it became evident that some functions would
 *     definitely require arguments, so for consistency, all of them are
 *     defined as functions not getters.
 *
 * Convention for order of arguments to WGSL functions:
 * 1. builtins
 * 2. outputs, ordered from most global (storage) to most local (workgroup),
 *    and from most permanent (arguments to the primitive) to least (temporaries
 *    declared within the primitive)
 * 3. inputs, ordered from most global (storage) to most local (workgroup)
 *    and from most permanent (arguments to the primitive) to least (temporaries
 *    declared within the primitive)
 *
 * General philosophy is to design kernels around an expectation that
 * subgroups are present and optimize kernel organization strategies for
 * that case; a non-subgroup-enabled function that provides fallback must
 * be correct but may not be optimized.
 */

import { BinOpAdd } from "./binop.mjs";

/** Names shared by both versions of wgScan: one function and one workgroup
 * temp per (scan type, binop, datatype, workgroup size), so that exclusive
 * and inclusive scans can both be declared in one module. */
function wgScanNames(env) {
  const scanType = env.scanType ?? "exclusive";
  if (scanType !== "exclusive" && scanType !== "inclusive") {
    throw new Error(`wgScan: scanType must be "exclusive" or "inclusive", not "${scanType}"`);
  }
  for (const necessary of ["binop", "datatype", "workgroupSize"]) {
    if (!(necessary in env)) {
      console.warn(`wgScan: field '${necessary}' must be set in env`);
    }
  }
  const Type = scanType === "inclusive" ? "Inclusive" : "Exclusive";
  const config = [env.binop, env.datatype, env.workgroupSize].join("_");
  const shortName = `wg${Type}Scan`;
  return {
    fnName: env.useLongFunctionName ? `${shortName}_${config}` : shortName,
    wgTemp: `wg_${scanType}_scan_temp_${config}`,
    T: env.datatype,
    identity: env.binop.identity,
    inclusive: scanType === "inclusive",
  };
}

export class wgslFunctions {
  constructor(args) {
    this.env = args;
  }
  commonDefinitions(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `
  struct Builtins {
    @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
    @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
    @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    @builtin(local_invocation_id) lid: vec3u /* 3D thread index within workgroup */,
    @builtin(subgroup_size) sgsz: u32, /* 32 on Apple GPUs */
    @builtin(subgroup_invocation_id) sgid: u32 /* 1D thread index within subgroup */
  }
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
  }`;
  }
  initializeSubgroupVars(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return "let sgsz: u32 = builtinsUniform.sgsz;\nlet sgid: u32 = builtinsNonuniform.sgid;";
  }
  declareSubgroupVarsInsideHelper(uniformName = "builtinsUniform", nonuniformName = "builtinsNonuniform") {
    return `let sgsz = ${uniformName}.sgsz;\n  let sgid = ${nonuniformName}.sgid;`;
  }
  enableSubgroupsIfAppropriate(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return "enable subgroups;";
  }
  subgroupEmulation(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return "";
  }
  roundUpDivU32(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `fn roundUpDivU32(a : u32, b : u32) -> u32 {
    return (a + b - 1) / b;
  }`;
  }
  computeLinearizedGridParameters(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `
    /* wgid is a linearized (1d) unique ID per wg;
     * gid is a linearized (1d) unique ID per thread */
    var wgid = builtins.wgid.z * builtins.nwg.y * builtins.nwg.x +
               builtins.wgid.y * builtins.nwg.x +
               builtins.wgid.x;
    var numThreadsPerWorkgroup: u32 = ${
      env.numThreadsPerWorkgroup ?? env.workgroupSize
    };
    var gid: u32 = wgid * numThreadsPerWorkgroup + builtins.lidx;
    var workgroupCount = builtins.nwg.z * builtins.nwg.y * builtins.nwg.x;
    var totalThreadCount = workgroupCount * numThreadsPerWorkgroup;`;
  }
  computeLinearizedGridParametersSplit(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /* "split" meaning "split builtin uniforms vs. nonuniforms" */
    return /* wgsl */ `
    /* wgid is a linearized (1d) unique ID per wg;
     * gid is a linearized (1d) unique ID per thread */
    var wgid = builtinsUniform.wgid.z * builtinsUniform.nwg.y * builtinsUniform.nwg.x +
               builtinsUniform.wgid.y * builtinsUniform.nwg.x +
               builtinsUniform.wgid.x;
    var numThreadsPerWorkgroup: u32 = ${
      env.numThreadsPerWorkgroup ?? env.workgroupSize
    };
    var gid: u32 = wgid * numThreadsPerWorkgroup + builtinsNonuniform.lidx;
    var workgroupCount = builtinsUniform.nwg.z * builtinsUniform.nwg.y * builtinsUniform.nwg.x;
    var totalThreadCount = workgroupCount * numThreadsPerWorkgroup;`;
  }
  vec4InclusiveScan(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `
    fn vec4InclusiveScan(in: vec4<${env.datatype}>) ->
      vec4<${env.datatype}> {
      /* vec4Scan(in) = [in.x, in.x+in.y, in.x+in.y+in.z, in.x+in.y+in.z+in.w] */
      var out: vec4<${env.datatype}> = in;
      out.y = binop(in.x,  in.y);
      out.z = binop(out.y, in.z);
      out.w = binop(out.z, in.w);
      return out;
    }`;
  }
  vec4ExclusiveScan(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `
    fn vec4ExclusiveScan(in: vec4<${env.datatype}>) ->
      vec4<${env.datatype}> {
      /* vec4Scan(in) = [in.x, in.x+in.y, in.x+in.y+in.z, in.x+in.y+in.z+in.w] */
      var out: vec4<${env.datatype}>;
      out.x = ${env.binop.identity};
      out.y = in.x;
      out.z = binop(in.x,  in.y);
      out.w = binop(out.z, in.z);
      return out;
    }`;
  }
  vec4InclusiveToExclusive(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `
    fn vec4InclusiveToExclusive(in: vec4<${env.datatype}>) ->
      vec4<${env.datatype}> {
      var out: vec4<${env.datatype}>;
      out.w = in.z;
      out.z = in.y;
      out.y = in.x;
      out.x = ${env.binop.identity};
      return out;
    }`;
  }
  vec4Reduce(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    // TODO: Don't special-case this. Worried about polyfilling dot with
    // int arguments, that it'll potentially do four multiplies
    if (env.binop instanceof BinOpAdd) {
      return /* wgsl */ `
      fn vec4Reduce(in: vec4<${env.datatype}>) -> ${env.datatype} {
        return dot(in, vec4<${env.datatype}>(1, 1, 1, 1));
      }
      `;
    } else {
      return /* wgsl */ `
      fn vec4Reduce(in: vec4<${env.datatype}>) -> ${env.datatype} {
        return binop(binop(binop(in.x, in.y), in.z), in.w);
      }`;
    }
  }
  vec4ScalarBinopV4(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /* if binop is +, this still seems just as efficient, unless there's a vec4 +, I guess? */
    // TODO: "WGSL has mixed vector-scalar arithmetic operators, so it's probably best to use those if you can."
    return /* wgsl */ `
    fn vec4ScalarBinopV4(scalar: ${env.datatype}, vector: vec4<${env.datatype}>) ->
    vec4<${env.datatype}> {
      var out: vec4<${env.datatype}>;
      out.x = binop(scalar, vector.x);
      out.y = binop(scalar, vector.y);
      out.z = binop(scalar, vector.z);
      out.w = binop(scalar, vector.w);
      return out;
    }
    `;
  }
  subgroupZero(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `
    fn isSubgroupZero(lidx: u32, sgsz: u32) -> bool {
      return lidx < sgsz;
    }`;
  }
  subgroupShuffle(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /* keep builtin */
    return "";
  }
  subgroupShuffleXor(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /* keep builtin */
    return "";
  }
  subgroupBallot(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /* keep builtin */
    return "";
  }
  subgroupMax(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /* keep builtin */
    return "";
  }
  subgroupAdd(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /* keep builtin */
    return "";
  }
  subgroupAny(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args);
    return "";
  }
  subgroupAll(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args);
    return "";
  }

  subgroupInclusiveOpScan(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /* helpful reference from Thomas Smith:
     *   https://github.com/b0nes164/GPUSorting/blob/main/GPUSortingCUDA/Utils.cuh
     */
    /** Would prefer to not make sgsz/sgid an argument here, but we need it for
     *  the subgroup-hardware-capable, non-hardware-supported-scan-op case
     *  An alternative would be putting sgsz/sgid in workgroup memory. */
    if (env.binop.subgroupInclusiveScanOp) {
      /* use the builtin subgroupInclusiveScanOp */
      return /* wgsl */ `
      fn subgroupInclusiveOpScan(in: ${env.datatype}, sgid: u32, sgsz: u32) ->
        ${env.datatype} {
        return ${env.binop.subgroupInclusiveScanOp}(in);
      }
      `;
    } else {
      /* emulate subgroupInclusiveScanOp with subgroupShuffleUp */
      /* for (int i = 1; i <= 16; i <<= 1) { // 16 = LANE_COUNT >> 1
       *   const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
       *   if (getLaneId() >= i) val += t;
       * }
       * return val;
       */
      return /* wgsl */ `
      fn subgroupInclusiveOpScan(in: ${env.datatype}, sgid: u32, sgsz: u32) ->
        ${env.datatype} {
        var i: u32;
        var val = in;
        for (i = 1; i <= (sgsz >> 1); i <<= 1) {
          let t: ${env.datatype} = subgroupShuffleUp(val, i);
          val = binop(select(${env.binop.identity}, t, sgid >= i), val);
        }
        return val;
      }
    `;
    }
  }
  subgroupReduce(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /* this will fail if subgroupReduceOp isn't defined; TODO is write it */
    return /* wgsl */ `
    fn subgroupReduce(in: ${env.datatype}) -> ${env.datatype} {
      return ${env.binop.subgroupReduceOp}(in);
    }
    `;
  }
  wgReduce(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env

    /** The normal case would be that we only need one function of each type
     * thus we can use the shortFnName for declaration and call, and can do
     * everything within a template string.
     * But if we need more flexibility (here, more than one reduce call in
     * a module), we should do it outside the template string. Until that is
     * necessary, this capability has not been used and is untested.
     *
     * Primitive-specific args are:
     * - wgTempIsArgument: if true, pass in a temp array for temporary use
     * - useLongFunctionName: use config-specific name, otherwise wgReduce
     * Default for all of these is "false".
     */
    const shortFnName = "wgReduce";
    /* every entry in params below needs to be a member of env */
    const params = ["binop", "datatype", "workgroupSize", "SUBGROUP_MIN_SIZE"];
    for (const necessary of params) {
      if (!(necessary in env)) {
        console.warn(`wgReduce: field '${necessary}' must be set in env`);
      }
    }
    const config = params.map((param) => env[param]).join("_");
    const wgTemp = env.wgTempIsArgument ? "wgTemp" : `wg_temp_${config}`;
    const declareAndUseLocalWgTemp = !env.wgTempIsArgument;
    const longFnName = `${shortFnName}_${config}`;
    const fnName = env?.useLongFunctionName ? longFnName : shortFnName;
    const fn = /* wgsl */ `
${
  declareAndUseLocalWgTemp
    ? `const TEMP_${longFnName}_MEM_SIZE = 2 * ${env.workgroupSize} / ${env.SUBGROUP_MIN_SIZE};
var<workgroup> ${wgTemp}: array<${env.datatype}, TEMP_${longFnName}_MEM_SIZE>;`
    : ""
}

fn ${fnName}(// in: ptr<storage, array<${env.datatype}>, read>,
             in: ${env.datatype},
             ${
               declareAndUseLocalWgTemp
                 ? ""
                 : `wgTemp: ptr<workgroup, array<${env.datatype}, MAX_PARTIALS_SIZE>>,`
             }
             builtinsUniform: BuiltinsUniform,
             builtinsNonuniform: BuiltinsNonuniform) -> ${env.datatype} {
  let lidx = builtinsNonuniform.lidx;
  ${env.fnDeclarations.declareSubgroupVarsInsideHelper("builtinsUniform", "builtinsNonuniform")}
  let BLOCK_DIM: u32 = ${env.workgroupSize};
  let sid = lidx / sgsz;
  let lane_log = u32(countTrailingZeros(sgsz)); /* log_2(sgsz) */
  /* workgroup size / subgroup size; how many partial reductions in this tile? */
  let local_spine: u32 = BLOCK_DIM >> lane_log;
  let aligned_size_base = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
  /* fix for aligned_size_base == 1 (needed when subgroup_size == BLOCK_DIM) */
  let aligned_size = select(aligned_size_base, BLOCK_DIM, aligned_size_base == 1);

  let t_red = in;
  let s_red = subgroupReduce(t_red);
  if (sgid == 0u) {
    ${wgTemp}[sid] = s_red;
  }
  workgroupBarrier();
  var f_red: ${env.datatype} = ${env.binop.identity};

  var offset = 0u;
  var top_offset = 0u;
  let lane_pred = sgid == 0u;
  if (sgsz > aligned_size) {
    /* don't enter the loop */
    f_red = ${wgTemp}[lidx + top_offset];
  } else {
    for (var j = sgsz; j <= aligned_size; j <<= lane_log) {
      let step = local_spine >> offset;
      let pred = lidx < step;
      f_red = subgroupReduce(
        select(${env.binop.identity},
        ${wgTemp}[lidx + top_offset],
        pred));
      if (pred && lane_pred) {
        ${wgTemp}[sid + step + top_offset] = f_red;
      }
      workgroupBarrier();
      top_offset += step;
      offset += lane_log;
    }
  }
  workgroupBarrier();
  return ${wgTemp}[top_offset];
}`;
    return fn;
  }
  /** Ballot-based digit matching across a subgroup: Algorithm 3 of Ashkiani
   * et al., "GPU Multisplit" (TOPC 2017), and the core of OneSweep's WLMS.
   * With one ballot per bit of the digit, each invocation finds the lanes of
   * its subgroup whose digit equals its own, how many of them are below it,
   * and the highest of them.
   *
   * Lane masks are u32, or vec4<u32> (lane i is bit i % 32 of word i / 32)
   * when the device's subgroups can be wider than 32 (env.wideSubgroups);
   * that choice is made when the WGSL is generated, so 32-wide devices get
   * the cheaper 32-bit code.
   *
   *   ${this.fnDeclarations.subgroupLaneMasks()}     at module scope: fn laneMaskBelow (wide only)
   *   let lane_mask_lt = ${this.fnDeclarations.subgroupLaneMaskLt()};   lanes below me
   *   ${this.fnDeclarations.subgroupDigitMatch({ key: "key", shift: "shift", bits: "RADIX_LOG" })}
   *
   * subgroupDigitMatch emits statements (not a function), so a caller's
   * generated code is the same as if written inline. They declare, in the
   * caller's scope: eq_mask (the matching lanes, me included), var <rank>
   * (default "rank": matching lanes below me) and let <peer> (default
   * "peer": the highest matching lane). They read sgsz, and lane_mask_lt
   * (or the name given as laneMaskLt). The digit is bits [shift, shift +
   * bits) of key; every invocation of the subgroup must take part. */
  subgroupLaneMasks(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    if (!env.wideSubgroups) return "";
    return /* wgsl */ `
    fn laneMaskBelow(n: u32) -> vec4<u32> {
      /* bits set for lanes 0..n-1 */
      var m = vec4<u32>(0u);
      for (var i = 0u; i < 4u; i += 1u) {
        let lo = i * 32u;
        m[i] = select(select((1u << (n - lo)) - 1u, 0xffffffffu, n >= lo + 32u), 0u, n <= lo);
      }
      return m;
    }`;
  }
  subgroupLaneMaskLt(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return env.wideSubgroups ? "laneMaskBelow(sgid)" : "(1u << sgid) - 1u";
  }
  subgroupDigitMatch(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    const { key, shift, bits } = args;
    const rank = args.rank ?? "rank";
    const peer = args.peer ?? "peer";
    const lt = args.laneMaskLt ?? "lane_mask_lt";
    const loop = (word) => /* wgsl */ `
      /* loop through the bits in this digit place */
      for (var k = 0u; k < ${bits}; k += 1u) {
        let curr_bit = 1u << (k + ${shift});
        let pred = (${key} & curr_bit) != 0u;
        let ballot = subgroupBallot(pred);
        eq_mask &= select(~${word}, ${word}, pred);
      }
      /* eq_mask marks the lanes that have the same digit as me (me included) */`;
    if (!env.wideSubgroups) {
      return /* wgsl */ `
      var eq_mask = 0xffffffffu;${loop("ballot.x")}
      var ${rank} = countOneBits(eq_mask & ${lt});
      /* the highest lane with my digit */
      let ${peer} = sgsz - countLeadingZeros(eq_mask) - 1u;`;
    }
    return /* wgsl */ `
      /* start from the lanes that exist, so ~ballot cannot mark lanes past sgsz */
      var eq_mask = laneMaskBelow(sgsz);${loop("ballot")}
      let below = eq_mask & ${lt};
      var ${rank} = countOneBits(below.x) + countOneBits(below.y) + countOneBits(below.z) + countOneBits(below.w);
      /* the highest lane with my digit (eq_mask includes me, so it is never zero) */
      let top_word = select(select(select(0u, 1u, eq_mask.y != 0u), 2u, eq_mask.z != 0u), 3u, eq_mask.w != 0u);
      let ${peer} = top_word * 32u + firstLeadingBit(eq_mask[top_word]);`;
  }

  /** A histogram in workgroup memory: the shared core of Histogram,
   * MultisplitPerm's histogram pass and similar kernels. It needs no
   * subgroups, so both function classes use this version.
   *
   *   ${this.fnDeclarations.wgHistogram({ name: "bins", numBins: 256, copies: 2 })}
   *
   * declares
   *
   *   var<workgroup> bins: array<atomic<u32>, numBins * copies>;
   *   fn binsClear(lidx: u32)          zero the bins; then workgroupBarrier()
   *   fn binsAdd(b: u32, lidx: u32)    count one sample in bin b (b >= numBins,
   *                                    e.g. BIN_NONE, is not counted)
   *   fn binsCount(b: u32) -> u32      after workgroupBarrier(): the
   *                                    workgroup's count for bin b
   *
   * copies > 1 gives each block of workgroupSize / copies invocations its
   * own copy of the bins, which cuts atomic contention when there are few
   * bins; binsCount sums the copies. Callers flush the counts themselves
   * (atomicAdd into a global histogram, or a store at an index of their
   * choosing), typically for (var b = lidx; b < numBins; b += workgroupSize). */
  wgHistogram(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    const name = env.name ?? "wgBins";
    const N = env.numBins;
    const C = env.copies ?? 1;
    const W = env.workgroupSize;
    if (!(N >= 1) || !(C >= 1) || W % C !== 0) {
      throw new Error(
        `wgHistogram: need numBins >= 1 and copies dividing the workgroup size (numBins ${N}, copies ${C}, workgroup ${W})`
      );
    }
    const slot = C > 1 ? `(lidx / ${W / C}u) * ${N}u + b` : "b";
    const count =
      C > 1
        ? Array.from({ length: C }, (_, c) => `atomicLoad(&${name}[${c * N}u + b])`).join(" + ")
        : `atomicLoad(&${name}[b])`;
    return /* wgsl */ `
var<workgroup> ${name}: array<atomic<u32>, ${N * C}>;
fn ${name}Clear(lidx: u32) {
  for (var i = lidx; i < ${N * C}u; i += ${W}u) { atomicStore(&${name}[i], 0u); }
}
fn ${name}Add(b: u32, lidx: u32) {
  if (b < ${N}u) { atomicAdd(&${name}[${slot}], 1u); }
}
fn ${name}Count(b: u32) -> u32 { return ${count}; }`;
  }

  wgScan(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /** Scan of one value per invocation across the workgroup, the scan
     * counterpart of wgReduce.
     *
     *   ${this.fnDeclarations.wgScan()}                           // declares wgExclusiveScan
     *   ${this.fnDeclarations.wgScan({ scanType: "inclusive" })}  // declares wgInclusiveScan
     *
     *   var total: T;
     *   let prefix = wgExclusiveScan(value, &total, builtinsUniform, builtinsNonuniform);
     *
     * Exclusive: binop over the values of all invocations with a smaller
     * local_invocation_index (identity for invocation 0). Inclusive: the
     * same plus the invocation's own value. Both write the reduction of the
     * whole workgroup to *total in every invocation. Works with any binop
     * (add, min, max, multiply, custom). Call from workgroup-uniform control
     * flow with every invocation; it contains workgroup barriers, including
     * one at the end, so calls can follow one another directly.
     *
     * Args: scanType ("exclusive", default, or "inclusive"),
     * useLongFunctionName (as in wgReduce). Requires binop (its wgslfn
     * declared as "binop"), datatype and workgroupSize; with subgroups, also
     * subgroupInclusiveOpScan declared in the module. Declares its own
     * module-scope workgroup memory, named by configuration.
     */
    const { fnName, wgTemp, T, identity, inclusive } = wgScanNames(env);
    /* Scan within each subgroup (subgroupInclusiveOpScan: the binop's own
     * subgroup scan if it has one, else shuffles), scan the subgroup totals
     * sgsz at a time, then combine. The temp array has one entry per
     * subgroup, sized for the smallest subgroup size WebGPU allows (4), not
     * the adapter's reported minimum: the size a pipeline actually gets is
     * not always the reported one. */
    return /* wgsl */ `
var<workgroup> ${wgTemp}: array<${T}, ${Math.ceil(env.workgroupSize / 4)}>;

fn ${fnName}(in: ${T},
             total: ptr<function, ${T}>,
             builtinsUniform: BuiltinsUniform,
             builtinsNonuniform: BuiltinsNonuniform) -> ${T} {
  let lidx = builtinsNonuniform.lidx;
  ${env.fnDeclarations.declareSubgroupVarsInsideHelper("builtinsUniform", "builtinsNonuniform")}
  let sid = lidx / sgsz;
  let numSubgroups = (${env.workgroupSize}u + sgsz - 1u) / sgsz;
  /* (1) inclusive scan within each subgroup; its last lane posts the total */
  let sgInclusive = subgroupInclusiveOpScan(in, sgid, sgsz);
  if (sgid == sgsz - 1u || lidx == ${env.workgroupSize - 1}u) {
    ${wgTemp}[sid] = sgInclusive;
  }
  workgroupBarrier();
  /* (2) exclusive scan of the subgroup totals, sgsz at a time. Every
   * subgroup computes it (subgroup operations need uniform control flow);
   * subgroup 0 writes the results back in place. */
  var carry: ${T} = ${identity};
  for (var base = 0u; base < numSubgroups; base += sgsz) {
    let i = base + sgid;
    let v = select(${identity}, ${wgTemp}[min(i, numSubgroups - 1u)], i < numSubgroups);
    let chunkInclusive = subgroupInclusiveOpScan(v, sgid, sgsz);
    let chunkExclusive = select(${identity}, subgroupShuffleUp(chunkInclusive, 1u), sgid > 0u);
    let chunkTotal = subgroupShuffle(chunkInclusive, sgsz - 1u);
    workgroupBarrier(); /* everyone has read this chunk before it is overwritten */
    if (sid == 0u && i < numSubgroups) {
      ${wgTemp}[i] = binop(carry, chunkExclusive);
    }
    carry = binop(carry, chunkTotal);
    workgroupBarrier();
  }
  /* (3) combine: the earlier subgroups' total, then the prefix in this subgroup */
  let inSubgroup = ${inclusive ? "sgInclusive" : `select(${identity}, subgroupShuffleUp(sgInclusive, 1u), sgid > 0u)`};
  let out = binop(${wgTemp}[sid], inSubgroup);
  *total = carry;
  workgroupBarrier(); /* ${wgTemp} may be reused by the next call */
  return out;
}`;
  }
}

/** Philosophy of subgroup fallbacks:
 * - Emulate subgroup operations via workgroup memory
 * - This requires workgroup barriers since we can make no assumptions
 *   about SIMD width / lockstep execution
 * - Given that we HAVE to have workgroup barriers, might as well support
 *   large workgroups
 * - So assume that the emulated subgroup size == workgroup size (1 subgroup
 *   per workgroup)
 * - Implementations are generally O(n log n) (Hillis-Steele / Kogge-Stone)
 */

export class wgslFunctionsWithoutSubgroupSupport extends wgslFunctions {
  constructor(env) {
    super(env);
  }
  commonDefinitions(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `
    struct Builtins {
      @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
      @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
      @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
      @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    }
    struct BuiltinsNonuniform {
      @builtin(global_invocation_id) gid: vec3u /* 3D thread id in compute shader grid */,
      @builtin(local_invocation_index) lidx: u32 /* 1D thread index within workgroup */,
    }
    struct BuiltinsUniform {
      @builtin(num_workgroups) nwg: vec3u /* == dispatch */,
      @builtin(workgroup_id) wgid: vec3u /* 3D workgroup id within compute shader grid */,
    }`;
  }
  initializeSubgroupVars(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `
    sgid = builtinsNonuniform.lidx % 32u;
    lidx_private = builtinsNonuniform.lidx;
    shuffle_count = 0u;`;
  }
  declareSubgroupVarsInsideHelper(uniformName, nonuniformName) {
    return "";
  }

  enableSubgroupsIfAppropriate(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return "/* don't enable subgroups */\n";
  }
  /* if this declaration works for you, put it at the top of your kernel file at module scope */
  subgroupEmulation(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `var<workgroup> wg_sw_subgroups: array<${env.datatype}, ${env.workgroupSize}>;
    var<workgroup> wg_sw_subgroups_u32_steps: array<array<atomic<u32>, ${env.workgroupSize}>, 6>;
    var<workgroup> wg_sw_subgroups_flag_steps: array<array<atomic<u32>, ${env.workgroupSize}>, 6>;
    var<workgroup> wg_thread_shuffle_count: array<atomic<u32>, ${env.workgroupSize}>;
    const sgsz: u32 = 32u;
    const SPIN_TIMEOUT: u32 = 5000000u;
    var<private> sgid: u32;
    var<private> lidx_private: u32;
    var<private> shuffle_count: u32;
 
    fn subgroupInclusiveAdd(in: u32) -> u32 {
      let prev_count = shuffle_count;
      shuffle_count += 1u;
      let base = lidx_private & ~31u;
      
      if (prev_count > 0u) {
        for (var i: u32 = 0; i < 32u; i += 1u) {
          let target_idx = base + i;
          var spun_fc = 0u;
          while (atomicLoad(&wg_thread_shuffle_count[target_idx]) < prev_count && spun_fc < SPIN_TIMEOUT) {
            spun_fc += 1u;
          }
        }
      }
      
      var red: u32 = in;
      var t: u32;
      
      var step_idx = 0u;
      for (var delta: u32 = 1; delta < 32u; delta <<= 1) {
        atomicStore(&wg_sw_subgroups_u32_steps[step_idx][lidx_private], red);
        atomicStore(&wg_sw_subgroups_flag_steps[step_idx][lidx_private], shuffle_count);
        
        var neighborIdx: i32 = i32(sgid) - i32(delta);
        if (neighborIdx >= 0) {
          let target_idx = base + u32(neighborIdx);
          var spun = 0u;
          while (atomicLoad(&wg_sw_subgroups_flag_steps[step_idx][target_idx]) < shuffle_count && spun < SPIN_TIMEOUT) {
            spun += 1u;
          }
          t = atomicLoad(&wg_sw_subgroups_u32_steps[step_idx][target_idx]);
          red = red + t;
        }
        step_idx += 1u;
      }
      atomicStore(&wg_thread_shuffle_count[lidx_private], shuffle_count);
      return red;
    }
 
    fn subgroupExclusiveAdd(in: u32) -> u32 {
      let inclusive = subgroupInclusiveAdd(in);
      let step_idx = 5u;
      atomicStore(&wg_sw_subgroups_u32_steps[step_idx][lidx_private], inclusive);
      atomicStore(&wg_sw_subgroups_flag_steps[step_idx][lidx_private], shuffle_count);
      
      var exclusive: u32 = 0;
      if (sgid > 0) {
        let target_idx = lidx_private - 1u;
        var spun = 0u;
        while (atomicLoad(&wg_sw_subgroups_flag_steps[step_idx][target_idx]) < shuffle_count && spun < SPIN_TIMEOUT) {
          spun += 1u;
        }
        exclusive = atomicLoad(&wg_sw_subgroups_u32_steps[step_idx][target_idx]);
      }
      atomicStore(&wg_thread_shuffle_count[lidx_private], shuffle_count);
      return exclusive;
    }
 
    fn subgroupAdd(in: u32) -> u32 {
      let inclusive = subgroupInclusiveAdd(in);
      let base = lidx_private & ~31u;
      let step_idx = 5u;
      atomicStore(&wg_sw_subgroups_u32_steps[step_idx][lidx_private], inclusive);
      atomicStore(&wg_sw_subgroups_flag_steps[step_idx][lidx_private], shuffle_count);
      
      let target_idx = base + 31u;
      var spun = 0u;
      while (atomicLoad(&wg_sw_subgroups_flag_steps[step_idx][target_idx]) < shuffle_count && spun < SPIN_TIMEOUT) {
        spun += 1u;
      }
      let final_val = atomicLoad(&wg_sw_subgroups_u32_steps[step_idx][target_idx]);
      atomicStore(&wg_thread_shuffle_count[lidx_private], shuffle_count);
      return final_val;
    }`;
  }
  subgroupZero(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `
    fn isSubgroupZero(lidx: u32, sgsz: u32) -> bool {
      return lidx < 32u;
    }`;
  }
  /* No subgroups: a Hillis-Steele scan in workgroup memory with barriers
   * only. Emulated subgroups would spin-wait on other invocations, which
   * needs forward-progress guarantees WebGPU does not make; barriers do not. */
  wgScan(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /** Scan (no-subgroup version) of one value per invocation across the workgroup, the scan
     * counterpart of wgReduce.
     *
     *   ${this.fnDeclarations.wgScan()}                           // declares wgExclusiveScan
     *   ${this.fnDeclarations.wgScan({ scanType: "inclusive" })}  // declares wgInclusiveScan
     *
     *   var total: T;
     *   let prefix = wgExclusiveScan(value, &total, builtinsUniform, builtinsNonuniform);
     *
     * Exclusive: binop over the values of all invocations with a smaller
     * local_invocation_index (identity for invocation 0). Inclusive: the
     * same plus the invocation's own value. Both write the reduction of the
     * whole workgroup to *total in every invocation. Works with any binop
     * (add, min, max, multiply, custom). Call from workgroup-uniform control
     * flow with every invocation; it contains workgroup barriers, including
     * one at the end, so calls can follow one another directly.
     *
     * Args: scanType ("exclusive", default, or "inclusive"),
     * useLongFunctionName (as in wgReduce). Requires binop (its wgslfn
     * declared as "binop"), datatype and workgroupSize; with subgroups, also
     * subgroupInclusiveOpScan declared in the module. Declares its own
     * module-scope workgroup memory, named by configuration.
     */
    const { fnName, wgTemp, T, identity, inclusive } = wgScanNames(env);
    const W = env.workgroupSize;
    return /* wgsl */ `
var<workgroup> ${wgTemp}: array<${T}, ${W}>;

fn ${fnName}(in: ${T},
             total: ptr<function, ${T}>,
             builtinsUniform: BuiltinsUniform,
             builtinsNonuniform: BuiltinsNonuniform) -> ${T} {
  let lidx = builtinsNonuniform.lidx;
  ${wgTemp}[lidx] = in;
  workgroupBarrier();
  for (var offset = 1u; offset < ${W}u; offset <<= 1u) {
    var earlier: ${T} = ${identity};
    if (lidx >= offset) {
      earlier = ${wgTemp}[lidx - offset];
    }
    workgroupBarrier();
    ${wgTemp}[lidx] = binop(earlier, ${wgTemp}[lidx]);
    workgroupBarrier();
  }
  /* ${wgTemp} now holds the inclusive scan */
  let out = ${inclusive ? `${wgTemp}[lidx]` : `select(${identity}, ${wgTemp}[max(lidx, 1u) - 1u], lidx > 0u)`};
  *total = ${wgTemp}[${W - 1}u];
  workgroupBarrier(); /* ${wgTemp} may be reused by the next call */
  return out;
}`;
  }

  subgroupAny(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `fn subgroupAny(pred: bool) -> bool {
  let prev_count = shuffle_count;
  shuffle_count += 1u;
  let base = lidx_private & ~31u;
  
  if (prev_count > 0u) {
    for (var i: u32 = 0; i < 32u; i += 1u) {
      let target_idx = base + i;
      var spun_fc = 0u;
      while (atomicLoad(&wg_thread_shuffle_count[target_idx]) < prev_count && spun_fc < SPIN_TIMEOUT) {
        spun_fc += 1u;
      }
    }
  }
  
  let step_idx = 0u;
  atomicStore(&wg_sw_subgroups_u32_steps[step_idx][lidx_private], select(0u, 1u, pred));
  atomicStore(&wg_sw_subgroups_flag_steps[step_idx][lidx_private], shuffle_count);
  var anyVal: u32 = 0u;
  for (var i: u32 = 0; i < 32u; i += 1) {
    let target_idx = base + i;
    var spun = 0u;
    while (atomicLoad(&wg_sw_subgroups_flag_steps[step_idx][target_idx]) < shuffle_count && spun < SPIN_TIMEOUT) {
      spun += 1u;
    }
    anyVal |= atomicLoad(&wg_sw_subgroups_u32_steps[step_idx][target_idx]);
  }
  atomicStore(&wg_thread_shuffle_count[lidx_private], shuffle_count);
  return anyVal != 0u;
}`;
  }
  subgroupAll(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `fn subgroupAll(pred: bool) -> bool {
  let prev_count = shuffle_count;
  shuffle_count += 1u;
  let base = lidx_private & ~31u;
  
  if (prev_count > 0u) {
    for (var i: u32 = 0; i < 32u; i += 1u) {
      let target_idx = base + i;
      var spun_fc = 0u;
      while (atomicLoad(&wg_thread_shuffle_count[target_idx]) < prev_count && spun_fc < SPIN_TIMEOUT) {
        spun_fc += 1u;
      }
    }
  }
  
  let step_idx = 0u;
  atomicStore(&wg_sw_subgroups_u32_steps[step_idx][lidx_private], select(0u, 1u, pred));
  atomicStore(&wg_sw_subgroups_flag_steps[step_idx][lidx_private], shuffle_count);
  var allVal: u32 = 1u;
  for (var i: u32 = 0; i < 32u; i += 1) {
    let target_idx = base + i;
    var spun = 0u;
    while (atomicLoad(&wg_sw_subgroups_flag_steps[step_idx][target_idx]) < shuffle_count && spun < SPIN_TIMEOUT) {
      spun += 1u;
    }
    allVal &= atomicLoad(&wg_sw_subgroups_u32_steps[step_idx][target_idx]);
  }
  atomicStore(&wg_thread_shuffle_count[lidx_private], shuffle_count);
  return allVal != 0u;
}`;
  }
  subgroupShuffle(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `fn subgroupShuffle(x: u32, source: u32) -> u32 {
  let prev_count = shuffle_count;
  shuffle_count += 1u;
  let base = lidx_private & ~31u;
  let target_idx = base + source;
  
  if (prev_count > 0u) {
    for (var i: u32 = 0; i < 32u; i += 1u) {
      let flow_target_idx = base + i;
      var spun_fc = 0u;
      while (atomicLoad(&wg_thread_shuffle_count[flow_target_idx]) < prev_count && spun_fc < SPIN_TIMEOUT) {
        spun_fc += 1u;
      }
    }
  }
  
  let step_idx = 0u;
  atomicStore(&wg_sw_subgroups_u32_steps[step_idx][lidx_private], x);
  atomicStore(&wg_sw_subgroups_flag_steps[step_idx][lidx_private], shuffle_count);
  var spun = 0u;
  while (atomicLoad(&wg_sw_subgroups_flag_steps[step_idx][target_idx]) < shuffle_count && spun < SPIN_TIMEOUT) {
    spun += 1u;
  }
  var shuffled: u32 = atomicLoad(&wg_sw_subgroups_u32_steps[step_idx][target_idx]);
  atomicStore(&wg_thread_shuffle_count[lidx_private], shuffle_count);
  return shuffled;
}`;
  }
  subgroupShuffleXor(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `fn subgroupShuffleXor(x: u32, mask: u32) -> u32 {
  let prev_count = shuffle_count;
  shuffle_count += 1u;
  let base = lidx_private & ~31u;
  let target_idx = base + (sgid ^ mask);
  
  if (prev_count > 0u) {
    for (var i: u32 = 0; i < 32u; i += 1u) {
      let flow_target_idx = base + i;
      var spun_fc = 0u;
      while (atomicLoad(&wg_thread_shuffle_count[flow_target_idx]) < prev_count && spun_fc < SPIN_TIMEOUT) {
        spun_fc += 1u;
      }
    }
  }
  
  let step_idx = 0u;
  atomicStore(&wg_sw_subgroups_u32_steps[step_idx][lidx_private], x);
  atomicStore(&wg_sw_subgroups_flag_steps[step_idx][lidx_private], shuffle_count);
  var spun = 0u;
  while (atomicLoad(&wg_sw_subgroups_flag_steps[step_idx][target_idx]) < shuffle_count && spun < SPIN_TIMEOUT) {
    spun += 1u;
  }
  var shuffled: u32 = atomicLoad(&wg_sw_subgroups_u32_steps[step_idx][target_idx]);
  atomicStore(&wg_thread_shuffle_count[lidx_private], shuffle_count);
  return shuffled;
}`;
  }
  subgroupBallot(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `fn subgroupBallot(pred: bool) -> vec4<u32> {
  const bitsPerOutput = 32u;
  var acc: u32 = select(0u, 1u, pred) << (sgid & (bitsPerOutput - 1));
  let base = lidx_private & ~31u;
  
  let prev_count = shuffle_count;
  shuffle_count += 1u;
  
  if (prev_count > 0u) {
    for (var i: u32 = 0; i < 32u; i += 1u) {
      let flow_target_idx = base + i;
      var spun_fc = 0u;
      while (atomicLoad(&wg_thread_shuffle_count[flow_target_idx]) < prev_count && spun_fc < SPIN_TIMEOUT) {
        spun_fc += 1u;
      }
    }
  }
  
  var step_idx = 0u;
  atomicStore(&wg_sw_subgroups_u32_steps[step_idx][lidx_private], acc);
  atomicStore(&wg_sw_subgroups_flag_steps[step_idx][lidx_private], shuffle_count);
  
  for (var i: u32 = 1; i < bitsPerOutput; i <<= 1) {
    var sourceID: u32 = select(sgid, sgid ^ i, (sgid ^ i) < 32u);
    let target_idx = base + sourceID;
    
    var spun = 0u;
    while (atomicLoad(&wg_sw_subgroups_flag_steps[step_idx][target_idx]) < shuffle_count && spun < SPIN_TIMEOUT) {
      spun += 1u;
    }
    acc |= atomicLoad(&wg_sw_subgroups_u32_steps[step_idx][target_idx]);
    
    step_idx += 1u;
    atomicStore(&wg_sw_subgroups_u32_steps[step_idx][lidx_private], acc);
    atomicStore(&wg_sw_subgroups_flag_steps[step_idx][lidx_private], shuffle_count);
  }
  
  var out: vec4u = vec4u(0);
  let target0 = base + 0u;
  var spun0 = 0u;
  while (atomicLoad(&wg_sw_subgroups_flag_steps[step_idx][target0]) < shuffle_count && spun0 < SPIN_TIMEOUT) {
    spun0 += 1u;
  }
  out[0] = atomicLoad(&wg_sw_subgroups_u32_steps[step_idx][target0]);
  atomicStore(&wg_thread_shuffle_count[lidx_private], shuffle_count);
  return out;
}`;
  }
  subgroupReduce(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `
fn subgroupReduce(in: ${env.datatype}) -> ${env.datatype} {
  wg_sw_subgroups[lidx_private] = in;
  var red: ${env.datatype} = in;
  let base = lidx_private & ~31u;
  for (var i: u32 = 1; i < 32u; i <<= 1) {
    workgroupBarrier();
    var neighbor: u32 = sgid ^ i;
    var neighborVal: ${env.datatype} =
      select(${env.binop.identity},
             wg_sw_subgroups[base + neighbor],
             neighbor < 32u);
    red = binop(red, neighborVal);
    workgroupBarrier();
    wg_sw_subgroups[lidx_private] = red;
  }
  workgroupBarrier();
  return red;
}`;
  }
  subgroupBinopIsU32Add(args = {}) {
    // eslint-disable-next-line no-unused-vars
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `fn binop(a : u32, b : u32) -> u32 {return a+b;}`;
  }
  subgroupAdd(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    return /* wgsl */ `fn subgroupAdd(in: ${env.datatype}) -> ${env.datatype} {
  return subgroupReduce(in);
}`;
  }
  subgroupInclusiveOpScan(args = {}) {
    const env = Object.assign(Object.create(this.env), args); // properties in args overwrite this.env
    /* this is almost certainly faster if we double-buffered */
    return /* wgsl */ `
fn subgroupInclusiveOpScan(in: ${env.datatype}, sgid: u32, sgsz: u32) ->
  ${env.datatype} {
  var red: ${env.datatype} = in;
  var t: ${env.datatype};
  let base = lidx_private & ~31u;
  for (var delta: u32 = 1; delta < 32u; delta <<= 1) {
    workgroupBarrier();
    wg_sw_subgroups[lidx_private] = red;
    workgroupBarrier();
    var neighborIdx: i32 = i32(sgid) - i32(delta);
    if (neighborIdx >= 0) {
      t = wg_sw_subgroups[base + u32(neighborIdx)];
      red = binop(t, red);
    }
  }
  workgroupBarrier();
  return red;
}`;
  }
}
