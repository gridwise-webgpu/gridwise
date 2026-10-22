import {
  BinOpAdd,
  BinOpMin,
  BinOpMax,
  BinOpMultiply,
} from "../src/binop.mjs";
import { datatypeToTypedArray, getBrowserName } from "../src/util.mjs";
import { DLDFScan } from "../src/scandldf.mjs";
import { HierarchicalScan } from "../src/scan.mjs";
import { randomBitsWGSL, randomBitsJS, pmMixJS, philox4x32JS } from "../src/random.mjs";
import {
  BasePrimitive,
  Kernel,
  AllocateBuffer,
  ExecutePrimitive,
} from "../src/primitive.mjs";
import { OneSweepSort } from "../src/onesweep.mjs";
import { MultisplitPerm } from "../src/multisplitperm.mjs";

export async function main(navigator) {
  const isNode = typeof window === "undefined";

  /* ------------------------------------------------------------------ */
  /* Runner factory — independent pass/total counters per suite           */
  /* ------------------------------------------------------------------ */

  function makeRunner(listEl, summaryEl) {
    let passed = 0;
    let total = 0;

    function report(name, success, message = "") {
      total++;
      if (success) passed++;
      const prefix = success ? "PASS" : "FAIL";
      const detail = message ? ` — ${message}` : "";
      console.log(`[${prefix}] ${name}${detail}`);
      if (listEl) {
        const li = document.createElement("li");
        li.className = success ? "pass" : "fail";
        li.textContent = `${name}${detail}`;
        listEl.appendChild(li);
      }
    }

    function summarize() {
      const msg = `Results: ${passed}/${total} tests passed`;
      console.log(`\n${msg}`);
      if (summaryEl) {
        summaryEl.textContent = msg;
        summaryEl.className = passed === total ? "all-pass" : "some-fail";
      }
    }

    async function runTest(name, testFn) {
      try {
        const errorStr = await testFn();
        report(name, errorStr === "", errorStr.split("\n")[0]);
      } catch (e) {
        report(name, false, e.message);
      }
    }

    return { report, summarize, runTest };
  }

  /* ------------------------------------------------------------------ */
  /* Device setup                                                         */
  /* ------------------------------------------------------------------ */

  const summaryEl = document.getElementById("summary");
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) {
    const msg = "Fatal: no GPU adapter";
    console.error(msg);
    if (summaryEl) {
      summaryEl.textContent = msg;
      summaryEl.className = "summary some-fail";
    }
    return;
  }
  let useEmu = false;
  if (!isNode) {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get("disableSubgroups") === "true") {
      useEmu = true;
    }
  }
  const hasSubgroups = adapter.features.has("subgroups") && !useEmu;
  const device = await adapter.requestDevice({
    requiredFeatures: [...(hasSubgroups ? ["subgroups"] : [])],
    requiredLimits: {
      maxComputeWorkgroupStorageSize: 32768 /* required for sort */,
    },
  });
  if (!device) {
    const msg = "Fatal: device creation failed";
    console.error(msg);
    if (summaryEl) {
      summaryEl.textContent = msg;
      summaryEl.className = "summary some-fail";
    }
    return;
  }

  if (!isNode) {
    const platformInfo = document.getElementById("platform-info");
    if (platformInfo) {
      const browserName = getBrowserName();
      const mode = hasSubgroups ? "hardware subgroups" : "emulated subgroups";
      platformInfo.textContent = `(${browserName}, ${mode})`;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Scan / reduce helpers                                                */
  /* ------------------------------------------------------------------ */

  function generateScanInput(datatype, binop, size) {
    const TypedArray = datatypeToTypedArray(datatype);
    const data = new TypedArray(size);
    const isMultiply = binop instanceof BinOpMultiply;
    for (let i = 0; i < size; i++) {
      if (isMultiply) {
        /* keep values tiny to avoid overflow: alternate 1 and 2 */
        data[i] = datatype === "u32" ? (i % 2 === 0 ? 1 : 2) : i % 2 === 0 ? 1 : 2;
      } else {
        switch (datatype) {
          case "u32":
            data[i] = Math.floor(Math.random() * 1000);
            break;
          case "i32":
            data[i] = Math.round((Math.random() - 0.5) * 2000);
            break;
          case "f32":
            /* use small integers: exactly representable, sum stays exact */
            data[i] = Math.round((Math.random() - 0.5) * 200);
            break;
        }
      }
    }
    return data;
  }

  async function runScanTest({ type, datatype, binop, size, Primitive = DLDFScan }) {
    const TypedArray = datatypeToTypedArray(datatype);
    const memsrc = generateScanInput(datatype, binop, size);
    const isReduce = type === "reduce";
    const destBytes = isReduce ? 4 : memsrc.byteLength;

    const primitive = new Primitive({ device, binop, type, datatype });

    const srcBuffer = device.createBuffer({
      size: memsrc.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(srcBuffer, 0, memsrc);

    const destBuffer = device.createBuffer({
      size: destBytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    const mappableDest = device.createBuffer({
      size: destBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    await primitive.execute({ inputBuffer: srcBuffer, outputBuffer: destBuffer });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(destBuffer, 0, mappableDest, 0, destBytes);
    device.queue.submit([encoder.finish()]);

    await mappableDest.mapAsync(GPUMapMode.READ);
    const memdest = new TypedArray(mappableDest.getMappedRange().slice());
    mappableDest.unmap();

    srcBuffer.destroy();
    destBuffer.destroy();
    mappableDest.destroy();

    return primitive.validate({ inputBuffer: memsrc, outputBuffer: memdest });
  }

  /* ------------------------------------------------------------------ */
  /* Sort helpers                                                         */
  /* ------------------------------------------------------------------ */

  function generateSortInput(datatype, size) {
    const TypedArray = datatypeToTypedArray(datatype);
    const data = new TypedArray(size);
    for (let i = 0; i < size; i++) {
      switch (datatype) {
        case "u32":
          /* Use 2^28 range so duplicate keys are extremely rare even at
           * multi-million element sizes (avoids sensitivity to sort stability) */
          data[i] = Math.floor(Math.random() * (1 << 28));
          break;
        case "i32":
          data[i] = Math.round((Math.random() - 0.5) * (1 << 28));
          break;
        case "f32":
          data[i] = (Math.random() - 0.5) * 2000;
          break;
      }
    }
    return data;
  }

  async function runSortTest({ type, datatype, direction, size }) {
    const TypedArray = datatypeToTypedArray(datatype);
    const memsrcKeys = generateSortInput(datatype, size);
    const isKeyValue = type === "keyvalue";

    let memsrcPayload = null;
    if (isKeyValue) {
      memsrcPayload = new Uint32Array(size);
      for (let i = 0; i < size; i++) memsrcPayload[i] = i;
    }

    const primitive = new OneSweepSort({
      device,
      datatype,
      inputLength: size,
      type: isKeyValue ? "keyvalue" : "keysonly",
      direction,
      copyOutputToTemp: true /* sorted output lands in keysTemp */,
    });

    const keysInOut = device.createBuffer({
      size: memsrcKeys.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(keysInOut, 0, memsrcKeys);

    const keysTemp = device.createBuffer({
      size: memsrcKeys.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    const mappableKeys = device.createBuffer({
      size: memsrcKeys.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const executeArgs = { keysInOut, keysTemp };

    let payloadInOut, payloadTemp, mappablePayload;
    if (isKeyValue) {
      payloadInOut = device.createBuffer({
        size: memsrcPayload.byteLength,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(payloadInOut, 0, memsrcPayload);

      payloadTemp = device.createBuffer({
        size: memsrcPayload.byteLength,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });

      mappablePayload = device.createBuffer({
        size: memsrcPayload.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      executeArgs.payloadInOut = payloadInOut;
      executeArgs.payloadTemp = payloadTemp;
    }

    await primitive.execute(executeArgs);

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(keysTemp, 0, mappableKeys, 0, keysTemp.size);
    if (isKeyValue) {
      encoder.copyBufferToBuffer(
        payloadTemp,
        0,
        mappablePayload,
        0,
        payloadTemp.size
      );
    }
    device.queue.submit([encoder.finish()]);

    await mappableKeys.mapAsync(GPUMapMode.READ);
    const memdestKeys = new TypedArray(mappableKeys.getMappedRange().slice());
    mappableKeys.unmap();

    let memdestPayload = null;
    if (isKeyValue) {
      await mappablePayload.mapAsync(GPUMapMode.READ);
      memdestPayload = new Uint32Array(mappablePayload.getMappedRange().slice());
      mappablePayload.unmap();
    }

    keysInOut.destroy();
    keysTemp.destroy();
    mappableKeys.destroy();
    if (isKeyValue) {
      payloadInOut.destroy();
      payloadTemp.destroy();
      mappablePayload.destroy();
    }

    const validateArgs = { inputKeys: memsrcKeys, outputKeys: memdestKeys };
    if (isKeyValue) {
      validateArgs.inputPayload = memsrcPayload;
      validateArgs.outputPayload = memdestPayload;
    }
    return primitive.validate(validateArgs);
  }

  /* ------------------------------------------------------------------ */
  /* Test definitions                                                     */
  /* ------------------------------------------------------------------ */

  /* [type, datatype, BinOpClass, size] */
  const scanTests = [
    /* cover all three scan types × all three datatypes with Add */
    ["exclusive", "i32", BinOpAdd, 256],
    ["exclusive", "u32", BinOpAdd, 256],
    ["exclusive", "f32", BinOpAdd, 256],
    ["inclusive", "i32", BinOpAdd, 256],
    ["inclusive", "u32", BinOpAdd, 256],
    ["inclusive", "f32", BinOpAdd, 256],
    ["reduce", "i32", BinOpAdd, 256],
    ["reduce", "u32", BinOpAdd, 256],
    ["reduce", "f32", BinOpAdd, 256],
    /* cover all binops */
    ["exclusive", "i32", BinOpMin, 256],
    ["exclusive", "i32", BinOpMax, 256],
    ["exclusive", "i32", BinOpMultiply, 16] /* small size avoids overflow */,
    ["inclusive", "u32", BinOpMin, 256],
    ["inclusive", "u32", BinOpMax, 256],
    ["reduce", "i32", BinOpMin, 256],
    ["reduce", "i32", BinOpMax, 256],
    /* edge-case sizes */
    ["exclusive", "i32", BinOpAdd, 4] /* smaller than one workgroup */,
    ["exclusive", "i32", BinOpAdd, 65536] /* spans many workgroups */,
    ["reduce", "u32", BinOpAdd, 65536],
  ];

  /* [type, datatype, direction, size] */
  const sortTests = [
    /* cover all datatypes ascending */
    ["keysonly", "u32", "ascending", 256],
    ["keysonly", "i32", "ascending", 256],
    ["keysonly", "f32", "ascending", 256],
    /* descending */
    ["keysonly", "u32", "descending", 256],
    ["keysonly", "i32", "descending", 256],
    ["keysonly", "f32", "descending", 256],
    /* edge-case sizes */
    ["keysonly", "u32", "ascending", 32] /* smaller than one tile */,
    ["keysonly", "u32", "ascending", 4096],
    /* key-value */
    ["keyvalue", "u32", "ascending", 256],
    ["keyvalue", "i32", "ascending", 256],
    ["keyvalue", "u32", "descending", 256],
    ["keyvalue", "i32", "descending", 256],
  ];

  /* Large tests: MB-sized buffers (4 bytes/element), sizes divisible by 4,
   * not powers of 2. BinOpMultiply omitted (overflow at large sizes). */
  const largeScanTests = [
    ["exclusive", "i32", BinOpAdd, 1_000_000],
    ["exclusive", "u32", BinOpAdd, 1_000_000],
    ["exclusive", "f32", BinOpAdd, 1_000_000],
    ["inclusive", "i32", BinOpAdd, 3_000_000],
    ["inclusive", "u32", BinOpAdd, 3_000_000],
    ["inclusive", "f32", BinOpAdd, 3_000_000],
    ["reduce", "i32", BinOpAdd, 1_000_000],
    ["reduce", "u32", BinOpAdd, 1_000_000],
    ["reduce", "f32", BinOpAdd, 1_000_000],
    ["exclusive", "i32", BinOpMin, 2_000_000],
    ["exclusive", "i32", BinOpMax, 2_000_000],
    ["reduce", "u32", BinOpMin, 3_000_000],
    ["reduce", "u32", BinOpMax, 3_000_000],
  ];

  const largeSortTests = [
    ["keysonly", "u32", "ascending",  1_000_000],
    ["keysonly", "i32", "ascending",  1_000_000],
    ["keysonly", "f32", "ascending",  1_000_000],
    ["keysonly", "u32", "descending", 3_000_000],
    ["keysonly", "i32", "descending", 3_000_000],
    ["keysonly", "f32", "descending", 3_000_000],
    ["keyvalue", "u32", "ascending",  2_000_000],
    ["keyvalue", "i32", "ascending",  2_000_000],
    ["keyvalue", "u32", "descending", 2_000_000],
    ["keyvalue", "i32", "descending", 2_000_000],
  ];

  /* ------------------------------------------------------------------ */
  /* Run small tests                                                      */
  /* ------------------------------------------------------------------ */

  /* destroy() frees the buffers a primitive allocated itself (here, DLDFScan's
   * spine and parameters) but not the caller's; executing again reallocates. */
  async function runDestroyTest() {
    const size = 65536;
    const binop = new BinOpAdd({ datatype: "u32" });
    const input = new Uint32Array(size).fill(1);
    const inputBuffer = device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inputBuffer, 0, input);
    const outputBuffer = device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const mappable = device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const primitive = new DLDFScan({ device, binop, type: "exclusive", datatype: "u32" });
    await primitive.execute({ inputBuffer, outputBuffer });
    primitive.destroy();
    let errors = "";
    if (primitive.hasBuffer("spine")) errors += "\nspine still registered after destroy()";
    if (!primitive.hasBuffer("inputBuffer")) errors += "\ncaller's inputBuffer was dropped";
    /* the caller's buffers must still work, and execute() must reallocate */
    device.queue.writeBuffer(outputBuffer, 0, new Uint32Array(size));
    await primitive.execute({ inputBuffer, outputBuffer });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputBuffer, 0, mappable, 0, size * 4);
    device.queue.submit([encoder.finish()]);
    await mappable.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(mappable.getMappedRange().slice());
    mappable.unmap();
    for (let i = 0; i < size && errors === ""; i++) {
      if (out[i] !== i) errors += `\nafter destroy(): element ${i} is ${out[i]}, expected ${i}`;
    }
    primitive.destroy();
    for (const b of [inputBuffer, outputBuffer, mappable]) b.destroy();
    return errors;
  }

  /* ExecutePrimitive: a primitive that runs a DLDFScan into a buffer it
   * allocates itself, then adds 1 to every element in its own kernel. The
   * scan's output buffer exists only as this primitive's AllocateBuffer, so
   * counting kernels (for GPU timing) has to allocate it early. */
  class ScanPlusOne extends BasePrimitive {
    constructor(args) {
      super(args);
      this.knownBuffers = ["inputBuffer", "outputBuffer"];
      this.workgroupSize = 256;
      this.scan = new DLDFScan({
        device: this.device,
        binop: new BinOpAdd({ datatype: "u32" }),
        type: "exclusive",
        datatype: "u32",
        ...(this.disableSubgroups && { disableSubgroups: true }),
      });
    }
    compute() {
      const size = this.getBuffer("inputBuffer").size;
      return [
        new AllocateBuffer({ label: "scanned", size, clearBufferOnReuse: false }),
        new ExecutePrimitive({
          primitive: this.scan,
          buffers: { inputBuffer: "inputBuffer", outputBuffer: "scanned" },
        }),
        new Kernel({
          kernel: /* wgsl */ `
@group(0) @binding(0) var<storage, read> scanned: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u,
                                      @builtin(num_workgroups) nwg: vec3u) {
  let i = id.x + id.y * nwg.x * 256u;
  if (i < arrayLength(&outputBuffer)) { outputBuffer[i] = scanned[i] + 1u; }
}`,
          bufferTypes: [["read-only-storage", "storage"]],
          bindings: [["scanned", "outputBuffer"]],
          getDispatchGeometry: () =>
            this.getSimpleDispatchGeometry({ workgroupCount: Math.ceil(size / 4 / 256) }),
        }),
      ];
    }
    destroy() {
      super.destroy();
      this.scan.destroy();
    }
  }

  async function runExecutePrimitiveTest() {
    const size = 65536 + 1024;
    const input = new Uint32Array(size);
    for (let i = 0; i < size; i++) input[i] = (i * 7 + 3) % 11;
    const inputBuffer = device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inputBuffer, 0, input);
    const outputBuffer = device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const mappable = device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const primitive = new ScanPlusOne({ device, ...(useEmu && { disableSubgroups: true }) });
    let errors = "";
    /* counting before the first execute must work (allocates "scanned" early) */
    primitive.registerBuffer({ label: "inputBuffer", buffer: inputBuffer, device });
    primitive.registerBuffer({ label: "outputBuffer", buffer: outputBuffer, device });
    const kernels = primitive.countKernels();
    if (kernels !== 2) errors += `\ncountKernels() is ${kernels}, expected 2 (the scan's and our own)`;
    await primitive.execute({ inputBuffer, outputBuffer });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputBuffer, 0, mappable, 0, size * 4);
    device.queue.submit([encoder.finish()]);
    await mappable.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(mappable.getMappedRange().slice());
    mappable.unmap();
    let sum = 0;
    for (let i = 0; i < size && errors === ""; i++) {
      if (out[i] !== sum + 1) errors += `\nelement ${i}: expected ${sum + 1}, saw ${out[i]}`;
      sum += input[i];
    }
    primitive.destroy();
    for (const b of [inputBuffer, outputBuffer, mappable]) b.destroy();
    return errors;
  }

  /* random.mjs: the JS Philox against Random123's known-answer vectors, then
   * the WGSL functions against their JS mirrors for 4096 inputs. */
  class RandomBitsTest extends BasePrimitive {
    constructor(args) {
      super(args);
      this.knownBuffers = ["outputBuffer"];
    }
    compute() {
      return [
        new Kernel({
          kernel: /* wgsl */ `
${randomBitsWGSL(this.rng)}
@group(0) @binding(0) var<storage, read_write> outputBuffer: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  var ctr = 0u;
  outputBuffer[4u * i] = pm_mix(i, 0x2545f491u);
  outputBuffer[4u * i + 1u] = pm_unmix(pm_mix(i, 77u), 77u);
  outputBuffer[4u * i + 2u] = pm_rand(i, i >> 3u, 0x5eedu);
  outputBuffer[4u * i + 3u] = pm_bounded(i + 1u, i, 9u, &ctr);
}`,
          bufferTypes: [["storage"]],
          bindings: [["outputBuffer"]],
          getDispatchGeometry: () => [16],
        }),
      ];
    }
  }

  async function runRandomBitsTest(rng) {
    let errors = "";
    if (!rng) {
      const hex = (a) => a.map((x) => x.toString(16).padStart(8, "0")).join(" ");
      const kat = [
        [[0, 0, 0, 0], [0, 0], "6627e8d5 e169c58d bc57ac4c 9b00dbd8"],
        [[0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff], [0xffffffff, 0xffffffff], "408f276d 41c83b0e a20bc7c6 6d5451fd"],
        [[0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], [0xa4093822, 0x299f31d0], "d16cfe09 94fdcceb 5001e420 24126ea1"],
      ];
      for (const [c, k, want] of kat) {
        const got = hex(philox4x32JS(c, k));
        if (got !== want) errors += `\nphilox4x32JS(${hex(c)}): ${got}, expected ${want}`;
      }
    }
    const n = 4096;
    const outputBuffer = device.createBuffer({
      size: n * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const mappable = device.createBuffer({
      size: n * 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const primitive = new RandomBitsTest({ device, rng });
    await primitive.execute({ outputBuffer });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputBuffer, 0, mappable, 0, n * 16);
    device.queue.submit([encoder.finish()]);
    await mappable.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(mappable.getMappedRange().slice());
    mappable.unmap();
    const R = randomBitsJS(rng);
    for (let i = 0; i < n && errors.split("\n").length < 5; i++) {
      const want = [pmMixJS(i, 0x2545f491), i, R.rand(i, i >>> 3, 0x5eed), R.bounded(i + 1, i, 9, { value: 0 })];
      for (let j = 0; j < 4; j++) {
        if (out[4 * i + j] !== want[j]) {
          errors += `\ninput ${i}, function ${["pm_mix", "pm_unmix", "pm_rand", "pm_bounded"][j]}: GPU ${out[4 * i + j]}, JS ${want[j]}`;
        }
      }
    }
    primitive.destroy();
    outputBuffer.destroy();
    mappable.destroy();
    return errors;
  }

  /* ------------------------------------------------------------------ */
  /* Random permutations                                                  */
  /* ------------------------------------------------------------------ */

  /* MultisplitPerm writes a permutation of 0..n-1; validate() checks it is a
   * permutation in group order and, when reproducible, that it equals the
   * CPU reference bit for bit. */
  async function runPermTest({ n, options }) {
    const primitive = new MultisplitPerm({
      device,
      n,
      seed: 0x5eed,
      ...(useEmu && { disableSubgroups: true }),
      ...options,
    });
    const outputBuffer = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const mappable = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    await primitive.execute({ outputBuffer });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputBuffer, 0, mappable, 0, n * 4);
    device.queue.submit([encoder.finish()]);
    await mappable.mapAsync(GPUMapMode.READ);
    const output = new Uint32Array(mappable.getMappedRange().slice());
    mappable.unmap();
    mappable.destroy();
    outputBuffer.destroy();
    const result = primitive.validate({ outputBuffer: output });
    primitive.destroy();
    return result;
  }

  /* [n, options] */
  const permTests = [
    [65536, {}],
    [65536, { reproducible: true }],
    [100003, { carry: "indices" }] /* not a power of two */,
    [131072, { reproducible: true, split: "3+3+1" }] /* three passes */,
    [5000, { reproducible: true, localCap: 256 }] /* overflow path */,
    [16384, { reproducible: true, rng: { key: [1, 2], rounds: 7 }, carry: "pairs" }],
  ];
  const largePermTests = [
    [1 << 20, { reproducible: true }] /* CPU reference: keep it modest */,
    [1 << 24, {}],
  ];

  const listEl = isNode ? null : document.getElementById("test-list");
  const { runTest, summarize } = makeRunner(listEl, summaryEl);

  for (const [type, datatype, BinOpClass, size] of scanTests) {
    const binop = new BinOpClass({ datatype });
    const opName = BinOpClass.name.replace("BinOp", "").toLowerCase();
    const name = `scan  ${type.padEnd(9)} ${datatype}  ${opName.padEnd(8)} size=${size}`;
    await runTest(name, () => runScanTest({ type, datatype, binop, size }));
  }

  /* the same scans with HierarchicalScan (reduce-then-scan built on wgScan
   * and ExecutePrimitive), which also runs without subgroups using barriers only */
  for (const [type, datatype, BinOpClass, size] of scanTests) {
    const binop = new BinOpClass({ datatype });
    const opName = BinOpClass.name.replace("BinOp", "").toLowerCase();
    const name = `hscan ${type.padEnd(9)} ${datatype}  ${opName.padEnd(8)} size=${size}`;
    await runTest(name, () => runScanTest({ type, datatype, binop, size, Primitive: HierarchicalScan }));
  }

  for (const [type, datatype, direction, size] of sortTests) {
    const name = `sort  ${type.padEnd(9)} ${datatype}  ${direction.padEnd(10)} size=${size}`;
    await runTest(name, () => runSortTest({ type, datatype, direction, size }));
  }

  await runTest("primitive destroy() frees its own buffers, then re-executes", runDestroyTest);
  await runTest("ExecutePrimitive: scan nested in another primitive", runExecutePrimitiveTest);
  await runTest("random bits: keyed hash, WGSL = JS; Philox known answers", () => runRandomBitsTest(null));
  await runTest("random bits: Philox4x32-7, WGSL = JS", () => runRandomBitsTest({ key: [0x12345678, 0x9abcdef0], rounds: 7 }));

  for (const [n, options] of permTests) {
    const name = `perm  MultisplitPerm n=${n} ${JSON.stringify(options)}`;
    await runTest(name, () => runPermTest({ n, options }));
  }

  summarize();

  /* ------------------------------------------------------------------ */
  /* Large tests                                                          */
  /* ------------------------------------------------------------------ */

  async function runLargeTests(largeListEl, largeSummaryEl) {
    console.log("\n--- Large Tests ---");
    const { runTest: runLarge, summarize: summarizeLarge } = makeRunner(
      largeListEl,
      largeSummaryEl
    );

    for (const [type, datatype, BinOpClass, size] of largeScanTests) {
      const binop = new BinOpClass({ datatype });
      const opName = BinOpClass.name.replace("BinOp", "").toLowerCase();
      const name = `scan  ${type.padEnd(9)} ${datatype}  ${opName.padEnd(8)} size=${size}`;
      await runLarge(name, () => runScanTest({ type, datatype, binop, size }));
    }
    for (const [type, datatype, BinOpClass, size] of largeScanTests) {
      const binop = new BinOpClass({ datatype });
      const opName = BinOpClass.name.replace("BinOp", "").toLowerCase();
      const name = `hscan ${type.padEnd(9)} ${datatype}  ${opName.padEnd(8)} size=${size}`;
      await runLarge(name, () => runScanTest({ type, datatype, binop, size, Primitive: HierarchicalScan }));
    }

    for (const [type, datatype, direction, size] of largeSortTests) {
      const name = `sort  ${type.padEnd(9)} ${datatype}  ${direction.padEnd(10)} size=${size}`;
      await runLarge(name, () => runSortTest({ type, datatype, direction, size }));
    }

    for (const [n, options] of largePermTests) {
      const name = `perm  MultisplitPerm n=${n} ${JSON.stringify(options)}`;
      await runLarge(name, () => runPermTest({ n, options }));
    }

    summarizeLarge();
  }

  if (!isNode) {
    const btn = document.getElementById("run-large-btn");
    btn.disabled = false;
    btn.onclick = async () => {
      btn.disabled = true;
      const largeSection = document.getElementById("large-section");
      largeSection.style.display = "block";
      const largeListEl = document.getElementById("large-test-list");
      const largeSummaryEl = document.getElementById("large-summary");
      await runLargeTests(largeListEl, largeSummaryEl);
      device.destroy();
    };
  } else {
    await runLargeTests(null, null);
    device.destroy();
  }
}

main(navigator);
