import {
  BinOpAdd,
  BinOpMin,
  BinOpMax,
  BinOpMultiply,
} from "../binop.mjs";
import { datatypeToTypedArray } from "../util.mjs";
import { DLDFScan } from "../scandldf.mjs";
import { OneSweepSort } from "../onesweep.mjs";

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

  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) {
    console.error("Fatal: no GPU adapter");
    return;
  }
  const hasSubgroups = adapter.features.has("subgroups");
  const device = await adapter.requestDevice({
    requiredFeatures: [...(hasSubgroups ? ["subgroups"] : [])],
    requiredLimits: {
      maxComputeWorkgroupStorageSize: 32768 /* required for sort */,
    },
  });
  if (!device) {
    console.error("Fatal: device creation failed");
    return;
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

  async function runScanTest({ type, datatype, binop, size }) {
    const TypedArray = datatypeToTypedArray(datatype);
    const memsrc = generateScanInput(datatype, binop, size);
    const isReduce = type === "reduce";
    const destBytes = isReduce ? 4 : memsrc.byteLength;

    const primitive = new DLDFScan({ device, binop, type, datatype });

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

  const listEl = isNode ? null : document.getElementById("test-list");
  const summaryEl = isNode ? null : document.getElementById("summary");
  const { runTest, summarize } = makeRunner(listEl, summaryEl);

  for (const [type, datatype, BinOpClass, size] of scanTests) {
    const binop = new BinOpClass({ datatype });
    const opName = BinOpClass.name.replace("BinOp", "").toLowerCase();
    const name = `scan  ${type.padEnd(9)} ${datatype}  ${opName.padEnd(8)} size=${size}`;
    await runTest(name, () => runScanTest({ type, datatype, binop, size }));
  }

  for (const [type, datatype, direction, size] of sortTests) {
    const name = `sort  ${type.padEnd(9)} ${datatype}  ${direction.padEnd(10)} size=${size}`;
    await runTest(name, () => runSortTest({ type, datatype, direction, size }));
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

    for (const [type, datatype, direction, size] of largeSortTests) {
      const name = `sort  ${type.padEnd(9)} ${datatype}  ${direction.padEnd(10)} size=${size}`;
      await runLarge(name, () => runSortTest({ type, datatype, direction, size }));
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
