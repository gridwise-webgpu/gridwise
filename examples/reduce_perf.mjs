import { Pane } from "https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js";
import { BinOpAdd, BinOpMax, BinOpMin, makeBinOp } from "../binop.mjs";
import {
  datatypeToTypedArray,
  logspaceRounded,
  datatypeToBytes,
} from "../util.mjs";
import { DLDFScan } from "../scandldf.mjs";

let Plot = await import(
  "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm"
);

/* set up a WebGPU device */
const adapter = await navigator.gpu?.requestAdapter();
const hasSubgroups = adapter.features.has("subgroups");
const hasTimestampQuery = adapter.features.has("timestamp-query");
const device = await adapter?.requestDevice({
  requiredFeatures: [
    ...(hasTimestampQuery ? ["timestamp-query"] : []),
    ...(hasSubgroups ? ["subgroups"] : []),
  ],
});

if (!device) {
  console.error("Fatal error: Device does not support WebGPU.");
}

/* set up the UI, with parameters stored in the "params" object */
const pane = new Pane();
const params = {
  /* defaults */
  datatype: "u32",
  binop: "add",
  inputLengthStart: 2 ** 20,
  inputLengthEnd: 2 ** 22,
  inputCount: 3,
  trials: 5,
};

pane.addBinding(params, "datatype", {
  options: {
    // what it shows : what it returns
    u32: "u32",
    i32: "i32",
    f32: "f32",
  },
});

pane.addBinding(params, "binop", {
  options: {
    // what it shows : what it returns
    add: "add",
    max: "max",
    min: "min",
  },
});

pane.addBinding(params, "inputLengthStart", { format: (v) => Math.floor(v) });
pane.addBinding(params, "inputLengthEnd", { format: (v) => Math.floor(v) });
pane.addBinding(params, "inputCount", { format: (v) => Math.floor(v) });
pane.addBinding(params, "trials", { format: (v) => Math.floor(v) });

const button = pane.addButton({
  title: "Start",
});

button.on("click", async () => {
  if (params.inputLengthStart % 4 !== 0) {
    params.inputLengthStart = Math.floor(params.inputLengthStart / 4) * 4;
  }
  if (params.inputLengthEnd % 4 !== 0) {
    params.inputLengthEnd = Math.floor(params.inputLengthEnd / 4) * 4;
  }
  params.inputCount = Math.floor(params.inputCount);
  params.trials = Math.floor(params.trials);

  /* because inputLength may change here, we need to refresh the pane */
  pane.refresh();
  const results = document.getElementById("webgpu-results");
  const validation = await buildAndRun();
  results.innerHTML = `<p>I ran this</p>
  <ul>
  <li>Primitive: reduce
  <li>Datatype: ${params.datatype}
  <li>Binop: ${params.binop}
  <li>Input length: ${params.inputCount} lengths from ${params.inputLengthStart} to ${params.inputLengthEnd} (items)
  </ul>
  <p>${validation}</p>`;
});
/* end of setting up the UI */

/* all of the work is in this function */
async function buildAndRun() {
  let returnStr = "";
  const results = new Array(); // push new rows (experiments) onto this

  for (const inputLength of logspaceRounded(
    params.inputLengthStart,
    params.inputLengthEnd,
    params.inputCount
  )) {
    /* generate an input dataset */
    const memsrc = new (datatypeToTypedArray(params.datatype))(inputLength);

    /* generate ~random input datasets that are friendly for reduce */
    for (let i = 0; i < inputLength; i++) {
      switch (params.datatype) {
        case "u32":
          /* roughly, [0, 32], ints */
          memsrc[i] = Math.floor(Math.random() * Math.pow(2, 5));
          break;
        case "f32":
        case "i32":
          /* roughly, [-1024, 1024], ints */
          memsrc[i] =
            (Math.random() < 0.5 ? 1 : -1) *
            Math.floor(Math.random() * Math.pow(2, 10));
          break;
      }
    }
    console.log("input array", memsrc);

    /* declare the primitive */
    const primitive = new DLDFScan({
      device,
      binop: makeBinOp({ op: params.binop, datatype: params.datatype }),
      type: "reduce",
      datatype: params.datatype,
    });

    /* size the output - reduce always outputs a single value (4 bytes) */
    const memdestBytes = 4;

    /* allocate/create buffers on the GPU to hold in/out data */
    const memsrcBuffer = device.createBuffer({
      label: `memory source buffer (${params.datatype})`,
      size: memsrc.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(memsrcBuffer, 0, memsrc);

    const memdestBuffer = device.createBuffer({
      label: "memory destination buffer",
      size: memdestBytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST /* COPY_DST necessary for initialization */,
    });

    const mappableMemdestBuffer = device.createBuffer({
      label: "mappable memory destination buffer",
      size: memdestBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    /* actually run the primitive */
    const primitiveOptions = {
      trials: params.trials,
      enableGPUTiming: hasTimestampQuery,
      enableCPUTiming: true,
    };

    /* call once, ignore result (warmup) */
    await primitive.execute({
      inputBuffer: memsrcBuffer,
      outputBuffer: memdestBuffer,
    });

    /* call trials times */
    await primitive.execute({
      inputBuffer: memsrcBuffer,
      outputBuffer: memdestBuffer,
      ...primitiveOptions,
    });

    /* Append CPU and GPU timing results to "results" array */
    let { gpuTotalTimeNS, cpuTotalTimeNS } = await primitive.getTimingResult();
    console.log(gpuTotalTimeNS, cpuTotalTimeNS);
    const result = {};
    if (gpuTotalTimeNS instanceof Array) {
      // gpuTotalTimeNS might be a list, in which case just sum it up
      result.gpuTotalTimeNSArray = gpuTotalTimeNS;
      gpuTotalTimeNS = gpuTotalTimeNS.reduce((x, a) => x + a, 0);
    }
    result.gputime = gpuTotalTimeNS / params.trials;
    result.cputime = cpuTotalTimeNS / params.trials;
    result.inputBytes = inputLength * datatypeToBytes(primitive.datatype);
    result.bandwidthGPU = primitive.bytesTransferred / result.gputime;
    result.bandwidthCPU = primitive.bytesTransferred / result.cputime;
    result.inputItemsPerSecondE9GPU = inputLength / result.gputime;
    result.inputItemsPerSecondE9CPU = inputLength / result.cputime;

    results.push({
      ...result,
      timing: "GPU",
      time: result.gputime,
      bandwidth: result.bandwidthGPU,
      inputItemsPerSecondE9: result.inputItemsPerSecondE9GPU,
    });
    results.push({
      ...result,
      timing: "CPU",
      time: result.cputime,
      bandwidth: result.bandwidthCPU,
      inputItemsPerSecondE9: result.inputItemsPerSecondE9CPU,
    });

    /* copy output back to host */
    const encoder = device.createCommandEncoder({
      label: "copy result CPU->GPU encoder",
    });
    encoder.copyBufferToBuffer(
      memdestBuffer,
      0,
      mappableMemdestBuffer,
      0,
      mappableMemdestBuffer.size
    );
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    await mappableMemdestBuffer.mapAsync(GPUMapMode.READ);
    const memdest = new (datatypeToTypedArray(params.datatype))(
      mappableMemdestBuffer.getMappedRange().slice()
    );
    mappableMemdestBuffer.unmap();

    console.log("output array", memdest);

    if (primitive.validate) {
      const errorstr = primitive.validate({
        inputBuffer: memsrc,
        outputBuffer: memdest,
      });

      if (errorstr === "") {
        returnStr += `Validation passed (input length: ${inputLength})<br/>`;
      } else {
        returnStr += `Validation failed (input length: ${inputLength})<br/>${errorstr}<br/>`;
      }
    } else {
      returnStr += `Validation not performed (input length: ${inputLength})<br/>`;
    }
  } /* end loop over input lengths */
  plotResults(results, adapter.info.description);
  return returnStr;
}

function plotResults(results, adapterDescription) {
  const container = document.querySelector("#plot");

  const maxBWs = {
    "Apple M1 Max": 400,
  };

  if (!results || results.length === 0) return;

  const plots = [
    {
      x: { field: "inputBytes", label: "Input array size (B)" },
      y: { field: "bandwidth", label: "Bandwidth (GB/s)" },
      stroke: { field: "timing" },
      caption:
        "BANDWIDTH | CPU timing (performance.now), GPU timing (timestamps)",
      maxBW: maxBWs[adapterDescription],
    },
    {
      x: { field: "inputBytes", label: "Input array size (B)" },
      y: { field: "time", label: "Runtime (ns)" },
      stroke: { field: "timing" },
      caption:
        "RUNTIME | CPU timing (performance.now), GPU timing (timestamps)",
    },
  ];

  for (let plot of plots) {
    const mark = plot.mark ?? "lineY";
    const xField = plot.x.field;
    const yField = plot.y.field;

    // --- DOMAIN & LAYOUT LOGIC ---
    let yDomain = undefined;
    let ruleValue = null;
    let ruleLabel = "";
    let labelPosition = "above"; // Default

    // 1. Calculate Data Extents (needed for Max X and Y)
    const yValues = results.map((r) => r[yField]);
    const xValues = results.map((r) => r[xField]);

    // Find the right-most X value to anchor the text horizontally
    const maxX = xValues.reduce((max, v) => Math.max(max, v || 0), 0);

    if (plot.maxBW) {
      const dataMin = yValues.reduce(
        (min, v) => (v > 0 ? Math.min(min, v) : min),
        Infinity
      );
      const dataMax = yValues.reduce((max, v) => Math.max(max, v || 0), 0);
      const validMin = dataMin === Infinity ? 1 : dataMin;
      const clampThreshold = dataMax * 10;

      if (plot.maxBW > clampThreshold) {
        // CASE A: Clamped (Off Scale) -> Line at top
        yDomain = [validMin, clampThreshold];
        ruleValue = clampThreshold;
        ruleLabel = `Max DRAM BW for ${adapterDescription}: ${plot.maxBW} GB/s (Off Scale)`;
        labelPosition = "below";
      } else {
        // CASE B: Normal View
        yDomain = [validMin, Math.max(dataMax, plot.maxBW)];
        ruleValue = plot.maxBW;
        ruleLabel = `Max DRAM BW for ${adapterDescription}: ${plot.maxBW} GB/s`;

        // If line is effectively at the ceiling, put text below
        labelPosition = plot.maxBW >= dataMax ? "below" : "above";
      }
    }

    const marks = [
      Plot[mark](results, {
        x: xField,
        y: yField,
        ...("fx" in plot && { fx: plot.fx.field }),
        ...("fy" in plot && { fy: plot.fy.field }),
        ...("stroke" in plot && { stroke: plot.stroke.field }),
        tip: true,
        marker: "circle",
      }),
      Plot.text(
        results,
        Plot.selectLast({
          x: xField,
          y: yField,
          ...("stroke" in plot && {
            z: plot.stroke.field,
            text: plot.stroke.field,
            fill: plot.stroke.field,
          }),
          textAnchor: "start",
          dx: 5,
        })
      ),
      Plot.text([plot.text_tl ?? ""], {
        lineWidth: 30,
        dx: 5,
        frameAnchor: "top-left",
      }),
      Plot.text(plot.text_br ?? "", {
        lineWidth: 30,
        dx: 5,
        frameAnchor: "bottom-right",
      }),
    ];

    // --- ADD MAX BW LINE & LABEL ---
    if (ruleValue !== null) {
      // 1. The Red Dotted Line
      marks.push(
        Plot.ruleY([ruleValue], {
          stroke: "red",
          strokeDasharray: "4,4",
          strokeWidth: 2,
        })
      );

      // 2. The Label (Positioned explicitly using data coordinates)
      // We create a tiny 1-item dataset for this label to avoid accessor confusion
      const labelData = [
        {
          x: maxX, // The rightmost X value
          y: ruleValue, // The calculated Y value
          text: ruleLabel,
        },
      ];

      marks.push(
        Plot.text(labelData, {
          x: "x",
          y: "y",
          text: "text",
          fill: "red",
          textAnchor: "end", // Align text to end at the rightmost data point
          fontWeight: "bold",
          dy: labelPosition === "below" ? 15 : -5,
          clip: false, // <--- CRITICAL: Allows text to render even if on the exact chart edge
        })
      );
    }

    const schema = {
      marks: marks,
      marginRight: 60,
      x: { type: "log", label: plot?.x?.label ?? "XLABEL", tickFormat: "~s" },
      y: {
        type: "log",
        label: plot?.y?.label ?? "YLABEL",
        grid: true,
        ...(yDomain && { domain: yDomain }),
      },
      ...("fx" in plot && { fx: { label: plot.fx.label } }),
      ...("fy" in plot && { fy: { label: plot.fy.label } }),
      color: { type: "ordinal", legend: true },
      width: 1280,
      title: plot?.title,
      subtitle: plot?.subtitle,
      caption: plot?.caption,
    };

    const plotted = Plot.plot(schema);
    container.append(plotted);
    container.append(document.createElement("hr"));
  }
}
