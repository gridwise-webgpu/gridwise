import { Pane } from "https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js";
import { BinOpAdd, BinOpMax, BinOpMin, makeBinOp } from "../binop.mjs";
import {
  datatypeToTypedArray,
  logspaceRounded,
  datatypeToBytes,
} from "../util.mjs";
import { DLDFScan } from "../scandldf.mjs";
import { OneSweepSort } from "../onesweep.mjs";

let Plot = await import(
  "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm"
);

/* set up a WebGPU device */
const adapter = await navigator.gpu?.requestAdapter();
const hasSubgroups = adapter.features.has("subgroups");
const hasTimestampQuery = adapter.features.has("timestamp-query");
const device = await adapter?.requestDevice({
  requiredLimits: {
    /* this larger-than-default is only necessary for sort */
    maxComputeWorkgroupStorageSize: 32768,
  },
  requiredFeatures: [
    ...(hasTimestampQuery ? ["timestamp-query"] : []),
    ...(hasSubgroups ? ["subgroups"] : []),
  ],
});

if (!device) {
  console.error("Fatal error: Device does not support WebGPU.");
}

function isScan(primitive) {
  return primitive === "exclusive" || primitive === "inclusive";
}
function isReduce(primitive) {
  return primitive === "reduce";
}
function isSort(primitive) {
  return primitive === "sort_keys" || primitive === "sort_values";
}

/* set up the UI, with parameters stored in the "params" object */
const pane = new Pane();
const params = {
  /* defaults */
  primitive: "exclusive",
  datatype: "u32",
  binop: "add",
  direction: "ascending",
  inputLengthStart: 2 ** 20,
  inputLengthEnd: 2 ** 22,
  inputCount: 3,
  trials: 5,
};
pane
  .addBinding(params, "primitive", {
    options: {
      // what it shows : what it returns
      exclusive_scan: "exclusive",
      inclusive_scan: "inclusive",
      reduce: "reduce",
      sort_keys: "sort_keys",
      sort_pairs: "sort_pairs",
    },
  })
  /* this specializes the options to each primitive */
  .on("change", (ev) => {
    switch (ev.value) {
      case "exclusive":
      case "inclusive":
      case "reduce":
        binopPane.hidden = false;
        directionPane.hidden = true;
        break;
      case "sort_keys":
      case "sort_pairs":
        binopPane.hidden = true;
        directionPane.hidden = false;
        break;
      default:
        console.error("Unhandled value in params/primitive binding", ev);
        break;
    }
  });
pane.addBinding(params, "datatype", {
  options: {
    // what it shows : what it returns
    u32: "u32",
    i32: "i32",
    f32: "f32",
  },
});
const binopPane = pane.addBinding(params, "binop", {
  options: {
    // what it shows : what it returns
    add: "add",
    max: "max",
    min: "min",
  },
});
const directionPane = pane.addBinding(params, "direction", {
  options: {
    // what it shows : what it returns
    ascending: "ascending",
    descending: "descending",
  },
});
directionPane.hidden = true;
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
  <li>Primitive: ${params.primitive}
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

    /*for values*/
    const payLoadsrc = new (datatypeToTypedArray(params.datatype))(inputLength);


    /* generate ~random input datasets that are friendly for a
     * particular primitive */
    for (let i = 0; i < inputLength; i++) {
      switch (params.primitive) {
        case "exclusive":
        case "inclusive":
        case "reduce":
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
          break;
        case "sort_keys":

          /* for sorting, we want all different values */
          switch (params.datatype) {
            case "u32":
              /* roughly, [0, 2^28] */
              memsrc[i] = Math.floor(Math.random() * Math.pow(2, 28));

              break;
            case "f32":
            case "i32":
              /* roughly, [-2^28, 2^28], ints */
              memsrc[i] =
                (Math.random() < 0.5 ? 1 : -1) *
                Math.floor(Math.random() * Math.pow(2, 28));
              break;
          }
          break;
        case "sort_pairs":
          switch (params.datatype) {
            case "u32":
              memsrc[i] = Math.floor(Math.random() * Math.pow(2, 28));
              payLoadsrc[i] = Math.floor(Math.random() * Math.pow(2, 28));
              break;
            case "f32":
            case "i32":
              memsrc[i] = (Math.random() < 0.5 ? 1 : -1) *
                Math.floor(Math.random() * Math.pow(2, 28));

              payLoadsrc[i] = (Math.random() < 0.5 ? 1 : -1) *
                Math.floor(Math.random() * Math.pow(2, 28))
              break;
          }
          break;
      }
    }
    console.log("input array", memsrc);

    /* declare the primitive */
    let primitive;
    switch (params.primitive) {
      case "sort_keys":
        primitive = new OneSweepSort({
          device,
          datatype: params.datatype,
          direction: params.direction,
          copyOutputToTemp: true,
          inputLength: inputLength,
        });
        break;
      case "sort_pairs":
        primitive = new OneSweepSort({
          device,
          datatype: params.datatype,
          direction: params.direction,
          copyOutputToTemp: true,
          type: "keyvalue",
          inputLength: inputLength,
        });
        break;
      default:
        primitive = new DLDFScan({
          device,
          binop: makeBinOp({ op: params.binop, datatype: params.datatype }),
          type: params.primitive,
          datatype: params.datatype,
        });
        break;
    }

    /* size the output */
    let memdestBytes;
    if (params.primitive === "reduce") {
      memdestBytes = 4;
    } else {
      memdestBytes = memsrc.byteLength;
    }

    /* allocate/create buffers on the GPU to hold in/out data */
    const memsrcBuffer = device.createBuffer({
      label: `memory source buffer (${params.datatype})`,
      size: memsrc.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC /* allows copy from this buffer for sort validation */,
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

    /*buffers for values*/
    const payLoadsrcBuffer = device.createBuffer({
      label: `value source buffer (${params.datatype})`,
      size: payLoadsrc.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(payLoadsrcBuffer, 0, payLoadsrc);

    const payloadDestBuffer = device.createBuffer({
      label: "value destination buffer",
      size: memdestBytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST
    });

    const mappablePayloadDestBuffer = device.createBuffer({
      label: "mappable value destination buffer",
      size: memdestBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    /* actually run the primitive */
    /* sort and scan have different arguments */
    const primitiveOptions = {
      trials: params.trials,
      enableGPUTiming: hasTimestampQuery,
      enableCPUTiming: true,
    };
    switch (params.primitive) {
      case "sort_keys":
        /* call once, ignore result */
        await primitive.execute({
          keysInOut: memsrcBuffer,
          keysTemp: memdestBuffer,
        });
        /* call trials times */
        await primitive.execute({
          keysInOut: memsrcBuffer,
          keysTemp: memdestBuffer,
          ...primitiveOptions,
        });
        break;

      case "sort_pairs":
        /*call once, ignore results*/
        await primitive.execute({
          keysInOut: memsrcBuffer,
          keysTemp: memdestBuffer,
          payloadInOut: payLoadsrcBuffer,
          payloadTemp: payloadDestBuffer,
        });
        /*call trials times*/
        await primitive.execute({
          keysInOut: memsrcBuffer,
          keysTemp: memdestBuffer,
          payloadInOut: payLoadsrcBuffer,
          payloadTemp: payloadDestBuffer,
          ...primitiveOptions,
        })
        break;

      default:
        /*call once, ignore results*/
        await primitive.execute({
          inputBuffer: memsrcBuffer,
          outputBuffer: memdestBuffer,
        });
        /*call trials times*/
        await primitive.execute({
          inputBuffer: memsrcBuffer,
          outputBuffer: memdestBuffer,
          ...primitiveOptions,
        });
        break;
    }

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
    if (params.primitive == "sort_pairs") {
      encoder.copyBufferToBuffer(
        payloadDestBuffer,
        0,
        mappablePayloadDestBuffer,
        0,
        mappablePayloadDestBuffer.size
      );
    }
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);



    await mappableMemdestBuffer.mapAsync(GPUMapMode.READ);
    const memdest = new (datatypeToTypedArray(params.datatype))(
      mappableMemdestBuffer.getMappedRange().slice()
    );
    mappableMemdestBuffer.unmap();

    /*Read values back to cpu(only for sort_pairs)*/
    let payloadDest;
    if (params.primitive == "sort_pairs") {
      await mappablePayloadDestBuffer.mapAsync(GPUMapMode.READ);
      payloadDest = new (datatypeToTypedArray(params.datatype))(
        mappablePayloadDestBuffer.getMappedRange().slice()
      );
      mappablePayloadDestBuffer.unmap();

    }
    console.log("output array", memdest);
    if (params.primitive === "sort_pairs") {
      console.log("output values", payloadDest);
    }

    if (primitive.validate) {
      let errorstr;
      switch (params.primitive) {
        case "sort_keys":
          errorstr = primitive.validate({
            inputKeys: memsrc,
            outputKeys: memdest,
          });
          break;
        case "sort_pairs":
          errorstr = primitive.validate({
            inputKeys: memsrc,
            outputKeys: memdest,
            inputPayload: payLoadsrc,
            outputPayload: payloadDest,

          });
          break;
        default:
          /* this validation is only possible because the output is copied
           * into outputBuffer by configuring the OneSweepSort primitive with
           * copyOutputToTemp: true */
          errorstr = primitive.validate({
            inputBuffer: memsrc,
            outputBuffer: memdest,
          });
          break;
      }
      if (errorstr === "") {
        returnStr += `Validation passed (input length: ${inputLength})<br/ >`;
      } else {
        returnStr += `Validation failed (input length: ${inputLength})<br/ >${errorstr}<br/ >`;
      }
    } else {
      returnStr += `Validation not performed (input length: ${inputLength})<br/ >`;
    }
  } /* end loop over input lengths */
  plotResults(results);
  return returnStr;
}

function plotResults(results) {
  console.log(results);
  const plots = [
    {
      x: { field: "inputBytes", label: "Input array size (B)" },
      y: { field: "bandwidth", label: "Bandwidth (GB/s)" },
      stroke: { field: "timing" },
      caption:
        "BANDWIDTH | CPU timing (performance.now), GPU timing (timestamps)",
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
    const schema = {
      marks: [
        Plot[mark](results, {
          x: plot.x.field,
          y: plot.y.field,
          ...("fx" in plot && { fx: plot.fx.field }),
          ...("fy" in plot && { fy: plot.fy.field }),
          ...("stroke" in plot && {
            stroke: plot.stroke.field,
          }),
          tip: true,
        }),
        Plot.text(
          results,
          Plot.selectLast({
            x: plot.x.field,
            y: plot.y.field,
            ...("stroke" in plot && {
              z: plot.stroke.field,
              text: plot.stroke.field,
            }),
            ...("fx" in plot && { fx: plot.fx.field }),
            ...("fy" in plot && { fy: plot.fy.field }),
            textAnchor: "start",
            clip: false,
            dx: 3,
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
      ],
      x: { type: "log", label: plot?.x?.label ?? "XLABEL" },
      y: { type: "log", label: plot?.y?.label ?? "YLABEL" },
      ...("fx" in plot && {
        fx: { label: plot.fx.label },
      }),
      ...("fy" in plot && {
        fy: { label: plot.fy.label },
      }),
      ...(("fx" in plot || "fy" in plot) && { grid: true }),
      color: { type: "ordinal", legend: true },
      width: 1280,
      title: plot?.title,
      subtitle: plot?.subtitle,
      caption: plot?.caption,
    };
    const plotted = Plot.plot(schema);
    const div = document.querySelector("#plot");
    div.append(plotted);
    div.append(document.createElement("hr"));
  }
}
