import { Pane } from "https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js";
import { BinOpAdd, BinOpMax, BinOpMin, makeBinOp } from "../binop.mjs";
import { datatypeToTypedArray } from "../util.mjs";
import { DLDFScan } from "../scandldf.mjs";
import { OneSweepSort } from "../onesweep.mjs";

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
  inputLength: 2 ** 20,
};
pane
  .addBinding(params, "primitive", {
    options: {
      // what it shows : what it returns
      exclusive_scan: "exclusive",
      inclusive_scan: "inclusive",
      reduce: "reduce",
      sort_keys: "sort_keys",
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
      case "sort_values":
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
pane.addBinding(params, "inputLength", { format: (v) => Math.floor(v) });
const button = pane.addButton({
  title: "Start",
});

button.on("click", async () => {
  if (params.inputLength % 4 !== 0) {
    params.inputLength = Math.floor(params.inputLength / 4) * 4;
  }
  /* because inputLength may change here, we need to refresh the pane */
  pane.refresh();
  const results = document.getElementById("webgpu-results");
  const validation = await buildAndRun();
  results.innerHTML = `<p>I ran this</p>
  <ul>
  <li>Primitive: ${params.primitive}
  <li>Datatype: ${params.datatype}
  <li>Binop: ${params.binop}
  <li>Input length: ${params.inputLength} (items)
  </ul>
  <p>${validation}</p>`;
});
/* end of setting up the UI */

/* all of the work is in this function */
async function buildAndRun() {
  /* generate an input dataset */
  if (params.inputLength % 4 !== 0) {
    console.warn(
      "Input length (currently: ",
      params.inputLength,
      ") must be divisible by 4 (output is likely to be incorrect) "
    );
  }
  const memsrc = new (datatypeToTypedArray(params.datatype))(
    params.inputLength
  );

  /* generate ~random input datasets that are friendly for a
   * particular primitive */
  for (let i = 0; i < params.inputLength; i++) {
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
      case "sort_values":
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
      default:
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

  /* actually run the primitive */
  /* sort and scan have different arguments */
  switch (params.primitive) {
    case "sort_keys":
      await primitive.execute({
        keysInOut: memsrcBuffer,
        keysTemp: memdestBuffer,
      });
      break;
    default:
      await primitive.execute({
        inputBuffer: memsrcBuffer,
        outputBuffer: memdestBuffer,
      });
      break;
  }

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
    let errorstr;
    switch (params.primitive) {
      case "sort_keys":
        errorstr = primitive.validate({
          inputKeys: memsrc,
          outputKeys: memdest,
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
      return "Validation passed";
    } else {
      return `Validation failed:\n${errorstr}`;
    }
  } else {
    return "Validation not performed";
  }
}
