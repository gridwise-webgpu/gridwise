import { BinOpAdd, BinOpMax, BinOpMin } from "../binop.mjs";
import { datatypeToTypedArray } from "../util.mjs";
import { OneSweepSort } from "../onesweep.mjs";

export async function main(navigator) {
  /* set up a WebGPU device */
  const adapter = await navigator.gpu?.requestAdapter();

  const hasSubgroups = adapter.features.has("subgroups");
  const hasTimestampQuery = adapter.features.has("timestamp-query");
  const device = await adapter?.requestDevice({
    requiredFeatures: [
      ...(hasTimestampQuery ? ["timestamp-query"] : []),
      ...(hasSubgroups ? ["subgroups"] : []),
    ],
    requiredLimits: {
      /* this larger-than-default is only necessary for sort */
      maxComputeWorkgroupStorageSize: 32768,
    },
  });

  if (!device) {
    console.error("Fatal error: Device does not support WebGPU.");
  }

  /* configure the primitive */
  /**
   * Choices for configuring this primitive:
   * datatype: "i32", "u32", "f32"
   * inputLength: any multiple of 4 up to max GPUBuffer length
   */
  const datatype = "i32";
  const inputLength = 2 ** 24; // this is item count, not byte count

  /* generate an input dataset */
  if (inputLength % 4 !== 0) {
    console.warn(
      "Input length (currently: ",
      inputLength,
      ") must be divisible by 4 (output is likely to be incorrect) ",
    );
  }
  const memsrc = new (datatypeToTypedArray(datatype))(inputLength);
  for (let i = 0; i < inputLength; i++) {
    /* for sorting, we want all different values */
    switch (datatype) {
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
  }
  console.log("input array", memsrc);

  /* declare the primitive */
  const sortKeysPrimitive = new OneSweepSort({
    device,
    datatype,
    inputLength: inputLength,
    copyOutputToTemp: true /* output is in memdestBuffer */,
  });

  const primitive = sortKeysPrimitive;

  /* size the tempBuffer */
  const memdestBytes = memsrc.byteLength;

  /* allocate/create buffers on the GPU to hold in/out data */
  const memsrcBuffer = device.createBuffer({
    label: `memory source buffer (${datatype})`,
    size: memsrc.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
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
  await primitive.execute({
    keysInOut: memsrcBuffer,
    keysTemp: memdestBuffer,
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
    mappableMemdestBuffer.size,
  );
  const commandBuffer = encoder.finish();
  device.queue.submit([commandBuffer]);

  await mappableMemdestBuffer.mapAsync(GPUMapMode.READ);
  const memdest = new (datatypeToTypedArray(datatype))(
    mappableMemdestBuffer.getMappedRange().slice(),
  );
  mappableMemdestBuffer.unmap();

  console.log("output", memdest);

  if (primitive.validate) {
    const errorstr = primitive.validate({
      inputKeys: memsrc,
      outputKeys: memdest,
    });
    if (errorstr === "") {
      console.info("Validation passed");
    } else {
      console.error(`Validation failed:\n${errorstr}`);
    }
  }
}

main(navigator);
