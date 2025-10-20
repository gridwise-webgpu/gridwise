import { BinOpAdd, BinOpMax, BinOpMin } from "../binop.mjs";
import { datatypeToTypedArray } from "../util.mjs";
import { DLDFScan } from "../scandldf.mjs";

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
    });

    if (!device) {
        console.error("Fatal error: Device does not support WebGPU.");
    }

    /* configure the primitive */
    /**
     * Choices for configuring this primitive:
     * datatype: "i32", "u32", "f32"
     * binop: anything in binop.mjs
     * - Make sure it's imported (at top of file)
     * - BinOpMin, BinOpMax, BinOpAdd will work
     * - BinOpMultiply will work but is likely to overflow
     * inputLength: any multiple of 4 up to max GPUBuffer length
     */
    const datatype = "i32";
    const binop = new BinOpAdd({ datatype });
    const inputLength = 2 ** 24; // this is item count, not byte count

    /* generate an input dataset */
    if (inputLength % 4 !== 0) {
        console.warn(
            "Input length (currently: ",
            inputLength,
            ") must be divisible by 4 (output is likely to be incorrect) "
        );
    }
    const memsrc = new (datatypeToTypedArray(datatype))(inputLength);
    /* the gymnastics below are to try to generate GPU-native {i,u,f}32
     * datatypes, there's probably an easier/faster way to do it */
    for (let i = 0; i < inputLength; i++) {
        switch (datatype) {
            case "u32":
                /* [0, 32], ints */
                memsrc[i] = Math.floor(Math.random() * Math.pow(2, 5));
                break;
            case "i32":
                /*  [-1024, 1024], ints */
                memsrc[i] =
                    (Math.random() < 0.5 ? 1 : -1) *
                    Math.floor(Math.random() * Math.pow(2, 10));
                break;
            case "f32":
                /* attempt to evenly distribute floats between [-1023, 1023] */
                memsrc[i] =
                    (Math.random() < 0.5 ? 1 : -1) *
                    Math.floor(Math.random() * Math.pow(2, 10));
                break;
        }
    }
    console.log("input", memsrc);

    /* declare the primitive */
    const reducePrimitive = new DLDFScan({
        device,
        binop,
        type: "reduce",
        datatype,
    });

    const primitive = reducePrimitive;

    /* size the output - reduce outputs a single value */
    const memdestBytes = 4; // always 4 bytes for a single value

    /* allocate/create buffers on the GPU to hold in/out data */
    const memsrcBuffer = device.createBuffer({
        label: `memory source buffer (${datatype})`,
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
    await primitive.execute({
        inputBuffer: memsrcBuffer,
        outputBuffer: memdestBuffer,
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
    const memdest = new (datatypeToTypedArray(datatype))(
        mappableMemdestBuffer.getMappedRange().slice()
    );
    mappableMemdestBuffer.unmap();

    console.log("output", memdest);

    if (primitive.validate) {
        const errorstr = primitive.validate({
            inputBuffer: memsrc,
            outputBuffer: memdest,
        });
        if (errorstr === "") {
            console.info("Validation passed");
        } else {
            console.error(`Validation failed:\n${errorstr}`);
        }
    }
}

main(navigator);