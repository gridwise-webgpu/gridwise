import { globals, create } from "webgpu";
import path from "path";
import { fileURLToPath } from "url";

// 1. Polyfill WebGPU globals (GPUBufferUsage, etc.)
Object.assign(globalThis, globals);

// 2. Setup dawn.node features
const gpuCreateFeatures = [
  "enable-dawn-features=use_user_defined_labels_in_backend",
];
const gpuInstance = create(gpuCreateFeatures);

// 3. Mock navigator
const navigatorMock = {
  gpu: gpuInstance,
};
Object.defineProperty(globalThis, "navigator", {
  value: navigatorMock,
  configurable: true,
  writable: true,
});

async function main() {
  const adapter = await gpuInstance.requestAdapter();
  if (!adapter) {
    console.error("Fatal error: could not request WebGPU adapter");
    process.exit(1);
  }

  const hasSubgroups = adapter.features.has("subgroups");
  const device = await adapter.requestDevice({
    requiredFeatures: [
      ...(hasSubgroups ? ["subgroups"] : []),
    ],
  });

  if (!device) {
    console.error("Fatal error: could not request WebGPU device");
    process.exit(1);
  }

  console.log(`Using GPU Device: ${adapter.info?.description || "unknown_gpu"}`);
  console.log(`Subgroups support: ${hasSubgroups ? "YES" : "NO"}`);
  console.log("-----------------------------------------");

  const { runSuite } = await import("./test_wgsl_functions.mjs");

  let totalTests = 0;
  let passedTests = 0;
  let failedTests = 0;

  await runSuite(device, (event) => {
    switch (event.type) {
      case "mode_start":
        console.log(`\n=== Running tests in [${event.mode}] mode ===`);
        break;
      case "test_result":
        totalTests++;
        if (event.passed) {
          passedTests++;
          console.log(`[PASS] ${event.name}`);
        } else {
          failedTests++;
          console.error(`[FAIL] ${event.name}`);
          console.error(`       Error: ${event.errorMsg}`);
        }
        break;
      case "suite_end":
        console.log("\n=========================================");
        console.log(`Results: ${passedTests}/${totalTests} tests passed (${failedTests} failed)`);
        if (failedTests > 0) {
          process.exit(1);
        } else {
          process.exit(0);
        }
    }
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
