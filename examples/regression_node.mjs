import { globals, create } from "webgpu";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// 0. Filter specific experimental getGPUInfo warnings from dawn.node
process.on("warning", (warning) => {
  if (warning.message && warning.message.includes("getGPUInfo")) {
    return;
  }
  console.warn(warning.name + ": " + warning.message);
});

// 1. Polyfill WebGPU globals (GPUBufferUsage, etc.)
Object.assign(globalThis, globals);

// 2. Setup dawn.node features
let resolveMain;
let mainFinished = new Promise((resolve) => {
  resolveMain = resolve;
});

let enableTiming = true;
let cachedDevice = null;

const gpuCreateFeatures = [
  "enable-dawn-features=use_user_defined_labels_in_backend",
];
let adapterDescription = "unknown_gpu";

const gpuInstance = create(gpuCreateFeatures);
const originalRequestAdapter = gpuInstance.requestAdapter;
gpuInstance.requestAdapter = async function (...args) {
  const adapter = await originalRequestAdapter.apply(this, args);
  if (!adapter) {
    resolveMain();
    return null;
  }
  const originalFeatures = adapter.features;
  const mockedFeatures = new Proxy(originalFeatures, {
    get(target, prop, receiver) {
      if (prop === "has") {
        return function(feature) {
          if (feature === "subgroups" && process.env.DISABLE_SUBGROUPS === "true") {
            return false;
          }
          return originalFeatures.has(feature);
        };
      }
      const val = Reflect.get(target, prop, receiver);
      return typeof val === "function" ? val.bind(target) : val;
    }
  });
  Object.defineProperty(adapter, "features", {
    value: mockedFeatures,
    configurable: true,
    writable: true,
  });
  if (adapter.info) {
    const vendor = adapter.info.vendor || "";
    const device = adapter.info.device || "";
    const arch = adapter.info.architecture || "";
    if (device) {
      if (vendor && !device.toLowerCase().startsWith(vendor.toLowerCase())) {
        adapterDescription = `${vendor} ${device}`;
      } else {
        adapterDescription = device;
      }
      if (arch) {
        adapterDescription += ` (${arch})`;
      }
    } else {
      adapterDescription = adapter.info.description || "unknown_gpu";
    }
  } else {
    adapterDescription = "unknown_gpu";
  }
  const originalRequestDevice = adapter.requestDevice;
  adapter.requestDevice = async function (deviceArgs = {}, ...extraArgs) {
    if (cachedDevice) {
      return cachedDevice;
    }
    try {
      deviceArgs.requiredFeatures = deviceArgs.requiredFeatures || [];
      if (adapter.features.has("timestamp-query") && !deviceArgs.requiredFeatures.includes("timestamp-query")) {
        deviceArgs.requiredFeatures.push("timestamp-query");
      }
      const device = await originalRequestDevice.call(this, deviceArgs, ...extraArgs);
      if (!device) {
        resolveMain();
        return null;
      }
      cachedDevice = device;
      const originalDestroy = device.destroy;
      device.destroy = function () {
        if (enableTiming) {
          originalDestroy.call(device);
          resolveMain();
        } else {
          resolveMain();
        }
      };
      return device;
    } catch (e) {
      resolveMain();
      throw e;
    }
  };
  return adapter;
};

const navigatorMock = {
  gpu: gpuInstance,
};
Object.defineProperty(globalThis, "navigator", {
  value: navigatorMock,
  configurable: true,
  writable: true,
});

// 3. Mock minimal document object required by regression.mjs
globalThis.document = {
  getElementById: () => null,
};

// 4. Import BasePrimitive to intercept execution
import { BasePrimitive } from "../primitive.mjs";

const perfResults = [];
const originalExecute = BasePrimitive.prototype.execute;

// Intercept BasePrimitive execute method to collect performance metrics
BasePrimitive.prototype.execute = async function (args = {}) {
  if (!enableTiming) {
    return await originalExecute.call(this, args);
  }

  args.enableCPUTiming = true;
  const hasTimestampQuery = this.device.features.has("timestamp-query");
  args.enableGPUTiming = hasTimestampQuery;

  let minGpuTimeNS = Infinity;
  let minCpuTimeNS = Infinity;
  let res;
  let inputLength = this.inputLength;

  for (let t = 0; t < 3; t++) {
    res = await originalExecute.call(this, args);

    if (inputLength === undefined || inputLength === null) {
      if (this.hasBuffer("inputBuffer")) {
        inputLength = this.getBuffer("inputBuffer").length;
      } else if (this.hasBuffer("keysInOut")) {
        inputLength = this.getBuffer("keysInOut").length;
      }
    }

    const { gpuTotalTimeNS, cpuTotalTimeNS } = await this.getTimingResult();
    
    let gpuTimeNS = gpuTotalTimeNS;
    if (Array.isArray(gpuTimeNS)) {
      gpuTimeNS = gpuTimeNS.reduce((a, b) => a + b, 0);
    }
    
    if (gpuTimeNS < minGpuTimeNS) {
      minGpuTimeNS = gpuTimeNS;
    }
    if (cpuTotalTimeNS < minCpuTimeNS) {
      minCpuTimeNS = cpuTotalTimeNS;
    }

    // Break early if it's a small test (only run 1 trial)
    if (inputLength === undefined || inputLength === null || inputLength < 1000000) {
      break;
    }
  }

  perfResults.push({
    label: this.label,
    type: this.type,
    datatype: this.datatype,
    inputLength,
    gpuTimeMs: minGpuTimeNS / 1e6,
    cpuTimeMs: minCpuTimeNS / 1e6,
  });

  return res;
};

// Intercept console.log to track PASS/FAIL counts
const testStatus = { passed: 0, failed: 0 };
const originalConsoleLog = console.log;
console.log = function (...msg) {
  if (enableTiming) {
    originalConsoleLog.apply(console, msg);
    const msgStr = msg.join(" ");
    if (msgStr.startsWith("[PASS]")) {
      testStatus.passed++;
    } else if (msgStr.startsWith("[FAIL]")) {
      testStatus.failed++;
    }
  }
};

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// 5a. Warmup run (JIT compiles everything silently)
console.log("Warming up shaders (JIT compilation phase)...");
enableTiming = false;
await import(`./regression.mjs?warmup=true`);
await mainFinished;

// Reset mainFinished promise and log interceptors for timed run
mainFinished = new Promise((resolve) => {
  resolveMain = resolve;
});
enableTiming = true;

// 5b. Import and run the original regression tests with timing
console.log("Starting Timed Regression Suite...");
await import(`./regression.mjs?timed=true`);
await mainFinished;

// Restore console.log
console.log = originalConsoleLog;

// 6. Compare with baseline performance file
console.log("\n================ PERFORMANCE COMPARISON ================");
console.log(`GPU Device: ${adapterDescription}`);

const baselinePath = path.join(__dirname, "perf_baseline.json");
let baselineData = {};
if (fs.existsSync(baselinePath)) {
  try {
    baselineData = JSON.parse(fs.readFileSync(baselinePath, "utf8"));
  } catch (e) {
    console.warn("Could not read baseline performance file:", e.message);
  }
}

const gpuBaseline = baselineData[adapterDescription] || {};
const newGpuBaseline = {};
const updateBaseline = process.env.UPDATE_BASELINE === "true";
let regressionDetected = false;

for (const res of perfResults) {
  const key = `${res.label}_${res.type || "default"}_${res.datatype}_size_${res.inputLength}`;
  newGpuBaseline[key] = { gpuTimeMs: res.gpuTimeMs, cpuTimeMs: res.cpuTimeMs };

  const base = gpuBaseline[key];
  if (base) {
    const baseGpu = typeof base.gpuTimeMs === "number" ? base.gpuTimeMs : 0;
    const baseCpu = typeof base.cpuTimeMs === "number" ? base.cpuTimeMs : 0;

    const slowdownGpu = baseGpu > 0 ? (res.gpuTimeMs - baseGpu) / baseGpu : 0;
    const slowdownCpu = baseCpu > 0 ? (res.cpuTimeMs - baseCpu) / baseCpu : 0;

    const thresholdGpu = 0.80; // 80% GPU slowdown threshold
    const thresholdCpu = 1.00; // 100% CPU slowdown threshold
    
    // Noise floor: ignore regressions where absolute slowdown is under 3.0ms (driver/scheduling noise)
    const absoluteNoiseFloorMs = 3.0;
    
    const isGpuRegressed = res.gpuTimeMs > 0 && baseGpu > 0 && slowdownGpu > thresholdGpu && (res.gpuTimeMs - baseGpu) > absoluteNoiseFloorMs;
    const isCpuRegressed = baseCpu > 0 && slowdownCpu > thresholdCpu && (res.cpuTimeMs - baseCpu) > absoluteNoiseFloorMs;

    if (isGpuRegressed || isCpuRegressed) {
      if (res.inputLength >= 1000000) {
        if (res.gpuTimeMs > 0 && baseGpu > 0) {
          if (isGpuRegressed) {
            regressionDetected = true;
          }
        } else if (isCpuRegressed) {
          regressionDetected = true;
        }
      }
      console.warn(`[REGRESSION] ${key}:`);
      if (isGpuRegressed) {
        console.warn(`  GPU: ${res.gpuTimeMs.toFixed(3)} ms (vs baseline ${baseGpu.toFixed(3)} ms, +${(slowdownGpu * 100).toFixed(1)}%)`);
      }
      if (isCpuRegressed) {
        console.warn(`  CPU: ${res.cpuTimeMs.toFixed(3)} ms (vs baseline ${baseCpu.toFixed(3)} ms, +${(slowdownCpu * 100).toFixed(1)}%)`);
      }
    } else {
      const compareStr = baseGpu > 0 ? `(vs baseline ${baseGpu.toFixed(3)} ms)` : "(no valid baseline)";
      console.log(`[OK] ${key}: GPU ${res.gpuTimeMs.toFixed(3)} ms ${compareStr}`);
    }
  } else {
    console.log(`[NEW] ${key}: GPU ${res.gpuTimeMs.toFixed(3)} ms`);
  }
}

if (updateBaseline) {
  baselineData[adapterDescription] = { ...gpuBaseline, ...newGpuBaseline };
  fs.writeFileSync(baselinePath, JSON.stringify(baselineData, null, 2), "utf8");
  console.log(`\nBaseline performance data for "${adapterDescription}" written to: ${baselinePath}`);
}

console.log("\n================ REGRESSION SUMMARY ================");
console.log(`Correctness: ${testStatus.passed} passed, ${testStatus.failed} failed`);

if (testStatus.failed > 0 || testStatus.passed === 0) {
  console.error(`FAIL: Functional correctness failures detected! (${testStatus.passed} passed, ${testStatus.failed} failed)`);
  process.exit(1);
}
const failOnRegression = process.env.FAIL_ON_REGRESSION === "true";
if (regressionDetected && !updateBaseline && failOnRegression) {
  console.error("FAIL: Performance regression detected!");
  process.exit(2);
}
console.log("All correctness and performance checks passed!");
process.exit(0);
