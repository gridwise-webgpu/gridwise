# Implementation Plan: Node.js WebGPU Regression Test Suite

This plan outlines how to enable running the existing correctness regression test suite (`examples/regression.mjs`) in Node.js using `dawn.node` without modifying any existing source code, and how to extend it to automatically detect performance regressions.

## Objectives
1. **Zero Source Code Changes**: Keep all original source code untouched (including `examples/regression.mjs`).
2. **Correctness Validation**: Run the full suite of small and large test cases inside Node.js using `dawn.node`.
3. **Performance Regression Detection**: Intercept execution to measure execution times (GPU and CPU) and compare them against a performance baseline file.

---

## Running the Regression Suite

The following npm commands are available in [package.json](../package.json) to execute correctness and performance regressions:

### 1. Verification Run (Advisory Performance Checks)
To execute the suite locally, checking for correctness errors and outputting performance warning comparisons:
```bash
npm run test:node
```
* **Exit Code 0**: If all correctness tests pass.
* **Timing Logs**: Prints timing comparisons (e.g., `[OK]` or `[REGRESSION]`) against the saved baseline for information.

### 2. Updating / Generating Performance Baselines
To save your machine's current timings as the new performance baseline (or to initialize the baseline file on a new device):
```bash
npm run test:node:update-baseline
```
* **Output File**: Saves timing profiles directly to [examples/perf_baseline.json](../examples/perf_baseline.json) grouped by your GPU model key (e.g., `"apple-m4 (metal-3)"`).

### 3. Strict Verification Run (Gated Performance CI Checks)
To run timing comparisons where a significant slowdown in large tests ($\ge 1\text{M}$ elements) will fail the validation and block a PR/build:
```bash
FAIL_ON_REGRESSION=true npm run test:node
```
* **Exit Code 2**: If any large test experiences a performance regression exceeding the threshold limits.

---

## Technical Approach

Node.js standard imports cache modules. By importing the class definitions (such as `BasePrimitive` from `primitive.mjs`) in a wrapper runner script *before* loading `examples/regression.mjs`, we can modify the class prototypes. The wrapper script:
1. Polyfills the browser-specific variables expected by the test suite (like `globalThis.navigator` and `globalThis.document`).
2. Wraps the `BasePrimitive.prototype.execute` method to force timing profiling on every test case.
3. Dynamically imports and runs `examples/regression.mjs`.
4. Captures execution metrics and compares them to a local JSON baseline.

---

## Implementation Details

The implementation is located at:
* **Node Runner Script**: [examples/regression_node.mjs](../examples/regression_node.mjs)

---

## GPGPU Performance Regression Philosophy

Evaluating microsecond-scale performance regressions on desktop/consumer GPUs (especially Apple Silicon M-series unified architectures) requires addressing unique hardware/driver constraints. The runner script implements the following strategies to prevent flakiness and false alarms:

### 1. JIT Compilation Warmup
WebGPU compute shaders are compiled to hardware-specific machine binaries at runtime. The very first execution of any pipeline will trigger driver JIT compiling and command buffer validation overhead, which can be thousands of times slower than subsequent runs.
* **Warmup Phase**: The script runs a silent compilation pass of the test suite before start.
* **Device Caching**: The WebGPU device context is cached and shared between the warmup run and the timed run. This preserves both the browser-level and driver-level pipeline caches, ensuring subsequent executions measure only execution performance.

### 2. Microsecond-Scale Driver Quantization & Noise Floor
Hardware timestamp queries under Metal/WebGPU resolve timings inside discrete clock-ticking intervals. On Apple Silicon, this timing granularity is typically quantized to **$65.536\text{ }\mu\text{s}$ ($0.065536\text{ ms}$)**.
* Small fluctuations can cause measurements to snap into neighboring quantization bins, falsely reporting a $100\%$ or $200\%$ slowdown on sub-millisecond tests.
* **Noise Floor Limits**: The test suite ignores regressions where the absolute GPU timing slowdown is under **$3.0\text{ ms}$**, ignoring sub-millisecond noise.

### 3. Multi-Trial Min Filter
GPGPU execution times can spike due to system thread scheduling delays, unified memory bus contention from other active software, and thermal throttling states.
* **3-Trial Minimums**: For large tests ($\ge 1\text{M}$ elements), the runner runs each benchmark 3 times and records the **minimum** duration. This filters out transient CPU thread submission lag and dynamic clock frequency adjustment delay.
* **Single-Trial for Small Tests**: Small tests run exactly 1 trial to keep the suite run-time under a few seconds.

### 4. Large-Scale Gating only
Dynamic memory bandwidth sharing and CPU-to-GPU command submission queue overhead make performance metrics highly volatile on small data workloads.
* Performance validation failures (exit code `2`) are strictly gated on **large tests ($\ge 1\text{M}$ elements)**.
* Small workload performance changes are logged as warnings (`[REGRESSION]`) for visibility but do not break builds or CI loops.

### 5. Advisory default mode
Because local developer machines are subject to variable workloads (browsers, IDEs, compiler daemons), performance checks run in **advisory mode** by default.
* Correctness validation remains strictly blocking (fails with exit code `1`).
* Performance checks log regressions clearly but exit with code `0`.
* Enforce strict performance gating in CI environments by setting the flag:
  ```bash
  FAIL_ON_REGRESSION=true npm run test:node
  ```
