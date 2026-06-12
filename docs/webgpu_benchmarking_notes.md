# Best Practices for Efficient WebGPU-Based Demo Benchmarking

When benchmarking graphical WebGPU applications within an agentic environment, a highly structured approach is required to prevent caching issues, capture runtime errors, and avoid long-running browser subagent exploration loops. Below are the key techniques that made the diagnosis and measurements efficient.

---

## 1. Embed an Automated Benchmark Runner in the Demo
Do not rely on the browser subagent to click buttons, drag sliders, or manually time transitions. Instead, inject a self-contained, automated benchmark script into the JavaScript execution path:
*   Use a `setTimeout` to trigger the benchmark after page load.
*   Use a sequence of `await sleep(...)` blocks to transition between states (baseline measurement, triggering actions, changing sizes).
*   Log structured progress strings (e.g., `=== STARTING AUTOMATED BENCHMARK ===`, `--- 10K Baseline Complete ---`).
*   Log numerical metrics periodically (e.g., calculated FPS and active particle counts) to console output.

---

## 2. Force Cache Busting for ES Modules
Browsers aggressively cache ES Modules (`.mjs` files), and standard HTTP server configurations may not successfully force a refresh:
*   **HTML script tags**: Add version query parameters to the import source:
    ```html
    <script type="module" src="interactive_demo_optimal.mjs?v=10"></script>
    ```
*   **Direct URL navigation**: Instruct the browser subagent to load the page with matching query parameters:
    ```
    http://localhost:8000/demos/interactive_demo.html?v=10
    ```
This guarantees that changes to shader WGSL or setup parameters on the filesystem are immediately parsed and executed by Chrome.

---

## 3. Wrap GPU Operations in try-catch Blocks
WebGPU functions (shader compilation, adapter requests, resource binding) are asynchronous and throw unhandled promise rejections or device lost errors on failure. 
*   Because browser automation log listeners can miss uncaught JS runtime exceptions:
    ```javascript
    try {
      // Main WebGPU setup and pipeline creation
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice();
      ...
    } catch (error) {
      console.error("CRITICAL RUNTIME ERROR:", error);
      console.error(error.stack);
    }
    ```
*   Explicitly catching and logging these errors to the console using `console.error` guarantees that the subagent's console log listener captures the exact Tint or Dawn compiler message.

---

## 4. Clear Console History on Load
Chrome preserves console history across reloads. If a script fails to compile on run $N$, the log capture on run $N+1$ may still contain the stale compile error:
*   Insert `console.clear()` right after imports at the top of your script.
*   *Warning*: Never put `console.clear()` *before* imports, as ES Module syntax mandates that import statements must be the first expressions in the file, otherwise a `SyntaxError` will occur.

---

## 5. Keep the Browser Subagent Focused
Do not let the browser subagent try to debug code, inspect source files, or browse local directories using `view-source:` or navigation screenshots. This is slow and prone to timeouts:
*   Use local codebase search tools (`grep_search`, `view_file`) on the parent agent to inspect files and find issues.
*   Instruct the browser subagent to perform **only** a single sequence:
    1. Open URL.
    2. Wait for the predetermined benchmark duration.
    3. Capture console logs.
    4. Report logs and exit.
