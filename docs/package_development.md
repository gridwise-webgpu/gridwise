---
layout: page
title: Package Development & Publishing Guide
permalink: /package-development/
---

# Developer Guide: Gridwise npm Package & CI Infrastructure

This guide explains how `gridwise` is structured as an npm package, how continuous integration (CI) tests run, useful scripts for package development, and step-by-step procedures for adding or modifying code in the repository.

---

## 1. Package Architecture Overview

Gridwise is published as a modern **ES Module (ESM)** npm package.

### File & Directory Layout
```text
gridwise/
├── src/                  # Core library source code (scan, sort, reduce, WGSL shaders, buffers)
├── index.mjs             # Main package ES Module entry point (re-exports public primitives)
├── index.d.ts            # TypeScript type definitions
├── package.json          # Package manifest, metadata, scripts, and export mappings
├── package-lock.json     # Deterministic dependency lockfile
├── README.md             # Package documentation displayed on npmjs.com
├── LICENSE.txt           # Apache-2.0 license
├── .github/workflows/    # GitHub Actions CI workflow definitions
├── examples/             # Code examples and regression test scripts
└── misc/                 # Utility scripts (headless runner, package verification)
```

### Key `package.json` Fields
- `"type": "module"`: Instructs Node.js and bundlers to treat all `.mjs` and `.js` files as native ES Modules.
- `"main": "./index.mjs"` & `"types": "./index.d.ts"`: Specifies main JS and TypeScript entry points.
- `"exports"`: Declares modern conditional exports:
  ```json
  "exports": {
    ".": {
      "types": "./index.d.ts",
      "import": "./index.mjs"
    }
  }
  ```
- `"files": ["src/", "index.mjs", "index.d.ts", "LICENSE.txt", "README.md"]`: Defines the whitelist of directories and files included in the published npm package tarball.
- `"sideEffects": false`: Enables tree-shaking for web bundlers (Vite, Rollup, Webpack).

---

## 2. Helpful Development & Verification Scripts

The repository includes several npm scripts and helper tools for local development and testing:

| Script Command | What It Does | When To Use |
| :--- | :--- | :--- |
| `npm run test:wgsl` | Runs WGSL helper function tests in both **hardware** and **emulated** subgroup modes. | Use when modifying WGSL shader logic, subgroup helpers, or WGSL functions in `src/wgslFunctions.mjs`. |
| `npm run test:node` | Runs the full 54-test correctness regression suite in Node.js via `webgpu` (Dawn). | Use as your primary test command when modifying primitive algorithms (`DLDFScan`, `OneSweepSort`, etc.). |
| `npm run test:node:update-baseline` | Updates local GPU performance baselines in `examples/perf_baseline.json`. | Use when making intentional performance optimizations or after benchmarking on a new GPU. |
| `npm run test:package` | Builds a package tarball (`npm pack`), extracts it in an isolated temporary directory, and verifies ES Module exports. | Use before pushing or publishing to ensure no exported symbols or package files are broken. |
| `node misc/run_headless_tests.js` | Launches a headless Chrome browser (Puppeteer) to execute browser-based regression tests. | Use to verify WebGPU execution inside a real browser environment. |
| `npm pack --dry-run` | Previews the exact list of files, unpacked size, and structure of the npm tarball without saving to disk. | Use whenever modifying `package.json` `"files"` or adding new top-level package files. |

---

## 3. Automated CI / Pull Request Checks

Whenever code is pushed to `main` or a pull request is opened, GitHub Actions automatically executes the workflow defined in [`.github/workflows/test.yml`](file:///.github/workflows/test.yml).

### CI Environment Details
- **Runner**: `macos-latest` (Apple Silicon runner with Metal GPU acceleration).
- **Execution Steps**:
  1. `actions/checkout@v4` — Checks out the repo code.
  2. `actions/setup-node@v4` — Sets up Node.js 22 environment.
  3. `npm ci` — Installs exact dependencies from `package-lock.json`.
  4. `node misc/run_headless_tests.js` — Executes browser regression tests.
  5. `npm run test:node` — Executes Node WebGPU correctness and timing checks.
  6. `npm run test:wgsl` — Executes WGSL function tests.

---

## 4. Developer Workflows & Guidelines

### Scenario A: Adding a New Source File
1. Place the new `.mjs` file inside the `src/` directory (e.g., `src/my_new_primitive.mjs`).
2. **Package inclusion**: Because `package.json` `"files"` specifies `"src/"`, any file inside `src/` is **automatically included** in the npm package when published. You do not need to update `package.json`.
3. **Public API Exports**: If the new file contains public classes or functions:
   - Export them in `index.mjs` (e.g., `export { MyPrimitive } from "./src/my_new_primitive.mjs";`).
   - Add TypeScript type definitions to `index.d.ts`.
   - Add the export name to the `expectedExports` list in `misc/test_package.mjs`.

### Scenario B: Adding or Updating Dependencies
1. To add a runtime or development dependency:
   ```bash
   npm install <package-name>           # Runtime dependency
   npm install --save-dev <package-name> # Development dependency
   ```
2. **Lockfile**: Always commit both [`package.json`](file:///package.json) AND [`package-lock.json`](file:///package-lock.json). This guarantees repeatable builds for CI and other contributors.

### Scenario C: Modifying Existing Code / Primitives
1. Make your code changes in `src/` or `examples/`.
2. Run local tests:
   ```bash
   npm run test:wgsl
   npm run test:node
   npm run test:package
   ```
3. Commit and push your changes.

### Scenario D: Publishing a New Release to npmjs.com
1. **Verify package integrity**:
   ```bash
   npm run test:package
   ```
   *(Note: `"prepublishOnly": "npm run test:package"` is configured in `package.json`, so `npm publish` will automatically run this test and abort if anything fails).*
2. **Bump version number**:
   ```bash
   npm version patch   # For bug fixes (0.1.0 -> 0.1.1)
   npm version minor   # For new features (0.1.0 -> 0.2.0)
   npm version major   # For breaking changes (0.1.0 -> 1.0.0)
   ```
3. **Publish to npm**:
   ```bash
   npm publish
   ```
