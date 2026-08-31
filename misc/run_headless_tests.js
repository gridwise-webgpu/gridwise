import puppeteer from 'puppeteer';
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

const mimeTypes = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.wasm': 'application/wasm',
};

function startServer(root, port = 8000) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const parsedUrl = new URL(req.url, `http://127.0.0.1:${port}`);
      const safePath = path.normalize(parsedUrl.pathname).replace(/^(\.\.[\/\\])+/, '');
      let filePath = path.join(root, safePath);

      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, 'index.html');
      }

      if (!fs.existsSync(filePath)) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('404 Not Found');
        return;
      }

      const ext = path.extname(filePath).toLowerCase();
      const contentType = mimeTypes[ext] || 'application/octet-stream';

      res.writeHead(200, {
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Cache-Control': 'no-store, no-cache, must-revalidate',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });

      fs.createReadStream(filePath).pipe(res);
    });

    server.listen(port, '127.0.0.1', () => {
      console.log(`Server listening on http://127.0.0.1:${port}`);
      resolve(server);
    });

    server.on('error', (err) => reject(err));
  });
}

async function runRegressionTests(browser) {
  const page = await browser.newPage();

  // Redirect browser console logs to node console
  page.on('console', msg => console.log(`[Browser Console] ${msg.text()}`));

  console.log("Navigating to regression tests page...");
  await page.goto('http://127.0.0.1:8000/examples/regression.html', {
    waitUntil: 'domcontentloaded',
    timeout: 30000
  });

  console.log("Waiting for tests to complete...");
  // Wait for #summary to change from "Running…"
  await page.waitForFunction(() => {
    const summary = document.getElementById('summary');
    return summary && summary.textContent !== 'Running…';
  }, { timeout: 60000 });

  const summaryText = await page.evaluate(() => {
    return document.getElementById('summary').textContent;
  });

  console.log("\n================ TEST SUMMARY ================");
  console.log(summaryText);
  console.log("==============================================\n");

  const failedTests = await page.evaluate(() => {
    const fails = Array.from(document.querySelectorAll('#test-list li.fail'));
    return fails.map(el => el.textContent);
  });

  await page.close();
  return failedTests;
}

/*
 * Smoke test for demos/interactive_demo.html.
 *
 * The demo is not covered by the regression suite and is not shipped in
 * the npm package, so it can rot silently. This is deliberately shallow:
 * it does not check that the simulation is correct, only that the page
 * comes up, every shader compiles, and each of the three primitive
 * buttons runs without raising an error.
 *
 * That is enough to catch the class of breakage that is easy to cause and
 * easy to miss: the demo declares its Particle and Params structs once
 * and interpolates them into seven shaders, so a field renamed or a
 * uniform resized in one place and not another is a compile error at
 * page load. Clicking Sort and Scan matters specifically because those
 * are the shaders that read Params.layoutInset - a field that no other
 * code path touches.
 */
async function runDemoSmokeTest(browser) {
  const page = await browser.newPage();
  const failures = [];

  page.on('pageerror', err => failures.push(`uncaught exception: ${err.message}`));
  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.text().includes('favicon')) {
      failures.push(`console error: ${msg.text()}`);
    }
  });

  console.log("Navigating to interactive demo...");
  await page.goto('http://127.0.0.1:8000/demos/interactive_demo.html', {
    waitUntil: 'domcontentloaded',
    timeout: 30000
  });

  // Give the demo time to request a device, compile all seven shaders and
  // build its buffers before looking at anything.
  await new Promise(r => setTimeout(r, 5000));

  const hasWebGPU = await page.evaluate(() => !!navigator.gpu);
  if (!hasWebGPU) {
    console.log("WebGPU unavailable in this browser; skipping demo smoke test.");
    await page.close();
    return [];
  }

  // A non-zero backing store means resizeCanvas ran and the context was
  // configured; zero means init bailed out somewhere.
  const canvas = await page.evaluate(() => {
    const c = document.getElementById('canvas');
    return { w: c.width, h: c.height };
  });
  if (!canvas.w || !canvas.h) {
    failures.push(`canvas has no backing store (${canvas.w}x${canvas.h})`);
  }

  /*
   * Regression guard: the canvas backing store must never exceed the
   * device's maxTextureDimension2D. Past that the swap-chain texture is
   * invalid and the demo renders nothing - a black canvas with the
   * controls still drawn on top, and no thrown error to notice.
   *
   * This is easy to reintroduce because resizeCanvas() multiplies the CSS
   * size by devicePixelRatio, so on a Retina display any window wider
   * than half the limit crosses it. A wide viewport is used here because
   * the default test window is nowhere near large enough to catch it.
   */
  const wide = await browser.newPage();
  try {
    await wide.setViewport({ width: 5000, height: 1000, deviceScaleFactor: 2 });
    await wide.goto('http://127.0.0.1:8000/demos/interactive_demo.html', {
      waitUntil: 'domcontentloaded',
      timeout: 30000
    });
    await new Promise(r => setTimeout(r, 5000));
    const big = await wide.evaluate(async () => {
      const c = document.getElementById('canvas');
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice();
      return { w: c.width, h: c.height, max: device.limits.maxTextureDimension2D };
    });
    if (big.w > big.max || big.h > big.max) {
      failures.push(
        `canvas ${big.w}x${big.h} exceeds maxTextureDimension2D ${big.max} ` +
        `at a 5000x1000 viewport; the demo will render black`
      );
    } else {
      console.log(`  wide viewport: canvas ${big.w}x${big.h} within limit ${big.max}`);
    }
  } finally {
    await wide.close();
  }

  // Every control the demo wires up must exist, or an addEventListener
  // call threw and the rest of the module never ran.
  const missing = await page.evaluate(() => {
    const ids = ['canvas', 'starSlider', 'starCount', 'sortBtn', 'scanBtn', 'reduceBtn', 'modeBtn'];
    return ids.filter(id => !document.getElementById(id));
  });
  if (missing.length) failures.push(`missing elements: ${missing.join(', ')}`);

  // Prove the module actually ran to completion before trusting anything
  // below. The event listeners are registered at the very bottom of
  // interactive_demo.mjs, so if init threw part way - a shader that failed
  // to compile, say - nothing is wired up and every check after this point
  // would pass vacuously. Toggling the mode button is the cheapest probe:
  // its label only changes if its listener exists.
  const modeLabel = await page.evaluate(() => {
    const b = document.getElementById('modeBtn');
    const before = b.textContent;
    b.click();
    return { before, after: b.textContent };
  });
  if (modeLabel.before === modeLabel.after) {
    failures.push(
      `demo did not finish initializing: mode button inert (still "${modeLabel.after}")`
    );
  }

  // Run each primitive. Sort and Scan exercise the shaders that read
  // Params.layoutInset; Reduce exercises the third pipeline.
  for (const id of ['sortBtn', 'scanBtn', 'reduceBtn']) {
    await page.evaluate(btn => document.getElementById(btn).click(), id);
    await new Promise(r => setTimeout(r, 2500));
    const err = await page.evaluate(() => {
      const el = document.getElementById('errorDisplay');
      return getComputedStyle(el).display !== 'none' ? el.textContent : null;
    });
    if (err) failures.push(`${id} raised: ${err}`);
    else console.log(`  ${id}: ok`);
  }

  await page.close();
  return failures;
}

async function main() {
  console.log("Starting local HTTP server...");
  const server = await startServer(projectRoot, 8000);

  let browser;
  let exitCode = 0;
  try {
    console.log("Launching headless browser with WebGPU support...");
    browser = await puppeteer.launch({
      headless: 'new',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--enable-unsafe-webgpu'
      ]
    });

    const failedTests = await runRegressionTests(browser);
    if (failedTests.length > 0) {
      console.error("Failed tests:");
      failedTests.forEach(t => console.error(`- ${t}`));
      exitCode = 1;
    } else {
      console.log("All regression tests passed successfully!");
    }

    console.log("\n=========== INTERACTIVE DEMO SMOKE ===========");
    const demoFailures = await runDemoSmokeTest(browser);
    if (demoFailures.length > 0) {
      console.error("Interactive demo smoke test failed:");
      demoFailures.forEach(f => console.error(`- ${f}`));
      exitCode = 1;
    } else {
      console.log("Interactive demo smoke test passed.");
    }
    console.log("==============================================\n");

    process.exit(exitCode);
  } catch (error) {
    console.error("An error occurred during test execution:", error);
    process.exit(1);
  } finally {
    if (browser) {
      await browser.close();
    }
    server.close();
  }
}

main();
