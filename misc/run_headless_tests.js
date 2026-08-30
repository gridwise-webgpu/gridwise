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

async function main() {
  console.log("Starting local HTTP server...");
  const server = await startServer(projectRoot, 8000);

  let browser;
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

    if (failedTests.length > 0) {
      console.error("Failed tests:");
      failedTests.forEach(t => console.error(`- ${t}`));
      process.exit(1);
    } else {
      console.log("All tests passed successfully!");
      process.exit(0);
    }

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
