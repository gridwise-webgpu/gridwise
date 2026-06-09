import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

async function main() {
  console.log("Starting local HTTP server...");
  // Start simple_cors_server.py in the project root
  const server = spawn('python3', ['simple_cors_server.py'], {
    cwd: projectRoot,
    stdio: 'inherit'
  });

  // Wait 2 seconds for server to start
  await new Promise(resolve => setTimeout(resolve, 2000));

  let browser;
  try {
    console.log("Launching headless browser with WebGPU support...");
    browser = await puppeteer.launch({
      headless: 'new',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--enable-unsafe-webgpu',
        '--use-webgpu-adapter=swiftshader'
      ]
    });

    const page = await browser.newPage();
    
    // Redirect browser console logs to node console
    page.on('console', msg => console.log(`[Browser Console] ${msg.text()}`));

    console.log("Navigating to regression tests page...");
    await page.goto('http://localhost:8000/examples/regression.html', {
      waitUntil: 'networkidle0',
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
    // Terminate server process
    server.kill();
  }
}

main();
