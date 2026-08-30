import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const testDir = path.join(projectRoot, "scratch", "test-tarball");

async function main() {
  console.log("=== Testing NPM Package Tarball & Export Integrity ===");

  // 1. Pack tarball
  console.log("1. Running 'npm pack'...");
  const packOutput = execSync("npm pack", { cwd: projectRoot, encoding: "utf8" }).trim();
  const tarballName = packOutput.split("\n").pop().trim();
  const tarballPath = path.join(projectRoot, tarballName);

  if (!fs.existsSync(tarballPath)) {
    console.error(`Error: Tarball not created at ${tarballPath}`);
    process.exit(1);
  }
  console.log(`   Generated: ${tarballName}`);

  // 2. Setup temporary test app directory
  if (fs.existsSync(testDir)) {
    fs.rmSync(testDir, { recursive: true, force: true });
  }
  fs.mkdirSync(testDir, { recursive: true });

  try {
    // 3. Extract tarball in test app directory
    console.log("2. Extracting tarball in isolated directory...");
    execSync(`tar -xzf "${tarballPath}" -C "${testDir}"`, { stdio: "inherit" });

    // 4. Import index.mjs from extracted package
    const packageIndexPath = path.join(testDir, "package", "index.mjs");
    console.log(`3. Importing entrypoint: ${packageIndexPath}...`);

    const pkg = await import(`file://${packageIndexPath}`);

    // 5. Verify exported symbols
    const expectedExports = [
      "BasePrimitive",
      "Kernel",
      "AllocateBuffer",
      "WriteGPUBuffer",
      "Buffer",
      "DLDFScan",
      "OneSweepSort",
      "BinOp",
      "BinOpAdd",
      "BinOpMin",
      "BinOpMax",
      "BinOpMultiply",
      "makeBinOp",
    ];

    let missingCount = 0;
    for (const exp of expectedExports) {
      if (pkg[exp] === undefined) {
        console.error(`   [FAIL] Export missing: ${exp}`);
        missingCount++;
      } else {
        console.log(`   [PASS] Export present: ${exp} (${typeof pkg[exp]})`);
      }
    }

    if (missingCount > 0) {
      console.error(`\nFAILED: ${missingCount} expected exports were missing!`);
      process.exit(1);
    }

    console.log("\n=============================================");
    console.log("SUCCESS: Package tarball and ES Module exports verified!");
    console.log("=============================================\n");
  } catch (err) {
    console.error("An error occurred during package verification:", err);
    process.exit(1);
  } finally {
    // Cleanup
    if (fs.existsSync(tarballPath)) {
      fs.unlinkSync(tarballPath);
    }
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
  }
}

main();
