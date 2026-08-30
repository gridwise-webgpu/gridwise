---
layout: page
title: npm Publishing Instructions
permalink: /npm-instructions/
---

# Gridwise npm Publishing Guide

This guide provides step-by-step instructions for publishing `gridwise` to npmjs.com — both for the **first-time release** and for **subsequent version updates**.

---

## 🚀 Part 1: First-Time Publishing Instructions

Follow these steps if you have never published an npm package before or are setting up your npm author account for the first time.

### Step 1.1: Create & Verify your npm Account
1. Go to [npmjs.com/signup](https://www.npmjs.com/signup) and create an account.
2. **Verify your email address** (npm requires email verification before allowing any publishes).
3. **Enable Two-Factor Authentication (2FA)** under Account Settings -> Two-Factor Authentication.

### Step 1.2: Log in via the Command Line
In your terminal, navigate to the `gridwise` project root and run:
```bash
npm login
```
- It will open a browser authentication page (or prompt for your username, password, and 2FA OTP code).
- Confirm you are logged in by running:
  ```bash
  npm whoami
  ```
  *(This should print your npm username).*

### Step 1.3: Run Pre-Flight Verification Checks
Before publishing, run local verification scripts to ensure no exported symbols or package files are broken:
```bash
# 1. Run Node.js correctness regression suite
npm run test:node

# 2. Run WGSL function unit tests
npm run test:wgsl

# 3. Test building and extracting the npm package tarball
npm run test:package
```

### Step 1.4: Dry-Run Preview
Preview the exact file list and tarball size that will be uploaded to npmjs.com:
```bash
npm publish --dry-run
```
- Verify that `src/`, `index.mjs`, `index.d.ts`, `LICENSE.txt`, and `README.md` are present.

### Step 1.5: Perform the First Release
Publish the initial release (e.g. `v0.1.0`):
```bash
npm publish
```
*(Note: `package.json` contains `"prepublishOnly": "npm run test:package"`, so `npm publish` automatically verifies package exports right before uploading).*

---

## 🔄 Part 2: Subsequent Release Instructions

Follow these steps whenever you add new features, fix bugs, or prepare a new release of `gridwise`.

### Step 2.1: Ensure `main` is Clean and Tested
1. Make sure all local changes are committed and pushed to `main`.
2. Ensure GitHub Actions CI build is green (**`PASSING`**).
3. Run tests locally:
   ```bash
   npm run test:node
   npm run test:package
   ```

### Step 2.2: Bump the Version Number (Semantic Versioning)
npm packages follow [Semantic Versioning (SemVer)](https://semver.org/): `MAJOR.MINOR.PATCH` (e.g., `0.1.0`).

Choose one of the following commands to bump the version:
```bash
# For bug fixes / patch updates (0.1.0 -> 0.1.1)
npm version patch

# For new features / non-breaking API additions (0.1.0 -> 0.2.0)
npm version minor

# For breaking API changes (0.1.0 -> 1.0.0)
npm version major
```
`npm version` will automatically:
- Update `"version"` in `package.json` and `package-lock.json`.
- Create a Git commit (e.g. `v0.1.1`).
- Create a Git tag (e.g. `v0.1.1`).

### Step 2.3: Push Version Commits & Tags to GitHub
Push the newly generated version commit and Git tag to `origin/main`:
```bash
git push origin main --tags
```

### Step 2.4: Publish to npm
Publish the new release:
```bash
npm publish
```

---

## 🛠️ Quick Reference Checklist

| Task | Command |
| :--- | :--- |
| **Check npm Login Status** | `npm whoami` |
| **Log into npm CLI** | `npm login` |
| **Test Local Package Tarball** | `npm run test:package` |
| **Preview Package (Dry Run)** | `npm publish --dry-run` |
| **Bump Patch Version** | `npm version patch` |
| **Bump Minor Version** | `npm version minor` |
| **Publish to npm** | `npm publish` |
