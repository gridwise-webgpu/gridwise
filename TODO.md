# Prioritized Gridwise Todo List

Based on the feedback sent by Lee Mighdoll, we have analyzed the recent additions to the codebase and compared them to the original list of comments. A significant number of high-impact features (such as CI workflows, TypeScript typings, and documentation restructure) have already been successfully addressed. 

Below is the prioritized list of remaining tasks, categorized by urgency, with details on recent accomplishments and the exact files to modify next.

---

## 🚨 Priority 1: Critical Bug Fixes & Compatibility

*No remaining tasks.*

---

## 📦 Priority 2: NPM Publishing Preparation

*No remaining tasks.*

---

## 🌐 Priority 3: Landing Page & UX Improvements

### 4. Upgrade Intro Page with Performance Claims & Snippet
* **Feedback**: "consider an eye catching claim about performance... worlds fastest implementation, based on just-published state of the art research... consider putting a code snippet on the home page."
* **Action Item**: Update `docs/index.md` to:
  * Highlight that Gridwise is the fastest WebGPU primitive implementation, based on decoupled fallback/lookback state-of-the-art research.
  * Add a code snippet demonstrating how simple it is to import and run a sort/scan operation (e.g., `await OneSweepSort({ ... }).execute();`).
* **Files to modify**:
  * [docs/index.md](file:///Users/jdowens/Documents/working/gridwise/docs/index.md)

### 5. Setup Redirection for `https://gridwise-webgpu.github.io/` 404
* **Feedback**: "https://gridwise-webgpu.github.io/ is a 404, should be https://gridwise-webgpu.github.io/gridwise/"
* **Action Item**: Setup a landing/redirect page at the root level of the GitHub Pages user/org page. This is usually done by creating a repository named `gridwise-webgpu.github.io` with an `index.html` that performs a redirect to `/gridwise/`.

---

## 🏎️ Priority 4: Interactive Demo Polish

### 6. Limit Particle Count Range in Galaxy Stars Demo
* **Feedback**: "on my m1max looks great at 10K but terribly slow at 100K particles. People are accustomed to gpu demos where particles run at high frame rates.. maybe limit the range?"
* **Action Item**:
  * Limit the maximum slider range in `demos/interactive_demo.html` from `100000` to a more reasonable ceiling (e.g., `30000` or `50000` particles) to ensure frame rates remain high even on varied devices.
* **Files to modify**:
  * [demos/interactive_demo.html](file:///Users/jdowens/Documents/working/gridwise/demos/interactive_demo.html#L198)

---

## 📝 Priority 5: Documentation Clarifications & Terminology

### 7. Clarify the Machine Lockup Phrasing & WebGPU safety net
* **Feedback**: "'this locks up the entire machine' sounds pretty scary! perhaps an opportunity to mention webgpu provides safety net."
* **Action Item**: Rephrase the sentence in `docs/scan-and-reduce.md` to soften the impact and explain how WebGPU's safety mechanisms (like device loss, robust page scheduling, or execution timeouts) protect the system from infinite stalls.
* **Files to modify**:
  * [docs/scan-and-reduce.md](file:///Users/jdowens/Documents/working/gridwise/docs/scan-and-reduce.md#L34)

### 8. Mark Gridwise's Choices in "Design Choice" Sections
* **Feedback**: "The 'design choice' sections - unclear which one gridwise has chosen. maybe put a green checkmark on the gridwise approach."
* **Action Item**: Edit `docs/primitive-design.md` to clearly indicate (e.g., via checkmarks ✅) the choices that Gridwise implements (e.g., "Use subgroups + emulation" and "Always use chained algorithms").
* **Files to modify**:
  * [docs/primitive-design.md](file:///Users/jdowens/Documents/working/gridwise/docs/primitive-design.md#L37)

### 9. Replace "emu" Abbreviation with "Emulated" or "Emulation"
* **Feedback**: "'emu' sounds like a bird... write out emulated?"
* **Action Item**: Conduct a thorough replace in the documentation to spell out "emulated" or "emulation" instead of the shorthand "emu".
* **Files to modify**:
  * [docs/subgroup-strategy.md](file:///Users/jdowens/Documents/working/gridwise/docs/subgroup-strategy.md)

### 10. Polish the BinOp Example Description and Order
* **Feedback**: "in the binop example, : `{ datatype = "..." }` is a little confusing... in binop example, consider putting the gpu stuff first, and the cpu stuff at the end with a comment that it's the cpu stuff."
* **Action Item**: 
  * Ensure the placeholder text in the examples is easy to read (using `{ datatype: "f32" }` rather than the confusing `{ datatype = "..." }` syntax).
  * Ensure `docs/binop.md` displays the GPU WGSL code generation first, and lists the CPU JavaScript validation method at the end of the constructor with clear comments.
* **Files to modify**:
  * [docs/binop.md](file:///Users/jdowens/Documents/working/gridwise/docs/binop.md)

---

## 🚀 Priority 6: Future Research & Customization

### 11. Add GPU vs. CPU Performance Comparison Chart on Front Page
* **Feedback**: "maybe even a small chart on the front page showing off sort on the gpu vs. cpu?"
* **Action Item**: Once stable baselines are recorded, generate a comparison plot and embed it in `docs/index.md` or a dedicated performance summary.

### 12. Create Educational Subpage
* **Feedback**: "consider noting briefly why gpu programmers will come to need scan, reduce and sort. Many casual viewers will see the site and not know yet."
* **Action Item**: Add an educational guide page linked from the home page.

---

## ✅ Completed Tasks (Recent Milestones)

Here is a record of the items from Lee's feedback that have already been resolved:

* **Split performance examples from user examples**: Completed via issue `#34` (commit `f841fa4`), creating separate Examples and Performance pages.
* **Back button on examples**: Completed via issue `#33` (commits `389b9e0`, `3825097`), adding standardized styled back buttons (`← Back to Examples` / `← Back to Performance`).
* **Restructure documentation index**: Completed via issue `#37` (commit `fb98ee8`), splitting documentation cleanly into Usage and Architecture sections.
* **Automated regression/PR verification (CI)**: Completed via issue `#39` (commit `c1b6442`), setting up GitHub Actions workflows and a headless Node-based runner.
* **TypeScript Types & Exports**: Completed via issue `#38` (commit `e882e11`), packaging index types and exports.
* **Robust WebGPU Initialization & Crash Prevention on Non-Supported Browsers**: Completed. Added validations for `navigator.gpu` and `adapter` to avoid script crashes on unsupported browsers (e.g. Firefox without WebGPU enabled) and render visible error messages in the DOM.
* **Verify NPM Package Configuration & Review `package.json`**: Completed. Verified `package.json` fields (main, types, exports, files array) to confirm npm compliance and verified that all required library source files are packaged correctly.
* **Create a Local Test App to Verify the Packed NPM Package**: Completed. Ran `npm pack` to generate the local package tarball and created a local verification app in [scratch/test-app/](file:///Users/jdowens/Documents/working/gridwise/scratch/test-app) to confirm proper resolution of types and import endpoints.
