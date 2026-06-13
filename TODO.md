# Prioritized Gridwise Todo List

Below is the prioritized list of remaining tasks for the Gridwise library, including outstanding feedback items and open GitHub issues.

---

## 🚨 Priority 1: Critical Bug Fixes & Compatibility

### 1. Resolve Memory Limit Exceeded in Key-Value Sort Validation (Issue #14)
* **Problem**: The JavaScript validation function for key-value sorting throws `RangeError: Invalid array length` when sorting large arrays containing more than $2^{26}$ (~67 million) elements.
* **Action Item**: Refactor the validation helper code to avoid allocating array sizes that exceed JavaScript engine limits (e.g., utilize chunked verification or typed arrays).

### 2. Reset Atomic Buffers Between Trials (Issue #13)
* **Problem**: Helper atomic buffers (like `spineBuffer` and `scanBump`) are not cleared between consecutive execution trials. Atomics can accumulate values over multiple trials, potentially degrading performance or yielding incorrect results.
* **Action Item**: Reset/clear atomic buffers before each trial run to ensure clean starting conditions.

---

## 🌐 Priority 2: Landing Page & UX Improvements

### 3. Upgrade Intro Page with Performance Claims & Snippet
* **Feedback**: "consider an eye catching claim about performance... worlds fastest implementation, based on just-published state of the art research... consider putting a code snippet on the home page."
* **Action Item**: Update `docs/index.md` to:
  * Highlight that Gridwise is the fastest WebGPU primitive implementation, based on decoupled fallback/lookback state-of-the-art research.
  * Add a code snippet demonstrating how simple it is to import and run a sort/scan operation (e.g., `await OneSweepSort({ ... }).execute();`).
* **Files to modify**:
  * [docs/index.md](file:///Users/jdowens/Documents/working/gridwise/docs/index.md)

### 4. Setup Redirection for `https://gridwise-webgpu.github.io/` 404
* **Feedback**: "https://gridwise-webgpu.github.io/ is a 404, should be https://gridwise-webgpu.github.io/gridwise/"
* **Action Item**: Setup a landing/redirect page at the root level of the GitHub Pages user/org page. This is usually done by creating a repository named `gridwise-webgpu.github.io` with an `index.html` that performs a redirect to `/gridwise/`.

---

## 📝 Priority 3: Documentation Clarifications & Terminology

### 5. Clarify the Machine Lockup Phrasing & WebGPU safety net
* **Feedback**: "'this locks up the entire machine' sounds pretty scary! perhaps an opportunity to mention webgpu provides safety net."
* **Action Item**: Rephrase the sentence in `docs/scan-and-reduce.md` to soften the impact and explain how WebGPU's safety mechanisms (like device loss, robust page scheduling, or execution timeouts) protect the system from infinite stalls.
* **Files to modify**:
  * [docs/scan-and-reduce.md](file:///Users/jdowens/Documents/working/gridwise/docs/scan-and-reduce.md#L34)

### 6. Mark Gridwise's Choices in "Design Choice" Sections
* **Feedback**: "The 'design choice' sections - unclear which one gridwise has chosen. maybe put a green checkmark on the gridwise approach."
* **Action Item**: Edit `docs/primitive-design.md` to clearly indicate (e.g., via checkmarks ✅) the choices that Gridwise implements (e.g., "Use subgroups + emulation" and "Always use chained algorithms").
* **Files to modify**:
  * [docs/primitive-design.md](file:///Users/jdowens/Documents/working/gridwise/docs/primitive-design.md#L37)

### 7. Replace "emu" Abbreviation with "Emulated" or "Emulation"
* **Feedback**: "'emu' sounds like a bird... write out emulated?"
* **Action Item**: Conduct a thorough replace in the documentation to spell out "emulated" or "emulation" instead of the shorthand "emu".
* **Files to modify**:
  * [docs/subgroup-strategy.md](file:///Users/jdowens/Documents/working/gridwise/docs/subgroup-strategy.md)

### 8. Polish the BinOp Example Description and Order (Issues #36, #35)
* **Feedback**: "in the binop example, : `{ datatype = "..." }` is a little confusing... in binop example, consider putting the gpu stuff first, and the cpu stuff at the end with a comment that it's the cpu stuff."
* **Action Item**: 
  * Ensure the placeholder text in the examples is easy to read (using `{ datatype: "f32" }` rather than the confusing `{ datatype = "..." }` syntax).
  * Ensure `docs/binop.md` displays the GPU WGSL code generation first, and lists the CPU JavaScript validation method at the end of the constructor with clear comments.
* **Files to modify**:
  * [docs/binop.md](file:///Users/jdowens/Documents/working/gridwise/docs/binop.md)

### 9. Accurate CPU Timing Measurement (Issue #9)
* **Feedback**: When CPU timing is active, `t0` could be set during a previous asynchronous GPU submission.
* **Action Item**: Ensure the GPU queue is idle/cleared before timing starts (e.g. flush the queue or run onSubmittedWorkDone() first).
* **Files to modify**:
  * [primitive.mjs](file:///Users/jdowens/Documents/working/gridwise/primitive.mjs)

### 10. Better Formatting of Documentation (Issue #8)
* **Feedback**: Currently Gridwise documentation is done with default Jekyll templates. Propose a cleaner documentation layout that does not treat documentation pages as blog posts (no dates or chronological ordering).
* **Action Item**: Improve Jekyll layouts for Gridwise documentation.

---

## 🚀 Priority 4: Future Research & Performance Benchmarking

### 11. Add GPU vs. CPU Performance Comparison Chart on Front Page
* **Feedback**: "maybe even a small chart on the front page showing off sort on the gpu vs. cpu?"
* **Action Item**: Once stable baselines are recorded, generate a comparison plot and embed it in `docs/index.md` or a dedicated performance summary.

### 12. Create Educational Subpage
* **Feedback**: "consider noting briefly why gpu programmers will come to need scan, reduce and sort. Many casual viewers will see the site and not know yet."
* **Action Item**: Add an educational guide page linked from the home page.

### 13. Verify Sort Performance & Data Volume (Issue #31)
* **Problem**: Plotted sort bandwidth in `scan_sort_perf.html` is modest.
* **Action Item**: Double check data volume calculations for sort operations and perform lower-level benchmarking to verify bandwidth accuracy.
