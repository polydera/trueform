// ============================================================================
// Test runner — UI + execution (imports harness for registry)
// ============================================================================

import { tests, log, setTf } from "./harness.mjs";

// ============================================================================
// Import test files — add new test modules here
// ============================================================================

await import("./test_ndarray.mjs");
await import("./test_primitives.mjs");
await import("./test_mesh.mjs");
await import("./test_io.mjs");
await import("./test_spatial.mjs");
await import("./test_bench_spatial.mjs");
await import("./test_geometry.mjs");
await import("./test_mesh_setters.mjs");
await import("./test_shallow_copy.mjs");
await import("./test_async_precompute.mjs");
await import("./test_histogram.mjs");
await import("./test_point_cloud.mjs");
await import("./test_registration.mjs");
await import("./test_remesh.mjs");
await import("./test_topology.mjs");
await import("./test_cdt.mjs");
await import("./test_reindex.mjs");
await import("./test_transformations.mjs");
await import("./test_clean.mjs");
await import("./test_cut.mjs");
await import("./test_intersect.mjs");
await import("./test_io_roundtrip.mjs");

// ============================================================================
// UI setup
// ============================================================================

const testListEl = document.getElementById("test-list");
const runBtn = document.getElementById("run-btn");

let testIndex = 0;
const testIndices = [];
const groups = []; // { cb, startCi, endCi } for group-level toggle

tests.forEach((t, i) => {
  if (t._marker) {
    // Close previous group
    if (groups.length > 0) groups[groups.length - 1].endCi = testIndex;

    const g = document.createElement("div");
    g.className = "test-group";

    const gcb = document.createElement("input");
    gcb.type = "checkbox";
    gcb.checked = true;
    gcb.className = "group-checkbox";

    const label = document.createElement("span");
    label.textContent = t.group;

    g.appendChild(gcb);
    g.appendChild(label);
    testListEl.appendChild(g);

    const grp = { cb: gcb, startCi: testIndex, endCi: testIndex };
    groups.push(grp);

    function toggleGroup(checked) {
      for (let ci = grp.startCi; ci < grp.endCi; ci++)
        document.getElementById(`cb-${ci}`).checked = checked;
    }

    gcb.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleGroup(gcb.checked);
    });

    g.addEventListener("click", (e) => {
      if (e.target !== gcb) {
        gcb.checked = !gcb.checked;
        toggleGroup(gcb.checked);
      }
    });

    return;
  }
  const ci = testIndex++;
  testIndices.push(i);
  const div = document.createElement("div");
  div.className = "test-item";
  div.innerHTML = `<div class="status status-idle" id="status-${ci}"></div>
    <input type="checkbox" id="cb-${ci}" checked>
    <label for="cb-${ci}">${t.name}</label>`;
  div.addEventListener("click", (e) => {
    if (e.target.tagName !== "INPUT") document.getElementById(`cb-${ci}`).click();
  });
  testListEl.appendChild(div);
});

// Close the last group
if (groups.length > 0) groups[groups.length - 1].endCi = testIndex;

const totalTests = testIndex;

document.getElementById("select-all-btn").onclick = () => {
  for (let i = 0; i < totalTests; i++) document.getElementById(`cb-${i}`).checked = true;
  for (const g of groups) g.cb.checked = true;
};
document.getElementById("select-none-btn").onclick = () => {
  for (let i = 0; i < totalTests; i++) document.getElementById(`cb-${i}`).checked = false;
  for (const g of groups) g.cb.checked = false;
};
document.getElementById("clear-btn").onclick = () => {
  document.getElementById("output").innerHTML = "";
  document.getElementById("summary").textContent = "";
  for (let i = 0; i < totalTests; i++)
    document.getElementById(`status-${i}`).className = "status status-idle";
};

// ============================================================================
// Runner
// ============================================================================

async function run() {
  runBtn.disabled = true;
  document.getElementById("output").innerHTML = "";
  let passed = 0, failed = 0;

  log("Loading WASM module...", "line-dim");
  try {
    const tf = await import("../dist/index.js");
    setTf(tf);
    log("Module loaded.\n", "line-dim");
  } catch (e) {
    log("Failed to load module: " + e.message, "line-fail");
    runBtn.disabled = false;
    return;
  }

  // Yield to browser so "Module loaded." paints before heavy sync tests
  await new Promise(r => setTimeout(r, 0));

  for (let ci = 0; ci < totalTests; ci++) {
    const cb = document.getElementById(`cb-${ci}`);
    const st = document.getElementById(`status-${ci}`);
    if (!cb.checked) { st.className = "status status-idle"; continue; }

    const t = tests[testIndices[ci]];
    st.className = "status status-running";
    log(t.name, "line-heading");

    try {
      await t.fn();
      st.className = "status status-pass";
      passed++;
    } catch (e) {
      st.className = "status status-fail";
      log("  FAIL: " + e.message, "line-fail");
      failed++;
    }
    log("");
  }

  const summaryEl = document.getElementById("summary");
  summaryEl.textContent = `${passed} passed` + (failed ? `, ${failed} failed` : "");
  summaryEl.style.color = failed ? "#f85149" : "#3fb950";
  runBtn.disabled = false;
}

runBtn.onclick = run;
