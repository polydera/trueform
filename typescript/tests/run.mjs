// ============================================================================
// Node.js test runner — runs all tests, prints only failures + summary
// Usage: node tests/run.mjs
// ============================================================================

import { tests, setTf } from "./harness.mjs";

// Import test modules (same as runner.mjs, minus browser-only ones)
await import("./test_ndarray.mjs");
await import("./test_primitives.mjs");
await import("./test_mesh.mjs");
await import("./test_spatial.mjs");
await import("./test_geometry.mjs");
await import("./test_mesh_setters.mjs");
await import("./test_shallow_copy.mjs");
await import("./test_async_precompute.mjs");
await import("./test_point_cloud.mjs");
await import("./test_registration.mjs");
await import("./test_remesh.mjs");
await import("./test_topology.mjs");
await import("./test_reindex.mjs");
await import("./test_transformations.mjs");
await import("./test_clean.mjs");
await import("./test_cut.mjs");
await import("./test_intersect.mjs");
await import("./test_io_roundtrip.mjs");

// Load WASM module
const tf = await import("../dist/index.js");
setTf(tf);

// Run
let passed = 0, failed = 0;
for (const t of tests) {
  if (t._marker) continue;
  try {
    await t.fn();
    passed++;
  } catch (e) {
    failed++;
    console.log(`FAIL: ${t.name}`);
    console.log(`  ${e.message}`);
  }
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
