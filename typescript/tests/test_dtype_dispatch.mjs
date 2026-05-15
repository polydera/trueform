import { describe, test, log, assert, getTf } from "./harness.mjs";

function expectThrows(fn, pattern, msg) {
  let threw = false;
  let actual = "";
  try { fn(); } catch (e) { threw = true; actual = String(e.message ?? e); }
  assert(threw, `${msg}: expected throw, no throw`);
  if (pattern instanceof RegExp) {
    assert(pattern.test(actual), `${msg}: expected /${pattern.source}/, got: ${actual}`);
  } else {
    assert(actual.includes(pattern), `${msg}: expected "${pattern}", got: ${actual}`);
  }
}

async function expectThrowsAsync(fn, pattern, msg) {
  let threw = false;
  let actual = "";
  try { await fn(); } catch (e) { threw = true; actual = String(e.message ?? e); }
  assert(threw, `${msg}: expected throw, no throw`);
  if (pattern instanceof RegExp) {
    assert(pattern.test(actual), `${msg}: expected /${pattern.source}/, got: ${actual}`);
  } else {
    assert(actual.includes(pattern), `${msg}: expected "${pattern}", got: ${actual}`);
  }
}

// ============================================================================
// FF (form + form) — assertSameDtype rejects mixed dtypes
// ============================================================================

describe("dtype dispatch :: FF rejects mixed", () => {

  test("spatial.distance2(mesh32, mesh64) throws", () => {
    const tf = getTf();
    const m32 = tf.sphereMesh(1, 8, 8);
    const m64 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });

    expectThrows(
      () => tf.distance2(m32, m64),
      /dtype mismatch/,
      "distance2(mesh32, mesh64)",
    );

    m32.delete(); m64.delete();
  });

  test("spatial.distance(mesh32, mesh64) throws", () => {
    const tf = getTf();
    const m32 = tf.sphereMesh(1, 8, 8);
    const m64 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });

    expectThrows(
      () => tf.distance(m32, m64),
      /dtype mismatch/,
      "distance(mesh32, mesh64)",
    );

    m32.delete(); m64.delete();
  });

  test("cut.booleanUnion(mesh32, mesh64) throws", () => {
    const tf = getTf();
    const m32 = tf.sphereMesh(1, 8, 8);
    const m64 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });

    expectThrows(
      () => tf.booleanUnion(m32, m64),
      /dtype mismatch/,
      "booleanUnion(mesh32, mesh64)",
    );

    m32.delete(); m64.delete();
  });

  test("intersect.intersectionCurves(mesh32, mesh64) throws", () => {
    const tf = getTf();
    const m32 = tf.sphereMesh(1, 8, 8);
    const m64 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });

    expectThrows(
      () => tf.intersectionCurves(m32, m64),
      /dtype mismatch/,
      "intersectionCurves(mesh32, mesh64)",
    );

    m32.delete(); m64.delete();
  });

  test("async: distance2(mesh32, mesh64) rejects", async () => {
    const tf = getTf();
    const m32 = tf.sphereMesh(1, 8, 8);
    const m64 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });

    await expectThrowsAsync(
      () => tf.async.distance2(m32, m64),
      /dtype mismatch/,
      "async distance2(mesh32, mesh64)",
    );

    m32.delete(); m64.delete();
  });

});

// ============================================================================
// FF same dtype — both float32 and float64 paths produce a result
// ============================================================================

describe("dtype dispatch :: FF same dtype", () => {

  test("distance2(mesh32, mesh32) → number", () => {
    const tf = getTf();
    const a = tf.sphereMesh(1, 8, 8);
    const b = tf.sphereMesh(1, 8, 8);
    const d = tf.distance2(a, b);
    assert(typeof d === "number", `expected number, got ${typeof d}`);
    assert(d >= 0, `non-negative dist^2: ${d}`);
    log(`  distance2 mesh32+mesh32 = ${d}`, "line-pass");
    a.delete(); b.delete();
  });

  test("distance2(mesh64, mesh64) → number", () => {
    const tf = getTf();
    const a = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const b = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const d = tf.distance2(a, b);
    assert(typeof d === "number", `expected number, got ${typeof d}`);
    assert(d >= 0, `non-negative dist^2: ${d}`);
    log(`  distance2 mesh64+mesh64 = ${d}`, "line-pass");
    a.delete(); b.delete();
  });

});

// ============================================================================
// FP (form + primitive) — form's dtype wins; primitive is upcast
// ============================================================================

describe("dtype dispatch :: FP form-dtype wins", () => {

  test("distance2(mesh32, point32) → float32", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8);
    const pt = tf.point(2, 0, 0);
    assert(pt.dtype === "float32", `point32 dtype: ${pt.dtype}`);

    const d = tf.distance2(mesh, pt);
    assert(typeof d === "number", `expected number, got ${typeof d}`);
    log(`  distance2 mesh32+point32 = ${d}`, "line-pass");

    mesh.delete(); pt.delete();
  });

  test("distance2(mesh64, point64) → float64", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const pt = tf.point(2, 0, 0, { dtype: "float64" });
    assert(pt.dtype === "float64", `point64 dtype: ${pt.dtype}`);

    const d = tf.distance2(mesh, pt);
    assert(typeof d === "number", `expected number, got ${typeof d}`);
    log(`  distance2 mesh64+point64 = ${d}`, "line-pass");

    mesh.delete(); pt.delete();
  });

  test("distance2(mesh64, point32) upcasts primitive", () => {
    const tf = getTf();
    const mesh = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const pt = tf.point(2, 0, 0);
    assert(pt.dtype === "float32", `start dtype: ${pt.dtype}`);

    const d = tf.distance2(mesh, pt);
    assert(typeof d === "number", `expected number, got ${typeof d}`);
    log(`  distance2 mesh64+point32 upcasts → ${d}`, "line-pass");

    mesh.delete(); pt.delete();
  });

});

// ============================================================================
// PP (primitive + primitive) — upcast to wider dtype
// ============================================================================

describe("dtype dispatch :: PP upcasts to wider", () => {

  test("distance2(point32, point32) returns float32 result", () => {
    const tf = getTf();
    const a = tf.point(0, 0, 0);
    const b = tf.point(3, 4, 0);
    const d = tf.distance2(a, b);
    assert(typeof d === "number", `expected number, got ${typeof d}`);
    assert(Math.abs(d - 25) < 1e-5, `expected 25, got ${d}`);
    log(`  distance2 pt32+pt32 = ${d}`, "line-pass");
    a.delete(); b.delete();
  });

  test("distance2(point64, point64) returns float64 result", () => {
    const tf = getTf();
    const a = tf.point(0, 0, 0, { dtype: "float64" });
    const b = tf.point(3, 4, 0, { dtype: "float64" });
    const d = tf.distance2(a, b);
    assert(typeof d === "number", `expected number, got ${typeof d}`);
    assert(Math.abs(d - 25) < 1e-12, `expected 25, got ${d}`);
    log(`  distance2 pt64+pt64 = ${d}`, "line-pass");
    a.delete(); b.delete();
  });

  test("distance2(point32, point64) upcasts to float64", () => {
    const tf = getTf();
    const a = tf.point(0, 0, 0);
    const b = tf.point(3, 4, 0, { dtype: "float64" });
    const d = tf.distance2(a, b);
    assert(typeof d === "number", `expected number, got ${typeof d}`);
    assert(Math.abs(d - 25) < 1e-12, `expected 25, got ${d}`);
    log(`  distance2 pt32+pt64 (upcast) = ${d}`, "line-pass");
    a.delete(); b.delete();
  });

});
