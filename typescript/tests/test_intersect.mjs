import { describe, test, log, assert, getTf } from "./harness.mjs";

// Two overlapping spheres for intersection tests
function twoSpheres() {
  const tf = getTf();
  const s0 = tf.sphereMesh(1, 16, 16);
  const s1 = tf.sphereMesh(1, 16, 16);
  s1.transformation = tf.makeTranslation(1, 0, 0);
  return { tf, s0, s1 };
}

function twoSpheresFloat64() {
  const tf = getTf();
  const s0 = tf.sphereMesh(1, 16, 16, { dtype: "float64" });
  const s1 = tf.sphereMesh(1, 16, 16, { dtype: "float64" });
  // prettier-ignore
  s1.transformation = tf.ndarray(new Float64Array([
    1, 0, 0, 1,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]), [4, 4]);
  return { tf, s0, s1 };
}

describe("Intersection curves", () => {

  test("intersectionCurves(m0, m1)", () => {
    const { tf, s0, s1 } = twoSpheres();
    const curves = tf.intersectionCurves(s0, s1);

    assert(curves.length > 0, `curves count: ${curves.length}`);
    assert(curves.points.shape[1] === 3, "3D points");
    log(`  intersectionCurves: ${curves.length} curves`, "line-pass");

    curves.delete(); s1.delete(); s0.delete();
  });

  test("selfIntersectionCurves (no SI)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8);
    const curves = tf.selfIntersectionCurves(sphere);

    assert(curves.length === 0, "sphere has no self-intersections");
    log("  selfIntersectionCurves (no SI)", "line-pass");

    curves.delete(); sphere.delete();
  });

});

describe("Intersection curves (N-mesh + mode)", () => {

  test("intersectionCurves([m0, m1, m2])", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 16, 16);
    const s1 = tf.sphereMesh(1, 16, 16);
    const s2 = tf.sphereMesh(1, 16, 16);
    s0.transformation = tf.makeTranslation(0.5, 0, 0);
    s1.transformation = tf.makeTranslation(-0.5, 0, 0);
    s2.transformation = tf.makeTranslation(0, 0.5, 0);

    const curves = tf.intersectionCurves([s0, s1, s2]);

    assert(curves.length >= 3, `expected >= 3 curves, got ${curves.length}`);
    assert(curves.points.shape[1] === 3, "3D points");
    log(`  N-mesh curves: ${curves.length} curves`, "line-pass");

    curves.delete(); s2.delete(); s1.delete(); s0.delete();
  });

  test("intersectionCurves with mode", () => {
    const { tf, s0, s1 } = twoSpheres();
    const curves = tf.intersectionCurves(s0, s1, { mode: "primitives" });

    assert(curves.length > 0, `curves count: ${curves.length}`);
    log(`  curves (primitives mode): ${curves.length} curves`, "line-pass");

    curves.delete(); s1.delete(); s0.delete();
  });

  test("selfIntersectionCurves with mode", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8);
    const curves = tf.selfIntersectionCurves(sphere, { mode: "primitives" });

    assert(curves.length === 0, "sphere has no self-intersections");
    log("  selfIntersectionCurves (primitives mode)", "line-pass");

    curves.delete(); sphere.delete();
  });

});

describe("Intersection curves (float64)", () => {

  test("intersectionCurves(m0, m1) (float64)", () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const curves = tf.intersectionCurves(s0, s1);

    assert(curves.dtype === "float64", "result curves dtype is float64");
    assert(curves.length > 0, `curves count: ${curves.length}`);
    assert(curves.points.shape[1] === 3, "3D points");
    log(`  intersectionCurves (f64): ${curves.length} curves`, "line-pass");

    curves.delete(); s1.delete(); s0.delete();
  });

  test("selfIntersectionCurves (no SI) (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const curves = tf.selfIntersectionCurves(sphere);

    assert(curves.dtype === "float64", "result curves dtype is float64");
    assert(curves.length === 0, "sphere has no self-intersections");
    log("  selfIntersectionCurves (no SI, f64)", "line-pass");

    curves.delete(); sphere.delete();
  });

});

describe("Intersection curves (N-mesh + mode) (float64)", () => {

  test("intersectionCurves([m0, m1, m2]) (float64)", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 16, 16, { dtype: "float64" });
    const s1 = tf.sphereMesh(1, 16, 16, { dtype: "float64" });
    const s2 = tf.sphereMesh(1, 16, 16, { dtype: "float64" });
    // prettier-ignore
    s0.transformation = tf.ndarray(new Float64Array([
      1, 0, 0, 0.5,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]), [4, 4]);
    // prettier-ignore
    s1.transformation = tf.ndarray(new Float64Array([
      1, 0, 0, -0.5,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]), [4, 4]);
    // prettier-ignore
    s2.transformation = tf.ndarray(new Float64Array([
      1, 0, 0, 0,
      0, 1, 0, 0.5,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]), [4, 4]);

    const curves = tf.intersectionCurves([s0, s1, s2]);

    assert(curves.dtype === "float64", "result curves dtype is float64");
    assert(curves.length >= 3, `expected >= 3 curves, got ${curves.length}`);
    assert(curves.points.shape[1] === 3, "3D points");
    log(`  N-mesh curves (f64): ${curves.length} curves`, "line-pass");

    curves.delete(); s2.delete(); s1.delete(); s0.delete();
  });

  test("intersectionCurves with mode (float64)", () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const curves = tf.intersectionCurves(s0, s1, { mode: "primitives" });

    assert(curves.dtype === "float64", "result curves dtype is float64");
    assert(curves.length > 0, `curves count: ${curves.length}`);
    log(`  curves (primitives mode, f64): ${curves.length} curves`, "line-pass");

    curves.delete(); s1.delete(); s0.delete();
  });

  test("selfIntersectionCurves with mode (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const curves = tf.selfIntersectionCurves(sphere, { mode: "primitives" });

    assert(curves.dtype === "float64", "result curves dtype is float64");
    assert(curves.length === 0, "sphere has no self-intersections");
    log("  selfIntersectionCurves (primitives mode, f64)", "line-pass");

    curves.delete(); sphere.delete();
  });

});

describe("Intersection curves (dtype mismatch)", () => {

  test("intersectionCurves(m0, m1) dtype mismatch throws", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { tf.intersectionCurves(s0, s1); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  intersectionCurves dtype mismatch throws`, "line-pass");
    s1.delete(); s0.delete();
  });

  test("intersectionCurves([...]) dtype mismatch throws", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8);
    const s2 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { tf.intersectionCurves([s0, s1, s2]); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  intersectionCurves N-mesh dtype mismatch throws`, "line-pass");
    s2.delete(); s1.delete(); s0.delete();
  });

  test("async: intersectionCurves(m0, m1) dtype mismatch throws", async () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { await tf.async.intersectionCurves(s0, s1); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  async intersectionCurves dtype mismatch throws`, "line-pass");
    s1.delete(); s0.delete();
  });

});
