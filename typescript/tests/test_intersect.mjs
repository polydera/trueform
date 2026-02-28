import { describe, test, log, assert, getTf } from "./harness.mjs";

// Two overlapping spheres for intersection tests
function twoSpheres() {
  const tf = getTf();
  const s0 = tf.sphereMesh(1, 16, 16);
  const s1 = tf.sphereMesh(1, 16, 16);
  s1.transformation = tf.makeTranslation(1, 0, 0);
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

describe("Isocontours", () => {

  test("isocontours single threshold", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0); // x-coords as scalar field

    const curves = tf.isocontours(plane, scalars, 0.0);
    assert(curves.length > 0, `curves count: ${curves.length}`);
    assert(curves.points.shape[1] === 3, "3D points");
    log(`  isocontours(single): ${curves.length} curves`, "line-pass");

    curves.delete(); scalars.delete(); plane.delete();
  });

  test("isocontours multiple thresholds", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const thresholds = new Float32Array([-1, 0, 1]);

    const curves = tf.isocontours(plane, scalars, thresholds);
    assert(curves.length > 0, `curves count: ${curves.length}`);
    log(`  isocontours(multi): ${curves.length} curves`, "line-pass");

    curves.delete(); scalars.delete(); plane.delete();
  });

});
