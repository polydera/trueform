import { describe, test, log, assert, getTf } from "./harness.mjs";

// Two overlapping spheres for boolean tests
function twoSpheres() {
  const tf = getTf();
  const s0 = tf.sphereMesh(1, 16, 16);
  const s1 = tf.sphereMesh(1, 16, 16);
  s1.transformation = tf.makeTranslation(1, 0, 0);
  return { tf, s0, s1 };
}

describe("Boolean operations", () => {

  test("booleanUnion", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanUnion(s0, s1);

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces,
      `labels length ${result.labels.length} matches faces ${result.mesh.numberOfFaces}`);
    log(`  union: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanUnion with curves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanUnion(s0, s1, { returnCurves: true });

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, `curves count: ${result.curves.length}`);
    log(`  union with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanIntersection", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanIntersection(s0, s1);

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    log(`  intersection: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanIntersection with curves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanIntersection(s0, s1, { returnCurves: true });

    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, "curves non-empty");
    log(`  intersection with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanDifference", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanDifference(s0, s1);

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    log(`  difference: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanDifference with curves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanDifference(s0, s1, { returnCurves: true });

    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, "curves non-empty");
    log(`  difference with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

});

describe("Isobands", () => {

  test("isobands basic", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0); // x-coords as scalar field
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues);
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match faces");
    log(`  isobands: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with curves", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, { returnCurves: true });
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, `curves count: ${result.curves.length}`);
    log(`  isobands with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with selectedBands", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, { selectedBands: [1] });
    assert(result.mesh.numberOfFaces > 0, "has faces");
    log(`  isobands selectedBands: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with selectedBands + curves", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, {
      selectedBands: [1],
      returnCurves: true,
    });
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.curves !== undefined, "has curves");
    log("  isobands selectedBands + curves", "line-pass");

    result.curves.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

});

describe("Embedded curves", () => {

  test("embeddedIntersectionCurves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.embeddedIntersectionCurves(s0, s1);

    assert(result.numberOfFaces >= s0.numberOfFaces,
      "embedded mesh has at least as many faces");
    log(`  embedded: ${result.numberOfFaces} faces`, "line-pass");

    result.delete(); s1.delete(); s0.delete();
  });

  test("embeddedIntersectionCurves with curves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.embeddedIntersectionCurves(s0, s1, { returnCurves: true });

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, "curves non-empty");
    log(`  embedded with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("embeddedSelfIntersectionCurves", () => {
    const tf = getTf();
    // A sphere has no self-intersections, so result = same mesh
    const sphere = tf.sphereMesh(1, 8, 8);
    const result = tf.embeddedSelfIntersectionCurves(sphere);

    assert(result.numberOfFaces === sphere.numberOfFaces,
      "no self-intersection → same face count");
    log("  embeddedSelfIntersection (no SI)", "line-pass");

    result.delete(); sphere.delete();
  });

  test("embeddedSelfIntersectionCurves with curves", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8);
    const result = tf.embeddedSelfIntersectionCurves(sphere, { returnCurves: true });

    assert(result.mesh.numberOfFaces === sphere.numberOfFaces, "same face count");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length === 0, "no SI curves");
    log("  embeddedSelfIntersection with curves (no SI)", "line-pass");

    result.curves.delete(); result.mesh.delete(); sphere.delete();
  });

});
