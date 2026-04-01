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
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  union: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanUnion with curves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanUnion(s0, s1, { returnCurves: true });

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, `curves count: ${result.curves.length}`);
    log(`  union with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanIntersection", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanIntersection(s0, s1);

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  intersection: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanIntersection with curves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanIntersection(s0, s1, { returnCurves: true });

    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, "curves non-empty");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  intersection with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanDifference", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanDifference(s0, s1);

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  difference: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanDifference with curves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.booleanDifference(s0, s1, { returnCurves: true });

    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, "curves non-empty");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  difference with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
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
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  isobands: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with curves", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, { returnCurves: true });
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, `curves count: ${result.curves.length}`);
    log(`  isobands with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

  test("isobands with selectedBands", () => {
    const tf = getTf();
    const plane = tf.planeMesh(4, 4, 20, 20);
    const scalars = plane.points.take(null, 0);
    const cutValues = new Float32Array([-1, 0, 1]);

    const result = tf.isobands(plane, scalars, cutValues, { selectedBands: [1] });
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  isobands selectedBands: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
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
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    assert(result.curves !== undefined, "has curves");
    log("  isobands selectedBands + curves", "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    scalars.delete(); plane.delete();
  });

});

describe("Embedded curves", () => {

  test("embeddedIntersectionCurves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.embeddedIntersectionCurves(s0, s1);

    assert(result.mesh.numberOfFaces >= s0.numberOfFaces,
      "embedded mesh has at least as many faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  embedded: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.mesh.delete(); s1.delete(); s0.delete();
  });

  test("embeddedIntersectionCurves with curves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.embeddedIntersectionCurves(s0, s1, { returnCurves: true });

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, "curves non-empty");
    log(`  embedded with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("embeddedSelfIntersectionCurves", () => {
    const tf = getTf();
    // A sphere has no self-intersections, so result = same mesh
    const sphere = tf.sphereMesh(1, 8, 8);
    const result = tf.embeddedSelfIntersectionCurves(sphere);

    assert(result.mesh.numberOfFaces === sphere.numberOfFaces,
      "no self-intersection → same face count");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log("  embeddedSelfIntersection (no SI)", "line-pass");

    result.faceLabels.delete(); result.mesh.delete(); sphere.delete();
  });

  test("embeddedSelfIntersectionCurves with curves", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8);
    const result = tf.embeddedSelfIntersectionCurves(sphere, { returnCurves: true });

    assert(result.mesh.numberOfFaces === sphere.numberOfFaces, "same face count");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length === 0, "no SI curves");
    log("  embeddedSelfIntersection with curves (no SI)", "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.mesh.delete(); sphere.delete();
  });

});

describe("Mesh arrangements", () => {

  test("meshArrangements([m0, m1])", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.meshArrangements([s0, s1]);

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.tagLabels.length === result.mesh.numberOfFaces, "tagLabels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  meshArrangements: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.tagLabels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("meshArrangements with curves", () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = tf.meshArrangements([s0, s1], { returnCurves: true });

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, `curves count: ${result.curves.length}`);
    log(`  meshArrangements with curves: ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.tagLabels.delete();
    result.mesh.delete(); s1.delete(); s0.delete();
  });

});

describe("Polygon arrangements", () => {

  test("polygonArrangements (no SI)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8);
    const result = tf.polygonArrangements(sphere);

    assert(result.mesh.numberOfFaces === sphere.numberOfFaces,
      "no self-intersection → same face count");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log("  polygonArrangements (no SI)", "line-pass");

    result.faceLabels.delete(); result.mesh.delete(); sphere.delete();
  });

  test("polygonArrangements with curves (no SI)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8);
    const result = tf.polygonArrangements(sphere, { returnCurves: true });

    assert(result.mesh.numberOfFaces === sphere.numberOfFaces, "same face count");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length === 0, "no SI curves");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log("  polygonArrangements with curves (no SI)", "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.mesh.delete();
    sphere.delete();
  });

});

// ============================================================================
// 3-sphere cross-API consistency
// ============================================================================

function threeSpheres() {
  const tf = getTf();
  const d = 1.0;
  const s0 = tf.sphereMesh(1, 40, 40);
  const s1 = tf.sphereMesh(1, 40, 40);
  const s2 = tf.sphereMesh(1, 40, 40);
  s1.transformation = tf.makeTranslation(d, 0, 0);
  s2.transformation = tf.makeTranslation(d / 2, d * Math.sqrt(3) / 2, 0);
  return { tf, s0, s1, s2 };
}

function curveStats(curves) {
  const paths = curves.paths;
  const n = paths.length;
  let nClosed = 0, nOpen = 0;
  const endpoints = new Set();
  const edgeCounts = [];
  for (let i = 0; i < n; i++) {
    const ids = paths.get(i).data;
    edgeCounts.push(ids.length - 1);
    if (ids.length >= 2 && ids[0] === ids[ids.length - 1]) {
      nClosed++;
    } else {
      nOpen++;
      if (ids.length >= 2) {
        endpoints.add(ids[0]);
        endpoints.add(ids[ids.length - 1]);
      }
    }
  }
  edgeCounts.sort((a, b) => a - b);
  return { n, edgeCounts, nClosed, nOpen, nEndpoints: endpoints.size };
}

describe("3-sphere arrangement consistency", () => {

  test("meshArrangements: 6 curves, 2 endpoints, matches non-manifold", () => {
    const { tf, s0, s1, s2 } = threeSpheres();

    const result = tf.meshArrangements([s0, s1, s2], { returnCurves: true });
    const cs = curveStats(result.curves);

    assert(cs.n === 6, `expected 6 curves, got ${cs.n}`);
    assert(cs.nEndpoints === 2, `expected 2 endpoints, got ${cs.nEndpoints}`);
    log(`  meshArr: ${cs.n} curves, ${cs.nClosed} closed, ${cs.nOpen} open, ${cs.nEndpoints} endpoints`, "line-pass");

    // non-manifold edges should match
    const nm = tf.nonManifoldEdges(result.mesh);
    const nmPaths = tf.connectEdgesToPaths(nm);
    const nmStats = { n: nmPaths.length };
    assert(nmStats.n === 6, `expected 6 nm paths, got ${nmStats.n}`);
    log(`  nm paths: ${nmStats.n}`, "line-pass");

    nmPaths.delete(); nm.delete();
    result.curves.delete(); result.faceLabels.delete();
    result.tagLabels.delete(); result.mesh.delete();
    s2.delete(); s1.delete(); s0.delete();
  });

  test("polygonArrangements on merged: same curve stats", () => {
    const { tf, s0, s1, s2 } = threeSpheres();

    // merge 3 spheres into one mesh
    const merged = tf.concatenateMeshes([s0, s1, s2]);

    const result = tf.polygonArrangements(merged, { returnCurves: true });
    const cs = curveStats(result.curves);

    assert(cs.n === 6, `expected 6 curves, got ${cs.n}`);
    assert(cs.nEndpoints === 2, `expected 2 endpoints, got ${cs.nEndpoints}`);
    log(`  polyArr: ${cs.n} curves, ${cs.nClosed} closed, ${cs.nOpen} open, ${cs.nEndpoints} endpoints`, "line-pass");

    // non-manifold check
    const nm = tf.nonManifoldEdges(result.mesh);
    const nmPaths = tf.connectEdgesToPaths(nm);
    assert(nmPaths.length === 6, `expected 6 nm paths, got ${nmPaths.length}`);
    log(`  nm paths: ${nmPaths.length}`, "line-pass");

    nmPaths.delete(); nm.delete();
    result.curves.delete(); result.faceLabels.delete(); result.mesh.delete();
    merged.delete(); s2.delete(); s1.delete(); s0.delete();
  });

  test("all curve sources agree: counts and endpoints", () => {
    const { tf, s0, s1, s2 } = threeSpheres();

    // 1. N-mesh intersection curves (primitives)
    const ic = tf.intersectionCurves([s0, s1, s2], { mode: "primitives" });
    const icStats = curveStats(ic);

    // 2. self_intersection_curves on merged
    const merged = tf.concatenateMeshes([s0, s1, s2]);
    const si = tf.selfIntersectionCurves(merged, { mode: "primitives" });
    const siStats = curveStats(si);

    // 3. mesh_arrangements curves
    const ma = tf.meshArrangements([s0, s1, s2], { returnCurves: true });
    const maStats = curveStats(ma.curves);

    // 4. polygon_arrangements curves
    const pa = tf.polygonArrangements(merged, { returnCurves: true });
    const paStats = curveStats(pa.curves);

    log(`  ic: ${icStats.n} curves, ${icStats.nEndpoints} endpoints`, "line-pass");
    log(`  si: ${siStats.n} curves, ${siStats.nEndpoints} endpoints`, "line-pass");
    log(`  ma: ${maStats.n} curves, ${maStats.nEndpoints} endpoints`, "line-pass");
    log(`  pa: ${paStats.n} curves, ${paStats.nEndpoints} endpoints`, "line-pass");

    // all must be 6 curves, 2 endpoints
    assert(icStats.n === 6, `ic: expected 6 curves, got ${icStats.n}`);
    assert(siStats.n === 6, `si: expected 6 curves, got ${siStats.n}`);
    assert(maStats.n === 6, `ma: expected 6 curves, got ${maStats.n}`);
    assert(paStats.n === 6, `pa: expected 6 curves, got ${paStats.n}`);

    assert(icStats.nEndpoints === 2, `ic: expected 2 endpoints, got ${icStats.nEndpoints}`);
    assert(siStats.nEndpoints === 2, `si: expected 2 endpoints, got ${siStats.nEndpoints}`);
    assert(maStats.nEndpoints === 2, `ma: expected 2 endpoints, got ${maStats.nEndpoints}`);
    assert(paStats.nEndpoints === 2, `pa: expected 2 endpoints, got ${paStats.nEndpoints}`);

    // edge counts must match across all sources
    const icEc = JSON.stringify(icStats.edgeCounts);
    const siEc = JSON.stringify(siStats.edgeCounts);
    const maEc = JSON.stringify(maStats.edgeCounts);
    const paEc = JSON.stringify(paStats.edgeCounts);
    assert(icEc === siEc, `ic vs si edge counts: ${icEc} vs ${siEc}`);
    assert(icEc === maEc, `ic vs ma edge counts: ${icEc} vs ${maEc}`);
    assert(icEc === paEc, `ic vs pa edge counts: ${icEc} vs ${paEc}`);
    log(`  edge counts: ${icEc}`, "line-pass");

    // point counts must match
    assert(ic.points.shape[0] === si.points.shape[0],
      `ic vs si points: ${ic.points.shape[0]} vs ${si.points.shape[0]}`);
    assert(ic.points.shape[0] === ma.curves.points.shape[0],
      `ic vs ma points: ${ic.points.shape[0]} vs ${ma.curves.points.shape[0]}`);
    assert(ic.points.shape[0] === pa.curves.points.shape[0],
      `ic vs pa points: ${ic.points.shape[0]} vs ${pa.curves.points.shape[0]}`);
    log(`  point count: ${ic.points.shape[0]}`, "line-pass");

    pa.curves.delete(); pa.faceLabels.delete(); pa.mesh.delete();
    ma.curves.delete(); ma.faceLabels.delete(); ma.tagLabels.delete(); ma.mesh.delete();
    si.delete(); ic.delete(); merged.delete();
    s2.delete(); s1.delete(); s0.delete();
  });

});

describe("3-sphere arrangement consistency (async)", () => {

  test("async: all curve sources agree", async () => {
    const { tf, s0, s1, s2 } = threeSpheres();

    // 1. N-mesh intersection curves (primitives)
    const ic = await tf.async.intersectionCurves([s0, s1, s2], { mode: "primitives" });
    const icStats = curveStats(ic);

    // 2. self_intersection_curves on merged
    const merged = tf.concatenateMeshes([s0, s1, s2]);
    const si = await tf.async.selfIntersectionCurves(merged, { mode: "primitives" });
    const siStats = curveStats(si);

    // 3. mesh_arrangements curves
    const ma = await tf.async.meshArrangements([s0, s1, s2], { returnCurves: true });
    const maStats = curveStats(ma.curves);

    // 4. polygon_arrangements curves
    const pa = await tf.async.polygonArrangements(merged, { returnCurves: true });
    const paStats = curveStats(pa.curves);

    log(`  ic: ${icStats.n} curves, ${icStats.nEndpoints} endpoints`, "line-pass");
    log(`  si: ${siStats.n} curves, ${siStats.nEndpoints} endpoints`, "line-pass");
    log(`  ma: ${maStats.n} curves, ${maStats.nEndpoints} endpoints`, "line-pass");
    log(`  pa: ${paStats.n} curves, ${paStats.nEndpoints} endpoints`, "line-pass");

    assert(icStats.n === 6, `ic: expected 6 curves, got ${icStats.n}`);
    assert(siStats.n === 6, `si: expected 6 curves, got ${siStats.n}`);
    assert(maStats.n === 6, `ma: expected 6 curves, got ${maStats.n}`);
    assert(paStats.n === 6, `pa: expected 6 curves, got ${paStats.n}`);

    assert(icStats.nEndpoints === 2, `ic: expected 2 endpoints, got ${icStats.nEndpoints}`);
    assert(siStats.nEndpoints === 2, `si: expected 2 endpoints, got ${siStats.nEndpoints}`);
    assert(maStats.nEndpoints === 2, `ma: expected 2 endpoints, got ${maStats.nEndpoints}`);
    assert(paStats.nEndpoints === 2, `pa: expected 2 endpoints, got ${paStats.nEndpoints}`);

    const icEc = JSON.stringify(icStats.edgeCounts);
    const siEc = JSON.stringify(siStats.edgeCounts);
    const maEc = JSON.stringify(maStats.edgeCounts);
    const paEc = JSON.stringify(paStats.edgeCounts);
    assert(icEc === siEc, `ic vs si edge counts: ${icEc} vs ${siEc}`);
    assert(icEc === maEc, `ic vs ma edge counts: ${icEc} vs ${maEc}`);
    assert(icEc === paEc, `ic vs pa edge counts: ${icEc} vs ${paEc}`);
    log(`  edge counts: ${icEc}`, "line-pass");

    assert(ic.points.shape[0] === si.points.shape[0],
      `ic vs si points: ${ic.points.shape[0]} vs ${si.points.shape[0]}`);
    assert(ic.points.shape[0] === ma.curves.points.shape[0],
      `ic vs ma points: ${ic.points.shape[0]} vs ${ma.curves.points.shape[0]}`);
    assert(ic.points.shape[0] === pa.curves.points.shape[0],
      `ic vs pa points: ${ic.points.shape[0]} vs ${pa.curves.points.shape[0]}`);
    log(`  point count: ${ic.points.shape[0]}`, "line-pass");

    pa.curves.delete(); pa.faceLabels.delete(); pa.mesh.delete();
    ma.curves.delete(); ma.faceLabels.delete(); ma.tagLabels.delete(); ma.mesh.delete();
    si.delete(); ic.delete(); merged.delete();
    s2.delete(); s1.delete(); s0.delete();
  });

});
