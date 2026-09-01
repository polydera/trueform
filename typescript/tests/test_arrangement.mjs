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

  test("boolean with a sheet operand cuts without enclosing", () => {
    const tf = getTf();
    // undeclared, the plane bounds no volume and nothing is inside it;
    // declared a sheet it is an oriented separator, so the two halves
    // are non-empty and together they are the whole sphere
    const sphere = tf.sphereMesh(1, 16, 16);
    const plane = tf.planeMesh(4, 4, 4, 4);

    const plain = tf.booleanIntersection(sphere, plane);
    assert(plain.mesh.numberOfFaces === 0, "no sheet declared: intersection empty");
    plain.faceLabels.delete(); plain.labels.delete(); plain.mesh.delete();

    const lower = tf.booleanIntersection(sphere, plane, { sheets: [1] });
    const upper = tf.booleanDifference(sphere, plane, { sheets: [1] });
    assert(lower.mesh.numberOfFaces > 0, "sheet: lower half is non-empty");
    assert(upper.mesh.numberOfFaces > 0, "sheet: upper half is non-empty");
    log(`  sheet halves: ${lower.mesh.numberOfFaces} + ${upper.mesh.numberOfFaces} faces`,
      "line-pass");

    upper.faceLabels.delete(); upper.labels.delete(); upper.mesh.delete();
    lower.faceLabels.delete(); lower.labels.delete(); lower.mesh.delete();
    plane.delete(); sphere.delete();
  });

  test("a sheet still separates when curves are requested", () => {
    const tf = getTf();
    // returnCurves and sheets are independent options; asking for the seam
    // must not cost the sheet declaration
    const sphere = tf.sphereMesh(1, 16, 16);
    const plane = tf.planeMesh(4, 4, 4, 4);

    const plain = tf.booleanIntersection(sphere, plane, { sheets: [1] });
    const withCurves = tf.booleanIntersection(sphere, plane,
      { sheets: [1], returnCurves: true });

    assert(plain.mesh.numberOfFaces > 0, "sheet: intersection is non-empty");
    assert(withCurves.mesh.numberOfFaces === plain.mesh.numberOfFaces,
      `curves path agrees: ${withCurves.mesh.numberOfFaces} === ${plain.mesh.numberOfFaces}`);
    assert(withCurves.curves.length > 0, `seam curves: ${withCurves.curves.length}`);

    withCurves.curves.delete(); withCurves.faceLabels.delete();
    withCurves.labels.delete(); withCurves.mesh.delete();
    plain.faceLabels.delete(); plain.labels.delete(); plain.mesh.delete();
    plane.delete(); sphere.delete();
  });

  test("a sheet still separates through the async path", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 16, 16);
    const plane = tf.planeMesh(4, 4, 4, 4);

    const sync = tf.booleanIntersection(sphere, plane, { sheets: [1] });
    const async_ = await tf.async.booleanIntersection(sphere, plane, { sheets: [1] });

    assert(async_.mesh.numberOfFaces === sync.mesh.numberOfFaces,
      `async agrees: ${async_.mesh.numberOfFaces} === ${sync.mesh.numberOfFaces}`);

    async_.faceLabels.delete(); async_.labels.delete(); async_.mesh.delete();
    sync.faceLabels.delete(); sync.labels.delete(); sync.mesh.delete();
    plane.delete(); sphere.delete();
  });

  test("boolean sheets option rejects an out-of-range operand id", () => {
    const { tf, s0, s1 } = twoSpheres();
    let threw = false;
    try {
      tf.booleanUnion(s0, s1, { sheets: [2] });
    } catch (e) {
      threw = true;
    }
    assert(threw, "sheets: [2] is rejected");
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

  test("async: booleanUnion", async () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = await tf.async.booleanUnion(s0, s1);

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces,
      `labels length ${result.labels.length} matches faces ${result.mesh.numberOfFaces}`);
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  async union: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("async: booleanIntersection", async () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = await tf.async.booleanIntersection(s0, s1);

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  async intersection: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("async: booleanDifference", async () => {
    const { tf, s0, s1 } = twoSpheres();
    const result = await tf.async.booleanDifference(s0, s1);

    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  async difference: ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
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

  test("meshArrangements refinedCdt: valid, more points than cdt", () => {
    const { tf, s0, s1 } = twoSpheres();
    const stock = tf.meshArrangements([s0, s1]);
    const refined = tf.meshArrangements([s0, s1], { triangulation: "refinedCdt" });

    assert(refined.mesh.numberOfFaces > 0, "refined has faces");
    assert(refined.tagLabels.length === refined.mesh.numberOfFaces, "tagLabels match");
    assert(refined.faceLabels.length === refined.mesh.numberOfFaces, "faceLabels match");
    assert(refined.mesh.numberOfPoints >= stock.mesh.numberOfPoints,
      `refined points ${refined.mesh.numberOfPoints} >= cdt ${stock.mesh.numberOfPoints}`);
    log(`  meshArrangements refinedCdt: ${refined.mesh.numberOfPoints} pts vs cdt ${stock.mesh.numberOfPoints}`, "line-pass");

    refined.faceLabels.delete(); refined.tagLabels.delete(); refined.mesh.delete();
    stock.faceLabels.delete(); stock.tagLabels.delete(); stock.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("async meshArrangements refinedCdt matches sync", async () => {
    const { tf, s0, s1 } = twoSpheres();
    const sync = tf.meshArrangements([s0, s1], { triangulation: "refinedCdt" });
    const asyncResult = await tf.async.meshArrangements([s0, s1], { triangulation: "refinedCdt" });

    assert(asyncResult.mesh.numberOfPoints === sync.mesh.numberOfPoints,
      "async refined points match sync");
    assert(asyncResult.mesh.numberOfFaces === sync.mesh.numberOfFaces,
      "async refined faces match sync");
    log("  async meshArrangements refinedCdt matches sync", "line-pass");

    asyncResult.faceLabels.delete(); asyncResult.tagLabels.delete(); asyncResult.mesh.delete();
    sync.faceLabels.delete(); sync.tagLabels.delete(); sync.mesh.delete();
    s1.delete(); s0.delete();
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

  test("polygonArrangements refinedCdt: valid, more points than cdt", () => {
    const { tf, s0, s1, s2 } = threeSpheres();
    const merged = tf.concatenateMeshes([s0, s1, s2]);

    const stock = tf.polygonArrangements(merged);
    const refined = tf.polygonArrangements(merged, { triangulation: "refinedCdt" });

    assert(refined.mesh.numberOfFaces > 0, "refined has faces");
    assert(refined.faceLabels.length === refined.mesh.numberOfFaces, "faceLabels match");
    assert(refined.mesh.numberOfPoints >= stock.mesh.numberOfPoints,
      `refined points ${refined.mesh.numberOfPoints} >= cdt ${stock.mesh.numberOfPoints}`);
    log(`  polygonArrangements refinedCdt: ${refined.mesh.numberOfPoints} pts vs cdt ${stock.mesh.numberOfPoints}`, "line-pass");

    refined.faceLabels.delete(); refined.mesh.delete();
    stock.faceLabels.delete(); stock.mesh.delete();
    merged.delete(); s2.delete(); s1.delete(); s0.delete();
  });

  test("async polygonArrangements refinedCdt matches sync", async () => {
    const { tf, s0, s1, s2 } = threeSpheres();
    const merged = tf.concatenateMeshes([s0, s1, s2]);

    const sync = tf.polygonArrangements(merged, { triangulation: "refinedCdt" });
    const asyncResult = await tf.async.polygonArrangements(merged, { triangulation: "refinedCdt" });

    assert(asyncResult.mesh.numberOfPoints === sync.mesh.numberOfPoints,
      "async refined points match sync");
    assert(asyncResult.mesh.numberOfFaces === sync.mesh.numberOfFaces,
      "async refined faces match sync");
    log("  async polygonArrangements refinedCdt matches sync", "line-pass");

    asyncResult.faceLabels.delete(); asyncResult.mesh.delete();
    sync.faceLabels.delete(); sync.mesh.delete();
    merged.delete(); s2.delete(); s1.delete(); s0.delete();
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

// ============================================================================
// Float64 mirror tests
// ============================================================================

function twoSpheresFloat64() {
  const tf = getTf();
  const s0 = tf.sphereMesh(1, 16, 16, { dtype: "float64" });
  const s1 = tf.sphereMesh(1, 16, 16, { dtype: "float64" });
  // makeTranslation emits float32; the setter converts to the mesh dtype.
  s1.transformation = tf.makeTranslation(1, 0, 0);
  return { tf, s0, s1 };
}

describe("Boolean operations (float64)", () => {

  test("booleanUnion (float64)", () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = tf.booleanUnion(s0, s1);

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces,
      `labels length ${result.labels.length} matches faces ${result.mesh.numberOfFaces}`);
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  union (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanUnion with curves (float64)", () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = tf.booleanUnion(s0, s1, { returnCurves: true });

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.curves.dtype === "float64", "result curves dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, `curves count: ${result.curves.length}`);
    log(`  union with curves (f64): ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanIntersection (float64)", () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = tf.booleanIntersection(s0, s1);

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  intersection (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanIntersection with curves (float64)", () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = tf.booleanIntersection(s0, s1, { returnCurves: true });

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.curves.dtype === "float64", "result curves dtype is float64");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, "curves non-empty");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  intersection with curves (f64): ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanDifference (float64)", () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = tf.booleanDifference(s0, s1);

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  difference (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("booleanDifference with curves (float64)", () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = tf.booleanDifference(s0, s1, { returnCurves: true });

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.curves.dtype === "float64", "result curves dtype is float64");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, "curves non-empty");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  difference with curves (f64): ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("async: booleanUnion (float64)", async () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = await tf.async.booleanUnion(s0, s1);

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces,
      `labels length ${result.labels.length} matches faces ${result.mesh.numberOfFaces}`);
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  async union (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("async: booleanIntersection (float64)", async () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = await tf.async.booleanIntersection(s0, s1);

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  async intersection (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("async: booleanDifference (float64)", async () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = await tf.async.booleanDifference(s0, s1);

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.labels.length === result.mesh.numberOfFaces, "labels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  async difference (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.labels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

});

describe("Boolean operations (dtype mismatch)", () => {

  test("booleanUnion dtype mismatch throws", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { tf.booleanUnion(s0, s1); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  booleanUnion dtype mismatch throws`, "line-pass");
    s1.delete(); s0.delete();
  });

  test("booleanIntersection dtype mismatch throws", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { tf.booleanIntersection(s0, s1); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  booleanIntersection dtype mismatch throws`, "line-pass");
    s1.delete(); s0.delete();
  });

  test("booleanDifference dtype mismatch throws", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { tf.booleanDifference(s0, s1); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  booleanDifference dtype mismatch throws`, "line-pass");
    s1.delete(); s0.delete();
  });

  test("async: booleanUnion dtype mismatch throws", async () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { await tf.async.booleanUnion(s0, s1); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  async booleanUnion dtype mismatch throws`, "line-pass");
    s1.delete(); s0.delete();
  });

});

describe("Mesh arrangements (float64)", () => {

  test("meshArrangements([m0, m1]) (float64)", () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = tf.meshArrangements([s0, s1]);

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.tagLabels.length === result.mesh.numberOfFaces, "tagLabels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  meshArrangements (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.tagLabels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("meshArrangements with curves (float64)", () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = tf.meshArrangements([s0, s1], { returnCurves: true });

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.curves.dtype === "float64", "result curves dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, `curves count: ${result.curves.length}`);
    log(`  meshArrangements with curves (f64): ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.tagLabels.delete();
    result.mesh.delete(); s1.delete(); s0.delete();
  });

  test("async: meshArrangements([m0, m1]) (float64)", async () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = await tf.async.meshArrangements([s0, s1]);

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.tagLabels.length === result.mesh.numberOfFaces, "tagLabels match");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log(`  async meshArrangements (f64): ${result.mesh.numberOfFaces} faces`, "line-pass");

    result.faceLabels.delete(); result.tagLabels.delete(); result.mesh.delete();
    s1.delete(); s0.delete();
  });

  test("async: meshArrangements with curves (float64)", async () => {
    const { tf, s0, s1 } = twoSpheresFloat64();
    const result = await tf.async.meshArrangements([s0, s1], { returnCurves: true });

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.curves.dtype === "float64", "result curves dtype is float64");
    assert(result.mesh.numberOfFaces > 0, "has faces");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length > 0, `curves count: ${result.curves.length}`);
    log(`  async meshArrangements with curves (f64): ${result.curves.length} curves`, "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.tagLabels.delete();
    result.mesh.delete(); s1.delete(); s0.delete();
  });

});

describe("Mesh arrangements (dtype mismatch)", () => {

  test("meshArrangements dtype mismatch throws", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { tf.meshArrangements([s0, s1]); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  meshArrangements dtype mismatch throws`, "line-pass");
    s1.delete(); s0.delete();
  });

  test("async: meshArrangements dtype mismatch throws", async () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 8, 8);
    const s1 = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    let threw = false;
    try { await tf.async.meshArrangements([s0, s1]); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  async meshArrangements dtype mismatch throws`, "line-pass");
    s1.delete(); s0.delete();
  });

});

describe("Polygon arrangements (float64)", () => {

  test("polygonArrangements (no SI) (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const result = tf.polygonArrangements(sphere);

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces === sphere.numberOfFaces,
      "no self-intersection → same face count");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log("  polygonArrangements (no SI) (f64)", "line-pass");

    result.faceLabels.delete(); result.mesh.delete(); sphere.delete();
  });

  test("polygonArrangements with curves (no SI) (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const result = tf.polygonArrangements(sphere, { returnCurves: true });

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.curves.dtype === "float64", "result curves dtype is float64");
    assert(result.mesh.numberOfFaces === sphere.numberOfFaces, "same face count");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length === 0, "no SI curves");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log("  polygonArrangements with curves (no SI) (f64)", "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.mesh.delete();
    sphere.delete();
  });

  test("async: polygonArrangements (no SI) (float64)", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const result = await tf.async.polygonArrangements(sphere);

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.mesh.numberOfFaces === sphere.numberOfFaces,
      "no self-intersection → same face count");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log("  async polygonArrangements (no SI) (f64)", "line-pass");

    result.faceLabels.delete(); result.mesh.delete(); sphere.delete();
  });

  test("async: polygonArrangements with curves (no SI) (float64)", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1, 8, 8, { dtype: "float64" });
    const result = await tf.async.polygonArrangements(sphere, { returnCurves: true });

    assert(result.mesh.dtype === "float64", "result mesh dtype is float64");
    assert(result.curves.dtype === "float64", "result curves dtype is float64");
    assert(result.mesh.numberOfFaces === sphere.numberOfFaces, "same face count");
    assert(result.curves !== undefined, "has curves");
    assert(result.curves.length === 0, "no SI curves");
    assert(result.faceLabels.length === result.mesh.numberOfFaces, "faceLabels match");
    log("  async polygonArrangements with curves (no SI) (f64)", "line-pass");

    result.curves.delete(); result.faceLabels.delete(); result.mesh.delete();
    sphere.delete();
  });

});
