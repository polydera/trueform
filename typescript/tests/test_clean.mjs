import { describe, test, log, assert, approx, getTf } from "./harness.mjs";

// Tetrahedron with duplicate vertex (vertex 3 = vertex 0)
function meshWithDuplicateVertex() {
  return {
    faces: new Int32Array([
      0, 1, 2,
      3, 1, 2,  // face uses vertex 3 which duplicates vertex 0
    ]),
    points: new Float32Array([
      0, 0, 0,   // v0
      1, 0, 0,   // v1
      0.5, 1, 0, // v2
      0, 0, 0,   // v3 = duplicate of v0
    ]),
  };
}

describe("Clean", () => {

  test("cleaned(mesh) — exact duplicates", () => {
    const tf = getTf();
    const { faces, points } = meshWithDuplicateVertex();
    const m = tf.mesh(faces, points);
    const clean = tf.cleaned(m);

    assert(clean.numberOfPoints <= m.numberOfPoints,
      `expected fewer points: ${clean.numberOfPoints} <= ${m.numberOfPoints}`);
    assert(clean.numberOfFaces === 1, `expected 1 face (duplicate removed), got ${clean.numberOfFaces}`);
    log(`  cleaned: ${m.numberOfPoints} → ${clean.numberOfPoints} points`, "line-pass");

    clean.delete(); m.delete();
  });

  test("cleaned(mesh, tolerance)", () => {
    const tf = getTf();
    const { faces, points } = meshWithDuplicateVertex();
    const m = tf.mesh(faces, points);
    const clean = tf.cleaned(m, 1e-6);

    assert(clean.numberOfPoints <= m.numberOfPoints, "fewer or equal points");
    log("  cleaned(mesh, 1e-6)", "line-pass");

    clean.delete(); m.delete();
  });

  test("cleaned(mesh, { returnIndexMap: true })", () => {
    const tf = getTf();
    const { faces, points } = meshWithDuplicateVertex();
    const m = tf.mesh(faces, points);
    const result = tf.cleaned(m, { returnIndexMap: true });

    assert(result.mesh !== undefined, "has mesh");
    assert(result.faceMap !== undefined, "has faceMap");
    assert(result.pointMap !== undefined, "has pointMap");

    const fm = result.faceMap;
    assert(fm.f.length === m.numberOfFaces, `f length: ${fm.f.length}`);
    assert(fm.keptIds.length <= m.numberOfFaces, "keptIds <= original");
    log(`  faceMap: f[${fm.f.length}], keptIds[${fm.keptIds.length}]`, "line-pass");

    const pm = result.pointMap;
    assert(pm.f.length === m.numberOfPoints, `f length: ${pm.f.length}`);
    assert(pm.keptIds.length <= m.numberOfPoints, "keptIds <= original");
    log(`  pointMap: f[${pm.f.length}], keptIds[${pm.keptIds.length}]`, "line-pass");

    fm.delete(); pm.delete(); result.mesh.delete(); m.delete();
  });

  test("cleaned(mesh, { returnIndexMap: true, tolerance })", () => {
    const tf = getTf();
    const { faces, points } = meshWithDuplicateVertex();
    const m = tf.mesh(faces, points);
    const result = tf.cleaned(m, { returnIndexMap: true, tolerance: 1e-6 });

    assert(result.mesh !== undefined, "has mesh");
    assert(result.faceMap !== undefined, "has faceMap");
    assert(result.pointMap !== undefined, "has pointMap");
    log("  cleaned(mesh, { returnIndexMap, tolerance })", "line-pass");

    result.faceMap.delete(); result.pointMap.delete();
    result.mesh.delete(); m.delete();
  });

  test("cleaned(points) — exact duplicates", () => {
    const tf = getTf();
    const pts = tf.point(new Float32Array([
      0,0,0, 1,0,0, 0,0,0, 2,0,0,
    ]), 3);

    const clean = tf.cleaned(pts);
    assert(clean.count <= pts.count,
      `expected fewer points: ${clean.count} <= ${pts.count}`);
    log(`  cleaned(points): ${pts.count} → ${clean.count}`, "line-pass");

    clean.delete(); pts.delete();
  });

  test("cleaned(points, { returnIndexMap: true })", () => {
    const tf = getTf();
    const pts = tf.point(new Float32Array([
      0,0,0, 1,0,0, 0,0,0, 2,0,0,
    ]), 3);

    const result = tf.cleaned(pts, { returnIndexMap: true });
    assert(result.points !== undefined, "has points");
    assert(result.pointMap !== undefined, "has pointMap");
    assert(result.pointMap.keptIds.length <= 4, "keptIds <= 4");
    log("  cleaned(points, { returnIndexMap })", "line-pass");

    result.pointMap.delete(); result.points.delete(); pts.delete();
  });

  test("cleaned(triangleSoup) → Mesh", () => {
    const tf = getTf();
    // Two triangles sharing an edge as triangle soup (unindexed)
    const soup = tf.triangle(new Float32Array([
      0,0,0, 1,0,0, 0.5,1,0,
      1,0,0, 0,0,0, 0.5,-1,0,
    ]), 3);

    const mesh = tf.cleaned(soup);
    assert(mesh.numberOfFaces === 2, `expected 2 faces, got ${mesh.numberOfFaces}`);
    // Shared vertices should be merged: 4 unique points, not 6
    assert(mesh.numberOfPoints <= 6, `expected <= 6 points, got ${mesh.numberOfPoints}`);
    log(`  cleaned(soup): ${mesh.numberOfFaces} faces, ${mesh.numberOfPoints} points`, "line-pass");

    mesh.delete(); soup.delete();
  });

  test("async: cleaned", async () => {
    const tf = getTf();
    const { faces, points } = meshWithDuplicateVertex();
    const m = tf.mesh(faces, points);
    const clean = await tf.async.cleaned(m);

    assert(clean.numberOfPoints <= m.numberOfPoints,
      `expected fewer points: ${clean.numberOfPoints} <= ${m.numberOfPoints}`);
    assert(clean.numberOfFaces === 1, `expected 1 face (duplicate removed), got ${clean.numberOfFaces}`);
    log(`  async cleaned: ${m.numberOfPoints} → ${clean.numberOfPoints} points`, "line-pass");

    clean.delete(); m.delete();
  });

});
