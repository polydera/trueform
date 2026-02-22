import { describe, test, log, assert, getTf } from "./harness.mjs";

// Two triangles sharing an edge (0-1)
//   2
//  / \
// 0---1
//  \ /
//   3
function twoTriangles() {
  return {
    faces: new Int32Array([0, 1, 2, 0, 3, 1]),
    points: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0.5, 1, 0,
      0.5, -1, 0,
    ]),
  };
}

// Tetrahedron: 4 faces, 4 vertices — closed manifold
function tetrahedron() {
  return {
    faces: new Int32Array([
      0, 1, 2,
      0, 3, 1,
      1, 3, 2,
      0, 2, 3,
    ]),
    points: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0.5, 1, 0,
      0.5, 0.5, 1,
    ]),
  };
}

describe("Mesh", () => {

  test("Create mesh and read back", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);

    assert(m.numberOfFaces === 2, `expected 2 faces, got ${m.numberOfFaces}`);
    log(`  numberOfFaces = ${m.numberOfFaces}`, "line-pass");

    assert(m.numberOfPoints === 4, `expected 4 points, got ${m.numberOfPoints}`);
    log(`  numberOfPoints = ${m.numberOfPoints}`, "line-pass");

    const fh = m.faces;
    const fd = fh.data;
    assert(fd.length === 6, `expected 6 face indices, got ${fd.length}`);
    assert(fd[0] === 0 && fd[1] === 1 && fd[2] === 2, "face 0 mismatch");
    assert(fd[3] === 0 && fd[4] === 3 && fd[5] === 1, "face 1 mismatch");
    log("  faces data matches input", "line-pass");

    const ph = m.points;
    const pd = ph.data;
    assert(pd.length === 12, `expected 12 point coords, got ${pd.length}`);
    log("  points data matches input", "line-pass");

    fh.delete();
    ph.delete();
    m.delete();
  });

  test("Independent handle ownership", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);

    const fh = m.faces;
    const ph = m.points;

    // Destroy mesh — handles should still be valid
    m.delete();

    assert(fh.data.length === 6, "faces data still accessible");
    log("  handles survive mesh.delete()", "line-pass");

    fh.delete();
    ph.delete();
  });

  test("Shared view", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const v = m.sharedView();

    assert(v.numberOfFaces === 2, "shared view has same face count");
    assert(v.numberOfPoints === 4, "shared view has same point count");

    const vf = v.faces;
    assert(vf.data[0] === 0, "shared view faces match");
    log("  sharedView shares data", "line-pass");

    vf.delete();
    v.delete();
    m.delete();
  });

  test("Face membership", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const fm = m.faceMembership;

    assert(fm.length === 4, `expected 4 blocks (one per vertex), got ${fm.length}`);
    log(`  fm.length = ${fm.length} (4 vertices)`, "line-pass");

    const offsets = fm.offsets;
    const data = fm.data;
    log(`  fm offsets length = ${offsets.length}`, "line-pass");
    log(`  fm data length = ${data.length}`, "line-pass");

    // Vertices 0 and 1 are shared by both faces
    // Check total membership count
    let totalMemberships = 0;
    for (let i = 0; i < fm.length; i++) {
      const block = fm.get(i);
      totalMemberships += block.length;
    }
    // 2 triangles * 3 vertices = 6 total memberships
    assert(totalMemberships === 6, `expected 6 total memberships, got ${totalMemberships}`);
    log(`  total memberships = ${totalMemberships}`, "line-pass");

    offsets.delete();
    data.delete();
    fm.delete();
    m.delete();
  });

  test("Manifold edge link", () => {
    const tf = getTf();
    const { faces, points } = tetrahedron();
    const m = tf.mesh(faces, points);
    const mel = m.manifoldEdgeLink;

    assert(mel.length === 12, `expected 12 mel entries (4*3), got ${mel.length}`);
    const d = mel.data;
    log(`  mel shape = [${mel.shape}]`, "line-pass");

    // Tetrahedron is closed — every edge has a neighbor (no -1 boundary)
    let hasBoundary = false;
    for (let i = 0; i < d.length; i++) {
      if (d[i] === -1) hasBoundary = true;
    }
    assert(!hasBoundary, "tetrahedron should have no boundary edges");
    log("  no boundary edges on closed tetrahedron", "line-pass");

    mel.delete();
    m.delete();
  });

  test("Manifold edge link (open mesh)", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const mel = m.manifoldEdgeLink;

    const d = mel.data;
    let boundaryCount = 0;
    let neighborCount = 0;
    for (let i = 0; i < d.length; i++) {
      if (d[i] === -1) boundaryCount++;
      else if (d[i] >= 0) neighborCount++;
    }
    assert(boundaryCount > 0, "open mesh should have boundary edges");
    assert(neighborCount > 0, "should have at least one shared edge");
    log(`  boundary edges: ${boundaryCount}, shared edges: ${neighborCount}`, "line-pass");

    mel.delete();
    m.delete();
  });

  test("Face link", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const fl = m.faceLink;

    assert(fl.length === 2, `expected 2 blocks (one per face), got ${fl.length}`);
    log(`  fl.length = ${fl.length}`, "line-pass");

    // Both faces share vertices 0 and 1, so they should be adjacent
    const block0 = fl.get(0);
    let found1 = false;
    for (let i = 0; i < block0.length; i++) {
      if (block0[i] === 1) found1 = true;
    }
    assert(found1, "face 0 should be adjacent to face 1");
    log("  face 0 adjacent to face 1", "line-pass");

    fl.delete();
    m.delete();
  });

  test("Vertex link", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const vl = m.vertexLink;

    assert(vl.length === 4, `expected 4 blocks (one per vertex), got ${vl.length}`);
    log(`  vl.length = ${vl.length}`, "line-pass");

    // Vertex 0 connects to vertices 1, 2, 3
    const block0 = vl.get(0);
    assert(block0.length === 3, `vertex 0 should have 3 neighbors, got ${block0.length}`);
    log(`  vertex 0 neighbors: [${Array.from(block0)}]`, "line-pass");

    vl.delete();
    m.delete();
  });

  test("Generation tracking (faces setter invalidates cache)", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);

    // Build topology
    const fm1 = m.faceMembership;
    assert(fm1.length === 4, "initial fm has 4 blocks");
    fm1.delete();

    // Replace faces with single triangle
    m.faces = new Int32Array([0, 1, 2]);
    assert(m.numberOfFaces === 1, "now 1 face");

    // Topology should rebuild
    const fm2 = m.faceMembership;
    assert(fm2.length === 4, "fm still has 4 blocks (4 vertices in points)");

    // But membership counts change: only 3 memberships now
    let totalMemberships = 0;
    for (let i = 0; i < fm2.length; i++) {
      const block = fm2.get(i);
      totalMemberships += block.length;
    }
    assert(totalMemberships === 3, `expected 3 memberships after faces setter, got ${totalMemberships}`);
    log("  topology rebuilt after faces setter", "line-pass");

    fm2.delete();
    m.delete();
  });

});
