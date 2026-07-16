import { describe, test, log, assert, approx, getTf } from "./harness.mjs";

// ============================================================================
// Shallow copy: mutation isolation + cache invalidation correctness
//
// These tests verify the contract of mesh.shallowCopy() / pc.shallowCopy():
//  - The copy and the original share the underlying buffers (cheap), but
//    each handle has its own cache slots and own (cleared) transformation.
//  - Mutating geometry on the copy must NOT affect the original.
//  - The copy's lazy caches (tree, normals, topology) must be rebuilt for
//    the copy's geometry, not silently reuse stale state from the original.
// ============================================================================

// Translate a flat [V*3] points buffer by (dx, dy, dz).
function translatePoints(pts, dx, dy, dz) {
  const out = new Float32Array(pts.length);
  for (let i = 0; i < pts.length; i += 3) {
    out[i + 0] = pts[i + 0] + dx;
    out[i + 1] = pts[i + 1] + dy;
    out[i + 2] = pts[i + 2] + dz;
  }
  return out;
}

describe("Mesh shallowCopy: mutation isolation", () => {

  // ==========================================================================
  test("set points on copy does NOT affect original", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const original = m.points;
    const baseline = new Float32Array(original.data);  // snapshot
    original.delete();

    const c = m.shallowCopy();
    const moved = translatePoints(baseline, 100, 0, 0);
    c.points = moved;

    // Original points unchanged
    const after = m.points;
    for (let i = 0; i < baseline.length; i++) {
      assert(after.data[i] === baseline[i],
        `original.points[${i}] changed: was ${baseline[i]}, now ${after.data[i]}`);
    }
    log("  original.points untouched after copy.points = ...", "line-pass");

    // Copy reflects the new positions
    const cp = c.points;
    assert(cp.data[0] === baseline[0] + 100, `copy.points[0] should be translated`);
    log("  copy.points reflects assignment", "line-pass");

    after.delete();
    cp.delete();
    c.delete();
    m.delete();
  });

  // ==========================================================================
  test("set faces on copy does NOT affect original", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const subdivided = tf.boxMesh(2, 2, 2, 2, 2, 2);

    const origFaceCount = m.numberOfFaces;
    const newFaceCount = subdivided.numberOfFaces;
    assert(origFaceCount !== newFaceCount, "subdivided box should have different face count");

    const c = m.shallowCopy();
    c.faces = subdivided.faces;
    c.points = subdivided.points;

    assert(m.numberOfFaces === origFaceCount,
      `original face count changed: ${origFaceCount} -> ${m.numberOfFaces}`);
    assert(c.numberOfFaces === newFaceCount,
      `copy face count not updated: expected ${newFaceCount}, got ${c.numberOfFaces}`);
    log(`  original kept ${origFaceCount} faces, copy has ${c.numberOfFaces}`, "line-pass");

    c.delete();
    subdivided.delete();
    m.delete();
  });

  // ==========================================================================
  test("set transformation on copy does NOT affect original", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    assert(m.transformation === null, "original should start with no transformation");

    const c = m.shallowCopy();
    const t = tf.makeTranslation(50, 0, 0);
    c.transformation = t;

    assert(m.transformation === null,
      "original.transformation should remain null after copy.transformation = ...");
    assert(c.transformation !== null,
      "copy.transformation should not be null after assignment");
    log("  original.transformation untouched", "line-pass");

    t.delete();
    c.delete();
    m.delete();
  });

  // ==========================================================================
  test("clearing transformation on copy does NOT affect original", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const t = tf.makeTranslation(7, 0, 0);
    m.transformation = t;
    t.delete();

    const c = m.shallowCopy();  // copy starts with cleared transformation
    assert(c.transformation === null, "shallowCopy should clear the transformation");

    // Now reassign on original; copy must remain null
    const t2 = tf.makeTranslation(11, 0, 0);
    m.transformation = t2;
    t2.delete();

    assert(c.transformation === null,
      "copy.transformation should still be null after original was reassigned");
    log("  transformation independence preserved both ways", "line-pass");

    c.delete();
    m.delete();
  });

});

describe("Mesh shallowCopy: original is preserved", () => {

  // ==========================================================================
  // Snapshot every observable on the original, take a shallowCopy, then
  // mutate the copy in every way we can — original must be byte-identical
  // to the snapshot at every step.
  test("original retains all data + caches after copy is taken and mutated", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 3, 4);

    // Pose the original
    const t = tf.makeTranslation(1, 2, 3);
    m.transformation = t;
    t.delete();

    // Force every lazy cache to materialize on the original
    const facesA = m.faces;
    const pointsA = m.points;
    const normalsA = m.normals;
    const pointNormalsA = m.pointNormals;
    const faceMembershipA_len = m.faceMembership.length;
    const faceLinkA_len = m.faceLink.length;
    const vertexLinkA_len = m.vertexLink.length;
    const manifoldA_shape = m.manifoldEdgeLink.shape.slice();
    const transformationA = m.transformation;

    // Snapshot raw bytes (so later we can prove ZERO drift)
    const facesSnap = new Int32Array(facesA.data);
    const pointsSnap = new Float32Array(pointsA.data);
    const normalsSnap = new Float32Array(normalsA.data);
    const pointNormalsSnap = new Float32Array(pointNormalsA.data);
    const transformationSnap = new Float32Array(transformationA.data);
    const numFacesA = m.numberOfFaces;
    const numPointsA = m.numberOfPoints;

    facesA.delete();
    pointsA.delete();
    normalsA.delete();
    pointNormalsA.delete();
    transformationA.delete();

    // Helper: assert original matches snapshot
    function checkOriginalIntact(label) {
      assert(m.numberOfFaces === numFacesA,
        `${label}: numberOfFaces drifted (${numFacesA} -> ${m.numberOfFaces})`);
      assert(m.numberOfPoints === numPointsA,
        `${label}: numberOfPoints drifted (${numPointsA} -> ${m.numberOfPoints})`);

      const f = m.faces;
      for (let i = 0; i < facesSnap.length; i++)
        assert(f.data[i] === facesSnap[i],
          `${label}: faces[${i}] drifted (${facesSnap[i]} -> ${f.data[i]})`);
      f.delete();

      const p = m.points;
      for (let i = 0; i < pointsSnap.length; i++)
        assert(p.data[i] === pointsSnap[i],
          `${label}: points[${i}] drifted (${pointsSnap[i]} -> ${p.data[i]})`);
      p.delete();

      const n = m.normals;
      assert(n.shape[0] === numFacesA, `${label}: normals shape drifted`);
      for (let i = 0; i < normalsSnap.length; i++)
        assert(Math.abs(n.data[i] - normalsSnap[i]) < 1e-6,
          `${label}: normals[${i}] drifted`);
      n.delete();

      const pn = m.pointNormals;
      assert(pn.shape[0] === numPointsA, `${label}: pointNormals shape drifted`);
      for (let i = 0; i < pointNormalsSnap.length; i++)
        assert(Math.abs(pn.data[i] - pointNormalsSnap[i]) < 1e-6,
          `${label}: pointNormals[${i}] drifted`);
      pn.delete();

      const fm = m.faceMembership;
      assert(fm.length === faceMembershipA_len,
        `${label}: faceMembership length drifted`);
      fm.delete();

      const fl = m.faceLink;
      assert(fl.length === faceLinkA_len,
        `${label}: faceLink length drifted`);
      fl.delete();

      const vl = m.vertexLink;
      assert(vl.length === vertexLinkA_len,
        `${label}: vertexLink length drifted`);
      vl.delete();

      const mel = m.manifoldEdgeLink;
      assert(mel.shape[0] === manifoldA_shape[0] && mel.shape[1] === manifoldA_shape[1],
        `${label}: manifoldEdgeLink shape drifted`);
      mel.delete();

      const tr = m.transformation;
      assert(tr !== null, `${label}: transformation should still be set`);
      for (let i = 0; i < transformationSnap.length; i++)
        assert(Math.abs(tr.data[i] - transformationSnap[i]) < 1e-6,
          `${label}: transformation[${i}] drifted`);
      tr.delete();
    }

    // -------- Step 1: just take the copy. Original must be intact.
    const c = m.shallowCopy();
    checkOriginalIntact("after shallowCopy()");
    log("  original intact immediately after shallowCopy()", "line-pass");

    // -------- Step 2: mutate the copy's points. Original still intact.
    const sub = tf.boxMesh(2, 3, 4, 2, 2, 2);
    c.points = sub.points;
    checkOriginalIntact("after copy.points = ...");
    log("  original intact after copy.points = ...", "line-pass");

    // -------- Step 3: mutate the copy's faces. Original still intact.
    c.faces = sub.faces;
    checkOriginalIntact("after copy.faces = ...");
    log("  original intact after copy.faces = ...", "line-pass");

    // -------- Step 4: set transformation on the copy. Original still intact.
    const t2 = tf.makeTranslation(50, 0, 0);
    c.transformation = t2;
    t2.delete();
    checkOriginalIntact("after copy.transformation = ...");
    log("  original intact after copy.transformation = ...", "line-pass");

    // -------- Step 5: trigger lazy caches on the copy. Original still intact.
    const cn = c.normals;
    const cfm = c.faceMembership;
    const cfl = c.faceLink;
    const cvl = c.vertexLink;
    const cmel = c.manifoldEdgeLink;
    const cpn = c.pointNormals;
    cn.delete(); cfm.delete(); cfl.delete(); cvl.delete(); cmel.delete(); cpn.delete();
    checkOriginalIntact("after copy lazy caches built");
    log("  original intact after copy built its own caches", "line-pass");

    // -------- Step 6: destroy the copy. Original still intact.
    c.delete();
    checkOriginalIntact("after copy.delete()");
    log("  original intact after copy.delete()", "line-pass");

    sub.delete();
    m.delete();
  });

});

// ============================================================================
// Cache lifecycle template — applied to every lazily-built mesh cache.
//
// For each cache: build on original → shallowCopy inherits it fresh →
// reassign the dependency on the copy → copy goes stale, original stays fresh
// → access cache on copy → copy is fresh again (rebuilt for new geometry) →
// original still fresh (its slot is independent).
// ============================================================================

describe("Mesh cache lifecycle: tree (deps: faces + points)", () => {

  test("buildTree → shallowCopy inherits → mutate copy → rebuild on copy", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const mh = m._handle;

    // 1. Build on original
    m.buildTree();
    assert(mh.is_tree_built() && mh.is_tree_fresh(), "orig: built + fresh");

    // 2. Shallow copy inherits it
    const c = m.shallowCopy();
    const ch = c._handle;
    assert(ch.is_tree_built() && ch.is_tree_fresh(),
      "copy inherits tree built + fresh");

    // 3. Reassign dependency (points) on copy → copy stale, orig fresh
    const cp = c.points;
    c.points = translatePoints(cp.data, 50, 0, 0);
    cp.delete();
    assert(ch.is_tree_built() && !ch.is_tree_fresh(),
      "copy: tree slot still set, but stale");
    assert(mh.is_tree_fresh(), "orig: tree still fresh");

    // 4. Access on copy (via buildTree) → copy fresh again, orig untouched
    c.buildTree();
    assert(ch.is_tree_built() && ch.is_tree_fresh(),
      "copy: tree rebuilt → fresh");
    assert(mh.is_tree_fresh(), "orig: tree still fresh (independent slot)");

    log("  tree lifecycle ok (mutate via points)", "line-pass");
    c.delete();
    m.delete();
  });

  test("same lifecycle, but mutate via faces", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const sub = tf.boxMesh(2, 2, 2, 2, 2, 2);
    m.buildTree();

    const c = m.shallowCopy();
    assert(c._handle.is_tree_fresh(), "copy inherits fresh tree");

    c.faces = sub.faces;
    c.points = sub.points;
    assert(!c._handle.is_tree_fresh(), "copy stale after set faces");
    assert(m._handle.is_tree_fresh(), "orig fresh");

    c.buildTree();
    assert(c._handle.is_tree_fresh(), "copy fresh after rebuild");
    assert(m._handle.is_tree_fresh(), "orig still fresh");

    log("  tree lifecycle ok (mutate via faces)", "line-pass");
    c.delete();
    sub.delete();
    m.delete();
  });

});

describe("Mesh cache lifecycle: normals (deps: faces + points)", () => {

  test("build → shallowCopy → mutate points on copy → access on copy", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const mh = m._handle;

    const n0 = m.normals; n0.delete();
    assert(mh.is_normals_fresh(), "orig: normals fresh");

    const c = m.shallowCopy();
    const ch = c._handle;
    assert(ch.is_normals_fresh(), "copy inherits fresh normals");

    const cp = c.points;
    c.points = translatePoints(cp.data, 0, 0, 0);  // bumps gen
    cp.delete();
    assert(ch.is_normals_built() && !ch.is_normals_fresh(),
      "copy: normals slot still set, stale");
    assert(mh.is_normals_fresh(), "orig: normals still fresh");

    const cn = c.normals;
    assert(ch.is_normals_fresh(), "copy: normals rebuilt → fresh");
    assert(mh.is_normals_fresh(), "orig: still fresh");
    cn.delete();

    log("  normals lifecycle ok", "line-pass");
    c.delete();
    m.delete();
  });

});

describe("Mesh cache lifecycle: point_normals (deps: faces + points)", () => {

  test("build → shallowCopy → mutate points on copy → access on copy", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const mh = m._handle;

    const pn0 = m.pointNormals; pn0.delete();
    assert(mh.is_point_normals_fresh(), "orig: point_normals fresh");

    const c = m.shallowCopy();
    const ch = c._handle;
    assert(ch.is_point_normals_fresh(), "copy inherits fresh point_normals");

    const cp = c.points;
    c.points = translatePoints(cp.data, 0, 0, 0);
    cp.delete();
    assert(ch.is_point_normals_built() && !ch.is_point_normals_fresh(),
      "copy: point_normals stale");
    assert(mh.is_point_normals_fresh(), "orig: point_normals fresh");

    const cpn = c.pointNormals;
    assert(ch.is_point_normals_fresh(), "copy: rebuilt → fresh");
    assert(mh.is_point_normals_fresh(), "orig: still fresh");
    cpn.delete();

    log("  point_normals lifecycle ok", "line-pass");
    c.delete();
    m.delete();
  });

});

describe("Mesh cache lifecycle: face_membership (deps: faces)", () => {

  test("build → shallowCopy → mutate faces on copy → access on copy", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const sub = tf.boxMesh(2, 2, 2, 2, 2, 2);
    const mh = m._handle;

    const fm0 = m.faceMembership; fm0.delete();
    assert(mh.is_face_membership_fresh(), "orig fresh");

    const c = m.shallowCopy();
    const ch = c._handle;
    assert(ch.is_face_membership_fresh(), "copy inherits fresh");

    c.faces = sub.faces;
    c.points = sub.points;
    assert(ch.is_face_membership_built() && !ch.is_face_membership_fresh(),
      "copy stale after set faces");
    assert(mh.is_face_membership_fresh(), "orig still fresh");

    const cfm = c.faceMembership;
    assert(ch.is_face_membership_fresh(), "copy fresh after rebuild");
    assert(mh.is_face_membership_fresh(), "orig still fresh");
    cfm.delete();

    log("  face_membership lifecycle ok", "line-pass");
    c.delete();
    sub.delete();
    m.delete();
  });

});

describe("Mesh cache lifecycle: face_link (deps: faces)", () => {

  test("build → shallowCopy → mutate faces on copy → access on copy", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const sub = tf.boxMesh(2, 2, 2, 2, 2, 2);
    const mh = m._handle;

    const fl0 = m.faceLink; fl0.delete();
    assert(mh.is_face_link_fresh(), "orig fresh");

    const c = m.shallowCopy();
    const ch = c._handle;
    assert(ch.is_face_link_fresh(), "copy inherits fresh");

    c.faces = sub.faces;
    c.points = sub.points;
    assert(ch.is_face_link_built() && !ch.is_face_link_fresh(), "copy stale");
    assert(mh.is_face_link_fresh(), "orig still fresh");

    const cfl = c.faceLink;
    assert(ch.is_face_link_fresh(), "copy fresh after rebuild");
    assert(mh.is_face_link_fresh(), "orig still fresh");
    cfl.delete();

    log("  face_link lifecycle ok", "line-pass");
    c.delete();
    sub.delete();
    m.delete();
  });

});

describe("Mesh cache lifecycle: vertex_link (deps: faces)", () => {

  test("build → shallowCopy → mutate faces on copy → access on copy", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const sub = tf.boxMesh(2, 2, 2, 2, 2, 2);
    const mh = m._handle;

    const vl0 = m.vertexLink; vl0.delete();
    assert(mh.is_vertex_link_fresh(), "orig fresh");

    const c = m.shallowCopy();
    const ch = c._handle;
    assert(ch.is_vertex_link_fresh(), "copy inherits fresh");

    c.faces = sub.faces;
    c.points = sub.points;
    assert(ch.is_vertex_link_built() && !ch.is_vertex_link_fresh(), "copy stale");
    assert(mh.is_vertex_link_fresh(), "orig still fresh");

    const cvl = c.vertexLink;
    assert(ch.is_vertex_link_fresh(), "copy fresh after rebuild");
    assert(mh.is_vertex_link_fresh(), "orig still fresh");
    cvl.delete();

    log("  vertex_link lifecycle ok", "line-pass");
    c.delete();
    sub.delete();
    m.delete();
  });

});

describe("Mesh cache lifecycle: manifold_edge_link (deps: faces)", () => {

  test("build → shallowCopy → mutate faces on copy → access on copy", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    const sub = tf.boxMesh(2, 2, 2, 2, 2, 2);
    const mh = m._handle;

    const mel0 = m.manifoldEdgeLink; mel0.delete();
    assert(mh.is_manifold_edge_link_fresh(), "orig fresh");

    const c = m.shallowCopy();
    const ch = c._handle;
    assert(ch.is_manifold_edge_link_fresh(), "copy inherits fresh");

    c.faces = sub.faces;
    c.points = sub.points;
    assert(ch.is_manifold_edge_link_built() && !ch.is_manifold_edge_link_fresh(),
      "copy stale");
    assert(mh.is_manifold_edge_link_fresh(), "orig still fresh");

    const cmel = c.manifoldEdgeLink;
    assert(ch.is_manifold_edge_link_fresh(), "copy fresh after rebuild");
    assert(mh.is_manifold_edge_link_fresh(), "orig still fresh");
    cmel.delete();

    log("  manifold_edge_link lifecycle ok", "line-pass");
    c.delete();
    sub.delete();
    m.delete();
  });

});

describe("Mesh shallowCopy: cache invalidation correctness", () => {

  // ==========================================================================
  // After setting new points on the copy, queries against the copy must
  // reflect the NEW geometry — proving the spatial tree was rebuilt for the
  // copy's points, not silently reused from the original.
  test("copy's tree rebuilds for new points (distance reflects new position)", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);  // centered at origin, ±1 extent

    // Build the tree on the original up front.
    m.buildTree();

    const c = m.shallowCopy();
    const orig = m.points;
    const moved = translatePoints(orig.data, 100, 0, 0);
    orig.delete();
    c.points = moved;

    // Copy now lives near (100,0,0). Distance from a probe at (100,0,0)
    // should be ≈ 0 (closest face within the box). If c reused the original
    // tree (bug), the distance would be ~99.
    const dCopy = tf.distance(c, tf.point(100, 0, 0));
    assert(dCopy < 2.0,
      `copy.distance from (100,0,0) should be ~0, got ${dCopy} (tree not rebuilt?)`);
    log(`  copy distance @ (100,0,0) = ${dCopy.toFixed(3)} (tree rebuilt OK)`, "line-pass");

    // Original is unchanged → distance from (0,0,0) is ≈ 0.
    const dOrig = tf.distance(m, tf.point(0, 0, 0));
    assert(dOrig < 2.0,
      `original.distance from (0,0,0) should be ~0, got ${dOrig}`);
    log(`  original distance @ (0,0,0) = ${dOrig.toFixed(3)} (unchanged)`, "line-pass");

    // Cross-check: original from (100,0,0) should be ~99, copy from (0,0,0) ~99
    const dOrigFar = tf.distance(m, tf.point(100, 0, 0));
    const dCopyFar = tf.distance(c, tf.point(0, 0, 0));
    assert(dOrigFar > 95 && dOrigFar < 105,
      `original.distance from (100,0,0) should be ~99, got ${dOrigFar}`);
    assert(dCopyFar > 95 && dCopyFar < 105,
      `copy.distance from (0,0,0) should be ~99, got ${dCopyFar}`);
    log(`  cross: orig@far=${dOrigFar.toFixed(2)}, copy@orig=${dCopyFar.toFixed(2)}`, "line-pass");

    c.delete();
    m.delete();
  });

  // ==========================================================================
  // After setting new faces on the copy, derived caches (normals, topology)
  // must be rebuilt for the new face count.
  test("copy's normals rebuild for new faces", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);

    // Touch normals on the original to force a build there.
    const nOrig = m.normals;
    const origCount = nOrig.shape[0];
    nOrig.delete();

    // Subdivide the copy → different face count.
    const sub = tf.boxMesh(2, 2, 2, 2, 2, 2);
    const c = m.shallowCopy();
    c.faces = sub.faces;
    c.points = sub.points;

    const nCopy = c.normals;
    assert(nCopy.shape[0] === sub.numberOfFaces,
      `copy.normals should have ${sub.numberOfFaces} entries, got ${nCopy.shape[0]}`);
    nCopy.delete();
    log(`  copy normals rebuilt for new face count (${sub.numberOfFaces})`, "line-pass");

    // Original's normals should still match its own face count.
    const nOrig2 = m.normals;
    assert(nOrig2.shape[0] === origCount,
      `original.normals should still have ${origCount} entries, got ${nOrig2.shape[0]}`);
    nOrig2.delete();
    log(`  original normals unchanged (${origCount} entries)`, "line-pass");

    c.delete();
    sub.delete();
    m.delete();
  });

  // ==========================================================================
  test("copy's topology caches rebuild for new faces", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);

    // Force build on original.
    const fmOrig = m.faceMembership;
    const origPoints = fmOrig.length;
    fmOrig.delete();

    const sub = tf.boxMesh(2, 2, 2, 2, 2, 2);
    const c = m.shallowCopy();
    c.faces = sub.faces;
    c.points = sub.points;

    const fmCopy = c.faceMembership;
    assert(fmCopy.length === sub.numberOfPoints,
      `copy.faceMembership should have ${sub.numberOfPoints} entries, got ${fmCopy.length}`);
    fmCopy.delete();

    const fmOrig2 = m.faceMembership;
    assert(fmOrig2.length === origPoints,
      `original.faceMembership should still have ${origPoints} entries, got ${fmOrig2.length}`);
    fmOrig2.delete();
    log(`  faceMembership: orig=${origPoints}, copy=${sub.numberOfPoints}`, "line-pass");

    c.delete();
    sub.delete();
    m.delete();
  });

});

describe("PointCloud cache lifecycle: tree (deps: points)", () => {

  test("build → shallowCopy inherits → mutate copy → rebuild on copy", () => {
    const tf = getTf();
    const data = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    const pc = tf.pointCloud(data);
    const ph = pc._handle;

    pc.buildTree();
    assert(ph.is_tree_fresh(), "orig: tree fresh");

    const c = pc.shallowCopy();
    const ch = c._handle;
    assert(ch.is_tree_fresh(), "copy inherits fresh tree");

    c.points = translatePoints(data, 50, 0, 0);
    assert(ch.is_tree_built() && !ch.is_tree_fresh(), "copy: tree stale");
    assert(ph.is_tree_fresh(), "orig: still fresh");

    c.buildTree();
    assert(ch.is_tree_fresh(), "copy: rebuilt → fresh");
    assert(ph.is_tree_fresh(), "orig: still fresh");

    log("  pc tree lifecycle ok", "line-pass");
    c.delete();
    pc.delete();
  });

});

describe("PointCloud shallowCopy: mutation isolation + cache invalidation", () => {

  // ==========================================================================
  test("original retains data after copy is taken and mutated", () => {
    const tf = getTf();
    const data = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    const pc = tf.pointCloud(data);

    const t = tf.makeTranslation(7, 0, 0);
    pc.transformation = t;
    t.delete();

    const ptsSnap = new Float32Array(pc.points.data);
    const trSnap = new Float32Array(pc.transformation.data);
    const numA = pc.numberOfPoints;

    function checkPcIntact(label) {
      assert(pc.numberOfPoints === numA, `${label}: numberOfPoints drifted`);
      const p = pc.points;
      for (let i = 0; i < ptsSnap.length; i++)
        assert(p.data[i] === ptsSnap[i], `${label}: points[${i}] drifted`);
      p.delete();
      const tr = pc.transformation;
      for (let i = 0; i < trSnap.length; i++)
        assert(Math.abs(tr.data[i] - trSnap[i]) < 1e-6,
          `${label}: transformation[${i}] drifted`);
      tr.delete();
    }

    const c = pc.shallowCopy();
    checkPcIntact("after shallowCopy()");

    c.points = translatePoints(data, 100, 0, 0);
    checkPcIntact("after copy.points = ...");

    const t2 = tf.makeTranslation(99, 0, 0);
    c.transformation = t2;
    t2.delete();
    checkPcIntact("after copy.transformation = ...");

    c.delete();
    checkPcIntact("after copy.delete()");

    log("  pc original intact through every copy mutation", "line-pass");
    pc.delete();
  });

  // ==========================================================================
  test("set points on copy does NOT affect original", () => {
    const tf = getTf();
    const data = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const pc = tf.pointCloud(data);

    const c = pc.shallowCopy();
    c.points = new Float32Array([10, 0, 0, 11, 0, 0, 10, 1, 0]);

    const orig = pc.points;
    assert(orig.data[0] === 0, "original pc.points[0] changed");
    assert(orig.data[3] === 1, "original pc.points[3] changed");
    orig.delete();

    const cp = c.points;
    assert(cp.data[0] === 10, "copy pc.points[0] not updated");
    cp.delete();
    log("  pc original points untouched, copy reflects assignment", "line-pass");

    c.delete();
    pc.delete();
  });

  // ==========================================================================
  test("copy's tree rebuilds for new points", () => {
    const tf = getTf();
    const data = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    const pc = tf.pointCloud(data);
    pc.buildTree();

    const c = pc.shallowCopy();
    c.points = translatePoints(data, 100, 0, 0);

    // Copy now near (100,0,0)
    const dCopy = tf.distance(c, tf.point(100, 0, 0));
    assert(dCopy < 2.0,
      `pc copy distance from (100,0,0) should be ~0, got ${dCopy}`);

    // Original near origin
    const dOrig = tf.distance(pc, tf.point(0, 0, 0));
    assert(dOrig < 2.0,
      `pc original distance from (0,0,0) should be ~0, got ${dOrig}`);

    log(`  pc copy=${dCopy.toFixed(3)}, original=${dOrig.toFixed(3)}`, "line-pass");

    c.delete();
    pc.delete();
  });

});

describe("NDArray shallowCopy: shared buffer, independent metadata", () => {

  test("values shared, shape independent", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const c = a.shallowCopy();

    assert(c.shape.length === 2 && c.shape[0] === 2 && c.shape[1] === 3,
      `copy keeps shape, got [${c.shape}]`);

    // in-place mutation through the copy is visible in the original
    c.data[0] = 42;
    assert(a.data[0] === 42, "buffer is shared");

    // reshaping the copy leaves the original's shape untouched
    const flat = c.reshape([6]);
    assert(a.shape.length === 2, "original shape independent");
    assert(flat.data[5] === 6, "reshaped view sees same data");
    flat.delete();

    // a copy is a distinct handle, not an alias: the original deletes
    // first and the copy stays valid over the shared storage
    a.delete();
    assert(c.data[0] === 42 && c.data[5] === 6,
      "copy valid after original deleted");
    assert(c.shape[0] === 2 && c.shape[1] === 3,
      "copy metadata valid after original deleted");

    log("  ndarray shallowCopy: shared values, independent lifetime", "line-pass");
    c.delete();
  });

  test("clone stays deep by contrast", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const deep = a.clone();
    deep.data[0] = 99;
    assert(a.data[0] === 1, "clone does not share the buffer");
    deep.delete();
    a.delete();
  });

});

describe("Primitive shallowCopy: preserves type", () => {

  test("point copy is a point over the same buffer", () => {
    const tf = getTf();
    const p = tf.point(1, 2, 3);
    const c = p.shallowCopy();

    assert(c.type === "point", `type preserved, got ${c.type}`);
    assert(c.shape.length === p.shape.length, "shape preserved");

    c.data[0] = 7;
    assert(p.data[0] === 7, "buffer is shared");

    // original deletes first; the copy stays valid over shared storage
    p.delete();
    assert(c.data[0] === 7 && c.data[2] === 3,
      "copy valid after original deleted");

    log("  primitive shallowCopy: type + buffer + lifetime", "line-pass");
    c.delete();
  });

});

describe("Curves shallowCopy: shared buffers, independent lifetime", () => {

  test("copy shares paths and points; deletes independently", () => {
    const tf = getTf();
    const s0 = tf.sphereMesh(1, 16, 16);
    const s1 = tf.sphereMesh(1, 16, 16);
    s1.transformation = tf.makeTranslation(0.8, 0, 0);
    const curves = tf.intersectionCurves(s0, s1);
    assert(curves.length > 0, "fixture produces curves");

    const c = curves.shallowCopy();
    assert(c.length === curves.length, "same path count");

    const p0 = curves.points;
    const p1 = c.points;
    assert(p0.shape[0] === p1.shape[0], "same point count");
    assert(p0.data[0] === p1.data[0], "same point data");
    p0.delete();
    p1.delete();

    // original deletes first; the copy stays valid over shared storage
    curves.delete();
    const p2 = c.points;
    assert(p2.shape[0] > 0, "copy valid after original deleted");
    p2.delete();

    log("  curves shallowCopy: shared storage, refcounted lifetime", "line-pass");
    c.delete();
    s0.delete();
    s1.delete();
  });

});
