import { describe, test, log, assert, getTf } from "./harness.mjs";

// ============================================================================
// Helpers (local copies so this file is self-contained)
// ============================================================================

function seededRandom(count, seed) {
  const out = new Float32Array(count);
  let s = seed >>> 0;
  for (let i = 0; i < count; i++) {
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
    out[i] = (s >>> 0) / 4294967296;
  }
  return out;
}
function randomPoints(count, seed) { return seededRandom(count * 3, seed); }

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

// ============================================================================
// Async precompute: cache must land on the JS-visible original.
//
// Each tf.async.compute*() / tf.async.buildTree() call dispatches work to a
// TBB worker. The worker must populate the cache on the same mesh_data the
// JS handle owns — otherwise a subsequent sync access on the original would
// rebuild from scratch.
//
// Inspectors (is_X_built / is_X_fresh on _handle) make this deterministic:
// no timing comparisons.
// ============================================================================

describe("Async precompute populates the original mesh", () => {

  test("async.buildTree", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    assert(!m._handle.is_tree_built(), "fresh: tree not built");

    await tf.async.buildTree(m);

    assert(m._handle.is_tree_built() && m._handle.is_tree_fresh(),
      "after async.buildTree: tree built + fresh on original");
    log("  async.buildTree → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computeNormals", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    assert(!m._handle.is_normals_built(), "fresh: normals not built");

    const n = await tf.async.computeNormals(m);
    assert(m._handle.is_normals_built() && m._handle.is_normals_fresh(),
      "after async.computeNormals: built + fresh on original");
    assert(n.shape[0] === m.numberOfFaces, "returned normals have correct shape");
    n.delete();
    log("  async.computeNormals → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computePointNormals", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    assert(!m._handle.is_point_normals_built(), "fresh: point_normals not built");

    const pn = await tf.async.computePointNormals(m);
    assert(m._handle.is_point_normals_built() &&
           m._handle.is_point_normals_fresh(),
      "after async.computePointNormals: built + fresh on original");
    assert(pn.shape[0] === m.numberOfPoints,
      "returned point_normals have correct shape");
    pn.delete();
    log("  async.computePointNormals → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computeFaceMembership", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    assert(!m._handle.is_face_membership_built(),
      "fresh: face_membership not built");

    const fm = await tf.async.computeFaceMembership(m);
    assert(m._handle.is_face_membership_built() &&
           m._handle.is_face_membership_fresh(),
      "after async.computeFaceMembership: built + fresh on original");
    fm.delete();
    log("  async.computeFaceMembership → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computeManifoldEdgeLink", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    assert(!m._handle.is_manifold_edge_link_built(),
      "fresh: manifold_edge_link not built");

    const mel = await tf.async.computeManifoldEdgeLink(m);
    assert(m._handle.is_manifold_edge_link_built() &&
           m._handle.is_manifold_edge_link_fresh(),
      "after async.computeManifoldEdgeLink: built + fresh on original");
    mel.delete();
    log("  async.computeManifoldEdgeLink → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computeFaceLink", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    assert(!m._handle.is_face_link_built(), "fresh: face_link not built");

    const fl = await tf.async.computeFaceLink(m);
    assert(m._handle.is_face_link_built() && m._handle.is_face_link_fresh(),
      "after async.computeFaceLink: built + fresh on original");
    fl.delete();
    log("  async.computeFaceLink → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computeVertexLink", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2);
    assert(!m._handle.is_vertex_link_built(), "fresh: vertex_link not built");

    const vl = await tf.async.computeVertexLink(m);
    assert(m._handle.is_vertex_link_built() && m._handle.is_vertex_link_fresh(),
      "after async.computeVertexLink: built + fresh on original");
    vl.delete();
    log("  async.computeVertexLink → built + fresh on original", "line-pass");
    m.delete();
  });

});

describe("Async precompute populates the original PointCloud", () => {

  test("async.buildTree on PointCloud", async () => {
    const tf = getTf();
    const pc = tf.pointCloud(new Float32Array([0,0,0, 1,0,0, 0,1,0, 1,1,0]));
    assert(!pc._handle.is_tree_built(), "fresh pc: tree not built");

    await tf.async.buildTree(pc);

    assert(pc._handle.is_tree_built() && pc._handle.is_tree_fresh(),
      "after async.buildTree(pc): built + fresh on original");
    log("  async.buildTree(pc) → built + fresh on original", "line-pass");
    pc.delete();
  });

});

// ============================================================================
// Async spatial queries must leave the tree built on the forms they used.
// These functions don't advertise themselves as precomputers — they build the
// tree as a side effect of the query. After the mesh_data shared_ptr refactor,
// that side effect is visible on the JS-visible handle.
// ============================================================================

describe("Async spatial queries leave tree built on forms", () => {

  test("async.distance (mesh × primitive) → mesh tree built+fresh", async () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const p = tf.point(0.5, 0, 2);

    assert(!m._handle.is_tree_built(), "fresh: tree not built");
    await tf.async.distance(m, p);
    assert(m._handle.is_tree_built() && m._handle.is_tree_fresh(),
      "after async.distance(mesh, p): tree built + fresh on mesh");
    log("  async.distance(mesh, p) → mesh tree built+fresh", "line-pass");

    p.delete(); m.delete();
  });

  test("async.distance (mesh × mesh) → both trees built+fresh", async () => {
    const tf = getTf();
    const a = tf.mesh(twoTriangles().faces, twoTriangles().points);
    const b = tf.mesh(twoTriangles().faces, twoTriangles().points);

    assert(!a._handle.is_tree_built() && !b._handle.is_tree_built(),
      "fresh: neither tree built");

    await tf.async.distance(a, b);

    assert(a._handle.is_tree_built() && a._handle.is_tree_fresh(),
      "after async.distance(a, b): a tree built + fresh");
    assert(b._handle.is_tree_built() && b._handle.is_tree_fresh(),
      "after async.distance(a, b): b tree built + fresh");
    log("  async.distance(mesh, mesh) → both trees built+fresh", "line-pass");

    a.delete(); b.delete();
  });

  test("async.intersects (mesh × mesh) → both trees built+fresh", async () => {
    const tf = getTf();
    const a = tf.mesh(twoTriangles().faces, twoTriangles().points);
    const b = tf.mesh(twoTriangles().faces, twoTriangles().points);

    await tf.async.intersects(a, b);

    assert(a._handle.is_tree_fresh(), "a tree fresh");
    assert(b._handle.is_tree_fresh(), "b tree fresh");
    log("  async.intersects(mesh, mesh) → both trees fresh", "line-pass");

    a.delete(); b.delete();
  });

  test("async.neighborSearch (mesh × point) → mesh tree built+fresh", async () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const q = tf.point(0.5, 0.5, 1);

    assert(!m._handle.is_tree_built(), "fresh: tree not built");
    const r = await tf.async.neighborSearch(m, q);

    assert(m._handle.is_tree_built() && m._handle.is_tree_fresh(),
      "after async.neighborSearch: tree built + fresh on mesh");
    log("  async.neighborSearch → mesh tree built+fresh", "line-pass");

    r.point.delete(); q.delete(); m.delete();
  });

  test("async.rayCast (ray × mesh) → mesh tree built+fresh", async () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const ray = tf.ray(tf.point(0.5, 0.5, 2), tf.vector(0, 0, -1));

    assert(!m._handle.is_tree_built(), "fresh: tree not built");
    const res = await tf.async.rayCast(ray, m);
    assert(res.hit === true, "sanity: ray should hit");

    assert(m._handle.is_tree_built() && m._handle.is_tree_fresh(),
      "after async.rayCast: tree built + fresh on mesh");
    log("  async.rayCast → mesh tree built+fresh", "line-pass");

    ray.delete(); m.delete();
  });

  test("async.distance2 (batch) → mesh tree built+fresh", async () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const pts = tf.point(new Float32Array([0,0,1, 0.5,0.5,2, 1,1,3]), 3);

    assert(!m._handle.is_tree_built(), "fresh: tree not built");
    const d = await tf.async.distance2(m, pts);

    assert(m._handle.is_tree_built() && m._handle.is_tree_fresh(),
      "after batch async.distance2: mesh tree built + fresh");
    log("  async.distance2 (batch) → mesh tree built+fresh", "line-pass");

    d.delete(); pts.delete(); m.delete();
  });

});

// ============================================================================
// Same precompute matrix on float64 Mesh / PointCloud. dispatch_ensure and
// dispatch_ensure_pc need float64 variants registered on the C++ side, plus
// dtype branching in the TS wrappers. These tests currently fail with
// "Expected NativeFloat32Mesh" until that's in place.
// ============================================================================

function twoTrianglesF64() {
  return {
    faces: new Int32Array([0, 1, 2, 0, 3, 1]),
    points: new Float64Array([
      0, 0, 0,
      1, 0, 0,
      0.5, 1, 0,
      0.5, -1, 0,
    ]),
  };
}

describe("Async precompute populates the original mesh (float64)", () => {

  test("async.buildTree (float64)", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    assert(m.dtype === "float64", "sanity: mesh dtype is float64");
    assert(!m._handle.is_tree_built(), "fresh: tree not built");

    await tf.async.buildTree(m);

    assert(m._handle.is_tree_built() && m._handle.is_tree_fresh(),
      "after async.buildTree (float64): tree built + fresh on original");
    log("  async.buildTree (float64) → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computeNormals (float64)", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    assert(!m._handle.is_normals_built(), "fresh: normals not built");

    const n = await tf.async.computeNormals(m);
    assert(m._handle.is_normals_built() && m._handle.is_normals_fresh(),
      "after async.computeNormals (float64): built + fresh on original");
    assert(n.shape[0] === m.numberOfFaces, "returned normals have correct shape");
    n.delete();
    log("  async.computeNormals (float64) → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computePointNormals (float64)", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    assert(!m._handle.is_point_normals_built(), "fresh: point_normals not built");

    const pn = await tf.async.computePointNormals(m);
    assert(m._handle.is_point_normals_built() &&
           m._handle.is_point_normals_fresh(),
      "after async.computePointNormals (float64): built + fresh on original");
    assert(pn.shape[0] === m.numberOfPoints,
      "returned point_normals have correct shape");
    pn.delete();
    log("  async.computePointNormals (float64) → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computeFaceMembership (float64)", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    assert(!m._handle.is_face_membership_built(),
      "fresh: face_membership not built");

    const fm = await tf.async.computeFaceMembership(m);
    assert(m._handle.is_face_membership_built() &&
           m._handle.is_face_membership_fresh(),
      "after async.computeFaceMembership (float64): built + fresh on original");
    fm.delete();
    log("  async.computeFaceMembership (float64) → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computeManifoldEdgeLink (float64)", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    assert(!m._handle.is_manifold_edge_link_built(),
      "fresh: manifold_edge_link not built");

    const mel = await tf.async.computeManifoldEdgeLink(m);
    assert(m._handle.is_manifold_edge_link_built() &&
           m._handle.is_manifold_edge_link_fresh(),
      "after async.computeManifoldEdgeLink (float64): built + fresh on original");
    mel.delete();
    log("  async.computeManifoldEdgeLink (float64) → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computeFaceLink (float64)", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    assert(!m._handle.is_face_link_built(), "fresh: face_link not built");

    const fl = await tf.async.computeFaceLink(m);
    assert(m._handle.is_face_link_built() && m._handle.is_face_link_fresh(),
      "after async.computeFaceLink (float64): built + fresh on original");
    fl.delete();
    log("  async.computeFaceLink (float64) → built + fresh on original", "line-pass");
    m.delete();
  });

  test("async.computeVertexLink (float64)", async () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    assert(!m._handle.is_vertex_link_built(), "fresh: vertex_link not built");

    const vl = await tf.async.computeVertexLink(m);
    assert(m._handle.is_vertex_link_built() && m._handle.is_vertex_link_fresh(),
      "after async.computeVertexLink (float64): built + fresh on original");
    vl.delete();
    log("  async.computeVertexLink (float64) → built + fresh on original", "line-pass");
    m.delete();
  });

});

describe("Mesh sync lazy accessors on float64", () => {
  // The Mesh getters call C++ wasm_mesh<double> methods directly. These are
  // separate from dispatch_ensure: if Float64Mesh embind registers the
  // methods, sync access works regardless of async wiring. Lock the contract
  // so a future binding regression is caught.

  test("mesh.normals (float64) builds + has correct shape", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    const n = m.normals;
    assert(n.dtype === "float64", `normals dtype ${n.dtype}`);
    assert(n.shape[0] === m.numberOfFaces, "normals shape matches face count");
    assert(m._handle.is_normals_built() && m._handle.is_normals_fresh(),
      "normals built + fresh after sync access");
    log("  mesh.normals (float64) → built + fresh", "line-pass");
    n.delete(); m.delete();
  });

  test("mesh.pointNormals (float64) builds + has correct shape", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    const pn = m.pointNormals;
    assert(pn.dtype === "float64", `pointNormals dtype ${pn.dtype}`);
    assert(pn.shape[0] === m.numberOfPoints, "pointNormals shape matches point count");
    assert(m._handle.is_point_normals_built() && m._handle.is_point_normals_fresh(),
      "pointNormals built + fresh after sync access");
    log("  mesh.pointNormals (float64) → built + fresh", "line-pass");
    pn.delete(); m.delete();
  });

  test("mesh.faceMembership (float64) builds", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    const fm = m.faceMembership;
    assert(m._handle.is_face_membership_built() && m._handle.is_face_membership_fresh(),
      "faceMembership built + fresh after sync access");
    log("  mesh.faceMembership (float64) → built + fresh", "line-pass");
    fm.delete(); m.delete();
  });

  test("mesh.manifoldEdgeLink (float64) builds", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    const mel = m.manifoldEdgeLink;
    assert(m._handle.is_manifold_edge_link_built() && m._handle.is_manifold_edge_link_fresh(),
      "manifoldEdgeLink built + fresh after sync access");
    log("  mesh.manifoldEdgeLink (float64) → built + fresh", "line-pass");
    mel.delete(); m.delete();
  });

  test("mesh.faceLink (float64) builds", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    const fl = m.faceLink;
    assert(m._handle.is_face_link_built() && m._handle.is_face_link_fresh(),
      "faceLink built + fresh after sync access");
    log("  mesh.faceLink (float64) → built + fresh", "line-pass");
    fl.delete(); m.delete();
  });

  test("mesh.vertexLink (float64) builds", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    const vl = m.vertexLink;
    assert(m._handle.is_vertex_link_built() && m._handle.is_vertex_link_fresh(),
      "vertexLink built + fresh after sync access");
    log("  mesh.vertexLink (float64) → built + fresh", "line-pass");
    vl.delete(); m.delete();
  });

  test("mesh.buildTree (float64) builds tree", () => {
    const tf = getTf();
    const m = tf.boxMesh(2, 2, 2, undefined, undefined, undefined, { dtype: "float64" });
    assert(!m._handle.is_tree_built(), "fresh: tree not built");
    m.buildTree();
    assert(m._handle.is_tree_built() && m._handle.is_tree_fresh(),
      "tree built + fresh after sync mesh.buildTree()");
    log("  mesh.buildTree (float64) → built + fresh", "line-pass");
    m.delete();
  });

});

describe("PointCloud sync lazy accessors on float64", () => {

  test("pointCloud.buildTree (float64) builds tree", () => {
    const tf = getTf();
    const pc = tf.pointCloud(new Float64Array([0,0,0, 1,0,0, 0,1,0, 1,1,0]));
    assert(pc.dtype === "float64", "sanity: pc dtype is float64");
    assert(!pc._handle.is_tree_built(), "fresh pc: tree not built");
    pc.buildTree();
    assert(pc._handle.is_tree_built() && pc._handle.is_tree_fresh(),
      "tree built + fresh after sync pc.buildTree()");
    log("  pointCloud.buildTree (float64) → built + fresh", "line-pass");
    pc.delete();
  });

});

describe("Async precompute populates the original PointCloud (float64)", () => {

  test("async.buildTree on PointCloud (float64)", async () => {
    const tf = getTf();
    const pc = tf.pointCloud(new Float64Array([0,0,0, 1,0,0, 0,1,0, 1,1,0]));
    assert(pc.dtype === "float64", "sanity: pc dtype is float64");
    assert(!pc._handle.is_tree_built(), "fresh pc: tree not built");

    await tf.async.buildTree(pc);

    assert(pc._handle.is_tree_built() && pc._handle.is_tree_fresh(),
      "after async.buildTree(pc, float64): built + fresh on original");
    log("  async.buildTree(pc, float64) → built + fresh on original", "line-pass");
    pc.delete();
  });

});

describe("Async registration leaves tree built on target point cloud", () => {

  test("async.fitIcpAlignment → target tree built+fresh", async () => {
    const tf = getTf();
    const pts0 = randomPoints(200, 42);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i]     = pts0[i]     + 0.1;
      pts1[i + 1] = pts0[i + 1] + 0.05;
      pts1[i + 2] = pts0[i + 2] + 0.1;
    }
    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);

    assert(!tgt._handle.is_tree_built(), "fresh target: tree not built");

    const T = await tf.async.fitIcpAlignment(src, tgt, { maxIterations: 20 });

    assert(tgt._handle.is_tree_built() && tgt._handle.is_tree_fresh(),
      "after async.fitIcpAlignment: target tree built + fresh");
    log("  async.fitIcpAlignment → target tree built+fresh", "line-pass");

    T.delete();
    src.delete();
    tgt.delete();
  });

});
