import { describe, test, log, assert, getTf } from "./harness.mjs";

describe("Topology", () => {

  // ==========================================================================
  test("isClosed / isOpen (sphere vs plane)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const plane = tf.planeMesh(10, 5);

    assert(tf.isClosed(sphere) === true, "sphere should be closed");
    assert(tf.isOpen(sphere) === false, "sphere should not be open");
    log("  sphere: closed=true, open=false", "line-pass");

    assert(tf.isClosed(plane) === false, "plane should not be closed");
    assert(tf.isOpen(plane) === true, "plane should be open");
    log("  plane: closed=false, open=true", "line-pass");

    plane.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("isManifold / isNonManifold (sphere)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    assert(tf.isManifold(sphere) === true, "sphere should be manifold");
    assert(tf.isNonManifold(sphere) === false, "sphere should not be non-manifold");
    log("  sphere: manifold=true, nonManifold=false", "line-pass");

    sphere.delete();
  });

  // ==========================================================================
  test("eulerCharacteristic (sphere = 2, torus = 0)", () => {
    const tf = getTf();

    // Sphere: genus 0 closed surface → χ = 2
    const sphere = tf.sphereMesh(1.0, 16, 16);
    const ec = tf.eulerCharacteristic(sphere);
    assert(ec === 2, `expected euler=2 for sphere, got ${ec}`);
    log(`  sphere euler = ${ec}`, "line-pass");

    // Torus (genus 1): box with a cylindrical hole → χ = 0
    const box = tf.boxMesh(4, 4, 4);
    const cyl = tf.cylinderMesh(0.5, 6, 32);
    const diff = tf.booleanDifference(box, cyl);
    const torus = diff.mesh;
    const ecTorus = tf.eulerCharacteristic(torus);
    assert(ecTorus === 0, `expected euler=0 for torus, got ${ecTorus}`);
    log(`  torus euler = ${ecTorus}`, "line-pass");

    diff.labels.delete();
    torus.delete();
    cyl.delete();
    box.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("boundaryEdges (sphere = 0, plane > 0)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const be = tf.boundaryEdges(sphere);
    assert(be.shape[0] === 0, `expected 0 boundary edges for sphere, got ${be.shape[0]}`);
    log("  sphere: 0 boundary edges", "line-pass");
    be.delete();

    const plane = tf.planeMesh(10, 5);
    const bePlane = tf.boundaryEdges(plane);
    assert(bePlane.shape[0] === 4, `expected 4 boundary edges for plane, got ${bePlane.shape[0]}`);
    assert(bePlane.shape[1] === 2, `expected shape[1]=2, got ${bePlane.shape[1]}`);
    log(`  plane: ${bePlane.shape[0]} boundary edges`, "line-pass");
    bePlane.delete();

    plane.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("nonManifoldEdges (sphere = 0)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const nme = tf.nonManifoldEdges(sphere);
    assert(nme.shape[0] === 0, `expected 0 non-manifold edges, got ${nme.shape[0]}`);
    log("  sphere: 0 non-manifold edges", "line-pass");
    nme.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("boundaryPaths (plane = 1 loop)", () => {
    const tf = getTf();
    const plane = tf.planeMesh(10, 5);
    const bp = tf.boundaryPaths(plane);
    assert(bp.length === 1, `expected 1 boundary path for plane, got ${bp.length}`);
    const path = bp.get(0);
    // Closed loop: last vertex == first vertex, so 4 edges → 5 vertices
    assert(path.shape[0] === 5, `expected 5 vertices in closed boundary loop, got ${path.shape[0]}`);
    const pathData = path.data;
    assert(pathData[0] === pathData[path.shape[0] - 1], "closed path: first == last vertex");
    log(`  plane: 1 closed boundary path with ${path.shape[0]} vertices`, "line-pass");
    path.delete();
    bp.delete();
    plane.delete();
  });

  // ==========================================================================
  test("kRings (sphere, k=1)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const kr = tf.kRings(sphere, 1);
    assert(kr.length === sphere.numberOfPoints, `expected ${sphere.numberOfPoints} rings, got ${kr.length}`);

    // Each vertex should have at least 3 neighbors in 1-ring
    const ring0 = kr.get(0);
    assert(ring0.shape[0] >= 3, `expected >=3 neighbors, got ${ring0.shape[0]}`);
    log(`  vertex 0 has ${ring0.shape[0]} 1-ring neighbors`, "line-pass");

    ring0.delete();
    kr.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("kRings inclusive", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const kr = tf.kRings(sphere, 1, true);
    const ring0 = kr.get(0);
    // Inclusive ring should contain the seed vertex itself
    const data = ring0.data;
    let containsSeed = false;
    for (let i = 0; i < data.length; i++) {
      if (data[i] === 0) containsSeed = true;
    }
    assert(containsSeed, "inclusive k-ring should contain seed vertex");
    log("  inclusive k-ring contains seed vertex", "line-pass");

    ring0.delete();
    kr.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("neighborhoods (sphere)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const nbr = tf.neighborhoods(sphere, 0.5);
    assert(nbr.length === sphere.numberOfPoints, `expected ${sphere.numberOfPoints} neighborhoods`);

    const n0 = nbr.get(0);
    assert(n0.shape[0] >= 1, `expected >=1 neighbors within radius, got ${n0.shape[0]}`);
    log(`  vertex 0: ${n0.shape[0]} neighbors within r=0.5`, "line-pass");

    n0.delete();
    nbr.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("connectEdgesToPaths", () => {
    const tf = getTf();
    // A simple loop: 0-1, 1-2, 2-0
    const edges = tf.ndarray(new Int32Array([0, 1, 1, 2, 2, 0]), [3, 2]);
    const paths = tf.connectEdgesToPaths(edges);
    assert(paths.length === 1, `expected 1 path, got ${paths.length}`);
    const path = paths.get(0);
    // Closed loop: 3 edges → 4 vertices (last == first)
    assert(path.shape[0] === 4, `expected 4 vertices in closed path, got ${path.shape[0]}`);
    const pathData = path.data;
    assert(pathData[0] === pathData[path.shape[0] - 1], "closed path: first == last");
    log(`  3 edges → 1 closed path of ${path.shape[0]} vertices`, "line-pass");
    path.delete();
    paths.delete();
    edges.delete();
  });

  // ==========================================================================
  test("labelConnectedComponents (OffsetBlockedBuffer — vertexLink)", () => {
    const tf = getTf();
    // Two separate spheres → 2 vertex-connected components
    const s1 = tf.sphereMesh(1.0, 8, 8);
    const s2 = tf.sphereMesh(1.0, 8, 8);

    // Offset s2 points
    const p2 = s2.points;
    const p2data = p2.data;
    const offset = new Float32Array(p2data.length);
    for (let i = 0; i < p2data.length; i += 3) {
      offset[i] = p2data[i] + 5;
      offset[i + 1] = p2data[i + 1];
      offset[i + 2] = p2data[i + 2];
    }

    // Combine faces (offset s2 indices)
    const f1 = s1.faces;
    const f2 = s2.faces;
    const nPts1 = s1.numberOfPoints;
    const f2data = f2.data;
    const combinedFaces = new Int32Array(f1.data.length + f2data.length);
    combinedFaces.set(f1.data);
    for (let i = 0; i < f2data.length; i++) {
      combinedFaces[f1.data.length + i] = f2data[i] + nPts1;
    }
    const combinedPoints = new Float32Array(p2data.length + s1.numberOfPoints * 3);
    const p1 = s1.points;
    combinedPoints.set(p1.data);
    combinedPoints.set(offset, s1.numberOfPoints * 3);

    const combined = tf.mesh(combinedFaces, combinedPoints);
    const vl = combined.vertexLink;
    const result = tf.labelConnectedComponents(vl);
    assert(result.nComponents === 2, `expected 2 components, got ${result.nComponents}`);
    assert(result.labels.shape[0] === combined.numberOfPoints,
      `expected ${combined.numberOfPoints} labels, got ${result.labels.shape[0]}`);
    log(`  2 spheres combined → ${result.nComponents} vertex components`, "line-pass");

    result.labels.delete();
    vl.delete();
    combined.delete();
    p1.delete();
    p2.delete();
    f1.delete();
    f2.delete();
    s1.delete();
    s2.delete();
  });

  // ==========================================================================
  test("labelConnectedComponents (NDArrayInt32 — manifoldEdgeLink)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const mel = sphere.manifoldEdgeLink;
    const result = tf.labelConnectedComponents(mel);
    assert(result.nComponents === 1, `expected 1 component for sphere, got ${result.nComponents}`);
    assert(result.labels.shape[0] === sphere.numberOfFaces,
      `expected ${sphere.numberOfFaces} labels, got ${result.labels.shape[0]}`);
    log(`  sphere → ${result.nComponents} manifold-edge component`, "line-pass");

    result.labels.delete();
    mel.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("connectedComponents convenience (edge, manifoldEdge, vertex)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    const edge = tf.connectedComponents(sphere, "edge");
    assert(edge.nComponents === 1, `expected 1 edge component, got ${edge.nComponents}`);
    log(`  sphere edge components: ${edge.nComponents}`, "line-pass");

    const me = tf.connectedComponents(sphere, "manifoldEdge");
    assert(me.nComponents === 1, `expected 1 manifoldEdge component, got ${me.nComponents}`);
    log(`  sphere manifoldEdge components: ${me.nComponents}`, "line-pass");

    const vtx = tf.connectedComponents(sphere, "vertex");
    assert(vtx.nComponents === 1, `expected 1 vertex component, got ${vtx.nComponents}`);
    log(`  sphere vertex components: ${vtx.nComponents}`, "line-pass");

    vtx.labels.delete();
    me.labels.delete();
    edge.labels.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("consistentlyOriented (preserves face/point count)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const oriented = tf.consistentlyOriented(sphere);

    assert(oriented.numberOfFaces === sphere.numberOfFaces,
      `face count: expected ${sphere.numberOfFaces}, got ${oriented.numberOfFaces}`);
    assert(oriented.numberOfPoints === sphere.numberOfPoints,
      `point count: expected ${sphere.numberOfPoints}, got ${oriented.numberOfPoints}`);
    log(`  ${oriented.numberOfFaces} faces, ${oriented.numberOfPoints} points (preserved)`, "line-pass");

    oriented.delete();
    sphere.delete();
  });

});
