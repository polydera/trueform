import { describe, test, log, assert, getTf } from "./harness.mjs";

describe("PointCloud", () => {

  // ==========================================================================
  test("pointCloud from Float32Array", () => {
    const tf = getTf();
    const pts = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const pc = tf.pointCloud(pts);

    assert(pc.dtype === "float32", `expected dtype float32, got ${pc.dtype}`);
    assert(pc.numberOfPoints === 3, `expected 3 points, got ${pc.numberOfPoints}`);
    log(`  pointCloud(Float32Array) → ${pc.numberOfPoints} points`, "line-pass");

    const p = pc.points;
    assert(p.dtype === "float32", `expected points dtype float32, got ${p.dtype}`);
    assert(p.shape[0] === 3 && p.shape[1] === 3, `points shape: [${p.shape}]`);
    log("  points shape [3, 3]", "line-pass");

    p.delete();
    pc.delete();
  });

  // ==========================================================================
  test("pointCloud from Float64Array (float64)", () => {
    const tf = getTf();
    const pts = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const pc = tf.pointCloud(pts);

    assert(pc.dtype === "float64", `expected dtype float64, got ${pc.dtype}`);
    assert(pc.numberOfPoints === 3, `expected 3 points, got ${pc.numberOfPoints}`);
    log(`  pointCloud(Float64Array) → ${pc.numberOfPoints} points`, "line-pass");

    const p = pc.points;
    assert(p.dtype === "float64", `expected points dtype float64, got ${p.dtype}`);
    assert(p.data instanceof Float64Array, "points data should be Float64Array");
    assert(p.shape[0] === 3 && p.shape[1] === 3, `points shape: [${p.shape}]`);
    log("  points shape [3, 3] (float64)", "line-pass");

    p.delete();
    pc.delete();
  });

  // ==========================================================================
  test("pointCloud from NDArray", () => {
    const tf = getTf();
    const pts = tf.ndarray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), [3, 3]);
    const pc = tf.pointCloud(pts);

    assert(pc.dtype === "float32", `expected dtype float32, got ${pc.dtype}`);
    assert(pc.numberOfPoints === 3, `expected 3 points, got ${pc.numberOfPoints}`);
    log(`  pointCloud(NDArray) → ${pc.numberOfPoints} points`, "line-pass");

    pts.delete();
    pc.delete();
  });

  // ==========================================================================
  test("pointCloud from NDArray (float64)", () => {
    const tf = getTf();
    const pts = tf.ndarray(new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), [3, 3]);
    const pc = tf.pointCloud(pts);

    assert(pc.dtype === "float64", `expected dtype float64, got ${pc.dtype}`);
    assert(pc.numberOfPoints === 3, `expected 3 points, got ${pc.numberOfPoints}`);
    log(`  pointCloud(NDArray float64) → ${pc.numberOfPoints} points`, "line-pass");

    pts.delete();
    pc.delete();
  });

  // ==========================================================================
  test("pointCloud from Mesh", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);
    const pc = tf.pointCloud(box);

    assert(pc.numberOfPoints === box.numberOfPoints,
      `expected ${box.numberOfPoints} points, got ${pc.numberOfPoints}`);
    log(`  pointCloud(box) → ${pc.numberOfPoints} points`, "line-pass");

    // Should have vertex link (copied from mesh)
    const vl = pc.vertexLink;
    assert(vl !== null, "expected vertexLink to be set");
    assert(vl.length === pc.numberOfPoints,
      `vertexLink blocks: ${vl.length}, expected ${pc.numberOfPoints}`);
    log(`  vertexLink: ${vl.length} blocks`, "line-pass");

    // Should have normals (point normals copied from mesh)
    const n = pc.normals;
    assert(n !== null, "expected normals to be set");
    assert(n.shape[0] === pc.numberOfPoints && n.shape[1] === 3,
      `normals shape: [${n.shape}]`);
    log(`  normals shape [${n.shape}]`, "line-pass");

    n.delete();
    vl.delete();
    pc.delete();
    box.delete();
  });

  // ==========================================================================
  test("vertexLink setter/getter roundtrip", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    // Get vertex link from mesh
    const vl = box.vertexLink;

    // Create empty point cloud and set vertex link
    const pc = tf.pointCloud(box.points.data);
    assert(pc.vertexLink === null, "expected no vertexLink initially");
    pc.vertexLink = vl;

    const vlBack = pc.vertexLink;
    assert(vlBack !== null, "expected vertexLink after set");
    assert(vlBack.length === vl.length,
      `expected ${vl.length} blocks, got ${vlBack.length}`);
    log(`  set/get vertexLink: ${vlBack.length} blocks`, "line-pass");

    vlBack.delete();
    vl.delete();
    pc.delete();
    box.delete();
  });

  // ==========================================================================
  test("normals setter/getter roundtrip", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    // Get point normals from mesh
    const pn = box.pointNormals;

    // Create empty point cloud and set normals
    const pc = tf.pointCloud(box.points.data);
    assert(pc.normals === null, "expected no normals initially");
    pc.normals = pn;

    const nBack = pc.normals;
    assert(nBack !== null, "expected normals after set");
    assert(nBack.shape[0] === pn.shape[0] && nBack.shape[1] === 3,
      `normals shape: [${nBack.shape}]`);
    log(`  set/get normals: [${nBack.shape}]`, "line-pass");

    nBack.delete();
    pn.delete();
    pc.delete();
    box.delete();
  });

  // ==========================================================================
  test("transformation setter/getter", () => {
    const tf = getTf();
    const pc = tf.pointCloud(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));

    assert(pc.transformation === null, "expected no transformation initially");

    const t = tf.ndarray(new Float32Array([
      1, 0, 0, 1,
      0, 1, 0, 2,
      0, 0, 1, 3,
      0, 0, 0, 1,
    ]), [4, 4]);
    pc.transformation = t;

    const tBack = pc.transformation;
    assert(tBack !== null, "expected transformation after set");
    assert(tBack.shape[0] === 4 && tBack.shape[1] === 4,
      `transformation shape: [${tBack.shape}]`);
    log("  transformation set/get OK", "line-pass");

    pc.transformation = null;
    assert(pc.transformation === null, "expected null after clear");
    log("  transformation clear OK", "line-pass");

    tBack.delete();
    t.delete();
    pc.delete();
  });

  // ==========================================================================
  test("shallowCopy shares data", () => {
    const tf = getTf();
    const pc = tf.pointCloud(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));

    const t = tf.ndarray(new Float32Array([
      1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1,
    ]), [4, 4]);
    pc.transformation = t;

    const view = pc.shallowCopy();
    assert(view.dtype === "float32", "shallow copy preserves dtype");
    assert(view.numberOfPoints === pc.numberOfPoints,
      "shallow copy has same point count");
    assert(view.transformation === null,
      "shallow copy has no transformation");
    log("  shallowCopy: same points, no transformation", "line-pass");

    t.delete();
    view.delete();
    pc.delete();
  });

  // ==========================================================================
  test("shallowCopy shares data (float64)", () => {
    const tf = getTf();
    const pc = tf.pointCloud(new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
    assert(pc.dtype === "float64", "source dtype is float64");

    const t = tf.ndarray(new Float64Array([
      1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1,
    ]), [4, 4]);
    pc.transformation = t;

    const view = pc.shallowCopy();
    assert(view.dtype === "float64", "shallow copy preserves float64 dtype");
    assert(view.numberOfPoints === pc.numberOfPoints,
      "shallow copy has same point count");
    assert(view.transformation === null,
      "shallow copy has no transformation");
    const vp = view.points;
    assert(vp.dtype === "float64", "shallow copy points dtype is float64");
    assert(vp.data instanceof Float64Array, "shallow copy points data is Float64Array");
    log("  shallowCopy preserves float64 dtype", "line-pass");

    vp.delete();
    t.delete();
    view.delete();
    pc.delete();
  });

  // ==========================================================================
  test("Independent handle ownership (float64)", () => {
    const tf = getTf();
    const pts = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const pc = tf.pointCloud(pts);

    const ph = pc.points;
    assert(ph.dtype === "float64", "points dtype is float64");

    pc.delete();

    assert(ph.data.length === 9, "points data still accessible after pc.delete()");
    assert(ph.data instanceof Float64Array, "points data is Float64Array after pc delete");
    log("  handles survive pc.delete() (float64)", "line-pass");

    ph.delete();
  });

  // ==========================================================================
  test("transformation setter/getter (float64)", () => {
    const tf = getTf();
    const pc = tf.pointCloud(new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
    assert(pc.dtype === "float64", "pc dtype is float64");

    assert(pc.transformation === null, "expected no transformation initially");

    const t = tf.ndarray(new Float64Array([
      1, 0, 0, 1,
      0, 1, 0, 2,
      0, 0, 1, 3,
      0, 0, 0, 1,
    ]), [4, 4]);
    pc.transformation = t;

    const tBack = pc.transformation;
    assert(tBack !== null, "expected transformation after set");
    assert(tBack.dtype === "float64", "transformation dtype is float64");
    assert(tBack.shape[0] === 4 && tBack.shape[1] === 4,
      `transformation shape: [${tBack.shape}]`);
    log("  transformation set/get OK (float64)", "line-pass");

    pc.transformation = null;
    assert(pc.transformation === null, "expected null after clear");
    log("  transformation clear OK (float64)", "line-pass");

    tBack.delete();
    t.delete();
    pc.delete();
  });

});
