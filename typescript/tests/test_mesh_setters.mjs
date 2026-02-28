import { describe, test, log, assert, getTf } from "./harness.mjs";

describe("Mesh Setters", () => {

  // ==========================================================================
  test("set/get normals roundtrip", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    // Get lazy-computed normals
    const n = box.normals;
    assert(n.shape[0] === box.numberOfFaces, `normals: [${n.shape}]`);

    // Create new mesh and set normals from the first
    const box2 = tf.boxMesh(2, 3, 4);
    box2.normals = n;

    const n2 = box2.normals;
    assert(n2.shape[0] === n.shape[0], "same number of normals after set");
    log("  set/get normals OK", "line-pass");

    n2.delete();
    n.delete();
    box2.delete();
    box.delete();
  });

  // ==========================================================================
  test("set/get pointNormals roundtrip", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    const pn = box.pointNormals;
    assert(pn.shape[0] === box.numberOfPoints, `pointNormals: [${pn.shape}]`);

    const box2 = tf.boxMesh(2, 3, 4);
    box2.pointNormals = pn;

    const pn2 = box2.pointNormals;
    assert(pn2.shape[0] === pn.shape[0], "same number of pointNormals after set");
    log("  set/get pointNormals OK", "line-pass");

    pn2.delete();
    pn.delete();
    box2.delete();
    box.delete();
  });

  // ==========================================================================
  test("set/get vertexLink roundtrip", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    const vl = box.vertexLink;
    assert(vl.length === box.numberOfPoints, `vertexLink: ${vl.length} blocks`);

    const box2 = tf.boxMesh(2, 3, 4);
    box2.vertexLink = vl;

    const vl2 = box2.vertexLink;
    assert(vl2.length === vl.length, "same vertexLink block count after set");
    log("  set/get vertexLink OK", "line-pass");

    vl2.delete();
    vl.delete();
    box2.delete();
    box.delete();
  });

  // ==========================================================================
  test("set/get faceMembership roundtrip", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    const fm = box.faceMembership;
    assert(fm.length === box.numberOfPoints, `faceMembership: ${fm.length} blocks`);

    const box2 = tf.boxMesh(2, 3, 4);
    box2.faceMembership = fm;

    const fm2 = box2.faceMembership;
    assert(fm2.length === fm.length, "same faceMembership block count after set");
    log("  set/get faceMembership OK", "line-pass");

    fm2.delete();
    fm.delete();
    box2.delete();
    box.delete();
  });

  // ==========================================================================
  test("set/get manifoldEdgeLink roundtrip", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    const mel = box.manifoldEdgeLink;
    assert(mel.shape[0] === box.numberOfFaces && mel.shape[1] === 3,
      `manifoldEdgeLink: [${mel.shape}]`);

    const box2 = tf.boxMesh(2, 3, 4);
    box2.manifoldEdgeLink = mel;

    const mel2 = box2.manifoldEdgeLink;
    assert(mel2.shape[0] === mel.shape[0], "same manifoldEdgeLink after set");
    log("  set/get manifoldEdgeLink OK", "line-pass");

    mel2.delete();
    mel.delete();
    box2.delete();
    box.delete();
  });

  // ==========================================================================
  test("set/get faceLink roundtrip", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    const fl = box.faceLink;
    assert(fl.length === box.numberOfFaces, `faceLink: ${fl.length} blocks`);

    const box2 = tf.boxMesh(2, 3, 4);
    box2.faceLink = fl;

    const fl2 = box2.faceLink;
    assert(fl2.length === fl.length, "same faceLink block count after set");
    log("  set/get faceLink OK", "line-pass");

    fl2.delete();
    fl.delete();
    box2.delete();
    box.delete();
  });

  // ==========================================================================
  test("setting faces invalidates caches", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    // Trigger lazy computation
    const n1 = box.normals;
    const vl1 = box.vertexLink;
    const nFaces1 = n1.shape[0];
    const vlLen1 = vl1.length;
    n1.delete();
    vl1.delete();

    // Change faces — subdivided box has different topology
    const box2 = tf.boxMesh(2, 3, 4, 2, 2, 2);
    box.faces = box2.faces;
    box.points = box2.points;

    // Caches should be recomputed for new topology
    const n2 = box.normals;
    const vl2 = box.vertexLink;
    assert(n2.shape[0] === box2.numberOfFaces,
      `expected ${box2.numberOfFaces} normals after face change, got ${n2.shape[0]}`);
    assert(vl2.length === box2.numberOfPoints,
      `expected ${box2.numberOfPoints} vl blocks after change, got ${vl2.length}`);
    log(`  caches invalidated: normals ${nFaces1} → ${n2.shape[0]}, vl ${vlLen1} → ${vl2.length}`, "line-pass");

    vl2.delete();
    n2.delete();
    box2.delete();
    box.delete();
  });

});
