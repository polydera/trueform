import { describe, test, log, assert, getTf } from "./harness.mjs";

// Two triangles sharing only vertex 0 — the pinch every NM-vertex verb names.
function bowtieMesh(tf, dtype) {
  const Points = dtype === "float64" ? Float64Array : Float32Array;
  return tf.mesh(
    new Int32Array([0, 1, 2, 0, 3, 4]),
    new Points([0, 0, 0, 1, 1, 0, 1, -1, 0, -1, 1, 0, -1, -1, 0]),
  );
}

// Two triangles sharing no vertex, one piercing the other's interior.
function crossingMesh(tf) {
  return tf.mesh(
    new Int32Array([0, 1, 2, 3, 4, 5]),
    new Float32Array([
      0, 0, 0, 2, 0, 0, 1, 2, 0,
      1, 0.5, -1, 1, 0.5, 1, 1, 1.5, 2,
    ]),
  );
}

describe("Diagnostics", () => {

  // ==========================================================================
  test("hasSelfIntersections (sphere false, crossing triangles true)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    assert(tf.hasSelfIntersections(sphere) === false, "sphere should not self-intersect");
    log("  sphere: no self-intersections", "line-pass");
    sphere.delete();

    const crossing = crossingMesh(tf);
    assert(tf.hasSelfIntersections(crossing) === true, "crossing triangles should self-intersect");
    log("  crossing triangles: self-intersections found", "line-pass");
    crossing.delete();
  });

  // ==========================================================================
  test("nonManifoldVertices (sphere = 0, bowtie names the pinch)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const none = tf.nonManifoldVertices(sphere);
    assert(none.shape[0] === 0, `expected 0 non-manifold vertices, got ${none.shape[0]}`);
    log("  sphere: 0 non-manifold vertices", "line-pass");
    none.delete();
    sphere.delete();

    const bowtie = bowtieMesh(tf, "float32");
    const nmv = tf.nonManifoldVertices(bowtie);
    assert(nmv.shape[0] === 1, `expected 1 non-manifold vertex, got ${nmv.shape[0]}`);
    assert(nmv.data[0] === 0, `expected vertex 0, got ${nmv.data[0]}`);
    log("  bowtie: vertex 0 named", "line-pass");
    nmv.delete();
    bowtie.delete();
  });

  // ==========================================================================
  test("splitNonManifoldVertices (bowtie: map + manifold after)", () => {
    const tf = getTf();
    const bowtie = bowtieMesh(tf, "float32");
    assert(tf.isManifold(bowtie) === false, "bowtie should not be manifold");

    const split = tf.splitNonManifoldVertices(bowtie);
    assert(split.mesh.numberOfFaces === 2, `face count preserved, got ${split.mesh.numberOfFaces}`);
    assert(split.mesh.numberOfPoints === 6, `expected 6 points, got ${split.mesh.numberOfPoints}`);
    assert(split.pointMap.shape[0] === 6, `expected map of 6, got ${split.pointMap.shape[0]}`);
    const map = split.pointMap.data;
    for (let i = 0; i < 5; i++) {
      assert(map[i] === i, `original point ${i} maps to itself`);
    }
    assert(map[5] === 0, `minted point copies vertex 0, got ${map[5]}`);
    assert(tf.isManifold(split.mesh) === true, "split mesh should be manifold");
    log("  bowtie split: 6 points, mint copies vertex 0, manifold after", "line-pass");

    split.pointMap.delete();
    split.mesh.delete();
    bowtie.delete();
  });

  // ==========================================================================
  test("boundaryRims (strip = 1 closed rim, sphere = 0, bowtie = 2)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const noRims = tf.boundaryRims(sphere);
    assert(noRims.vertices.length === 0, `expected 0 rims for sphere, got ${noRims.vertices.length}`);
    log("  sphere: 0 rims", "line-pass");
    noRims.closed.delete();
    noRims.faces.delete();
    noRims.vertices.delete();
    sphere.delete();

    const strip = tf.planeMesh(4, 1, 4, 1);
    const rims = tf.boundaryRims(strip);
    assert(rims.vertices.length === 1, `expected 1 rim for strip, got ${rims.vertices.length}`);
    assert(rims.closed.data[0] === 1, "strip rim should be closed");
    const rimVertices = rims.vertices.get(0);
    const rimFaces = rims.faces.get(0);
    // Closed rim of n vertices has n edges, each carried by one face.
    assert(rimVertices.shape[0] === 10, `expected 10 rim vertices, got ${rimVertices.shape[0]}`);
    assert(rimFaces.shape[0] === 10, `expected 10 rim faces, got ${rimFaces.shape[0]}`);
    log(`  strip: 1 closed rim of ${rimVertices.shape[0]} vertices`, "line-pass");
    rimFaces.delete();
    rimVertices.delete();
    rims.closed.delete();
    rims.faces.delete();
    rims.vertices.delete();
    strip.delete();

    // A pinch splits the rim.
    const bowtie = bowtieMesh(tf, "float32");
    const pinched = tf.boundaryRims(bowtie);
    assert(pinched.vertices.length === 2, `expected 2 rims for bowtie, got ${pinched.vertices.length}`);
    assert(pinched.closed.data[0] === 1 && pinched.closed.data[1] === 1, "bowtie rims both closed");
    log("  bowtie: pinch splits into 2 closed rims", "line-pass");
    pinched.closed.delete();
    pinched.faces.delete();
    pinched.vertices.delete();
    bowtie.delete();
  });

  // ==========================================================================
  test("faceQuality invariants (sphere)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const F = sphere.numberOfFaces;
    const fq = tf.faceQuality(sphere);
    assert(fq.quality.shape[0] === F, `quality shape [${fq.quality.shape[0]}] != [${F}]`);
    assert(fq.minAngle.shape[0] === F, "minAngle shape");
    assert(fq.maxAngle.shape[0] === F, "maxAngle shape");
    assert(fq.aspectRatio.shape[0] === F, "aspectRatio shape");
    const q = fq.quality.data, lo = fq.minAngle.data, hi = fq.maxAngle.data, ar = fq.aspectRatio.data;
    for (let i = 0; i < F; i++) {
      assert(q[i] > 0 && q[i] <= 1.0000001, `quality[${i}] = ${q[i]} out of (0, 1]`);
      assert(lo[i] > 0, `minAngle[${i}] = ${lo[i]} not positive`);
      assert(lo[i] <= hi[i], `minAngle[${i}] > maxAngle[${i}]`);
      assert(hi[i] < Math.PI, `maxAngle[${i}] = ${hi[i]} not below pi`);
      assert(ar[i] >= 1, `aspectRatio[${i}] = ${ar[i]} below 1`);
    }
    log(`  ${F} faces: quality in (0, 1], angles ordered, aspect >= 1`, "line-pass");
    fq.aspectRatio.delete();
    fq.maxAngle.delete();
    fq.minAngle.delete();
    fq.quality.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("dihedralAngles (box: 12 right + 6 flat)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const da = tf.dihedralAngles(box);
    assert(da.edges.shape[0] === 18, `expected 18 edges, got ${da.edges.shape[0]}`);
    assert(da.edges.shape[1] === 2, "edges shape [N, 2]");
    assert(da.angles.shape[0] === 18, `expected 18 angles, got ${da.angles.shape[0]}`);
    let flat = 0, right = 0;
    const a = da.angles.data;
    for (let i = 0; i < a.length; i++) {
      if (Math.abs(a[i]) < 1e-3) flat++;
      if (Math.abs(a[i] - Math.PI / 2) < 1e-3) right++;
    }
    assert(flat === 6, `expected 6 flat diagonals, got ${flat}`);
    assert(right === 12, `expected 12 right-angle edges, got ${right}`);
    log("  box: 12 right-angle edges, 6 flat diagonals", "line-pass");
    da.angles.delete();
    da.edges.delete();
    box.delete();
  });

  // ==========================================================================
  // ASYNC VARIANTS
  // ==========================================================================

  test("async: hasSelfIntersections", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    assert(await tf.async.hasSelfIntersections(sphere) === false, "sphere clean");
    sphere.delete();
    const crossing = crossingMesh(tf);
    assert(await tf.async.hasSelfIntersections(crossing) === true, "crossing found");
    log("  async hasSelfIntersections OK", "line-pass");
    crossing.delete();
  });

  test("async: nonManifoldVertices", async () => {
    const tf = getTf();
    const bowtie = bowtieMesh(tf, "float32");
    const nmv = await tf.async.nonManifoldVertices(bowtie);
    assert(nmv.shape[0] === 1 && nmv.data[0] === 0, "bowtie pinch named");
    log("  async nonManifoldVertices: vertex 0", "line-pass");
    nmv.delete();
    bowtie.delete();
  });

  test("async: splitNonManifoldVertices", async () => {
    const tf = getTf();
    const bowtie = bowtieMesh(tf, "float32");
    const split = await tf.async.splitNonManifoldVertices(bowtie);
    assert(split.mesh.numberOfPoints === 6, "6 points after split");
    assert(split.pointMap.data[5] === 0, "mint copies vertex 0");
    assert(await tf.async.isManifold(split.mesh) === true, "manifold after");
    log("  async splitNonManifoldVertices OK", "line-pass");
    split.pointMap.delete();
    split.mesh.delete();
    bowtie.delete();
  });

  test("async: boundaryRims", async () => {
    const tf = getTf();
    const strip = tf.planeMesh(4, 1, 4, 1);
    const rims = await tf.async.boundaryRims(strip);
    assert(rims.vertices.length === 1, `expected 1 rim, got ${rims.vertices.length}`);
    assert(rims.closed.data[0] === 1, "closed rim");
    log(`  async boundaryRims: ${rims.vertices.length} rim`, "line-pass");
    rims.closed.delete();
    rims.faces.delete();
    rims.vertices.delete();
    strip.delete();
  });

  test("async: faceQuality", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const fq = await tf.async.faceQuality(sphere);
    assert(fq.quality.shape[0] === sphere.numberOfFaces, "quality per face");
    log(`  async faceQuality: ${fq.quality.shape[0]} faces`, "line-pass");
    fq.aspectRatio.delete();
    fq.maxAngle.delete();
    fq.minAngle.delete();
    fq.quality.delete();
    sphere.delete();
  });

  test("async: dihedralAngles", async () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const da = await tf.async.dihedralAngles(box);
    assert(da.edges.shape[0] === 18 && da.angles.shape[0] === 18, "18 shared edges");
    log("  async dihedralAngles: 18 edges", "line-pass");
    da.angles.delete();
    da.edges.delete();
    box.delete();
  });

  // ==========================================================================
  // FLOAT64 MIRRORS
  // ==========================================================================

  test("diagnostics family (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10, { dtype: "float64" });
    assert(sphere.dtype === "float64", "sphere dtype");
    assert(tf.hasSelfIntersections(sphere) === false, "sphere(f64) clean");

    const fq = tf.faceQuality(sphere);
    assert(fq.quality.dtype === "float64", "quality dtype follows mesh");
    assert(fq.quality.shape[0] === sphere.numberOfFaces, "quality per face");
    fq.aspectRatio.delete();
    fq.maxAngle.delete();
    fq.minAngle.delete();
    fq.quality.delete();

    const da = tf.dihedralAngles(sphere);
    assert(da.angles.dtype === "float64", "angles dtype follows mesh");
    da.angles.delete();
    da.edges.delete();

    const rims = tf.boundaryRims(sphere);
    assert(rims.vertices.length === 0, "sphere(f64): 0 rims");
    rims.closed.delete();
    rims.faces.delete();
    rims.vertices.delete();
    sphere.delete();

    const bowtie = bowtieMesh(tf, "float64");
    const nmv = tf.nonManifoldVertices(bowtie);
    assert(nmv.shape[0] === 1 && nmv.data[0] === 0, "bowtie(f64) pinch named");
    nmv.delete();

    const split = tf.splitNonManifoldVertices(bowtie);
    assert(split.mesh.dtype === "float64", "split mesh dtype");
    assert(split.mesh.numberOfPoints === 6, "6 points after split");
    assert(tf.isManifold(split.mesh) === true, "manifold after");
    log("  float64 mirrors OK", "line-pass");
    split.pointMap.delete();
    split.mesh.delete();
    bowtie.delete();
  });

});
