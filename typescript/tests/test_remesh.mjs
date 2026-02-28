import { describe, test, log, assert, getTf } from "./harness.mjs";

describe("Remesh", () => {

  // ==========================================================================
  test("decimated (box to 50%)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);
    const origFaces = box.numberOfFaces; // 12

    const dec = tf.decimated(box, 0.5);
    assert(dec.numberOfFaces <= origFaces, `face count should decrease: ${dec.numberOfFaces} <= ${origFaces}`);
    assert(dec.numberOfFaces > 0, "should have at least 1 face");
    assert(dec.numberOfPoints > 0, "should have at least 1 point");
    assert(dec.numberOfPoints <= box.numberOfPoints, `point count should decrease: ${dec.numberOfPoints} <= ${box.numberOfPoints}`);
    log(`  box 12 faces → ${dec.numberOfFaces} faces, ${dec.numberOfPoints} points`, "line-pass");

    dec.delete();
    box.delete();
  });

  // ==========================================================================
  test("decimated (sphere — preserves rough shape)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 20, 20);
    const origFaces = sphere.numberOfFaces;

    const dec = tf.decimated(sphere, 0.3);
    assert(dec.numberOfFaces < origFaces, `face count should decrease`);
    assert(dec.numberOfFaces > 0, "should have faces");
    log(`  sphere ${origFaces} → ${dec.numberOfFaces} faces`, "line-pass");

    // Volume should be roughly preserved (within 50%)
    const origVol = tf.volume(sphere);
    const decVol = tf.volume(dec);
    const ratio = decVol / origVol;
    assert(ratio > 0.5 && ratio < 1.5, `volume ratio ${ratio.toFixed(3)} should be near 1`);
    log(`  volume ratio = ${ratio.toFixed(3)}`, "line-pass");

    dec.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("decimated with options", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 15, 15);

    const dec = tf.decimated(sphere, 0.5, {
      maxAspectRatio: 20,
      preserveBoundary: false,
      stabilizer: 1e-3,
    });
    assert(dec.numberOfFaces > 0, "should produce valid mesh");
    assert(dec.numberOfFaces <= sphere.numberOfFaces, "should reduce faces");
    log(`  decimated with options → ${dec.numberOfFaces} faces`, "line-pass");

    dec.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("isotropicRemeshed (box — basic)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4, 3, 3, 3);
    const mel = tf.meanEdgeLength(box);

    // Remesh to 2× mean edge length → fewer, more uniform triangles
    const rem = tf.isotropicRemeshed(box, mel * 2.0);
    assert(rem.numberOfFaces > 0, "should have faces");
    assert(rem.numberOfPoints > 0, "should have points");
    log(`  box ${box.numberOfFaces} faces → remeshed ${rem.numberOfFaces} faces (target=${(mel * 2).toFixed(3)})`, "line-pass");

    rem.delete();
    box.delete();
  });

  // ==========================================================================
  test("isotropicRemeshed (sphere — edge lengths converge)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const targetLen = tf.meanEdgeLength(sphere);

    const rem = tf.isotropicRemeshed(sphere, targetLen, { iterations: 5 });
    assert(rem.numberOfFaces > 0, "should have faces");

    // After remeshing, edge lengths should be more uniform
    const minEl = tf.minEdgeLength(rem);
    const maxEl = tf.maxEdgeLength(rem);
    const ratio = maxEl / minEl;
    // Ratio should be reasonable (not perfect due to curvature, but < 10)
    assert(ratio < 10, `edge length ratio ${ratio.toFixed(2)} should be < 10`);
    log(`  edge length range: [${minEl.toFixed(4)}, ${maxEl.toFixed(4)}], ratio=${ratio.toFixed(2)}`, "line-pass");

    rem.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("isotropicRemeshed with quadric", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const targetLen = tf.meanEdgeLength(sphere);

    const rem = tf.isotropicRemeshed(sphere, targetLen, {
      useQuadric: true,
      iterations: 3,
    });
    assert(rem.numberOfFaces > 0, "should produce valid mesh with quadric");
    log(`  quadric remesh → ${rem.numberOfFaces} faces`, "line-pass");

    rem.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("pipeline: decimate then isotropic remesh", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 20, 20);
    const origFaces = sphere.numberOfFaces;

    // Step 1: Decimate to 30%
    const dec = tf.decimated(sphere, 0.3);
    assert(dec.numberOfFaces < origFaces, "decimation should reduce faces");

    // Step 2: Isotropic remesh the decimated result
    const mel = tf.meanEdgeLength(dec);
    const rem = tf.isotropicRemeshed(dec, mel, { useQuadric: true });
    assert(rem.numberOfFaces > 0, "remesh should produce valid mesh");
    log(`  ${origFaces} → decimate ${dec.numberOfFaces} → remesh ${rem.numberOfFaces} faces`, "line-pass");

    // Volume should still be roughly preserved
    const origVol = tf.volume(sphere);
    const remVol = tf.volume(rem);
    const ratio = remVol / origVol;
    assert(ratio > 0.3 && ratio < 2.0, `volume ratio ${ratio.toFixed(3)} should be reasonable`);
    log(`  volume ratio = ${ratio.toFixed(3)}`, "line-pass");

    rem.delete();
    dec.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("decimated (result mesh has valid topology)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 15, 15);

    const dec = tf.decimated(sphere, 0.5);

    // Accessing topology on result should work (half_edges pre-cached)
    const fm = dec.faceMembership;
    assert(fm.length === dec.numberOfPoints, `faceMembership size ${fm.length} should equal ${dec.numberOfPoints} points`);
    log(`  decimated mesh faceMembership size = ${fm.length}`, "line-pass");

    const mel = dec.manifoldEdgeLink;
    assert(mel.shape[0] === dec.numberOfFaces, `manifoldEdgeLink rows ${mel.shape[0]} should equal ${dec.numberOfFaces} faces`);
    log(`  decimated mesh manifoldEdgeLink shape [${mel.shape}]`, "line-pass");

    mel.delete();
    fm.delete();
    dec.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("isotropicRemeshed with preserveBoundary", () => {
    const tf = getTf();
    // Use a plane (has boundary edges) instead of closed mesh
    const plane = tf.planeMesh(10, 10, 5, 5);
    const mel = tf.meanEdgeLength(plane);

    const rem = tf.isotropicRemeshed(plane, mel, {
      preserveBoundary: true,
      iterations: 2,
    });
    assert(rem.numberOfFaces > 0, "should produce valid mesh");
    log(`  plane ${plane.numberOfFaces} faces → remeshed ${rem.numberOfFaces} faces (preserveBoundary)`, "line-pass");

    rem.delete();
    plane.delete();
  });

});
