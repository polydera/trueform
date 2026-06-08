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
      minQuality: 0.3,
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

  // ==========================================================================
  test("simplified (sphere — error budget reduces faces)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 20, 20);
    const origFaces = sphere.numberOfFaces;

    const sim = tf.simplified(sphere, { errorRel: 0.01 });
    assert(sim.numberOfFaces < origFaces, `face count should decrease: ${sim.numberOfFaces} < ${origFaces}`);
    assert(sim.numberOfFaces > 0, "should have faces");
    assert(sim.numberOfPoints > 0, "should have points");

    // Volume roughly preserved (error budget keeps curved detail)
    const ratio = tf.volume(sim) / tf.volume(sphere);
    assert(ratio > 0.5 && ratio < 1.5, `volume ratio ${ratio.toFixed(3)} should be near 1`);
    log(`  sphere ${origFaces} → ${sim.numberOfFaces} faces, volume ratio ${ratio.toFixed(3)}`, "line-pass");

    sim.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("simplified (box — flat regions collapse)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4, 4, 4, 4);
    const origFaces = box.numberOfFaces;

    // Flat box faces should collapse for ~0 quadric error.
    const sim = tf.simplified(box);
    assert(sim.numberOfFaces > 0, "should have at least 1 face");
    assert(sim.numberOfFaces <= origFaces, `face count should not increase: ${sim.numberOfFaces} <= ${origFaces}`);
    log(`  box ${origFaces} faces → ${sim.numberOfFaces} faces`, "line-pass");

    sim.delete();
    box.delete();
  });

  // ==========================================================================
  test("simplified with options", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 15, 15);

    const sim = tf.simplified(sphere, {
      errorRel: 0.005,
      optimizeIterations: 2,
      minQuality: 0.2,
      preserveBoundary: false,
      stabilizer: 1e-3,
    });
    assert(sim.numberOfFaces > 0, "should produce valid mesh");
    assert(sim.numberOfFaces <= sphere.numberOfFaces, "should not increase faces");
    log(`  simplified with options → ${sim.numberOfFaces} faces`, "line-pass");

    sim.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("simplified (result mesh has valid topology — he cached)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 15, 15);

    const sim = tf.simplified(sphere, { errorRel: 0.01 });

    // half_edges are cached on the result by set_half_edges — topology access works.
    const fm = sim.faceMembership;
    assert(fm.length === sim.numberOfPoints, `faceMembership size ${fm.length} should equal ${sim.numberOfPoints} points`);
    const mel = sim.manifoldEdgeLink;
    assert(mel.shape[0] === sim.numberOfFaces, `manifoldEdgeLink rows ${mel.shape[0]} should equal ${sim.numberOfFaces} faces`);
    log(`  simplified mesh faceMembership=${fm.length}, manifoldEdgeLink=[${mel.shape}]`, "line-pass");

    mel.delete();
    fm.delete();
    sim.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("async: decimated (box to 50%)", async () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);
    const origFaces = box.numberOfFaces; // 12

    const dec = await tf.async.decimated(box, 0.5);
    assert(dec.numberOfFaces <= origFaces, `face count should decrease: ${dec.numberOfFaces} <= ${origFaces}`);
    assert(dec.numberOfFaces > 0, "should have at least 1 face");
    assert(dec.numberOfPoints > 0, "should have at least 1 point");
    assert(dec.numberOfPoints <= box.numberOfPoints, `point count should decrease: ${dec.numberOfPoints} <= ${box.numberOfPoints}`);
    log(`  async: box 12 faces → ${dec.numberOfFaces} faces, ${dec.numberOfPoints} points`, "line-pass");

    dec.delete();
    box.delete();
  });

  // ==========================================================================
  test("async: isotropicRemeshed (box — basic)", async () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4, 3, 3, 3);
    const mel = tf.meanEdgeLength(box);

    const rem = await tf.async.isotropicRemeshed(box, mel * 2.0);
    assert(rem.numberOfFaces > 0, "should have faces");
    assert(rem.numberOfPoints > 0, "should have points");
    log(`  async: box ${box.numberOfFaces} faces → remeshed ${rem.numberOfFaces} faces (target=${(mel * 2).toFixed(3)})`, "line-pass");

    rem.delete();
    box.delete();
  });

  // ==========================================================================
  test("async: simplified (sphere)", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 20, 20);
    const origFaces = sphere.numberOfFaces;

    const sim = await tf.async.simplified(sphere, { errorRel: 0.01 });
    assert(sim.numberOfFaces < origFaces, `face count should decrease: ${sim.numberOfFaces} < ${origFaces}`);
    assert(sim.numberOfFaces > 0, "should have faces");
    assert(sim.numberOfPoints > 0, "should have points");
    log(`  async: sphere ${origFaces} → ${sim.numberOfFaces} faces`, "line-pass");

    sim.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("decimated (box to 50%, float64)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4, undefined, undefined, undefined, { dtype: "float64" });
    const origFaces = box.numberOfFaces; // 12

    const dec = tf.decimated(box, 0.5);
    assert(dec.dtype === "float64", `expected dtype float64, got ${dec.dtype}`);
    assert(dec.numberOfFaces <= origFaces, `face count should decrease: ${dec.numberOfFaces} <= ${origFaces}`);
    assert(dec.numberOfFaces > 0, "should have at least 1 face");
    assert(dec.numberOfPoints > 0, "should have at least 1 point");
    assert(dec.numberOfPoints <= box.numberOfPoints, `point count should decrease: ${dec.numberOfPoints} <= ${box.numberOfPoints}`);
    log(`  box(float64) 12 faces → ${dec.numberOfFaces} faces, ${dec.numberOfPoints} points`, "line-pass");

    dec.delete();
    box.delete();
  });

  // ==========================================================================
  test("decimated (sphere — preserves rough shape, float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 20, 20, { dtype: "float64" });
    const origFaces = sphere.numberOfFaces;

    const dec = tf.decimated(sphere, 0.3);
    assert(dec.dtype === "float64", `expected dtype float64, got ${dec.dtype}`);
    assert(dec.numberOfFaces < origFaces, `face count should decrease`);
    assert(dec.numberOfFaces > 0, "should have faces");
    log(`  sphere(float64) ${origFaces} → ${dec.numberOfFaces} faces`, "line-pass");

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
  test("decimated with options (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 15, 15, { dtype: "float64" });

    const dec = tf.decimated(sphere, 0.5, {
      minQuality: 0.3,
      preserveBoundary: false,
      stabilizer: 1e-3,
    });
    assert(dec.dtype === "float64", `expected dtype float64, got ${dec.dtype}`);
    assert(dec.numberOfFaces > 0, "should produce valid mesh");
    assert(dec.numberOfFaces <= sphere.numberOfFaces, "should reduce faces");
    log(`  decimated(float64) with options → ${dec.numberOfFaces} faces`, "line-pass");

    dec.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("isotropicRemeshed (box — basic, float64)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4, 3, 3, 3, { dtype: "float64" });
    const mel = tf.meanEdgeLength(box);

    // Remesh to 2× mean edge length → fewer, more uniform triangles
    const rem = tf.isotropicRemeshed(box, mel * 2.0);
    assert(rem.dtype === "float64", `expected dtype float64, got ${rem.dtype}`);
    assert(rem.numberOfFaces > 0, "should have faces");
    assert(rem.numberOfPoints > 0, "should have points");
    log(`  box(float64) ${box.numberOfFaces} faces → remeshed ${rem.numberOfFaces} faces (target=${(mel * 2).toFixed(3)})`, "line-pass");

    rem.delete();
    box.delete();
  });

  // ==========================================================================
  test("isotropicRemeshed (sphere — edge lengths converge, float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10, { dtype: "float64" });
    const targetLen = tf.meanEdgeLength(sphere);

    const rem = tf.isotropicRemeshed(sphere, targetLen, { iterations: 5 });
    assert(rem.dtype === "float64", `expected dtype float64, got ${rem.dtype}`);
    assert(rem.numberOfFaces > 0, "should have faces");

    // After remeshing, edge lengths should be more uniform
    const minEl = tf.minEdgeLength(rem);
    const maxEl = tf.maxEdgeLength(rem);
    const ratio = maxEl / minEl;
    // Ratio should be reasonable (not perfect due to curvature, but < 10)
    assert(ratio < 10, `edge length ratio ${ratio.toFixed(2)} should be < 10`);
    log(`  edge length range(float64): [${minEl.toFixed(4)}, ${maxEl.toFixed(4)}], ratio=${ratio.toFixed(2)}`, "line-pass");

    rem.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("isotropicRemeshed with quadric (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10, { dtype: "float64" });
    const targetLen = tf.meanEdgeLength(sphere);

    const rem = tf.isotropicRemeshed(sphere, targetLen, {
      useQuadric: true,
      iterations: 3,
    });
    assert(rem.dtype === "float64", `expected dtype float64, got ${rem.dtype}`);
    assert(rem.numberOfFaces > 0, "should produce valid mesh with quadric");
    log(`  quadric remesh(float64) → ${rem.numberOfFaces} faces`, "line-pass");

    rem.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("pipeline: decimate then isotropic remesh (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 20, 20, { dtype: "float64" });
    const origFaces = sphere.numberOfFaces;

    // Step 1: Decimate to 30%
    const dec = tf.decimated(sphere, 0.3);
    assert(dec.dtype === "float64", `expected dtype float64, got ${dec.dtype}`);
    assert(dec.numberOfFaces < origFaces, "decimation should reduce faces");

    // Step 2: Isotropic remesh the decimated result
    const mel = tf.meanEdgeLength(dec);
    const rem = tf.isotropicRemeshed(dec, mel, { useQuadric: true });
    assert(rem.dtype === "float64", `expected dtype float64, got ${rem.dtype}`);
    assert(rem.numberOfFaces > 0, "remesh should produce valid mesh");
    log(`  ${origFaces} → decimate(float64) ${dec.numberOfFaces} → remesh ${rem.numberOfFaces} faces`, "line-pass");

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
  test("decimated (result mesh has valid topology, float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 15, 15, { dtype: "float64" });

    const dec = tf.decimated(sphere, 0.5);
    assert(dec.dtype === "float64", `expected dtype float64, got ${dec.dtype}`);

    // Accessing topology on result should work (half_edges pre-cached)
    const fm = dec.faceMembership;
    assert(fm.length === dec.numberOfPoints, `faceMembership size ${fm.length} should equal ${dec.numberOfPoints} points`);
    log(`  decimated(float64) mesh faceMembership size = ${fm.length}`, "line-pass");

    const mel = dec.manifoldEdgeLink;
    assert(mel.shape[0] === dec.numberOfFaces, `manifoldEdgeLink rows ${mel.shape[0]} should equal ${dec.numberOfFaces} faces`);
    log(`  decimated(float64) mesh manifoldEdgeLink shape [${mel.shape}]`, "line-pass");

    mel.delete();
    fm.delete();
    dec.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("isotropicRemeshed with preserveBoundary (float64)", () => {
    const tf = getTf();
    // Use a plane (has boundary edges) instead of closed mesh
    const plane = tf.planeMesh(10, 10, 5, 5, { dtype: "float64" });
    const mel = tf.meanEdgeLength(plane);

    const rem = tf.isotropicRemeshed(plane, mel, {
      preserveBoundary: true,
      iterations: 2,
    });
    assert(rem.dtype === "float64", `expected dtype float64, got ${rem.dtype}`);
    assert(rem.numberOfFaces > 0, "should produce valid mesh");
    log(`  plane(float64) ${plane.numberOfFaces} faces → remeshed ${rem.numberOfFaces} faces (preserveBoundary)`, "line-pass");

    rem.delete();
    plane.delete();
  });

  // ==========================================================================
  test("async: decimated (box to 50%, float64)", async () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4, undefined, undefined, undefined, { dtype: "float64" });
    const origFaces = box.numberOfFaces; // 12

    const dec = await tf.async.decimated(box, 0.5);
    assert(dec.dtype === "float64", `expected dtype float64, got ${dec.dtype}`);
    assert(dec.numberOfFaces <= origFaces, `face count should decrease: ${dec.numberOfFaces} <= ${origFaces}`);
    assert(dec.numberOfFaces > 0, "should have at least 1 face");
    assert(dec.numberOfPoints > 0, "should have at least 1 point");
    assert(dec.numberOfPoints <= box.numberOfPoints, `point count should decrease: ${dec.numberOfPoints} <= ${box.numberOfPoints}`);
    log(`  async: box(float64) 12 faces → ${dec.numberOfFaces} faces, ${dec.numberOfPoints} points`, "line-pass");

    dec.delete();
    box.delete();
  });

  // ==========================================================================
  test("async: isotropicRemeshed (box — basic, float64)", async () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4, 3, 3, 3, { dtype: "float64" });
    const mel = tf.meanEdgeLength(box);

    const rem = await tf.async.isotropicRemeshed(box, mel * 2.0);
    assert(rem.dtype === "float64", `expected dtype float64, got ${rem.dtype}`);
    assert(rem.numberOfFaces > 0, "should have faces");
    assert(rem.numberOfPoints > 0, "should have points");
    log(`  async: box(float64) ${box.numberOfFaces} faces → remeshed ${rem.numberOfFaces} faces (target=${(mel * 2).toFixed(3)})`, "line-pass");

    rem.delete();
    box.delete();
  });

  // ==========================================================================
  test("simplified (sphere — error budget reduces faces, float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 20, 20, { dtype: "float64" });
    const origFaces = sphere.numberOfFaces;

    const sim = tf.simplified(sphere, { errorRel: 0.01 });
    assert(sim.dtype === "float64", `expected dtype float64, got ${sim.dtype}`);
    assert(sim.numberOfFaces < origFaces, `face count should decrease: ${sim.numberOfFaces} < ${origFaces}`);
    assert(sim.numberOfFaces > 0, "should have faces");

    const ratio = tf.volume(sim) / tf.volume(sphere);
    assert(ratio > 0.5 && ratio < 1.5, `volume ratio ${ratio.toFixed(3)} should be near 1`);
    log(`  sphere(float64) ${origFaces} → ${sim.numberOfFaces} faces, volume ratio ${ratio.toFixed(3)}`, "line-pass");

    sim.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("simplified with options (float64)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 15, 15, { dtype: "float64" });

    const sim = tf.simplified(sphere, {
      errorRel: 0.005,
      optimizeIterations: 2,
      minQuality: 0.2,
      preserveBoundary: false,
    });
    assert(sim.dtype === "float64", `expected dtype float64, got ${sim.dtype}`);
    assert(sim.numberOfFaces > 0, "should produce valid mesh");
    assert(sim.numberOfFaces <= sphere.numberOfFaces, "should not increase faces");
    log(`  simplified(float64) with options → ${sim.numberOfFaces} faces`, "line-pass");

    sim.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("async: simplified (sphere, float64)", async () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 20, 20, { dtype: "float64" });
    const origFaces = sphere.numberOfFaces;

    const sim = await tf.async.simplified(sphere, { errorRel: 0.01 });
    assert(sim.dtype === "float64", `expected dtype float64, got ${sim.dtype}`);
    assert(sim.numberOfFaces < origFaces, `face count should decrease: ${sim.numberOfFaces} < ${origFaces}`);
    assert(sim.numberOfFaces > 0, "should have faces");
    log(`  async: sphere(float64) ${origFaces} → ${sim.numberOfFaces} faces`, "line-pass");

    sim.delete();
    sphere.delete();
  });

  // ==========================================================================
  // preserve_regions
  // ==========================================================================
  function boxWithLabels() {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2, 8, 8, 8);
    const n = box.numberOfFaces;
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) labels[i] = i < n / 2 ? 0 : 1;
    return { box, labels, n };
  }

  const REGION_OPS = [
    ["decimated", (tf, box, opts) => tf.decimated(box, 0.5, opts)],
    ["simplified", (tf, box, opts) => tf.simplified(box, opts)],
    ["isotropicRemeshed",
     (tf, box, opts) => tf.isotropicRemeshed(box, 2.0 * tf.meanEdgeLength(box), opts)],
  ];

  for (const [name, call] of REGION_OPS) {
    test(`preserve_regions: ${name} returns { mesh, regions }`, () => {
      const tf = getTf();
      const { box, labels } = boxWithLabels();
      const r = call(tf, box, { preserveRegions: labels });
      assert(r.mesh && r.regions, `${name}: expected { mesh, regions }`);
      assert(r.regions.dtype === "int32", `${name}: regions should be int32`);
      assert(r.regions.length === r.mesh.numberOfFaces,
        `${name}: one label per output face (${r.regions.length} vs ${r.mesh.numberOfFaces})`);
      log(`  ${name}: ${r.mesh.numberOfFaces} faces, ${r.regions.length} labels`, "line-pass");
      r.mesh.delete();
      box.delete();
    });

    test(`preserve_regions: ${name} without regions returns a Mesh`, () => {
      const tf = getTf();
      const { box } = boxWithLabels();
      const m = call(tf, box, {});
      assert(m.regions === undefined, `${name}: no regions -> bare Mesh`);
      assert(m.numberOfFaces > 0, `${name}: should produce a mesh`);
      m.delete();
      box.delete();
    });

    test(`preserve_regions: ${name} wrong-size (incl. empty) throws`, () => {
      const tf = getTf();
      const { box, n } = boxWithLabels();
      for (const bad of [new Int32Array(0), new Int32Array(n - 1)]) {
        let threw = false;
        try { call(tf, box, { preserveRegions: bad }); }
        catch { threw = true; }
        assert(threw, `${name}: wrong-size labels (${bad.length}) should throw`);
      }
      box.delete();
    });
  }

  // ==========================================================================
  test("preserve_regions: async decimated returns { mesh, regions }", async () => {
    const tf = getTf();
    const { box, labels } = boxWithLabels();
    const r = await tf.async.decimated(box, 0.5, { preserveRegions: labels });
    assert(r.mesh && r.regions, "async: expected { mesh, regions }");
    assert(r.regions.length === r.mesh.numberOfFaces,
      "async: one label per output face");
    log(`  async decimated: ${r.mesh.numberOfFaces} faces, ${r.regions.length} labels`, "line-pass");
    r.mesh.delete();
    box.delete();
  });

});
