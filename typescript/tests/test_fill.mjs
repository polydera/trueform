import { describe, test, log, assert, getTf } from "./harness.mjs";

function punchedSphere(tf, dtype) {
  const sphere = tf.sphereMesh(1.0, 24, 24, { dtype });
  const faces = sphere.faces.data;
  const points = sphere.points.data;
  const nf = sphere.numberOfFaces;
  const maskData = new Int8Array(nf);
  for (let f = 0; f < nf; f++) {
    let z = 0;
    for (let c = 0; c < 3; c++) z += points[3 * faces[3 * f + c] + 2];
    maskData[f] = z / 3 < 0.8 ? 1 : 0;
  }
  const mask = tf.ndarray(maskData, [nf]);
  const punched = tf.reindexedByMask(sphere, mask);
  mask.delete();
  sphere.delete();
  return punched;
}

describe("Fill", () => {

  for (const dtype of ["float32", "float64"]) {
    test(`fillHoles + filledMesh closes a punched sphere (${dtype})`, () => {
      const tf = getTf();
      const mesh = punchedSphere(tf, dtype);
      const rims = tf.boundaryRims(mesh);
      assert(rims.vertices.length === 1, `expected 1 rim, got ${rims.vertices.length}`);

      const fill = tf.fillHoles(mesh, rims);
      assert(fill.status.length === 1, "one group");
      assert(fill.status.data[0] === tf.HoleFillStatus.filled, `status ${fill.status.data[0]}`);
      assert(fill.charted.data[0] === tf.HoleChartStatus.admitted, `charted ${fill.charted.data[0]}`);
      assert(fill.triangles.shape[1] === 3, "triangle rows");
      assert(fill.triangles.shape[0] > 0, "patch is not empty");
      assert(fill.offending.shape[0] === 1 && fill.offending.data[0] === -1, "no offending edge");

      const filled = tf.filledMesh(mesh, fill);
      assert(filled.numberOfPoints === mesh.numberOfPoints + fill.mintedPoints.shape[0], "points = input + minted");
      assert(tf.isClosed(filled), "filled mesh is closed");
      log(`  ${dtype}: ${fill.triangles.shape[0]} patch triangles, ${fill.mintedPoints.shape[0]} minted`, "line-pass");

      filled.delete();
      tf.deleteHoleFill(fill);
      rims.closed.delete(); rims.faces.delete(); rims.vertices.delete();
      mesh.delete();
    });
  }

  test("fillHoles without rim splits publishes none", () => {
    const tf = getTf();
    const mesh = punchedSphere(tf, "float32");
    const rims = tf.boundaryRims(mesh);
    const fill = tf.fillHoles(mesh, rims, { splitRim: false, minQuality: 0, fairing: false });
    assert(fill.splits.shape[0] === 0, "no splits");
    assert(fill.planFaces.length === 0, "no plans");
    const filled = tf.filledMesh(mesh, fill);
    assert(tf.isClosed(filled), "filled mesh is closed");
    filled.delete();
    tf.deleteHoleFill(fill);
    rims.closed.delete(); rims.faces.delete(); rims.vertices.delete();
    mesh.delete();
  });

  test("async: fillHoles + filledMesh match sync", async () => {
    const tf = getTf();
    const mesh = punchedSphere(tf, "float64");
    const rims = tf.boundaryRims(mesh);
    const sync = tf.fillHoles(mesh, rims);
    const fill = await tf.async.fillHoles(mesh, rims);
    assert(fill.triangles.length === sync.triangles.length, "same patch size");
    const a = fill.triangles.data, b = sync.triangles.data;
    for (let i = 0; i < a.length; i++) assert(a[i] === b[i], "same triangles");
    // The fill's lanes own their storage: dropping the rims and input first is safe.
    rims.closed.delete(); rims.faces.delete(); rims.vertices.delete();
    const filled = await tf.async.filledMesh(mesh, fill);
    mesh.delete();
    assert(tf.isClosed(filled), "async filled mesh is closed");
    log(`  async: ${filled.numberOfFaces} faces`, "line-pass");
    filled.delete();
    tf.deleteHoleFill(fill);
    tf.deleteHoleFill(sync);
  });

  test("fillHoles on a closed mesh holds no groups", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);
    const rims = tf.boundaryRims(sphere);
    const fill = tf.fillHoles(sphere, rims);
    assert(fill.status.length === 0, "no groups");
    const filled = tf.filledMesh(sphere, fill);
    assert(filled.numberOfFaces === sphere.numberOfFaces, "unchanged faces");
    filled.delete();
    tf.deleteHoleFill(fill);
    rims.closed.delete(); rims.faces.delete(); rims.vertices.delete();
    sphere.delete();
  });

});
