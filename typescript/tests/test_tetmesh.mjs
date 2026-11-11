import { describe, test, log, assert, getTf } from "./harness.mjs";

function volumeOf(result) {
  const t = result.tetrahedra.data, p = result.points.data;
  let sum = 0;
  for (let i = 0; i < t.length; i += 4) {
    const a = 3 * t[i], b = 3 * t[i + 1], c = 3 * t[i + 2], d = 3 * t[i + 3];
    const u = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
    const v = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
    const w = [p[d] - p[a], p[d + 1] - p[a + 1], p[d + 2] - p[a + 2]];
    sum += u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
      u[2] * (v[0] * w[1] - v[1] * w[0]);
  }
  return sum / 6;
}

function sameArray(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

describe("Tetmesh", () => {

  for (const dtype of ["float32", "float64"]) {
    test(`tetMesh fills a box in float64 points (${dtype})`, () => {
      const tf = getTf();
      const box = tf.boxMesh(1, 2, 3, 2, 2, 2, { dtype });
      const result = tf.tetMesh(box, { returnReport: true });
      assert(result.status === tf.TetMeshStatus.meshed, `status ${result.status}`);
      assert(result.report.status === result.status, "the report states the status");
      assert(result.report.drains === 0, "no drain on a clean surface");
      assert(result.tetrahedra.shape[1] === 4 && result.tetrahedra.shape[0] > 0, "tetrahedra rows");
      assert(result.points.dtype === "float64", `points ${result.points.dtype}`);
      assert(Math.abs(volumeOf(result) - 6) < 1e-9, `volume ${volumeOf(result)}`);
      assert(typeof result.report.census.faces.covered === "number", "nested census");
      log(`  ${dtype}: ${result.tetrahedra.shape[0]} tets`, "line-pass");
      tf.deleteTetMesh(result);
      box.delete();
    });
  }

  test("tetMesh index map covers every face", () => {
    const tf = getTf();
    const box = tf.boxMesh(1, 2, 3, 2, 2, 2);
    const result = tf.tetMesh(box, { returnIndexMap: true });
    assert(result.status === tf.TetMeshStatus.meshed, "meshed");
    assert(result.report === undefined, "the report only when asked");
    assert(result.faceFacetOffsets.length === box.numberOfFaces + 1, "one block per face");
    assert(result.facetTriangles.shape[1] === 3, "triangle rows");
    const offsets = result.facetOffsets.data;
    assert(offsets[offsets.length - 1] === result.facetTriangles.shape[0], "offsets close");
    const plain = tf.tetMesh(box);
    assert(sameArray(plain.tetrahedra.data, result.tetrahedra.data), "same cells");
    tf.deleteTetMesh(plain);
    tf.deleteTetMesh(result);
    box.delete();
  });

  test("tetMesh as given: same cells on a clean box, refusal on a crossing", () => {
    const tf = getTf();
    const box = tf.boxMesh(1, 2, 3, 2, 2, 2);
    const resolved = tf.tetMesh(box);
    const given = tf.tetMesh(box, { splitConstraints: false, returnReport: true });
    assert(given.report.drains === 0, "no drain as given");
    assert(sameArray(resolved.tetrahedra.data, given.tetrahedra.data), "same cells");
    tf.deleteTetMesh(resolved);
    tf.deleteTetMesh(given);
    box.delete();
    const a = tf.boxMesh(2, 2, 2);
    const b = tf.boxMesh(2, 2, 2);
    b.transformation = tf.makeTranslation(0.5, 0.75, 1.25);
    const crossed = tf.concatenateMeshes([a, b]);
    const meshed = tf.tetMesh(crossed);
    const refused = tf.tetMesh(crossed, { splitConstraints: false });
    assert(meshed.status === tf.TetMeshStatus.meshed, "resolved meshes");
    assert(refused.status === tf.TetMeshStatus.refused, `as given ${refused.status}`);
    tf.deleteTetMesh(meshed);
    tf.deleteTetMesh(refused);
    crossed.delete(); a.delete(); b.delete();
  });

  test("tetMesh reads the transformation", () => {
    const tf = getTf();
    const box = tf.boxMesh(1, 2, 3, 2, 2, 2);
    const plain = tf.tetMesh(box);
    box.transformation = tf.makeTranslation(5, -2, 7);
    const moved = tf.tetMesh(box);
    assert(sameArray(plain.tetrahedra.data, moved.tetrahedra.data), "same cells");
    const p = plain.points.data, q = moved.points.data;
    const shift = [5, -2, 7];
    let worst = 0;
    for (let i = 0; i < p.length; i++) worst = Math.max(worst, Math.abs(q[i] - p[i] - shift[i % 3]));
    assert(worst < 1e-6, `worst ${worst}`);
    tf.deleteTetMesh(plain);
    tf.deleteTetMesh(moved);
    box.delete();
  });

  test("async: tetMesh matches sync and owns its storage", async () => {
    const tf = getTf();
    const box = tf.boxMesh(1, 2, 3, 2, 2, 2, { dtype: "float64" });
    const sync = tf.tetMesh(box, { returnIndexMap: true });
    const result = await tf.async.tetMesh(box, { returnIndexMap: true });
    box.delete();
    assert(sameArray(sync.tetrahedra.data, result.tetrahedra.data), "same cells");
    assert(sameArray(sync.facetTriangles.data, result.facetTriangles.data), "same facets");
    assert(Math.abs(volumeOf(result) - 6) < 1e-9, "volume after the input is gone");
    tf.deleteTetMesh(sync);
    tf.deleteTetMesh(result);
  });
});
