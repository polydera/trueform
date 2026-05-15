import { describe, test, log, assert, getTf } from "./harness.mjs";

function ringPoints(tf, nBoundary, nSteiner = 0, dtype = "float32") {
  // First nBoundary points form a regular polygon outline; remaining
  // are random points inside the unit square.
  const total = nBoundary + nSteiner;
  const Ctor = dtype === "float64" ? Float64Array : Float32Array;
  const data = new Ctor(total * 2);
  for (let i = 0; i < nBoundary; i++) {
    const t = (2 * Math.PI * i) / nBoundary;
    data[2 * i] = 0.5 + 0.45 * Math.cos(t);
    data[2 * i + 1] = 0.5 + 0.45 * Math.sin(t);
  }
  for (let i = nBoundary; i < total; i++) {
    data[2 * i] = Math.random();
    data[2 * i + 1] = Math.random();
  }
  return tf.ndarray(data, [total, 2]);
}

function closingEdges(tf, nBoundary) {
  const e = new Int32Array(nBoundary * 2);
  for (let i = 0; i < nBoundary; i++) {
    e[2 * i] = i;
    e[2 * i + 1] = (i + 1) % nBoundary;
  }
  return tf.ndarray(e, [nBoundary, 2]);
}

describe("CDT (constrained Delaunay)", () => {

  // ==========================================================================
  test("cdt: convex hull triangulation of N random points", () => {
    const tf = getTf();
    const points = ringPoints(tf, 0, 50);

    const { faces, points: out } = tf.cdt(points);

    assert(faces.shape.length === 2 && faces.shape[1] === 3,
           `expected (K, 3) faces, got ${JSON.stringify(faces.shape)}`);
    assert(faces.dtype === "int32", `expected int32 faces, got ${faces.dtype}`);
    assert(out.shape.length === 2 && out.shape[1] === 2,
           `expected (M, 2) out points, got ${JSON.stringify(out.shape)}`);
    assert(out.dtype === "float32",
           `expected float32 out points, got ${out.dtype}`);
    log(`  ${faces.shape[0]} triangles over ${out.shape[0]} points`,
        "line-pass");

    faces.delete(); out.delete(); points.delete();
  });

  // ==========================================================================
  test("cdt: convex hull triangulation of N random points (float64)", () => {
    const tf = getTf();
    const points = ringPoints(tf, 0, 50, "float64");

    const { faces, points: out } = tf.cdt(points);

    assert(faces.shape.length === 2 && faces.shape[1] === 3,
           `expected (K, 3) faces, got ${JSON.stringify(faces.shape)}`);
    assert(faces.dtype === "int32", `expected int32 faces, got ${faces.dtype}`);
    assert(out.shape.length === 2 && out.shape[1] === 2,
           `expected (M, 2) out points, got ${JSON.stringify(out.shape)}`);
    assert(out.dtype === "float64",
           `expected float64 out points, got ${out.dtype}`);
    log(`  ${faces.shape[0]} triangles over ${out.shape[0]} points`,
        "line-pass");

    faces.delete(); out.delete(); points.delete();
  });

  // ==========================================================================
  test("cdt: closed polygon outline + Steiner points", () => {
    const tf = getTf();
    const nBoundary = 32;
    const points = ringPoints(tf, nBoundary, 64);
    const edges = closingEdges(tf, nBoundary);

    const { faces, points: out } = tf.cdt(points, { edges });
    const fullHull = tf.cdt(points);

    assert(faces.shape[0] < fullHull.faces.shape[0],
           "interior triangulation should have strictly fewer faces than " +
           "the unconstrained convex-hull triangulation");
    log(`  interior=${faces.shape[0]}  hull=${fullHull.faces.shape[0]}`,
        "line-pass");

    faces.delete(); out.delete(); edges.delete(); points.delete();
    fullHull.faces.delete(); fullHull.points.delete();
  });

  // ==========================================================================
  test("cdt: closed polygon outline + Steiner points (float64)", () => {
    const tf = getTf();
    const nBoundary = 32;
    const points = ringPoints(tf, nBoundary, 64, "float64");
    const edges = closingEdges(tf, nBoundary);

    const { faces, points: out } = tf.cdt(points, { edges });
    const fullHull = tf.cdt(points);

    assert(out.dtype === "float64",
           `expected float64 out points, got ${out.dtype}`);
    assert(faces.shape[0] < fullHull.faces.shape[0],
           "interior triangulation should have strictly fewer faces than " +
           "the unconstrained convex-hull triangulation");
    log(`  interior=${faces.shape[0]}  hull=${fullHull.faces.shape[0]}`,
        "line-pass");

    faces.delete(); out.delete(); edges.delete(); points.delete();
    fullHull.faces.delete(); fullHull.points.delete();
  });

  // ==========================================================================
  test("cdt: edgeMask all-false drops all triangles", () => {
    const tf = getTf();
    const nBoundary = 16;
    const points = ringPoints(tf, nBoundary);
    const edges = closingEdges(tf, nBoundary);
    // Mark every constraint as non-boundary — parity never flips, so
    // the parity filter (interior == odd) keeps no triangles.
    const edgeMask = tf.ndarray(new Int8Array(nBoundary), [nBoundary]);

    const { faces, points: out } = tf.cdt(points, { edges, edgeMask });
    assert(faces.shape[0] === 0,
           `expected 0 interior triangles, got ${faces.shape[0]}`);
    log("  all-false mask → empty interior", "line-pass");

    faces.delete(); out.delete();
    edgeMask.delete(); edges.delete(); points.delete();
  });

  // ==========================================================================
  test("cdt: edgeMask all-false drops all triangles (float64)", () => {
    const tf = getTf();
    const nBoundary = 16;
    const points = ringPoints(tf, nBoundary, 0, "float64");
    const edges = closingEdges(tf, nBoundary);
    // Mark every constraint as non-boundary — parity never flips, so
    // the parity filter (interior == odd) keeps no triangles.
    const edgeMask = tf.ndarray(new Int8Array(nBoundary), [nBoundary]);

    const { faces, points: out } = tf.cdt(points, { edges, edgeMask });
    assert(out.dtype === "float64",
           `expected float64 out points, got ${out.dtype}`);
    assert(faces.shape[0] === 0,
           `expected 0 interior triangles, got ${faces.shape[0]}`);
    log("  all-false mask → empty interior", "line-pass");

    faces.delete(); out.delete();
    edgeMask.delete(); edges.delete(); points.delete();
  });

  // ==========================================================================
  test("cdt: returnIndexMap round-trip", () => {
    const tf = getTf();
    const nBoundary = 32;
    const points = ringPoints(tf, nBoundary, 64);
    const edges = closingEdges(tf, nBoundary);

    const r = tf.cdt(points, { edges, returnIndexMap: true });
    const { faces, points: out, indexMap } = r;

    assert(indexMap.f.shape[0] === points.shape[0],
           `f.size should equal input count`);
    assert(indexMap.keptIds.shape[0] === out.shape[0],
           `keptIds.size should equal output count`);

    // Round-trip: out_points[f[i]] ≈ points[i] for inputs that survived
    // the parity filter (sentinel = f.size).
    const sentinel = indexMap.f.shape[0];
    const f = indexMap.f.data;
    const ptsIn = points.data;
    const ptsOut = out.data;
    let preserved = 0;
    let maxErr = 0;
    for (let i = 0; i < indexMap.f.shape[0]; i++) {
      const j = f[i];
      if (j === sentinel) continue;
      preserved++;
      const dx = ptsOut[2 * j] - ptsIn[2 * i];
      const dy = ptsOut[2 * j + 1] - ptsIn[2 * i + 1];
      const e = Math.hypot(dx, dy);
      if (e > maxErr) maxErr = e;
    }
    assert(preserved >= nBoundary,
           `at least the boundary inputs (${nBoundary}) should be preserved, ` +
           `got ${preserved}`);
    assert(maxErr < 1e-2,
           `index-map round-trip error too large: ${maxErr}`);
    log(`  preserved=${preserved}/${indexMap.f.shape[0]}  ` +
        `maxErr=${maxErr.toExponential(2)}`, "line-pass");

    faces.delete(); out.delete(); indexMap.delete();
    edges.delete(); points.delete();
  });

  // ==========================================================================
  test("cdt: returnIndexMap round-trip (float64)", () => {
    const tf = getTf();
    const nBoundary = 32;
    const points = ringPoints(tf, nBoundary, 64, "float64");
    const edges = closingEdges(tf, nBoundary);

    const r = tf.cdt(points, { edges, returnIndexMap: true });
    const { faces, points: out, indexMap } = r;

    assert(out.dtype === "float64",
           `expected float64 out points, got ${out.dtype}`);
    assert(indexMap.f.shape[0] === points.shape[0],
           `f.size should equal input count`);
    assert(indexMap.keptIds.shape[0] === out.shape[0],
           `keptIds.size should equal output count`);

    // Round-trip: out_points[f[i]] ≈ points[i] for inputs that survived
    // the parity filter (sentinel = f.size).
    const sentinel = indexMap.f.shape[0];
    const f = indexMap.f.data;
    const ptsIn = points.data;
    const ptsOut = out.data;
    let preserved = 0;
    let maxErr = 0;
    for (let i = 0; i < indexMap.f.shape[0]; i++) {
      const j = f[i];
      if (j === sentinel) continue;
      preserved++;
      const dx = ptsOut[2 * j] - ptsIn[2 * i];
      const dy = ptsOut[2 * j + 1] - ptsIn[2 * i + 1];
      const e = Math.hypot(dx, dy);
      if (e > maxErr) maxErr = e;
    }
    assert(preserved >= nBoundary,
           `at least the boundary inputs (${nBoundary}) should be preserved, ` +
           `got ${preserved}`);
    assert(maxErr < 1e-2,
           `index-map round-trip error too large: ${maxErr}`);
    log(`  preserved=${preserved}/${indexMap.f.shape[0]}  ` +
        `maxErr=${maxErr.toExponential(2)}`, "line-pass");

    faces.delete(); out.delete(); indexMap.delete();
    edges.delete(); points.delete();
  });

  // ==========================================================================
  test("async cdt produces the same result as sync", async () => {
    const tf = getTf();
    const nBoundary = 16;
    const points = ringPoints(tf, nBoundary, 32);
    const edges = closingEdges(tf, nBoundary);

    const sync = tf.cdt(points, { edges });
    const asyncR = await tf.async.cdt(points, { edges });

    assert(sync.faces.shape[0] === asyncR.faces.shape[0],
           `face count mismatch: sync=${sync.faces.shape[0]} ` +
           `async=${asyncR.faces.shape[0]}`);
    assert(sync.points.shape[0] === asyncR.points.shape[0],
           `point count mismatch`);
    log(`  ${sync.faces.shape[0]} triangles, sync == async`, "line-pass");

    sync.faces.delete(); sync.points.delete();
    asyncR.faces.delete(); asyncR.points.delete();
    edges.delete(); points.delete();
  });

  // ==========================================================================
  test("async cdt produces the same result as sync (float64)", async () => {
    const tf = getTf();
    const nBoundary = 16;
    const points = ringPoints(tf, nBoundary, 32, "float64");
    const edges = closingEdges(tf, nBoundary);

    const sync = tf.cdt(points, { edges });
    const asyncR = await tf.async.cdt(points, { edges });

    assert(sync.points.dtype === "float64",
           `expected float64 sync points, got ${sync.points.dtype}`);
    assert(asyncR.points.dtype === "float64",
           `expected float64 async points, got ${asyncR.points.dtype}`);
    assert(sync.faces.shape[0] === asyncR.faces.shape[0],
           `face count mismatch: sync=${sync.faces.shape[0]} ` +
           `async=${asyncR.faces.shape[0]}`);
    assert(sync.points.shape[0] === asyncR.points.shape[0],
           `point count mismatch`);
    log(`  ${sync.faces.shape[0]} triangles, sync == async`, "line-pass");

    sync.faces.delete(); sync.points.delete();
    asyncR.faces.delete(); asyncR.points.delete();
    edges.delete(); points.delete();
  });
});
