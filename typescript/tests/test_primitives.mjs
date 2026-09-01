import { describe, test, log, assert, approx, getTf } from "./harness.mjs";

// ============================================================================
// NDArray-based primitive factories
// ============================================================================

describe("Primitive factories (NDArray overloads)", () => {

  test("point from NDArray (single)", () => {
    const tf = getTf();
    const arr = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const p = tf.point(arr);

    assert(p.type === "point", `type: ${p.type}`);
    assert(p.ndim === 1, `ndim: ${p.ndim}`);
    approx(p.data[0], 1); approx(p.data[1], 2); approx(p.data[2], 3);
    log("  point(ndarray[3]) → point[3]", "line-pass");

    p.delete();
  });

  test("point from NDArray (batch)", () => {
    const tf = getTf();
    const arr = tf.ndarray(new Float32Array([1,2,3,4,5,6]), [2, 3]);
    const p = tf.point(arr);

    assert(p.type === "point", `type: ${p.type}`);
    assert(p.isBatch === true, `isBatch: ${p.isBatch}`);
    assert(p.shape[0] === 2 && p.shape[1] === 3, `shape: [${p.shape}]`);
    log("  point(ndarray[2,3]) → batch point[2,3]", "line-pass");

    p.delete();
  });

  test("vector from NDArray", () => {
    const tf = getTf();
    const arr = tf.ndarray(new Float32Array([0, 1, 0]), [3]);
    const v = tf.vector(arr);

    assert(v.type === "vector", `type: ${v.type}`);
    approx(v.data[1], 1);
    log("  vector(ndarray[3]) → vector[3]", "line-pass");

    v.delete();
  });

  test("segment from two NDArrays (single)", () => {
    const tf = getTf();
    const starts = tf.ndarray(new Float32Array([0, 0, 0]), [3]);
    const ends = tf.ndarray(new Float32Array([1, 1, 1]), [3]);

    const s = tf.segment(starts, ends);
    assert(s.type === "segment", `type: ${s.type}`);
    assert(s.shape[0] === 2 && s.shape[1] === 3, `shape: [${s.shape}]`);
    approx(s.data[0], 0); approx(s.data[3], 1);
    log("  segment(ndarray[3], ndarray[3]) → segment[2,3]", "line-pass");

    s.delete(); starts.delete(); ends.delete();
  });

  test("segment from two NDArrays (batch)", () => {
    const tf = getTf();
    const starts = tf.ndarray(new Float32Array([0,0,0, 1,1,1]), [2, 3]);
    const ends = tf.ndarray(new Float32Array([2,2,2, 3,3,3]), [2, 3]);

    const s = tf.segment(starts, ends);
    assert(s.type === "segment", `type: ${s.type}`);
    assert(s.isBatch === true, `isBatch: ${s.isBatch}`);
    assert(s.shape[0] === 2 && s.shape[1] === 2 && s.shape[2] === 3, `shape: [${s.shape}]`);
    approx(s.data[0], 0, "seg0 start x");
    approx(s.data[3], 2, "seg0 end x");
    approx(s.data[6], 1, "seg1 start x");
    approx(s.data[9], 3, "seg1 end x");
    log("  segment(ndarray[2,3], ndarray[2,3]) → batch segment[2,2,3]", "line-pass");

    s.delete(); starts.delete(); ends.delete();
  });

  test("ray from two NDArrays (single)", () => {
    const tf = getTf();
    const origin = tf.ndarray(new Float32Array([0, 0, 0]), [3]);
    const dir = tf.ndarray(new Float32Array([1, 0, 0]), [3]);

    const r = tf.ray(origin, dir);
    assert(r.type === "ray", `type: ${r.type}`);
    assert(r.shape[0] === 2 && r.shape[1] === 3, `shape: [${r.shape}]`);
    approx(r.data[0], 0, "origin x"); approx(r.data[3], 1, "dir x");
    log("  ray(ndarray[3], ndarray[3]) → ray[2,3]", "line-pass");

    r.delete(); origin.delete(); dir.delete();
  });

  test("ray from two NDArrays (batch)", () => {
    const tf = getTf();
    const origins = tf.ndarray(new Float32Array([0,0,0, 1,1,1, 2,2,2]), [3, 3]);
    const dirs = tf.ndarray(new Float32Array([1,0,0, 0,1,0, 0,0,1]), [3, 3]);

    const r = tf.ray(origins, dirs);
    assert(r.type === "ray", `type: ${r.type}`);
    assert(r.isBatch === true, `isBatch`);
    assert(r.shape[0] === 3 && r.shape[1] === 2 && r.shape[2] === 3, `shape: [${r.shape}]`);
    log("  ray(ndarray[3,3], ndarray[3,3]) → batch ray[3,2,3]", "line-pass");

    r.delete(); origins.delete(); dirs.delete();
  });

  test("triangle from three NDArrays (single)", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float32Array([0, 0, 0]), [3]);
    const b = tf.ndarray(new Float32Array([1, 0, 0]), [3]);
    const c = tf.ndarray(new Float32Array([0, 1, 0]), [3]);

    const tri = tf.triangle(a, b, c);
    assert(tri.type === "triangle", `type: ${tri.type}`);
    assert(tri.shape[0] === 3 && tri.shape[1] === 3, `shape: [${tri.shape}]`);
    approx(tri.data[0], 0); approx(tri.data[3], 1); approx(tri.data[7], 1);
    log("  triangle(ndarray[3], ndarray[3], ndarray[3]) → triangle[3,3]", "line-pass");

    tri.delete(); a.delete(); b.delete(); c.delete();
  });

  test("triangle from three NDArrays (batch)", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float32Array([0,0,0, 1,1,1]), [2, 3]);
    const b = tf.ndarray(new Float32Array([1,0,0, 2,1,1]), [2, 3]);
    const c = tf.ndarray(new Float32Array([0,1,0, 1,2,1]), [2, 3]);

    const tri = tf.triangle(a, b, c);
    assert(tri.type === "triangle", `type: ${tri.type}`);
    assert(tri.isBatch === true, `isBatch`);
    assert(tri.shape[0] === 2 && tri.shape[1] === 3 && tri.shape[2] === 3, `shape: [${tri.shape}]`);
    log("  triangle(ndarray[2,3]x3) → batch triangle[2,3,3]", "line-pass");

    tri.delete(); a.delete(); b.delete(); c.delete();
  });

  test("line from two NDArrays", () => {
    const tf = getTf();
    const origin = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const dir = tf.ndarray(new Float32Array([0, 0, 1]), [3]);

    const l = tf.line(origin, dir);
    assert(l.type === "line", `type: ${l.type}`);
    assert(l.shape[0] === 2 && l.shape[1] === 3, `shape: [${l.shape}]`);
    approx(l.data[0], 1); approx(l.data[5], 1);
    log("  line(ndarray[3], ndarray[3]) → line[2,3]", "line-pass");

    l.delete(); origin.delete(); dir.delete();
  });

  test("aabb from two NDArrays", () => {
    const tf = getTf();
    const mins = tf.ndarray(new Float32Array([0, 0, 0]), [3]);
    const maxs = tf.ndarray(new Float32Array([1, 2, 3]), [3]);

    const box = tf.aabb(mins, maxs);
    assert(box.type === "aabb", `type: ${box.type}`);
    assert(box.shape[0] === 2 && box.shape[1] === 3, `shape: [${box.shape}]`);
    approx(box.data[0], 0); approx(box.data[3], 1);
    approx(box.data[4], 2); approx(box.data[5], 3);
    log("  aabb(ndarray[3], ndarray[3]) → aabb[2,3]", "line-pass");

    box.delete(); mins.delete(); maxs.delete();
  });

  test("polygon from NDArray", () => {
    const tf = getTf();
    const arr = tf.ndarray(new Float32Array([0,0, 1,0, 1,1, 0,1]), [4, 2]);
    const p = tf.polygon(arr);

    assert(p.type === "polygon", `type: ${p.type}`);
    assert(p.shape[0] === 4 && p.shape[1] === 2, `shape: [${p.shape}]`);
    approx(p.data[2], 1); approx(p.data[5], 1);
    log("  polygon(ndarray[4,2]) → polygon[4,2]", "line-pass");

    p.delete();
  });

});

// ============================================================================
// Variadic, flat-array, and Primitive-based overloads
// ============================================================================

describe("Primitive factories (other overloads)", () => {

  // ---------- point ----------
  test("point variadic", () => {
    const tf = getTf();
    const p = tf.point(1, 2, 3);
    assert(p.type === "point", `type: ${p.type}`);
    assert(p.isBatch === false, "single not batch");
    approx(p.data[0], 1); approx(p.data[1], 2); approx(p.data[2], 3);
    log("  point(1, 2, 3)", "line-pass");
    p.delete();
  });

  test("point from Float32Array", () => {
    const tf = getTf();
    const p = tf.point(new Float32Array([4, 5, 6]));
    assert(p.type === "point");
    approx(p.data[0], 4);
    log("  point(Float32Array)", "line-pass");
    p.delete();
  });

  test("point from number[]", () => {
    const tf = getTf();
    const p = tf.point([7, 8, 9]);
    assert(p.type === "point");
    approx(p.data[0], 7);
    log("  point([7,8,9])", "line-pass");
    p.delete();
  });

  test("point batch from flat + dims", () => {
    const tf = getTf();
    const p = tf.point(new Float32Array([1,2,3, 4,5,6]), 3);
    assert(p.type === "point");
    assert(p.isBatch === true, "should be batch");
    assert(p.count === 2, `count: ${p.count}`);
    assert(p.shape[0] === 2 && p.shape[1] === 3, `shape: [${p.shape}]`);
    log("  point(data, 3) → batch[2,3]", "line-pass");

    const p0 = p.at(0);
    assert(p0.isBatch === false, "at(0) not batch");
    approx(p0.data[0], 1);
    log("  batch.at(0) works", "line-pass");
    p0.delete();
    p.delete();
  });

  // ---------- vector ----------
  test("vector variadic", () => {
    const tf = getTf();
    const v = tf.vector(0, 1, 0);
    assert(v.type === "vector");
    approx(v.data[1], 1);
    log("  vector(0, 1, 0)", "line-pass");
    v.delete();
  });

  test("vector from Float32Array", () => {
    const tf = getTf();
    const v = tf.vector(new Float32Array([1, 0, 0]));
    assert(v.type === "vector");
    approx(v.data[0], 1);
    log("  vector(Float32Array)", "line-pass");
    v.delete();
  });

  test("vector batch from flat + dims", () => {
    const tf = getTf();
    const v = tf.vector(new Float32Array([1,0,0, 0,1,0, 0,0,1]), 3);
    assert(v.isBatch === true);
    assert(v.count === 3, `count: ${v.count}`);
    log("  vector(data, 3) → batch[3,3]", "line-pass");
    v.delete();
  });

  // ---------- segment from Primitives ----------
  test("segment from Point + Point", () => {
    const tf = getTf();
    const a = tf.point(0, 0, 0);
    const b = tf.point(1, 1, 1);
    const s = tf.segment(a, b);
    assert(s.type === "segment");
    assert(s.shape[0] === 2 && s.shape[1] === 3, `shape: [${s.shape}]`);
    approx(s.data[0], 0); approx(s.data[3], 1);
    log("  segment(Point, Point)", "line-pass");
    s.delete(); b.delete(); a.delete();
  });

  test("segment from flat array", () => {
    const tf = getTf();
    const s = tf.segment([0,0,0, 1,1,1]);
    assert(s.type === "segment");
    assert(s.shape[0] === 2 && s.shape[1] === 3, `shape: [${s.shape}]`);
    log("  segment([0,0,0, 1,1,1])", "line-pass");
    s.delete();
  });

  test("segment batch from flat + dims", () => {
    const tf = getTf();
    const s = tf.segment(new Float32Array([0,0,0, 1,1,1, 2,2,2, 3,3,3]), 3);
    assert(s.isBatch === true);
    assert(s.count === 2, `count: ${s.count}`);
    log("  segment(data, 3) → batch[2,2,3]", "line-pass");
    s.delete();
  });

  // ---------- triangle from Primitives ----------
  test("triangle from Point × 3", () => {
    const tf = getTf();
    const a = tf.point(0, 0, 0);
    const b = tf.point(1, 0, 0);
    const c = tf.point(0, 1, 0);
    const tri = tf.triangle(a, b, c);
    assert(tri.type === "triangle");
    assert(tri.shape[0] === 3 && tri.shape[1] === 3, `shape: [${tri.shape}]`);
    approx(tri.data[3], 1);
    log("  triangle(Point, Point, Point)", "line-pass");
    tri.delete(); c.delete(); b.delete(); a.delete();
  });

  test("triangle from flat array", () => {
    const tf = getTf();
    const tri = tf.triangle([0,0,0, 1,0,0, 0,1,0]);
    assert(tri.type === "triangle");
    assert(tri.shape[0] === 3 && tri.shape[1] === 3, `shape: [${tri.shape}]`);
    log("  triangle([...])", "line-pass");
    tri.delete();
  });

  test("triangle batch from flat + dims", () => {
    const tf = getTf();
    const tri = tf.triangle(new Float32Array([
      0,0,0, 1,0,0, 0,1,0,
      1,1,1, 2,1,1, 1,2,1,
    ]), 3);
    assert(tri.isBatch === true);
    assert(tri.count === 2, `count: ${tri.count}`);
    log("  triangle(data, 3) → batch[2,3,3]", "line-pass");
    tri.delete();
  });

  // ---------- ray from Primitives ----------
  test("ray from Point + Vector", () => {
    const tf = getTf();
    const o = tf.point(0, 0, 0);
    const d = tf.vector(0, 0, 1);
    const r = tf.ray(o, d);
    assert(r.type === "ray");
    assert(r.shape[0] === 2 && r.shape[1] === 3, `shape: [${r.shape}]`);
    approx(r.data[5], 1);
    log("  ray(Point, Vector)", "line-pass");
    r.delete(); d.delete(); o.delete();
  });

  test("ray from flat array", () => {
    const tf = getTf();
    const r = tf.ray([0,0,0, 1,0,0]);
    assert(r.type === "ray");
    assert(r.shape[0] === 2 && r.shape[1] === 3, `shape: [${r.shape}]`);
    log("  ray([...])", "line-pass");
    r.delete();
  });

  test("ray batch from flat + dims", () => {
    const tf = getTf();
    const r = tf.ray(new Float32Array([
      0,0,0, 1,0,0,
      0,0,0, 0,1,0,
    ]), 3);
    assert(r.isBatch === true);
    assert(r.count === 2, `count: ${r.count}`);
    log("  ray(data, 3) → batch[2,2,3]", "line-pass");
    r.delete();
  });

  // ---------- line from Primitives ----------
  test("line from Point + Vector", () => {
    const tf = getTf();
    const o = tf.point(1, 2, 3);
    const d = tf.vector(0, 0, 1);
    const l = tf.line(o, d);
    assert(l.type === "line");
    assert(l.shape[0] === 2 && l.shape[1] === 3, `shape: [${l.shape}]`);
    approx(l.data[0], 1); approx(l.data[5], 1);
    log("  line(Point, Vector)", "line-pass");
    l.delete(); d.delete(); o.delete();
  });

  test("line from flat array", () => {
    const tf = getTf();
    const l = tf.line([1,2,3, 0,0,1]);
    assert(l.type === "line");
    assert(l.shape[0] === 2 && l.shape[1] === 3, `shape: [${l.shape}]`);
    log("  line([...])", "line-pass");
    l.delete();
  });

  test("line batch from flat + dims", () => {
    const tf = getTf();
    const l = tf.line(new Float32Array([
      0,0,0, 1,0,0,
      0,0,0, 0,1,0,
    ]), 3);
    assert(l.isBatch === true);
    assert(l.count === 2, `count: ${l.count}`);
    log("  line(data, 3) → batch[2,2,3]", "line-pass");
    l.delete();
  });

  // ---------- plane ----------
  test("plane from Vector + offset", () => {
    const tf = getTf();
    const n = tf.vector(0, 0, 1);
    const pl = tf.plane(n, 5);
    assert(pl.type === "plane");
    approx(pl.data[2], 1); approx(pl.data[3], 5);
    log("  plane(Vector, offset)", "line-pass");
    pl.delete(); n.delete();
  });

  test("plane from NDArray + offset", () => {
    const tf = getTf();
    const n = tf.ndarray(new Float32Array([0, 1, 0]), [3]);
    const pl = tf.plane(n, 3);
    assert(pl.type === "plane");
    approx(pl.data[1], 1); approx(pl.data[3], 3);
    log("  plane(NDArray, offset)", "line-pass");
    pl.delete(); n.delete();
  });

  test("plane batch from packed NDArray coefficients", () => {
    const tf = getTf();
    const coefficients = tf.ndarray(new Float32Array([
      1, 0, 0, -2,
      0, 1, 0, 3,
    ]), [2, 4]);
    const planes = tf.plane(coefficients);

    assert(planes.type === "plane");
    assert(planes.isBatch === true, "should be a batch");
    assert(planes.count === 2, `count: ${planes.count}`);
    assert(planes.shape[0] === 2 && planes.shape[1] === 4, `shape: [${planes.shape}]`);

    coefficients.delete();
    approx(planes.data[3], -2, "first offset remains owned by planes");
    approx(planes.data[7], 3, "second offset remains owned by planes");
    log("  plane(ndarray[2,4]) → batch plane[2,4]", "line-pass");
    planes.delete();
  });

  test("packed plane owns dtype conversion and rejects malformed shapes", () => {
    const tf = getTf();
    const coefficients = tf.ndarray(new Float32Array([0, 0, 1, -4]), [4]);
    const plane64 = tf.plane(coefficients, { dtype: "float64" });

    coefficients.delete();
    assert(plane64.dtype === "float64", `dtype: ${plane64.dtype}`);
    approx(plane64.data[3], -4, "converted plane remains independently owned");
    plane64.delete();

    const malformed = tf.ndarray(new Float32Array([0, 0, 1]), [3]);
    let message = "";
    try { tf.plane(malformed); }
    catch (cause) { message = cause instanceof Error ? cause.message : String(cause); }
    assert(
      /plane coefficients must have shape \[4\] or \[N,4\]/.test(message),
      `expected packed-plane shape rejection, got: ${message}`,
    );
    malformed.delete();
    log("  packed plane validates shape and owns dtype conversion", "line-pass");
  });

  test("plane from flat [a,b,c,d]", () => {
    const tf = getTf();
    const pl = tf.plane([0, 0, 1, 2]);
    assert(pl.type === "plane");
    approx(pl.data[2], 1); approx(pl.data[3], 2);
    log("  plane([0,0,1,2])", "line-pass");
    pl.delete();
  });

  // ---------- aabb from Primitives ----------
  test("aabb from Point + Point", () => {
    const tf = getTf();
    const lo = tf.point(0, 0, 0);
    const hi = tf.point(1, 2, 3);
    const box = tf.aabb(lo, hi);
    assert(box.type === "aabb");
    assert(box.shape[0] === 2 && box.shape[1] === 3, `shape: [${box.shape}]`);
    approx(box.data[0], 0); approx(box.data[5], 3);
    log("  aabb(Point, Point)", "line-pass");
    box.delete(); hi.delete(); lo.delete();
  });

  test("aabb from flat array", () => {
    const tf = getTf();
    const box = tf.aabb([0,0,0, 1,2,3]);
    assert(box.type === "aabb");
    assert(box.shape[0] === 2 && box.shape[1] === 3, `shape: [${box.shape}]`);
    log("  aabb([0,0,0, 1,2,3])", "line-pass");
    box.delete();
  });

  test("aabb batch from flat + dims", () => {
    const tf = getTf();
    const box = tf.aabb(new Float32Array([
      0,0,0, 1,1,1,
      2,2,2, 3,3,3,
    ]), 3);
    assert(box.isBatch === true);
    assert(box.count === 2, `count: ${box.count}`);
    log("  aabb(data, 3) → batch[2,2,3]", "line-pass");
    box.delete();
  });

  // ---------- aabbFrom ----------
  test("aabbFrom NDArray points", () => {
    const tf = getTf();
    const points = tf.ndarray(new Float32Array([
      1, 2, 3,
      4, 5, 6,
      -1, 0, 10,
    ]), [3, 3]);

    const box = tf.aabbFrom(points);
    assert(box.type === "aabb", `type: ${box.type}`);
    assert(box.shape[0] === 2 && box.shape[1] === 3, `shape: [${box.shape}]`);
    approx(box.data[0], -1); approx(box.data[1], 0); approx(box.data[2], 3);
    approx(box.data[3], 4); approx(box.data[4], 5); approx(box.data[5], 10);
    log("  aabbFrom(ndarray[3,3]) → aabb[2,3]", "line-pass");

    box.delete(); points.delete();
  });

  test("aabbFrom mesh", () => {
    const tf = getTf();
    const faces = tf.ndarray(new Int32Array([0, 1, 2]), [1, 3]);
    const points = tf.ndarray(new Float32Array([
      0, 0, 0,
      3, 0, 0,
      0, 4, 0,
    ]), [3, 3]);
    const m = tf.mesh(faces, points);

    const box = tf.aabbFrom(m);
    assert(box.type === "aabb", `type: ${box.type}`);
    assert(box.shape[0] === 2 && box.shape[1] === 3, `shape: [${box.shape}]`);
    approx(box.data[0], 0); approx(box.data[1], 0); approx(box.data[2], 0);
    approx(box.data[3], 3); approx(box.data[4], 4); approx(box.data[5], 0);
    log("  aabbFrom(mesh) → aabb[2,3]", "line-pass");

    box.delete(); m.delete();
  });

  // ---------- obbFrom ----------
  test("obbFrom mesh", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);
    const obb = tf.obbFrom(box);

    assert(obb.origin.shape[0] === 3, `origin shape: [${obb.origin.shape}]`);
    assert(obb.axes.shape[0] === 3 && obb.axes.shape[1] === 3, `axes shape: [${obb.axes.shape}]`);
    assert(obb.extent.shape[0] === 3, `extent shape: [${obb.extent.shape}]`);

    // Extents should be close to box dimensions (2, 3, 4) in some order
    const e = Array.from(obb.extent.data).sort((a, b) => a - b);
    assert(Math.abs(e[0] - 2) < 0.1, `smallest extent ~2, got ${e[0]}`);
    assert(Math.abs(e[1] - 3) < 0.1, `middle extent ~3, got ${e[1]}`);
    assert(Math.abs(e[2] - 4) < 0.1, `largest extent ~4, got ${e[2]}`);
    log(`  obbFrom(box) → extents [${e.map(v => v.toFixed(2))}]`, "line-pass");

    obb.extent.delete(); obb.axes.delete(); obb.origin.delete();
    box.delete();
  });

  test("obbFrom NDArray points", () => {
    const tf = getTf();
    const points = tf.ndarray(new Float32Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1,
    ]), [4, 3]);

    const obb = tf.obbFrom(points);
    assert(obb.origin.shape[0] === 3, "origin [3]");
    assert(obb.axes.shape[0] === 3 && obb.axes.shape[1] === 3, "axes [3,3]");
    assert(obb.extent.shape[0] === 3, "extent [3]");
    log("  obbFrom(points) → origin/axes/extent shapes OK", "line-pass");

    obb.extent.delete(); obb.axes.delete(); obb.origin.delete();
    points.delete();
  });

  test("obbFrom float64 mesh", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.5, 16, 16, { dtype: "float64" });
    const obb = tf.obbFrom(sphere);
    assert(obb.origin.dtype === "float64", `origin dtype=${obb.origin.dtype}`);
    assert(obb.axes.dtype === "float64", `axes dtype=${obb.axes.dtype}`);
    assert(obb.extent.dtype === "float64", `extent dtype=${obb.extent.dtype}`);
    const e = Array.from(obb.extent.data).sort((a, b) => a - b);
    for (const v of e) assert(Math.abs(v - 3.0) < 0.1, `extent ~3.0 (diameter), got ${v}`);
    log("  obbFrom(sphere f64) → float64 in/out", "line-pass");
    obb.extent.delete(); obb.axes.delete(); obb.origin.delete();
    sphere.delete();
  });

  test("obbFrom async (float32 + float64)", async () => {
    const tf = getTf();
    const s32 = tf.sphereMesh(1.0, 16, 16);
    const s64 = tf.sphereMesh(1.0, 16, 16, { dtype: "float64" });
    const o32 = await tf.async.obbFrom(s32);
    const o64 = await tf.async.obbFrom(s64);
    assert(o32.extent.dtype === "float32", `async f32 dtype=${o32.extent.dtype}`);
    assert(o64.extent.dtype === "float64", `async f64 dtype=${o64.extent.dtype}`);
    log("  async obbFrom dispatch preserves dtype", "line-pass");
    o32.extent.delete(); o32.axes.delete(); o32.origin.delete();
    o64.extent.delete(); o64.axes.delete(); o64.origin.delete();
    s32.delete(); s64.delete();
  });

  // ---------- polygon ----------
  test("polygon from Point[]", () => {
    const tf = getTf();
    const pts = [
      tf.point(0, 0, 0),
      tf.point(1, 0, 0),
      tf.point(1, 1, 0),
      tf.point(0, 1, 0),
    ];
    const poly = tf.polygon(pts);
    assert(poly.type === "polygon");
    assert(poly.shape[0] === 4 && poly.shape[1] === 3, `shape: [${poly.shape}]`);
    approx(poly.data[3], 1);
    log("  polygon(Point[])", "line-pass");
    poly.delete();
    for (const p of pts) p.delete();
  });

  test("polygon from flat + dims", () => {
    const tf = getTf();
    const poly = tf.polygon(new Float32Array([0,0,0, 1,0,0, 1,1,0, 0,1,0]), 3);
    assert(poly.type === "polygon");
    assert(poly.shape[0] === 4 && poly.shape[1] === 3, `shape: [${poly.shape}]`);
    log("  polygon(data, 3)", "line-pass");
    poly.delete();
  });

  // ---------- batch properties ----------
  test("isBatch false on single", () => {
    const tf = getTf();
    const p = tf.point(1, 2, 3);
    assert(p.isBatch === false, `single isBatch: ${p.isBatch}`);
    assert(p.count === 1, `single count: ${p.count}`);
    log("  single: isBatch=false, count=1", "line-pass");
    p.delete();
  });

});
