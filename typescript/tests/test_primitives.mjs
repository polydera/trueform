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
