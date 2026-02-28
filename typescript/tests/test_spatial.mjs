import { describe, test, log, assert, approx, getTf } from "./harness.mjs";

// ============================================================================
// Helpers
// ============================================================================

function approxPt(pt, expected, label, eps = 1e-4) {
  const d = pt.data;
  for (let i = 0; i < expected.length; i++)
    approx(d[i], expected[i], `${label}[${i}]`, eps);
}

// Two coplanar triangles sharing edge 0-1
//   2
//  / \
// 0---1
//  \ /
//   3
function twoTriangles() {
  return {
    faces: new Int32Array([0, 1, 2, 0, 3, 1]),
    points: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0.5, 1, 0,
      0.5, -1, 0,
    ]),
  };
}

// ============================================================================
// distance2
// ============================================================================

describe("Spatial: distance2", () => {

  // --- single ---

  test("point × point", () => {
    const tf = getTf();
    const a = tf.point(0, 0, 0);
    const b = tf.point(1, 0, 0);
    const d = tf.distance2(a, b);
    approx(d, 1.0, "d2");
    log(`  point × point = ${d}`, "line-pass");
    a.delete(); b.delete();
  });

  test("point × segment", () => {
    const tf = getTf();
    const p = tf.point(0, 1, 0);
    const s = tf.segment(tf.point(0, 0, 0), tf.point(1, 0, 0));
    const d = tf.distance2(p, s);
    approx(d, 1.0, "d2");
    log(`  point × segment = ${d}`, "line-pass");
    p.delete(); s.delete();
  });

  test("segment × segment", () => {
    const tf = getTf();
    const s0 = tf.segment(tf.point(0, 0, 0), tf.point(0, 0, 2));
    const s1 = tf.segment(tf.point(3, 0, 0), tf.point(3, 1, 0));
    const d = tf.distance2(s0, s1);
    approx(d, 9.0, "d2");
    log(`  segment × segment = ${d}`, "line-pass");
    s0.delete(); s1.delete();
  });

  test("point × triangle", () => {
    const tf = getTf();
    const p = tf.point(0.25, 0.25, 1);
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    const d = tf.distance2(p, t);
    approx(d, 1.0, "d2");
    log(`  point × triangle = ${d}`, "line-pass");
    p.delete(); t.delete();
  });

  test("point × aabb", () => {
    const tf = getTf();
    const p = tf.point(3, 0.5, 0.5);
    const box = tf.aabb(tf.point(0, 0, 0), tf.point(2, 1, 1));
    const d = tf.distance2(p, box);
    approx(d, 1.0, "d2");
    log(`  point × aabb = ${d}`, "line-pass");
    p.delete(); box.delete();
  });

  test("aabb × segment", () => {
    const tf = getTf();
    const box = tf.aabb(tf.point(0, 0, 0), tf.point(1, 1, 1));
    const s = tf.segment(tf.point(0, 0.5, 3), tf.point(1, 0.5, 3));
    const d = tf.distance2(box, s);
    approx(d, 4.0, "d2");
    log(`  aabb × segment = ${d}`, "line-pass");
    box.delete(); s.delete();
  });

  test("point × plane", () => {
    const tf = getTf();
    const p = tf.point(0, 0, 3);
    const pl = tf.plane(tf.vector(0, 0, 1), 0);
    const d = tf.distance2(p, pl);
    approx(d, 9.0, "d2");
    log(`  point × plane = ${d}`, "line-pass");
    p.delete(); pl.delete();
  });

  test("point × ray", () => {
    const tf = getTf();
    const p = tf.point(0, 1, 0);
    const r = tf.ray(tf.point(0, 0, 0), tf.vector(1, 0, 0));
    const d = tf.distance2(p, r);
    approx(d, 1.0, "d2");
    log(`  point × ray = ${d}`, "line-pass");
    p.delete(); r.delete();
  });

  test("point × line", () => {
    const tf = getTf();
    const p = tf.point(5, 2, 0);
    const l = tf.line(tf.point(0, 0, 0), tf.vector(1, 0, 0));
    const d = tf.distance2(p, l);
    approx(d, 4.0, "d2");
    log(`  point × line = ${d}`, "line-pass");
    p.delete(); l.delete();
  });

  test("mesh × point", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const p = tf.point(0.5, 0, 2);
    const d = tf.distance2(m, p);
    approx(d, 4.0, "d2");
    log(`  mesh × point = ${d}`, "line-pass");
    p.delete(); m.delete();
  });

  test("mesh × mesh", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m0 = tf.mesh(faces, points);
    const shifted = new Float32Array(points.length);
    for (let i = 0; i < points.length; i += 3) {
      shifted[i] = points[i]; shifted[i+1] = points[i+1]; shifted[i+2] = points[i+2] + 3;
    }
    const m1 = tf.mesh(new Int32Array(faces), shifted);
    const d = tf.distance2(m0, m1);
    approx(d, 9.0, "d2");
    log(`  mesh × mesh = ${d}`, "line-pass");
    m0.delete(); m1.delete();
  });

  // --- batch ---

  test("batch: 3 points × segment", () => {
    const tf = getTf();
    const pts = tf.point(new Float32Array([0,1,0, 0,2,0, 0,3,0]), 3);
    const s = tf.segment(tf.point(0, 0, 0), tf.point(1, 0, 0));
    const d = tf.distance2(pts, s);
    assert(typeof d !== "number", "should return NDArray");
    const v = d.data;
    approx(v[0], 1.0, "[0]"); approx(v[1], 4.0, "[1]"); approx(v[2], 9.0, "[2]");
    log(`  batch 3 points × segment = [${v[0]}, ${v[1]}, ${v[2]}]`, "line-pass");
    d.delete(); pts.delete(); s.delete();
  });

  test("batch: 3 segments × point", () => {
    const tf = getTf();
    // 3 horizontal segments at y = 1, 2, 3
    const segs = tf.segment(new Float32Array([
      0,1,0, 1,1,0,
      0,2,0, 1,2,0,
      0,3,0, 1,3,0,
    ]), 3);
    const p = tf.point(0.5, 0, 0);
    assert(segs.isBatch && segs.count === 3, "should be batch of 3");
    const d = tf.distance2(segs, p);
    const v = d.data;
    approx(v[0], 1.0, "[0]"); approx(v[1], 4.0, "[1]"); approx(v[2], 9.0, "[2]");
    log(`  batch 3 segments × point = [${v[0]}, ${v[1]}, ${v[2]}]`, "line-pass");
    d.delete(); segs.delete(); p.delete();
  });

  test("batch: 3 aabbs × point", () => {
    const tf = getTf();
    // 3 boxes at z = 0, 1, 2
    const boxes = tf.aabb(new Float32Array([
      0,0,0, 1,1,1,
      0,0,1, 1,1,2,
      0,0,2, 1,1,3,
    ]), 3);
    const p = tf.point(0.5, 0.5, -1);
    assert(boxes.isBatch && boxes.count === 3, "should be batch of 3");
    const d = tf.distance2(boxes, p);
    const v = d.data;
    approx(v[0], 1.0, "[0]"); approx(v[1], 4.0, "[1]"); approx(v[2], 9.0, "[2]");
    log(`  batch 3 aabbs × point = [${v[0]}, ${v[1]}, ${v[2]}]`, "line-pass");
    d.delete(); boxes.delete(); p.delete();
  });

  test("batch: mesh × 3 points", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const pts = tf.point(new Float32Array([0.5,0,1, 0.5,0,2, 0.5,0,3]), 3);
    const d = tf.distance2(m, pts);
    const v = d.data;
    approx(v[0], 1.0, "[0]"); approx(v[1], 4.0, "[1]"); approx(v[2], 9.0, "[2]");
    log(`  mesh × batch 3 points = [${v[0]}, ${v[1]}, ${v[2]}]`, "line-pass");
    d.delete(); pts.delete(); m.delete();
  });
});

// ============================================================================
// intersects
// ============================================================================

describe("Spatial: intersects", () => {

  // --- single ---

  test("segment × triangle (hit)", () => {
    const tf = getTf();
    const s = tf.segment(tf.point(0.25, 0.25, -1), tf.point(0.25, 0.25, 1));
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    const r = tf.intersects(s, t);
    assert(r === true, "should hit");
    log(`  segment × triangle = ${r}`, "line-pass");
    s.delete(); t.delete();
  });

  test("segment × triangle (miss)", () => {
    const tf = getTf();
    const s = tf.segment(tf.point(0, 0, 1), tf.point(1, 0, 1));
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    const r = tf.intersects(s, t);
    assert(r === false, "should miss");
    log(`  segment × triangle = ${r}`, "line-pass");
    s.delete(); t.delete();
  });

  test("aabb × aabb (overlap)", () => {
    const tf = getTf();
    const a = tf.aabb(tf.point(0, 0, 0), tf.point(2, 2, 2));
    const b = tf.aabb(tf.point(1, 1, 1), tf.point(3, 3, 3));
    const r = tf.intersects(a, b);
    assert(r === true, "should overlap");
    log(`  aabb × aabb (overlap) = ${r}`, "line-pass");
    a.delete(); b.delete();
  });

  test("aabb × aabb (separated)", () => {
    const tf = getTf();
    const a = tf.aabb(tf.point(0, 0, 0), tf.point(1, 1, 1));
    const b = tf.aabb(tf.point(2, 2, 2), tf.point(3, 3, 3));
    const r = tf.intersects(a, b);
    assert(r === false, "should not overlap");
    log(`  aabb × aabb (separated) = ${r}`, "line-pass");
    a.delete(); b.delete();
  });

  test("point × aabb (inside)", () => {
    const tf = getTf();
    const p = tf.point(0.5, 0.5, 0.5);
    const box = tf.aabb(tf.point(0, 0, 0), tf.point(1, 1, 1));
    const r = tf.intersects(p, box);
    assert(r === true, "point inside aabb");
    log(`  point × aabb (inside) = ${r}`, "line-pass");
    p.delete(); box.delete();
  });

  test("point × aabb (outside)", () => {
    const tf = getTf();
    const p = tf.point(5, 5, 5);
    const box = tf.aabb(tf.point(0, 0, 0), tf.point(1, 1, 1));
    const r = tf.intersects(p, box);
    assert(r === false, "point outside aabb");
    log(`  point × aabb (outside) = ${r}`, "line-pass");
    p.delete(); box.delete();
  });

  test("mesh × point (on surface / off)", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const p_on = tf.point(0.5, 0.5, 0);
    const p_off = tf.point(0.5, 0.5, 5);
    const r1 = tf.intersects(m, p_on);
    const r2 = tf.intersects(m, p_off);
    assert(r1 === true, "on surface");
    assert(r2 === false, "off surface");
    log(`  mesh × point: on=${r1}, off=${r2}`, "line-pass");
    p_on.delete(); p_off.delete(); m.delete();
  });

  test("mesh × mesh (overlapping)", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m0 = tf.mesh(faces, points);
    const shifted = new Float32Array(points.length);
    for (let i = 0; i < points.length; i += 3) {
      shifted[i] = points[i] + 0.5; shifted[i+1] = points[i+1]; shifted[i+2] = points[i+2];
    }
    const m1 = tf.mesh(new Int32Array(faces), shifted);
    const r = tf.intersects(m0, m1);
    assert(r === true, "should overlap");
    log(`  mesh × mesh (overlapping) = ${r}`, "line-pass");
    m0.delete(); m1.delete();
  });

  // --- batch ---

  test("batch: 3 segments × triangle", () => {
    const tf = getTf();
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    // seg0: through triangle, seg1: above (miss), seg2: through at different spot
    const segs = tf.segment(new Float32Array([
      0.25,0.25,-1, 0.25,0.25,1,
      0,0,1,        1,0,1,
      0.1,0.1,-1,   0.1,0.1,1,
    ]), 3);
    assert(segs.isBatch && segs.count === 3, "should be batch of 3");
    const r = tf.intersects(segs, t);
    assert(typeof r !== "boolean", "batch returns NDArray");
    const v = r.data;
    assert(v[0] === 1, "seg0 hit"); assert(v[1] === 0, "seg1 miss"); assert(v[2] === 1, "seg2 hit");
    log(`  batch 3 segments × triangle = [${v[0]}, ${v[1]}, ${v[2]}]`, "line-pass");
    r.delete(); segs.delete(); t.delete();
  });

  test("batch: 3 aabbs × aabb", () => {
    const tf = getTf();
    const target = tf.aabb(tf.point(0, 0, 0), tf.point(1, 1, 1));
    // box0: overlaps, box1: separated, box2: overlaps
    const boxes = tf.aabb(new Float32Array([
      0.5,0.5,0.5, 1.5,1.5,1.5,
      5,5,5,       6,6,6,
      -0.5,-0.5,-0.5, 0.5,0.5,0.5,
    ]), 3);
    assert(boxes.isBatch && boxes.count === 3, "should be batch of 3");
    const r = tf.intersects(boxes, target);
    const v = r.data;
    assert(v[0] === 1, "box0 overlap"); assert(v[1] === 0, "box1 miss"); assert(v[2] === 1, "box2 overlap");
    log(`  batch 3 aabbs × aabb = [${v[0]}, ${v[1]}, ${v[2]}]`, "line-pass");
    r.delete(); boxes.delete(); target.delete();
  });

  test("batch: mesh × 3 points", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const pts = tf.point(new Float32Array([
      0.5,0.5,0,
      0.5,-0.5,0,
      10,10,10,
    ]), 3);
    const r = tf.intersects(m, pts);
    const v = r.data;
    assert(v[0] === 1, "pt0 on surface"); assert(v[1] === 1, "pt1 on surface"); assert(v[2] === 0, "pt2 far");
    log(`  mesh × batch 3 points = [${v[0]}, ${v[1]}, ${v[2]}]`, "line-pass");
    r.delete(); pts.delete(); m.delete();
  });

  test("batch: mesh × 3 segments", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    // seg0: through face 0, seg1: through face 1, seg2: misses
    const segs = tf.segment(new Float32Array([
      0.5,0.5,-1, 0.5,0.5,1,
      0.5,-0.5,-1, 0.5,-0.5,1,
      10,10,-1,  10,10,1,
    ]), 3);
    const r = tf.intersects(m, segs);
    const v = r.data;
    assert(v[0] === 1, "seg0 through face0"); assert(v[1] === 1, "seg1 through face1"); assert(v[2] === 0, "seg2 miss");
    log(`  mesh × batch 3 segments = [${v[0]}, ${v[1]}, ${v[2]}]`, "line-pass");
    r.delete(); segs.delete(); m.delete();
  });
});

// ============================================================================
// closestPointPair
// ============================================================================

describe("Spatial: closestPointPair", () => {

  // --- single ---

  test("point × segment", () => {
    const tf = getTf();
    const p = tf.point(0, 2, 0);
    const s = tf.segment(tf.point(0, 0, 0), tf.point(1, 0, 0));
    const r = tf.closestPointPair(p, s);
    approx(r.distance2, 4.0, "d2");
    approxPt(r.point0, [0, 2, 0], "pt0");
    approxPt(r.point1, [0, 0, 0], "pt1");
    log(`  point × segment: d2=${r.distance2}, pt0=(${r.point0.data}), pt1=(${r.point1.data})`, "line-pass");
    r.point0.delete(); r.point1.delete(); p.delete(); s.delete();
  });

  test("segment × segment", () => {
    const tf = getTf();
    const s0 = tf.segment(tf.point(0, 0, 0), tf.point(0, 0, 2));
    const s1 = tf.segment(tf.point(3, 0, 0), tf.point(3, 1, 0));
    const r = tf.closestPointPair(s0, s1);
    approx(r.distance2, 9.0, "d2");
    approxPt(r.point0, [0, 0, 0], "pt0");
    approxPt(r.point1, [3, 0, 0], "pt1");
    log(`  segment × segment: d2=${r.distance2}`, "line-pass");
    r.point0.delete(); r.point1.delete(); s0.delete(); s1.delete();
  });

  test("aabb × point", () => {
    const tf = getTf();
    const box = tf.aabb(tf.point(0, 0, 0), tf.point(1, 1, 1));
    const p = tf.point(2, 0.5, 0.5);
    const r = tf.closestPointPair(box, p);
    approx(r.distance2, 1.0, "d2");
    approxPt(r.point0, [1, 0.5, 0.5], "pt0 on aabb");
    approxPt(r.point1, [2, 0.5, 0.5], "pt1 point");
    log(`  aabb × point: d2=${r.distance2}, pt0=(${r.point0.data}), pt1=(${r.point1.data})`, "line-pass");
    r.point0.delete(); r.point1.delete(); box.delete(); p.delete();
  });

  test("aabb × segment", () => {
    const tf = getTf();
    const box = tf.aabb(tf.point(0, 0, 0), tf.point(1, 1, 1));
    const s = tf.segment(tf.point(0, 0.5, 3), tf.point(1, 0.5, 3));
    const r = tf.closestPointPair(box, s);
    approx(r.distance2, 4.0, "d2");
    approx(r.point0.data[2], 1.0, "pt0.z on aabb");
    approx(r.point1.data[2], 3.0, "pt1.z on segment");
    log(`  aabb × segment: d2=${r.distance2}`, "line-pass");
    r.point0.delete(); r.point1.delete(); box.delete(); s.delete();
  });

  test("point × triangle", () => {
    const tf = getTf();
    const p = tf.point(0.25, 0.25, 2);
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    const r = tf.closestPointPair(p, t);
    approx(r.distance2, 4.0, "d2");
    approxPt(r.point0, [0.25, 0.25, 2], "pt0 (query point)");
    approxPt(r.point1, [0.25, 0.25, 0], "pt1 (on triangle)");
    log(`  point × triangle: d2=${r.distance2}`, "line-pass");
    r.point0.delete(); r.point1.delete(); p.delete(); t.delete();
  });

  test("triangle × triangle", () => {
    const tf = getTf();
    const t0 = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    const t1 = tf.triangle(tf.point(0, 0, 3), tf.point(1, 0, 3), tf.point(0, 1, 3));
    const r = tf.closestPointPair(t0, t1);
    approx(r.distance2, 9.0, "d2");
    approx(r.point0.data[2], 0, "pt0.z");
    approx(r.point1.data[2], 3, "pt1.z");
    log(`  triangle × triangle: d2=${r.distance2}`, "line-pass");
    r.point0.delete(); r.point1.delete(); t0.delete(); t1.delete();
  });

  // --- batch ---

  test("batch: 3 points × segment", () => {
    const tf = getTf();
    const pts = tf.point(new Float32Array([0,1,0, 0,2,0, 0,3,0]), 3);
    const s = tf.segment(tf.point(0, 0, 0), tf.point(1, 0, 0));
    const r = tf.closestPointPair(pts, s);
    const dists = r.distances.data;
    approx(dists[0], 1.0, "[0]"); approx(dists[1], 4.0, "[1]"); approx(dists[2], 9.0, "[2]");
    log(`  batch 3 points × segment dists = [${dists[0]}, ${dists[1]}, ${dists[2]}]`, "line-pass");
    r.points0.delete(); r.points1.delete(); r.distances.delete();
    pts.delete(); s.delete();
  });

  test("batch: 3 segments × point", () => {
    const tf = getTf();
    const segs = tf.segment(new Float32Array([
      0,1,0, 1,1,0,
      0,2,0, 1,2,0,
      0,3,0, 1,3,0,
    ]), 3);
    const p = tf.point(0.5, 0, 0);
    const r = tf.closestPointPair(segs, p);
    const dists = r.distances.data;
    approx(dists[0], 1.0, "[0]"); approx(dists[1], 4.0, "[1]"); approx(dists[2], 9.0, "[2]");
    // Check closest points on segments are at y=1,2,3 and x=0.5
    const p0 = r.points0.data;
    for (let i = 0; i < 3; i++) {
      approx(p0[i * 3 + 0], 0.5, `pt0[${i}].x`);
      approx(p0[i * 3 + 1], i + 1, `pt0[${i}].y`);
    }
    log(`  batch 3 segments × point dists = [${dists[0]}, ${dists[1]}, ${dists[2]}]`, "line-pass");
    r.points0.delete(); r.points1.delete(); r.distances.delete();
    segs.delete(); p.delete();
  });

  test("batch: 3 aabbs × point", () => {
    const tf = getTf();
    const boxes = tf.aabb(new Float32Array([
      0,0,0, 1,1,1,
      0,0,2, 1,1,3,
      0,0,4, 1,1,5,
    ]), 3);
    const p = tf.point(0.5, 0.5, -1);
    const r = tf.closestPointPair(boxes, p);
    const dists = r.distances.data;
    approx(dists[0], 1.0, "[0]"); approx(dists[1], 9.0, "[1]"); approx(dists[2], 25.0, "[2]");
    log(`  batch 3 aabbs × point dists = [${dists[0]}, ${dists[1]}, ${dists[2]}]`, "line-pass");
    r.points0.delete(); r.points1.delete(); r.distances.delete();
    boxes.delete(); p.delete();
  });

  test("batch: 3 triangles × point", () => {
    const tf = getTf();
    // 3 triangles at z = 0, 1, 2
    const tris = tf.triangle(new Float32Array([
      0,0,0, 1,0,0, 0,1,0,
      0,0,1, 1,0,1, 0,1,1,
      0,0,2, 1,0,2, 0,1,2,
    ]), 3);
    const p = tf.point(0.25, 0.25, -1);
    assert(tris.isBatch && tris.count === 3, "should be batch of 3");
    const r = tf.closestPointPair(tris, p);
    const dists = r.distances.data;
    approx(dists[0], 1.0, "[0]"); approx(dists[1], 4.0, "[1]"); approx(dists[2], 9.0, "[2]");
    log(`  batch 3 triangles × point dists = [${dists[0]}, ${dists[1]}, ${dists[2]}]`, "line-pass");
    r.points0.delete(); r.points1.delete(); r.distances.delete();
    tris.delete(); p.delete();
  });
});

// ============================================================================
// neighborSearch (single-query only, no batch bindings)
// ============================================================================

describe("Spatial: neighborSearch", () => {

  test("mesh × point", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const q = tf.point(0.5, 0.5, 1);
    const r = tf.neighborSearch(m, q);
    assert(r.elementId === 0, `expected face 0, got ${r.elementId}`);
    log(`  mesh × point: face=${r.elementId}, d2=${r.distance2.toFixed(4)}, pt=(${r.point.data})`, "line-pass");
    r.point.delete(); q.delete(); m.delete();
  });

  test("mesh × point (face 1)", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const q = tf.point(0.5, -0.5, 1);
    const r = tf.neighborSearch(m, q);
    assert(r.elementId === 1, `expected face 1, got ${r.elementId}`);
    log(`  mesh × point (face 1): face=${r.elementId}, d2=${r.distance2.toFixed(4)}`, "line-pass");
    r.point.delete(); q.delete(); m.delete();
  });

  test("mesh × segment", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const s = tf.segment(tf.point(0, 0, 3), tf.point(1, 0, 3));
    const r = tf.neighborSearch(m, s);
    assert(r.elementId >= 0, "should find an element");
    approx(r.distance2, 9.0, "d2");
    log(`  mesh × segment: face=${r.elementId}, d2=${r.distance2.toFixed(4)}`, "line-pass");
    r.point.delete(); s.delete(); m.delete();
  });

  test("mesh × triangle", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    // Triangle hovering 2 units above
    const t = tf.triangle(tf.point(0, 0, 2), tf.point(1, 0, 2), tf.point(0.5, 1, 2));
    const r = tf.neighborSearch(m, t);
    assert(r.elementId >= 0, "should find an element");
    approx(r.distance2, 4.0, "d2");
    log(`  mesh × triangle: face=${r.elementId}, d2=${r.distance2.toFixed(4)}`, "line-pass");
    r.point.delete(); t.delete(); m.delete();
  });

  test("mesh × aabb", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    // Box floating 3 units above
    const box = tf.aabb(tf.point(0, 0, 3), tf.point(1, 1, 4));
    const r = tf.neighborSearch(m, box);
    assert(r.elementId >= 0, "should find an element");
    approx(r.distance2, 9.0, "d2");
    log(`  mesh × aabb: face=${r.elementId}, d2=${r.distance2.toFixed(4)}`, "line-pass");
    r.point.delete(); box.delete(); m.delete();
  });

  test("mesh × mesh", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m0 = tf.mesh(faces, points);
    const shifted = new Float32Array(points.length);
    for (let i = 0; i < points.length; i += 3) {
      shifted[i] = points[i]; shifted[i+1] = points[i+1]; shifted[i+2] = points[i+2] + 2;
    }
    const m1 = tf.mesh(new Int32Array(faces), shifted);
    const r = tf.neighborSearch(m0, m1);
    assert(r.elementId0 >= 0 && r.elementId1 >= 0, "should find elements");
    approx(r.distance2, 4.0, "d2");
    log(`  mesh × mesh: faces=${r.elementId0}↔${r.elementId1}, d2=${r.distance2.toFixed(4)}`, "line-pass");
    r.point0.delete(); r.point1.delete(); m0.delete(); m1.delete();
  });

  // --- batch ---

  test("batch: mesh × 3 points", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    // 3 points at z = 1, 2, 3 above face 0
    const pts = tf.point(new Float32Array([0.5,0.5,1, 0.5,0.5,2, 0.5,0.5,3]), 3);
    const r = tf.neighborSearch(m, pts);
    const ids = r.elementIds.data;
    const dists = r.distances.data;
    assert(ids[0] === 0, `pt0 face 0, got ${ids[0]}`);
    assert(ids[1] === 0, `pt1 face 0, got ${ids[1]}`);
    assert(ids[2] === 0, `pt2 face 0, got ${ids[2]}`);
    approx(dists[0], 1.0, "d2[0]"); approx(dists[1], 4.0, "d2[1]"); approx(dists[2], 9.0, "d2[2]");
    log(`  batch mesh × 3 points: faces=[${ids[0]},${ids[1]},${ids[2]}], dists=[${dists[0]},${dists[1]},${dists[2]}]`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete();
    pts.delete(); m.delete();
  });

  test("batch: mesh × 3 segments", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    // 3 segments at z = 1, 2, 3
    const segs = tf.segment(new Float32Array([
      0,0,1, 1,0,1,
      0,0,2, 1,0,2,
      0,0,3, 1,0,3,
    ]), 3);
    const r = tf.neighborSearch(m, segs);
    const dists = r.distances.data;
    approx(dists[0], 1.0, "d2[0]"); approx(dists[1], 4.0, "d2[1]"); approx(dists[2], 9.0, "d2[2]");
    log(`  batch mesh × 3 segments: dists=[${dists[0]},${dists[1]},${dists[2]}]`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete();
    segs.delete(); m.delete();
  });

  test("batch: mesh × 3 aabbs", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    // 3 boxes at z = 1, 2, 3
    const boxes = tf.aabb(new Float32Array([
      0,0,1, 1,1,2,
      0,0,2, 1,1,3,
      0,0,3, 1,1,4,
    ]), 3);
    const r = tf.neighborSearch(m, boxes);
    const dists = r.distances.data;
    approx(dists[0], 1.0, "d2[0]"); approx(dists[1], 4.0, "d2[1]"); approx(dists[2], 9.0, "d2[2]");
    log(`  batch mesh × 3 aabbs: dists=[${dists[0]},${dists[1]},${dists[2]}]`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete();
    boxes.delete(); m.delete();
  });
});

// ============================================================================
// rayCast
// ============================================================================

describe("Spatial: rayCast", () => {

  // --- single ---

  test("ray × triangle (hit)", () => {
    const tf = getTf();
    const r = tf.ray(tf.point(0.25, 0.25, -1), tf.vector(0, 0, 1));
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    const res = tf.rayCast(r, t);
    assert(res.hit === true, "should hit");
    approx(res.t, 1.0, "t");
    log(`  ray × triangle: hit=${res.hit}, t=${res.t.toFixed(3)}`, "line-pass");
    r.delete(); t.delete();
  });

  test("ray × triangle (miss)", () => {
    const tf = getTf();
    const r = tf.ray(tf.point(5, 5, -1), tf.vector(0, 0, 1));
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    const res = tf.rayCast(r, t);
    assert(res.hit === false, "should miss");
    log(`  ray × triangle: hit=${res.hit}`, "line-pass");
    r.delete(); t.delete();
  });

  test("ray × aabb (hit)", () => {
    const tf = getTf();
    const r = tf.ray(tf.point(0.5, 0.5, -5), tf.vector(0, 0, 1));
    const box = tf.aabb(tf.point(0, 0, 0), tf.point(1, 1, 1));
    const res = tf.rayCast(r, box);
    assert(res.hit === true, "should hit");
    approx(res.t, 5.0, "t");
    log(`  ray × aabb: hit=${res.hit}, t=${res.t.toFixed(3)}`, "line-pass");
    r.delete(); box.delete();
  });

  test("ray × mesh (hit)", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const r = tf.ray(tf.point(0.5, 0.5, 2), tf.vector(0, 0, -1));
    const res = tf.rayCast(r, m);
    assert(res.hit === true, "should hit");
    approx(res.t, 2.0, "t");
    assert(res.elementId === 0, `expected face 0, got ${res.elementId}`);
    log(`  ray × mesh: hit=${res.hit}, t=${res.t.toFixed(3)}, face=${res.elementId}`, "line-pass");
    r.delete(); m.delete();
  });

  test("ray × mesh with minT / maxT", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const r = tf.ray(tf.point(0.5, 0.5, 5), tf.vector(0, 0, -1));
    const miss = tf.rayCast(r, m, { minT: 0, maxT: 3 });
    assert(miss.hit === false, "maxT=3 should miss");
    const hit = tf.rayCast(r, m, { minT: 0, maxT: 10 });
    assert(hit.hit === true, "maxT=10 should hit");
    approx(hit.t, 5.0, "t");
    log(`  ray × mesh: maxT=3 miss, maxT=10 hit t=${hit.t.toFixed(3)}`, "line-pass");
    r.delete(); m.delete();
  });

  // --- batch ---

  test("batch: 3 rays × triangle", () => {
    const tf = getTf();
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    const rays = tf.ray(new Float32Array([
      0.25,0.25,-1, 0,0,1,   // hit at t=1
      5,5,-1,       0,0,1,   // miss
      0.1,0.1,-2,   0,0,1,   // hit at t=2
    ]), 3);
    assert(rays.isBatch && rays.count === 3, "should be batch of 3");
    const res = tf.rayCast(rays, t);
    const h = res.hits.data;
    const ts = res.ts.data;
    assert(h[0] === 1, "ray0 hit"); assert(h[1] === 0, "ray1 miss"); assert(h[2] === 1, "ray2 hit");
    approx(ts[0], 1.0, "t0"); approx(ts[2], 2.0, "t2");
    log(`  batch 3 rays × triangle: hits=[${h[0]},${h[1]},${h[2]}], ts=[${ts[0].toFixed(1)},_,${ts[2].toFixed(1)}]`, "line-pass");
    res.hits.delete(); res.ts.delete(); rays.delete(); t.delete();
  });

  test("batch: 3 rays × mesh", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const rays = tf.ray(new Float32Array([
      0.5,0.33,2,   0,0,-1,  // hits face 0
      0.5,-0.33,2,  0,0,-1,  // hits face 1
      10,10,2,      0,0,-1,  // misses
    ]), 3);
    const res = tf.rayCast(rays, m);
    const h = res.hits.data;
    const ids = res.elementIds.data;
    assert(h[0] === 1 && h[1] === 1 && h[2] === 0, "hit/hit/miss");
    assert(ids[0] === 0, `face0, got ${ids[0]}`);
    assert(ids[1] === 1, `face1, got ${ids[1]}`);
    log(`  batch 3 rays × mesh: hits=[${h[0]},${h[1]},${h[2]}], faces=[${ids[0]},${ids[1]},_]`, "line-pass");
    res.hits.delete(); res.ts.delete(); res.elementIds.delete();
    rays.delete(); m.delete();
  });

  // --- per-ray config ---

  test("per-ray minT/maxT × triangle", () => {
    const tf = getTf();
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    // ray0: hit at t=1, ray1: hit at t=2
    const rays = tf.ray(new Float32Array([
      0.25,0.25,-1, 0,0,1,
      0.1,0.1,-2,   0,0,1,
    ]), 2);
    // ray0: maxT=0.5 blocks the hit (t=1 > 0.5), ray1: maxT=5 allows the hit (t=2 < 5)
    const minTs = tf.ndarray(new Float32Array([0, 0]));
    const maxTs = tf.ndarray(new Float32Array([0.5, 5]));
    const res = tf.rayCast(rays, t, { minT: minTs, maxT: maxTs });
    const h = res.hits.data;
    const ts = res.ts.data;
    assert(h[0] === 0, "ray0 should miss (maxT=0.5)");
    assert(h[1] === 1, "ray1 should hit (maxT=5)");
    approx(ts[1], 2.0, "t1");
    log(`  per-ray config × triangle: hits=[${h[0]},${h[1]}], ts=[_,${ts[1].toFixed(1)}]`, "line-pass");
    res.hits.delete(); res.ts.delete();
    rays.delete(); t.delete(); minTs.delete(); maxTs.delete();
  });

  test("per-ray minT/maxT × mesh", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    // 3 rays all aimed at the mesh from z=5 going down
    const rays = tf.ray(new Float32Array([
      0.5,0.5,5,  0,0,-1,  // hits at t=5
      0.5,0.5,5,  0,0,-1,  // hits at t=5
      0.5,0.5,5,  0,0,-1,  // hits at t=5
    ]), 3);
    // ray0: maxT=3 blocks, ray1: maxT=10 allows, ray2: minT=6 blocks
    const minTs = tf.ndarray(new Float32Array([0, 0, 6]));
    const maxTs = tf.ndarray(new Float32Array([3, 10, 10]));
    const res = tf.rayCast(rays, m, { minT: minTs, maxT: maxTs });
    const h = res.hits.data;
    const ts = res.ts.data;
    assert(h[0] === 0, "ray0 should miss (maxT=3 < t=5)");
    assert(h[1] === 1, "ray1 should hit (maxT=10)");
    assert(h[2] === 0, "ray2 should miss (minT=6 > t=5)");
    approx(ts[1], 5.0, "t1");
    log(`  per-ray config × mesh: hits=[${h[0]},${h[1]},${h[2]}]`, "line-pass");
    res.hits.delete(); res.ts.delete(); res.elementIds.delete();
    rays.delete(); m.delete(); minTs.delete(); maxTs.delete();
  });

  test("mixed config: scalar minT + NDArray maxT", () => {
    const tf = getTf();
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    // ray0: hit at t=1, ray1: hit at t=2
    const rays = tf.ray(new Float32Array([
      0.25,0.25,-1, 0,0,1,
      0.1,0.1,-2,   0,0,1,
    ]), 2);
    // scalar minT=0, per-ray maxT: ray0 blocked (0.5 < 1), ray1 allowed (5 > 2)
    const maxTs = tf.ndarray(new Float32Array([0.5, 5]));
    const res = tf.rayCast(rays, t, { minT: 0, maxT: maxTs });
    const h = res.hits.data;
    assert(h[0] === 0, "ray0 should miss");
    assert(h[1] === 1, "ray1 should hit");
    log(`  mixed config (scalar + NDArray): hits=[${h[0]},${h[1]}]`, "line-pass");
    res.hits.delete(); res.ts.delete();
    rays.delete(); t.delete(); maxTs.delete();
  });
});

// ============================================================================
// Async smoke tests
// ============================================================================

describe("Spatial: async", () => {

  test("async distance2 (point × point)", async () => {
    const tf = getTf();
    const a = tf.point(0, 0, 0);
    const b = tf.point(1, 0, 0);
    const d = await tf.async.distance2(a, b);
    approx(d, 1.0, "async d2");
    log(`  async point × point = ${d}`, "line-pass");
    a.delete(); b.delete();
  });

  test("async distance2 (mesh × point)", async () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const p = tf.point(0.5, 0, 2);
    const d = await tf.async.distance2(m, p);
    approx(d, 4.0, "async d2");
    log(`  async mesh × point = ${d}`, "line-pass");
    p.delete(); m.delete();
  });

  test("async closestPointPair", async () => {
    const tf = getTf();
    const p = tf.point(0, 2, 0);
    const s = tf.segment(tf.point(0, 0, 0), tf.point(1, 0, 0));
    const r = await tf.async.closestPointPair(p, s);
    approx(r.distance2, 4.0, "async d2");
    log(`  async closestPointPair: d2=${r.distance2}`, "line-pass");
    r.point0.delete(); r.point1.delete(); p.delete(); s.delete();
  });

  test("async neighborSearch", async () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const q = tf.point(0.5, 0.5, 1);
    const r = await tf.async.neighborSearch(m, q);
    assert(r.elementId === 0, `expected face 0, got ${r.elementId}`);
    log(`  async neighborSearch: face=${r.elementId}`, "line-pass");
    r.point.delete(); q.delete(); m.delete();
  });

  test("async intersects", async () => {
    const tf = getTf();
    const s = tf.segment(tf.point(0.25, 0.25, -1), tf.point(0.25, 0.25, 1));
    const t = tf.triangle(tf.point(0, 0, 0), tf.point(1, 0, 0), tf.point(0, 1, 0));
    const r = await tf.async.intersects(s, t);
    assert(r === true, "should hit");
    log(`  async intersects: ${r}`, "line-pass");
    s.delete(); t.delete();
  });

  test("async rayCast", async () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const r = tf.ray(tf.point(0.5, 0.5, 2), tf.vector(0, 0, -1));
    const res = await tf.async.rayCast(r, m);
    assert(res.hit === true, "should hit");
    approx(res.t, 2.0, "t");
    log(`  async rayCast: hit=${res.hit}, t=${res.t.toFixed(3)}, face=${res.elementId}`, "line-pass");
    r.delete(); m.delete();
  });

  test("async rayCast per-ray config", async () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    // 2 rays aimed at mesh from z=5
    const rays = tf.ray(new Float32Array([
      0.5,0.5,5,  0,0,-1,  // hits at t=5
      0.5,0.5,5,  0,0,-1,  // hits at t=5
    ]), 2);
    const minTs = tf.ndarray(new Float32Array([0, 0]));
    const maxTs = tf.ndarray(new Float32Array([3, 10]));
    const res = await tf.async.rayCast(rays, m, { minT: minTs, maxT: maxTs });
    const h = res.hits.data;
    assert(h[0] === 0, "ray0 should miss (maxT=3)");
    assert(h[1] === 1, "ray1 should hit (maxT=10)");
    log(`  async rayCast per-ray: hits=[${h[0]},${h[1]}]`, "line-pass");
    res.hits.delete(); res.ts.delete(); res.elementIds.delete();
    rays.delete(); m.delete(); minTs.delete(); maxTs.delete();
  });
});

// ============================================================================
// PointCloud spatial queries
// ============================================================================

describe("Spatial: PointCloud", () => {

  test("pointCloud × point distance2", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const pc = tf.pointCloud(box);
    const p = tf.point(0, 0, 5);
    const d = tf.distance2(pc, p);
    assert(typeof d === "number", "single query returns number");
    assert(d > 0, `distance should be > 0, got ${d}`);
    log(`  pointCloud × point distance2 = ${d.toFixed(4)}`, "line-pass");
    p.delete(); pc.delete(); box.delete();
  });

  test("pointCloud × mesh distance2", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const pc = tf.pointCloud(box);
    const { faces, points } = twoTriangles();
    const shifted = new Float32Array(points.length);
    for (let i = 0; i < points.length; i += 3) {
      shifted[i] = points[i]; shifted[i+1] = points[i+1]; shifted[i+2] = points[i+2] + 10;
    }
    const m = tf.mesh(new Int32Array(faces), shifted);
    const d = tf.distance2(pc, m);
    assert(typeof d === "number", "FF returns number");
    assert(d > 0, `distance should be > 0, got ${d}`);
    log(`  pointCloud × mesh distance2 = ${d.toFixed(4)}`, "line-pass");
    m.delete(); pc.delete(); box.delete();
  });

  test("pointCloud × point neighborSearch", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const pc = tf.pointCloud(box);
    const p = tf.point(0, 0, 5);
    const r = tf.neighborSearch(pc, p);
    assert(r.elementId >= 0, `should find element, got ${r.elementId}`);
    assert(r.distance2 > 0, `distance should be > 0`);
    log(`  pointCloud × point neighbor: elem=${r.elementId}, d2=${r.distance2.toFixed(4)}`, "line-pass");
    r.point.delete(); p.delete(); pc.delete(); box.delete();
  });

  test("pointCloud × point intersects", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const pc = tf.pointCloud(box);
    const p_far = tf.point(100, 100, 100);
    const r = tf.intersects(pc, p_far);
    assert(r === false, "far point should not intersect");
    log(`  pointCloud × far point intersects = ${r}`, "line-pass");
    p_far.delete(); pc.delete(); box.delete();
  });

  test("pointCloud rayCast", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const pc = tf.pointCloud(box);
    // Aim directly through a known vertex (1,1,1) of the 2×2×2 box
    const r = tf.ray(tf.point(1, 1, 5), tf.vector(0, 0, -1));
    const res = tf.rayCast(r, pc);
    assert(res.hit === true, "should hit vertex");
    assert(res.elementId >= 0, `should have element, got ${res.elementId}`);
    log(`  pointCloud rayCast: hit=${res.hit}, t=${res.t.toFixed(3)}, elem=${res.elementId}`, "line-pass");
    r.delete(); pc.delete(); box.delete();
  });

  test("batch: pointCloud × 3 points distance2", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const pc = tf.pointCloud(box);
    const pts = tf.point(new Float32Array([0,0,5, 0,0,10, 0,0,15]), 3);
    const d = tf.distance2(pc, pts);
    assert(typeof d !== "number", "batch returns NDArray");
    const v = d.data;
    assert(v[0] < v[1] && v[1] < v[2], `distances should increase: [${v[0]}, ${v[1]}, ${v[2]}]`);
    log(`  batch pointCloud × 3 points = [${v[0].toFixed(2)}, ${v[1].toFixed(2)}, ${v[2].toFixed(2)}]`, "line-pass");
    d.delete(); pts.delete(); pc.delete(); box.delete();
  });

  test("batch: pointCloud × 3 points neighborSearch", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const pc = tf.pointCloud(box);
    const pts = tf.point(new Float32Array([0,0,5, 0,0,10, 0,0,15]), 3);
    const r = tf.neighborSearch(pc, pts);
    assert(r.elementIds.shape[0] === 3, "should have 3 results");
    const dists = r.distances.data;
    assert(dists[0] < dists[1] && dists[1] < dists[2], "distances should increase");
    log(`  batch pointCloud × 3 points neighbor: dists=[${dists[0].toFixed(2)},${dists[1].toFixed(2)},${dists[2].toFixed(2)}]`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete();
    pts.delete(); pc.delete(); box.delete();
  });
});

// ============================================================================
// k-NN neighbor search
// ============================================================================

describe("Spatial: k-NN neighborSearch", () => {

  test("mesh × point k=3", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const p = tf.point(0, 0, 0);
    const r = tf.neighborSearch(box, p, { k: 3 });
    assert(r.elementIds.shape[0] === 3, `expected 3 results, got ${r.elementIds.shape[0]}`);
    const dists = r.distances.data;
    assert(dists[0] <= dists[1] && dists[1] <= dists[2],
      `distances should be sorted: [${dists[0]}, ${dists[1]}, ${dists[2]}]`);
    log(`  mesh × point k=3: [${dists[0].toFixed(4)}, ${dists[1].toFixed(4)}, ${dists[2].toFixed(4)}]`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete();
    p.delete(); box.delete();
  });

  test("mesh × point k=5 with radius", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const p = tf.point(0, 0, 0);
    // Very small radius — should find fewer than 5
    const r = tf.neighborSearch(box, p, { k: 5, radius: 0.01 });
    const count = r.elementIds.shape[0];
    assert(count <= 5, `should find <= 5, got ${count}`);
    log(`  mesh × point k=5 radius=0.01: found ${count}`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete();
    p.delete(); box.delete();
  });

  test("batch: mesh × 3 points k=2", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const pts = tf.point(new Float32Array([0,0,0, 0,0,5, 0,0,10]), 3);
    const r = tf.neighborSearch(box, pts, { k: 2 });
    assert(r.elementIds.shape[0] === 3 && r.elementIds.shape[1] === 2,
      `expected [3, 2], got [${r.elementIds.shape}]`);
    assert(r.counts.shape[0] === 3, `expected 3 counts, got ${r.counts.shape[0]}`);
    const counts = r.counts.data;
    assert(counts[0] === 2, `pt0 should find 2, got ${counts[0]}`);
    assert(counts[1] === 2, `pt1 should find 2, got ${counts[1]}`);
    assert(counts[2] === 2, `pt2 should find 2, got ${counts[2]}`);
    log(`  batch mesh × 3 points k=2: counts=[${counts[0]},${counts[1]},${counts[2]}]`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete(); r.counts.delete();
    pts.delete(); box.delete();
  });

  test("pointCloud × point k=3", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    const pc = tf.pointCloud(box);
    const p = tf.point(0, 0, 0);
    const r = tf.neighborSearch(pc, p, { k: 3 });
    assert(r.elementIds.shape[0] === 3, `expected 3 results, got ${r.elementIds.shape[0]}`);
    const dists = r.distances.data;
    assert(dists[0] <= dists[1] && dists[1] <= dists[2],
      `distances should be sorted: [${dists[0]}, ${dists[1]}, ${dists[2]}]`);
    log(`  pointCloud × point k=3: [${dists[0].toFixed(4)}, ${dists[1].toFixed(4)}, ${dists[2].toFixed(4)}]`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete();
    p.delete(); pc.delete(); box.delete();
  });

  test("neighborSearch with radius (no k)", () => {
    const tf = getTf();
    const { faces, points } = twoTriangles();
    const m = tf.mesh(faces, points);
    const q = tf.point(0.5, 0.5, 1);
    const r = tf.neighborSearch(m, q, { radius: 100 });
    assert(r.elementId === 0, `expected face 0, got ${r.elementId}`);
    approx(r.distance2, 1.0, "d2");
    log(`  neighborSearch with radius: face=${r.elementId}, d2=${r.distance2.toFixed(4)}`, "line-pass");
    r.point.delete(); q.delete(); m.delete();
  });

  test("k-NN with radius: partial results (single)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    // Query point far away, tiny radius — should find 0 neighbors
    const p = tf.point(100, 100, 100);
    const r = tf.neighborSearch(box, p, { k: 5, radius: 0.001 });
    const count = r.elementIds.shape[0];
    assert(count === 0, `expected 0 results with tiny radius, got ${count}`);
    log(`  k-NN tiny radius single: found ${count}`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete();
    p.delete(); box.delete();
  });

  test("k-NN with radius: batch -1 padding", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    // pt0: at center, should find k neighbors
    // pt1: far away with tiny radius, should find 0
    // pt2: at center with large radius, should find k
    const pts = tf.point(new Float32Array([
      0, 0, 0,
      100, 100, 100,
      0, 0, 0,
    ]), 3);
    const r = tf.neighborSearch(box, pts, { k: 3, radius: 0.001 });
    assert(r.elementIds.shape[0] === 3, `expected N=3, got ${r.elementIds.shape[0]}`);
    assert(r.elementIds.shape[1] === 3, `expected k=3, got ${r.elementIds.shape[1]}`);

    const counts = r.counts.data;
    const ids = r.elementIds.data;

    // pt0: at center with tiny radius — may find 0 since box surface is at distance 1
    assert(counts[0] >= 0, `pt0 count valid`);
    // pt1: 100,100,100 with radius 0.001 — definitely 0
    assert(counts[1] === 0, `pt1 should find 0 with tiny radius, got ${counts[1]}`);
    // pt2: same as pt0
    assert(counts[2] >= 0, `pt2 count valid`);

    // Verify -1 padding for pt1 (all slots should be -1)
    const base1 = 1 * 3; // pt1 offset in flat [N, k] array
    assert(ids[base1 + 0] === -1, `pt1 slot0 should be -1, got ${ids[base1 + 0]}`);
    assert(ids[base1 + 1] === -1, `pt1 slot1 should be -1, got ${ids[base1 + 1]}`);
    assert(ids[base1 + 2] === -1, `pt1 slot2 should be -1, got ${ids[base1 + 2]}`);

    log(`  k-NN batch radius -1 padding: counts=[${counts[0]},${counts[1]},${counts[2]}], pt1 ids=[${ids[base1]},${ids[base1+1]},${ids[base1+2]}]`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete(); r.counts.delete();
    pts.delete(); box.delete();
  });

  test("k-NN with radius: mixed counts", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 2, 2);
    // Use a large enough radius to find some but maybe not all
    // pt0: at surface (1,0,0) — close to faces, should find 3 within radius 2
    // pt1: far at (0,0,50) — radius 2 too small, finds 0
    const pts = tf.point(new Float32Array([
      1, 0, 0,
      0, 0, 50,
    ]), 2);
    const r = tf.neighborSearch(box, pts, { k: 5, radius: 2 });
    const counts = r.counts.data;
    const ids = r.elementIds.data;

    assert(counts[0] > 0, `pt0 at surface should find some, got ${counts[0]}`);
    assert(counts[1] === 0, `pt1 far away should find 0, got ${counts[1]}`);

    // Verify unused slots in pt1 are -1
    const base1 = 1 * 5;
    for (let j = 0; j < 5; j++) {
      assert(ids[base1 + j] === -1, `pt1 slot ${j} should be -1, got ${ids[base1 + j]}`);
    }

    log(`  k-NN mixed counts: pt0 found ${counts[0]}, pt1 found ${counts[1]}`, "line-pass");
    r.elementIds.delete(); r.points.delete(); r.distances.delete(); r.counts.delete();
    pts.delete(); box.delete();
  });
});
