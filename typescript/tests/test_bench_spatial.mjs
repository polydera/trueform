import { describe, test, log, getTf } from "./harness.mjs";

// ============================================================================
// Helpers
// ============================================================================

/** Generate a grid mesh: (gridN+1)^2 vertices, 2*gridN^2 triangles. */
function gridMesh(gridN) {
  const nv = (gridN + 1) * (gridN + 1);
  const nf = 2 * gridN * gridN;
  const points = new Float32Array(nv * 3);
  const faces = new Int32Array(nf * 3);

  for (let j = 0; j <= gridN; j++) {
    for (let i = 0; i <= gridN; i++) {
      const idx = j * (gridN + 1) + i;
      points[idx * 3 + 0] = i / gridN;
      points[idx * 3 + 1] = j / gridN;
      points[idx * 3 + 2] = 0;
    }
  }

  let fi = 0;
  for (let j = 0; j < gridN; j++) {
    for (let i = 0; i < gridN; i++) {
      const v00 = j * (gridN + 1) + i;
      const v10 = v00 + 1;
      const v01 = v00 + (gridN + 1);
      const v11 = v01 + 1;
      faces[fi++] = v00; faces[fi++] = v10; faces[fi++] = v11;
      faces[fi++] = v00; faces[fi++] = v11; faces[fi++] = v01;
    }
  }

  return { faces, points };
}

/** Generate N random 3D points in [0,1]^3. */
function randomPoints(n) {
  const f = new Float32Array(n * 3);
  for (let i = 0; i < f.length; i++) f[i] = Math.random();
  return f;
}

/** Generate N random segments (2 endpoints each). */
function randomSegments(n) {
  const f = new Float32Array(n * 6);
  for (let i = 0; i < f.length; i++) f[i] = Math.random();
  return f;
}

/** Generate N random rays (origin in [0,1]^3, direction normalized). */
function randomRays(n) {
  const f = new Float32Array(n * 6);
  for (let i = 0; i < n; i++) {
    const o = i * 6;
    f[o + 0] = Math.random(); f[o + 1] = Math.random(); f[o + 2] = Math.random() + 1;
    let dx = Math.random() - 0.5, dy = Math.random() - 0.5, dz = -(Math.random() + 0.1);
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    f[o + 3] = dx / len; f[o + 4] = dy / len; f[o + 5] = dz / len;
  }
  return f;
}

/** Generate N random AABBs (min in [0,0.5]^3, max = min + [0,0.5]^3). */
function randomAABBs(n) {
  const f = new Float32Array(n * 6);
  for (let i = 0; i < n; i++) {
    const o = i * 6;
    for (let j = 0; j < 3; j++) {
      const lo = Math.random() * 0.5;
      f[o + j] = lo;
      f[o + 3 + j] = lo + Math.random() * 0.5;
    }
  }
  return f;
}

/** Time a function, return ms. */
function timeMs(fn) {
  const t0 = performance.now();
  fn();
  return performance.now() - t0;
}

/** Pad string to width. */
function pad(s, w, align = "right") {
  s = String(s);
  if (align === "right") return s.padStart(w);
  return s.padEnd(w);
}

/** Format a results table row. */
function fmtRow(label, ms) {
  return `  ${pad(label, 32, "left")} │ ${pad(ms.toFixed(2), 9)}`;
}

function header() {
  return `  ${pad("operation", 32, "left")} │ ${pad("time (ms)", 9)}`;
}

function separator() {
  return "  " + "─".repeat(32) + "─┼─" + "─".repeat(9);
}

// ============================================================================
// Benchmarks
// ============================================================================

describe("Bench: auto-parallel spatial", () => {

  test("spatial batch benchmarks", () => {
    const tf = getTf();

    // --- Setup ---
    const GRID = 50; // 5000 triangles
    const { faces, points } = gridMesh(GRID);
    const mesh = tf.mesh(faces, points);
    const tri = tf.triangle(tf.point(0.25, 0.25, 0), tf.point(0.75, 0.25, 0), tf.point(0.5, 0.75, 0));
    const seg = tf.segment(tf.point(0.3, 0.3, 0.5), tf.point(0.7, 0.7, 0.5));

    const sizes = [100, 1000, 10000, 100000];

    log("Grid mesh: " + (GRID * GRID * 2) + " triangles", "line-dim");
    log("");

    for (const N of sizes) {
      log(`──── N = ${N} ────`, "line-heading");
      log(header());
      log(separator());

      // Pre-generate batch data
      const ptData = randomPoints(N);
      const segData = randomSegments(N);
      const rayData = randomRays(N);
      const aabbData = randomAABBs(N);

      // ── PP: distance2 (points × segment) ──
      {
        const pts = tf.point(ptData, 3);
        const ms = timeMs(() => tf.distance2(pts, seg));
        log(fmtRow("distance2 PP pts×seg", ms));
        pts.delete();
      }

      // ── PP: distance2 (segments × point) ──
      {
        const segs = tf.segment(segData, 3);
        const p = tf.point(0.5, 0.5, 0.5);
        const ms = timeMs(() => tf.distance2(segs, p));
        log(fmtRow("distance2 PP segs×pt", ms));
        segs.delete(); p.delete();
      }

      // ── FP: distance2 (mesh × points) ──
      {
        const pts = tf.point(ptData, 3);
        const ms = timeMs(() => tf.distance2(mesh, pts));
        log(fmtRow("distance2 FP mesh×pts", ms));
        pts.delete();
      }

      // ── PP: intersects (segments × triangle) ──
      {
        const segs = tf.segment(segData, 3);
        const ms = timeMs(() => tf.intersects(segs, tri));
        log(fmtRow("intersects PP segs×tri", ms));
        segs.delete();
      }

      // ── FP: intersects (mesh × points) ──
      {
        const pts = tf.point(ptData, 3);
        const ms = timeMs(() => tf.intersects(mesh, pts));
        log(fmtRow("intersects FP mesh×pts", ms));
        pts.delete();
      }

      // ── FP: intersects (mesh × segments) ──
      {
        const segs = tf.segment(segData, 3);
        const ms = timeMs(() => tf.intersects(mesh, segs));
        log(fmtRow("intersects FP mesh×segs", ms));
        segs.delete();
      }

      // ── PP: closestPointPair (points × segment) ──
      {
        const pts = tf.point(ptData, 3);
        const ms = timeMs(() => { const r = tf.closestPointPair(pts, seg); r.points0.delete(); r.points1.delete(); r.distances.delete(); });
        log(fmtRow("closestPtPair PP pts×seg", ms));
        pts.delete();
      }

      // ── PP: closestPointPair (aabbs × point) ──
      {
        const boxes = tf.aabb(aabbData, 3);
        const p = tf.point(0.5, 0.5, 0.5);
        const ms = timeMs(() => { const r = tf.closestPointPair(boxes, p); r.points0.delete(); r.points1.delete(); r.distances.delete(); });
        log(fmtRow("closestPtPair PP aabbs×pt", ms));
        boxes.delete(); p.delete();
      }

      // ── FP: neighborSearch (mesh × points) ──
      {
        const pts = tf.point(ptData, 3);
        const ms = timeMs(() => { const r = tf.neighborSearch(mesh, pts); r.elementIds.delete(); r.points.delete(); r.distances.delete(); });
        log(fmtRow("neighborSearch FP mesh×pts", ms));
        pts.delete();
      }

      // ── FP: neighborSearch (mesh × segments) ──
      {
        const segs = tf.segment(segData, 3);
        const ms = timeMs(() => { const r = tf.neighborSearch(mesh, segs); r.elementIds.delete(); r.points.delete(); r.distances.delete(); });
        log(fmtRow("neighborSearch FP mesh×segs", ms));
        segs.delete();
      }

      // ── PP: rayCast (rays × triangle) ──
      {
        const rays = tf.ray(rayData, 3);
        const ms = timeMs(() => { const r = tf.rayCast(rays, tri); r.hits.delete(); r.ts.delete(); });
        log(fmtRow("rayCast PP rays×tri", ms));
        rays.delete();
      }

      // ── FP: rayCast (rays × mesh) ──
      {
        const rays = tf.ray(rayData, 3);
        const ms = timeMs(() => { const r = tf.rayCast(rays, mesh); r.hits.delete(); r.ts.delete(); r.elementIds.delete(); });
        log(fmtRow("rayCast FP rays×mesh", ms));
        rays.delete();
      }

      log("");
    }

    // Cleanup
    tri.delete(); seg.delete(); mesh.delete();
    log("Done.", "line-dim");
  });

});
