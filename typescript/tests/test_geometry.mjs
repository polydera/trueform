import { describe, test, log, assert, getTf } from "./harness.mjs";

describe("Geometry", () => {

  // ==========================================================================
  test("triangulate Mesh (triangles → identity)", () => {
    const tf = getTf();
    // Two triangles sharing edge 0-1
    const faces = new Int32Array([0, 1, 2, 0, 3, 1]);
    const points = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0.5, 1, 0,
      0.5, -1, 0,
    ]);
    const m = tf.mesh(faces, points);
    const tri = tf.triangulate(m);

    assert(tri.numberOfFaces === 2, `expected 2 faces, got ${tri.numberOfFaces}`);
    assert(tri.numberOfPoints === 4, `expected 4 points, got ${tri.numberOfPoints}`);
    log("  tri mesh → 2 faces, 4 points", "line-pass");

    const fh = tri.faces;
    assert(fh.shape[0] === 2 && fh.shape[1] === 3, `face shape: [${fh.shape}]`);
    log("  face shape [2, 3]", "line-pass");

    fh.delete();
    tri.delete();
    m.delete();
  });

  // ==========================================================================
  test("triangulate MeshLike fixed (quads)", () => {
    const tf = getTf();
    // Two quads:
    //  3---2    7---6
    //  |   |    |   |
    //  0---1    4---5
    const faces = tf.ndarray(new Int32Array([
      0, 1, 2, 3,
      4, 5, 6, 7,
    ]), [2, 4]);
    const points = tf.ndarray(new Float32Array([
      0, 0, 0,   1, 0, 0,   1, 1, 0,   0, 1, 0,
      2, 0, 0,   3, 0, 0,   3, 1, 0,   2, 1, 0,
    ]), [8, 3]);

    const tri = tf.triangulate({ faces, points });

    assert(tri.numberOfFaces === 4, `expected 4 faces, got ${tri.numberOfFaces}`);
    assert(tri.numberOfPoints === 8, `expected 8 points, got ${tri.numberOfPoints}`);
    log("  2 quads → 4 triangles", "line-pass");

    const fh = tri.faces;
    assert(fh.shape[1] === 3, `expected tri faces, got shape[1]=${fh.shape[1]}`);
    log("  output faces are triangles", "line-pass");

    fh.delete();
    tri.delete();
    points.delete();
    faces.delete();
  });

  // ==========================================================================
  test("triangulate MeshLike fixed (pentagons)", () => {
    const tf = getTf();
    // Single pentagon: 5 vertices → 3 triangles
    const faces = tf.ndarray(new Int32Array([0, 1, 2, 3, 4]), [1, 5]);
    const cos72 = Math.cos(2 * Math.PI / 5);
    const sin72 = Math.sin(2 * Math.PI / 5);
    const cos144 = Math.cos(4 * Math.PI / 5);
    const sin144 = Math.sin(4 * Math.PI / 5);
    const points = tf.ndarray(new Float32Array([
      1, 0, 0,
      cos72, sin72, 0,
      cos144, sin144, 0,
      cos144, -sin144, 0,
      cos72, -sin72, 0,
    ]), [5, 3]);

    const tri = tf.triangulate({ faces, points });

    assert(tri.numberOfFaces === 3, `expected 3 faces, got ${tri.numberOfFaces}`);
    assert(tri.numberOfPoints === 5, `expected 5 points, got ${tri.numberOfPoints}`);
    log("  1 pentagon → 3 triangles", "line-pass");

    tri.delete();
    points.delete();
    faces.delete();
  });

  // ==========================================================================
  test("triangulate Polygon (single, 3D)", () => {
    const tf = getTf();
    // Square polygon in 3D
    const poly = tf.polygon(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 1, 0,
      0, 1, 0,
    ]), 3);

    const tri = tf.triangulate(poly);

    assert(tri.numberOfFaces === 2, `expected 2 faces, got ${tri.numberOfFaces}`);
    assert(tri.numberOfPoints === 4, `expected 4 points, got ${tri.numberOfPoints}`);
    log("  square polygon → 2 triangles", "line-pass");

    const fh = tri.faces;
    assert(fh.shape[1] === 3, "output faces are triangles");
    log("  output faces shape OK", "line-pass");

    fh.delete();
    tri.delete();
    poly.delete();
  });

  // ==========================================================================
  test("triangulate Polygon (triangle, identity)", () => {
    const tf = getTf();
    const poly = tf.polygon(new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0.5, 1, 0,
    ]), 3);

    const tri = tf.triangulate(poly);

    assert(tri.numberOfFaces === 1, `expected 1 face, got ${tri.numberOfFaces}`);
    assert(tri.numberOfPoints === 3, `expected 3 points, got ${tri.numberOfPoints}`);
    log("  triangle polygon → 1 triangle", "line-pass");

    tri.delete();
    poly.delete();
  });

  // ==========================================================================
  test("triangulate Polygon (batch, 3D)", () => {
    const tf = getTf();
    // Batch of 2 quads: shape [2, 4, 3]
    const batch = tf.ndarray(new Float32Array([
      // Quad 1
      0, 0, 0,   1, 0, 0,   1, 1, 0,   0, 1, 0,
      // Quad 2
      2, 0, 0,   3, 0, 0,   3, 1, 0,   2, 1, 0,
    ]), [2, 4, 3]);
    const poly = tf.polygon(batch);

    const tri = tf.triangulate(poly);

    assert(tri.numberOfFaces === 4, `expected 4 faces, got ${tri.numberOfFaces}`);
    log("  batch 2 quads → 4 triangles", "line-pass");

    // Points should be 8 (after cleaning, all unique)
    assert(tri.numberOfPoints === 8, `expected 8 points, got ${tri.numberOfPoints}`);
    log("  8 unique points", "line-pass");

    tri.delete();
    poly.delete();
    batch.delete();
  });

  // ==========================================================================
  test("offsetBlockedBuffer factory", () => {
    const tf = getTf();
    // Two paths: [0,1,2] and [3,4]
    const offsets = tf.ndarray(new Int32Array([0, 3, 5]), [3]);
    const data = tf.ndarray(new Int32Array([0, 1, 2, 3, 4]), [5]);

    const obb = tf.offsetBlockedBuffer(offsets, data);
    assert(obb.length === 2, `expected 2 blocks, got ${obb.length}`);

    const b0 = obb.get(0);
    assert(b0.length === 3, `block 0 size: ${b0.length}`);
    const b0d = b0.data;
    assert(b0d[0] === 0 && b0d[1] === 1 && b0d[2] === 2, `block 0 data: [${b0d}]`);

    const b1 = obb.get(1);
    assert(b1.length === 2, `block 1 size: ${b1.length}`);
    const b1d = b1.data;
    assert(b1d[0] === 3 && b1d[1] === 4, `block 1 data: [${b1d}]`);

    log("  offsetBlockedBuffer([0,3,5], [0,1,2,3,4]) → 2 blocks", "line-pass");

    b1.delete();
    b0.delete();
    obb.delete();
    data.delete();
    offsets.delete();
  });

  // ==========================================================================
  test("curves from offsetBlockedBuffer + points", () => {
    const tf = getTf();
    // Two curves: [0→1→2] and [3→4]
    const offsets = tf.ndarray(new Int32Array([0, 3, 5]), [3]);
    const data = tf.ndarray(new Int32Array([0, 1, 2, 3, 4]), [5]);
    const points = tf.ndarray(new Float32Array([
      0,0,0, 1,0,0, 2,0,0, 0,1,0, 1,1,0,
    ]), [5, 3]);

    const paths = tf.offsetBlockedBuffer(offsets, data);
    const c = tf.curves(paths, points);

    assert(c.length === 2, `expected 2 curves, got ${c.length}`);

    const pts = c.points;
    assert(pts.shape[0] === 5 && pts.shape[1] === 3, `points shape: [${pts.shape}]`);

    log("  curves from obb + points → 2 curves, 5 points", "line-pass");

    // Make tube mesh from curves
    const tubes = tf.tubeMesh(c, 0.1, 6);
    assert(tubes.numberOfFaces > 0, `tube faces: ${tubes.numberOfFaces}`);
    assert(tubes.numberOfPoints > 0, `tube points: ${tubes.numberOfPoints}`);
    log(`  tubeMesh → ${tubes.numberOfFaces} faces, ${tubes.numberOfPoints} points`, "line-pass");

    tubes.delete();
    pts.delete();
    c.delete();
    paths.delete();
    points.delete();
    data.delete();
    offsets.delete();
  });

  // ==========================================================================
  test("sphereMesh", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    // stacks=10, segments=10:
    // vertices = 2 + (10-1)*10 = 92
    // faces = 2 * 10 * (10-1) = 180
    assert(sphere.numberOfPoints === 92, `expected 92 points, got ${sphere.numberOfPoints}`);
    assert(sphere.numberOfFaces === 180, `expected 180 faces, got ${sphere.numberOfFaces}`);
    log(`  sphere(1, 10, 10) → ${sphere.numberOfFaces} faces, ${sphere.numberOfPoints} points`, "line-pass");

    const pts = sphere.points;
    assert(pts.shape[1] === 3, "points are 3D");
    log("  points shape OK", "line-pass");
    pts.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("cylinderMesh", () => {
    const tf = getTf();
    const cyl = tf.cylinderMesh(1.0, 2.0, 20);

    // vertices = 2 + 2*20 = 42
    // faces = 4*20 = 80
    assert(cyl.numberOfPoints === 42, `expected 42 points, got ${cyl.numberOfPoints}`);
    assert(cyl.numberOfFaces === 80, `expected 80 faces, got ${cyl.numberOfFaces}`);
    log(`  cylinder(1, 2, 20) → ${cyl.numberOfFaces} faces, ${cyl.numberOfPoints} points`, "line-pass");

    cyl.delete();
  });

  // ==========================================================================
  test("boxMesh (simple)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 1, 3);

    assert(box.numberOfPoints === 8, `expected 8 points, got ${box.numberOfPoints}`);
    assert(box.numberOfFaces === 12, `expected 12 faces, got ${box.numberOfFaces}`);
    log(`  box(2,1,3) → ${box.numberOfFaces} faces, ${box.numberOfPoints} points`, "line-pass");

    box.delete();
  });

  // ==========================================================================
  test("boxMesh (subdivided)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 1, 3, 2, 2, 2);

    // subdivided has more verts/faces than simple
    assert(box.numberOfFaces > 12, `expected >12 faces, got ${box.numberOfFaces}`);
    assert(box.numberOfPoints > 8, `expected >8 points, got ${box.numberOfPoints}`);
    log(`  box(2,1,3, 2,2,2) → ${box.numberOfFaces} faces, ${box.numberOfPoints} points`, "line-pass");

    box.delete();
  });

  // ==========================================================================
  test("planeMesh (simple)", () => {
    const tf = getTf();
    const plane = tf.planeMesh(10, 5);

    assert(plane.numberOfPoints === 4, `expected 4 points, got ${plane.numberOfPoints}`);
    assert(plane.numberOfFaces === 2, `expected 2 faces, got ${plane.numberOfFaces}`);
    log(`  plane(10,5) → ${plane.numberOfFaces} faces, ${plane.numberOfPoints} points`, "line-pass");

    plane.delete();
  });

  // ==========================================================================
  test("area (box)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);
    const a = tf.area(box);
    const expected = 2 * (2 * 3 + 3 * 4 + 2 * 4); // 52
    assert(Math.abs(a - expected) < 0.01, `expected area ~${expected}, got ${a}`);
    log(`  box(2,3,4) area = ${a}`, "line-pass");
    box.delete();
  });

  // ==========================================================================
  test("volume and signedVolume (box)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);
    const sv = tf.signedVolume(box);
    const v = tf.volume(box);
    const expected = 24; // 2*3*4
    assert(Math.abs(v - expected) < 0.01, `expected volume ~${expected}, got ${v}`);
    assert(Math.abs(Math.abs(sv) - expected) < 0.01, `expected |signedVolume| ~${expected}, got ${sv}`);
    log(`  box(2,3,4) volume = ${v}, signedVolume = ${sv}`, "line-pass");
    box.delete();
  });

  // ==========================================================================
  test("meanEdgeLength, minEdgeLength, maxEdgeLength (box)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);
    const mel = tf.meanEdgeLength(box);
    const minEl = tf.minEdgeLength(box);
    const maxEl = tf.maxEdgeLength(box);
    // Simple box edges: 2, 3, 4, and diagonals sqrt(4+9)=3.606, sqrt(9+16)=5, sqrt(4+16)=4.472
    assert(minEl > 0, `min edge length should be > 0, got ${minEl}`);
    assert(maxEl >= minEl, `max >= min: ${maxEl} >= ${minEl}`);
    assert(mel >= minEl && mel <= maxEl, `mean in [min, max]: ${mel}`);
    log(`  box(2,3,4) edgeLengths: min=${minEl.toFixed(3)}, mean=${mel.toFixed(3)}, max=${maxEl.toFixed(3)}`, "line-pass");
    box.delete();
  });

  // ==========================================================================
  test("planeMesh (subdivided)", () => {
    const tf = getTf();
    const plane = tf.planeMesh(10, 5, 4, 3);

    // (4+1)*(3+1) = 20 vertices, 4*3*2 = 24 faces
    assert(plane.numberOfPoints === 20, `expected 20 points, got ${plane.numberOfPoints}`);
    assert(plane.numberOfFaces === 24, `expected 24 faces, got ${plane.numberOfFaces}`);
    log(`  plane(10,5, 4,3) → ${plane.numberOfFaces} faces, ${plane.numberOfPoints} points`, "line-pass");

    plane.delete();
  });

  // ==========================================================================
  test("normals (box)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    const n = box.normals;
    assert(n.shape[0] === box.numberOfFaces, `expected ${box.numberOfFaces} normals, got ${n.shape[0]}`);
    assert(n.shape[1] === 3, `expected 3D normals, got shape[1]=${n.shape[1]}`);
    log(`  box normals shape [${n.shape}]`, "line-pass");

    // Each normal should be unit length
    const data = n.data;
    for (let i = 0; i < n.shape[0]; i++) {
      const x = data[i * 3], y = data[i * 3 + 1], z = data[i * 3 + 2];
      const len = Math.sqrt(x * x + y * y + z * z);
      assert(Math.abs(len - 1.0) < 0.01, `normal ${i} not unit length: ${len}`);
    }
    log("  all normals are unit length", "line-pass");

    n.delete();
    box.delete();
  });

  // ==========================================================================
  test("pointNormals (box)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    const pn = box.pointNormals;
    assert(pn.shape[0] === box.numberOfPoints, `expected ${box.numberOfPoints} point normals, got ${pn.shape[0]}`);
    assert(pn.shape[1] === 3, `expected 3D normals, got shape[1]=${pn.shape[1]}`);
    log(`  box pointNormals shape [${pn.shape}]`, "line-pass");

    // Each point normal should be unit length
    const data = pn.data;
    for (let i = 0; i < pn.shape[0]; i++) {
      const x = data[i * 3], y = data[i * 3 + 1], z = data[i * 3 + 2];
      const len = Math.sqrt(x * x + y * y + z * z);
      assert(Math.abs(len - 1.0) < 0.01, `point normal ${i} not unit length: ${len}`);
    }
    log("  all point normals are unit length", "line-pass");

    pn.delete();
    box.delete();
  });

  // ==========================================================================
  test("positivelyOriented (reverses negative winding)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4);

    // Reverse each face's winding to get negative orientation
    const fh = box.faces;
    const faceData = fh.data;
    const reversed = new Int32Array(faceData.length);
    for (let i = 0; i < faceData.length; i += 3) {
      reversed[i]     = faceData[i];
      reversed[i + 1] = faceData[i + 2];
      reversed[i + 2] = faceData[i + 1];
    }
    const ph = box.points;
    const negMesh = tf.mesh(reversed, ph.data);

    // Confirm it's negatively oriented
    const svNeg = tf.signedVolume(negMesh);
    assert(svNeg < 0, `expected negative signedVolume, got ${svNeg}`);
    log(`  reversed mesh signedVolume = ${svNeg.toFixed(3)} (negative)`, "line-pass");

    // Fix orientation
    const oriented = tf.positivelyOriented(negMesh);
    const svPos = tf.signedVolume(oriented);
    assert(svPos > 0, `expected positive signedVolume, got ${svPos}`);
    log(`  positivelyOriented signedVolume = ${svPos.toFixed(3)} (positive)`, "line-pass");

    oriented.delete();
    negMesh.delete();
    ph.delete();
    fh.delete();
    box.delete();
  });

  // ==========================================================================
  test("principalCurvatures (sphere)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(2.0, 20, 20);

    const { k0, k1 } = tf.principalCurvatures(sphere);
    assert(k0.shape[0] === sphere.numberOfPoints, `expected ${sphere.numberOfPoints} k0 values`);
    assert(k1.shape[0] === sphere.numberOfPoints, `expected ${sphere.numberOfPoints} k1 values`);
    log(`  k0 shape [${k0.shape}], k1 shape [${k1.shape}]`, "line-pass");

    // For a sphere of radius 2, curvatures should be ~0.5 (1/r)
    const meanK0 = tf.mean(k0);
    const meanK1 = tf.mean(k1);
    assert(Math.abs(meanK0 - 0.5) < 0.15, `expected mean k0 ~0.5, got ${meanK0}`);
    assert(Math.abs(meanK1 - 0.5) < 0.15, `expected mean k1 ~0.5, got ${meanK1}`);
    log(`  mean k0=${meanK0.toFixed(3)}, k1=${meanK1.toFixed(3)} (expected ~0.5)`, "line-pass");

    k0.delete();
    k1.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("principalDirections (sphere)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(2.0, 10, 10);

    const { k0, k1, d0, d1 } = tf.principalDirections(sphere);
    assert(d0.shape[0] === sphere.numberOfPoints && d0.shape[1] === 3, `d0 shape [${d0.shape}]`);
    assert(d1.shape[0] === sphere.numberOfPoints && d1.shape[1] === 3, `d1 shape [${d1.shape}]`);
    log(`  d0 [${d0.shape}], d1 [${d1.shape}]`, "line-pass");

    // Directions should be unit length
    const d0Norms = tf.norm(d0, 1);
    const minNorm = tf.min(d0Norms);
    const maxNorm = tf.max(d0Norms);
    assert(minNorm > 0.9 && maxNorm < 1.1, `d0 norms in [${minNorm}, ${maxNorm}]`);
    log(`  d0 norms in [${minNorm.toFixed(3)}, ${maxNorm.toFixed(3)}]`, "line-pass");

    d0Norms.delete();
    k0.delete();
    k1.delete();
    d0.delete();
    d1.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("shapeIndex (sphere → ~1, plane → ~0)", () => {
    const tf = getTf();

    // Sphere: convex → shape index close to 1
    const sphere = tf.sphereMesh(2.0, 20, 20);
    const siSphere = tf.shapeIndex(sphere);
    const meanSI = tf.mean(siSphere);
    assert(meanSI > 0.5, `expected sphere mean SI > 0.5, got ${meanSI}`);
    log(`  sphere mean SI = ${meanSI.toFixed(3)}`, "line-pass");
    siSphere.delete();
    sphere.delete();

    // Plane: flat → shape index close to 0
    const plane = tf.planeMesh(10, 10, 5, 5);
    const siPlane = tf.shapeIndex(plane);
    const meanAbsSI = tf.mean(tf.abs(siPlane));
    assert(meanAbsSI < 0.1, `expected plane mean |SI| < 0.1, got ${meanAbsSI}`);
    log(`  plane mean |SI| = ${meanAbsSI.toFixed(3)}`, "line-pass");
    siPlane.delete();
    plane.delete();
  });

  // ==========================================================================
  test("laplacianSmoothed (sphere — reduces curvature range)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    const smoothed = tf.laplacianSmoothed(sphere, 3, 0.5);
    assert(smoothed.numberOfFaces === sphere.numberOfFaces, "face count preserved");
    assert(smoothed.numberOfPoints === sphere.numberOfPoints, "point count preserved");
    log(`  smoothed mesh: ${smoothed.numberOfFaces} faces, ${smoothed.numberOfPoints} points`, "line-pass");

    // Smoothing should move vertices — points should differ from original
    const origPts = sphere.points;
    const smoothPts = smoothed.points;
    const diff = tf.abs(smoothPts.sub(origPts));
    const maxDiff = tf.max(diff);
    assert(maxDiff > 0.001, `expected points to move, max diff = ${maxDiff}`);
    log(`  max point displacement = ${maxDiff.toFixed(4)}`, "line-pass");

    diff.delete();
    smoothPts.delete();
    origPts.delete();
    smoothed.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("taubinSmoothed (sphere — volume-preserving)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    const smoothed = tf.taubinSmoothed(sphere, 5, 0.5, 0.1);
    assert(smoothed.numberOfFaces === sphere.numberOfFaces, "face count preserved");
    assert(smoothed.numberOfPoints === sphere.numberOfPoints, "point count preserved");
    log(`  smoothed mesh: ${smoothed.numberOfFaces} faces, ${smoothed.numberOfPoints} points`, "line-pass");

    // Taubin smoothing is volume-preserving — volume should be close to original
    const origVol = tf.volume(sphere);
    const smoothVol = tf.volume(smoothed);
    const volRatio = smoothVol / origVol;
    assert(volRatio > 0.8 && volRatio < 1.2, `expected volume ratio ~1, got ${volRatio}`);
    log(`  volume ratio = ${volRatio.toFixed(4)} (original=${origVol.toFixed(4)}, smoothed=${smoothVol.toFixed(4)})`, "line-pass");

    smoothed.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("sharpEdges (subdivided box → 12 sharp edges)", () => {
    const tf = getTf();
    const box = tf.boxMesh(2, 3, 4, 4, 4, 4);

    const edges = tf.sharpEdges(box, 30);
    assert(edges.shape[1] === 2, `expected shape [N, 2], got shape[1]=${edges.shape[1]}`);
    // 12 box edges × 4 subdivisions = 48 sharp edge segments
    assert(edges.shape[0] === 48, `expected 48 sharp edges, got ${edges.shape[0]}`);
    log(`  box(2,3,4, 4,4,4) → ${edges.shape[0]} sharp edges at 30°`, "line-pass");

    edges.delete();
    box.delete();
  });

  // ==========================================================================
  test("reverseWinding (flips signed volume)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 10, 10);

    const sv = tf.signedVolume(sphere);
    assert(sv > 0, `expected positive signedVolume, got ${sv}`);
    log(`  sphere signedVolume = ${sv.toFixed(4)} (positive)`, "line-pass");

    const flipped = tf.reverseWinding(sphere);
    const svFlipped = tf.signedVolume(flipped);
    assert(svFlipped < 0, `expected negative signedVolume, got ${svFlipped}`);
    log(`  reversed signedVolume = ${svFlipped.toFixed(4)} (negative)`, "line-pass");

    flipped.delete();
    sphere.delete();
  });

  // ==========================================================================
  test("normals (plane — all same direction)", () => {
    const tf = getTf();
    const plane = tf.planeMesh(10, 5);

    const n = plane.normals;
    assert(n.shape[0] === 2, `expected 2 face normals, got ${n.shape[0]}`);

    // XY plane normals should point in +Z or -Z
    const data = n.data;
    for (let i = 0; i < 2; i++) {
      const x = data[i * 3], y = data[i * 3 + 1], z = data[i * 3 + 2];
      assert(Math.abs(x) < 0.01 && Math.abs(y) < 0.01, `plane normal should be along Z, got (${x},${y},${z})`);
      assert(Math.abs(Math.abs(z) - 1.0) < 0.01, `plane normal Z component should be ±1, got ${z}`);
    }
    log("  plane normals point along Z axis", "line-pass");

    n.delete();
    plane.delete();
  });

});
