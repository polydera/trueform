import { describe, test, log, assert, approx, getTf } from "./harness.mjs";

// ============================================================================
// Helpers
// ============================================================================

function applyTransform(pts, T) {
  // pts: Float32Array [N*3], T: Float32Array [16] (row-major 4x4)
  // Returns new Float32Array [N*3]
  const n = pts.length / 3;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = pts[i * 3], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
    out[i * 3]     = T[0] * x + T[1] * y + T[2]  * z + T[3];
    out[i * 3 + 1] = T[4] * x + T[5] * y + T[6]  * z + T[7];
    out[i * 3 + 2] = T[8] * x + T[9] * y + T[10] * z + T[11];
  }
  return out;
}

function rotationZ(angleDeg) {
  const a = angleDeg * Math.PI / 180;
  const c = Math.cos(a), s = Math.sin(a);
  // Row-major 4x4
  return new Float32Array([
    c, -s,  0, 0,
    s,  c,  0, 0,
    0,  0,  1, 0,
    0,  0,  0, 1,
  ]);
}

function rotTransZ(angleDeg, tx, ty, tz) {
  const a = angleDeg * Math.PI / 180;
  const c = Math.cos(a), s = Math.sin(a);
  return new Float32Array([
    c, -s,  0, tx,
    s,  c,  0, ty,
    0,  0,  1, tz,
    0,  0,  0,  1,
  ]);
}

function matMul4x4(A, B) {
  const C = new Float32Array(16);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += A[i * 4 + k] * B[k * 4 + j];
      C[i * 4 + j] = sum;
    }
  return C;
}

function seededRandom(n, seed) {
  // Simple LCG for reproducible tests (returns values in [0, 1))
  let s = seed;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
    out[i] = (s >>> 0) / 4294967296;
  }
  return out;
}

function randomPoints(count, seed) {
  return seededRandom(count * 3, seed);
}

function randomCenteredPoints(count, seed) {
  const pts = seededRandom(count * 3, seed);
  for (let i = 0; i < pts.length; i++) pts[i] = pts[i] * 2 - 1;
  return pts;
}

function elongatedBox(count, lx, ly, lz, seed) {
  const raw = seededRandom(count * 3, seed);
  for (let i = 0; i < count; i++) {
    raw[i * 3]     = (raw[i * 3]     - 0.5) * lx;
    raw[i * 3 + 1] = (raw[i * 3 + 1] - 0.5) * ly;
    raw[i * 3 + 2] = (raw[i * 3 + 2] - 0.5) * lz;
  }
  return raw;
}

// ============================================================================
// Chamfer Error
// ============================================================================

describe("Chamfer Error", () => {

  test("identical clouds → ~0", () => {
    const tf = getTf();
    const pts = randomPoints(50, 42);
    const src = tf.pointCloud(pts);
    const tgt = tf.pointCloud(new Float32Array(pts));

    const err = tf.chamferError(src, tgt);
    assert(err < 1e-5, `expected ~0, got ${err}`);
    log(`  error = ${err}`, "line-pass");

    src.delete();
    tgt.delete();
  });

  test("known offset (cube vertices shifted x+1)", () => {
    const tf = getTf();
    const pts0 = new Float32Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1,
      1,1,0, 1,0,1, 0,1,1, 1,1,1,
    ]);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i] = pts0[i] + 1;
      pts1[i+1] = pts0[i+1];
      pts1[i+2] = pts0[i+2];
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);

    // Half vertices (x=1) have NN distance 0, half (x=0) distance 1. Mean = 0.5
    const err = tf.chamferError(src, tgt);
    approx(err, 0.5, "chamfer error", 1e-4);
    log(`  error = ${err} (expected 0.5)`, "line-pass");

    src.delete();
    tgt.delete();
  });

  test("asymmetry", () => {
    const tf = getTf();
    const pts0 = new Float32Array([0, 0, 0]);
    const pts1 = new Float32Array([1, 0, 0, 2, 0, 0]);

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);

    const err01 = tf.chamferError(src, tgt);
    const err10 = tf.chamferError(tgt, src);

    approx(err01, 1.0, "src→tgt", 1e-4);
    approx(err10, 1.5, "tgt→src", 1e-4);
    assert(Math.abs(err01 - err10) > 0.1, "should be asymmetric");
    log(`  src→tgt=${err01}, tgt→src=${err10}`, "line-pass");

    src.delete();
    tgt.delete();
  });

  // ==========================================================================
  test("async: chamferError (known offset)", async () => {
    const tf = getTf();
    const pts0 = new Float32Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1,
      1,1,0, 1,0,1, 0,1,1, 1,1,1,
    ]);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i] = pts0[i] + 1;
      pts1[i+1] = pts0[i+1];
      pts1[i+2] = pts0[i+2];
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);

    const err = await tf.async.chamferError(src, tgt);
    approx(err, 0.5, "async chamfer error", 1e-4);
    log(`  async: error = ${err} (expected 0.5)`, "line-pass");

    src.delete();
    tgt.delete();
  });

});

// ============================================================================
// Rigid Alignment
// ============================================================================

describe("Rigid Alignment", () => {

  test("identity (same points)", () => {
    const tf = getTf();
    const pts = new Float32Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1, 1,1,0, 1,0,1, 0,1,1, 1,1,1,
    ]);
    const src = tf.pointCloud(pts);
    const tgt = tf.pointCloud(new Float32Array(pts));

    const T = tf.fitRigidAlignment(src, tgt);
    assert(T.shape[0] === 4 && T.shape[1] === 4, `expected [4,4], got [${T.shape}]`);

    // Should be near identity
    const d = T.data;
    approx(d[0], 1, "T[0,0]", 1e-4);
    approx(d[5], 1, "T[1,1]", 1e-4);
    approx(d[10], 1, "T[2,2]", 1e-4);
    approx(d[15], 1, "T[3,3]", 1e-4);
    approx(d[3], 0, "T[0,3]", 1e-4);
    approx(d[7], 0, "T[1,3]", 1e-4);
    approx(d[11], 0, "T[2,3]", 1e-4);
    log("  near-identity matrix", "line-pass");

    T.delete();
    src.delete();
    tgt.delete();
  });

  test("pure translation", () => {
    const tf = getTf();
    const pts0 = new Float32Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1,
    ]);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i]   = pts0[i]   + 1;
      pts1[i+1] = pts0[i+1] + 2;
      pts1[i+2] = pts0[i+2] + 3;
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = tf.fitRigidAlignment(src, tgt);
    const d = T.data;

    // Apply and verify
    const aligned = applyTransform(pts0, d);
    for (let i = 0; i < pts1.length; i++) {
      assert(Math.abs(aligned[i] - pts1[i]) < 1e-3,
        `point mismatch at ${i}: ${aligned[i]} vs ${pts1[i]}`);
    }
    log("  translation [1,2,3] recovered", "line-pass");

    T.delete();
    src.delete();
    tgt.delete();
  });

  test("rotation + translation", () => {
    const tf = getTf();
    const pts0 = new Float32Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1, 1,1,1,
    ]);
    const Ttrue = rotTransZ(45, 10, -5, 3);
    const pts1 = applyTransform(pts0, Ttrue);

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = tf.fitRigidAlignment(src, tgt);
    const d = T.data;

    const aligned = applyTransform(pts0, d);
    for (let i = 0; i < pts1.length; i++) {
      assert(Math.abs(aligned[i] - pts1[i]) < 1e-3,
        `point mismatch at ${i}: ${aligned[i]} vs ${pts1[i]}`);
    }
    log("  rot(45°)+trans recovered", "line-pass");

    T.delete();
    src.delete();
    tgt.delete();
  });

  test("output is valid rigid (det(R)≈1, R·Rᵀ≈I, last row [0,0,0,1])", () => {
    const tf = getTf();
    const pts0 = randomCenteredPoints(20, 42);
    const Ttrue = rotTransZ(30, 1, 2, 3);
    const pts1 = applyTransform(pts0, Ttrue);

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = tf.fitRigidAlignment(src, tgt);
    const d = T.data;

    // Last row
    approx(d[12], 0, "T[3,0]", 1e-5);
    approx(d[13], 0, "T[3,1]", 1e-5);
    approx(d[14], 0, "T[3,2]", 1e-5);
    approx(d[15], 1, "T[3,3]", 1e-5);
    log("  last row [0,0,0,1]", "line-pass");

    // det(R) ≈ 1
    const R = [d[0],d[1],d[2], d[4],d[5],d[6], d[8],d[9],d[10]];
    const det = R[0]*(R[4]*R[8]-R[5]*R[7]) - R[1]*(R[3]*R[8]-R[5]*R[6]) + R[2]*(R[3]*R[7]-R[4]*R[6]);
    assert(Math.abs(det - 1.0) < 1e-3, `det(R) should be ~1, got ${det}`);
    log(`  det(R) = ${det.toFixed(6)}`, "line-pass");

    T.delete();
    src.delete();
    tgt.delete();
  });

  // ==========================================================================
  test("async: fitRigidAlignment (rotation + translation)", async () => {
    const tf = getTf();
    const pts0 = new Float32Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1, 1,1,1,
    ]);
    const Ttrue = rotTransZ(45, 10, -5, 3);
    const pts1 = applyTransform(pts0, Ttrue);

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = await tf.async.fitRigidAlignment(src, tgt);
    const d = T.data;

    const aligned = applyTransform(pts0, d);
    for (let i = 0; i < pts1.length; i++) {
      assert(Math.abs(aligned[i] - pts1[i]) < 1e-3,
        `point mismatch at ${i}: ${aligned[i]} vs ${pts1[i]}`);
    }
    log("  async: rot(45)+trans recovered", "line-pass");

    T.delete();
    src.delete();
    tgt.delete();
  });

});

// ============================================================================
// ICP Alignment
// ============================================================================

describe("ICP Alignment", () => {

  test("identity (same cloud)", () => {
    const tf = getTf();
    const pts = randomPoints(100, 42);
    const src = tf.pointCloud(pts);
    const tgt = tf.pointCloud(new Float32Array(pts));

    const T = tf.fitIcpAlignment(src, tgt, { maxIterations: 10 });
    assert(T.shape[0] === 4 && T.shape[1] === 4, `expected [4,4], got [${T.shape}]`);

    const d = T.data;
    approx(d[0], 1, "T[0,0]", 1e-3);
    approx(d[5], 1, "T[1,1]", 1e-3);
    approx(d[10], 1, "T[2,2]", 1e-3);
    approx(d[15], 1, "T[3,3]", 1e-3);
    log("  near-identity", "line-pass");

    T.delete();
    src.delete();
    tgt.delete();
  });

  test("small translation converges", () => {
    const tf = getTf();
    const pts0 = randomPoints(200, 42);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i]   = pts0[i]   + 0.1;
      pts1[i+1] = pts0[i+1] + 0.05;
      pts1[i+2] = pts0[i+2] + 0.1;
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = tf.fitIcpAlignment(src, tgt, { maxIterations: 50 });
    const d = T.data;

    const aligned = applyTransform(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.01, `ICP should converge, got error ${err}`);
    log(`  chamfer error after ICP = ${err.toFixed(6)}`, "line-pass");

    T.delete();
    alignedPC.delete();
    src.delete();
    tgt.delete();
  });

  test("small rotation converges", () => {
    const tf = getTf();
    const pts0 = randomCenteredPoints(200, 42);
    const Ttrue = rotationZ(10);
    const pts1 = applyTransform(pts0, Ttrue);

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = tf.fitIcpAlignment(src, tgt, { maxIterations: 50 });
    const d = T.data;

    const aligned = applyTransform(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.01, `ICP should converge, got error ${err}`);
    log(`  chamfer error after ICP = ${err.toFixed(6)}`, "line-pass");

    T.delete();
    alignedPC.delete();
    src.delete();
    tgt.delete();
  });

  test("rotation + translation converges", () => {
    const tf = getTf();
    const pts0 = randomCenteredPoints(200, 42);
    const Ttrue = rotTransZ(15, 0.2, 0.15, 0.1);
    const pts1 = applyTransform(pts0, Ttrue);

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = tf.fitIcpAlignment(src, tgt, { maxIterations: 50 });
    const d = T.data;

    const aligned = applyTransform(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.05, `ICP should converge, got error ${err}`);
    log(`  chamfer error after ICP = ${err.toFixed(6)}`, "line-pass");

    T.delete();
    alignedPC.delete();
    src.delete();
    tgt.delete();
  });

  test("with normals (point-to-plane from mesh)", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 20, 20);
    const target = tf.PointCloud.fromMesh(sphere);

    // Transform source
    const srcPts = sphere.points;
    const Ttrue = rotTransZ(10, 0.1, 0.1, 0.05);
    const srcTransformed = applyTransform(srcPts.data, Ttrue);
    const source = tf.pointCloud(srcTransformed);

    const T = tf.fitIcpAlignment(source, target, { maxIterations: 50 });
    const d = T.data;

    const aligned = applyTransform(srcTransformed, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, target);
    assert(err < 0.05, `p2plane ICP should converge, got error ${err}`);
    log(`  p2plane chamfer = ${err.toFixed(6)}`, "line-pass");

    T.delete();
    alignedPC.delete();
    source.delete();
    target.delete();
    srcPts.delete();
    sphere.delete();
  });

  test("different k values", () => {
    const tf = getTf();
    const pts0 = randomPoints(100, 42);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i] = pts0[i] + 0.1;
      pts1[i+1] = pts0[i+1] + 0.1;
      pts1[i+2] = pts0[i+2] + 0.1;
    }
    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);

    for (const k of [1, 3, 5]) {
      const T = tf.fitIcpAlignment(src, tgt, { maxIterations: 20, k });
      assert(T.shape[0] === 4 && T.shape[1] === 4, `k=${k}: shape [${T.shape}]`);
      T.delete();
    }
    log("  k=1,3,5 all produce [4,4]", "line-pass");

    src.delete();
    tgt.delete();
  });

  test("with subsampling", () => {
    const tf = getTf();
    const pts0 = randomPoints(500, 42);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i] = pts0[i] + 0.1;
      pts1[i+1] = pts0[i+1] + 0.1;
      pts1[i+2] = pts0[i+2] + 0.1;
    }
    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);

    const T = tf.fitIcpAlignment(src, tgt, { maxIterations: 30, nSamples: 100 });
    assert(T.shape[0] === 4 && T.shape[1] === 4, `shape [${T.shape}]`);
    log("  nSamples=100 accepted", "line-pass");

    T.delete();
    src.delete();
    tgt.delete();
  });

  test("with outlier rejection", () => {
    const tf = getTf();
    const pts0 = randomPoints(100, 42);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i] = pts0[i] + 0.1;
      pts1[i+1] = pts0[i+1] + 0.1;
      pts1[i+2] = pts0[i+2] + 0.1;
    }
    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);

    const T = tf.fitIcpAlignment(src, tgt, { maxIterations: 20, outlierProportion: 0.1 });
    assert(T.shape[0] === 4 && T.shape[1] === 4, `shape [${T.shape}]`);
    log("  outlierProportion=0.1 accepted", "line-pass");

    T.delete();
    src.delete();
    tgt.delete();
  });

  test("different cloud sizes", () => {
    const tf = getTf();
    const pts0 = randomPoints(50, 42);
    const pts1 = randomPoints(100, 99);
    // Shift pts1 slightly
    for (let i = 0; i < pts1.length; i += 3) pts1[i] += 0.1;

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);

    const T = tf.fitIcpAlignment(src, tgt, { maxIterations: 20 });
    assert(T.shape[0] === 4 && T.shape[1] === 4, `shape [${T.shape}]`);
    log("  50 vs 100 points OK", "line-pass");

    T.delete();
    src.delete();
    tgt.delete();
  });

  // ==========================================================================
  test("async: fitIcpAlignment (small translation converges)", async () => {
    const tf = getTf();
    const pts0 = randomPoints(200, 42);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i]   = pts0[i]   + 0.1;
      pts1[i+1] = pts0[i+1] + 0.05;
      pts1[i+2] = pts0[i+2] + 0.1;
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = await tf.async.fitIcpAlignment(src, tgt, { maxIterations: 50 });
    const d = T.data;

    const aligned = applyTransform(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.01, `async ICP should converge, got error ${err}`);
    log(`  async: chamfer error after ICP = ${err.toFixed(6)}`, "line-pass");

    T.delete();
    alignedPC.delete();
    src.delete();
    tgt.delete();
  });

});

// ============================================================================
// OBB Alignment
// ============================================================================

describe("OBB Alignment", () => {

  test("translation (elongated box)", () => {
    const tf = getTf();
    const pts0 = elongatedBox(200, 4, 2, 1, 123);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i]   = pts0[i]   + 3;
      pts1[i+1] = pts0[i+1] + 5;
      pts1[i+2] = pts0[i+2] - 2;
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = tf.fitObbAlignment(src, tgt);
    assert(T.shape[0] === 4 && T.shape[1] === 4, `shape [${T.shape}]`);

    const d = T.data;
    const aligned = applyTransform(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.5, `OBB alignment error too high: ${err}`);
    log(`  chamfer after OBB = ${err.toFixed(4)}`, "line-pass");

    T.delete();
    alignedPC.delete();
    src.delete();
    tgt.delete();
  });

  test("rotation + translation (elongated box)", () => {
    const tf = getTf();
    const pts0 = elongatedBox(200, 4, 2, 1, 123);
    const Ttrue = rotTransZ(45, 5, -3, 2);
    const pts1 = applyTransform(pts0, Ttrue);

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = tf.fitObbAlignment(src, tgt);
    const d = T.data;

    const aligned = applyTransform(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.5, `OBB alignment error too high: ${err}`);
    log(`  chamfer after OBB = ${err.toFixed(4)}`, "line-pass");

    T.delete();
    alignedPC.delete();
    src.delete();
    tgt.delete();
  });

  test("sampleSize parameter", () => {
    const tf = getTf();
    const pts0 = elongatedBox(200, 4, 2, 1, 123);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i] = pts0[i] + 1;
      pts1[i+1] = pts0[i+1] + 2;
      pts1[i+2] = pts0[i+2] + 3;
    }
    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);

    const T10 = tf.fitObbAlignment(src, tgt, { sampleSize: 10 });
    const T200 = tf.fitObbAlignment(src, tgt, { sampleSize: 200 });
    assert(T10.shape[0] === 4, "sampleSize=10 OK");
    assert(T200.shape[0] === 4, "sampleSize=200 OK");
    log("  sampleSize=10,200 both accepted", "line-pass");

    T10.delete();
    T200.delete();
    src.delete();
    tgt.delete();
  });

  test("OBB as ICP initializer", () => {
    const tf = getTf();
    const pts0 = elongatedBox(200, 4, 2, 1, 123);
    const Ttrue = rotTransZ(30, 3, -2, 1);
    const pts1 = applyTransform(pts0, Ttrue);

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);

    // Coarse OBB alignment
    const Tobb = tf.fitObbAlignment(src, tgt);
    const obbData = Tobb.data;
    const afterObb = applyTransform(pts0, obbData);
    const afterObbPC = tf.pointCloud(afterObb);
    const errObb = tf.chamferError(afterObbPC, tgt);

    // Refine with ICP using OBB as init
    src.transformation = Tobb;
    const Ticp = tf.fitIcpAlignment(src, tgt, { maxIterations: 50 });
    const icpData = Ticp.data;

    // Compose: total = icp @ obb
    const Ttotal = matMul4x4(icpData, obbData);
    const afterIcp = applyTransform(pts0, Ttotal);
    const afterIcpPC = tf.pointCloud(afterIcp);
    const errIcp = tf.chamferError(afterIcpPC, tgt);

    assert(errIcp < 0.01, `ICP+OBB should converge, got error ${errIcp}`);
    log(`  OBB err=${errObb.toFixed(4)}, ICP refined=${errIcp.toFixed(4)}`, "line-pass");

    Ticp.delete();
    Tobb.delete();
    afterIcpPC.delete();
    afterObbPC.delete();
    src.delete();
    tgt.delete();
  });

  // ==========================================================================
  test("async: fitObbAlignment (translation)", async () => {
    const tf = getTf();
    const pts0 = elongatedBox(200, 4, 2, 1, 123);
    const pts1 = new Float32Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i]   = pts0[i]   + 3;
      pts1[i+1] = pts0[i+1] + 5;
      pts1[i+2] = pts0[i+2] - 2;
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = await tf.async.fitObbAlignment(src, tgt);
    assert(T.shape[0] === 4 && T.shape[1] === 4, `shape [${T.shape}]`);

    const d = T.data;
    const aligned = applyTransform(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.5, `async OBB alignment error too high: ${err}`);
    log(`  async: chamfer after OBB = ${err.toFixed(4)}`, "line-pass");

    T.delete();
    alignedPC.delete();
    src.delete();
    tgt.delete();
  });

});

// ============================================================================
// buildTree
// ============================================================================

describe("buildTree", () => {

  test("Mesh.buildTree() — second call is cached", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 30, 30);

    const t0 = performance.now();
    sphere.buildTree();
    const first = performance.now() - t0;

    const t1 = performance.now();
    sphere.buildTree();
    const second = performance.now() - t1;

    assert(second < first + 0.1, `second call should be cached: first=${first.toFixed(3)}ms, second=${second.toFixed(3)}ms`);
    log(`  first=${first.toFixed(3)}ms, second=${second.toFixed(3)}ms (cached)`, "line-pass");

    sphere.delete();
  });

  test("PointCloud.buildTree() — second call is cached", () => {
    const tf = getTf();
    const pts = randomPoints(5000, 42);
    const pc = tf.pointCloud(pts);

    const t0 = performance.now();
    pc.buildTree();
    const first = performance.now() - t0;

    const t1 = performance.now();
    pc.buildTree();
    const second = performance.now() - t1;

    assert(second < first + 0.1, `second call should be cached: first=${first.toFixed(3)}ms, second=${second.toFixed(3)}ms`);
    log(`  first=${first.toFixed(3)}ms, second=${second.toFixed(3)}ms (cached)`, "line-pass");

    pc.delete();
  });

});

// ============================================================================
// float64 dispatch
// ============================================================================

function applyTransform64(pts, T) {
  // pts: Float64Array [N*3], T: Float64Array [16] (row-major 4x4)
  const n = pts.length / 3;
  const out = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = pts[i * 3], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
    out[i * 3]     = T[0] * x + T[1] * y + T[2]  * z + T[3];
    out[i * 3 + 1] = T[4] * x + T[5] * y + T[6]  * z + T[7];
    out[i * 3 + 2] = T[8] * x + T[9] * y + T[10] * z + T[11];
  }
  return out;
}

function rotTransZ64(angleDeg, tx, ty, tz) {
  const a = angleDeg * Math.PI / 180;
  const c = Math.cos(a), s = Math.sin(a);
  return new Float64Array([
    c, -s,  0, tx,
    s,  c,  0, ty,
    0,  0,  1, tz,
    0,  0,  0,  1,
  ]);
}

describe("Registration (float64)", () => {

  test("chamferError (float64): identical clouds → ~0", () => {
    const tf = getTf();
    const pts = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]);
    const src = tf.pointCloud(pts);
    const tgt = tf.pointCloud(new Float64Array(pts));
    assert(src.dtype === "float64" && tgt.dtype === "float64", "both float64");

    const err = tf.chamferError(src, tgt);
    assert(typeof err === "number", "returns number");
    assert(err < 1e-10, `expected ~0, got ${err}`);
    log(`  f64 chamferError = ${err}`, "line-pass");

    src.delete(); tgt.delete();
  });

  test("chamferError (float64): known offset", () => {
    const tf = getTf();
    const pts0 = new Float64Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1,
      1,1,0, 1,0,1, 0,1,1, 1,1,1,
    ]);
    const pts1 = new Float64Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i] = pts0[i] + 1;
      pts1[i+1] = pts0[i+1];
      pts1[i+2] = pts0[i+2];
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    assert(src.dtype === "float64" && tgt.dtype === "float64", "both float64");

    const err = tf.chamferError(src, tgt);
    approx(err, 0.5, "f64 chamfer error", 1e-10);
    log(`  f64 chamferError = ${err} (expected 0.5)`, "line-pass");

    src.delete(); tgt.delete();
  });

  test("fitRigidAlignment (float64): rotation + translation", () => {
    const tf = getTf();
    const pts0 = new Float64Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1, 1,1,1,
    ]);
    const Ttrue = rotTransZ64(45, 10, -5, 3);
    const pts1 = applyTransform64(pts0, Ttrue);

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    assert(src.dtype === "float64" && tgt.dtype === "float64", "both float64");

    const T = tf.fitRigidAlignment(src, tgt);
    assert(T.dtype === "float64", `expected T dtype float64, got ${T.dtype}`);
    assert(T.shape[0] === 4 && T.shape[1] === 4, `expected [4,4], got [${T.shape}]`);
    const d = T.data;
    assert(d instanceof Float64Array, `expected Float64Array, got ${d.constructor.name}`);

    const aligned = applyTransform64(pts0, d);
    for (let i = 0; i < pts1.length; i++) {
      assert(Math.abs(aligned[i] - pts1[i]) < 1e-7,
        `point mismatch at ${i}: ${aligned[i]} vs ${pts1[i]}`);
    }
    log("  f64 rigid: rot(45)+trans recovered", "line-pass");

    T.delete(); src.delete(); tgt.delete();
  });

  test("fitIcpAlignment (float64): small translation converges", () => {
    const tf = getTf();
    const pts0 = new Float64Array(randomPoints(200, 42));
    const pts1 = new Float64Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i]   = pts0[i]   + 0.1;
      pts1[i+1] = pts0[i+1] + 0.05;
      pts1[i+2] = pts0[i+2] + 0.1;
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    assert(src.dtype === "float64" && tgt.dtype === "float64", "both float64");

    const T = tf.fitIcpAlignment(src, tgt, { maxIterations: 50 });
    assert(T.dtype === "float64", `expected T dtype float64, got ${T.dtype}`);
    assert(T.shape[0] === 4 && T.shape[1] === 4, `expected [4,4], got [${T.shape}]`);
    const d = T.data;
    assert(d instanceof Float64Array, `expected Float64Array, got ${d.constructor.name}`);

    const aligned = applyTransform64(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.01, `f64 ICP should converge, got error ${err}`);
    log(`  f64 ICP chamfer = ${err.toFixed(8)}`, "line-pass");

    T.delete(); alignedPC.delete(); src.delete(); tgt.delete();
  });

  test("fitIcpAlignment (float64) with normals", () => {
    const tf = getTf();
    const sphere = tf.sphereMesh(1.0, 20, 20, { dtype: "float64" });
    const target = tf.PointCloud.fromMesh(sphere);
    assert(target.dtype === "float64", `expected target dtype float64, got ${target.dtype}`);

    const srcPts = sphere.points;
    const Ttrue = rotTransZ64(10, 0.1, 0.1, 0.05);
    const srcTransformed = applyTransform64(srcPts.data, Ttrue);
    const source = tf.pointCloud(srcTransformed);
    assert(source.dtype === "float64", `expected source dtype float64, got ${source.dtype}`);

    const T = tf.fitIcpAlignment(source, target, { maxIterations: 50 });
    assert(T.dtype === "float64", `expected T dtype float64, got ${T.dtype}`);
    assert(T.shape[0] === 4 && T.shape[1] === 4, `expected [4,4], got [${T.shape}]`);
    const d = T.data;
    assert(d instanceof Float64Array, `expected Float64Array, got ${d.constructor.name}`);

    const aligned = applyTransform64(srcTransformed, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, target);
    assert(err < 0.05, `f64 p2plane ICP should converge, got error ${err}`);
    log(`  f64 p2plane chamfer = ${err.toFixed(8)}`, "line-pass");

    T.delete();
    alignedPC.delete();
    source.delete();
    target.delete();
    srcPts.delete();
    sphere.delete();
  });

  test("fitObbAlignment (float64): translation (elongated box)", () => {
    const tf = getTf();
    const ptsF32 = elongatedBox(200, 4, 2, 1, 123);
    const pts0 = new Float64Array(ptsF32);
    const pts1 = new Float64Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i]   = pts0[i]   + 3;
      pts1[i+1] = pts0[i+1] + 5;
      pts1[i+2] = pts0[i+2] - 2;
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    assert(src.dtype === "float64" && tgt.dtype === "float64", "both float64");

    const T = tf.fitObbAlignment(src, tgt);
    assert(T.dtype === "float64", `expected T dtype float64, got ${T.dtype}`);
    assert(T.shape[0] === 4 && T.shape[1] === 4, `shape [${T.shape}]`);
    const d = T.data;
    assert(d instanceof Float64Array, `expected Float64Array, got ${d.constructor.name}`);

    const aligned = applyTransform64(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.5, `f64 OBB alignment error too high: ${err}`);
    log(`  f64 OBB chamfer = ${err.toFixed(6)}`, "line-pass");

    T.delete(); alignedPC.delete(); src.delete(); tgt.delete();
  });

  test("async: chamferError (float64)", async () => {
    const tf = getTf();
    const pts0 = new Float64Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1,
      1,1,0, 1,0,1, 0,1,1, 1,1,1,
    ]);
    const pts1 = new Float64Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i] = pts0[i] + 1;
      pts1[i+1] = pts0[i+1];
      pts1[i+2] = pts0[i+2];
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const err = await tf.async.chamferError(src, tgt);
    approx(err, 0.5, "async f64 chamferError", 1e-10);
    log(`  async f64 chamferError = ${err}`, "line-pass");

    src.delete(); tgt.delete();
  });

  test("async: fitRigidAlignment (float64)", async () => {
    const tf = getTf();
    const pts0 = new Float64Array([
      0,0,0, 1,0,0, 0,1,0, 0,0,1, 1,1,1,
    ]);
    const Ttrue = rotTransZ64(45, 10, -5, 3);
    const pts1 = applyTransform64(pts0, Ttrue);

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = await tf.async.fitRigidAlignment(src, tgt);
    assert(T.dtype === "float64", `expected T dtype float64, got ${T.dtype}`);
    const d = T.data;
    assert(d instanceof Float64Array, `expected Float64Array, got ${d.constructor.name}`);

    const aligned = applyTransform64(pts0, d);
    for (let i = 0; i < pts1.length; i++) {
      assert(Math.abs(aligned[i] - pts1[i]) < 1e-7,
        `point mismatch at ${i}: ${aligned[i]} vs ${pts1[i]}`);
    }
    log("  async f64 rigid: rot(45)+trans recovered", "line-pass");

    T.delete(); src.delete(); tgt.delete();
  });

  test("async: fitIcpAlignment (float64)", async () => {
    const tf = getTf();
    const pts0 = new Float64Array(randomPoints(200, 42));
    const pts1 = new Float64Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i]   = pts0[i]   + 0.1;
      pts1[i+1] = pts0[i+1] + 0.05;
      pts1[i+2] = pts0[i+2] + 0.1;
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = await tf.async.fitIcpAlignment(src, tgt, { maxIterations: 50 });
    assert(T.dtype === "float64", `expected T dtype float64, got ${T.dtype}`);
    const d = T.data;
    assert(d instanceof Float64Array, `expected Float64Array, got ${d.constructor.name}`);

    const aligned = applyTransform64(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.01, `async f64 ICP should converge, got error ${err}`);
    log(`  async f64 ICP chamfer = ${err.toFixed(8)}`, "line-pass");

    T.delete(); alignedPC.delete(); src.delete(); tgt.delete();
  });

  test("async: fitObbAlignment (float64)", async () => {
    const tf = getTf();
    const ptsF32 = elongatedBox(200, 4, 2, 1, 123);
    const pts0 = new Float64Array(ptsF32);
    const pts1 = new Float64Array(pts0.length);
    for (let i = 0; i < pts0.length; i += 3) {
      pts1[i]   = pts0[i]   + 3;
      pts1[i+1] = pts0[i+1] + 5;
      pts1[i+2] = pts0[i+2] - 2;
    }

    const src = tf.pointCloud(pts0);
    const tgt = tf.pointCloud(pts1);
    const T = await tf.async.fitObbAlignment(src, tgt);
    assert(T.dtype === "float64", `expected T dtype float64, got ${T.dtype}`);
    const d = T.data;
    assert(d instanceof Float64Array, `expected Float64Array, got ${d.constructor.name}`);

    const aligned = applyTransform64(pts0, d);
    const alignedPC = tf.pointCloud(aligned);
    const err = tf.chamferError(alignedPC, tgt);
    assert(err < 0.5, `async f64 OBB error too high: ${err}`);
    log(`  async f64 OBB chamfer = ${err.toFixed(6)}`, "line-pass");

    T.delete(); alignedPC.delete(); src.delete(); tgt.delete();
  });

});

// ============================================================================
// dtype mismatch
// ============================================================================

describe("Registration (dtype mismatch)", () => {

  test("chamferError dtype mismatch throws", () => {
    const tf = getTf();
    const ptsF32 = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const ptsF64 = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const src = tf.pointCloud(ptsF32);
    const tgt = tf.pointCloud(ptsF64);
    let threw = false;
    try { tf.chamferError(src, tgt); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  chamferError dtype mismatch throws`, "line-pass");
    src.delete(); tgt.delete();
  });

  test("fitRigidAlignment dtype mismatch throws", () => {
    const tf = getTf();
    const ptsF32 = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    const ptsF64 = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    const src = tf.pointCloud(ptsF32);
    const tgt = tf.pointCloud(ptsF64);
    let threw = false;
    try { tf.fitRigidAlignment(src, tgt); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  fitRigidAlignment dtype mismatch throws`, "line-pass");
    src.delete(); tgt.delete();
  });

  test("fitIcpAlignment dtype mismatch throws", () => {
    const tf = getTf();
    const ptsF32 = new Float32Array(randomPoints(50, 7));
    const ptsF64 = new Float64Array(ptsF32);
    const src = tf.pointCloud(ptsF32);
    const tgt = tf.pointCloud(ptsF64);
    let threw = false;
    try { tf.fitIcpAlignment(src, tgt); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  fitIcpAlignment dtype mismatch throws`, "line-pass");
    src.delete(); tgt.delete();
  });

  test("fitObbAlignment dtype mismatch throws", () => {
    const tf = getTf();
    const ptsF32 = elongatedBox(50, 4, 2, 1, 3);
    const ptsF64 = new Float64Array(ptsF32);
    const src = tf.pointCloud(ptsF32);
    const tgt = tf.pointCloud(ptsF64);
    let threw = false;
    try { tf.fitObbAlignment(src, tgt); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  fitObbAlignment dtype mismatch throws`, "line-pass");
    src.delete(); tgt.delete();
  });

  test("async chamferError dtype mismatch throws", async () => {
    const tf = getTf();
    const ptsF32 = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const ptsF64 = new Float64Array(ptsF32);
    const src = tf.pointCloud(ptsF32);
    const tgt = tf.pointCloud(ptsF64);
    let threw = false;
    try { await tf.async.chamferError(src, tgt); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  async chamferError dtype mismatch throws`, "line-pass");
    src.delete(); tgt.delete();
  });

  test("async fitIcpAlignment dtype mismatch throws", async () => {
    const tf = getTf();
    const ptsF32 = new Float32Array(randomPoints(50, 7));
    const ptsF64 = new Float64Array(ptsF32);
    const src = tf.pointCloud(ptsF32);
    const tgt = tf.pointCloud(ptsF64);
    let threw = false;
    try { await tf.async.fitIcpAlignment(src, tgt); } catch (e) {
      if (/dtype mismatch/.test(e.message)) threw = true;
    }
    assert(threw, "expected dtype mismatch error");
    log(`  async fitIcpAlignment dtype mismatch throws`, "line-pass");
    src.delete(); tgt.delete();
  });

});
