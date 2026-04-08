import { describe, test, log, assert, approx, getTf } from "./harness.mjs";

describe("Transformations", () => {

  test("makeTranslation(x, y, z)", () => {
    const tf = getTf();
    const t = tf.makeTranslation(1, 2, 3);
    assert(t.shape[0] === 4 && t.shape[1] === 4, `shape: [${t.shape}]`);
    // Row-major: translation in column 3 (indices 3, 7, 11)
    approx(t.data[0], 1, "R[0,0]"); approx(t.data[3], 1, "tx");
    approx(t.data[5], 1, "R[1,1]"); approx(t.data[7], 2, "ty");
    approx(t.data[10], 1, "R[2,2]"); approx(t.data[11], 3, "tz");
    approx(t.data[15], 1, "w");
    log("  makeTranslation(1, 2, 3)", "line-pass");
    t.delete();
  });

  test("makeTranslation([x, y, z])", () => {
    const tf = getTf();
    const t = tf.makeTranslation([4, 5, 6]);
    approx(t.data[3], 4); approx(t.data[7], 5); approx(t.data[11], 6);
    log("  makeTranslation([4, 5, 6])", "line-pass");
    t.delete();
  });

  test("makeTranslation(ndarray)", () => {
    const tf = getTf();
    const v = tf.ndarray(new Float32Array([7, 8, 9]), [3]);
    const t = tf.makeTranslation(v);
    approx(t.data[3], 7); approx(t.data[7], 8); approx(t.data[11], 9);
    log("  makeTranslation(ndarray)", "line-pass");
    t.delete(); v.delete();
  });

  test("makeRotation 90° around x", () => {
    const tf = getTf();
    const r = tf.makeRotation(90, "x");
    assert(r.shape[0] === 4 && r.shape[1] === 4, "4x4");
    // Rotation around x: y→z, z→-y
    approx(r.data[0], 1, "R[0,0]");
    approx(r.data[5], 0, "R[1,1] ≈ cos90", 1e-5);
    approx(r.data[6], -1, "R[1,2] ≈ -sin90", 1e-5);
    approx(r.data[9], 1, "R[2,1] ≈ sin90", 1e-5);
    approx(r.data[10], 0, "R[2,2] ≈ cos90", 1e-5);
    log("  makeRotation(90, 'x')", "line-pass");
    r.delete();
  });

  test("makeRotation 90° around y", () => {
    const tf = getTf();
    const r = tf.makeRotation(90, "y");
    // Rotation around y: z→x, x→-z
    approx(r.data[0], 0, "R[0,0]", 1e-5);
    approx(r.data[2], 1, "R[0,2]", 1e-5);
    approx(r.data[5], 1, "R[1,1]");
    approx(r.data[8], -1, "R[2,0]", 1e-5);
    approx(r.data[10], 0, "R[2,2]", 1e-5);
    log("  makeRotation(90, 'y')", "line-pass");
    r.delete();
  });

  test("makeRotation 90° around z", () => {
    const tf = getTf();
    const r = tf.makeRotation(90, "z");
    // Rotation around z: x→y, y→-x
    approx(r.data[0], 0, "R[0,0]", 1e-5);
    approx(r.data[1], -1, "R[0,1]", 1e-5);
    approx(r.data[4], 1, "R[1,0]", 1e-5);
    approx(r.data[5], 0, "R[1,1]", 1e-5);
    approx(r.data[10], 1, "R[2,2]");
    log("  makeRotation(90, 'z')", "line-pass");
    r.delete();
  });

  test("makeRotation arbitrary axis", () => {
    const tf = getTf();
    // 180° around z-axis via [0,0,1]
    const r = tf.makeRotation(180, [0, 0, 1]);
    approx(r.data[0], -1, "R[0,0]", 1e-5);
    approx(r.data[5], -1, "R[1,1]", 1e-5);
    approx(r.data[10], 1, "R[2,2]");
    log("  makeRotation(180, [0,0,1])", "line-pass");
    r.delete();
  });

  test("makeRotation with pivot", () => {
    const tf = getTf();
    // 90° around z with pivot at (1, 0, 0)
    // Point (1,0,0) should stay at (1,0,0)
    const r = tf.makeRotation(90, "z", [1, 0, 0]);
    assert(r.shape[0] === 4 && r.shape[1] === 4, "4x4");
    // Apply to pivot: R * pivot + t should = pivot
    const px = r.data[0]*1 + r.data[1]*0 + r.data[2]*0 + r.data[3];
    const py = r.data[4]*1 + r.data[5]*0 + r.data[6]*0 + r.data[7];
    approx(px, 1, "pivot x preserved", 1e-5);
    approx(py, 0, "pivot y preserved", 1e-5);
    log("  makeRotation(90, 'z', pivot)", "line-pass");
    r.delete();
  });

  test("makeRandomRotation", () => {
    const tf = getTf();
    const r = tf.makeRandomRotation();
    assert(r.shape[0] === 4 && r.shape[1] === 4, "4x4");
    // Last row should be [0, 0, 0, 1]
    approx(r.data[12], 0); approx(r.data[13], 0);
    approx(r.data[14], 0); approx(r.data[15], 1);
    log("  makeRandomRotation()", "line-pass");
    r.delete();
  });

  test("makeRandomRotation with pivot", () => {
    const tf = getTf();
    const r = tf.makeRandomRotation([1, 2, 3]);
    assert(r.shape[0] === 4 && r.shape[1] === 4, "4x4");
    approx(r.data[15], 1, "w=1");
    log("  makeRandomRotation(pivot)", "line-pass");
    r.delete();
  });

  // ==========================================================================
  test("inverted 4x4: T * T^-1 = I", () => {
    const tf = getTf();
    const t = tf.makeTranslation(1, 2, 3);
    const r = tf.makeRotation(37, "y");
    const m = t.matMul(r);

    const inv = tf.inverted(m);
    assert(inv.shape[0] === 4 && inv.shape[1] === 4, "4x4");

    // Verify M * M^-1 ≈ I
    const id = m.matMul(inv);
    const d = id.data;
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) {
        const expected = i === j ? 1 : 0;
        approx(d[i * 4 + j], expected, `I[${i},${j}]`);
      }
    log("  inverted 4x4: M * M^-1 ≈ I", "line-pass");

    id.delete(); inv.delete(); m.delete(); r.delete(); t.delete();
  });

  // ==========================================================================
  test("inverted 3x3 (2D affine)", () => {
    const tf = getTf();
    // 2D affine: rotation + translation
    // | cos -sin tx |
    // | sin  cos ty |
    // |  0    0   1 |
    const c = Math.cos(Math.PI / 4);
    const s = Math.sin(Math.PI / 4);
    const m = tf.ndarray(new Float32Array([
      c, -s, 2,
      s,  c, 3,
      0,  0, 1,
    ]), [3, 3]);

    const inv = tf.inverted(m);
    assert(inv.shape[0] === 3 && inv.shape[1] === 3, "3x3");

    const id = m.matMul(inv);
    const d = id.data;
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        const expected = i === j ? 1 : 0;
        approx(d[i * 3 + j], expected, `I[${i},${j}]`);
      }
    log("  inverted 3x3: M * M^-1 ≈ I", "line-pass");

    id.delete(); inv.delete(); m.delete();
  });

  // ==========================================================================
  test("inverted throws on invalid shape", () => {
    const tf = getTf();
    const m = tf.ndarray(new Float32Array(4), [2, 2]);
    let threw = false;
    try { tf.inverted(m); } catch (e) { threw = true; }
    assert(threw, "expected throw on 2x2 matrix");
    log("  inverted 2x2 → throws", "line-pass");
    m.delete();
  });

});
