import { describe, test, log, assert, approx, getTf } from "./harness.mjs";

describe("NDArray", () => {

  // ==========================================================================
  test("arithmetic", () => {
    const tf = getTf();

    // add — same shape
    const a1 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const b1 = tf.ndarray(new Float32Array([10, 20, 30, 40, 50, 60]), [2, 3]);
    const c1 = a1.add(b1);
    approx(c1.data[0], 11); approx(c1.data[3], 44); approx(c1.data[5], 66);
    log("  add same shape", "line-pass");
    c1.delete(); b1.delete(); a1.delete();

    // add — broadcasting [N,3] + [3]
    const a2 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const b2 = tf.ndarray(new Float32Array([10, 20, 30]), [3]);
    const c2 = a2.add(b2);
    approx(c2.data[0], 11); approx(c2.data[3], 14);
    assert(c2.shape[0] === 2 && c2.shape[1] === 3, "broadcast shape");
    log("  add broadcast [2,3]+[3]", "line-pass");
    c2.delete(); b2.delete(); a2.delete();

    // add — scalar
    const a3 = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const c3 = a3.add(10);
    approx(c3.data[0], 11); approx(c3.data[2], 13);
    log("  add scalar", "line-pass");
    c3.delete(); a3.delete();

    // sub
    const a4 = tf.ndarray(new Float32Array([10, 20, 30]), [3]);
    const b4 = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const c4 = a4.sub(b4);
    approx(c4.data[0], 9); approx(c4.data[2], 27);
    log("  sub", "line-pass");
    c4.delete(); b4.delete(); a4.delete();

    // sub — scalar
    const a5 = tf.ndarray(new Float32Array([10, 20, 30]), [3]);
    const c5 = a5.sub(5);
    approx(c5.data[0], 5); approx(c5.data[2], 25);
    log("  sub scalar", "line-pass");
    c5.delete(); a5.delete();

    // mul
    const a6 = tf.ndarray(new Float32Array([2, 3, 4]), [3]);
    const b6 = tf.ndarray(new Float32Array([5, 6, 7]), [3]);
    const c6 = a6.mul(b6);
    approx(c6.data[0], 10); approx(c6.data[2], 28);
    log("  mul element-wise", "line-pass");
    c6.delete(); b6.delete(); a6.delete();

    // mul — scalar
    const a7 = tf.ndarray(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const c7 = a7.mul(3);
    approx(c7.data[0], 3); approx(c7.data[3], 12);
    log("  mul scalar", "line-pass");
    c7.delete(); a7.delete();

    // div
    const a8 = tf.ndarray(new Float32Array([6, 8, 10]), [3]);
    const b8 = tf.ndarray(new Float32Array([2, 4, 5]), [3]);
    const c8 = a8.div(b8);
    approx(c8.data[0], 3); approx(c8.data[1], 2); approx(c8.data[2], 2);
    log("  div", "line-pass");
    c8.delete(); b8.delete(); a8.delete();

    // div — scalar
    const a9 = tf.ndarray(new Float32Array([10, 20, 30]), [3]);
    const c9 = a9.div(5);
    approx(c9.data[0], 2); approx(c9.data[2], 6);
    log("  div scalar", "line-pass");
    c9.delete(); a9.delete();

    // div — broadcast
    const a10 = tf.ndarray(new Float32Array([2, 4, 6, 8]), [2, 2]);
    const b10 = tf.ndarray(new Float32Array([2, 1]), [1, 2]);
    const c10 = a10.div(b10);
    approx(c10.data[0], 1); approx(c10.data[1], 4); approx(c10.data[2], 3); approx(c10.data[3], 8);
    log("  div broadcast", "line-pass");
    c10.delete(); b10.delete(); a10.delete();

    // int32 add + mul
    const ai = tf.ndarray(new Int32Array([1, 2, 3]), [3]);
    const bi = tf.ndarray(new Int32Array([4, 5, 6]), [3]);
    const ci = ai.add(bi);
    assert(ci.data[0] === 5 && ci.data[2] === 9, "int add");
    const di = ai.mul(bi);
    assert(di.data[0] === 4 && di.data[2] === 18, "int mul");
    log("  int32 add + mul", "line-pass");
    di.delete(); ci.delete(); bi.delete(); ai.delete();

    // in-place: add_, mul_, div_, chaining
    const ip1 = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const ip2 = tf.ndarray(new Float32Array([10, 20, 30]), [3]);
    ip1.add_(ip2);
    approx(ip1.data[0], 11); approx(ip1.data[2], 33);
    log("  add_ buffer", "line-pass");
    ip2.delete(); ip1.delete();

    const ip3 = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    ip3.add_(100);
    approx(ip3.data[0], 101);
    log("  add_ scalar", "line-pass");
    ip3.delete();

    const ip4 = tf.ndarray(new Float32Array([2, 4, 6]), [3]);
    ip4.mul_(0.5);
    approx(ip4.data[0], 1); approx(ip4.data[2], 3);
    log("  mul_ scalar", "line-pass");
    ip4.delete();

    const ip5 = tf.ndarray(new Float32Array([12, 9, 6]), [3]);
    ip5.div_(3);
    approx(ip5.data[0], 4); approx(ip5.data[2], 2);
    log("  div_ scalar", "line-pass");
    ip5.delete();

    const ip6 = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    ip6.add_(10).mul_(2);
    approx(ip6.data[0], 22); approx(ip6.data[2], 26);
    log("  chaining add_.mul_", "line-pass");
    ip6.delete();
  });

  // ==========================================================================
  test("unary math", () => {
    const tf = getTf();

    // neg
    const a1 = tf.ndarray(new Float32Array([1, -2, 3]), [3]);
    const b1 = tf.neg(a1);
    approx(b1.data[0], -1); approx(b1.data[1], 2); approx(b1.data[2], -3);
    log("  neg", "line-pass");
    b1.delete(); a1.delete();

    const ai = tf.ndarray(new Int32Array([10, -20]), [2]);
    const bi = tf.neg(ai);
    assert(bi.data[0] === -10 && bi.data[1] === 20, "neg int32");
    log("  neg int32", "line-pass");
    bi.delete(); ai.delete();

    // abs
    const a3 = tf.ndarray(new Float32Array([-1, 2, -3]), [3]);
    const b3 = tf.abs(a3);
    approx(b3.data[0], 1); approx(b3.data[2], 3);
    log("  abs", "line-pass");
    b3.delete(); a3.delete();

    // sqrt
    const a5 = tf.ndarray(new Float32Array([4, 9, 16]), [3]);
    const b5 = tf.sqrt(a5);
    approx(b5.data[0], 2); approx(b5.data[1], 3); approx(b5.data[2], 4);
    log("  sqrt", "line-pass");
    b5.delete(); a5.delete();

    const a6 = tf.ndarray(new Float32Array([25, 1]), [2]);
    tf.sqrt_(a6);
    approx(a6.data[0], 5); approx(a6.data[1], 1);
    log("  sqrt_ in-place", "line-pass");
    a6.delete();
  });

  // ==========================================================================
  test("math functions", () => {
    const tf = getTf();

    // sin / cos / tan
    const a1 = tf.ndarray(new Float32Array([0, Math.PI / 2, Math.PI]), [3]);
    const s = tf.sin(a1);
    approx(s.data[0], 0); approx(s.data[1], 1); approx(s.data[2], 0, "sin(pi)", 1e-5);
    log("  sin", "line-pass");
    const c = tf.cos(a1);
    approx(c.data[0], 1); approx(c.data[1], 0, "cos(pi/2)", 1e-5); approx(c.data[2], -1);
    log("  cos", "line-pass");
    const t = tf.tan(tf.ndarray(new Float32Array([0, Math.PI / 4]), [2]));
    approx(t.data[0], 0); approx(t.data[1], 1);
    log("  tan", "line-pass");
    t.delete(); c.delete(); s.delete(); a1.delete();

    // asin / acos / atan
    const a2 = tf.ndarray(new Float32Array([0, 0.5, 1]), [3]);
    const as2 = tf.asin(a2);
    approx(as2.data[0], 0); approx(as2.data[2], Math.PI / 2);
    log("  asin", "line-pass");
    const ac2 = tf.acos(a2);
    approx(ac2.data[0], Math.PI / 2); approx(ac2.data[2], 0);
    log("  acos", "line-pass");
    const at2 = tf.atan(a2);
    approx(at2.data[0], 0); approx(at2.data[1], Math.atan(0.5));
    log("  atan", "line-pass");
    at2.delete(); ac2.delete(); as2.delete(); a2.delete();

    // exp / log / log2 / log10
    const a3 = tf.ndarray(new Float32Array([0, 1, 2]), [3]);
    const e = tf.exp(a3);
    approx(e.data[0], 1); approx(e.data[1], Math.E);
    log("  exp", "line-pass");
    e.delete(); a3.delete();

    const a4 = tf.ndarray(new Float32Array([1, Math.E, Math.E * Math.E]), [3]);
    const l = tf.log(a4);
    approx(l.data[0], 0); approx(l.data[1], 1); approx(l.data[2], 2);
    log("  log", "line-pass");
    l.delete(); a4.delete();

    const a5 = tf.ndarray(new Float32Array([1, 2, 4, 8]), [4]);
    const l2 = tf.log2(a5);
    approx(l2.data[0], 0); approx(l2.data[1], 1); approx(l2.data[2], 2); approx(l2.data[3], 3);
    log("  log2", "line-pass");
    l2.delete(); a5.delete();

    const a6 = tf.ndarray(new Float32Array([1, 10, 100, 1000]), [4]);
    const l10 = tf.log10(a6);
    approx(l10.data[0], 0); approx(l10.data[1], 1); approx(l10.data[2], 2); approx(l10.data[3], 3);
    log("  log10", "line-pass");
    l10.delete(); a6.delete();

    // floor / ceil / round
    const a7 = tf.ndarray(new Float32Array([1.2, 2.7, -0.3, -1.8]), [4]);
    const fl = tf.floor(a7);
    approx(fl.data[0], 1); approx(fl.data[1], 2); approx(fl.data[2], -1); approx(fl.data[3], -2);
    log("  floor", "line-pass");
    const ce = tf.ceil(a7);
    approx(ce.data[0], 2); approx(ce.data[1], 3); approx(ce.data[2], 0); approx(ce.data[3], -1);
    log("  ceil", "line-pass");
    const rn = tf.round(a7);
    approx(rn.data[0], 1); approx(rn.data[1], 3); approx(rn.data[2], 0); approx(rn.data[3], -2);
    log("  round", "line-pass");
    rn.delete(); ce.delete(); fl.delete(); a7.delete();

    // pow
    const a8 = tf.ndarray(new Float32Array([2, 3, 4]), [3]);
    const p = tf.pow(a8, 2);
    approx(p.data[0], 4); approx(p.data[1], 9); approx(p.data[2], 16);
    log("  pow", "line-pass");
    p.delete(); a8.delete();

    const a9 = tf.ndarray(new Float32Array([4, 9]), [2]);
    tf.pow_(a9, 0.5);
    approx(a9.data[0], 2); approx(a9.data[1], 3);
    log("  pow_ in-place", "line-pass");
    a9.delete();

    // clip
    const a10 = tf.ndarray(new Float32Array([-2, 0, 3, 5, 10]), [5]);
    const cl = tf.clip(a10, 0, 5);
    approx(cl.data[0], 0); approx(cl.data[2], 3); approx(cl.data[4], 5);
    log("  clip", "line-pass");
    cl.delete(); a10.delete();

    // in-place trig
    const a11 = tf.ndarray(new Float32Array([0, Math.PI / 2]), [2]);
    tf.sin_(a11);
    approx(a11.data[0], 0); approx(a11.data[1], 1);
    log("  sin_ in-place", "line-pass");
    a11.delete();

    const a12 = tf.ndarray(new Float32Array([0, 1]), [2]);
    tf.exp_(a12);
    approx(a12.data[0], 1); approx(a12.data[1], Math.E);
    log("  exp_ in-place", "line-pass");
    a12.delete();
  });

  // ==========================================================================
  test("logical ops", () => {
    const tf = getTf();

    const a1 = tf.ndarray(new Int8Array([1, 0, 1, 0]), [4]).as("bool");
    const b1 = a1.not();
    assert(b1.data[0] === 0 && b1.data[1] === 1 && b1.data[2] === 0 && b1.data[3] === 1, "not");
    log("  not", "line-pass");
    b1.delete(); a1.delete();

    const a2 = tf.ndarray(new Int8Array([1, 1, 0, 0]), [4]).as("bool");
    const b2 = tf.ndarray(new Int8Array([1, 0, 1, 0]), [4]).as("bool");
    const c2 = a2.and(b2);
    assert(c2.data[0] === 1 && c2.data[1] === 0 && c2.data[2] === 0 && c2.data[3] === 0, "and");
    log("  and", "line-pass");
    c2.delete(); b2.delete(); a2.delete();

    const a3 = tf.ndarray(new Int8Array([1, 1, 0, 0]), [4]).as("bool");
    const b3 = tf.ndarray(new Int8Array([1, 0, 1, 0]), [4]).as("bool");
    const c3 = a3.or(b3);
    assert(c3.data[0] === 1 && c3.data[1] === 1 && c3.data[2] === 1 && c3.data[3] === 0, "or");
    log("  or", "line-pass");
    c3.delete(); b3.delete(); a3.delete();
  });

  // ==========================================================================
  test("matMul", () => {
    const tf = getTf();

    // 2D: 2x3 @ 3x2
    const a1 = tf.ndarray(new Float32Array([1,2,3,4,5,6]), [2, 3]);
    const b1 = tf.ndarray(new Float32Array([7,8,9,10,11,12]), [3, 2]);
    const c1 = a1.matMul(b1);
    assert(c1.shape[0] === 2 && c1.shape[1] === 2, "2D shape");
    approx(c1.data[0], 58); approx(c1.data[1], 64);
    approx(c1.data[2], 139); approx(c1.data[3], 154);
    log("  2x3 @ 3x2", "line-pass");
    c1.delete(); b1.delete(); a1.delete();

    // identity
    const eye = tf.ndarray(new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]), [4, 4]);
    const m = tf.ndarray(new Float32Array([1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16]), [4, 4]);
    const c2 = eye.matMul(m);
    for (let i = 0; i < 16; i++) approx(c2.data[i], i + 1, `identity [${i}]`);
    log("  identity @ M = M", "line-pass");
    c2.delete(); m.delete(); eye.delete();

    // batch broadcast: [B,M,K] x [B,K,N] → [B,M,N]
    const a3 = tf.ndarray(new Float32Array([
      1,0, 0,1,   // batch 0: identity 2x2
      2,0, 0,2,   // batch 1: scale by 2
    ]), [2, 2, 2]);
    const b3 = tf.ndarray(new Float32Array([
      1,2, 3,4,   // batch 0
      5,6, 7,8,   // batch 1
    ]), [2, 2, 2]);
    const c3 = a3.matMul(b3);
    assert(c3.shape[0] === 2 && c3.shape[1] === 2 && c3.shape[2] === 2, "batch shape [2,2,2]");
    // batch 0: I @ [[1,2],[3,4]] = [[1,2],[3,4]]
    approx(c3.data[0], 1); approx(c3.data[1], 2);
    approx(c3.data[2], 3); approx(c3.data[3], 4);
    // batch 1: 2I @ [[5,6],[7,8]] = [[10,12],[14,16]]
    approx(c3.data[4], 10); approx(c3.data[5], 12);
    approx(c3.data[6], 14); approx(c3.data[7], 16);
    log("  batch [B,M,K] x [B,K,N]", "line-pass");
    c3.delete(); b3.delete(); a3.delete();

    // broadcast: [M,K] x [B,K,N] → [B,M,N]
    const a4 = tf.ndarray(new Float32Array([1,2, 3,4]), [2, 2]); // single matrix
    const b4 = tf.ndarray(new Float32Array([
      1,0, 0,1,   // batch 0: identity
      0,1, 1,0,   // batch 1: swap cols
    ]), [2, 2, 2]);
    const c4 = a4.matMul(b4);
    assert(c4.shape[0] === 2 && c4.shape[1] === 2 && c4.shape[2] === 2, "broadcast shape [2,2,2]");
    // batch 0: [[1,2],[3,4]] @ I = [[1,2],[3,4]]
    approx(c4.data[0], 1); approx(c4.data[1], 2);
    approx(c4.data[2], 3); approx(c4.data[3], 4);
    // batch 1: [[1,2],[3,4]] @ [[0,1],[1,0]] = [[2,1],[4,3]]
    approx(c4.data[4], 2); approx(c4.data[5], 1);
    approx(c4.data[6], 4); approx(c4.data[7], 3);
    log("  broadcast [M,K] x [B,K,N]", "line-pass");
    c4.delete(); b4.delete(); a4.delete();

    // broadcast: [B,M,K] x [K,N] → [B,M,N]
    const a5 = tf.ndarray(new Float32Array([
      1,0, 0,1,   // batch 0: identity
      2,0, 0,2,   // batch 1: scale
    ]), [2, 2, 2]);
    const b5 = tf.ndarray(new Float32Array([5,6, 7,8]), [2, 2]); // single matrix
    const c5 = a5.matMul(b5);
    assert(c5.shape[0] === 2 && c5.shape[1] === 2 && c5.shape[2] === 2, "broadcast shape");
    // batch 0: I @ [[5,6],[7,8]] = [[5,6],[7,8]]
    approx(c5.data[0], 5); approx(c5.data[1], 6);
    approx(c5.data[2], 7); approx(c5.data[3], 8);
    // batch 1: 2I @ [[5,6],[7,8]] = [[10,12],[14,16]]
    approx(c5.data[4], 10); approx(c5.data[5], 12);
    approx(c5.data[6], 14); approx(c5.data[7], 16);
    log("  broadcast [B,M,K] x [K,N]", "line-pass");
    c5.delete(); b5.delete(); a5.delete();
  });

  // ==========================================================================
  test("dot & cross", () => {
    const tf = getTf();

    // dot 1D
    const a1 = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const b1 = tf.ndarray(new Float32Array([4, 5, 6]), [3]);
    const r1 = tf.dot(a1, b1);
    approx(r1.data[0], 32);
    log("  dot 1D: 32", "line-pass");
    r1.delete(); b1.delete(); a1.delete();

    // dot batch
    const a2 = tf.ndarray(new Float32Array([1,0,0, 0,1,0]), [2, 3]);
    const b2 = tf.ndarray(new Float32Array([1,1,1, 1,1,1]), [2, 3]);
    const r2 = tf.dot(a2, b2);
    assert(r2.shape[0] === 2, "dot batch shape");
    approx(r2.data[0], 1); approx(r2.data[1], 1);
    log("  dot batch", "line-pass");
    r2.delete(); b2.delete(); a2.delete();

    // cross 1D
    const a3 = tf.ndarray(new Float32Array([1, 0, 0]), [3]);
    const b3 = tf.ndarray(new Float32Array([0, 1, 0]), [3]);
    const r3 = tf.cross(a3, b3);
    approx(r3.data[0], 0); approx(r3.data[1], 0); approx(r3.data[2], 1);
    log("  cross: x * y = z", "line-pass");
    r3.delete(); b3.delete(); a3.delete();

    // cross batch
    const a4 = tf.ndarray(new Float32Array([1,0,0, 0,1,0]), [2, 3]);
    const b4 = tf.ndarray(new Float32Array([0,1,0, 0,0,1]), [2, 3]);
    const r4 = tf.cross(a4, b4);
    assert(r4.shape[0] === 2 && r4.shape[1] === 3, "cross batch shape");
    approx(r4.data[0], 0); approx(r4.data[1], 0); approx(r4.data[2], 1);
    approx(r4.data[3], 1); approx(r4.data[4], 0); approx(r4.data[5], 0);
    log("  cross batch", "line-pass");
    r4.delete(); b4.delete(); a4.delete();
  });

  // ==========================================================================
  test("normalize", () => {
    const tf = getTf();

    // no axis
    const a1 = tf.ndarray(new Float32Array([3, 4]), [2]);
    const n1 = tf.normalize(a1);
    approx(n1.data[0], 0.6); approx(n1.data[1], 0.8);
    approx(a1.data[0], 3, "original unchanged");
    log("  normalize [3,4] → [0.6, 0.8]", "line-pass");
    n1.delete(); a1.delete();

    // 2D flattened
    const a2 = tf.ndarray(new Float32Array([1, 2, 2]), [1, 3]);
    const n2 = tf.normalize(a2);
    approx(n2.data[0], 1/3); approx(n2.data[1], 2/3); approx(n2.data[2], 2/3);
    log("  normalize flattened", "line-pass");
    n2.delete(); a2.delete();

    // axis=1
    const a3 = tf.ndarray(new Float32Array([3, 4, 0, 5]), [2, 2]);
    const n3 = tf.normalize(a3, 1);
    approx(n3.data[0], 0.6); approx(n3.data[1], 0.8);
    approx(n3.data[2], 0.0); approx(n3.data[3], 1.0);
    log("  normalize axis=1", "line-pass");
    n3.delete(); a3.delete();

    // axis=0
    const a4 = tf.ndarray(new Float32Array([3, 0, 4, 1]), [2, 2]);
    const n4 = tf.normalize(a4, 0);
    approx(n4.data[0], 0.6); approx(n4.data[1], 0.0);
    approx(n4.data[2], 0.8); approx(n4.data[3], 1.0);
    log("  normalize axis=0", "line-pass");
    n4.delete(); a4.delete();

    // in-place
    const a5 = tf.ndarray(new Float32Array([3, 4]), [2]);
    tf.normalize_(a5);
    approx(a5.data[0], 0.6); approx(a5.data[1], 0.8);
    log("  normalize_ in-place", "line-pass");
    a5.delete();

    // in-place axis=1
    const a6 = tf.ndarray(new Float32Array([3, 4, 0, 5]), [2, 2]);
    tf.normalize_(a6, 1);
    approx(a6.data[0], 0.6); approx(a6.data[1], 0.8);
    approx(a6.data[2], 0.0); approx(a6.data[3], 1.0);
    log("  normalize_ axis=1", "line-pass");
    a6.delete();
  });

  // ==========================================================================
  test("reductions", () => {
    const tf = getTf();

    // flat float32
    const a1 = tf.ndarray(new Float32Array([1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]), [3,3]);
    approx(a1.sum(), 49.5); approx(a1.min(), 1.5); approx(a1.max(), 9.5); approx(a1.mean(), 5.5);
    log("  sum/min/max/mean float32", "line-pass");

    // axis
    const s0 = a1.sum(0);
    approx(s0.data[0], 13.5); approx(s0.data[1], 16.5); approx(s0.data[2], 19.5);
    log("  sum axis=0", "line-pass");
    s0.delete();

    const s1 = a1.sum(1);
    approx(s1.data[0], 7.5); approx(s1.data[1], 16.5); approx(s1.data[2], 25.5);
    log("  sum axis=1", "line-pass");
    s1.delete();

    const m1 = a1.mean(1);
    approx(m1.data[0], 2.5); approx(m1.data[1], 5.5); approx(m1.data[2], 8.5);
    log("  mean axis=1", "line-pass");
    m1.delete();
    a1.delete();

    // flat int32
    const a2 = tf.ndarray(new Int32Array([0,1,2,3,4,5]), [2,3]);
    assert(a2.sum() === 15, "int sum"); assert(a2.min() === 0, "int min"); assert(a2.max() === 5, "int max");
    approx(a2.mean(), 2.5, "int mean");
    log("  sum/min/max/mean int32", "line-pass");
    a2.delete();

    // standalone
    const a3 = tf.ndarray(new Float32Array([1, 2, 3, 4]), [4]);
    approx(tf.sum(a3), 10); approx(tf.min(a3), 1); approx(tf.max(a3), 4); approx(tf.mean(a3), 2.5);
    log("  tf.sum/min/max/mean", "line-pass");
    a3.delete();

    // argmin / argmax
    const a4 = tf.ndarray(new Float32Array([3, 1, 5, 2]), [4]);
    assert(a4.argmin() === 1, "argmin"); assert(a4.argmax() === 2, "argmax");
    assert(tf.argmin(a4) === 1, "tf.argmin"); assert(tf.argmax(a4) === 2, "tf.argmax");
    log("  argmin/argmax", "line-pass");
    a4.delete();

    const a5 = tf.ndarray(new Float32Array([1, 3, 4, 2]), [2, 2]);
    const r5 = a5.argmin(1);
    assert(r5.data[0] === 0 && r5.data[1] === 1, "argmin axis=1");
    log("  argmin axis=1", "line-pass");
    r5.delete(); a5.delete();

    const a6 = tf.ndarray(new Float32Array([1, 5, 4, 2]), [2, 2]);
    const r6 = a6.argmax(0);
    assert(r6.data[0] === 1 && r6.data[1] === 0, "argmax axis=0");
    log("  argmax axis=0", "line-pass");
    r6.delete(); a6.delete();

    // any / all
    const ab1 = tf.ndarray(new Int8Array([0, 0, 1, 0]), [4]).as("bool");
    const ab2 = tf.ndarray(new Int8Array([0, 0, 0, 0]), [4]).as("bool");
    assert(ab1.any() === 1 && ab2.any() === 0, "any");
    log("  any", "line-pass");
    ab2.delete(); ab1.delete();

    const al1 = tf.ndarray(new Int8Array([1, 1, 1]), [3]).as("bool");
    const al2 = tf.ndarray(new Int8Array([1, 0, 1]), [3]).as("bool");
    assert(al1.all() === 1 && al2.all() === 0, "all");
    assert(tf.any(al2) === 1 && tf.all(al2) === 0, "standalone any/all");
    log("  all + standalone", "line-pass");
    al2.delete(); al1.delete();
  });

  // ==========================================================================
  test("reductions (async)", async () => {
    const tf = getTf();
    const floats = tf.ndarray(new Float32Array([1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]), [3,3]);
    const ints = tf.ndarray(new Int32Array([0,1,2,3,4,5]), [2,3]);

    approx(await tf.async.sum(floats), 49.5, "async sum");
    log("  async sum", "line-pass");
    approx(await tf.async.mean(ints), 2.5, "async mean");
    log("  async mean", "line-pass");

    const asum1 = await tf.async.sum(floats, 1);
    approx(asum1.data[0], 7.5); approx(asum1.data[2], 25.5);
    log("  async sum axis=1", "line-pass");
    asum1.delete();

    ints.delete(); floats.delete();
  });

  // ==========================================================================
  test("creation", () => {
    const tf = getTf();

    const z = tf.zeros("float32", [2, 3]);
    assert(z.shape[0] === 2 && z.shape[1] === 3 && z.length === 6, "zeros shape");
    for (let i = 0; i < 6; i++) approx(z.data[i], 0);
    log("  zeros", "line-pass");
    z.delete();

    const o = tf.ones("int32", [4]);
    for (let i = 0; i < 4; i++) assert(o.data[i] === 1);
    log("  ones", "line-pass");
    o.delete();

    const f = tf.full("float32", [3], 7.5);
    for (let i = 0; i < 3; i++) approx(f.data[i], 7.5);
    log("  full", "line-pass");
    f.delete();

    const ar = tf.arange("int32", 0, 5);
    assert(ar.shape[0] === 5);
    for (let i = 0; i < 5; i++) assert(ar.data[i] === i);
    log("  arange", "line-pass");
    ar.delete();

    const ars = tf.arange("float32", 0, 1, 0.25);
    assert(ars.shape[0] === 4);
    approx(ars.data[1], 0.25); approx(ars.data[2], 0.5);
    log("  arange with step", "line-pass");
    ars.delete();

    const ls = tf.linspace(0, 1, 5);
    approx(ls.data[0], 0); approx(ls.data[2], 0.5); approx(ls.data[4], 1);
    log("  linspace", "line-pass");
    ls.delete();

    const a = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const c = a.clone();
    a.mul_(2);
    approx(c.data[0], 1, "clone unaffected");
    approx(a.data[0], 2, "original changed");
    log("  clone", "line-pass");
    c.delete(); a.delete();
  });

  // ==========================================================================
  test("stack & concatenate", () => {
    const tf = getTf();

    // stack axis=0
    const s1a = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const s1b = tf.ndarray(new Float32Array([4, 5, 6]), [3]);
    const s1 = tf.stack([s1a, s1b]);
    assert(s1.ndim === 2 && s1.shape[0] === 2 && s1.shape[1] === 3, "stack shape");
    approx(s1.data[0], 1); approx(s1.data[3], 4);
    log("  stack axis=0", "line-pass");
    s1.delete(); s1b.delete(); s1a.delete();

    // stack axis=1 2D
    const s2a = tf.ndarray(new Float32Array([1,2,3,4,5,6]), [2,3]);
    const s2b = tf.ndarray(new Float32Array([7,8,9,10,11,12]), [2,3]);
    const s2 = tf.stack([s2a, s2b], 1);
    assert(s2.ndim === 3 && s2.shape[0] === 2 && s2.shape[1] === 2 && s2.shape[2] === 3, "stack shape");
    approx(s2.data[0], 1); approx(s2.data[3], 7);
    log("  stack axis=1 (2D)", "line-pass");
    s2.delete(); s2b.delete(); s2a.delete();

    // stack int32
    const s3a = tf.ndarray(new Int32Array([10, 20]), [2]);
    const s3b = tf.ndarray(new Int32Array([30, 40]), [2]);
    const s3 = tf.stack([s3a, s3b]);
    assert(s3.data[0] === 10 && s3.data[2] === 30, "stack int32");
    log("  stack int32", "line-pass");
    s3.delete(); s3b.delete(); s3a.delete();

    // stack 3 arrays
    const s4a = tf.ndarray(new Float32Array([1, 2]), [2]);
    const s4b = tf.ndarray(new Float32Array([3, 4]), [2]);
    const s4c = tf.ndarray(new Float32Array([5, 6]), [2]);
    const s4 = tf.stack([s4a, s4b, s4c]);
    assert(s4.shape[0] === 3 && s4.shape[1] === 2, "stack 3 shape");
    log("  stack 3 arrays", "line-pass");
    s4.delete(); s4c.delete(); s4b.delete(); s4a.delete();

    // stack negative axis
    const s5a = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const s5b = tf.ndarray(new Float32Array([4, 5, 6]), [3]);
    const s5 = tf.stack([s5a, s5b], -1);
    assert(s5.shape[0] === 3 && s5.shape[1] === 2, "stack axis=-1 shape");
    approx(s5.data[0], 1); approx(s5.data[1], 4);
    log("  stack axis=-1", "line-pass");
    s5.delete(); s5b.delete(); s5a.delete();

    // concatenate 1D
    const c1a = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const c1b = tf.ndarray(new Float32Array([4, 5]), [2]);
    const c1 = tf.concatenate([c1a, c1b]);
    assert(c1.shape[0] === 5, "concat shape");
    approx(c1.data[3], 4);
    log("  concatenate 1D", "line-pass");
    c1.delete(); c1b.delete(); c1a.delete();

    // concatenate 2D axis=0
    const c2a = tf.ndarray(new Float32Array([1,2,3,4,5,6]), [2,3]);
    const c2b = tf.ndarray(new Float32Array([7,8,9]), [1,3]);
    const c2 = tf.concatenate([c2a, c2b]);
    assert(c2.shape[0] === 3 && c2.shape[1] === 3, "concat 2D shape");
    approx(c2.data[6], 7);
    log("  concatenate 2D axis=0", "line-pass");
    c2.delete(); c2b.delete(); c2a.delete();

    // concatenate 2D axis=1
    const c3a = tf.ndarray(new Float32Array([1,2,3,4]), [2,2]);
    const c3b = tf.ndarray(new Float32Array([5,6]), [2,1]);
    const c3 = tf.concatenate([c3a, c3b], 1);
    assert(c3.shape[0] === 2 && c3.shape[1] === 3, "concat axis=1 shape");
    approx(c3.data[2], 5); approx(c3.data[5], 6);
    log("  concatenate 2D axis=1", "line-pass");
    c3.delete(); c3b.delete(); c3a.delete();

    // concatenate int32
    const c4a = tf.ndarray(new Int32Array([1, 2]), [2]);
    const c4b = tf.ndarray(new Int32Array([3, 4, 5]), [3]);
    const c4 = tf.concatenate([c4a, c4b]);
    assert(c4.shape[0] === 5 && c4.data[4] === 5, "concat int32");
    log("  concatenate int32", "line-pass");
    c4.delete(); c4b.delete(); c4a.delete();

    // concatenate 3 arrays
    const c5a = tf.ndarray(new Float32Array([1]), [1]);
    const c5b = tf.ndarray(new Float32Array([2, 3]), [2]);
    const c5c = tf.ndarray(new Float32Array([4, 5, 6]), [3]);
    const c5 = tf.concatenate([c5a, c5b, c5c]);
    assert(c5.shape[0] === 6, "concat 3 shape");
    for (let i = 0; i < 6; i++) approx(c5.data[i], i + 1);
    log("  concatenate 3 arrays", "line-pass");
    c5.delete(); c5c.delete(); c5b.delete(); c5a.delete();
  });

  // ==========================================================================
  test("tile", () => {
    const tf = getTf();

    // 1D scalar reps
    const t1 = tf.tile(tf.ndarray(new Float32Array([1, 2, 3]), [3]), 3);
    assert(t1.shape[0] === 9, "tile 1D shape");
    approx(t1.data[0], 1); approx(t1.data[3], 1); approx(t1.data[6], 1);
    log("  tile 1D x3", "line-pass");
    t1.delete();

    // 1D with leading axis
    const t2 = tf.tile(tf.ndarray(new Float32Array([1, 2, 3]), [3]), [4, 1]);
    assert(t2.ndim === 2 && t2.shape[0] === 4 && t2.shape[1] === 3, "tile leading shape");
    for (let r = 0; r < 4; r++) approx(t2.data[r * 3], 1);
    log("  tile [3] x [4,1]", "line-pass");
    t2.delete();

    // 2D axis 0
    const t3 = tf.tile(tf.ndarray(new Float32Array([1,2,3,4]), [2, 2]), [3, 1]);
    assert(t3.shape[0] === 6 && t3.shape[1] === 2, "tile 2D axis 0 shape");
    approx(t3.data[0], 1); approx(t3.data[4], 1);
    log("  tile 2D [3,1]", "line-pass");
    t3.delete();

    // 2D axis 1
    const t4 = tf.tile(tf.ndarray(new Float32Array([1,2,3,4]), [2, 2]), [1, 3]);
    assert(t4.shape[0] === 2 && t4.shape[1] === 6, "tile 2D axis 1 shape");
    approx(t4.data[0], 1); approx(t4.data[2], 1);
    log("  tile 2D [1,3]", "line-pass");
    t4.delete();

    // int32
    const t5 = tf.tile(tf.ndarray(new Int32Array([10, 20]), [2]), 2);
    assert(t5.shape[0] === 4 && t5.data[0] === 10 && t5.data[2] === 10, "tile int32");
    log("  tile int32", "line-pass");
    t5.delete();
  });

  // ==========================================================================
  test("indexing", () => {
    const tf = getTf();

    // take 1D
    const g1 = tf.ndarray(new Float32Array([10, 20, 30, 40, 50]), [5]);
    const gi = tf.ndarray(new Int32Array([4, 1, 0]), [3]);
    const gr = g1.take(gi);
    assert(gr.shape[0] === 3, "take shape");
    approx(gr.data[0], 50); approx(gr.data[1], 20); approx(gr.data[2], 10);
    log("  take 1D", "line-pass");
    gr.delete(); gi.delete(); g1.delete();

    // take 2D
    const g2 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
    const gi2 = tf.ndarray(new Int32Array([2, 0]), [2]);
    const gr2 = g2.take(gi2);
    assert(gr2.shape[0] === 2 && gr2.shape[1] === 2, "take 2D shape");
    approx(gr2.data[0], 5); approx(gr2.data[2], 1);
    log("  take 2D", "line-pass");
    gr2.delete(); gi2.delete(); g2.delete();

    // booleanIndex
    const b1 = tf.ndarray(new Float32Array([10, 20, 30, 40]), [4]);
    const bm = tf.ndarray(new Int8Array([1, 0, 1, 0]), [4]).as("bool");
    const br = b1.booleanIndex(bm);
    assert(br.shape[0] === 2, "booleanIndex shape");
    approx(br.data[0], 10); approx(br.data[1], 30);
    log("  booleanIndex", "line-pass");
    br.delete(); bm.delete(); b1.delete();

    // where
    const wc = tf.ndarray(new Int8Array([1, 0, 1, 0]), [4]).as("bool");
    const wx = tf.ndarray(new Float32Array([1, 2, 3, 4]), [4]);
    const wy = tf.ndarray(new Float32Array([10, 20, 30, 40]), [4]);
    const wr = tf.where(wc, wx, wy);
    approx(wr.data[0], 1); approx(wr.data[1], 20); approx(wr.data[2], 3); approx(wr.data[3], 40);
    log("  where", "line-pass");
    wr.delete(); wy.delete(); wx.delete(); wc.delete();

    // take with axis — 2D axis=1: select columns
    const ta1 = tf.ndarray(new Float32Array([1,2,3, 4,5,6]), [2, 3]);
    const tai1 = tf.ndarray(new Int32Array([2, 0]), [2]);
    const tar1 = ta1.take(tai1, 1);
    assert(tar1.shape[0] === 2 && tar1.shape[1] === 2, "take axis=1 shape");
    approx(tar1.data[0], 3); approx(tar1.data[1], 1);  // row 0: col2, col0
    approx(tar1.data[2], 6); approx(tar1.data[3], 4);  // row 1: col2, col0
    log("  take axis=1", "line-pass");
    tar1.delete(); tai1.delete(); ta1.delete();

    // take free function
    const ta2 = tf.ndarray(new Float32Array([10, 20, 30, 40, 50]), [5]);
    const tai2 = tf.ndarray(new Int32Array([3, 1]), [2]);
    const tar2 = tf.take(ta2, tai2);
    assert(tar2.shape[0] === 2, "take free fn shape");
    approx(tar2.data[0], 40); approx(tar2.data[1], 20);
    log("  take free function", "line-pass");
    tar2.delete(); tai2.delete(); ta2.delete();

    // takeAlongAxis — select per-row indices along axis=1
    const taa1 = tf.ndarray(new Float32Array([10,20,30, 40,50,60]), [2, 3]);
    const taai = tf.ndarray(new Int32Array([2, 0]), [2, 1]);
    const taar = taa1.takeAlongAxis(taai, 1);
    assert(taar.shape[0] === 2 && taar.shape[1] === 1, "takeAlongAxis shape");
    approx(taar.data[0], 30);  // row 0: index 2
    approx(taar.data[1], 40);  // row 1: index 0
    log("  takeAlongAxis", "line-pass");
    taar.delete(); taai.delete(); taa1.delete();

    // takeAlongAxis free function
    const taa2 = tf.ndarray(new Float32Array([1,2,3, 4,5,6]), [2, 3]);
    const taai2 = tf.ndarray(new Int32Array([1, 2]), [2, 1]);
    const taar2 = tf.takeAlongAxis(taa2, taai2, 1);
    approx(taar2.data[0], 2);  // row 0: index 1
    approx(taar2.data[1], 6);  // row 1: index 2
    log("  takeAlongAxis free function", "line-pass");
    taar2.delete(); taai2.delete(); taa2.delete();
  });

  // ==========================================================================
  test("sort & argsort", () => {
    const tf = getTf();

    // sort 1D
    const s1 = tf.ndarray(new Float32Array([3, 1, 4, 1, 5, 9, 2, 6]), [8]);
    const s1r = tf.sort(s1);
    assert(s1r.shape[0] === 8, "sort 1D shape");
    approx(s1r.data[0], 1); approx(s1r.data[1], 1); approx(s1r.data[2], 2);
    approx(s1r.data[3], 3); approx(s1r.data[7], 9);
    approx(s1.data[0], 3, "original unchanged");
    log("  sort 1D", "line-pass");
    s1r.delete(); s1.delete();

    // sort 2D — lexicographic row sort
    const s2 = tf.ndarray(new Float32Array([
      3, 1,
      1, 2,
      1, 0,
      2, 5,
    ]), [4, 2]);
    const s2r = tf.sort(s2);
    assert(s2r.shape[0] === 4 && s2r.shape[1] === 2, "sort 2D shape");
    // Expected order: [1,0], [1,2], [2,5], [3,1]
    approx(s2r.data[0], 1); approx(s2r.data[1], 0);
    approx(s2r.data[2], 1); approx(s2r.data[3], 2);
    approx(s2r.data[4], 2); approx(s2r.data[5], 5);
    approx(s2r.data[6], 3); approx(s2r.data[7], 1);
    log("  sort 2D lexicographic", "line-pass");
    s2r.delete(); s2.delete();

    // sort_ in-place
    const s3 = tf.ndarray(new Float32Array([5, 2, 8, 1]), [4]);
    tf.sort_(s3);
    approx(s3.data[0], 1); approx(s3.data[1], 2);
    approx(s3.data[2], 5); approx(s3.data[3], 8);
    log("  sort_ in-place", "line-pass");
    s3.delete();

    // sort int32
    const s4 = tf.ndarray(new Int32Array([30, 10, 20]), [3]);
    const s4r = tf.sort(s4);
    assert(s4r.data[0] === 10 && s4r.data[1] === 20 && s4r.data[2] === 30, "sort int32");
    log("  sort int32", "line-pass");
    s4r.delete(); s4.delete();

    // sort method
    const s5 = tf.ndarray(new Float32Array([9, 3, 7]), [3]);
    const s5r = s5.sort();
    approx(s5r.data[0], 3); approx(s5r.data[1], 7); approx(s5r.data[2], 9);
    log("  sort method", "line-pass");
    s5r.delete(); s5.delete();

    // argsort 1D
    const a1 = tf.ndarray(new Float32Array([30, 10, 20]), [3]);
    const a1r = tf.argsort(a1);
    assert(a1r.shape[0] === 3, "argsort shape");
    assert(a1r.data[0] === 1 && a1r.data[1] === 2 && a1r.data[2] === 0, "argsort values");
    log("  argsort 1D", "line-pass");
    a1r.delete(); a1.delete();

    // argsort 2D — lexicographic row permutation
    const a2 = tf.ndarray(new Float32Array([
      3, 1,
      1, 2,
      1, 0,
    ]), [3, 2]);
    const a2r = tf.argsort(a2);
    assert(a2r.shape[0] === 3, "argsort 2D shape");
    // Expected: row 2 [1,0] < row 1 [1,2] < row 0 [3,1] → perm [2, 1, 0]
    assert(a2r.data[0] === 2 && a2r.data[1] === 1 && a2r.data[2] === 0, "argsort 2D values");
    log("  argsort 2D lexicographic", "line-pass");
    a2r.delete(); a2.delete();

    // argsort method
    const a3 = tf.ndarray(new Float32Array([5, 1, 3]), [3]);
    const a3r = a3.argsort();
    assert(a3r.data[0] === 1 && a3r.data[1] === 2 && a3r.data[2] === 0, "argsort method");
    log("  argsort method", "line-pass");
    a3r.delete(); a3.delete();

    // argsort + take roundtrip = sort
    const rt = tf.ndarray(new Float32Array([5, 2, 8, 1, 3]), [5]);
    const perm = tf.argsort(rt);
    const sorted = tf.take(rt, perm);
    approx(sorted.data[0], 1); approx(sorted.data[1], 2); approx(sorted.data[2], 3);
    approx(sorted.data[3], 5); approx(sorted.data[4], 8);
    log("  argsort + take = sort", "line-pass");
    sorted.delete(); perm.delete(); rt.delete();
  });

  // ==========================================================================
  test("set operations", () => {
    const tf = getTf();

    // unique 1D
    const u1 = tf.ndarray(new Float32Array([1, 1, 2, 3, 3, 3, 5]), [7]);
    const u1r = tf.unique(u1);
    assert(u1r.shape[0] === 4, "unique 1D shape");
    approx(u1r.data[0], 1); approx(u1r.data[1], 2);
    approx(u1r.data[2], 3); approx(u1r.data[3], 5);
    log("  unique 1D", "line-pass");
    u1r.delete(); u1.delete();

    // unique 2D — duplicate rows
    const u2 = tf.ndarray(new Float32Array([
      1, 2,
      1, 2,
      3, 4,
      3, 4,
      5, 6,
    ]), [5, 2]);
    const u2r = tf.unique(u2);
    assert(u2r.shape[0] === 3 && u2r.shape[1] === 2, "unique 2D shape");
    approx(u2r.data[0], 1); approx(u2r.data[1], 2);
    approx(u2r.data[2], 3); approx(u2r.data[3], 4);
    approx(u2r.data[4], 5); approx(u2r.data[5], 6);
    log("  unique 2D", "line-pass");
    u2r.delete(); u2.delete();

    // unique int32
    const u3 = tf.ndarray(new Int32Array([5, 5, 10, 10, 10, 20]), [6]);
    const u3r = tf.unique(u3);
    assert(u3r.shape[0] === 3, "unique int32 shape");
    assert(u3r.data[0] === 5 && u3r.data[1] === 10 && u3r.data[2] === 20, "unique int32 values");
    log("  unique int32", "line-pass");
    u3r.delete(); u3.delete();

    // unique — all same
    const u4 = tf.ndarray(new Float32Array([7, 7, 7]), [3]);
    const u4r = tf.unique(u4);
    assert(u4r.shape[0] === 1, "unique all same");
    approx(u4r.data[0], 7);
    log("  unique all same", "line-pass");
    u4r.delete(); u4.delete();

    // unique — already unique
    const u5 = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const u5r = tf.unique(u5);
    assert(u5r.shape[0] === 3, "unique already unique");
    log("  unique already unique", "line-pass");
    u5r.delete(); u5.delete();

    // setUnion 1D
    const su1a = tf.ndarray(new Float32Array([1, 3, 5, 7]), [4]);
    const su1b = tf.ndarray(new Float32Array([2, 3, 6, 7]), [4]);
    const su1r = tf.setUnion(su1a, su1b);
    assert(su1r.shape[0] === 6, "setUnion shape");
    approx(su1r.data[0], 1); approx(su1r.data[1], 2); approx(su1r.data[2], 3);
    approx(su1r.data[3], 5); approx(su1r.data[4], 6); approx(su1r.data[5], 7);
    log("  setUnion 1D", "line-pass");
    su1r.delete(); su1b.delete(); su1a.delete();

    // setUnion — disjoint
    const su2a = tf.ndarray(new Float32Array([1, 2]), [2]);
    const su2b = tf.ndarray(new Float32Array([3, 4]), [2]);
    const su2r = tf.setUnion(su2a, su2b);
    assert(su2r.shape[0] === 4, "setUnion disjoint shape");
    approx(su2r.data[0], 1); approx(su2r.data[3], 4);
    log("  setUnion disjoint", "line-pass");
    su2r.delete(); su2b.delete(); su2a.delete();

    // setUnion int32
    const su3a = tf.ndarray(new Int32Array([1, 3]), [2]);
    const su3b = tf.ndarray(new Int32Array([2, 3]), [2]);
    const su3r = tf.setUnion(su3a, su3b);
    assert(su3r.shape[0] === 3, "setUnion int32 shape");
    assert(su3r.data[0] === 1 && su3r.data[1] === 2 && su3r.data[2] === 3, "setUnion int32");
    log("  setUnion int32", "line-pass");
    su3r.delete(); su3b.delete(); su3a.delete();

    // setIntersection 1D
    const si1a = tf.ndarray(new Float32Array([1, 2, 3, 5, 7]), [5]);
    const si1b = tf.ndarray(new Float32Array([2, 3, 6, 7, 8]), [5]);
    const si1r = tf.setIntersection(si1a, si1b);
    assert(si1r.shape[0] === 3, "setIntersection shape");
    approx(si1r.data[0], 2); approx(si1r.data[1], 3); approx(si1r.data[2], 7);
    log("  setIntersection 1D", "line-pass");
    si1r.delete(); si1b.delete(); si1a.delete();

    // setIntersection — empty result
    const si2a = tf.ndarray(new Float32Array([1, 2]), [2]);
    const si2b = tf.ndarray(new Float32Array([3, 4]), [2]);
    const si2r = tf.setIntersection(si2a, si2b);
    assert(si2r.shape[0] === 0, "setIntersection empty");
    log("  setIntersection empty", "line-pass");
    si2r.delete(); si2b.delete(); si2a.delete();

    // setIntersection — identical
    const si3a = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const si3b = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const si3r = tf.setIntersection(si3a, si3b);
    assert(si3r.shape[0] === 3, "setIntersection identical");
    log("  setIntersection identical", "line-pass");
    si3r.delete(); si3b.delete(); si3a.delete();

    // setDifference 1D
    const sd1a = tf.ndarray(new Float32Array([1, 2, 3, 5, 7]), [5]);
    const sd1b = tf.ndarray(new Float32Array([2, 3, 6]), [3]);
    const sd1r = tf.setDifference(sd1a, sd1b);
    assert(sd1r.shape[0] === 3, "setDifference shape");
    approx(sd1r.data[0], 1); approx(sd1r.data[1], 5); approx(sd1r.data[2], 7);
    log("  setDifference 1D", "line-pass");
    sd1r.delete(); sd1b.delete(); sd1a.delete();

    // setDifference — nothing removed
    const sd2a = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const sd2b = tf.ndarray(new Float32Array([4, 5]), [2]);
    const sd2r = tf.setDifference(sd2a, sd2b);
    assert(sd2r.shape[0] === 3, "setDifference nothing removed");
    log("  setDifference nothing removed", "line-pass");
    sd2r.delete(); sd2b.delete(); sd2a.delete();

    // setDifference — all removed
    const sd3a = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const sd3b = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const sd3r = tf.setDifference(sd3a, sd3b);
    assert(sd3r.shape[0] === 0, "setDifference all removed");
    log("  setDifference all removed", "line-pass");
    sd3r.delete(); sd3b.delete(); sd3a.delete();

    // setDifference int32
    const sd4a = tf.ndarray(new Int32Array([10, 20, 30, 40]), [4]);
    const sd4b = tf.ndarray(new Int32Array([20, 40]), [2]);
    const sd4r = tf.setDifference(sd4a, sd4b);
    assert(sd4r.shape[0] === 2, "setDifference int32 shape");
    assert(sd4r.data[0] === 10 && sd4r.data[1] === 30, "setDifference int32");
    log("  setDifference int32", "line-pass");
    sd4r.delete(); sd4b.delete(); sd4a.delete();
  });

  // ==========================================================================
  test("reshape & shape ops", () => {
    const tf = getTf();

    // reshape method
    const r1 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [6]);
    const r1r = r1.reshape([2, 3]);
    assert(r1r.ndim === 2 && r1r.shape[0] === 2 && r1r.shape[1] === 3, "reshape");
    log("  reshape [6] → [2,3]", "line-pass");
    r1.delete();

    // shape setter
    const r2 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [6]);
    r2.shape = [2, 3];
    assert(r2.ndim === 2 && r2.shape[0] === 2 && r2.shape[1] === 3, "shape setter");
    r2.shape = [3, 2];
    assert(r2.shape[0] === 3 && r2.shape[1] === 2, "shape setter 2");
    r2.shape = [6];
    assert(r2.ndim === 1, "shape setter 3");
    log("  shape setter", "line-pass");
    r2.delete();

    // shape setter — size mismatch
    const r3 = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    let threw = false;
    try { r3.shape = [2, 2]; } catch (e) { threw = true; }
    assert(threw, "reshape mismatch throws");
    log("  reshape mismatch throws", "line-pass");
    r3.delete();

    // flatten
    const f1 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const f1r = f1.flatten();
    assert(f1r.ndim === 1 && f1r.shape[0] === 6, "flatten");
    log("  flatten", "line-pass");
    f1.delete();

    // squeeze
    const sq1 = tf.ndarray(new Float32Array([1, 2, 3]), [1, 3, 1]);
    const sq1r = sq1.squeeze();
    assert(sq1r.ndim === 1 && sq1r.shape[0] === 3, "squeeze");
    log("  squeeze", "line-pass");
    sq1.delete();

    // squeeze specific axis
    const sq2 = tf.ndarray(new Float32Array([1, 2, 3]), [1, 3, 1]);
    const sq2r = sq2.squeeze(0);
    assert(sq2r.ndim === 2 && sq2r.shape[0] === 3 && sq2r.shape[1] === 1, "squeeze(0)");
    log("  squeeze(0)", "line-pass");
    sq2.delete();

    // unsqueeze
    const u1 = tf.ndarray(new Float32Array([1, 2, 3]), [3]);
    const u1r = u1.unsqueeze(0);
    assert(u1r.ndim === 2 && u1r.shape[0] === 1 && u1r.shape[1] === 3, "unsqueeze");
    log("  unsqueeze(0)", "line-pass");
    u1.delete();

    // transpose .T
    const t1 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const t1r = t1.T;
    assert(t1r.shape[0] === 3 && t1r.shape[1] === 2, "transpose shape");
    approx(t1r.data[0], 1); approx(t1r.data[1], 4);
    approx(t1r.data[2], 2); approx(t1r.data[3], 5);
    log("  .T transpose", "line-pass");
    t1r.delete(); t1.delete();

    // transpose with axes
    const t2 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2]);
    const t2r = t2.transpose([2, 0, 1]);
    approx(t2r.data[0], 1); approx(t2r.data[1], 3); approx(t2r.data[4], 2); approx(t2r.data[5], 4);
    log("  transpose with axes", "line-pass");
    t2r.delete(); t2.delete();
  });

  // ==========================================================================
  test("assign", () => {
    const tf = getTf();

    // scalar fill
    const a1 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    a1.assign(0);
    for (let i = 0; i < 6; i++) approx(a1.data[i], 0);
    log("  assign scalar", "line-pass");
    a1.delete();

    // array copy (same shape)
    const a2 = tf.ndarray(new Float32Array([0, 0, 0, 0, 0, 0]), [2, 3]);
    const a2v = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    a2.assign(a2v);
    approx(a2.data[0], 1); approx(a2.data[5], 6);
    log("  assign array (same shape)", "line-pass");
    a2v.delete(); a2.delete();

    // array broadcast [3] → [2,3]
    const a3 = tf.ndarray(new Float32Array([0, 0, 0, 0, 0, 0]), [2, 3]);
    const a3v = tf.ndarray(new Float32Array([10, 20, 30]), [3]);
    a3.assign(a3v);
    approx(a3.data[0], 10); approx(a3.data[1], 20); approx(a3.data[2], 30);
    approx(a3.data[3], 10); approx(a3.data[4], 20); approx(a3.data[5], 30);
    log("  assign array broadcast [3]→[2,3]", "line-pass");
    a3v.delete(); a3.delete();

    // indexed scalar
    const a4 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
    const a4i = tf.ndarray(new Int32Array([0, 2]), [2]);
    a4.assign(a4i, 99);
    approx(a4.data[0], 99); approx(a4.data[1], 99);  // row 0
    approx(a4.data[2], 3);  approx(a4.data[3], 4);    // row 1 unchanged
    approx(a4.data[4], 99); approx(a4.data[5], 99);  // row 2
    log("  assign indexed scalar", "line-pass");
    a4i.delete(); a4.delete();

    // indexed array (exact match)
    const a5 = tf.ndarray(new Float32Array([0, 0, 0, 0, 0, 0]), [3, 2]);
    const a5i = tf.ndarray(new Int32Array([1, 2]), [2]);
    const a5v = tf.ndarray(new Float32Array([10, 20, 30, 40]), [2, 2]);
    a5.assign(a5i, a5v);
    approx(a5.data[0], 0);  approx(a5.data[1], 0);   // row 0 unchanged
    approx(a5.data[2], 10); approx(a5.data[3], 20);  // row 1
    approx(a5.data[4], 30); approx(a5.data[5], 40);  // row 2
    log("  assign indexed array", "line-pass");
    a5v.delete(); a5i.delete(); a5.delete();

    // indexed array broadcast [2] → [3,2] (assigns same row to 3 indices)
    const a5b = tf.ndarray(new Float32Array([0, 0, 0, 0, 0, 0]), [3, 2]);
    const a5bi = tf.ndarray(new Int32Array([0, 1, 2]), [3]);
    const a5bv = tf.ndarray(new Float32Array([7, 8]), [2]);
    a5b.assign(a5bi, a5bv);
    approx(a5b.data[0], 7); approx(a5b.data[1], 8);
    approx(a5b.data[2], 7); approx(a5b.data[3], 8);
    approx(a5b.data[4], 7); approx(a5b.data[5], 8);
    log("  assign indexed array broadcast [2]→[3,2]", "line-pass");
    a5bv.delete(); a5bi.delete(); a5b.delete();

    // masked scalar
    const a6 = tf.ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
    const a6m = tf.ndarray(new Int8Array([1, 0, 1]), [3]).as("bool");
    a6.assign(a6m, 0);
    approx(a6.data[0], 0); approx(a6.data[1], 0);   // row 0 masked
    approx(a6.data[2], 3); approx(a6.data[3], 4);    // row 1 unchanged
    approx(a6.data[4], 0); approx(a6.data[5], 0);   // row 2 masked
    log("  assign masked scalar", "line-pass");
    a6m.delete(); a6.delete();

    // masked array (exact match)
    const a7 = tf.ndarray(new Float32Array([0, 0, 0, 0, 0, 0]), [3, 2]);
    const a7m = tf.ndarray(new Int8Array([0, 1, 1]), [3]).as("bool");
    const a7v = tf.ndarray(new Float32Array([10, 20, 30, 40]), [2, 2]);
    a7.assign(a7m, a7v);
    approx(a7.data[0], 0);  approx(a7.data[1], 0);   // row 0 unchanged
    approx(a7.data[2], 10); approx(a7.data[3], 20);  // row 1
    approx(a7.data[4], 30); approx(a7.data[5], 40);  // row 2
    log("  assign masked array", "line-pass");
    a7v.delete(); a7m.delete(); a7.delete();

    // masked array broadcast [2] → [2,2] (same row to all masked)
    const a7b = tf.ndarray(new Float32Array([0, 0, 0, 0, 0, 0]), [3, 2]);
    const a7bm = tf.ndarray(new Int8Array([1, 0, 1]), [3]).as("bool");
    const a7bv = tf.ndarray(new Float32Array([5, 6]), [2]);
    a7b.assign(a7bm, a7bv);
    approx(a7b.data[0], 5); approx(a7b.data[1], 6);  // row 0
    approx(a7b.data[2], 0); approx(a7b.data[3], 0);  // row 1 unchanged
    approx(a7b.data[4], 5); approx(a7b.data[5], 6);  // row 2
    log("  assign masked array broadcast [2]→[2,2]", "line-pass");
    a7bv.delete(); a7bm.delete(); a7b.delete();

    // int32
    const a8 = tf.ndarray(new Int32Array([1, 2, 3, 4]), [4]);
    a8.assign(42);
    assert(a8.data[0] === 42 && a8.data[3] === 42, "int32 assign scalar");
    log("  assign int32", "line-pass");
    a8.delete();
  });

  // ==========================================================================
  test("eye", () => {
    const tf = getTf();

    // 3x3 float32
    const e1 = tf.eye("float32", 3);
    assert(e1.shape[0] === 3 && e1.shape[1] === 3, "eye shape");
    approx(e1.data[0], 1); approx(e1.data[1], 0); approx(e1.data[2], 0);
    approx(e1.data[3], 0); approx(e1.data[4], 1); approx(e1.data[5], 0);
    approx(e1.data[6], 0); approx(e1.data[7], 0); approx(e1.data[8], 1);
    log("  eye float32 3x3", "line-pass");
    e1.delete();

    // 2x2 int32
    const e2 = tf.eye("int32", 2);
    assert(e2.shape[0] === 2 && e2.shape[1] === 2, "eye int32 shape");
    assert(e2.data[0] === 1 && e2.data[1] === 0 && e2.data[2] === 0 && e2.data[3] === 1, "eye int32 values");
    log("  eye int32 2x2", "line-pass");
    e2.delete();

    // 1x1
    const e3 = tf.eye("float32", 1);
    assert(e3.shape[0] === 1 && e3.shape[1] === 1, "eye 1x1 shape");
    approx(e3.data[0], 1);
    log("  eye 1x1", "line-pass");
    e3.delete();
  });

  // ==========================================================================
  test("norm", () => {
    const tf = getTf();

    // global norm — [3, 4] → sqrt(9+16) = 5
    const n1 = tf.ndarray(new Float32Array([3, 4]), [2]);
    const n1r = tf.norm(n1);
    approx(n1r, 5);
    log("  norm global [3,4]→5", "line-pass");
    n1.delete();

    // global norm — [1, 2, 2] → sqrt(1+4+4) = 3
    const n2 = tf.ndarray(new Float32Array([1, 2, 2]), [3]);
    const n2r = tf.norm(n2);
    approx(n2r, 3);
    log("  norm global [1,2,2]→3", "line-pass");
    n2.delete();

    // norm along axis 1 — [N,3]: per-row vector magnitude
    const n3 = tf.ndarray(new Float32Array([3, 4, 0, 0, 0, 5, 1, 2, 2]), [3, 3]);
    const n3r = tf.norm(n3, 1);
    assert(n3r.shape[0] === 3, "norm axis shape");
    approx(n3r.data[0], 5);     // sqrt(9+16+0)
    approx(n3r.data[1], 5);     // sqrt(0+0+25)
    approx(n3r.data[2], 3);     // sqrt(1+4+4)
    log("  norm axis=1 per-row", "line-pass");
    n3r.delete(); n3.delete();

    // norm along axis 0 — column norms
    const n4 = tf.ndarray(new Float32Array([1, 0, 0, 1]), [2, 2]);
    const n4r = tf.norm(n4, 0);
    assert(n4r.shape[0] === 2, "norm axis=0 shape");
    approx(n4r.data[0], 1);     // sqrt(1+0)
    approx(n4r.data[1], 1);     // sqrt(0+1)
    log("  norm axis=0 column norms", "line-pass");
    n4r.delete(); n4.delete();

    // norm int32
    const n5 = tf.ndarray(new Int32Array([3, 4]), [2]);
    const n5r = tf.norm(n5);
    approx(n5r, 5);
    log("  norm int32", "line-pass");
    n5.delete();
  });

  // ==========================================================================
  test("atan2", () => {
    const tf = getTf();

    // basic atan2(y, x)
    const y1 = tf.ndarray(new Float32Array([1, 0, -1, 0]), [4]);
    const x1 = tf.ndarray(new Float32Array([0, 1, 0, -1]), [4]);
    const r1 = tf.atan2(y1, x1);
    assert(r1.shape[0] === 4, "atan2 shape");
    approx(r1.data[0], Math.PI / 2);    // atan2(1, 0) = pi/2
    approx(r1.data[1], 0);              // atan2(0, 1) = 0
    approx(r1.data[2], -Math.PI / 2);   // atan2(-1, 0) = -pi/2
    approx(r1.data[3], Math.PI);        // atan2(0, -1) = pi
    log("  atan2 basic", "line-pass");
    r1.delete(); x1.delete(); y1.delete();

    // atan2 broadcasting — [N,1] with [1,M]
    const y2 = tf.ndarray(new Float32Array([1, -1]), [2, 1]);
    const x2 = tf.ndarray(new Float32Array([1, -1]), [1, 2]);
    const r2 = tf.atan2(y2, x2);
    assert(r2.shape[0] === 2 && r2.shape[1] === 2, "atan2 broadcast shape");
    approx(r2.data[0], Math.atan2(1, 1));     // pi/4
    approx(r2.data[1], Math.atan2(1, -1));    // 3pi/4
    approx(r2.data[2], Math.atan2(-1, 1));    // -pi/4
    approx(r2.data[3], Math.atan2(-1, -1));   // -3pi/4
    log("  atan2 broadcast [2,1]×[1,2]→[2,2]", "line-pass");
    r2.delete(); x2.delete(); y2.delete();

    // atan2 same shape — angle of 2D vectors
    const vy = tf.ndarray(new Float32Array([0, 1, 1]), [3]);
    const vx = tf.ndarray(new Float32Array([1, 1, 0]), [3]);
    const angles = tf.atan2(vy, vx);
    approx(angles.data[0], 0);              // (1,0) → 0
    approx(angles.data[1], Math.PI / 4);    // (1,1) → pi/4
    approx(angles.data[2], Math.PI / 2);    // (0,1) → pi/2
    log("  atan2 vector angles", "line-pass");
    angles.delete(); vx.delete(); vy.delete();
  });

});
