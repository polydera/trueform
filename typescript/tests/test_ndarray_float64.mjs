/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */

import { describe, test, log, assert, approx, getTf } from "./harness.mjs";

describe("NDArray float64", () => {

  // ==========================================================================
  test("float64 NDArray from Float64Array", () => {
    const tf = getTf();
    const arr = tf.ndarray(new Float64Array([1.5, 2.5, 3.5]));
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    assert(arr.length === 3, `length=${arr.length}`);
    assert(arr.shape[0] === 3, `shape[0]=${arr.shape[0]}`);
    approx(arr.data[0], 1.5);
    approx(arr.data[1], 2.5);
    approx(arr.data[2], 3.5);
    log("  Float64Array round-trip", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("float64 NDArray from raw number[]", () => {
    const tf = getTf();
    const arr = tf.ndarray([1.5, 2.5, 3.5]);
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    assert(arr.length === 3, `length=${arr.length}`);
    approx(arr.data[0], 1.5);
    approx(arr.data[2], 3.5);
    log("  raw number[] defaults to float64", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("float64 NDArray with explicit shape", () => {
    const tf = getTf();
    const arr = tf.ndarray(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    assert(arr.ndim === 2, `ndim=${arr.ndim}`);
    assert(arr.shape[0] === 2 && arr.shape[1] === 3, `shape=${arr.shape}`);
    assert(arr.length === 6, `length=${arr.length}`);
    approx(arr.data[5], 6);
    log("  2D float64 from Float64Array", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("float64 preserves double-precision values", () => {
    const tf = getTf();
    // Value that loses precision in float32 but survives in float64
    const v = 1.0000000001;
    const arr = tf.ndarray(new Float64Array([v]));
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    // float64 should preserve this exactly; float32 cannot
    assert(arr.data[0] === v, `expected ${v}, got ${arr.data[0]}`);
    log("  double-precision preserved", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("as float64 converts from float32", () => {
    const tf = getTf();
    const src = tf.ndarray(new Float32Array([1.5, 2.5, 3.5]));
    const arr = src.as("float64");
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    // 1.5 and 2.5 are exactly representable in float32, so the cast must
    // preserve them bit-for-bit when widened to float64.
    assert(arr.data[0] === 1.5, `data[0]=${arr.data[0]}`);
    assert(arr.data[1] === 2.5, `data[1]=${arr.data[1]}`);
    log("  float32 -> float64 cast", "line-pass");
    arr.delete();
    src.delete();
  });

  // ==========================================================================
  test("as float32 converts from float64 (downcast)", () => {
    const tf = getTf();
    const src = tf.ndarray(new Float64Array([1.5, 2.5, 3.5]));
    const arr = src.as("float32");
    assert(arr.dtype === "float32", `dtype=${arr.dtype}`);
    approx(arr.data[0], 1.5);
    log("  float64 -> float32 downcast", "line-pass");
    arr.delete();
    src.delete();
  });

  // ==========================================================================
  test("as int32 converts from float64", () => {
    const tf = getTf();
    const src = tf.ndarray(new Float64Array([1.5, 2.5, 3.5]));
    const arr = src.as("int32");
    assert(arr.dtype === "int32", `dtype=${arr.dtype}`);
    assert(arr.data[0] === 1, `expected 1, got ${arr.data[0]}`);
    log("  float64 -> int32 cast", "line-pass");
    arr.delete();
    src.delete();
  });

  // ==========================================================================
  test("float64 NDArray destroy is idempotent", () => {
    const tf = getTf();
    const arr = tf.ndarray(new Float64Array([1, 2, 3]));
    arr.delete();
    arr.delete(); // should not throw
    log("  double-delete is safe", "line-pass");
  });

  // ==========================================================================
  test("float64 NDArray row view", () => {
    const tf = getTf();
    const arr = tf.ndarray(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const r0 = arr.row(0);
    assert(r0.dtype === "float64", `row dtype=${r0.dtype}`);
    assert(r0.length === 3, `row length=${r0.length}`);
    approx(r0.data[0], 1);
    approx(r0.data[2], 3);
    const r1 = arr.row(1);
    approx(r1.data[0], 4);
    approx(r1.data[2], 6);
    log("  row view on 2D float64", "line-pass");
    r1.delete();
    r0.delete();
    arr.delete();
  });

  // ==========================================================================
  test("zeros float64", () => {
    const tf = getTf();
    const arr = tf.zeros("float64", [5]);
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    assert(arr.length === 5, `length=${arr.length}`);
    for (let i = 0; i < 5; i++) assert(arr.data[i] === 0, `data[${i}]=${arr.data[i]}`);
    log("  zeros('float64', [5])", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("ones float64", () => {
    const tf = getTf();
    const arr = tf.ones("float64", [4]);
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    assert(arr.length === 4, `length=${arr.length}`);
    for (let i = 0; i < 4; i++) assert(arr.data[i] === 1, `data[${i}]=${arr.data[i]}`);
    log("  ones('float64', [4])", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("full float64", () => {
    const tf = getTf();
    const v = 7.0000000001; // value that loses precision in float32
    const arr = tf.full("float64", [3], v);
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    for (let i = 0; i < 3; i++) assert(arr.data[i] === v, `data[${i}]=${arr.data[i]}`);
    log("  full('float64', [3], v) preserves precision", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("eye float64", () => {
    const tf = getTf();
    const arr = tf.eye("float64", 3);
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    assert(arr.shape[0] === 3 && arr.shape[1] === 3, `shape=${arr.shape}`);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const expected = i === j ? 1 : 0;
        assert(arr.data[i * 3 + j] === expected,
          `data[${i},${j}]=${arr.data[i * 3 + j]} expected ${expected}`);
      }
    }
    log("  eye('float64', 3)", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("arange float64 (stop only)", () => {
    const tf = getTf();
    const arr = tf.arange("float64", 5);
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    assert(arr.length === 5, `length=${arr.length}`);
    for (let i = 0; i < 5; i++) assert(arr.data[i] === i, `data[${i}]=${arr.data[i]}`);
    log("  arange('float64', 5)", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("arange float64 (start, stop, step)", () => {
    const tf = getTf();
    const arr = tf.arange("float64", 1, 10, 2);
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    const expected = [1, 3, 5, 7, 9];
    assert(arr.length === expected.length, `length=${arr.length}`);
    for (let i = 0; i < expected.length; i++)
      assert(arr.data[i] === expected[i], `data[${i}]=${arr.data[i]}`);
    log("  arange('float64', 1, 10, 2)", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("linspace returns float64 by default", () => {
    const tf = getTf();
    const arr = tf.linspace(0, 1, 11);
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    assert(arr.length === 11, `length=${arr.length}`);
    assert(arr.data[0] === 0, `data[0]=${arr.data[0]}`);
    assert(arr.data[10] === 1, `data[10]=${arr.data[10]}`);
    approx(arr.data[5], 0.5, "midpoint", 1e-15);
    log("  linspace(0,1,11) → float64", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("random float64", () => {
    const tf = getTf();
    const arr = tf.random("float64", [4, 5], 0, 1);
    assert(arr.dtype === "float64", `dtype=${arr.dtype}`);
    assert(arr.length === 20, `length=${arr.length}`);
    for (let i = 0; i < 20; i++) {
      const v = arr.data[i];
      assert(v >= 0 && v < 1, `data[${i}]=${v} out of [0,1)`);
    }
    log("  random('float64', [4,5], 0, 1)", "line-pass");
    arr.delete();
  });

  // ==========================================================================
  test("stack float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([1, 2, 3]));
    const b = tf.ndarray(new Float64Array([4, 5, 6]));
    const s = tf.stack([a, b]);
    assert(s.dtype === "float64", `dtype=${s.dtype}`);
    assert(s.ndim === 2 && s.shape[0] === 2 && s.shape[1] === 3, `shape=${s.shape}`);
    assert(s.data[0] === 1 && s.data[3] === 4, "values");
    log("  stack float64", "line-pass");
    s.delete(); b.delete(); a.delete();
  });

  // ==========================================================================
  test("concatenate float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([1, 2]));
    const b = tf.ndarray(new Float64Array([3, 4, 5]));
    const c = tf.concatenate([a, b]);
    assert(c.dtype === "float64", `dtype=${c.dtype}`);
    assert(c.length === 5, `length=${c.length}`);
    const expected = [1, 2, 3, 4, 5];
    for (let i = 0; i < 5; i++)
      assert(c.data[i] === expected[i], `data[${i}]=${c.data[i]}`);
    log("  concatenate float64", "line-pass");
    c.delete(); b.delete(); a.delete();
  });

  // ==========================================================================
  test("tile float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([1, 2, 3]));
    const t = tf.tile(a, 3);
    assert(t.dtype === "float64", `dtype=${t.dtype}`);
    assert(t.length === 9, `length=${t.length}`);
    const expected = [1, 2, 3, 1, 2, 3, 1, 2, 3];
    for (let i = 0; i < 9; i++)
      assert(t.data[i] === expected[i], `data[${i}]=${t.data[i]}`);
    log("  tile float64 x3", "line-pass");
    t.delete(); a.delete();
  });

  // ==========================================================================
  test("where float64", () => {
    const tf = getTf();
    const cond = tf.ndarray(new Int8Array([1, 0, 1, 0])).as("bool");
    const x = tf.ndarray(new Float64Array([1, 2, 3, 4]));
    const y = tf.ndarray(new Float64Array([10, 20, 30, 40]));
    const r = tf.where(cond, x, y);
    assert(r.dtype === "float64", `dtype=${r.dtype}`);
    assert(r.data[0] === 1 && r.data[1] === 20 && r.data[2] === 3 && r.data[3] === 40, "values");
    log("  where float64", "line-pass");
    r.delete(); y.delete(); x.delete(); cond.delete();
  });

  // ==========================================================================
  test("sort float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([3, 1, 4, 1, 5, 9, 2, 6]));
    const s = tf.sort(a);
    assert(s.dtype === "float64", `dtype=${s.dtype}`);
    const expected = [1, 1, 2, 3, 4, 5, 6, 9];
    for (let i = 0; i < 8; i++)
      assert(s.data[i] === expected[i], `data[${i}]=${s.data[i]}`);
    assert(a.data[0] === 3, "original unchanged");
    log("  sort float64", "line-pass");
    s.delete(); a.delete();
  });

  // ==========================================================================
  test("sort_ in-place float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([5, 2, 8, 1]));
    tf.sort_(a);
    assert(a.dtype === "float64", `dtype=${a.dtype}`);
    const expected = [1, 2, 5, 8];
    for (let i = 0; i < 4; i++)
      assert(a.data[i] === expected[i], `data[${i}]=${a.data[i]}`);
    log("  sort_ float64", "line-pass");
    a.delete();
  });

  // ==========================================================================
  test("argsort float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([30, 10, 20]));
    const p = tf.argsort(a);
    assert(p.dtype === "int32", `dtype=${p.dtype}`);
    assert(p.data[0] === 1 && p.data[1] === 2 && p.data[2] === 0, "permutation");
    log("  argsort float64", "line-pass");
    p.delete(); a.delete();
  });

  // ==========================================================================
  test("unique float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([1, 1, 2, 3, 3, 3, 5]));
    const u = tf.unique(a);
    assert(u.dtype === "float64", `dtype=${u.dtype}`);
    assert(u.length === 4, `length=${u.length}`);
    const expected = [1, 2, 3, 5];
    for (let i = 0; i < 4; i++)
      assert(u.data[i] === expected[i], `data[${i}]=${u.data[i]}`);
    log("  unique float64", "line-pass");
    u.delete(); a.delete();
  });

  // ==========================================================================
  test("setUnion float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([1, 3, 5, 7]));
    const b = tf.ndarray(new Float64Array([2, 3, 6, 7]));
    const r = tf.setUnion(a, b);
    assert(r.dtype === "float64", `dtype=${r.dtype}`);
    const expected = [1, 2, 3, 5, 6, 7];
    assert(r.length === expected.length, `length=${r.length}`);
    for (let i = 0; i < expected.length; i++)
      assert(r.data[i] === expected[i], `data[${i}]=${r.data[i]}`);
    log("  setUnion float64", "line-pass");
    r.delete(); b.delete(); a.delete();
  });

  // ==========================================================================
  test("setIntersection float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([1, 2, 3, 5, 7]));
    const b = tf.ndarray(new Float64Array([2, 3, 6, 7, 8]));
    const r = tf.setIntersection(a, b);
    assert(r.dtype === "float64", `dtype=${r.dtype}`);
    const expected = [2, 3, 7];
    assert(r.length === expected.length, `length=${r.length}`);
    for (let i = 0; i < expected.length; i++)
      assert(r.data[i] === expected[i], `data[${i}]=${r.data[i]}`);
    log("  setIntersection float64", "line-pass");
    r.delete(); b.delete(); a.delete();
  });

  // ==========================================================================
  test("setDifference float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([1, 2, 3, 5, 7]));
    const b = tf.ndarray(new Float64Array([2, 3, 6]));
    const r = tf.setDifference(a, b);
    assert(r.dtype === "float64", `dtype=${r.dtype}`);
    const expected = [1, 5, 7];
    assert(r.length === expected.length, `length=${r.length}`);
    for (let i = 0; i < expected.length; i++)
      assert(r.data[i] === expected[i], `data[${i}]=${r.data[i]}`);
    log("  setDifference float64", "line-pass");
    r.delete(); b.delete(); a.delete();
  });

  // ==========================================================================
  test("take float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([10, 20, 30, 40, 50]));
    const i = tf.ndarray(new Int32Array([4, 1, 0]));
    const r = tf.take(a, i);
    assert(r.dtype === "float64", `dtype=${r.dtype}`);
    assert(r.data[0] === 50 && r.data[1] === 20 && r.data[2] === 10, "values");
    log("  take float64", "line-pass");
    r.delete(); i.delete(); a.delete();
  });

  // ==========================================================================
  test("takeAlongAxis float64", () => {
    const tf = getTf();
    const a = tf.ndarray(new Float64Array([10, 20, 30, 40, 50, 60]), [2, 3]);
    const idx = tf.ndarray(new Int32Array([2, 0]), [2, 1]);
    const r = tf.takeAlongAxis(a, idx, 1);
    assert(r.dtype === "float64", `dtype=${r.dtype}`);
    assert(r.data[0] === 30 && r.data[1] === 40, "values");
    log("  takeAlongAxis float64", "line-pass");
    r.delete(); idx.delete(); a.delete();
  });

});

// ============================================================================
// Numeric-op mirror of test_ndarray.mjs for the float64 dtype.
// Every arithmetic / reduction / math / comparison / assign test that exercises
// a `_float32` binding gets a matching `_float64` driver here. This is the
// regression net that would have caught `add_inplace_float64` missing — and
// will catch the same class of bug for any other dtype-symmetric op.
// ============================================================================

describe("NDArray numeric ops (float64)", () => {

  // ==========================================================================
  test("arithmetic (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    // add — same shape
    const a1 = tf.ndarray(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const b1 = tf.ndarray(new Float64Array([10, 20, 30, 40, 50, 60]), [2, 3]);
    const c1 = a1.add(b1);
    assert(c1.dtype === "float64", `dtype=${c1.dtype}`);
    approx(c1.data[0], 11, "add same shape", EPS);
    approx(c1.data[3], 44, "add same shape", EPS);
    approx(c1.data[5], 66, "add same shape", EPS);
    log("  add same shape (f64)", "line-pass");
    c1.delete(); b1.delete(); a1.delete();

    // add — broadcasting [N,3] + [3]
    const a2 = tf.ndarray(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const b2 = tf.ndarray(new Float64Array([10, 20, 30]), [3]);
    const c2 = a2.add(b2);
    assert(c2.dtype === "float64", `dtype=${c2.dtype}`);
    approx(c2.data[0], 11, "broadcast", EPS); approx(c2.data[3], 14, "broadcast", EPS);
    assert(c2.shape[0] === 2 && c2.shape[1] === 3, "broadcast shape");
    log("  add broadcast [2,3]+[3] (f64)", "line-pass");
    c2.delete(); b2.delete(); a2.delete();

    // add — scalar
    const a3 = tf.ndarray(new Float64Array([1, 2, 3]), [3]);
    const c3 = a3.add(10);
    assert(c3.dtype === "float64", `dtype=${c3.dtype}`);
    approx(c3.data[0], 11, "add scalar", EPS); approx(c3.data[2], 13, "add scalar", EPS);
    log("  add scalar (f64)", "line-pass");
    c3.delete(); a3.delete();

    // sub
    const a4 = tf.ndarray(new Float64Array([10, 20, 30]), [3]);
    const b4 = tf.ndarray(new Float64Array([1, 2, 3]), [3]);
    const c4 = a4.sub(b4);
    assert(c4.dtype === "float64", `dtype=${c4.dtype}`);
    approx(c4.data[0], 9, "sub", EPS); approx(c4.data[2], 27, "sub", EPS);
    log("  sub (f64)", "line-pass");
    c4.delete(); b4.delete(); a4.delete();

    // sub — scalar
    const a5 = tf.ndarray(new Float64Array([10, 20, 30]), [3]);
    const c5 = a5.sub(5);
    approx(c5.data[0], 5, "sub scalar", EPS); approx(c5.data[2], 25, "sub scalar", EPS);
    log("  sub scalar (f64)", "line-pass");
    c5.delete(); a5.delete();

    // mul
    const a6 = tf.ndarray(new Float64Array([2, 3, 4]), [3]);
    const b6 = tf.ndarray(new Float64Array([5, 6, 7]), [3]);
    const c6 = a6.mul(b6);
    assert(c6.dtype === "float64", `dtype=${c6.dtype}`);
    approx(c6.data[0], 10, "mul", EPS); approx(c6.data[2], 28, "mul", EPS);
    log("  mul element-wise (f64)", "line-pass");
    c6.delete(); b6.delete(); a6.delete();

    // mul — scalar
    const a7 = tf.ndarray(new Float64Array([1, 2, 3, 4]), [2, 2]);
    const c7 = a7.mul(3);
    approx(c7.data[0], 3, "mul scalar", EPS); approx(c7.data[3], 12, "mul scalar", EPS);
    log("  mul scalar (f64)", "line-pass");
    c7.delete(); a7.delete();

    // div
    const a8 = tf.ndarray(new Float64Array([6, 8, 10]), [3]);
    const b8 = tf.ndarray(new Float64Array([2, 4, 5]), [3]);
    const c8 = a8.div(b8);
    approx(c8.data[0], 3, "div", EPS); approx(c8.data[1], 2, "div", EPS); approx(c8.data[2], 2, "div", EPS);
    log("  div (f64)", "line-pass");
    c8.delete(); b8.delete(); a8.delete();

    // div — scalar
    const a9 = tf.ndarray(new Float64Array([10, 20, 30]), [3]);
    const c9 = a9.div(5);
    approx(c9.data[0], 2, "div scalar", EPS); approx(c9.data[2], 6, "div scalar", EPS);
    log("  div scalar (f64)", "line-pass");
    c9.delete(); a9.delete();

    // div — broadcast
    const a10 = tf.ndarray(new Float64Array([2, 4, 6, 8]), [2, 2]);
    const b10 = tf.ndarray(new Float64Array([2, 1]), [1, 2]);
    const c10 = a10.div(b10);
    approx(c10.data[0], 1, "div bc", EPS); approx(c10.data[1], 4, "div bc", EPS);
    approx(c10.data[2], 3, "div bc", EPS); approx(c10.data[3], 8, "div bc", EPS);
    log("  div broadcast (f64)", "line-pass");
    c10.delete(); b10.delete(); a10.delete();

    // double-precision values survive arithmetic (regression for the
    // float-precision-leak class of bug)
    const dpA = tf.ndarray(new Float64Array([1.0000000001]));
    const dpB = tf.ndarray(new Float64Array([2.0000000002]));
    const dpC = dpA.add(dpB);
    approx(dpC.data[0], 3.0000000003, "double precision add", 1e-15);
    log("  add preserves double precision", "line-pass");
    dpC.delete(); dpB.delete(); dpA.delete();

    // in-place: add_, mul_, div_, chaining (this is what `add_inplace_float64`
    // missing would have broken — explicitly drive every in-place op)
    const ip1 = tf.ndarray(new Float64Array([1, 2, 3]), [3]);
    const ip2 = tf.ndarray(new Float64Array([10, 20, 30]), [3]);
    ip1.add_(ip2);
    assert(ip1.dtype === "float64", "in-place keeps dtype");
    approx(ip1.data[0], 11, "add_ buf", EPS); approx(ip1.data[2], 33, "add_ buf", EPS);
    log("  add_ buffer (f64)", "line-pass");
    ip2.delete(); ip1.delete();

    const ip3 = tf.ndarray(new Float64Array([1, 2, 3]), [3]);
    ip3.add_(100);
    approx(ip3.data[0], 101, "add_ scalar", EPS);
    log("  add_ scalar (f64)", "line-pass");
    ip3.delete();

    const ip4 = tf.ndarray(new Float64Array([2, 4, 6]), [3]);
    ip4.mul_(0.5);
    approx(ip4.data[0], 1, "mul_ scalar", EPS); approx(ip4.data[2], 3, "mul_ scalar", EPS);
    log("  mul_ scalar (f64)", "line-pass");
    ip4.delete();

    const ip5 = tf.ndarray(new Float64Array([12, 9, 6]), [3]);
    ip5.div_(3);
    approx(ip5.data[0], 4, "div_ scalar", EPS); approx(ip5.data[2], 2, "div_ scalar", EPS);
    log("  div_ scalar (f64)", "line-pass");
    ip5.delete();

    const ip6 = tf.ndarray(new Float64Array([1, 2, 3]), [3]);
    ip6.add_(10).mul_(2);
    approx(ip6.data[0], 22, "chain", EPS); approx(ip6.data[2], 26, "chain", EPS);
    log("  chaining add_.mul_ (f64)", "line-pass");
    ip6.delete();

    // sub_ in-place (scalar + buffer)
    const ip7 = tf.ndarray(new Float64Array([10, 20, 30]), [3]);
    ip7.sub_(5);
    approx(ip7.data[0], 5, "sub_ scalar", EPS); approx(ip7.data[2], 25, "sub_ scalar", EPS);
    const ip7b = tf.ndarray(new Float64Array([1, 2, 3]), [3]);
    ip7.sub_(ip7b);
    approx(ip7.data[0], 4, "sub_ buf", EPS); approx(ip7.data[2], 22, "sub_ buf", EPS);
    log("  sub_ in-place (f64)", "line-pass");
    ip7b.delete(); ip7.delete();
  });

  // ==========================================================================
  test("unary math (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    // neg
    const a1 = tf.ndarray(new Float64Array([1, -2, 3]), [3]);
    const b1 = tf.neg(a1);
    assert(b1.dtype === "float64", `dtype=${b1.dtype}`);
    approx(b1.data[0], -1, "neg", EPS); approx(b1.data[1], 2, "neg", EPS); approx(b1.data[2], -3, "neg", EPS);
    log("  neg (f64)", "line-pass");
    b1.delete(); a1.delete();

    // abs
    const a3 = tf.ndarray(new Float64Array([-1, 2, -3]), [3]);
    const b3 = tf.abs(a3);
    assert(b3.dtype === "float64", `dtype=${b3.dtype}`);
    approx(b3.data[0], 1, "abs", EPS); approx(b3.data[2], 3, "abs", EPS);
    log("  abs (f64)", "line-pass");
    b3.delete(); a3.delete();

    // sqrt
    const a5 = tf.ndarray(new Float64Array([4, 9, 16]), [3]);
    const b5 = tf.sqrt(a5);
    assert(b5.dtype === "float64", `dtype=${b5.dtype}`);
    approx(b5.data[0], 2, "sqrt", EPS); approx(b5.data[1], 3, "sqrt", EPS); approx(b5.data[2], 4, "sqrt", EPS);
    log("  sqrt (f64)", "line-pass");
    b5.delete(); a5.delete();

    // sqrt_ in-place
    const a6 = tf.ndarray(new Float64Array([25, 1]), [2]);
    tf.sqrt_(a6);
    approx(a6.data[0], 5, "sqrt_", EPS); approx(a6.data[1], 1, "sqrt_", EPS);
    log("  sqrt_ in-place (f64)", "line-pass");
    a6.delete();
  });

  // ==========================================================================
  test("math functions (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    // sin / cos / tan
    const a1 = tf.ndarray(new Float64Array([0, Math.PI / 2, Math.PI]), [3]);
    const s = tf.sin(a1);
    assert(s.dtype === "float64", `dtype=${s.dtype}`);
    approx(s.data[0], 0, "sin(0)", EPS); approx(s.data[1], 1, "sin(pi/2)", EPS);
    approx(s.data[2], 0, "sin(pi)", 1e-12);
    log("  sin (f64)", "line-pass");
    const c = tf.cos(a1);
    approx(c.data[0], 1, "cos(0)", EPS); approx(c.data[1], 0, "cos(pi/2)", 1e-12); approx(c.data[2], -1, "cos(pi)", EPS);
    log("  cos (f64)", "line-pass");
    const t = tf.tan(tf.ndarray(new Float64Array([0, Math.PI / 4]), [2]));
    approx(t.data[0], 0, "tan(0)", EPS); approx(t.data[1], 1, "tan(pi/4)", EPS);
    log("  tan (f64)", "line-pass");
    t.delete(); c.delete(); s.delete(); a1.delete();

    // asin / acos / atan
    const a2 = tf.ndarray(new Float64Array([0, 0.5, 1]), [3]);
    const as2 = tf.asin(a2);
    approx(as2.data[0], 0, "asin", EPS); approx(as2.data[2], Math.PI / 2, "asin", EPS);
    log("  asin (f64)", "line-pass");
    const ac2 = tf.acos(a2);
    approx(ac2.data[0], Math.PI / 2, "acos", EPS); approx(ac2.data[2], 0, "acos", EPS);
    log("  acos (f64)", "line-pass");
    const at2 = tf.atan(a2);
    approx(at2.data[0], 0, "atan", EPS); approx(at2.data[1], Math.atan(0.5), "atan", EPS);
    log("  atan (f64)", "line-pass");
    at2.delete(); ac2.delete(); as2.delete(); a2.delete();

    // exp / log / log2 / log10
    const a3 = tf.ndarray(new Float64Array([0, 1, 2]), [3]);
    const e = tf.exp(a3);
    approx(e.data[0], 1, "exp(0)", EPS); approx(e.data[1], Math.E, "exp(1)", EPS);
    log("  exp (f64)", "line-pass");
    e.delete(); a3.delete();

    const a4 = tf.ndarray(new Float64Array([1, Math.E, Math.E * Math.E]), [3]);
    const l = tf.log(a4);
    approx(l.data[0], 0, "log(1)", EPS); approx(l.data[1], 1, "log(e)", EPS); approx(l.data[2], 2, "log(e²)", EPS);
    log("  log (f64)", "line-pass");
    l.delete(); a4.delete();

    const a5 = tf.ndarray(new Float64Array([1, 2, 4, 8]), [4]);
    const l2 = tf.log2(a5);
    approx(l2.data[0], 0, "log2", EPS); approx(l2.data[1], 1, "log2", EPS); approx(l2.data[2], 2, "log2", EPS); approx(l2.data[3], 3, "log2", EPS);
    log("  log2 (f64)", "line-pass");
    l2.delete(); a5.delete();

    const a6 = tf.ndarray(new Float64Array([1, 10, 100, 1000]), [4]);
    const l10 = tf.log10(a6);
    approx(l10.data[0], 0, "log10", EPS); approx(l10.data[1], 1, "log10", EPS); approx(l10.data[2], 2, "log10", EPS); approx(l10.data[3], 3, "log10", EPS);
    log("  log10 (f64)", "line-pass");
    l10.delete(); a6.delete();

    // floor / ceil / round
    const a7 = tf.ndarray(new Float64Array([1.2, 2.7, -0.3, -1.8]), [4]);
    const fl = tf.floor(a7);
    approx(fl.data[0], 1, "floor", EPS); approx(fl.data[1], 2, "floor", EPS); approx(fl.data[2], -1, "floor", EPS); approx(fl.data[3], -2, "floor", EPS);
    log("  floor (f64)", "line-pass");
    const ce = tf.ceil(a7);
    approx(ce.data[0], 2, "ceil", EPS); approx(ce.data[1], 3, "ceil", EPS); approx(ce.data[2], 0, "ceil", EPS); approx(ce.data[3], -1, "ceil", EPS);
    log("  ceil (f64)", "line-pass");
    const rn = tf.round(a7);
    approx(rn.data[0], 1, "round", EPS); approx(rn.data[1], 3, "round", EPS); approx(rn.data[2], 0, "round", EPS); approx(rn.data[3], -2, "round", EPS);
    log("  round (f64)", "line-pass");
    rn.delete(); ce.delete(); fl.delete(); a7.delete();

    // pow
    const a8 = tf.ndarray(new Float64Array([2, 3, 4]), [3]);
    const p = tf.pow(a8, 2);
    approx(p.data[0], 4, "pow", EPS); approx(p.data[1], 9, "pow", EPS); approx(p.data[2], 16, "pow", EPS);
    log("  pow (f64)", "line-pass");
    p.delete(); a8.delete();

    // pow_ in-place
    const a9 = tf.ndarray(new Float64Array([4, 9]), [2]);
    tf.pow_(a9, 0.5);
    approx(a9.data[0], 2, "pow_", EPS); approx(a9.data[1], 3, "pow_", EPS);
    log("  pow_ in-place (f64)", "line-pass");
    a9.delete();

    // clip
    const a10 = tf.ndarray(new Float64Array([-2, 0, 3, 5, 10]), [5]);
    const cl = tf.clip(a10, 0, 5);
    approx(cl.data[0], 0, "clip", EPS); approx(cl.data[2], 3, "clip", EPS); approx(cl.data[4], 5, "clip", EPS);
    log("  clip (f64)", "line-pass");
    cl.delete(); a10.delete();

    // in-place trig
    const a11 = tf.ndarray(new Float64Array([0, Math.PI / 2]), [2]);
    tf.sin_(a11);
    approx(a11.data[0], 0, "sin_", EPS); approx(a11.data[1], 1, "sin_", EPS);
    log("  sin_ in-place (f64)", "line-pass");
    a11.delete();

    const a12 = tf.ndarray(new Float64Array([0, 1]), [2]);
    tf.exp_(a12);
    approx(a12.data[0], 1, "exp_", EPS); approx(a12.data[1], Math.E, "exp_", EPS);
    log("  exp_ in-place (f64)", "line-pass");
    a12.delete();
  });

  // ==========================================================================
  test("isNaN (float64)", () => {
    const tf = getTf();

    const a = tf.ndarray(new Float64Array([1, NaN, 3, NaN, 5]));
    assert(a.dtype === "float64", `dtype=${a.dtype}`);
    const mask = a.isNaN();
    assert(mask.dtype === "bool", `mask dtype: ${mask.dtype}`);
    assert(mask.data[0] === 0 && mask.data[1] === 1, "detects NaN");
    assert(mask.data[2] === 0 && mask.data[3] === 1 && mask.data[4] === 0, "non-NaN is 0");
    log("  .isNaN() method (f64)", "line-pass");
    mask.delete();

    // Free function
    const mask2 = tf.isNaN(a);
    assert(mask2.data[1] === 1 && mask2.data[4] === 0, "free function");
    log("  tf.isNaN() free function (f64)", "line-pass");
    mask2.delete();

    // No NaN
    const b = tf.ndarray(new Float64Array([1, 2, 3]));
    const mask3 = b.isNaN();
    assert(mask3.data[0] === 0 && mask3.data[1] === 0 && mask3.data[2] === 0, "no NaN");
    log("  no NaN → all false (f64)", "line-pass");
    mask3.delete(); b.delete();

    // All NaN
    const c = tf.ndarray(new Float64Array([NaN, NaN]));
    const mask4 = c.isNaN();
    assert(mask4.data[0] === 1 && mask4.data[1] === 1, "all NaN");
    log("  all NaN → all true (f64)", "line-pass");
    mask4.delete(); c.delete(); a.delete();
  });

  // ==========================================================================
  test("matMul (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    // 2D: 2x3 @ 3x2
    const a1 = tf.ndarray(new Float64Array([1,2,3,4,5,6]), [2, 3]);
    const b1 = tf.ndarray(new Float64Array([7,8,9,10,11,12]), [3, 2]);
    const c1 = a1.matMul(b1);
    assert(c1.dtype === "float64", `dtype=${c1.dtype}`);
    assert(c1.shape[0] === 2 && c1.shape[1] === 2, "2D shape");
    approx(c1.data[0], 58, "mm", EPS); approx(c1.data[1], 64, "mm", EPS);
    approx(c1.data[2], 139, "mm", EPS); approx(c1.data[3], 154, "mm", EPS);
    log("  2x3 @ 3x2 (f64)", "line-pass");
    c1.delete(); b1.delete(); a1.delete();

    // identity
    const eye = tf.ndarray(new Float64Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]), [4, 4]);
    const m = tf.ndarray(new Float64Array([1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16]), [4, 4]);
    const c2 = eye.matMul(m);
    for (let i = 0; i < 16; i++) approx(c2.data[i], i + 1, `identity [${i}]`, EPS);
    log("  identity @ M = M (f64)", "line-pass");
    c2.delete(); m.delete(); eye.delete();

    // batch broadcast: [B,M,K] x [B,K,N] → [B,M,N]
    const a3 = tf.ndarray(new Float64Array([
      1,0, 0,1,
      2,0, 0,2,
    ]), [2, 2, 2]);
    const b3 = tf.ndarray(new Float64Array([
      1,2, 3,4,
      5,6, 7,8,
    ]), [2, 2, 2]);
    const c3 = a3.matMul(b3);
    assert(c3.shape[0] === 2 && c3.shape[1] === 2 && c3.shape[2] === 2, "batch shape");
    approx(c3.data[0], 1, "mm batch", EPS); approx(c3.data[1], 2, "mm batch", EPS);
    approx(c3.data[2], 3, "mm batch", EPS); approx(c3.data[3], 4, "mm batch", EPS);
    approx(c3.data[4], 10, "mm batch", EPS); approx(c3.data[5], 12, "mm batch", EPS);
    approx(c3.data[6], 14, "mm batch", EPS); approx(c3.data[7], 16, "mm batch", EPS);
    log("  batch [B,M,K] x [B,K,N] (f64)", "line-pass");
    c3.delete(); b3.delete(); a3.delete();
  });

  // ==========================================================================
  test("dot & cross (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    // dot 1D → returns number
    const a1 = tf.ndarray(new Float64Array([1, 2, 3]), [3]);
    const b1 = tf.ndarray(new Float64Array([4, 5, 6]), [3]);
    const r1 = tf.dot(a1, b1);
    approx(r1, 32, "dot scalar", EPS);
    log("  dot 1D: 32 (f64)", "line-pass");
    b1.delete(); a1.delete();

    // dot batch
    const a2 = tf.ndarray(new Float64Array([1,0,0, 0,1,0]), [2, 3]);
    const b2 = tf.ndarray(new Float64Array([1,1,1, 1,1,1]), [2, 3]);
    const r2 = tf.dot(a2, b2);
    assert(r2.dtype === "float64", `dtype=${r2.dtype}`);
    assert(r2.shape[0] === 2, "dot batch shape");
    approx(r2.data[0], 1, "dot batch", EPS); approx(r2.data[1], 1, "dot batch", EPS);
    log("  dot batch (f64)", "line-pass");
    r2.delete(); b2.delete(); a2.delete();

    // cross 1D
    const a3 = tf.ndarray(new Float64Array([1, 0, 0]), [3]);
    const b3 = tf.ndarray(new Float64Array([0, 1, 0]), [3]);
    const r3 = tf.cross(a3, b3);
    assert(r3.dtype === "float64", `dtype=${r3.dtype}`);
    approx(r3.data[0], 0, "cross", EPS); approx(r3.data[1], 0, "cross", EPS); approx(r3.data[2], 1, "cross", EPS);
    log("  cross: x * y = z (f64)", "line-pass");
    r3.delete(); b3.delete(); a3.delete();

    // cross batch
    const a4 = tf.ndarray(new Float64Array([1,0,0, 0,1,0]), [2, 3]);
    const b4 = tf.ndarray(new Float64Array([0,1,0, 0,0,1]), [2, 3]);
    const r4 = tf.cross(a4, b4);
    assert(r4.shape[0] === 2 && r4.shape[1] === 3, "cross batch shape");
    approx(r4.data[0], 0, "cross batch", EPS); approx(r4.data[1], 0, "cross batch", EPS); approx(r4.data[2], 1, "cross batch", EPS);
    approx(r4.data[3], 1, "cross batch", EPS); approx(r4.data[4], 0, "cross batch", EPS); approx(r4.data[5], 0, "cross batch", EPS);
    log("  cross batch (f64)", "line-pass");
    r4.delete(); b4.delete(); a4.delete();
  });

  // ==========================================================================
  test("normalize (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    // no axis
    const a1 = tf.ndarray(new Float64Array([3, 4]), [2]);
    const n1 = tf.normalize(a1);
    assert(n1.dtype === "float64", `dtype=${n1.dtype}`);
    approx(n1.data[0], 0.6, "norm", EPS); approx(n1.data[1], 0.8, "norm", EPS);
    approx(a1.data[0], 3, "original unchanged", EPS);
    log("  normalize [3,4] → [0.6, 0.8] (f64)", "line-pass");
    n1.delete(); a1.delete();

    // 2D flattened
    const a2 = tf.ndarray(new Float64Array([1, 2, 2]), [1, 3]);
    const n2 = tf.normalize(a2);
    approx(n2.data[0], 1/3, "norm", EPS); approx(n2.data[1], 2/3, "norm", EPS); approx(n2.data[2], 2/3, "norm", EPS);
    log("  normalize flattened (f64)", "line-pass");
    n2.delete(); a2.delete();

    // axis=1
    const a3 = tf.ndarray(new Float64Array([3, 4, 0, 5]), [2, 2]);
    const n3 = tf.normalize(a3, 1);
    approx(n3.data[0], 0.6, "n3", EPS); approx(n3.data[1], 0.8, "n3", EPS);
    approx(n3.data[2], 0.0, "n3", EPS); approx(n3.data[3], 1.0, "n3", EPS);
    log("  normalize axis=1 (f64)", "line-pass");
    n3.delete(); a3.delete();

    // axis=0
    const a4 = tf.ndarray(new Float64Array([3, 0, 4, 1]), [2, 2]);
    const n4 = tf.normalize(a4, 0);
    approx(n4.data[0], 0.6, "n4", EPS); approx(n4.data[1], 0.0, "n4", EPS);
    approx(n4.data[2], 0.8, "n4", EPS); approx(n4.data[3], 1.0, "n4", EPS);
    log("  normalize axis=0 (f64)", "line-pass");
    n4.delete(); a4.delete();

    // in-place
    const a5 = tf.ndarray(new Float64Array([3, 4]), [2]);
    tf.normalize_(a5);
    approx(a5.data[0], 0.6, "norm_", EPS); approx(a5.data[1], 0.8, "norm_", EPS);
    log("  normalize_ in-place (f64)", "line-pass");
    a5.delete();

    // in-place axis=1
    const a6 = tf.ndarray(new Float64Array([3, 4, 0, 5]), [2, 2]);
    tf.normalize_(a6, 1);
    approx(a6.data[0], 0.6, "norm_ ax", EPS); approx(a6.data[1], 0.8, "norm_ ax", EPS);
    approx(a6.data[2], 0.0, "norm_ ax", EPS); approx(a6.data[3], 1.0, "norm_ ax", EPS);
    log("  normalize_ axis=1 (f64)", "line-pass");
    a6.delete();
  });

  // ==========================================================================
  test("reductions (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    // flat
    const a1 = tf.ndarray(new Float64Array([1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]), [3,3]);
    approx(a1.sum(), 49.5, "sum", EPS); approx(a1.min(), 1.5, "min", EPS);
    approx(a1.max(), 9.5, "max", EPS); approx(a1.mean(), 5.5, "mean", EPS);
    log("  sum/min/max/mean (f64)", "line-pass");

    // axis
    const s0 = a1.sum(0);
    assert(s0.dtype === "float64", `dtype=${s0.dtype}`);
    approx(s0.data[0], 13.5, "s0", EPS); approx(s0.data[1], 16.5, "s0", EPS); approx(s0.data[2], 19.5, "s0", EPS);
    log("  sum axis=0 (f64)", "line-pass");
    s0.delete();

    const s1 = a1.sum(1);
    approx(s1.data[0], 7.5, "s1", EPS); approx(s1.data[1], 16.5, "s1", EPS); approx(s1.data[2], 25.5, "s1", EPS);
    log("  sum axis=1 (f64)", "line-pass");
    s1.delete();

    const m1 = a1.mean(1);
    approx(m1.data[0], 2.5, "m1", EPS); approx(m1.data[1], 5.5, "m1", EPS); approx(m1.data[2], 8.5, "m1", EPS);
    log("  mean axis=1 (f64)", "line-pass");
    m1.delete();
    a1.delete();

    // standalone
    const a3 = tf.ndarray(new Float64Array([1, 2, 3, 4]), [4]);
    approx(tf.sum(a3), 10, "tf.sum", EPS); approx(tf.min(a3), 1, "tf.min", EPS);
    approx(tf.max(a3), 4, "tf.max", EPS); approx(tf.mean(a3), 2.5, "tf.mean", EPS);
    log("  tf.sum/min/max/mean (f64)", "line-pass");
    a3.delete();

    // argmin / argmax
    const a4 = tf.ndarray(new Float64Array([3, 1, 5, 2]), [4]);
    assert(a4.argmin() === 1, "argmin"); assert(a4.argmax() === 2, "argmax");
    assert(tf.argmin(a4) === 1, "tf.argmin"); assert(tf.argmax(a4) === 2, "tf.argmax");
    log("  argmin/argmax (f64)", "line-pass");
    a4.delete();

    const a5 = tf.ndarray(new Float64Array([1, 3, 4, 2]), [2, 2]);
    const r5 = a5.argmin(1);
    assert(r5.data[0] === 0 && r5.data[1] === 1, "argmin axis=1");
    log("  argmin axis=1 (f64)", "line-pass");
    r5.delete(); a5.delete();

    const a6 = tf.ndarray(new Float64Array([1, 5, 4, 2]), [2, 2]);
    const r6 = a6.argmax(0);
    assert(r6.data[0] === 1 && r6.data[1] === 0, "argmax axis=0");
    log("  argmax axis=0 (f64)", "line-pass");
    r6.delete(); a6.delete();
  });

  // ==========================================================================
  test("reductions async (float64)", async () => {
    const tf = getTf();
    const EPS = 1e-12;
    const floats = tf.ndarray(new Float64Array([1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]), [3,3]);

    approx(await tf.async.sum(floats), 49.5, "async sum", EPS);
    log("  async sum (f64)", "line-pass");
    approx(await tf.async.mean(floats), 5.5, "async mean", EPS);
    log("  async mean (f64)", "line-pass");

    const asum1 = await tf.async.sum(floats, 1);
    assert(asum1.dtype === "float64", `dtype=${asum1.dtype}`);
    approx(asum1.data[0], 7.5, "asum1", EPS); approx(asum1.data[2], 25.5, "asum1", EPS);
    log("  async sum axis=1 (f64)", "line-pass");
    asum1.delete();

    floats.delete();
  });

  // ==========================================================================
  test("assign (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    // scalar fill
    const a1 = tf.ndarray(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    a1.assign(0);
    for (let i = 0; i < 6; i++) approx(a1.data[i], 0, `[${i}]`, EPS);
    log("  assign scalar (f64)", "line-pass");
    a1.delete();

    // array copy (same shape)
    const a2 = tf.ndarray(new Float64Array([0, 0, 0, 0, 0, 0]), [2, 3]);
    const a2v = tf.ndarray(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    a2.assign(a2v);
    approx(a2.data[0], 1, "a2", EPS); approx(a2.data[5], 6, "a2", EPS);
    log("  assign array (same shape) (f64)", "line-pass");
    a2v.delete(); a2.delete();

    // array broadcast [3] → [2,3]
    const a3 = tf.ndarray(new Float64Array([0, 0, 0, 0, 0, 0]), [2, 3]);
    const a3v = tf.ndarray(new Float64Array([10, 20, 30]), [3]);
    a3.assign(a3v);
    approx(a3.data[0], 10, "a3", EPS); approx(a3.data[1], 20, "a3", EPS); approx(a3.data[2], 30, "a3", EPS);
    approx(a3.data[3], 10, "a3", EPS); approx(a3.data[4], 20, "a3", EPS); approx(a3.data[5], 30, "a3", EPS);
    log("  assign array broadcast [3]→[2,3] (f64)", "line-pass");
    a3v.delete(); a3.delete();

    // indexed scalar
    const a4 = tf.ndarray(new Float64Array([1, 2, 3, 4, 5, 6]), [3, 2]);
    const a4i = tf.ndarray(new Int32Array([0, 2]), [2]);
    a4.assign(a4i, 99);
    approx(a4.data[0], 99, "a4", EPS); approx(a4.data[1], 99, "a4", EPS);
    approx(a4.data[2], 3, "a4", EPS); approx(a4.data[3], 4, "a4", EPS);
    approx(a4.data[4], 99, "a4", EPS); approx(a4.data[5], 99, "a4", EPS);
    log("  assign indexed scalar (f64)", "line-pass");
    a4i.delete(); a4.delete();

    // indexed array (exact match)
    const a5 = tf.ndarray(new Float64Array([0, 0, 0, 0, 0, 0]), [3, 2]);
    const a5i = tf.ndarray(new Int32Array([1, 2]), [2]);
    const a5v = tf.ndarray(new Float64Array([10, 20, 30, 40]), [2, 2]);
    a5.assign(a5i, a5v);
    approx(a5.data[0], 0, "a5", EPS);  approx(a5.data[1], 0, "a5", EPS);
    approx(a5.data[2], 10, "a5", EPS); approx(a5.data[3], 20, "a5", EPS);
    approx(a5.data[4], 30, "a5", EPS); approx(a5.data[5], 40, "a5", EPS);
    log("  assign indexed array (f64)", "line-pass");
    a5v.delete(); a5i.delete(); a5.delete();

    // masked scalar
    const a6 = tf.ndarray(new Float64Array([1, 2, 3, 4, 5, 6]), [3, 2]);
    const a6m = tf.ndarray(new Int8Array([1, 0, 1]), [3]).as("bool");
    a6.assign(a6m, 0);
    approx(a6.data[0], 0, "a6", EPS); approx(a6.data[1], 0, "a6", EPS);
    approx(a6.data[2], 3, "a6", EPS); approx(a6.data[3], 4, "a6", EPS);
    approx(a6.data[4], 0, "a6", EPS); approx(a6.data[5], 0, "a6", EPS);
    log("  assign masked scalar (f64)", "line-pass");
    a6m.delete(); a6.delete();

    // masked array (exact match)
    const a7 = tf.ndarray(new Float64Array([0, 0, 0, 0, 0, 0]), [3, 2]);
    const a7m = tf.ndarray(new Int8Array([0, 1, 1]), [3]).as("bool");
    const a7v = tf.ndarray(new Float64Array([10, 20, 30, 40]), [2, 2]);
    a7.assign(a7m, a7v);
    approx(a7.data[0], 0, "a7", EPS);  approx(a7.data[1], 0, "a7", EPS);
    approx(a7.data[2], 10, "a7", EPS); approx(a7.data[3], 20, "a7", EPS);
    approx(a7.data[4], 30, "a7", EPS); approx(a7.data[5], 40, "a7", EPS);
    log("  assign masked array (f64)", "line-pass");
    a7v.delete(); a7m.delete(); a7.delete();
  });

  // ==========================================================================
  test("norm (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    // global norm — [3, 4] → 5
    const n1 = tf.ndarray(new Float64Array([3, 4]), [2]);
    const n1r = tf.norm(n1);
    approx(n1r, 5, "norm scalar", EPS);
    log("  norm global [3,4]→5 (f64)", "line-pass");
    n1.delete();

    // global norm — [1, 2, 2] → 3
    const n2 = tf.ndarray(new Float64Array([1, 2, 2]), [3]);
    approx(tf.norm(n2), 3, "norm", EPS);
    log("  norm global [1,2,2]→3 (f64)", "line-pass");
    n2.delete();

    // norm along axis 1 — [N,3]: per-row vector magnitude
    const n3 = tf.ndarray(new Float64Array([3, 4, 0, 0, 0, 5, 1, 2, 2]), [3, 3]);
    const n3r = tf.norm(n3, 1);
    assert(n3r.dtype === "float64", `dtype=${n3r.dtype}`);
    assert(n3r.shape[0] === 3, "norm axis shape");
    approx(n3r.data[0], 5, "n3", EPS); approx(n3r.data[1], 5, "n3", EPS); approx(n3r.data[2], 3, "n3", EPS);
    log("  norm axis=1 per-row (f64)", "line-pass");
    n3r.delete(); n3.delete();

    // norm along axis 0
    const n4 = tf.ndarray(new Float64Array([1, 0, 0, 1]), [2, 2]);
    const n4r = tf.norm(n4, 0);
    assert(n4r.shape[0] === 2, "norm axis=0 shape");
    approx(n4r.data[0], 1, "n4", EPS); approx(n4r.data[1], 1, "n4", EPS);
    log("  norm axis=0 column norms (f64)", "line-pass");
    n4r.delete(); n4.delete();

    // regression: large array triggers the parallel reduce path
    const nLarge = 100_000;
    const n6 = tf.full("float64", [nLarge], 1);
    approx(n6.norm(), Math.sqrt(nLarge),
      "norm large array (parallel reduce correctness, f64)", 1e-9);
    log("  norm large array (parallel reducer correctness, f64)", "line-pass");
    n6.delete();
  });

  // ==========================================================================
  test("norm instance method (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;
    // Per-row norm on [3, 3]
    const pts = tf.ndarray(new Float64Array([
      3, 4, 0,
      0, 0, 5,
      1, 0, 0,
    ]), [3, 3]);

    const norms = pts.norm(1);
    assert(norms.dtype === "float64", `dtype=${norms.dtype}`);
    assert(norms.shape.length === 1 && norms.shape[0] === 3, `expected [3], got [${norms.shape}]`);
    approx(norms.data[0], 5.0, "norm of [3,4,0]", EPS);
    approx(norms.data[1], 5.0, "norm of [0,0,5]", EPS);
    approx(norms.data[2], 1.0, "norm of [1,0,0]", EPS);
    log("  pts.norm(1) per-row norms (f64)", "line-pass");

    // Global norm
    const simple = tf.ndarray(new Float64Array([3, 4]), [2]);
    approx(simple.norm(), 5.0, "global norm", EPS);
    log("  [3,4].norm() = 5 (f64)", "line-pass");

    simple.delete();
    norms.delete();
    pts.delete();
  });

  // ==========================================================================
  test("atan2 (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    // basic atan2(y, x)
    const y1 = tf.ndarray(new Float64Array([1, 0, -1, 0]), [4]);
    const x1 = tf.ndarray(new Float64Array([0, 1, 0, -1]), [4]);
    const r1 = tf.atan2(y1, x1);
    assert(r1.dtype === "float64", `dtype=${r1.dtype}`);
    assert(r1.shape[0] === 4, "atan2 shape");
    approx(r1.data[0], Math.PI / 2, "atan2", EPS);
    approx(r1.data[1], 0, "atan2", EPS);
    approx(r1.data[2], -Math.PI / 2, "atan2", EPS);
    approx(r1.data[3], Math.PI, "atan2", EPS);
    log("  atan2 basic (f64)", "line-pass");
    r1.delete(); x1.delete(); y1.delete();

    // atan2 broadcasting — [N,1] with [1,M]
    const y2 = tf.ndarray(new Float64Array([1, -1]), [2, 1]);
    const x2 = tf.ndarray(new Float64Array([1, -1]), [1, 2]);
    const r2 = tf.atan2(y2, x2);
    assert(r2.shape[0] === 2 && r2.shape[1] === 2, "atan2 broadcast shape");
    approx(r2.data[0], Math.atan2(1, 1), "atan2 bc", EPS);
    approx(r2.data[1], Math.atan2(1, -1), "atan2 bc", EPS);
    approx(r2.data[2], Math.atan2(-1, 1), "atan2 bc", EPS);
    approx(r2.data[3], Math.atan2(-1, -1), "atan2 bc", EPS);
    log("  atan2 broadcast (f64)", "line-pass");
    r2.delete(); x2.delete(); y2.delete();
  });

  // ==========================================================================
  test("mod & mod_ (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;

    const a = tf.ndarray(new Float64Array([5, 7, 10, 13]), [4]);
    const r = tf.mod(a, 3);
    assert(r.dtype === "float64", `dtype=${r.dtype}`);
    approx(r.data[0], 2, "mod", EPS); approx(r.data[1], 1, "mod", EPS); approx(r.data[2], 1, "mod", EPS); approx(r.data[3], 1, "mod", EPS);
    log("  mod(a, 3) (f64)", "line-pass");
    r.delete();

    // mod with NDArray
    const b = tf.ndarray(new Float64Array([3, 4, 5, 6]), [4]);
    const r2 = tf.mod(a, b);
    approx(r2.data[0], 2, "mod(a,b)", EPS); approx(r2.data[1], 3, "mod(a,b)", EPS);
    log("  mod(a, b) (f64)", "line-pass");
    r2.delete(); b.delete();

    // mod_ in-place
    const c = tf.ndarray(new Float64Array([10, 11, 12]), [3]);
    tf.mod_(c, 5);
    approx(c.data[0], 0, "mod_", EPS); approx(c.data[1], 1, "mod_", EPS); approx(c.data[2], 2, "mod_", EPS);
    log("  mod_ in-place (f64)", "line-pass");
    c.delete(); a.delete();
  });

  // ==========================================================================
  test("clip_ in-place (float64)", () => {
    const tf = getTf();
    const EPS = 1e-12;
    const a = tf.ndarray(new Float64Array([-1, 0, 3, 5, 10, 15]), [6]);
    a.clip_(0, 10);
    assert(a.dtype === "float64", `dtype=${a.dtype}`);
    approx(a.data[0], 0, "clamped min", EPS);
    approx(a.data[2], 3, "unchanged", EPS);
    approx(a.data[5], 10, "clamped max", EPS);
    log("  clip_ in-place (f64)", "line-pass");
    a.delete();
  });

});
