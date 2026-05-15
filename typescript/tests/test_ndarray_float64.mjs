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
