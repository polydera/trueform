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

import { native } from "../native";
import { NDArray } from "../ndarray/NDArray";
import { ndarray } from "../ndarray/factories";
import type { NDArrayFloat32 } from "../ndarray/NDArray";

/** Axis specifier: principal axis name or arbitrary unit vector. */
export type Axis = "x" | "y" | "z" | [number, number, number] | NDArray;

const DEG_TO_RAD = Math.PI / 180;

function toXYZ(v: [number, number, number] | NDArray): [number, number, number] {
  if (Array.isArray(v)) return v;
  const d = v.data as Float32Array;
  return [d[0], d[1], d[2]];
}

/**
 * Returns a 4x4 translation matrix as NDArrayFloat32 [4, 4], row-major.
 */
export function makeTranslation(v: [number, number, number] | NDArray): NDArrayFloat32;
export function makeTranslation(x: number, y: number, z: number): NDArrayFloat32;
export function makeTranslation(
  xOrV: number | [number, number, number] | NDArray, y?: number, z?: number,
): NDArrayFloat32 {
  let tx: number, ty: number, tz: number;
  if (typeof xOrV === "number") {
    tx = xOrV; ty = y!; tz = z!;
  } else {
    [tx, ty, tz] = toXYZ(xOrV);
  }
  // prettier-ignore
  return ndarray(new Float32Array([
    1, 0, 0, tx,
    0, 1, 0, ty,
    0, 0, 1, tz,
    0, 0, 0, 1,
  ]), [4, 4]);
}

/**
 * Returns a 4x4 rotation matrix as NDArrayFloat32 [4, 4], row-major.
 *
 * @param degrees Rotation angle in degrees.
 * @param axis Principal axis ("x", "y", "z") or unit vector.
 * @param pivot Optional pivot point.
 */
export function makeRotation(
  degrees: number,
  axis: Axis,
  pivot?: [number, number, number] | NDArray,
): NDArrayFloat32 {
  const rad = degrees * DEG_TO_RAD;
  const c = Math.cos(rad);
  const s = Math.sin(rad);

  let out: Float32Array;

  if (axis === "x") {
    // prettier-ignore
    out = new Float32Array([
      1, 0,  0, 0,
      0, c, -s, 0,
      0, s,  c, 0,
      0, 0,  0, 1,
    ]);
  } else if (axis === "y") {
    // prettier-ignore
    out = new Float32Array([
       c, 0, s, 0,
       0, 1, 0, 0,
      -s, 0, c, 0,
       0, 0, 0, 1,
    ]);
  } else if (axis === "z") {
    // prettier-ignore
    out = new Float32Array([
      c, -s, 0, 0,
      s,  c, 0, 0,
      0,  0, 1, 0,
      0,  0, 0, 1,
    ]);
  } else {
    const [x, y, z] = toXYZ(axis);
    const t = 1 - c;
    // prettier-ignore
    out = new Float32Array([
      t*x*x + c,     t*x*y - s*z,   t*x*z + s*y,   0,
      t*x*y + s*z,   t*y*y + c,     t*y*z - s*x,   0,
      t*x*z - s*y,   t*y*z + s*x,   t*z*z + c,     0,
      0,             0,             0,             1,
    ]);
  }

  if (pivot) {
    const [px, py, pz] = toXYZ(pivot);
    // T = translate(pivot) * R * translate(-pivot)
    // translation column = pivot - R * pivot
    out[3] = px - (out[0] * px + out[1] * py + out[2] * pz);
    out[7] = py - (out[4] * px + out[5] * py + out[6] * pz);
    out[11] = pz - (out[8] * px + out[9] * py + out[10] * pz);
  }

  return ndarray(out, [4, 4]);
}

/**
 * Invert an affine transformation matrix.
 *
 * Supports 4x4 (3D affine) and 3x3 (2D affine) matrices. Output dtype
 * matches input (float32 in / float32 out, float64 in / float64 out).
 *
 * @param m The 4x4 or 3x3 transformation matrix.
 * @returns The inverted matrix with the same shape and dtype.
 * @throws If the matrix is not 4x4 or 3x3.
 */
export function inverted(m: NDArray): NDArray {
  if (m.shape.length !== 2 || m.shape[0] !== m.shape[1] ||
      (m.shape[0] !== 4 && m.shape[0] !== 3)) {
    throw new Error(
      `inverted: expected 4x4 or 3x3 matrix, got [${m.shape.join(", ")}]`,
    );
  }
  const dt = m.dtype === "float64" ? "float64" : "float32";
  const safe = m.dtype === dt ? m : m.as(dt);
  return new NDArray(native()[`inverted_${dt}`](safe._handle), dt);
}

/**
 * Returns a 4x4 random rotation matrix as NDArrayFloat32 [4, 4], row-major.
 *
 * @param pivot Optional pivot point.
 */
export function makeRandomRotation(
  pivot?: [number, number, number] | NDArray,
): NDArrayFloat32 {
  const r0 = Math.random() * 2 - 1;
  const r1 = Math.random() * 2 - 1;
  const r2 = Math.random() * 2 - 1;
  const len = Math.sqrt(r0 * r0 + r1 * r1 + r2 * r2) || 1;
  const axis: [number, number, number] = [r0 / len, r1 / len, r2 / len];
  const degrees = Math.random() * 360;
  return makeRotation(degrees, axis, pivot);
}
