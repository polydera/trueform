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

import { ndarray } from "../ndarray/factories";
import type { NDArrayFloat32 } from "../ndarray/NDArray";

/** Axis specifier: principal axis name or arbitrary unit vector [x, y, z]. */
export type Axis = "x" | "y" | "z" | [number, number, number];

const DEG_TO_RAD = Math.PI / 180;

/**
 * Returns a 4x4 translation matrix as NDArrayFloat32 [4, 4], row-major.
 */
export function makeTranslation(x: number, y: number, z: number): NDArrayFloat32 {
  // prettier-ignore
  return ndarray(new Float32Array([
    1, 0, 0, x,
    0, 1, 0, y,
    0, 0, 1, z,
    0, 0, 0, 1,
  ]), [4, 4]);
}

/**
 * Returns a 4x4 rotation matrix as NDArrayFloat32 [4, 4], row-major.
 *
 * @param degrees Rotation angle in degrees.
 * @param axis Principal axis ("x", "y", "z") or unit vector [x, y, z].
 * @param pivot Optional pivot point [x, y, z].
 */
export function makeRotation(
  degrees: number,
  axis: Axis,
  pivot?: [number, number, number],
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
    const [x, y, z] = axis;
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
    const [px, py, pz] = pivot;
    // T = translate(pivot) * R * translate(-pivot)
    // translation column = pivot - R * pivot
    out[3] = px - (out[0] * px + out[1] * py + out[2] * pz);
    out[7] = py - (out[4] * px + out[5] * py + out[6] * pz);
    out[11] = pz - (out[8] * px + out[9] * py + out[10] * pz);
  }

  return ndarray(out, [4, 4]);
}
