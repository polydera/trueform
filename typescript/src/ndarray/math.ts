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
import { NDArray, NDArrayFloat32 } from "./NDArray";
import { norm } from "./reductions";

function nd(dtype: string): string {
  return dtype === "bool" ? "int8" : dtype;
}

// ============================================================================
// Unary math functions (float32 only)
// ============================================================================

export function sqrt(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().sqrt_float32(arr._handle), "float32");
}
export function sqrt_(arr: NDArray): NDArray {
  native().sqrt_inplace_float32(arr._handle);
  return arr;
}

export function sin(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().sin_float32(arr._handle), "float32");
}
export function sin_(arr: NDArray): NDArray {
  native().sin_inplace_float32(arr._handle);
  return arr;
}

export function cos(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().cos_float32(arr._handle), "float32");
}
export function cos_(arr: NDArray): NDArray {
  native().cos_inplace_float32(arr._handle);
  return arr;
}

export function tan(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().tan_float32(arr._handle), "float32");
}
export function tan_(arr: NDArray): NDArray {
  native().tan_inplace_float32(arr._handle);
  return arr;
}

export function asin(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().asin_float32(arr._handle), "float32");
}
export function asin_(arr: NDArray): NDArray {
  native().asin_inplace_float32(arr._handle);
  return arr;
}

export function acos(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().acos_float32(arr._handle), "float32");
}
export function acos_(arr: NDArray): NDArray {
  native().acos_inplace_float32(arr._handle);
  return arr;
}

export function atan(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().atan_float32(arr._handle), "float32");
}
export function atan_(arr: NDArray): NDArray {
  native().atan_inplace_float32(arr._handle);
  return arr;
}

/** Element-wise atan2(y, x). Broadcasts. */
export function atan2(y: NDArray, x: NDArray): NDArrayFloat32 {
  return new NDArray(native().atan2_float32(y._handle, x._handle), "float32");
}

export function exp(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().exp_float32(arr._handle), "float32");
}
export function exp_(arr: NDArray): NDArray {
  native().exp_inplace_float32(arr._handle);
  return arr;
}

export function log(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().log_float32(arr._handle), "float32");
}
export function log_(arr: NDArray): NDArray {
  native().log_inplace_float32(arr._handle);
  return arr;
}

export function log2(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().log2_float32(arr._handle), "float32");
}
export function log2_(arr: NDArray): NDArray {
  native().log2_inplace_float32(arr._handle);
  return arr;
}

export function log10(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().log10_float32(arr._handle), "float32");
}
export function log10_(arr: NDArray): NDArray {
  native().log10_inplace_float32(arr._handle);
  return arr;
}

export function floor(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().floor_float32(arr._handle), "float32");
}
export function floor_(arr: NDArray): NDArray {
  native().floor_inplace_float32(arr._handle);
  return arr;
}

export function ceil(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().ceil_float32(arr._handle), "float32");
}
export function ceil_(arr: NDArray): NDArray {
  native().ceil_inplace_float32(arr._handle);
  return arr;
}

export function round(arr: NDArray): NDArrayFloat32 {
  return new NDArray(native().round_float32(arr._handle), "float32");
}
export function round_(arr: NDArray): NDArray {
  native().round_inplace_float32(arr._handle);
  return arr;
}

// ============================================================================
// pow (float32, takes scalar exponent)
// ============================================================================

export function pow(arr: NDArray, exponent: number): NDArrayFloat32 {
  return new NDArray(native().pow_float32(arr._handle, exponent), "float32");
}
export function pow_(arr: NDArray, exponent: number): NDArray {
  native().pow_inplace_float32(arr._handle, exponent);
  return arr;
}

// ============================================================================
// Multi-dtype wrappers (abs, neg, clip work on all dtypes)
// ============================================================================

export function abs(arr: NDArray): NDArray {
  return new NDArray(native()[`abs_${nd(arr.dtype)}`](arr._handle), arr.dtype);
}

export function neg(arr: NDArray): NDArray {
  return new NDArray(native()[`neg_${nd(arr.dtype)}`](arr._handle), arr.dtype);
}

export function clip(arr: NDArray, lo: number, hi: number): NDArray {
  return new NDArray(
    native()[`clip_${nd(arr.dtype)}`](arr._handle, lo, hi),
    arr.dtype,
  );
}

// ============================================================================
// dot / cross
// ============================================================================

/** Dot product along last axis. [N,3] x [N,3] -> [N]. Broadcasts batch dims. */
export function dot(a: NDArray, b: NDArray): NDArray {
  const nd_ = nd(a.dtype);
  return new NDArray(native()[`dot_${nd_}`](a._handle, b._handle), a.dtype);
}

/** Cross product along last axis (must be 3). [N,3] x [N,3] -> [N,3]. Broadcasts batch dims. */
export function cross(a: NDArray, b: NDArray): NDArray {
  const nd_ = nd(a.dtype);
  return new NDArray(native()[`cross_${nd_}`](a._handle, b._handle), a.dtype);
}

// ============================================================================
// Compound operations
// ============================================================================

/** L2-normalize an array, returning a new array. */
export function normalize(arr: NDArray, axis?: number): NDArray {
  if (axis === undefined) {
    return arr.div(norm(arr) as number);
  }
  const n = norm(arr, axis) as NDArray;
  const expanded = n.unsqueeze(axis);
  n.delete();
  const result = arr.div(expanded);
  expanded.delete();
  return result;
}

/** L2-normalize an array in-place. */
export function normalize_(arr: NDArray, axis?: number): NDArray {
  if (axis === undefined) {
    return arr.div_(norm(arr) as number);
  }
  const n = norm(arr, axis) as NDArray;
  const expanded = n.unsqueeze(axis);
  n.delete();
  arr.div_(expanded);
  expanded.delete();
  return arr;
}
