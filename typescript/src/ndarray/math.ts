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
import { NDArray, NDArrayBool } from "./NDArray";
import { norm } from "./reductions";

function nd(dtype: string): string {
  return dtype === "bool" ? "int8" : dtype;
}

// ============================================================================
// Unary math functions — float dtypes only
// ============================================================================

/** Element-wise square root. */
export function sqrt(arr: NDArray): NDArray {
  return new NDArray(native()[`sqrt_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise square root, in-place. */
export function sqrt_(arr: NDArray): NDArray {
  native()[`sqrt_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise sine (radians). */
export function sin(arr: NDArray): NDArray {
  return new NDArray(native()[`sin_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise sine, in-place. */
export function sin_(arr: NDArray): NDArray {
  native()[`sin_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise cosine (radians). */
export function cos(arr: NDArray): NDArray {
  return new NDArray(native()[`cos_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise cosine, in-place. */
export function cos_(arr: NDArray): NDArray {
  native()[`cos_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise tangent (radians). */
export function tan(arr: NDArray): NDArray {
  return new NDArray(native()[`tan_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise tangent, in-place. */
export function tan_(arr: NDArray): NDArray {
  native()[`tan_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise arc sine. */
export function asin(arr: NDArray): NDArray {
  return new NDArray(native()[`asin_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise arc sine, in-place. */
export function asin_(arr: NDArray): NDArray {
  native()[`asin_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise arc cosine. */
export function acos(arr: NDArray): NDArray {
  return new NDArray(native()[`acos_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise arc cosine, in-place. */
export function acos_(arr: NDArray): NDArray {
  native()[`acos_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise arc tangent. */
export function atan(arr: NDArray): NDArray {
  return new NDArray(native()[`atan_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise arc tangent, in-place. */
export function atan_(arr: NDArray): NDArray {
  native()[`atan_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise atan2(y, x). Broadcasts. */
export function atan2(y: NDArray, x: NDArray): NDArray {
  return new NDArray(
    native()[`atan2_${y.dtype}`](y._handle, x._handle), y.dtype,
  );
}

/** Element-wise exponential (e^x). */
export function exp(arr: NDArray): NDArray {
  return new NDArray(native()[`exp_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise exponential, in-place. */
export function exp_(arr: NDArray): NDArray {
  native()[`exp_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise natural logarithm. */
export function log(arr: NDArray): NDArray {
  return new NDArray(native()[`log_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise natural logarithm, in-place. */
export function log_(arr: NDArray): NDArray {
  native()[`log_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise base-2 logarithm. */
export function log2(arr: NDArray): NDArray {
  return new NDArray(native()[`log2_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise base-2 logarithm, in-place. */
export function log2_(arr: NDArray): NDArray {
  native()[`log2_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise base-10 logarithm. */
export function log10(arr: NDArray): NDArray {
  return new NDArray(native()[`log10_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise base-10 logarithm, in-place. */
export function log10_(arr: NDArray): NDArray {
  native()[`log10_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise floor. */
export function floor(arr: NDArray): NDArray {
  return new NDArray(native()[`floor_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise floor, in-place. */
export function floor_(arr: NDArray): NDArray {
  native()[`floor_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise ceiling. */
export function ceil(arr: NDArray): NDArray {
  return new NDArray(native()[`ceil_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise ceiling, in-place. */
export function ceil_(arr: NDArray): NDArray {
  native()[`ceil_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

/** Element-wise rounding to nearest integer. */
export function round(arr: NDArray): NDArray {
  return new NDArray(native()[`round_${arr.dtype}`](arr._handle), arr.dtype);
}
/** Element-wise rounding, in-place. */
export function round_(arr: NDArray): NDArray {
  native()[`round_inplace_${arr.dtype}`](arr._handle);
  return arr;
}

// ============================================================================
// pow (float, takes scalar exponent)
// ============================================================================

/** Element-wise power. */
export function pow(arr: NDArray, exponent: number): NDArray {
  return new NDArray(
    native()[`pow_${arr.dtype}`](arr._handle, exponent), arr.dtype,
  );
}
/** Element-wise power, in-place. */
export function pow_(arr: NDArray, exponent: number): NDArray {
  native()[`pow_inplace_${arr.dtype}`](arr._handle, exponent);
  return arr;
}

// ============================================================================
// Multi-dtype wrappers (abs, neg, clip work on all dtypes)
// ============================================================================

/** Element-wise absolute value. Supports all dtypes. */
export function abs(arr: NDArray): NDArray {
  return new NDArray(native()[`abs_${nd(arr.dtype)}`](arr._handle), arr.dtype);
}

/** Element-wise negation. Supports all dtypes. */
export function neg(arr: NDArray): NDArray {
  return new NDArray(native()[`neg_${nd(arr.dtype)}`](arr._handle), arr.dtype);
}

/** Clamp values to `[lo, hi]`. Supports all dtypes. */
export function clip(arr: NDArray, lo: number, hi: number): NDArray {
  return new NDArray(
    native()[`clip_${nd(arr.dtype)}`](arr._handle, lo, hi),
    arr.dtype,
  );
}

// ============================================================================
// dot / cross
// ============================================================================

/** Dot product along last axis. [N,3] x [N,3] -> [N]. Broadcasts batch dims. Returns number for 1D inputs. */
export function dot(a: NDArray, b: NDArray): number | NDArray {
  const nd_ = nd(a.dtype);
  const handle = native()[`dot_${nd_}`](a._handle, b._handle);
  if (a.ndim === 1 && b.ndim === 1) {
    const val = handle.data()[0];
    handle.delete();
    return val;
  }
  return new NDArray(handle, a.dtype);
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

// ============================================================================
// mod (truncated remainder — matches JS % operator)
// ============================================================================

/** Element-wise remainder (truncated, matching JS `%`). Supports all dtypes. */
export function mod(a: NDArray, b: NDArray | number): NDArray {
  return a.mod(b);
}

/** Element-wise remainder in-place. */
export function mod_(a: NDArray, b: NDArray | number): NDArray {
  return a.mod_(b);
}

// ============================================================================
// isNaN
// ============================================================================

/** Element-wise NaN detection. Returns true where value is NaN. */
export function isNaN(arr: NDArray): NDArrayBool {
  return arr.isNaN();
}
