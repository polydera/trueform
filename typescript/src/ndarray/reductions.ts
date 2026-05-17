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
import { NDArray } from "./NDArray";

function wrapResult(raw: any, dtype: string): number | NDArray {
  if (typeof raw === "number") return raw;
  return new NDArray(raw, dtype);
}

function nd(dtype: string): string {
  return dtype === "bool" ? "int8" : dtype;
}

function floatDtype(dtype: string): string {
  return dtype === "float64" ? "float64" : "float32";
}

/** Sum of elements. Without axis: scalar. With axis: reduced NDArray. */
export function sum(arr: NDArray, axis: number = -1): number | NDArray {
  const out = arr.dtype === "float32" || arr.dtype === "float64"
    ? arr.dtype : "int32";
  return wrapResult(native()[`sum_${nd(arr.dtype)}`](arr._handle, axis), out);
}

/** Minimum value. Without axis: scalar. With axis: reduced NDArray. */
export function min(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native()[`min_${nd(arr.dtype)}`](arr._handle, axis), arr.dtype);
}

/** Maximum value. Without axis: scalar. With axis: reduced NDArray. */
export function max(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native()[`max_${nd(arr.dtype)}`](arr._handle, axis), arr.dtype);
}

/** Arithmetic mean. Without axis: scalar. With axis: reduced NDArray (float). */
export function mean(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(
    native()[`mean_${nd(arr.dtype)}`](arr._handle, axis), floatDtype(arr.dtype),
  );
}

/** L2 norm. Without axis: scalar. With axis: per-slice norms (float). */
export function norm(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(
    native()[`norm_${nd(arr.dtype)}`](arr._handle, axis), floatDtype(arr.dtype),
  );
}

/** Index of minimum value. Without axis: flat index. With axis: per-slice indices. */
export function argmin(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native()[`argmin_${nd(arr.dtype)}`](arr._handle, axis), "int32");
}

/** Index of maximum value. Without axis: flat index. With axis: per-slice indices. */
export function argmax(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native()[`argmax_${nd(arr.dtype)}`](arr._handle, axis), "int32");
}

/** True if any element is nonzero. Without axis: scalar. With axis: per-slice. */
export function any(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native().any_int8(arr._handle, axis), "bool");
}

/** True if all elements are nonzero. Without axis: scalar. With axis: per-slice. */
export function all(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native().all_int8(arr._handle, axis), "bool");
}
