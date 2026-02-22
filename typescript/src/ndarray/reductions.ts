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

export function sum(arr: NDArray, axis: number = -1): number | NDArray {
  const outDtype = arr.dtype === "float32" ? "float32" : "int32";
  return wrapResult(native()[`sum_${nd(arr.dtype)}`](arr._handle, axis), outDtype);
}

export function min(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native()[`min_${nd(arr.dtype)}`](arr._handle, axis), arr.dtype);
}

export function max(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native()[`max_${nd(arr.dtype)}`](arr._handle, axis), arr.dtype);
}

export function mean(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native()[`mean_${nd(arr.dtype)}`](arr._handle, axis), "float32");
}

export function norm(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native()[`norm_${nd(arr.dtype)}`](arr._handle, axis), "float32");
}

export function argmin(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native()[`argmin_${nd(arr.dtype)}`](arr._handle, axis), "int32");
}

export function argmax(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native()[`argmax_${nd(arr.dtype)}`](arr._handle, axis), "int32");
}

export function any(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native().any_int8(arr._handle, axis), "bool");
}

export function all(arr: NDArray, axis: number = -1): number | NDArray {
  return wrapResult(native().all_int8(arr._handle, axis), "bool");
}
