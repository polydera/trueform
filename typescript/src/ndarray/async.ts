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

import { native, dispatcher } from "../native";
import { NDArray } from "./NDArray";

function wrapResult(raw: any, dtype: string): number | NDArray {
  if (typeof raw === "number") return raw;
  return new NDArray(raw, dtype);
}

function nd(dtype: string): string {
  return dtype === "bool" ? "int8" : dtype;
}

export async function sum(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  const outDtype = arr.dtype === "float32" ? "float32" : "int32";
  return dispatcher().run(
    () => native()[`dispatch_sum_${nd(arr.dtype)}`](arr._handle, axis),
    (raw) => wrapResult(raw, outDtype),
  );
}

export async function min(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_min_${nd(arr.dtype)}`](arr._handle, axis),
    (raw) => wrapResult(raw, arr.dtype),
  );
}

export async function max(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_max_${nd(arr.dtype)}`](arr._handle, axis),
    (raw) => wrapResult(raw, arr.dtype),
  );
}

export async function mean(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_mean_${nd(arr.dtype)}`](arr._handle, axis),
    (raw) => wrapResult(raw, "float32"),
  );
}

export async function norm(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_norm_${nd(arr.dtype)}`](arr._handle, axis),
    (raw) => wrapResult(raw, "float32"),
  );
}

export async function atan2(y: NDArray, x: NDArray): Promise<NDArray> {
  return dispatcher().run(
    () => native().dispatch_atan2_float32(y._handle, x._handle),
    (raw) => new NDArray(raw, "float32"),
  );
}

export async function argmin(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_argmin_${nd(arr.dtype)}`](arr._handle, axis),
    (raw) => wrapResult(raw, "int32"),
  );
}

export async function argmax(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_argmax_${nd(arr.dtype)}`](arr._handle, axis),
    (raw) => wrapResult(raw, "int32"),
  );
}

export async function any(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  return dispatcher().run(
    () => native().dispatch_any_int8(arr._handle, axis),
    (raw) => wrapResult(raw, "bool"),
  );
}

export async function all(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  return dispatcher().run(
    () => native().dispatch_all_int8(arr._handle, axis),
    (raw) => wrapResult(raw, "bool"),
  );
}

export async function sort(arr: NDArray): Promise<NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_sort_${nd(arr.dtype)}`](arr._handle),
    (raw) => new NDArray(raw, arr.dtype),
  );
}

export async function sort_(arr: NDArray): Promise<NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_sort_inplace_${nd(arr.dtype)}`](arr._handle),
    () => arr,
  );
}

export async function argsort(arr: NDArray): Promise<NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_argsort_${nd(arr.dtype)}`](arr._handle),
    (raw) => new NDArray(raw, "int32"),
  );
}

export async function unique(arr: NDArray): Promise<NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_unique_${nd(arr.dtype)}`](arr._handle),
    (raw) => new NDArray(raw, arr.dtype),
  );
}

export async function setUnion(a: NDArray, b: NDArray): Promise<NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_set_union_${nd(a.dtype)}`](a._handle, b._handle),
    (raw) => new NDArray(raw, a.dtype),
  );
}

export async function setIntersection(a: NDArray, b: NDArray): Promise<NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_set_intersection_${nd(a.dtype)}`](a._handle, b._handle),
    (raw) => new NDArray(raw, a.dtype),
  );
}

export async function setDifference(a: NDArray, b: NDArray): Promise<NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_set_difference_${nd(a.dtype)}`](a._handle, b._handle),
    (raw) => new NDArray(raw, a.dtype),
  );
}
