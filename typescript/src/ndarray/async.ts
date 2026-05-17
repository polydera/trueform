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
import { NDArray, NDArrayInt32, NDArrayFloat32 } from "./NDArray";
import type {
  BincountOptions,
  HistogramOptions,
  HistogramResult,
} from "./histogram";

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

export async function sum(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  const outDtype = arr.dtype === "float32" || arr.dtype === "float64"
    ? arr.dtype : "int32";
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
    (raw) => wrapResult(raw, floatDtype(arr.dtype)),
  );
}

export async function norm(
  arr: NDArray,
  axis: number = -1,
): Promise<number | NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_norm_${nd(arr.dtype)}`](arr._handle, axis),
    (raw) => wrapResult(raw, floatDtype(arr.dtype)),
  );
}

export async function atan2(y: NDArray, x: NDArray): Promise<NDArray> {
  return dispatcher().run(
    () => native()[`dispatch_atan2_${y.dtype}`](y._handle, x._handle),
    (raw) => new NDArray(raw, y.dtype),
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

// ============================================================================
// bincount / histogram
// ============================================================================

const wrapHistInt = (raw: any): HistogramResult => ({
  counts: new NDArray(raw.counts, "int32") as NDArrayInt32,
  edges: new NDArray(raw.edges, "float32") as NDArrayFloat32,
});

const wrapHistFloat = (raw: any): HistogramResult => ({
  counts: new NDArray(raw.counts, "float32") as NDArrayFloat32,
  edges: new NDArray(raw.edges, "float32") as NDArrayFloat32,
});

const asInt32 = (a: NDArray): NDArrayInt32 =>
  (a.dtype === "int32" ? a : a.as("int32")) as NDArrayInt32;

const asFloat32 = (a: NDArray): NDArrayFloat32 =>
  (a.dtype === "float32" ? a : a.as("float32")) as NDArrayFloat32;

export async function bincount(
  x: NDArray,
  opts?: BincountOptions,
): Promise<NDArrayInt32 | NDArrayFloat32> {
  const safe = asInt32(x);
  const minLen = opts?.minLength ?? 0;
  if (opts?.weights) {
    const w = asFloat32(opts.weights);
    return dispatcher().run(
      () => native().dispatch_bincount_weighted_int32(
        safe._handle, w._handle, minLen),
      (raw) => new NDArray(raw, "float32") as NDArrayFloat32,
    );
  }
  return dispatcher().run(
    () => native().dispatch_bincount_int32(safe._handle, minLen),
    (raw) => new NDArray(raw, "int32") as NDArrayInt32,
  );
}

export async function histogram(
  x: NDArray,
  opts?: HistogramOptions,
): Promise<HistogramResult> {
  const safe = asFloat32(x);
  const bins = opts?.bins ?? 10;
  const weights = opts?.weights ? asFloat32(opts.weights) : undefined;
  const density = opts?.density ?? false;
  const n = native();
  const d = dispatcher();

  if (typeof bins === "number") {
    const [lo, hi] = opts?.range ?? [NaN, NaN];
    if (density)
      return d.run(
        () => n.dispatch_histogram_density_equal_width_float32(
          safe._handle, bins, lo, hi, weights?._handle),
        wrapHistFloat,
      );
    if (weights)
      return d.run(
        () => n.dispatch_histogram_equal_width_weighted_float32(
          safe._handle, weights._handle, bins, lo, hi),
        wrapHistFloat,
      );
    return d.run(
      () => n.dispatch_histogram_equal_width_float32(
        safe._handle, bins, lo, hi),
      wrapHistInt,
    );
  }

  const edges = asFloat32(bins);
  if (density)
    return d.run(
      () => n.dispatch_histogram_density_edges_float32(
        safe._handle, edges._handle, weights?._handle),
      wrapHistFloat,
    );
  if (weights)
    return d.run(
      () => n.dispatch_histogram_edges_weighted_float32(
        safe._handle, weights._handle, edges._handle),
      wrapHistFloat,
    );
  return d.run(
    () => n.dispatch_histogram_edges_float32(safe._handle, edges._handle),
    wrapHistInt,
  );
}
