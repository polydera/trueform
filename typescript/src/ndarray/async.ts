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
import { coerce, disposeOwned } from "../internal/owned";
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

export async function bincount(
  x: NDArray,
  opts?: BincountOptions,
): Promise<NDArrayInt32 | NDArrayFloat32> {
  const owned: NDArray[] = [];
  try {
    const safe = coerce(x, "int32", owned);
    const minLen = opts?.minLength ?? 0;
    if (opts?.weights) {
      const w = coerce(opts.weights, "float32", owned);
      return await dispatcher().run(
        () => native().dispatch_bincount_weighted_int32(
          safe._handle, w._handle, minLen),
        (raw) => new NDArray(raw, "float32") as NDArrayFloat32,
      );
    }
    return await dispatcher().run(
      () => native().dispatch_bincount_int32(safe._handle, minLen),
      (raw) => new NDArray(raw, "int32") as NDArrayInt32,
    );
  } finally {
    disposeOwned(owned);
  }
}

export async function histogram(
  x: NDArray,
  opts?: HistogramOptions,
): Promise<HistogramResult> {
  const owned: NDArray[] = [];
  try {
    const safe = coerce(x, "float32", owned);
    const bins = opts?.bins ?? 10;
    const weights = opts?.weights ? coerce(opts.weights, "float32", owned) : undefined;
    const density = opts?.density ?? false;
    const n = native();
    const d = dispatcher();

    if (typeof bins === "number") {
      const [lo, hi] = opts?.range ?? [NaN, NaN];
      if (density)
        return await d.run(
          () => n.dispatch_histogram_density_equal_width_float32(
            safe._handle, bins, lo, hi, weights?._handle),
          wrapHistFloat,
        );
      if (weights)
        return await d.run(
          () => n.dispatch_histogram_equal_width_weighted_float32(
            safe._handle, weights._handle, bins, lo, hi),
          wrapHistFloat,
        );
      return await d.run(
        () => n.dispatch_histogram_equal_width_float32(
          safe._handle, bins, lo, hi),
        wrapHistInt,
      );
    }

    const edges = coerce(bins, "float32", owned);
    if (density)
      return await d.run(
        () => n.dispatch_histogram_density_edges_float32(
          safe._handle, edges._handle, weights?._handle),
        wrapHistFloat,
      );
    if (weights)
      return await d.run(
        () => n.dispatch_histogram_edges_weighted_float32(
          safe._handle, weights._handle, edges._handle),
        wrapHistFloat,
      );
    return await d.run(
      () => n.dispatch_histogram_edges_float32(safe._handle, edges._handle),
      wrapHistInt,
    );
  } finally {
    disposeOwned(owned);
  }
}
