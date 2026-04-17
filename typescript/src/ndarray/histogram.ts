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
import { NDArray, NDArrayInt32, NDArrayFloat32 } from "./NDArray";

// ============================================================================
// Types
// ============================================================================

/** Options for {@link bincount}. */
export interface BincountOptions {
  /** Optional weights. Any numeric NDArray of the same length as x; coerced to float32. */
  weights?: NDArray;
  /** Minimum output length; padded with zeros. Default 0. */
  minLength?: number;
}

/** Options for {@link histogram}. */
export interface HistogramOptions {
  /**
   * Either the number of equal-width bins (default 10) or explicit
   * bin edges as a monotone NDArray of length `num_bins + 1` (coerced to
   * float32).
   */
  bins?: number | NDArray;
  /** Lower/upper bounds for equal-width binning. Auto (min, max) if omitted. */
  range?: [number, number];
  /** Optional weights. Any numeric NDArray of the same length as x; coerced to float32. */
  weights?: NDArray;
  /** If true, normalize to a PDF so the integral is 1. Default false. */
  density?: boolean;
}

/** Result of {@link histogram}. */
export interface HistogramResult {
  /** Bin counts. int32 if unweighted and non-density; float32 otherwise. */
  counts: NDArrayInt32 | NDArrayFloat32;
  /** Bin edges, length `counts.length + 1`. Always float32. */
  edges: NDArrayFloat32;
}

// ============================================================================
// Internal result wrappers
// ============================================================================

function wrapInt(raw: any): HistogramResult {
  return {
    counts: new NDArray(raw.counts, "int32") as NDArrayInt32,
    edges: new NDArray(raw.edges, "float32") as NDArrayFloat32,
  };
}

function wrapFloat(raw: any): HistogramResult {
  return {
    counts: new NDArray(raw.counts, "float32") as NDArrayFloat32,
    edges: new NDArray(raw.edges, "float32") as NDArrayFloat32,
  };
}

function coerceInt32(a: NDArray): NDArrayInt32 {
  return (a.dtype === "int32" ? a : a.as("int32")) as NDArrayInt32;
}

function coerceFloat32(a: NDArray): NDArrayFloat32 {
  return (a.dtype === "float32" ? a : a.as("float32")) as NDArrayFloat32;
}

// ============================================================================
// bincount
// ============================================================================

/**
 * Count occurrences of each non-negative integer in `x`.
 *
 * `result[i]` is the number of times `i` appears in `x`. Output length is
 * `max(max(x) + 1, minLength)`. With `weights`, the result holds the sum of
 * weights at each index (float32). Throws on negative values.
 *
 * Any integer-valued NDArray is accepted — coerced to int32 internally.
 */
export function bincount(
  x: NDArray,
  opts?: BincountOptions,
): NDArrayInt32 | NDArrayFloat32 {
  const safe = coerceInt32(x);
  const minLen = opts?.minLength ?? 0;
  if (opts?.weights) {
    const w = coerceFloat32(opts.weights);
    return new NDArray(
      native().bincount_weighted_int32(safe._handle, w._handle, minLen),
      "float32",
    ) as NDArrayFloat32;
  }
  return new NDArray(
    native().bincount_int32(safe._handle, minLen),
    "int32",
  ) as NDArrayInt32;
}

// ============================================================================
// histogram
// ============================================================================

/**
 * Compute the histogram of `x`. Semantics match NumPy's `np.histogram`:
 *
 * - `bins` as int → equal-width bins over `range` (auto-min/max if omitted).
 * - `bins` as NDArray → explicit monotone bin edges (length num_bins + 1).
 * - `weights` → sum of weights per bin instead of counting.
 * - `density` → normalize so the integral is 1 (PDF).
 * - NaN and out-of-range values are dropped. Upper edge is inclusive.
 *
 * Any numeric NDArray is accepted — coerced to float32 internally.
 *
 * Returns `{ counts, edges }`. `counts` is int32 unless `weights` or
 * `density` is set, in which case it's float32.
 */
export function histogram(
  x: NDArray,
  opts?: HistogramOptions,
): HistogramResult {
  const safe = coerceFloat32(x);
  const bins = opts?.bins ?? 10;
  const weights = opts?.weights ? coerceFloat32(opts.weights) : undefined;
  const density = opts?.density ?? false;
  const n = native();

  if (typeof bins === "number") {
    const [lo, hi] = opts?.range ?? [NaN, NaN];
    if (density)
      return wrapFloat(n.histogram_density_equal_width_float32(
        safe._handle, bins, lo, hi, weights?._handle));
    if (weights)
      return wrapFloat(n.histogram_equal_width_weighted_float32(
        safe._handle, weights._handle, bins, lo, hi));
    return wrapInt(n.histogram_equal_width_float32(safe._handle, bins, lo, hi));
  }

  const edges = coerceFloat32(bins);
  if (density)
    return wrapFloat(n.histogram_density_edges_float32(
      safe._handle, edges._handle, weights?._handle));
  if (weights)
    return wrapFloat(n.histogram_edges_weighted_float32(
      safe._handle, weights._handle, edges._handle));
  return wrapInt(n.histogram_edges_float32(safe._handle, edges._handle));
}
