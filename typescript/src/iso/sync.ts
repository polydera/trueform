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
import { NDArray, NDArrayInt32, NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import type { FloatDtype } from "../ndarray/dtype";
import { Mesh } from "../form/Mesh";
import { Curves } from "../form/Curves";
import { assertSameDtype } from "../internal/dtype";

/** Result of isobands slicing. */
export interface IsobandsResult {
  /** The result mesh with band regions. */
  mesh: Mesh;
  /** Per-face band ID. */
  labels: NDArrayInt32;
  /** Per-face origin: which face in the original mesh. */
  faceLabels: NDArrayInt32;
}

/** Result of isobands slicing with isocontour curves. */
export interface IsobandsResultWithCurves {
  /** The result mesh with band regions. */
  mesh: Mesh;
  /** Per-face band ID. */
  labels: NDArrayInt32;
  /** Per-face origin: which face in the original mesh. */
  faceLabels: NDArrayInt32;
  /** Isocontour polylines. */
  curves: Curves;
}

function wrapIsobands(raw: any, dt: FloatDtype): IsobandsResult {
  return {
    mesh: new Mesh(raw.mesh, dt),
    labels: new NDArray(raw.labels, "int32"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

function wrapIsobandsWithCurves(
  raw: any, dt: FloatDtype,
): IsobandsResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh, dt),
    labels: new NDArray(raw.labels, "int32"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    curves: new Curves(raw.curves, dt),
  };
}

// ============================================================================
// Isocontours
// ============================================================================

/** Extract isocontour curves from a scalar field at one or more thresholds. */
export function isocontours(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  threshold: number | Float32Array | Float64Array | number[],
): Curves {
  assertSameDtype([mesh, scalars], ["mesh", "scalars"]);
  const dt = mesh.dtype as FloatDtype;
  if (typeof threshold === "number") {
    return new Curves(
      native()[`isocontours_${dt}`](mesh._handle, scalars._handle, threshold),
      dt,
    );
  }
  return new Curves(
    native()[`isocontours_multi_${dt}`](
      mesh._handle, scalars._handle, threshold,
    ),
    dt,
  );
}

// ============================================================================
// Isobands
// ============================================================================

/**
 * Slice a mesh by scalar field isocontours into band regions.
 *
 * @param mesh - Input mesh
 * @param scalars - Per-vertex scalar values [V] (must share mesh dtype)
 * @param cutValues - Isocontour thresholds. N values create N+1 bands.
 * @param opts.selectedBands - Band indices to keep (0-indexed)
 * @param opts.returnCurves - If true, include isocontour polylines
 */
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  cutValues: Float32Array | Float64Array | number[],
): IsobandsResult;
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  cutValues: Float32Array | Float64Array | number[],
  opts: { returnCurves: true },
): IsobandsResultWithCurves;
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  cutValues: Float32Array | Float64Array | number[],
  opts: { selectedBands: number[] },
): IsobandsResult;
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  cutValues: Float32Array | Float64Array | number[],
  opts: { selectedBands: number[], returnCurves: true },
): IsobandsResultWithCurves;
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  cutValues: Float32Array | Float64Array | number[],
  opts?: { returnCurves?: boolean, selectedBands?: number[] },
): IsobandsResult | IsobandsResultWithCurves {
  assertSameDtype([mesh, scalars], ["mesh", "scalars"]);
  const dt = mesh.dtype as FloatDtype;
  const sb = opts?.selectedBands;
  if (opts?.returnCurves) {
    if (sb) {
      return wrapIsobandsWithCurves(
        native()[`isobands_with_curves_selected_${dt}`](
          mesh._handle, scalars._handle, cutValues, sb,
        ), dt,
      );
    }
    return wrapIsobandsWithCurves(
      native()[`isobands_with_curves_${dt}`](
        mesh._handle, scalars._handle, cutValues,
      ), dt,
    );
  }
  if (sb) {
    return wrapIsobands(
      native()[`isobands_selected_${dt}`](
        mesh._handle, scalars._handle, cutValues, sb,
      ), dt,
    );
  }
  return wrapIsobands(
    native()[`isobands_${dt}`](mesh._handle, scalars._handle, cutValues), dt,
  );
}
