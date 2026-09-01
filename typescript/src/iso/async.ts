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
import { NDArray, NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import type { FloatDtype } from "../ndarray/dtype";
import { Mesh } from "../form/Mesh";
import { Curves } from "../form/Curves";
import { assertSameDtype } from "../internal/dtype";

import type { IsobandsResult, IsobandsResultWithCurves } from "./sync";

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

/** Extract isocontour curves at one or more thresholds, off the main thread. */
export async function isocontours(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  threshold: number | Float32Array | Float64Array | number[],
): Promise<Curves> {
  assertSameDtype([mesh, scalars], ["mesh", "scalars"]);
  const dt = mesh.dtype as FloatDtype;
  if (typeof threshold === "number") {
    return dispatcher().run(
      () =>
        native()[`dispatch_isocontours_${dt}`](
          mesh._handle, scalars._handle, threshold,
        ),
      (raw) => new Curves(raw, dt),
    );
  }
  return dispatcher().run(
    () =>
      native()[`dispatch_isocontours_multi_${dt}`](
        mesh._handle, scalars._handle, threshold,
      ),
    (raw) => new Curves(raw, dt),
  );
}

// ============================================================================
// Isobands
// ============================================================================

export async function isobands(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  cutValues: Float32Array | Float64Array | number[],
  opts: { returnCurves: true, selectedBands?: number[] },
): Promise<IsobandsResultWithCurves>;
export async function isobands(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  cutValues: Float32Array | Float64Array | number[],
  opts?: { selectedBands?: number[] },
): Promise<IsobandsResult>;
export async function isobands(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  cutValues: Float32Array | Float64Array | number[],
  opts?: { returnCurves?: true, selectedBands?: number[] },
): Promise<IsobandsResult | IsobandsResultWithCurves> {
  assertSameDtype([mesh, scalars], ["mesh", "scalars"]);
  const dt = mesh.dtype as FloatDtype;
  const sb = opts?.selectedBands;
  if (opts?.returnCurves) {
    if (sb) {
      return dispatcher().run(
        () => native()[`dispatch_isobands_with_curves_selected_${dt}`](
          mesh._handle, scalars._handle, cutValues, sb,
        ),
        (raw) => wrapIsobandsWithCurves(raw, dt),
      );
    }
    return dispatcher().run(
      () => native()[`dispatch_isobands_with_curves_${dt}`](
        mesh._handle, scalars._handle, cutValues,
      ),
      (raw) => wrapIsobandsWithCurves(raw, dt),
    );
  }
  if (sb) {
    return dispatcher().run(
      () => native()[`dispatch_isobands_selected_${dt}`](
        mesh._handle, scalars._handle, cutValues, sb,
      ),
      (raw) => wrapIsobands(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_isobands_${dt}`](
      mesh._handle, scalars._handle, cutValues,
    ),
    (raw) => wrapIsobands(raw, dt),
  );
}
