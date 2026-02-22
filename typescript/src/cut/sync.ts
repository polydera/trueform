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
import { NDArray, NDArrayInt8, NDArrayInt32, NDArrayFloat32 } from "../ndarray/NDArray";
import { Mesh } from "../form/Mesh";
import { Curves } from "../form/Curves";

export interface LabeledCutResult {
  mesh: Mesh;
  labels: NDArrayInt8;
}

export interface LabeledCutResultWithCurves {
  mesh: Mesh;
  labels: NDArrayInt8;
  curves: Curves;
}

export interface IsobandsResult {
  mesh: Mesh;
  labels: NDArrayInt32;
}

export interface IsobandsResultWithCurves {
  mesh: Mesh;
  labels: NDArrayInt32;
  curves: Curves;
}

export interface CutResultWithCurves {
  mesh: Mesh;
  curves: Curves;
}

function wrapLabeled(raw: any): LabeledCutResult {
  return {
    mesh: new Mesh(raw.mesh),
    labels: new NDArray(raw.labels, "int8"),
  };
}

function wrapLabeledWithCurves(raw: any): LabeledCutResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh),
    labels: new NDArray(raw.labels, "int8"),
    curves: new Curves(raw.curves),
  };
}

function wrapIsobands(raw: any): IsobandsResult {
  return {
    mesh: new Mesh(raw.mesh),
    labels: new NDArray(raw.labels, "int32"),
  };
}

function wrapIsobandsWithCurves(raw: any): IsobandsResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh),
    labels: new NDArray(raw.labels, "int32"),
    curves: new Curves(raw.curves),
  };
}

function wrapCutWithCurves(raw: any): CutResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh),
    curves: new Curves(raw.curves),
  };
}

// ============================================================================
// Booleans
// ============================================================================

export function booleanUnion(m0: Mesh, m1: Mesh): LabeledCutResult;
export function booleanUnion(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true },
): LabeledCutResultWithCurves;
export function booleanUnion(
  m0: Mesh, m1: Mesh, opts?: { returnCurves: true },
): LabeledCutResult | LabeledCutResultWithCurves {
  if (opts?.returnCurves) {
    return wrapLabeledWithCurves(
      native().boolean_union_with_curves(m0._handle, m1._handle),
    );
  }
  return wrapLabeled(native().boolean_union(m0._handle, m1._handle));
}

export function booleanIntersection(m0: Mesh, m1: Mesh): LabeledCutResult;
export function booleanIntersection(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true },
): LabeledCutResultWithCurves;
export function booleanIntersection(
  m0: Mesh, m1: Mesh, opts?: { returnCurves: true },
): LabeledCutResult | LabeledCutResultWithCurves {
  if (opts?.returnCurves) {
    return wrapLabeledWithCurves(
      native().boolean_intersection_with_curves(m0._handle, m1._handle),
    );
  }
  return wrapLabeled(native().boolean_intersection(m0._handle, m1._handle));
}

export function booleanDifference(m0: Mesh, m1: Mesh): LabeledCutResult;
export function booleanDifference(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true },
): LabeledCutResultWithCurves;
export function booleanDifference(
  m0: Mesh, m1: Mesh, opts?: { returnCurves: true },
): LabeledCutResult | LabeledCutResultWithCurves {
  if (opts?.returnCurves) {
    return wrapLabeledWithCurves(
      native().boolean_difference_with_curves(m0._handle, m1._handle),
    );
  }
  return wrapLabeled(native().boolean_difference(m0._handle, m1._handle));
}

// ============================================================================
// Isobands
// ============================================================================

export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
): IsobandsResult;
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts: { returnCurves: true },
): IsobandsResultWithCurves;
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts?: { returnCurves: true },
): IsobandsResult | IsobandsResultWithCurves {
  if (opts?.returnCurves) {
    return wrapIsobandsWithCurves(
      native().isobands_with_curves(mesh._handle, scalars._handle, cutValues),
    );
  }
  return wrapIsobands(
    native().isobands(mesh._handle, scalars._handle, cutValues),
  );
}

// ============================================================================
// Embedded intersection curves
// ============================================================================

export function embeddedIntersectionCurves(m0: Mesh, m1: Mesh): Mesh;
export function embeddedIntersectionCurves(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true },
): CutResultWithCurves;
export function embeddedIntersectionCurves(
  m0: Mesh, m1: Mesh, opts?: { returnCurves: true },
): Mesh | CutResultWithCurves {
  if (opts?.returnCurves) {
    return wrapCutWithCurves(
      native().embedded_intersection_curves_with_curves(
        m0._handle, m1._handle,
      ),
    );
  }
  return new Mesh(
    native().embedded_intersection_curves(m0._handle, m1._handle),
  );
}

// ============================================================================
// Embedded self-intersection curves
// ============================================================================

export function embeddedSelfIntersectionCurves(mesh: Mesh): Mesh;
export function embeddedSelfIntersectionCurves(
  mesh: Mesh, opts: { returnCurves: true },
): CutResultWithCurves;
export function embeddedSelfIntersectionCurves(
  mesh: Mesh, opts?: { returnCurves: true },
): Mesh | CutResultWithCurves {
  if (opts?.returnCurves) {
    return wrapCutWithCurves(
      native().embedded_self_intersection_curves_with_curves(mesh._handle),
    );
  }
  return new Mesh(
    native().embedded_self_intersection_curves(mesh._handle),
  );
}
