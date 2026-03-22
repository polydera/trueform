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
import { NDArray, NDArrayFloat32 } from "../ndarray/NDArray";
import { Mesh } from "../form/Mesh";
import { Curves } from "../form/Curves";

import type {
  LabeledCutResult,
  LabeledCutResultWithCurves,
  IsobandsResult,
  IsobandsResultWithCurves,
  CutResultWithCurves,
} from "./sync";

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

export async function booleanUnion(
  m0: Mesh, m1: Mesh,
): Promise<LabeledCutResult> {
  return dispatcher().run(
    () => native().dispatch_boolean_union(m0._handle, m1._handle),
    (raw) => wrapLabeled(raw),
  );
}

export async function booleanUnionWithCurves(
  m0: Mesh, m1: Mesh,
): Promise<LabeledCutResultWithCurves> {
  return dispatcher().run(
    () => native().dispatch_boolean_union_with_curves(m0._handle, m1._handle),
    (raw) => wrapLabeledWithCurves(raw),
  );
}

export async function booleanIntersection(
  m0: Mesh, m1: Mesh,
): Promise<LabeledCutResult> {
  return dispatcher().run(
    () => native().dispatch_boolean_intersection(m0._handle, m1._handle),
    (raw) => wrapLabeled(raw),
  );
}

export async function booleanIntersectionWithCurves(
  m0: Mesh, m1: Mesh,
): Promise<LabeledCutResultWithCurves> {
  return dispatcher().run(
    () =>
      native().dispatch_boolean_intersection_with_curves(
        m0._handle, m1._handle,
      ),
    (raw) => wrapLabeledWithCurves(raw),
  );
}

export async function booleanDifference(
  m0: Mesh, m1: Mesh,
): Promise<LabeledCutResult> {
  return dispatcher().run(
    () => native().dispatch_boolean_difference(m0._handle, m1._handle),
    (raw) => wrapLabeled(raw),
  );
}

export async function booleanDifferenceWithCurves(
  m0: Mesh, m1: Mesh,
): Promise<LabeledCutResultWithCurves> {
  return dispatcher().run(
    () =>
      native().dispatch_boolean_difference_with_curves(
        m0._handle, m1._handle,
      ),
    (raw) => wrapLabeledWithCurves(raw),
  );
}

// ============================================================================
// Isobands
// ============================================================================

export async function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts?: { selectedBands?: number[] },
): Promise<IsobandsResult> {
  const sb = opts?.selectedBands;
  if (sb) {
    return dispatcher().run(
      () => native().dispatch_isobands_selected(mesh._handle, scalars._handle, cutValues, sb),
      (raw) => wrapIsobands(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_isobands(mesh._handle, scalars._handle, cutValues),
    (raw) => wrapIsobands(raw),
  );
}

export async function isobandsWithCurves(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts?: { selectedBands?: number[] },
): Promise<IsobandsResultWithCurves> {
  const sb = opts?.selectedBands;
  if (sb) {
    return dispatcher().run(
      () => native().dispatch_isobands_with_curves_selected(
        mesh._handle, scalars._handle, cutValues, sb,
      ),
      (raw) => wrapIsobandsWithCurves(raw),
    );
  }
  return dispatcher().run(
    () =>
      native().dispatch_isobands_with_curves(
        mesh._handle, scalars._handle, cutValues,
      ),
    (raw) => wrapIsobandsWithCurves(raw),
  );
}

// ============================================================================
// Embedded intersection curves
// ============================================================================

export async function embeddedIntersectionCurves(
  m0: Mesh, m1: Mesh,
): Promise<Mesh> {
  return dispatcher().run(
    () =>
      native().dispatch_embedded_intersection_curves(m0._handle, m1._handle),
    (raw) => new Mesh(raw),
  );
}

export async function embeddedIntersectionCurvesWithCurves(
  m0: Mesh, m1: Mesh,
): Promise<CutResultWithCurves> {
  return dispatcher().run(
    () =>
      native().dispatch_embedded_intersection_curves_with_curves(
        m0._handle, m1._handle,
      ),
    (raw) => wrapCutWithCurves(raw),
  );
}

// ============================================================================
// Mesh Arrangement
// ============================================================================

import type {
  MeshArrangementResult,
  MeshArrangementResultWithCurves,
} from "./sync";

function wrapArrangement(raw: any): MeshArrangementResult {
  return {
    mesh: new Mesh(raw.mesh),
    tagLabels: new NDArray(raw.tagLabels, "int32"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

function wrapArrangementWithCurves(raw: any): MeshArrangementResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh),
    tagLabels: new NDArray(raw.tagLabels, "int32"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    curves: new Curves(raw.curves),
  };
}

export async function meshArrangement(
  meshes: Mesh[],
): Promise<MeshArrangementResult>;
export async function meshArrangement(
  meshes: Mesh[], opts: { returnCurves: true },
): Promise<MeshArrangementResultWithCurves>;
export async function meshArrangement(
  meshes: Mesh[], opts?: { returnCurves: true },
): Promise<MeshArrangementResult | MeshArrangementResultWithCurves> {
  const handles = meshes.map(m => m._handle);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native().dispatch_mesh_arrangement_with_curves(handles),
      (raw) => wrapArrangementWithCurves(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_mesh_arrangement(handles),
    (raw) => wrapArrangement(raw),
  );
}

// ============================================================================
// Embedded self-intersection curves
// ============================================================================

export async function embeddedSelfIntersectionCurves(
  mesh: Mesh,
): Promise<Mesh> {
  return dispatcher().run(
    () =>
      native().dispatch_embedded_self_intersection_curves(mesh._handle),
    (raw) => new Mesh(raw),
  );
}

export async function embeddedSelfIntersectionCurvesWithCurves(
  mesh: Mesh,
): Promise<CutResultWithCurves> {
  return dispatcher().run(
    () =>
      native().dispatch_embedded_self_intersection_curves_with_curves(
        mesh._handle,
      ),
    (raw) => wrapCutWithCurves(raw),
  );
}
