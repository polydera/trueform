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
import { IntersectOpts, buildMode } from "../intersect/sync";

import type {
  LabeledCutResult,
  LabeledCutResultWithCurves,
  IsobandsResult,
  IsobandsResultWithCurves,
  CutResult,
  CutResultWithCurves,
  MeshArrangementResult,
  MeshArrangementResultWithCurves,
  PolygonArrangementResult,
  PolygonArrangementResultWithCurves,
} from "./sync";

function wrapLabeled(raw: any): LabeledCutResult {
  return {
    mesh: new Mesh(raw.mesh),
    labels: new NDArray(raw.labels, "int8"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

function wrapLabeledWithCurves(raw: any): LabeledCutResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh),
    labels: new NDArray(raw.labels, "int8"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    curves: new Curves(raw.curves),
  };
}

function wrapIsobands(raw: any): IsobandsResult {
  return {
    mesh: new Mesh(raw.mesh),
    labels: new NDArray(raw.labels, "int32"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

function wrapIsobandsWithCurves(raw: any): IsobandsResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh),
    labels: new NDArray(raw.labels, "int32"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    curves: new Curves(raw.curves),
  };
}

function wrapCut(raw: any): CutResult {
  return {
    mesh: new Mesh(raw.mesh),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

function wrapCutWithCurves(raw: any): CutResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    curves: new Curves(raw.curves),
  };
}

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

function wrapPolygonArrangement(raw: any): PolygonArrangementResult {
  return {
    mesh: new Mesh(raw.mesh),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

function wrapPolygonArrangementWithCurves(raw: any): PolygonArrangementResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    curves: new Curves(raw.curves),
  };
}

// ============================================================================
// Booleans
// ============================================================================

export async function booleanUnion(
  m0: Mesh, m1: Mesh,
): Promise<LabeledCutResult>;
export async function booleanUnion(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true },
): Promise<LabeledCutResultWithCurves>;
export async function booleanUnion(
  m0: Mesh, m1: Mesh, opts?: { returnCurves?: true },
): Promise<LabeledCutResult | LabeledCutResultWithCurves> {
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native().dispatch_boolean_union_with_curves(m0._handle, m1._handle),
      (raw) => wrapLabeledWithCurves(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_boolean_union(m0._handle, m1._handle),
    (raw) => wrapLabeled(raw),
  );
}

export async function booleanIntersection(
  m0: Mesh, m1: Mesh,
): Promise<LabeledCutResult>;
export async function booleanIntersection(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true },
): Promise<LabeledCutResultWithCurves>;
export async function booleanIntersection(
  m0: Mesh, m1: Mesh, opts?: { returnCurves?: true },
): Promise<LabeledCutResult | LabeledCutResultWithCurves> {
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native().dispatch_boolean_intersection_with_curves(m0._handle, m1._handle),
      (raw) => wrapLabeledWithCurves(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_boolean_intersection(m0._handle, m1._handle),
    (raw) => wrapLabeled(raw),
  );
}

export async function booleanDifference(
  m0: Mesh, m1: Mesh,
): Promise<LabeledCutResult>;
export async function booleanDifference(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true },
): Promise<LabeledCutResultWithCurves>;
export async function booleanDifference(
  m0: Mesh, m1: Mesh, opts?: { returnCurves?: true },
): Promise<LabeledCutResult | LabeledCutResultWithCurves> {
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native().dispatch_boolean_difference_with_curves(m0._handle, m1._handle),
      (raw) => wrapLabeledWithCurves(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_boolean_difference(m0._handle, m1._handle),
    (raw) => wrapLabeled(raw),
  );
}

// ============================================================================
// Isobands
// ============================================================================

export async function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts?: { selectedBands?: number[] },
): Promise<IsobandsResult>;
export async function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts: { returnCurves: true, selectedBands?: number[] },
): Promise<IsobandsResultWithCurves>;
export async function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts?: { returnCurves?: true, selectedBands?: number[] },
): Promise<IsobandsResult | IsobandsResultWithCurves> {
  const sb = opts?.selectedBands;
  if (opts?.returnCurves) {
    if (sb) {
      return dispatcher().run(
        () => native().dispatch_isobands_with_curves_selected(
          mesh._handle, scalars._handle, cutValues, sb,
        ),
        (raw) => wrapIsobandsWithCurves(raw),
      );
    }
    return dispatcher().run(
      () => native().dispatch_isobands_with_curves(
        mesh._handle, scalars._handle, cutValues,
      ),
      (raw) => wrapIsobandsWithCurves(raw),
    );
  }
  if (sb) {
    return dispatcher().run(
      () => native().dispatch_isobands_selected(
        mesh._handle, scalars._handle, cutValues, sb,
      ),
      (raw) => wrapIsobands(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_isobands(mesh._handle, scalars._handle, cutValues),
    (raw) => wrapIsobands(raw),
  );
}

// ============================================================================
// Embedded intersection curves
// ============================================================================

export async function embeddedIntersectionCurves(
  m0: Mesh, m1: Mesh, opts?: IntersectOpts,
): Promise<CutResult>;
export async function embeddedIntersectionCurves(
  m0: Mesh, m1: Mesh, opts: IntersectOpts & { returnCurves: true },
): Promise<CutResultWithCurves>;
export async function embeddedIntersectionCurves(
  m0: Mesh, m1: Mesh, opts?: IntersectOpts & { returnCurves?: true },
): Promise<CutResult | CutResultWithCurves> {
  const mode = buildMode(opts, "primitives", false, false);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native().dispatch_embedded_intersection_curves_with_curves(
        m0._handle, m1._handle, mode,
      ),
      (raw) => wrapCutWithCurves(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_embedded_intersection_curves(
      m0._handle, m1._handle, mode,
    ),
    (raw) => wrapCut(raw),
  );
}

// ============================================================================
// Mesh Arrangement
// ============================================================================

export async function meshArrangements(
  meshes: Mesh[], opts?: IntersectOpts,
): Promise<MeshArrangementResult>;
export async function meshArrangements(
  meshes: Mesh[], opts: IntersectOpts & { returnCurves: true },
): Promise<MeshArrangementResultWithCurves>;
export async function meshArrangements(
  meshes: Mesh[], opts?: IntersectOpts & { returnCurves?: true },
): Promise<MeshArrangementResult | MeshArrangementResultWithCurves> {
  const rc = opts?.resolveCrossings ?? (meshes.length > 2);
  const mode = buildMode(opts, "primitives", rc, false);
  const handles = meshes.map(m => m._handle);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native().dispatch_mesh_arrangements_with_curves(handles, mode),
      (raw) => wrapArrangementWithCurves(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_mesh_arrangements(handles, mode),
    (raw) => wrapArrangement(raw),
  );
}

// ============================================================================
// Embedded self-intersection curves
// ============================================================================

export async function embeddedSelfIntersectionCurves(
  mesh: Mesh, opts?: IntersectOpts,
): Promise<CutResult>;
export async function embeddedSelfIntersectionCurves(
  mesh: Mesh, opts: IntersectOpts & { returnCurves: true },
): Promise<CutResultWithCurves>;
export async function embeddedSelfIntersectionCurves(
  mesh: Mesh, opts?: IntersectOpts & { returnCurves?: true },
): Promise<CutResult | CutResultWithCurves> {
  const mode = buildMode(opts, "primitives", true, true);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native().dispatch_embedded_self_intersection_curves_with_curves(
        mesh._handle, mode,
      ),
      (raw) => wrapCutWithCurves(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_embedded_self_intersection_curves(
      mesh._handle, mode,
    ),
    (raw) => wrapCut(raw),
  );
}

// ============================================================================
// Polygon arrangements (self-intersection arrangements)
// ============================================================================

export async function polygonArrangements(
  mesh: Mesh, opts?: IntersectOpts,
): Promise<PolygonArrangementResult>;
export async function polygonArrangements(
  mesh: Mesh, opts: IntersectOpts & { returnCurves: true },
): Promise<PolygonArrangementResultWithCurves>;
export async function polygonArrangements(
  mesh: Mesh, opts?: IntersectOpts & { returnCurves?: true },
): Promise<PolygonArrangementResult | PolygonArrangementResultWithCurves> {
  const mode = buildMode(opts, "primitives", true, true);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native().dispatch_polygon_arrangements_with_curves(
        mesh._handle, mode,
      ),
      (raw) => wrapPolygonArrangementWithCurves(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_polygon_arrangements(mesh._handle, mode),
    (raw) => wrapPolygonArrangement(raw),
  );
}
