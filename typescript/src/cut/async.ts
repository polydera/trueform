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
import { IntersectOpts, buildMode, getTolerance } from "../intersect/sync";
import { ArrangementOpts, getTriangulation } from "./sync";
import { assertSameDtype } from "../internal/dtype";

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

function wrapLabeled(raw: any, dt: "float32" | "float64"): LabeledCutResult {
  return {
    mesh: new Mesh(raw.mesh, dt),
    labels: new NDArray(raw.labels, "int8"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

function wrapLabeledWithCurves(
  raw: any, dt: "float32" | "float64",
): LabeledCutResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh, dt),
    labels: new NDArray(raw.labels, "int8"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    curves: new Curves(raw.curves, dt),
  };
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

function wrapCut(raw: any, dt: "float32" | "float64"): CutResult {
  return {
    mesh: new Mesh(raw.mesh, dt),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

function wrapCutWithCurves(
  raw: any, dt: "float32" | "float64",
): CutResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh, dt),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    curves: new Curves(raw.curves, dt),
  };
}

function wrapArrangement(
  raw: any, dt: "float32" | "float64",
): MeshArrangementResult {
  return {
    mesh: new Mesh(raw.mesh, dt),
    tagLabels: new NDArray(raw.tagLabels, "int32"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

function wrapArrangementWithCurves(
  raw: any, dt: "float32" | "float64",
): MeshArrangementResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh, dt),
    tagLabels: new NDArray(raw.tagLabels, "int32"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    curves: new Curves(raw.curves, dt),
  };
}

function wrapPolygonArrangement(
  raw: any, dt: "float32" | "float64",
): PolygonArrangementResult {
  return {
    mesh: new Mesh(raw.mesh, dt),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

function wrapPolygonArrangementWithCurves(
  raw: any, dt: "float32" | "float64",
): PolygonArrangementResultWithCurves {
  return {
    mesh: new Mesh(raw.mesh, dt),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    curves: new Curves(raw.curves, dt),
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
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native()[`dispatch_boolean_union_with_curves_${dt}`](m0._handle, m1._handle),
      (raw) => wrapLabeledWithCurves(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_boolean_union_${dt}`](m0._handle, m1._handle),
    (raw) => wrapLabeled(raw, dt),
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
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native()[`dispatch_boolean_intersection_with_curves_${dt}`](m0._handle, m1._handle),
      (raw) => wrapLabeledWithCurves(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_boolean_intersection_${dt}`](m0._handle, m1._handle),
    (raw) => wrapLabeled(raw, dt),
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
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native()[`dispatch_boolean_difference_with_curves_${dt}`](m0._handle, m1._handle),
      (raw) => wrapLabeledWithCurves(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_boolean_difference_${dt}`](m0._handle, m1._handle),
    (raw) => wrapLabeled(raw, dt),
  );
}

// ============================================================================
// Isobands
// ============================================================================

export async function isobands(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  cutValues: Float32Array | Float64Array | number[],
  opts?: { selectedBands?: number[] },
): Promise<IsobandsResult>;
export async function isobands(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  cutValues: Float32Array | Float64Array | number[],
  opts: { returnCurves: true, selectedBands?: number[] },
): Promise<IsobandsResultWithCurves>;
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
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  const mode = buildMode(opts, "primitives", false, false);
  const tolerance = getTolerance(opts);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native()[`dispatch_embedded_intersection_curves_with_curves_${dt}`](
        m0._handle, m1._handle, mode, tolerance,
      ),
      (raw) => wrapCutWithCurves(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_embedded_intersection_curves_${dt}`](
      m0._handle, m1._handle, mode, tolerance,
    ),
    (raw) => wrapCut(raw, dt),
  );
}

// ============================================================================
// Mesh Arrangement
// ============================================================================

export async function meshArrangements(
  meshes: Mesh[], opts?: ArrangementOpts,
): Promise<MeshArrangementResult>;
export async function meshArrangements(
  meshes: Mesh[], opts: ArrangementOpts & { returnCurves: true },
): Promise<MeshArrangementResultWithCurves>;
export async function meshArrangements(
  meshes: Mesh[], opts?: ArrangementOpts & { returnCurves?: true },
): Promise<MeshArrangementResult | MeshArrangementResultWithCurves> {
  assertSameDtype(
    meshes,
    meshes.map((_, i) => `meshes[${i}]`),
  );
  const dt = meshes[0].dtype;
  const rc = opts?.resolveCrossings ?? (meshes.length > 2);
  const mode = buildMode(opts, "primitives", rc, false);
  const tolerance = getTolerance(opts);
  const triangulation = getTriangulation(opts);
  const handles = meshes.map(m => m._handle);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native()[`dispatch_mesh_arrangements_with_curves_${dt}`](handles, mode, tolerance, triangulation),
      (raw) => wrapArrangementWithCurves(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_mesh_arrangements_${dt}`](handles, mode, tolerance, triangulation),
    (raw) => wrapArrangement(raw, dt),
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
  const dt = mesh.dtype;
  const mode = buildMode(opts, "primitives", true, true);
  const tolerance = getTolerance(opts);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native()[
        `dispatch_embedded_self_intersection_curves_with_curves_${dt}`
      ](mesh._handle, mode, tolerance),
      (raw) => wrapCutWithCurves(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_embedded_self_intersection_curves_${dt}`](
      mesh._handle, mode, tolerance,
    ),
    (raw) => wrapCut(raw, dt),
  );
}

// ============================================================================
// Polygon arrangements (self-intersection arrangements)
// ============================================================================

export async function polygonArrangements(
  mesh: Mesh, opts?: ArrangementOpts,
): Promise<PolygonArrangementResult>;
export async function polygonArrangements(
  mesh: Mesh, opts: ArrangementOpts & { returnCurves: true },
): Promise<PolygonArrangementResultWithCurves>;
export async function polygonArrangements(
  mesh: Mesh, opts?: ArrangementOpts & { returnCurves?: true },
): Promise<PolygonArrangementResult | PolygonArrangementResultWithCurves> {
  const dt = mesh.dtype;
  const mode = buildMode(opts, "primitives", true, true);
  const tolerance = getTolerance(opts);
  const triangulation = getTriangulation(opts);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native()[`dispatch_polygon_arrangements_with_curves_${dt}`](
        mesh._handle, mode, tolerance, triangulation,
      ),
      (raw) => wrapPolygonArrangementWithCurves(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_polygon_arrangements_${dt}`](mesh._handle, mode, tolerance, triangulation),
    (raw) => wrapPolygonArrangement(raw, dt),
  );
}
