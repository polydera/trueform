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
import { IntersectOpts, buildMode } from "../intersect/sync";

/** Result of a boolean operation. */
export interface LabeledCutResult {
  /** The result mesh. */
  mesh: Mesh;
  /** Per-face region labels. */
  labels: NDArrayInt8;
  /** Per-face origin: which face in the original mesh. */
  faceLabels: NDArrayInt32;
}

/** Result of a boolean operation with intersection curves. */
export interface LabeledCutResultWithCurves {
  /** The result mesh. */
  mesh: Mesh;
  /** Per-face region labels. */
  labels: NDArrayInt8;
  /** Per-face origin: which face in the original mesh. */
  faceLabels: NDArrayInt32;
  /** Intersection curves. */
  curves: Curves;
}

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

/** Result of embedded intersection. */
export interface CutResult {
  /** The mesh with embedded intersection edges. */
  mesh: Mesh;
  /** Per-face origin: which face in the original mesh. */
  faceLabels: NDArrayInt32;
}

/** Result of embedded intersection with curves. */
export interface CutResultWithCurves {
  /** The mesh with embedded intersection edges. */
  mesh: Mesh;
  /** Per-face origin: which face in the original mesh. */
  faceLabels: NDArrayInt32;
  /** Intersection curves. */
  curves: Curves;
}

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

// ============================================================================
// Booleans
// ============================================================================

/** Boolean union of two meshes. Result is the volume covered by either mesh. */
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

/** Boolean intersection of two meshes. Result is the volume covered by both. */
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

/** Boolean difference: m0 minus m1. Result is m0 with m1 subtracted. */
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

/**
 * Slice a mesh by scalar field isocontours into band regions.
 *
 * @param mesh - Input mesh
 * @param scalars - Per-vertex scalar values [V]
 * @param cutValues - Isocontour thresholds. N values create N+1 bands.
 * @param opts.selectedBands - Band indices to keep (0-indexed)
 * @param opts.returnCurves - If true, include isocontour polylines
 */
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
): IsobandsResult;
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts: { returnCurves: true },
): IsobandsResultWithCurves;
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts: { selectedBands: number[] },
): IsobandsResult;
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts: { selectedBands: number[], returnCurves: true },
): IsobandsResultWithCurves;
export function isobands(
  mesh: Mesh, scalars: NDArrayFloat32, cutValues: Float32Array,
  opts?: { returnCurves?: boolean, selectedBands?: number[] },
): IsobandsResult | IsobandsResultWithCurves {
  const sb = opts?.selectedBands;
  if (opts?.returnCurves) {
    if (sb) {
      return wrapIsobandsWithCurves(
        native().isobands_with_curves_selected(mesh._handle, scalars._handle, cutValues, sb),
      );
    }
    return wrapIsobandsWithCurves(
      native().isobands_with_curves(mesh._handle, scalars._handle, cutValues),
    );
  }
  if (sb) {
    return wrapIsobands(
      native().isobands_selected(mesh._handle, scalars._handle, cutValues, sb),
    );
  }
  return wrapIsobands(
    native().isobands(mesh._handle, scalars._handle, cutValues),
  );
}

// ============================================================================
// Embedded intersection curves
// ============================================================================

/** Embed intersection curves of m0 and m1 as edges in m0. */
export function embeddedIntersectionCurves(m0: Mesh, m1: Mesh): CutResult;
export function embeddedIntersectionCurves(
  m0: Mesh, m1: Mesh, opts: IntersectOpts & { returnCurves: true },
): CutResultWithCurves;
export function embeddedIntersectionCurves(
  m0: Mesh, m1: Mesh, opts: IntersectOpts,
): CutResult;
export function embeddedIntersectionCurves(
  m0: Mesh, m1: Mesh, opts?: IntersectOpts & { returnCurves?: true },
): CutResult | CutResultWithCurves {
  const mode = buildMode(opts, "primitives", false, false);
  if (opts?.returnCurves) {
    return wrapCutWithCurves(
      native().embedded_intersection_curves_with_curves(
        m0._handle, m1._handle, mode,
      ),
    );
  }
  return wrapCut(
    native().embedded_intersection_curves(m0._handle, m1._handle, mode),
  );
}

// ============================================================================
// Mesh Arrangement
// ============================================================================

/** Result of mesh arrangement. */
export interface MeshArrangementResult {
  /** The merged mesh with all faces split along intersection curves. */
  mesh: Mesh;
  /** Per-face tag: which input mesh each face came from. */
  tagLabels: NDArrayInt32;
  /** Per-face origin: which face in the original mesh. */
  faceLabels: NDArrayInt32;
}

/** Result of mesh arrangement with intersection curves. */
export interface MeshArrangementResultWithCurves {
  /** The merged mesh. */
  mesh: Mesh;
  /** Per-face tag. */
  tagLabels: NDArrayInt32;
  /** Per-face origin. */
  faceLabels: NDArrayInt32;
  /** Intersection curves. */
  curves: Curves;
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

/**
 * Build a mesh arrangement from N meshes.
 *
 * Splits all faces along intersection curves and merges into one mesh.
 * Each face is tagged with its source mesh and original face index.
 */
export function meshArrangements(meshes: Mesh[]): MeshArrangementResult;
export function meshArrangements(
  meshes: Mesh[], opts: IntersectOpts & { returnCurves: true },
): MeshArrangementResultWithCurves;
export function meshArrangements(
  meshes: Mesh[], opts: IntersectOpts,
): MeshArrangementResult;
export function meshArrangements(
  meshes: Mesh[], opts?: IntersectOpts & { returnCurves?: true },
): MeshArrangementResult | MeshArrangementResultWithCurves {
  const rc = opts?.resolveCrossings ?? (meshes.length > 2);
  const mode = buildMode(opts, "primitives", rc, false);
  const handles = meshes.map(m => m._handle);
  if (opts?.returnCurves) {
    return wrapArrangementWithCurves(
      native().mesh_arrangements_with_curves(handles, mode),
    );
  }
  return wrapArrangement(native().mesh_arrangements(handles, mode));
}

// ============================================================================
// Embedded self-intersection curves
// ============================================================================

/** Detect and resolve self-intersections, splitting faces along intersection curves. */
export function embeddedSelfIntersectionCurves(mesh: Mesh): CutResult;
export function embeddedSelfIntersectionCurves(
  mesh: Mesh, opts: IntersectOpts & { returnCurves: true },
): CutResultWithCurves;
export function embeddedSelfIntersectionCurves(
  mesh: Mesh, opts: IntersectOpts,
): CutResult;
export function embeddedSelfIntersectionCurves(
  mesh: Mesh, opts?: IntersectOpts & { returnCurves?: true },
): CutResult | CutResultWithCurves {
  const mode = buildMode(opts, "primitives", true, true);
  if (opts?.returnCurves) {
    return wrapCutWithCurves(
      native().embedded_self_intersection_curves_with_curves(mesh._handle, mode),
    );
  }
  return wrapCut(
    native().embedded_self_intersection_curves(mesh._handle, mode),
  );
}

// ============================================================================
// Polygon arrangements (self-intersection arrangements)
// ============================================================================

/** Result of polygon arrangement (self-intersection decomposition). */
export interface PolygonArrangementResult {
  /** The split mesh with faces subdivided at self-intersection curves. */
  mesh: Mesh;
  /** Per-face origin: which original face each output face came from. */
  faceLabels: NDArrayInt32;
}

/** Result of polygon arrangement with intersection curves. */
export interface PolygonArrangementResultWithCurves {
  /** The split mesh. */
  mesh: Mesh;
  /** Per-face origin. */
  faceLabels: NDArrayInt32;
  /** Self-intersection curves. */
  curves: Curves;
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

/**
 * Decompose a mesh at its self-intersection curves.
 *
 * Splits all faces along self-intersection curves and returns
 * the subdivided mesh with per-face labels identifying original faces.
 */
export function polygonArrangements(mesh: Mesh): PolygonArrangementResult;
export function polygonArrangements(
  mesh: Mesh, opts: IntersectOpts & { returnCurves: true },
): PolygonArrangementResultWithCurves;
export function polygonArrangements(
  mesh: Mesh, opts: IntersectOpts,
): PolygonArrangementResult;
export function polygonArrangements(
  mesh: Mesh, opts?: IntersectOpts & { returnCurves?: true },
): PolygonArrangementResult | PolygonArrangementResultWithCurves {
  const mode = buildMode(opts, "primitives", true, true);
  if (opts?.returnCurves) {
    return wrapPolygonArrangementWithCurves(
      native().polygon_arrangements_with_curves(mesh._handle, mode),
    );
  }
  return wrapPolygonArrangement(
    native().polygon_arrangements(mesh._handle, mode),
  );
}
