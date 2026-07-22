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
import {
  NDArray, NDArrayInt8, NDArrayInt32, NDArrayFloat32, NDArrayFloat64,
} from "../ndarray/NDArray";
import type { FloatDtype } from "../ndarray/dtype";
import { Mesh } from "../form/Mesh";
import { Curves } from "../form/Curves";
import { IntersectOpts, buildMode, getTolerance } from "../intersect/sync";
import { assertSameDtype } from "../internal/dtype";

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
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  if (opts?.returnCurves) {
    return wrapLabeledWithCurves(
      native()[`boolean_union_with_curves_${dt}`](m0._handle, m1._handle), dt,
    );
  }
  return wrapLabeled(
    native()[`boolean_union_${dt}`](m0._handle, m1._handle), dt,
  );
}

/** Boolean intersection of two meshes. Result is the volume covered by both. */
export function booleanIntersection(m0: Mesh, m1: Mesh): LabeledCutResult;
export function booleanIntersection(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true },
): LabeledCutResultWithCurves;
export function booleanIntersection(
  m0: Mesh, m1: Mesh, opts?: { returnCurves: true },
): LabeledCutResult | LabeledCutResultWithCurves {
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  if (opts?.returnCurves) {
    return wrapLabeledWithCurves(
      native()[`boolean_intersection_with_curves_${dt}`](m0._handle, m1._handle), dt,
    );
  }
  return wrapLabeled(
    native()[`boolean_intersection_${dt}`](m0._handle, m1._handle), dt,
  );
}

/** Boolean difference: m0 minus m1. Result is m0 with m1 subtracted. */
export function booleanDifference(m0: Mesh, m1: Mesh): LabeledCutResult;
export function booleanDifference(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true },
): LabeledCutResultWithCurves;
export function booleanDifference(
  m0: Mesh, m1: Mesh, opts?: { returnCurves: true },
): LabeledCutResult | LabeledCutResultWithCurves {
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  if (opts?.returnCurves) {
    return wrapLabeledWithCurves(
      native()[`boolean_difference_with_curves_${dt}`](m0._handle, m1._handle), dt,
    );
  }
  return wrapLabeled(
    native()[`boolean_difference_${dt}`](m0._handle, m1._handle), dt,
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
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  const mode = buildMode(opts, "primitives", false, false);
  const tolerance = getTolerance(opts);
  if (opts?.returnCurves) {
    return wrapCutWithCurves(
      native()[`embedded_intersection_curves_with_curves_${dt}`](
        m0._handle, m1._handle, mode, tolerance,
      ), dt,
    );
  }
  return wrapCut(
    native()[`embedded_intersection_curves_${dt}`](
      m0._handle, m1._handle, mode, tolerance,
    ), dt,
  );
}

// ============================================================================
// Mesh Arrangement
// ============================================================================

/**
 * Intersection options plus the arrangement-only cut-surface triangulation
 * choice. Triangulation is meaningless for curve/boolean queries, so it
 * stays out of the shared {@link IntersectOpts}.
 */
export interface ArrangementOpts extends IntersectOpts {
  /**
   * Cut-surface triangulation: "cdt" (plain constrained Delaunay per cut
   * loop, default) or "refinedCdt" (quality refinement of the cut surface;
   * shared boundaries stay watertight by construction).
   */
  triangulation?: "cdt" | "refinedCdt";
}

const TRIANGULATION_MAP = { cdt: 0, refinedCdt: 1 } as const;

/** @internal */
export function getTriangulation(opts?: ArrangementOpts): number {
  return TRIANGULATION_MAP[opts?.triangulation ?? "cdt"];
}

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

/**
 * Build a mesh arrangement from N meshes.
 *
 * Splits all faces along intersection curves and merges into one mesh.
 * Each face is tagged with its source mesh and original face index.
 */
export function meshArrangements(meshes: Mesh[]): MeshArrangementResult;
export function meshArrangements(
  meshes: Mesh[], opts: ArrangementOpts & { returnCurves: true },
): MeshArrangementResultWithCurves;
export function meshArrangements(
  meshes: Mesh[], opts: ArrangementOpts,
): MeshArrangementResult;
export function meshArrangements(
  meshes: Mesh[], opts?: ArrangementOpts & { returnCurves?: true },
): MeshArrangementResult | MeshArrangementResultWithCurves {
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
    return wrapArrangementWithCurves(
      native()[`mesh_arrangements_with_curves_${dt}`](handles, mode, tolerance, triangulation), dt,
    );
  }
  return wrapArrangement(
    native()[`mesh_arrangements_${dt}`](handles, mode, tolerance, triangulation), dt,
  );
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
  const dt = mesh.dtype;
  const mode = buildMode(opts, "primitives", true, true);
  const tolerance = getTolerance(opts);
  if (opts?.returnCurves) {
    return wrapCutWithCurves(
      native()[`embedded_self_intersection_curves_with_curves_${dt}`](
        mesh._handle, mode, tolerance,
      ), dt,
    );
  }
  return wrapCut(
    native()[`embedded_self_intersection_curves_${dt}`](mesh._handle, mode, tolerance),
    dt,
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

/**
 * Decompose a mesh at its self-intersection curves.
 *
 * Splits all faces along self-intersection curves and returns
 * the subdivided mesh with per-face labels identifying original faces.
 */
export function polygonArrangements(mesh: Mesh): PolygonArrangementResult;
export function polygonArrangements(
  mesh: Mesh, opts: ArrangementOpts & { returnCurves: true },
): PolygonArrangementResultWithCurves;
export function polygonArrangements(
  mesh: Mesh, opts: ArrangementOpts,
): PolygonArrangementResult;
export function polygonArrangements(
  mesh: Mesh, opts?: ArrangementOpts & { returnCurves?: true },
): PolygonArrangementResult | PolygonArrangementResultWithCurves {
  const dt = mesh.dtype;
  const mode = buildMode(opts, "primitives", true, true);
  const tolerance = getTolerance(opts);
  const triangulation = getTriangulation(opts);
  if (opts?.returnCurves) {
    return wrapPolygonArrangementWithCurves(
      native()[`polygon_arrangements_with_curves_${dt}`](mesh._handle, mode, tolerance, triangulation),
      dt,
    );
  }
  return wrapPolygonArrangement(
    native()[`polygon_arrangements_${dt}`](mesh._handle, mode, tolerance, triangulation), dt,
  );
}
