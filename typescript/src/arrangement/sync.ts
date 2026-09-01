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
import { NDArray, NDArrayInt32 } from "../ndarray/NDArray";
import { Mesh } from "../form/Mesh";
import { Curves } from "../form/Curves";
import { IntersectOpts, buildMode, getTolerance } from "../intersect/sync";
import { assertSameDtype } from "../internal/dtype";

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
