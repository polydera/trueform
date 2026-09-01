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
import { NDArray } from "../ndarray/NDArray";
import { Mesh } from "../form/Mesh";
import { Curves } from "../form/Curves";
import { buildMode, getTolerance } from "../intersect/sync";
import { ArrangementOpts, getTriangulation } from "./sync";
import { assertSameDtype } from "../internal/dtype";

import type {
  MeshArrangementResult,
  MeshArrangementResultWithCurves,
  PolygonArrangementResult,
  PolygonArrangementResultWithCurves,
} from "./sync";

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
// Mesh Arrangement
// ============================================================================

export async function meshArrangements(
  meshes: Mesh[], opts: ArrangementOpts & { returnCurves: true },
): Promise<MeshArrangementResultWithCurves>;
export async function meshArrangements(
  meshes: Mesh[], opts?: ArrangementOpts,
): Promise<MeshArrangementResult>;
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
// Polygon arrangements (self-intersection arrangements)
// ============================================================================

export async function polygonArrangements(
  mesh: Mesh, opts: ArrangementOpts & { returnCurves: true },
): Promise<PolygonArrangementResultWithCurves>;
export async function polygonArrangements(
  mesh: Mesh, opts?: ArrangementOpts,
): Promise<PolygonArrangementResult>;
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
