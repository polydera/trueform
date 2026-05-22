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
import { Mesh } from "../form/Mesh";
import { IndexMap } from "../core/IndexMap";
import { Primitive, type Triangle, type Point } from "../primitive";
import type { FloatDtype } from "../ndarray/dtype";

export interface CleanedMeshResult {
  mesh: Mesh;
  faceMap: IndexMap;
  pointMap: IndexMap;
}

export interface CleanedPointsResult {
  points: Point;
  pointMap: IndexMap;
}

/**
 * Options for `cleaned`.
 *
 * - `tolerance`: merge vertices within this distance (default 0 = exact).
 * - `returnIndexMap`: return index maps for attribute reindexing.
 * - `removeDuplicatePrimitives`: drop duplicate faces / edges (default true).
 *   Has no effect on point cleaning or triangle-soup cleaning.
 * - `removeUnreferencedPoints`: drop vertices not referenced by any face /
 *   edge (default true). Has no effect on point cleaning or triangle-soup
 *   cleaning.
 */
export interface CleanOpts {
  tolerance?: number;
  returnIndexMap?: boolean;
  removeDuplicatePrimitives?: boolean;
  removeUnreferencedPoints?: boolean;
}

/** Clean a triangle soup (batch) into an indexed Mesh. */
export function cleaned(input: Triangle, tolOrOpts?: number | CleanOpts): Mesh;
/** Clean a mesh by removing duplicate/degenerate vertices and faces. */
export function cleaned(input: Mesh, tolOrOpts?: number | CleanOpts): Mesh;
/** Clean a mesh and return index maps. */
export function cleaned(
  input: Mesh,
  opts: CleanOpts & { returnIndexMap: true },
): CleanedMeshResult;
/** Clean a point batch by removing duplicate points. */
export function cleaned(input: Point, tolOrOpts?: number | CleanOpts): Point;
/** Clean a point batch and return an index map. */
export function cleaned(
  input: Point,
  opts: CleanOpts & { returnIndexMap: true },
): CleanedPointsResult;
export function cleaned(
  input: Triangle | Mesh | Point,
  tolOrOpts?: number | CleanOpts,
): Mesh | Point | CleanedMeshResult | CleanedPointsResult {
  const isOpts = typeof tolOrOpts === "object" && tolOrOpts !== null;
  const wantMaps = isOpts && tolOrOpts.returnIndexMap === true;
  const tol =
    typeof tolOrOpts === "number"
      ? tolOrOpts
      : isOpts
        ? (tolOrOpts.tolerance ?? 0)
        : 0;
  const removeDup = isOpts
    ? (tolOrOpts.removeDuplicatePrimitives ?? true)
    : true;
  const removeUnref = isOpts
    ? (tolOrOpts.removeUnreferencedPoints ?? true)
    : true;

  // Triangle soup → Mesh
  if (input instanceof Primitive && input.type === "triangle") {
    const dt = input.dtype as FloatDtype;
    return new Mesh(
      native()[`cleaned_polygon_soup_${dt}`](
        input._handle,
        tol,
        removeDup,
        removeUnref,
      ),
      dt,
    );
  }

  // Mesh → Mesh (optionally with index maps)
  if (input instanceof Mesh) {
    const dt = input.dtype as FloatDtype;
    if (wantMaps) {
      const raw = native()[`cleaned_mesh_with_maps_${dt}`](
        input._handle,
        tol,
        removeDup,
        removeUnref,
      );
      return {
        mesh: new Mesh(raw.mesh, dt),
        faceMap: new IndexMap(raw.faceMap),
        pointMap: new IndexMap(raw.pointMap),
      };
    }
    return new Mesh(
      native()[`cleaned_mesh_${dt}`](
        input._handle,
        tol,
        removeDup,
        removeUnref,
      ),
      dt,
    );
  }

  // Point batch → Point (optionally with index map)
  const dt = input.dtype as FloatDtype;
  if (wantMaps) {
    const raw = native()[`cleaned_points_with_map_${dt}`](
      input._handle,
      tol,
      removeDup,
      removeUnref,
    );
    return {
      points: new Primitive(raw.points, "point", dt) as Point,
      pointMap: new IndexMap(raw.pointMap),
    };
  }
  return new Primitive(
    native()[`cleaned_points_${dt}`](
      input._handle,
      tol,
      removeDup,
      removeUnref,
    ),
    "point",
    dt,
  ) as Point;
}
