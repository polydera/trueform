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

export interface CleanedMeshResult {
  mesh: Mesh;
  faceMap: IndexMap;
  pointMap: IndexMap;
}

export interface CleanedPointsResult {
  points: Point;
  pointMap: IndexMap;
}

/** Clean a triangle soup (batch) into an indexed Mesh. */
export function cleaned(input: Triangle, tolerance?: number): Mesh;
/** Clean a mesh by removing duplicate/degenerate vertices and faces. */
export function cleaned(input: Mesh, tolerance?: number): Mesh;
/** Clean a mesh and return index maps. */
export function cleaned(
  input: Mesh,
  opts: { returnIndexMap: true; tolerance?: number },
): CleanedMeshResult;
/** Clean a point batch by removing duplicate points. */
export function cleaned(input: Point, tolerance?: number): Point;
/** Clean a point batch and return an index map. */
export function cleaned(
  input: Point,
  opts: { returnIndexMap: true; tolerance?: number },
): CleanedPointsResult;
export function cleaned(
  input: Triangle | Mesh | Point,
  tolOrOpts?: number | { returnIndexMap: true; tolerance?: number },
): Mesh | Point | CleanedMeshResult | CleanedPointsResult {
  const wantMaps = typeof tolOrOpts === "object" && tolOrOpts.returnIndexMap;
  const tol =
    typeof tolOrOpts === "number"
      ? tolOrOpts
      : typeof tolOrOpts === "object"
        ? (tolOrOpts.tolerance ?? 0)
        : 0;

  // Triangle soup → Mesh
  if (input instanceof Primitive && input.type === "triangle") {
    return new Mesh(native().cleaned_polygon_soup(input._handle, tol));
  }

  // Mesh → Mesh (optionally with index maps)
  if (input instanceof Mesh) {
    if (wantMaps) {
      const raw = native().cleaned_mesh_with_maps(input._handle, tol);
      return {
        mesh: new Mesh(raw.mesh),
        faceMap: new IndexMap(raw.faceMap),
        pointMap: new IndexMap(raw.pointMap),
      };
    }
    return new Mesh(native().cleaned_mesh(input._handle, tol));
  }

  // Point batch → Point (optionally with index map)
  if (wantMaps) {
    const raw = native().cleaned_points_with_map(input._handle, tol);
    return {
      points: new Primitive(raw.points, "point") as Point,
      pointMap: new IndexMap(raw.pointMap),
    };
  }
  return new Primitive(
    native().cleaned_points(input._handle, tol),
    "point",
  ) as Point;
}
