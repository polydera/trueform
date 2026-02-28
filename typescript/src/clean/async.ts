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
import { Mesh } from "../form/Mesh";
import { IndexMap } from "../core/IndexMap";
import { Primitive, type Triangle, type Point } from "../primitive";
import type { CleanedMeshResult, CleanedPointsResult } from "./sync";

/** Clean a polygon soup (batch) into an indexed Mesh, off the main thread. */
export async function cleaned(
  input: Triangle,
  tolerance?: number,
): Promise<Mesh>;
/** Clean a mesh, off the main thread. */
export async function cleaned(
  input: Mesh,
  tolerance?: number,
): Promise<Mesh>;
/** Clean a mesh and return index maps, off the main thread. */
export async function cleaned(
  input: Mesh,
  opts: { returnIndexMap: true; tolerance?: number },
): Promise<CleanedMeshResult>;
/** Clean a point batch, off the main thread. */
export async function cleaned(
  input: Point,
  tolerance?: number,
): Promise<Point>;
/** Clean a point batch and return an index map, off the main thread. */
export async function cleaned(
  input: Point,
  opts: { returnIndexMap: true; tolerance?: number },
): Promise<CleanedPointsResult>;
export async function cleaned(
  input: Triangle | Mesh | Point,
  tolOrOpts?: number | { returnIndexMap: true; tolerance?: number },
): Promise<Mesh | Point | CleanedMeshResult | CleanedPointsResult> {
  const wantMaps = typeof tolOrOpts === "object" && tolOrOpts.returnIndexMap;
  const tol =
    typeof tolOrOpts === "number"
      ? tolOrOpts
      : typeof tolOrOpts === "object"
        ? (tolOrOpts.tolerance ?? 0)
        : 0;

  // Triangle soup → Mesh
  if (input instanceof Primitive && input.type === "triangle") {
    return dispatcher().run(
      () => native().dispatch_cleaned_polygon_soup(input._handle, tol),
      (raw) => new Mesh(raw),
    );
  }

  // Mesh → Mesh (optionally with index maps)
  if (input instanceof Mesh) {
    if (wantMaps) {
      return dispatcher().run(
        () => native().dispatch_cleaned_mesh_with_maps(input._handle, tol),
        (raw) => ({
          mesh: new Mesh(raw.mesh),
          faceMap: new IndexMap(raw.faceMap),
          pointMap: new IndexMap(raw.pointMap),
        }),
      );
    }
    return dispatcher().run(
      () => native().dispatch_cleaned_mesh(input._handle, tol),
      (raw) => new Mesh(raw),
    );
  }

  // Point batch → Point (optionally with index map)
  if (wantMaps) {
    return dispatcher().run(
      () => native().dispatch_cleaned_points_with_map(input._handle, tol),
      (raw) => ({
        points: new Primitive(raw.points, "point") as Point,
        pointMap: new IndexMap(raw.pointMap),
      }),
    );
  }
  return dispatcher().run(
    () => native().dispatch_cleaned_points(input._handle, tol),
    (raw) => new Primitive(raw, "point") as Point,
  );
}
