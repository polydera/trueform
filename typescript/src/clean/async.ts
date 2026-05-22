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
import type { FloatDtype } from "../ndarray/dtype";
import type { CleanedMeshResult, CleanedPointsResult, CleanOpts } from "./sync";

/** Clean a polygon soup (batch) into an indexed Mesh, off the main thread. */
export async function cleaned(
  input: Triangle,
  tolOrOpts?: number | CleanOpts,
): Promise<Mesh>;
/** Clean a mesh, off the main thread. */
export async function cleaned(
  input: Mesh,
  tolOrOpts?: number | CleanOpts,
): Promise<Mesh>;
/** Clean a mesh and return index maps, off the main thread. */
export async function cleaned(
  input: Mesh,
  opts: CleanOpts & { returnIndexMap: true },
): Promise<CleanedMeshResult>;
/** Clean a point batch, off the main thread. */
export async function cleaned(
  input: Point,
  tolOrOpts?: number | CleanOpts,
): Promise<Point>;
/** Clean a point batch and return an index map, off the main thread. */
export async function cleaned(
  input: Point,
  opts: CleanOpts & { returnIndexMap: true },
): Promise<CleanedPointsResult>;
export async function cleaned(
  input: Triangle | Mesh | Point,
  tolOrOpts?: number | CleanOpts,
): Promise<Mesh | Point | CleanedMeshResult | CleanedPointsResult> {
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
    return dispatcher().run(
      () =>
        native()[`dispatch_cleaned_polygon_soup_${dt}`](
          input._handle,
          tol,
          removeDup,
          removeUnref,
        ),
      (raw) => new Mesh(raw, dt),
    );
  }

  // Mesh → Mesh (optionally with index maps)
  if (input instanceof Mesh) {
    const dt = input.dtype as FloatDtype;
    if (wantMaps) {
      return dispatcher().run(
        () =>
          native()[`dispatch_cleaned_mesh_with_maps_${dt}`](
            input._handle,
            tol,
            removeDup,
            removeUnref,
          ),
        (raw) => ({
          mesh: new Mesh(raw.mesh, dt),
          faceMap: new IndexMap(raw.faceMap),
          pointMap: new IndexMap(raw.pointMap),
        }),
      );
    }
    return dispatcher().run(
      () =>
        native()[`dispatch_cleaned_mesh_${dt}`](
          input._handle,
          tol,
          removeDup,
          removeUnref,
        ),
      (raw) => new Mesh(raw, dt),
    );
  }

  // Point batch → Point (optionally with index map)
  const dt = input.dtype as FloatDtype;
  if (wantMaps) {
    return dispatcher().run(
      () =>
        native()[`dispatch_cleaned_points_with_map_${dt}`](
          input._handle,
          tol,
          removeDup,
          removeUnref,
        ),
      (raw) => ({
        points: new Primitive(raw.points, "point", dt) as Point,
        pointMap: new IndexMap(raw.pointMap),
      }),
    );
  }
  return dispatcher().run(
    () =>
      native()[`dispatch_cleaned_points_${dt}`](
        input._handle,
        tol,
        removeDup,
        removeUnref,
      ),
    (raw) => new Primitive(raw, "point", dt) as Point,
  );
}
