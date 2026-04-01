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
import { Curves } from "../form/Curves";
import { NDArrayFloat32 } from "../ndarray/NDArray";
import { IntersectOpts, buildMode } from "./sync";

/** Compute intersection curves between two meshes, off the main thread. */
export async function intersectionCurves(
  m0: Mesh, m1: Mesh, opts?: IntersectOpts,
): Promise<Curves>;
/** Compute intersection curves from N meshes, off the main thread. */
export async function intersectionCurves(
  meshes: Mesh[], opts?: IntersectOpts,
): Promise<Curves>;
export async function intersectionCurves(
  m0OrMeshes: Mesh | Mesh[],
  m1OrOpts?: Mesh | IntersectOpts,
  opts?: IntersectOpts,
): Promise<Curves> {
  if (Array.isArray(m0OrMeshes)) {
    const o = m1OrOpts as IntersectOpts | undefined;
    const rc = o?.resolveCrossings ?? (m0OrMeshes.length > 2);
    const mode = buildMode(o, "sos", rc, false);
    const handles = m0OrMeshes.map(m => m._handle);
    return dispatcher().run(
      () => native().dispatch_intersection_curves_list(handles, mode),
      (raw) => new Curves(raw),
    );
  }
  const m1 = m1OrOpts as Mesh;
  const mode = buildMode(opts, "sos", false, false);
  return dispatcher().run(
    () => native().dispatch_intersection_curves(m0OrMeshes._handle, m1._handle, mode),
    (raw) => new Curves(raw),
  );
}

/** Find curves where a mesh intersects itself, off the main thread. */
export async function selfIntersectionCurves(
  mesh: Mesh, opts?: IntersectOpts,
): Promise<Curves> {
  const mode = buildMode(opts, "sos", true, true);
  return dispatcher().run(
    () => native().dispatch_self_intersection_curves(mesh._handle, mode),
    (raw) => new Curves(raw),
  );
}

/** Extract isocontour curves at one or more thresholds, off the main thread. */
export async function isocontours(
  mesh: Mesh, scalars: NDArrayFloat32, threshold: number | Float32Array,
): Promise<Curves> {
  if (typeof threshold === "number") {
    return dispatcher().run(
      () =>
        native().dispatch_isocontours(mesh._handle, scalars._handle, threshold),
      (raw) => new Curves(raw),
    );
  }
  return dispatcher().run(
    () =>
      native().dispatch_isocontours_multi(
        mesh._handle, scalars._handle, threshold,
      ),
    (raw) => new Curves(raw),
  );
}
