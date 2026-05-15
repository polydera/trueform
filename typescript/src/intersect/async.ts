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
import { assertSameDtype } from "../internal/dtype";
import { Mesh } from "../form/Mesh";
import { Curves } from "../form/Curves";
import { NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import type { FloatDtype } from "../ndarray/dtype";
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
    if (m0OrMeshes.length === 0) {
      throw new Error("intersectionCurves: empty mesh array");
    }
    assertSameDtype(
      m0OrMeshes,
      m0OrMeshes.map((_, i) => `meshes[${i}]`),
    );
    const dt = m0OrMeshes[0].dtype as FloatDtype;
    const o = m1OrOpts as IntersectOpts | undefined;
    const rc = o?.resolveCrossings ?? (m0OrMeshes.length > 2);
    const mode = buildMode(o, "sos", rc, false);
    const handles = m0OrMeshes.map(m => m._handle);
    return dispatcher().run(
      () => native()[`dispatch_intersection_curves_list_${dt}`](handles, mode),
      (raw) => new Curves(raw, dt),
    );
  }
  const m1 = m1OrOpts as Mesh;
  assertSameDtype([m0OrMeshes, m1], ["mesh0", "mesh1"]);
  const dt = m0OrMeshes.dtype as FloatDtype;
  const mode = buildMode(opts, "sos", false, false);
  return dispatcher().run(
    () => native()[`dispatch_intersection_curves_${dt}`](
      m0OrMeshes._handle, m1._handle, mode,
    ),
    (raw) => new Curves(raw, dt),
  );
}

/** Find curves where a mesh intersects itself, off the main thread. */
export async function selfIntersectionCurves(
  mesh: Mesh, opts?: IntersectOpts,
): Promise<Curves> {
  const dt = mesh.dtype as FloatDtype;
  const mode = buildMode(opts, "sos", true, true);
  return dispatcher().run(
    () => native()[`dispatch_self_intersection_curves_${dt}`](
      mesh._handle, mode,
    ),
    (raw) => new Curves(raw, dt),
  );
}

/** Extract isocontour curves at one or more thresholds, off the main thread. */
export async function isocontours(
  mesh: Mesh, scalars: NDArrayFloat32 | NDArrayFloat64,
  threshold: number | Float32Array | Float64Array | number[],
): Promise<Curves> {
  assertSameDtype([mesh, scalars], ["mesh", "scalars"]);
  const dt = mesh.dtype as FloatDtype;
  if (typeof threshold === "number") {
    return dispatcher().run(
      () =>
        native()[`dispatch_isocontours_${dt}`](
          mesh._handle, scalars._handle, threshold,
        ),
      (raw) => new Curves(raw, dt),
    );
  }
  return dispatcher().run(
    () =>
      native()[`dispatch_isocontours_multi_${dt}`](
        mesh._handle, scalars._handle, threshold,
      ),
    (raw) => new Curves(raw, dt),
  );
}
