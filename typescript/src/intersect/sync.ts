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
import { Curves } from "../form/Curves";
import { NDArrayFloat32 } from "../ndarray/NDArray";

/** Intersection mode: "sos" (fast) or "primitives" (handles shared geometry). */
export type IntersectMode = "sos" | "primitives";

const MODE_MAP: Record<IntersectMode, number> = { sos: 0, primitives: 1 };

export interface IntersectionCurvesOpts {
  mode?: IntersectMode;
}

/** Compute intersection curves between two meshes. */
export function intersectionCurves(m0: Mesh, m1: Mesh): Curves;
/** Compute intersection curves between two meshes with options. */
export function intersectionCurves(
  m0: Mesh, m1: Mesh, opts: IntersectionCurvesOpts,
): Curves;
/** Compute intersection curves from N meshes. */
export function intersectionCurves(
  meshes: Mesh[], opts?: IntersectionCurvesOpts,
): Curves;
export function intersectionCurves(
  m0OrMeshes: Mesh | Mesh[],
  m1OrOpts?: Mesh | IntersectionCurvesOpts,
  opts?: IntersectionCurvesOpts,
): Curves {
  if (Array.isArray(m0OrMeshes)) {
    const mode = MODE_MAP[(m1OrOpts as IntersectionCurvesOpts)?.mode ?? "sos"];
    const handles = m0OrMeshes.map(m => m._handle);
    return new Curves(native().intersection_curves_list(handles, mode));
  }
  const m1 = m1OrOpts as Mesh;
  const mode = MODE_MAP[opts?.mode ?? "sos"];
  return new Curves(
    native().intersection_curves(m0OrMeshes._handle, m1._handle, mode),
  );
}

/** Find curves where a mesh intersects itself. */
export function selfIntersectionCurves(
  mesh: Mesh, opts?: IntersectionCurvesOpts,
): Curves {
  const mode = MODE_MAP[opts?.mode ?? "sos"];
  return new Curves(
    native().self_intersection_curves(mesh._handle, mode),
  );
}

/** Extract isocontour curves from a scalar field at one or more thresholds. */
export function isocontours(
  mesh: Mesh, scalars: NDArrayFloat32, threshold: number | Float32Array,
): Curves {
  if (typeof threshold === "number") {
    return new Curves(
      native().isocontours(mesh._handle, scalars._handle, threshold),
    );
  }
  return new Curves(
    native().isocontours_multi(mesh._handle, scalars._handle, threshold),
  );
}
