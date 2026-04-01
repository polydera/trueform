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

/** Options for controlling intersection computation. */
export interface IntersectOpts {
  /** Intersection mode: "sos" (fast) or "primitives" (handles shared geometry). */
  mode?: "sos" | "primitives";
  /** Resolve crossings between different contours on the same face. */
  resolveCrossings?: boolean;
  /** Resolve self-crossings within a single contour. */
  resolveSelfCrossings?: boolean;
}

const MODE_MAP: Record<string, number> = { sos: 1, primitives: 2 };
const RESOLVE_CROSSINGS = 4;
const RESOLVE_SELF_CROSSINGS = 8;

export function buildMode(
  opts: IntersectOpts | undefined,
  defaultMode: string,
  defaultResolveCrossings: boolean,
  defaultResolveSelfCrossings: boolean,
): number {
  let m = MODE_MAP[opts?.mode ?? defaultMode];
  if (opts?.resolveCrossings ?? defaultResolveCrossings) m |= RESOLVE_CROSSINGS;
  if (opts?.resolveSelfCrossings ?? defaultResolveSelfCrossings) m |= RESOLVE_SELF_CROSSINGS;
  return m;
}

/** Compute intersection curves between two meshes. */
export function intersectionCurves(m0: Mesh, m1: Mesh): Curves;
/** Compute intersection curves between two meshes with options. */
export function intersectionCurves(
  m0: Mesh, m1: Mesh, opts: IntersectOpts,
): Curves;
/** Compute intersection curves from N meshes. */
export function intersectionCurves(
  meshes: Mesh[], opts?: IntersectOpts,
): Curves;
export function intersectionCurves(
  m0OrMeshes: Mesh | Mesh[],
  m1OrOpts?: Mesh | IntersectOpts,
  opts?: IntersectOpts,
): Curves {
  if (Array.isArray(m0OrMeshes)) {
    const o = m1OrOpts as IntersectOpts | undefined;
    const rc = o?.resolveCrossings ?? (m0OrMeshes.length > 2);
    const mode = buildMode(o, "sos", rc, false);
    const handles = m0OrMeshes.map(m => m._handle);
    return new Curves(native().intersection_curves_list(handles, mode));
  }
  const m1 = m1OrOpts as Mesh;
  const mode = buildMode(opts, "sos", false, false);
  return new Curves(
    native().intersection_curves(m0OrMeshes._handle, m1._handle, mode),
  );
}

/** Find curves where a mesh intersects itself. */
export function selfIntersectionCurves(
  mesh: Mesh, opts?: IntersectOpts,
): Curves {
  const mode = buildMode(opts, "sos", true, true);
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
