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
import { assertSameDtype } from "../internal/dtype";
import { Mesh } from "../form/Mesh";
import { Curves } from "../form/Curves";
import type { FloatDtype } from "../ndarray/dtype";

/** Options for controlling intersection computation. */
export interface IntersectOpts {
  /**
   * The classifier the run states its contacts with. "primitives"
   * (default) classifies shared edges/vertices and coplanar contacts;
   * "sos" perturbs every contact into a crossing, so no contact is ever
   * coplanar — coplanar walls then do not pool and the regions they
   * would have separated stay joined. Crossings between contours are
   * resolved whenever the operands can make such a pair.
   */
  mode?: "sos" | "primitives";
  /** World-coordinate distance an input vertex may move to reach the lattice (0 = exact). */
  tolerance?: number;
  /**
   * Also intersect each mesh with itself — required when a mesh can
   * self-overlap, e.g. meshes concatenated into one operand. Composed
   * with the classifier into the one mode that crosses; a mesh against
   * itself alone states it by its own arity (selfIntersectionCurves).
   */
  within?: boolean;
}

const MODE_MAP: Record<string, number> = { sos: 1, primitives: 2 };

export function buildMode(opts: IntersectOpts | undefined): number {
  return MODE_MAP[opts?.mode ?? "primitives"] | (opts?.within ? 4 : 0);
}


export function getTolerance(opts: IntersectOpts | undefined): number {
  return opts?.tolerance ?? 0.0;
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
    if (m0OrMeshes.length === 0) {
      throw new Error("intersectionCurves: empty mesh array");
    }
    assertSameDtype(
      m0OrMeshes,
      m0OrMeshes.map((_, i) => `meshes[${i}]`),
    );
    const dt = m0OrMeshes[0].dtype as FloatDtype;
    const o = m1OrOpts as IntersectOpts | undefined;
    const mode = buildMode(o);
    const tolerance = getTolerance(o);
    const handles = m0OrMeshes.map(m => m._handle);
    return new Curves(
      native()[`intersection_curves_list_${dt}`](handles, mode, tolerance), dt,
    );
  }
  const m1 = m1OrOpts as Mesh;
  assertSameDtype([m0OrMeshes, m1], ["mesh0", "mesh1"]);
  const dt = m0OrMeshes.dtype as FloatDtype;
  const mode = buildMode(opts);
  const tolerance = getTolerance(opts);
  return new Curves(
    native()[`intersection_curves_${dt}`](m0OrMeshes._handle, m1._handle, mode, tolerance),
    dt,
  );
}

/** Find curves where a mesh intersects itself. */
export function selfIntersectionCurves(
  mesh: Mesh, opts?: IntersectOpts,
): Curves {
  const dt = mesh.dtype as FloatDtype;
  const mode = buildMode(opts);
  const tolerance = getTolerance(opts);
  return new Curves(
    native()[`self_intersection_curves_${dt}`](mesh._handle, mode, tolerance), dt,
  );
}

/** True if the mesh meets itself. Faces sharing a vertex or an edge are
 *  neighbors rather than contacts; everything else that touches counts,
 *  decided exactly, stopped at the first contact found. Completes the
 *  mesh's tree, face membership and edge link into its cache, so a
 *  later query pays for none of them. */
export function hasSelfIntersections(mesh: Mesh): boolean {
  return native()[`has_self_intersections_${mesh.dtype}`](mesh._handle);
}
