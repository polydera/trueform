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
import { NDArray } from "../ndarray/NDArray";
import type {
  NDArrayFloat32, NDArrayFloat64, NDArrayInt32, NDArrayInt8,
} from "../ndarray/NDArray";
import type { FloatDtype } from "../ndarray/dtype";
import { Mesh } from "../form/Mesh";
import type { BoundaryRimsResult } from "../topology/sync";

/** What one hole group's fill produced (C++ `tf::hole_fill_status`). */
export const HoleFillStatus = {
  filled: 0,
  refusedOpen: 1,
  refusedInvalidRim: 2,
  refusedHostFace: 3,
  refusedProtectionDepth: 4,
  refusedTable: 5,
  refusedResource: 6,
  refusedRimSplit: 7,
} as const;

/** How the chart tier read a group's hole (C++ `tf::hole_chart_status`).
 *  `admitted` continues the host and `rimPlane` is the flat lid of an
 *  exactly planar rim; every other value is the reason the chart refused
 *  and handed the rim to the table. */
export const HoleChartStatus = {
  notAttempted: 0,
  admitted: 1,
  rimPlane: 2,
  noArea: 3,
  steep: 4,
  folded: 5,
  collarArity: 6,
  uncovered: 7,
  unsampled: 8,
  unsolved: 9,
  forbiddenChord: 10,
  resource: 11,
  outOfLattice: 12,
} as const;

/** How a group's patch left refinement (C++ `tf::hole_refine_status`). */
export const HoleRefineStatus = {
  notAttempted: 0,
  floorMet: 1,
  stalled: 2,
  budgetExhausted: 3,
} as const;

/** How a group's patch left fairing (C++ `tf::hole_fair_status`). */
export const HoleFairStatus = {
  notAttempted: 0,
  faired: 1,
  fairingOff: 2,
  fairingRefused: 3,
} as const;

/** How one fill shapes its patches. */
export interface FillHolesOpts {
  /** Quality floor a coarse patch is refined toward; 0 asks for none. Default 0.3. */
  minQuality?: number;
  /** Whether a refined coarse patch is faired against its host. Default true. */
  fairing?: boolean;
  /** Whether a tier may split the host's original rim edges. Default true. */
  splitRim?: boolean;
}

/** What {@link fillHoles} made of a mesh's rims, one group per rim.
 *
 * Entry g of every per-group lane belongs to group g; jagged lanes are rows
 * beside [G + 1] offsets. A corner below the mesh's point count is one of its
 * vertices; at or above it, it names `mintedPoints` read in order. */
export interface HoleFill {
  /** [G + 1] the run of rims each group fills. */
  groupOffsets: NDArrayInt32;
  /** [G + 1] offsets of each group's triangles. */
  triangleOffsets: NDArrayInt32;
  /** [T, 3] patch triangles. */
  triangles: NDArrayInt32;
  /** [G + 1] offsets of each group's minted points. */
  mintedOffsets: NDArrayInt32;
  /** [M, 3] minted points, in the mesh's dtype. */
  mintedPoints: NDArrayFloat32 | NDArrayFloat64;
  /** [G] values of {@link HoleFillStatus}. */
  status: NDArrayInt8;
  /** [G] values of {@link HoleChartStatus}. */
  charted: NDArrayInt8;
  /** [G] values of {@link HoleRefineStatus}. */
  refined: NDArrayInt8;
  /** [G] values of {@link HoleFairStatus}. */
  faired: NDArrayInt8;
  /** [G] the largest angle across each seam in radians, -1 where not computed. */
  seamMaxAngle: NDArrayFloat32 | NDArrayFloat64;
  /** [G, 2] the (rim, edge) a status speaks of, (-1, -1) where none. */
  offending: NDArrayInt32;
  /** [S, 4] the original edges split, as (face, v0, v1, point). */
  splits: NDArrayInt32;
  /** [S] each split's parameter from v0, a numerator over 64. */
  splitParameters: NDArrayInt32;
  /** [P] the carrying faces those splits reached, ascending. */
  planFaces: NDArrayInt32;
  /** [P + 1] offsets of each plan face's triangles. */
  planOffsets: NDArrayInt32;
  /** [Q, 3] the triangles the plan faces become. */
  planTriangles: NDArrayInt32;
}

/** @internal */
export function wrapHoleFill(raw: any, dt: FloatDtype): HoleFill {
  return {
    groupOffsets: new NDArray(raw.groupOffsets, "int32"),
    triangleOffsets: new NDArray(raw.triangleOffsets, "int32"),
    triangles: new NDArray(raw.triangles, "int32"),
    mintedOffsets: new NDArray(raw.mintedOffsets, "int32"),
    mintedPoints: new NDArray(raw.mintedPoints, dt) as NDArrayFloat32 | NDArrayFloat64,
    status: new NDArray(raw.status, "int8"),
    charted: new NDArray(raw.charted, "int8"),
    refined: new NDArray(raw.refined, "int8"),
    faired: new NDArray(raw.faired, "int8"),
    seamMaxAngle: new NDArray(raw.seamMaxAngle, dt) as NDArrayFloat32 | NDArrayFloat64,
    offending: new NDArray(raw.offending, "int32"),
    splits: new NDArray(raw.splits, "int32"),
    splitParameters: new NDArray(raw.splitParameters, "int32"),
    planFaces: new NDArray(raw.planFaces, "int32"),
    planOffsets: new NDArray(raw.planOffsets, "int32"),
    planTriangles: new NDArray(raw.planTriangles, "int32"),
  };
}

/** @internal */
export function fillArgs(rims: BoundaryRimsResult, opts?: FillHolesOpts): unknown[] {
  return [
    rims.vertices._handle, rims.faces._handle, rims.closed._handle,
    opts?.minQuality ?? 0.3, opts?.fairing ?? true, opts?.splitRim ?? true,
  ];
}

/** @internal */
export function filledArgs(fill: HoleFill): unknown[] {
  return [
    fill.triangleOffsets._handle, fill.triangles._handle,
    fill.mintedOffsets._handle, fill.mintedPoints._handle,
    fill.planFaces._handle, fill.planOffsets._handle, fill.planTriangles._handle,
  ];
}

/** Release every lane of a fill. */
export function deleteHoleFill(fill: HoleFill): void {
  for (const lane of Object.values(fill)) (lane as NDArray<any>).delete();
}

/**
 * Fill a mesh's boundary rims with patches of its own kind.
 *
 * One rim is one group, handed down a ladder of tiers: exhaustive enumeration
 * for a rim of at most five vertices; then the hole read as a height field
 * over the plane its rim spans, continuing the surface around it — or, where
 * that chart refuses and the rim lies exactly in one plane, the flat lid in
 * that plane; and otherwise a table over the tetrahedralization of its sites.
 * Replay the result with {@link filledMesh}.
 *
 * @param mesh - The mesh the rims are read against
 * @param rims - The rims to fill, as {@link boundaryRims} returns them
 * @param opts - How the patches are shaped
 */
export function fillHoles(mesh: Mesh, rims: BoundaryRimsResult, opts?: FillHolesOpts): HoleFill {
  const dt = mesh.dtype as FloatDtype;
  const raw = native()[`fill_holes_${dt}`](mesh._handle, ...fillArgs(rims, opts));
  return wrapHoleFill(raw, dt);
}

/**
 * The mesh a fill states: the input with what it changed. A face no split
 * reached is copied, a carrying face a split reached is replaced by its plan,
 * and the patches follow; the points are the input's, then the minted ones.
 *
 * @param mesh - The mesh the fill was stated against
 * @param fill - The result of {@link fillHoles} on that mesh
 */
export function filledMesh(mesh: Mesh, fill: HoleFill): Mesh {
  const dt = mesh.dtype as FloatDtype;
  return new Mesh(native()[`filled_mesh_${dt}`](mesh._handle, ...filledArgs(fill)), dt);
}
