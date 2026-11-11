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
import type { NDArrayFloat64, NDArrayInt32 } from "../ndarray/NDArray";
import type { FloatDtype } from "../ndarray/dtype";
import { Mesh } from "../form/Mesh";

/** What {@link tetMesh} made of its input (C++ `tf::tet_mesh_status`). */
export const TetMeshStatus = {
  meshed: 0,
  refused: 1,
  unwound: 2,
  rankDeficient: 3,
  capacity: 4,
  open: 5,
} as const;

/** The recovery's face routes (C++ `tf::topology::cdt::dt3::tet_face_census`). */
export interface TetFaceCensus {
  covered: number; passes: number; flipped: number; points: number;
  unplaced: number; onPiece: number; refused: number; openRim: number;
  crossed: number; offLattice: number; budget: number; pierced: number;
  standing: number;
}

/** The constraint recovery's census. */
export interface TetConstraintCensus {
  segmentRounds: number; shellSplits: number; segmentsRefused: number;
  unrestoredFacets: number; faces: TetFaceCensus; flipRecovered: number;
  searchRecovered: number; splitRows: number; hullBarred: number;
  restoredBeforeFaces: boolean; retiredCrossing: number;
  retiredNeighbour: number; revivedSegments: number; collapsed: number;
  collapseRefused: number; stalledDrains: number; budgetDrains: number;
}

/** The status of a {@link tetMesh} call and what it counted
 *  (C++ `tf::tet_mesh_report`). */
export interface TetMeshReport {
  /** A value of {@link TetMeshStatus}. */
  status: number;
  refusedSegments: number; refusedDepth: number; refusedBudget: number;
  refusedFaces: number; rung: number; drains: number;
  pairsNamed: number; pairsCut: number; trianglesCut: number;
  droppedDegenerate: number; stitchedTriangles: number;
  failedTriangles: number;
  pairsUnresolved: number; inputOpenEdges: number;
  recoveredOpenEdges: number;
  storageBits: number;
  inputBits: number;
  frameBits: number; deepSites: number; deepest: number;
  cornerBitsNeeded: number; minAngle: number; flat: number; inverted: number;
  negativeCells: number; census: TetConstraintCensus;
}

/** How one {@link tetMesh} call runs. */
export interface TetMeshOpts {
  /** Cut the crossings the recovery finds, where it finds them (default true);
   *  without it the surface is meshed as given and a crossing in it refuses. */
  splitConstraints?: boolean;
  /** Plane pooling, in the mesh's coordinates (default 0): near-coplanar
   *  faces pool onto one plane at the entrance, and it is the band of every
   *  cut round. */
  tolerance?: number;
  /** Sites a run may hold per site its entrance states, an integer.
   *  Default: the library's. */
  siteBudgetRatio?: number;
  /** Also return the report, measured only when asked. */
  returnReport?: boolean;
  /** Also return the index map. */
  returnIndexMap?: boolean;
}

/** A tet mesh. The arrays own WASM storage; release them with {@link deleteTetMesh}. */
export interface TetMeshResult {
  /** [T, 4] corners into `points`. */
  tetrahedra: NDArrayInt32;
  /** [P, 3] world positions, float64 whatever the mesh's dtype. */
  points: NDArrayFloat64;
  /** A value of {@link TetMeshStatus}. */
  status: number;
  /** Present only under `returnReport`. */
  report?: TetMeshReport;
}

/** A tet mesh with where its input surface stands in it (C++ `tf::tet_mesh_index_map`). */
export interface TetMeshResultWithIndexMap extends TetMeshResult {
  /** [K + 1] offsets of each facet's triangles. */
  facetOffsets: NDArrayInt32;
  /** [N, 3] boundary triangles into `points`, wound as their facet is. */
  facetTriangles: NDArrayInt32;
  /** [F + 1] offsets of each input face's facets. */
  faceFacetOffsets: NDArrayInt32;
  /** The facets each input face lies in, ascending. */
  faceFacets: NDArrayInt32;
}

/** @internal */
export function wrapTetMesh(
  raw: any, opts?: TetMeshOpts,
): TetMeshResult | TetMeshResultWithIndexMap {
  const out: any = {
    tetrahedra: new NDArray(raw.tetrahedra, "int32"),
    points: new NDArray(raw.points, "float64"),
    status: raw.status,
  };
  if (opts?.returnReport) out.report = raw.report;
  if (raw.facetOffsets !== undefined) {
    out.facetOffsets = new NDArray(raw.facetOffsets, "int32");
    out.facetTriangles = new NDArray(raw.facetTriangles, "int32");
    out.faceFacetOffsets = new NDArray(raw.faceFacetOffsets, "int32");
    out.faceFacets = new NDArray(raw.faceFacets, "int32");
  }
  return out;
}

/** @internal */
export function tetMeshCall(mesh: Mesh, opts?: TetMeshOpts): [string, unknown[]] {
  const dt = mesh.dtype as FloatDtype;
  const name = opts?.returnIndexMap ? `tet_mesh_with_index_map_${dt}` : `tet_mesh_${dt}`;
  const budget = opts?.siteBudgetRatio;
  if (budget !== undefined && (!Number.isInteger(budget) || budget < 0))
    throw new Error(`siteBudgetRatio must be a non-negative integer, got ${budget}`);
  return [name, [
    mesh._handle, opts?.splitConstraints ?? true, opts?.tolerance ?? 0,
    budget !== undefined, budget ?? 0, opts?.returnReport ?? false,
  ]];
}

/** Release every array of a tet mesh. */
export function deleteTetMesh(result: TetMeshResult | TetMeshResultWithIndexMap): void {
  for (const [key, lane] of Object.entries(result))
    if (key !== "report" && key !== "status") (lane as NDArray<any>).delete();
}

/**
 * Tetrahedralize the solid a closed triangle surface winds.
 *
 * Crossing faces are cut at their canonical intersections, every face and
 * edge is recovered exactly as a constraint, and the cells the surface winds
 * positively are kept. The mesh's transformation places it: the points come
 * back in world space, in float64, because they are lattice positions finer
 * than a float32 holds.
 *
 * @param mesh - A triangle mesh
 * @param opts - `splitConstraints`, the tolerance, the site budget, and whether to
 *   return the report and the index map
 */
export function tetMesh(mesh: Mesh, opts: TetMeshOpts & { returnIndexMap: true }): TetMeshResultWithIndexMap;
export function tetMesh(mesh: Mesh, opts?: TetMeshOpts): TetMeshResult;
export function tetMesh(mesh: Mesh, opts?: TetMeshOpts): TetMeshResult | TetMeshResultWithIndexMap {
  const [name, args] = tetMeshCall(mesh, opts);
  return wrapTetMesh(native()[name](...args), opts);
}
