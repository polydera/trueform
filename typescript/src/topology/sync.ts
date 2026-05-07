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
import {
  NDArray,
  type NDArrayInt32,
  type NDArrayFloat32,
  type NDArrayBool,
} from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { Mesh } from "../form/Mesh";
import { IndexMap } from "../core/IndexMap";

// ============ Boolean queries ============

/** True if the mesh has no boundary edges (every edge shared by exactly 2 faces). */
export function isClosed(m: Mesh): boolean {
  return native().is_closed(m._handle);
}

/** True if the mesh has at least one boundary edge. */
export function isOpen(m: Mesh): boolean {
  return native().is_open(m._handle);
}

/** True if every edge is shared by at most 2 faces. */
export function isManifold(m: Mesh): boolean {
  return native().is_manifold(m._handle);
}

/** True if any edge is shared by more than 2 faces. */
export function isNonManifold(m: Mesh): boolean {
  return native().is_non_manifold(m._handle);
}

// ============ Scalar queries ============

/** Euler characteristic: V - E + F. */
export function eulerCharacteristic(m: Mesh): number {
  return native().euler_characteristic(m._handle);
}

// ============ Edge results ============

/** Boundary edges as an Int32 NDArray of shape [N, 2]. */
export function boundaryEdges(m: Mesh): NDArrayInt32 {
  return new NDArray(native().boundary_edges(m._handle), "int32");
}

/** Non-manifold edges (shared by >2 faces) as an Int32 NDArray of shape [N, 2]. */
export function nonManifoldEdges(m: Mesh): NDArrayInt32 {
  return new NDArray(native().non_manifold_edges(m._handle), "int32");
}

// ============ Path / neighborhood results ============

/** Boundary loops as paths of vertex indices. */
export function boundaryPaths(m: Mesh): OffsetBlockedBuffer {
  return new OffsetBlockedBuffer(native().boundary_paths(m._handle));
}

/** K-ring neighborhoods for all vertices. */
export function kRings(m: Mesh, k: number, inclusive?: boolean): OffsetBlockedBuffer {
  return new OffsetBlockedBuffer(native().k_rings(m._handle, k, inclusive ?? false));
}

/** Radius-based neighborhoods for all vertices. */
export function neighborhoods(m: Mesh, radius: number, inclusive?: boolean): OffsetBlockedBuffer {
  return new OffsetBlockedBuffer(native().neighborhoods(m._handle, radius, inclusive ?? false));
}

/** Connect edge pairs into continuous vertex paths. */
export function connectEdgesToPaths(edges: NDArrayInt32): OffsetBlockedBuffer {
  return new OffsetBlockedBuffer(native().connect_edges_to_paths(edges._handle));
}

// ============ Connected components ============

/** Connected component labeling result. */
export interface ConnectedComponentsResult {
  labels: NDArrayInt32;
  nComponents: number;
}

function wrapComponentsResult(raw: any): ConnectedComponentsResult {
  return { labels: new NDArray(raw.labels, "int32"), nComponents: raw.nComponents };
}

/** Label connected components from variable-width connectivity (e.g. vertexLink, faceLink). */
export function labelConnectedComponents(connectivity: OffsetBlockedBuffer): ConnectedComponentsResult;
/** Label connected components from fixed-width [N, K] connectivity (-1 = no neighbor). */
export function labelConnectedComponents(connectivity: NDArrayInt32): ConnectedComponentsResult;
export function labelConnectedComponents(connectivity: OffsetBlockedBuffer | NDArrayInt32): ConnectedComponentsResult {
  if (connectivity instanceof OffsetBlockedBuffer) {
    return wrapComponentsResult(native().label_connected_components_obb(connectivity._handle));
  }
  return wrapComponentsResult(native().label_connected_components_ndarray(connectivity._handle));
}

/** Type of connected component analysis. */
export type ComponentType = "edge" | "manifoldEdge" | "vertex";

/** Compute connected components of a mesh using the specified connectivity type. */
export function connectedComponents(m: Mesh, type: ComponentType): ConnectedComponentsResult {
  switch (type) {
    case "edge": return labelConnectedComponents(m.faceLink);
    case "vertex": return labelConnectedComponents(m.vertexLink);
    case "manifoldEdge": return labelConnectedComponents(m.manifoldEdgeLink);
  }
}

// ============ Mesh mutation ============

/** Return a new mesh with consistently oriented faces (via manifold edge voting). */
export function consistentlyOriented(m: Mesh): Mesh {
  return new Mesh(native().consistently_oriented(m._handle));
}

// ============ Constrained Delaunay triangulation ============

/** 2D triangulation: triangle indices + vertex coordinates. */
export interface CdtResult {
  /** [K, 3] int32 triangle indices into `points`. */
  faces: NDArrayInt32;
  /** [M, 2] float32 vertex coordinates. */
  points: NDArrayFloat32;
}

/** 2D triangulation + the input-to-output point index map. */
export interface CdtResultWithMap extends CdtResult {
  /** Maps input-point indices (size = N) to output-point indices.
   *  `f[i] === f.size` (sentinel) means input point `i` fell on a
   *  triangle dropped by the parity filter; `keptIds[k] === f.size`
   *  marks output slots that are synthetic intersection vertices
   *  created during constraint arrangement. */
  indexMap: IndexMap;
}

/** Optional second-argument shape for `cdt`. */
export interface CdtOptions {
  /** [M, 2] int32 array of constraint edges. */
  edges?: NDArrayInt32;
  /** [M] bool array marking which constraints are region boundaries
   *  (default: all true). Non-boundary constrained edges are preserved
   *  but do not flip the parity used by the interior filter — useful
   *  for feature lines or polygon diagonals that should appear as
   *  edges without splitting the interior. */
  edgeMask?: NDArrayBool;
}

/** Constrained Delaunay triangulation (interior triangles only). */
export function cdt(
  points: NDArrayFloat32,
  options?: CdtOptions,
): CdtResult;

/** Constrained Delaunay triangulation, additionally returning the
 *  input-point-to-output-point index map. */
export function cdt(
  points: NDArrayFloat32,
  options: CdtOptions & { returnIndexMap: true },
): CdtResultWithMap;

export function cdt(
  points: NDArrayFloat32,
  options: CdtOptions & { returnIndexMap?: boolean } = {},
): CdtResult | CdtResultWithMap {
  const wantMap = options.returnIndexMap === true;
  const edges = options.edges;
  const mask = options.edgeMask;

  const wrap = (raw: any): CdtResult => ({
    faces: new NDArray(raw.faces, "int32"),
    points: new NDArray(raw.points, "float32"),
  });
  const wrapWithMap = (raw: any): CdtResultWithMap => ({
    faces: new NDArray(raw.faces, "int32"),
    points: new NDArray(raw.points, "float32"),
    indexMap: new IndexMap(raw.indexMap),
  });

  if (!edges) {
    if (wantMap) {
      return wrapWithMap(native().make_cdt_with_maps(points._handle));
    }
    return wrap(native().make_cdt(points._handle));
  }

  if (mask) {
    if (wantMap) {
      return wrapWithMap(
        native().make_cdt_edges_masked_with_maps(
          points._handle, edges._handle, mask._handle,
        ),
      );
    }
    return wrap(
      native().make_cdt_edges_masked(
        points._handle, edges._handle, mask._handle,
      ),
    );
  }

  if (wantMap) {
    return wrapWithMap(
      native().make_cdt_edges_with_maps(points._handle, edges._handle),
    );
  }
  return wrap(native().make_cdt_edges(points._handle, edges._handle));
}
