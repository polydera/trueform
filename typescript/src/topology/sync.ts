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
import { NDArray, type NDArrayInt32 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { Mesh } from "../form/Mesh";

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
