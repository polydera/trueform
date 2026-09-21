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
  type NDArrayFloat64,
  type NDArrayBool,
} from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { Mesh } from "../form/Mesh";
import { IndexMap } from "../core/IndexMap";

// ============ Boolean queries ============

/** True if the mesh has no boundary edges (every edge shared by exactly 2 faces). */
export function isClosed(m: Mesh): boolean {
  return native()[`is_closed_${m.dtype}`](m._handle);
}

/** True if the mesh has at least one boundary edge. */
export function isOpen(m: Mesh): boolean {
  return native()[`is_open_${m.dtype}`](m._handle);
}

/** True if every edge is shared by at most 2 faces and every vertex's faces are one fan. */
export function isManifold(m: Mesh): boolean {
  return native()[`is_manifold_${m.dtype}`](m._handle);
}

/** True if any edge is shared by 3+ faces or any vertex's faces split into several fans. */
export function isNonManifold(m: Mesh): boolean {
  return native()[`is_non_manifold_${m.dtype}`](m._handle);
}

// ============ Scalar queries ============

/** Euler characteristic: V - E + F. */
export function eulerCharacteristic(m: Mesh): number {
  return native()[`euler_characteristic_${m.dtype}`](m._handle);
}

// ============ Edge results ============

/** Boundary edges as an Int32 NDArray of shape [N, 2]. */
export function boundaryEdges(m: Mesh): NDArrayInt32 {
  return new NDArray(native()[`boundary_edges_${m.dtype}`](m._handle), "int32");
}

/** Non-manifold edges (shared by >2 faces) as an Int32 NDArray of shape [N, 2]. */
export function nonManifoldEdges(m: Mesh): NDArrayInt32 {
  return new NDArray(native()[`non_manifold_edges_${m.dtype}`](m._handle), "int32");
}

// ============ Vertex results ============

/** The vertices whose faces are not one fan, ascending, as an Int32 NDArray
 *  of shape [N]. A vertex is non-manifold when an edge at it is shared by
 *  3+ faces, or its faces fall into several pieces meeting at the vertex
 *  alone. */
export function nonManifoldVertices(m: Mesh): NDArrayInt32 {
  return new NDArray(native()[`non_manifold_vertices_${m.dtype}`](m._handle), "int32");
}

// ============ Path / neighborhood results ============

/** Boundary loops as paths of vertex indices. */
export function boundaryPaths(m: Mesh): OffsetBlockedBuffer {
  return new OffsetBlockedBuffer(native()[`boundary_paths_${m.dtype}`](m._handle));
}

/** The boundary as rims: per rim its vertex path, the face carrying each of
 *  its edges, and whether it closes. */
export interface BoundaryRimsResult {
  /** Block i is rim i's vertex ids, in the order it is walked. */
  vertices: OffsetBlockedBuffer;
  /** Block i names the face carrying each of rim i's edges, so rim edge k
   *  runs from vertex k to vertex k + 1 and is carried by face k alone. */
  faces: OffsetBlockedBuffer;
  /** [R] bool: whether rim i's last edge runs back to its first vertex. A
   *  closed rim of n vertices has n edges, an open one n - 1. */
  closed: NDArrayBool;
}

/** Assemble the mesh's boundary edges into rims. A rim ends where the
 *  boundary stops passing straight through, so a pinch splits it. */
export function boundaryRims(m: Mesh): BoundaryRimsResult {
  const raw = native()[`boundary_rims_${m.dtype}`](m._handle);
  return {
    vertices: new OffsetBlockedBuffer(raw.vertices),
    faces: new OffsetBlockedBuffer(raw.faces),
    closed: new NDArray(raw.closed, "bool"),
  };
}

/** K-ring neighborhoods for all vertices. */
export function kRings(m: Mesh, k: number, inclusive?: boolean): OffsetBlockedBuffer {
  return new OffsetBlockedBuffer(native()[`k_rings_${m.dtype}`](m._handle, k, inclusive ?? false));
}

/** Radius-based neighborhoods for all vertices. */
export function neighborhoods(m: Mesh, radius: number, inclusive?: boolean): OffsetBlockedBuffer {
  return new OffsetBlockedBuffer(native()[`neighborhoods_${m.dtype}`](m._handle, radius, inclusive ?? false));
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
  const dt = m.dtype;
  return new Mesh(native()[`consistently_oriented_${dt}`](m._handle), dt);
}

/** The separated mesh and the input point each of its points copies. */
export interface SplitNonManifoldVerticesResult {
  mesh: Mesh;
  /** [P] int32: for each output point the input point it copies, an
   *  original itself. */
  pointMap: NDArrayInt32;
}

/** Return a new mesh with every fan at a vertex given a vertex of its own.
 *  Faces keep their ids, arity and winding; the points are the input's
 *  followed by the minted copies. A vertex on a 3+-face edge is left
 *  untouched — separating its fans would tear that edge into boundary
 *  copies — and nonManifoldVertices still names it. */
export function splitNonManifoldVertices(m: Mesh): SplitNonManifoldVerticesResult {
  const dt = m.dtype;
  const raw = native()[`split_non_manifold_vertices_${dt}`](m._handle);
  return {
    mesh: new Mesh(raw.mesh, dt),
    pointMap: new NDArray(raw.pointMap, "int32"),
  };
}

// ============ Constrained Delaunay triangulation ============

/** 2D triangulation: triangle indices + vertex coordinates. */
export interface CdtResult {
  /** [K, 3] int32 triangle indices into `points`. */
  faces: NDArrayInt32;
  /** [M, 2] vertex coordinates (dtype follows the input points). */
  points: NDArrayFloat32 | NDArrayFloat64;
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
  /** Resolve crossings between constraints by creating a point there
   *  (default: true). With `false` the constraints are demanded verbatim
   *  and the result is empty if any two of them cross — that emptiness is
   *  the answer to "do these constraints cross", not an error. Ignored
   *  when no `edges` are given. */
  splitConstraints?: boolean;
}

/** Constrained Delaunay triangulation (interior triangles only). */
export function cdt(
  points: NDArrayFloat32 | NDArrayFloat64,
  options?: CdtOptions,
): CdtResult;

/** Constrained Delaunay triangulation, additionally returning the
 *  input-point-to-output-point index map. */
export function cdt(
  points: NDArrayFloat32 | NDArrayFloat64,
  options: CdtOptions & { returnIndexMap: true },
): CdtResultWithMap;

export function cdt(
  points: NDArrayFloat32 | NDArrayFloat64,
  options: CdtOptions & { returnIndexMap?: boolean } = {},
): CdtResult | CdtResultWithMap {
  const wantMap = options.returnIndexMap === true;
  const edges = options.edges;
  const mask = options.edgeMask;
  const split = options.splitConstraints !== false;
  const dt = points.dtype;

  const wrap = (raw: any): CdtResult => ({
    faces: new NDArray(raw.faces, "int32"),
    points: new NDArray(raw.points, dt) as NDArrayFloat32 | NDArrayFloat64,
  });
  const wrapWithMap = (raw: any): CdtResultWithMap => ({
    faces: new NDArray(raw.faces, "int32"),
    points: new NDArray(raw.points, dt) as NDArrayFloat32 | NDArrayFloat64,
    indexMap: new IndexMap(raw.indexMap),
  });

  if (!edges) {
    if (wantMap) {
      return wrapWithMap(native()[`make_cdt_with_maps_${dt}`](points._handle));
    }
    return wrap(native()[`make_cdt_${dt}`](points._handle));
  }

  if (mask) {
    if (wantMap) {
      return wrapWithMap(
        native()[`make_cdt_edges_masked_with_maps_${dt}`](
          points._handle, edges._handle, mask._handle, split,
        ),
      );
    }
    return wrap(
      native()[`make_cdt_edges_masked_${dt}`](
        points._handle, edges._handle, mask._handle, split,
      ),
    );
  }

  if (wantMap) {
    return wrapWithMap(
      native()[`make_cdt_edges_with_maps_${dt}`](
        points._handle, edges._handle, split,
      ),
    );
  }
  return wrap(
    native()[`make_cdt_edges_${dt}`](points._handle, edges._handle, split),
  );
}

// ============ Volumetric domains ============

const _DC_EXCLUDE_OUTER_SHELL = 1;
const _DC_IGNORE_OPEN_FRAGMENTS = 2;

/** @internal — also used by the async wrapper. */
export function _buildDomainConfig(opts?: DomainLabelsOptions): number {
  let c = 0;
  if (opts?.excludeOuterShell) c |= _DC_EXCLUDE_OUTER_SHELL;
  if (opts?.ignoreOpenFragments) c |= _DC_IGNORE_OPEN_FRAGMENTS;
  return c;
}

/** Options for {@link domainLabels}. */
export interface DomainLabelsOptions {
  /**
   * Drop face-sides bounding open fragments (faces in MEL components
   * carrying boundary edges) by parking them at the sentinel label
   * `nDomains`.
   */
  ignoreOpenFragments?: boolean;
  /**
   * Fold the unbounded universe domain into the same sentinel, so only
   * bounded interior domains receive ids.
   */
  excludeOuterShell?: boolean;
}

/** Per-face per-side volumetric domain labels for a 3D polygon mesh. */
export interface DomainLabelsResult {
  /**
   * Shape `(nFaces, 2)`, int32. `labels[f, 0]` is the domain containing face
   * `f` with REVERSED winding (the side `f`'s stored normal points INTO);
   * `labels[f, 1]` is the domain containing it with FORWARD winding (the side
   * the stored normal points AWAY FROM). `splitIntoDomains` reverses side-0
   * emissions and keeps side-1 emissions to produce outward-oriented submeshes.
   */
  labels: NDArrayInt32;
  /** Number of distinct bounded domains. */
  nDomains: number;
  /**
   * Label carried by outer-shell sides. Equals `nDomains` under
   * `excludeOuterShell`; otherwise no face-side carries this label.
   */
  outerShellLabel: number;
}

function _wrapDomainLabels(raw: any): DomainLabelsResult {
  return {
    labels: new NDArray(raw.labels, "int32"),
    nDomains: raw.nDomains,
    outerShellLabel: raw.outerShellLabel,
  };
}

/**
 * Compute per-face per-side volumetric domain labels for a 3D polygon mesh.
 *
 * A non-manifold surface mesh bounds multiple 3D regions ("domains"). This
 * partitions space into volumetric components and returns one label per
 * face per side.
 */
export function domainLabels(m: Mesh, opts?: DomainLabelsOptions): DomainLabelsResult {
  const cfg = _buildDomainConfig(opts);
  return _wrapDomainLabels(native()[`make_domain_labels_${m.dtype}`](m._handle, cfg));
}
