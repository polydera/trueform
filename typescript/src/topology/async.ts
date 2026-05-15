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
import type {
  ConnectedComponentsResult,
  ComponentType,
  CdtResult,
  CdtResultWithMap,
  CdtOptions,
} from "./sync";

function wrapComponentsResult(raw: any): ConnectedComponentsResult {
  return { labels: new NDArray(raw.labels, "int32"), nComponents: raw.nComponents };
}

// ============ Boolean queries ============

export async function isClosed(m: Mesh): Promise<boolean> {
  return dispatcher().run(() => native()[`dispatch_is_closed_${m.dtype}`](m._handle), (v) => v);
}

export async function isOpen(m: Mesh): Promise<boolean> {
  return dispatcher().run(() => native()[`dispatch_is_open_${m.dtype}`](m._handle), (v) => v);
}

export async function isManifold(m: Mesh): Promise<boolean> {
  return dispatcher().run(() => native()[`dispatch_is_manifold_${m.dtype}`](m._handle), (v) => v);
}

export async function isNonManifold(m: Mesh): Promise<boolean> {
  return dispatcher().run(() => native()[`dispatch_is_non_manifold_${m.dtype}`](m._handle), (v) => v);
}

// ============ Scalar queries ============

export async function eulerCharacteristic(m: Mesh): Promise<number> {
  return dispatcher().run(() => native()[`dispatch_euler_characteristic_${m.dtype}`](m._handle), (v) => v);
}

// ============ Edge results ============

export async function boundaryEdges(m: Mesh): Promise<NDArrayInt32> {
  return dispatcher().run(
    () => native()[`dispatch_boundary_edges_${m.dtype}`](m._handle),
    (raw) => new NDArray(raw, "int32"),
  );
}

export async function nonManifoldEdges(m: Mesh): Promise<NDArrayInt32> {
  return dispatcher().run(
    () => native()[`dispatch_non_manifold_edges_${m.dtype}`](m._handle),
    (raw) => new NDArray(raw, "int32"),
  );
}

// ============ Path / neighborhood results ============

export async function boundaryPaths(m: Mesh): Promise<OffsetBlockedBuffer> {
  return dispatcher().run(
    () => native()[`dispatch_boundary_paths_${m.dtype}`](m._handle),
    (raw) => new OffsetBlockedBuffer(raw),
  );
}

export async function kRings(m: Mesh, k: number, inclusive?: boolean): Promise<OffsetBlockedBuffer> {
  const inc = inclusive ?? false;
  return dispatcher().run(
    () => native()[`dispatch_k_rings_${m.dtype}`](m._handle, k, inc),
    (raw) => new OffsetBlockedBuffer(raw),
  );
}

export async function neighborhoods(m: Mesh, radius: number, inclusive?: boolean): Promise<OffsetBlockedBuffer> {
  const inc = inclusive ?? false;
  return dispatcher().run(
    () => native()[`dispatch_neighborhoods_${m.dtype}`](m._handle, radius, inc),
    (raw) => new OffsetBlockedBuffer(raw),
  );
}

export async function connectEdgesToPaths(edges: NDArrayInt32): Promise<OffsetBlockedBuffer> {
  return dispatcher().run(
    () => native().dispatch_connect_edges_to_paths(edges._handle),
    (raw) => new OffsetBlockedBuffer(raw),
  );
}

// ============ Connected components ============

export async function labelConnectedComponents(connectivity: OffsetBlockedBuffer): Promise<ConnectedComponentsResult>;
export async function labelConnectedComponents(connectivity: NDArrayInt32): Promise<ConnectedComponentsResult>;
export async function labelConnectedComponents(connectivity: OffsetBlockedBuffer | NDArrayInt32): Promise<ConnectedComponentsResult> {
  if (connectivity instanceof OffsetBlockedBuffer) {
    return dispatcher().run(
      () => native().dispatch_label_connected_components_obb(connectivity._handle),
      wrapComponentsResult,
    );
  }
  return dispatcher().run(
    () => native().dispatch_label_connected_components_ndarray(connectivity._handle),
    wrapComponentsResult,
  );
}

export async function connectedComponents(m: Mesh, type: ComponentType): Promise<ConnectedComponentsResult> {
  switch (type) {
    case "edge": return labelConnectedComponents(m.faceLink);
    case "vertex": return labelConnectedComponents(m.vertexLink);
    case "manifoldEdge": return labelConnectedComponents(m.manifoldEdgeLink);
  }
}

// ============ Mesh mutation ============

export async function consistentlyOriented(m: Mesh): Promise<Mesh> {
  const dt = m.dtype;
  return dispatcher().run(
    () => native()[`dispatch_consistently_oriented_${dt}`](m._handle),
    (raw) => new Mesh(raw, dt),
  );
}

// ============ Precompute ============

/** Compute face membership off the main thread. Result is cached on the mesh. */
export async function computeFaceMembership(m: Mesh): Promise<OffsetBlockedBuffer> {
  await dispatcher().run(() => native().dispatch_ensure(m._handle, 3));
  return m.faceMembership;
}

/** Compute manifold edge link off the main thread. Result is cached on the mesh. */
export async function computeManifoldEdgeLink(m: Mesh): Promise<NDArrayInt32> {
  await dispatcher().run(() => native().dispatch_ensure(m._handle, 4));
  return m.manifoldEdgeLink;
}

/** Compute face link off the main thread. Result is cached on the mesh. */
export async function computeFaceLink(m: Mesh): Promise<OffsetBlockedBuffer> {
  await dispatcher().run(() => native().dispatch_ensure(m._handle, 5));
  return m.faceLink;
}

/** Compute vertex link off the main thread. Result is cached on the mesh. */
export async function computeVertexLink(m: Mesh): Promise<OffsetBlockedBuffer> {
  await dispatcher().run(() => native().dispatch_ensure(m._handle, 6));
  return m.vertexLink;
}

// ============ Constrained Delaunay triangulation ============

/** Constrained Delaunay triangulation, off the main thread. */
export async function cdt(
  points: NDArrayFloat32 | NDArrayFloat64,
  options?: CdtOptions,
): Promise<CdtResult>;

/** Constrained Delaunay triangulation + index map, off the main thread. */
export async function cdt(
  points: NDArrayFloat32 | NDArrayFloat64,
  options: CdtOptions & { returnIndexMap: true },
): Promise<CdtResultWithMap>;

export async function cdt(
  points: NDArrayFloat32 | NDArrayFloat64,
  options: CdtOptions & { returnIndexMap?: boolean } = {},
): Promise<CdtResult | CdtResultWithMap> {
  const wantMap = options.returnIndexMap === true;
  const edges = options.edges;
  const mask = options.edgeMask;
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
      return dispatcher().run(
        () => native()[`dispatch_make_cdt_with_maps_${dt}`](points._handle),
        wrapWithMap,
      );
    }
    return dispatcher().run(
      () => native()[`dispatch_make_cdt_${dt}`](points._handle),
      wrap,
    );
  }

  if (mask) {
    if (wantMap) {
      return dispatcher().run(
        () => native()[`dispatch_make_cdt_edges_masked_with_maps_${dt}`](
          points._handle, edges._handle, mask._handle,
        ),
        wrapWithMap,
      );
    }
    return dispatcher().run(
      () => native()[`dispatch_make_cdt_edges_masked_${dt}`](
        points._handle, edges._handle, mask._handle,
      ),
      wrap,
    );
  }

  if (wantMap) {
    return dispatcher().run(
      () => native()[`dispatch_make_cdt_edges_with_maps_${dt}`](
        points._handle, edges._handle,
      ),
      wrapWithMap,
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_make_cdt_edges_${dt}`](
      points._handle, edges._handle,
    ),
    wrap,
  );
}
