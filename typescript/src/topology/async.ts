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
import { NDArray, type NDArrayInt32 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { Mesh } from "../form/Mesh";
import type { ConnectedComponentsResult, ComponentType } from "./sync";

function wrapComponentsResult(raw: any): ConnectedComponentsResult {
  return { labels: new NDArray(raw.labels, "int32"), nComponents: raw.nComponents };
}

// ============ Boolean queries ============

export async function isClosed(m: Mesh): Promise<boolean> {
  return dispatcher().run(() => native().dispatch_is_closed(m._handle), (v) => v);
}

export async function isOpen(m: Mesh): Promise<boolean> {
  return dispatcher().run(() => native().dispatch_is_open(m._handle), (v) => v);
}

export async function isManifold(m: Mesh): Promise<boolean> {
  return dispatcher().run(() => native().dispatch_is_manifold(m._handle), (v) => v);
}

export async function isNonManifold(m: Mesh): Promise<boolean> {
  return dispatcher().run(() => native().dispatch_is_non_manifold(m._handle), (v) => v);
}

// ============ Scalar queries ============

export async function eulerCharacteristic(m: Mesh): Promise<number> {
  return dispatcher().run(() => native().dispatch_euler_characteristic(m._handle), (v) => v);
}

// ============ Edge results ============

export async function boundaryEdges(m: Mesh): Promise<NDArrayInt32> {
  return dispatcher().run(
    () => native().dispatch_boundary_edges(m._handle),
    (raw) => new NDArray(raw, "int32"),
  );
}

export async function nonManifoldEdges(m: Mesh): Promise<NDArrayInt32> {
  return dispatcher().run(
    () => native().dispatch_non_manifold_edges(m._handle),
    (raw) => new NDArray(raw, "int32"),
  );
}

// ============ Path / neighborhood results ============

export async function boundaryPaths(m: Mesh): Promise<OffsetBlockedBuffer> {
  return dispatcher().run(
    () => native().dispatch_boundary_paths(m._handle),
    (raw) => new OffsetBlockedBuffer(raw),
  );
}

export async function kRings(m: Mesh, k: number, inclusive?: boolean): Promise<OffsetBlockedBuffer> {
  const inc = inclusive ?? false;
  return dispatcher().run(
    () => native().dispatch_k_rings(m._handle, k, inc),
    (raw) => new OffsetBlockedBuffer(raw),
  );
}

export async function neighborhoods(m: Mesh, radius: number, inclusive?: boolean): Promise<OffsetBlockedBuffer> {
  const inc = inclusive ?? false;
  return dispatcher().run(
    () => native().dispatch_neighborhoods(m._handle, radius, inc),
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
  return dispatcher().run(
    () => native().dispatch_consistently_oriented(m._handle),
    (raw) => new Mesh(raw),
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
