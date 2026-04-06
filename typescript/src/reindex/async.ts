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
import { Mesh } from "../form/Mesh";
import { NDArray, type NDArrayInt32, type NDArrayBool } from "../ndarray/NDArray";
import { IndexMap } from "../core/IndexMap";
import type { ReindexedMeshResult, SplitResult } from "./sync";

// ============================================================================
// reindexed
// ============================================================================

/** Apply an index map to an NDArray, off the main thread. */
export async function reindexed(data: NDArray, map: IndexMap): Promise<NDArray>;
/** Apply face and point index maps to a Mesh, off the main thread. */
export async function reindexed(
  data: Mesh,
  faceMap: IndexMap,
  pointMap: IndexMap,
): Promise<Mesh>;
export async function reindexed(
  data: NDArray | Mesh,
  mapOrFaceMap: IndexMap,
  pointMap?: IndexMap,
): Promise<NDArray | Mesh> {
  if (data instanceof Mesh) {
    return dispatcher().run(
      () =>
        native().dispatch_reindexed_mesh(
          data._handle,
          { f: mapOrFaceMap.f._handle, keptIds: mapOrFaceMap.keptIds._handle },
          { f: pointMap!.f._handle, keptIds: pointMap!.keptIds._handle },
        ),
      (raw) => new Mesh(raw),
    );
  }
  return data.take(mapOrFaceMap.keptIds);
}

// ============================================================================
// reindexedByMask
// ============================================================================

/** Filter a mesh by a boolean face mask, off the main thread. */
export async function reindexedByMask(
  m: Mesh,
  mask: NDArrayBool,
): Promise<Mesh>;
/** Filter a mesh by a boolean face mask with index maps, off the main thread. */
export async function reindexedByMask(
  m: Mesh,
  mask: NDArrayBool,
  opts: { returnIndexMap: true },
): Promise<ReindexedMeshResult>;
export async function reindexedByMask(
  m: Mesh,
  mask: NDArrayBool,
  opts?: { returnIndexMap: true },
): Promise<Mesh | ReindexedMeshResult> {
  const safeMask = mask.dtype === "bool" ? mask : mask.as("bool");
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () =>
        native().dispatch_reindexed_by_mask_with_maps(m._handle, safeMask._handle),
      (raw) => ({
        mesh: new Mesh(raw.mesh),
        faceMap: new IndexMap(raw.faceMap),
        pointMap: new IndexMap(raw.pointMap),
      }),
    );
  }
  return dispatcher().run(
    () => native().dispatch_reindexed_by_mask(m._handle, safeMask._handle),
    (raw) => new Mesh(raw),
  );
}

// ============================================================================
// reindexedByIds
// ============================================================================

/** Extract specific faces by IDs, off the main thread. */
export async function reindexedByIds(
  m: Mesh,
  ids: NDArrayInt32,
): Promise<Mesh>;
/** Extract specific faces with index maps, off the main thread. */
export async function reindexedByIds(
  m: Mesh,
  ids: NDArrayInt32,
  opts: { returnIndexMap: true },
): Promise<ReindexedMeshResult>;
export async function reindexedByIds(
  m: Mesh,
  ids: NDArrayInt32,
  opts?: { returnIndexMap: true },
): Promise<Mesh | ReindexedMeshResult> {
  const safeIds = ids.dtype === "int32" ? ids : ids.as("int32");
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () =>
        native().dispatch_reindexed_by_ids_with_maps(m._handle, safeIds._handle),
      (raw) => ({
        mesh: new Mesh(raw.mesh),
        faceMap: new IndexMap(raw.faceMap),
        pointMap: new IndexMap(raw.pointMap),
      }),
    );
  }
  return dispatcher().run(
    () => native().dispatch_reindexed_by_ids(m._handle, safeIds._handle),
    (raw) => new Mesh(raw),
  );
}

// ============================================================================
// reindexedByMaskOnPoints
// ============================================================================

/** Filter mesh by point mask, off the main thread. */
export async function reindexedByMaskOnPoints(
  m: Mesh,
  mask: NDArrayBool,
): Promise<Mesh>;
/** Filter mesh by point mask with index maps, off the main thread. */
export async function reindexedByMaskOnPoints(
  m: Mesh,
  mask: NDArrayBool,
  opts: { returnIndexMap: true },
): Promise<ReindexedMeshResult>;
export async function reindexedByMaskOnPoints(
  m: Mesh,
  mask: NDArrayBool,
  opts?: { returnIndexMap: true },
): Promise<Mesh | ReindexedMeshResult> {
  const safeMask = mask.dtype === "bool" ? mask : mask.as("bool");
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () =>
        native().dispatch_reindexed_by_mask_on_points_with_maps(
          m._handle,
          safeMask._handle,
        ),
      (raw) => ({
        mesh: new Mesh(raw.mesh),
        faceMap: new IndexMap(raw.faceMap),
        pointMap: new IndexMap(raw.pointMap),
      }),
    );
  }
  return dispatcher().run(
    () =>
      native().dispatch_reindexed_by_mask_on_points(m._handle, safeMask._handle),
    (raw) => new Mesh(raw),
  );
}

// ============================================================================
// reindexedByIdsOnPoints
// ============================================================================

/** Filter mesh by point IDs, off the main thread. */
export async function reindexedByIdsOnPoints(
  m: Mesh,
  ids: NDArrayInt32,
): Promise<Mesh>;
/** Filter mesh by point IDs with index maps, off the main thread. */
export async function reindexedByIdsOnPoints(
  m: Mesh,
  ids: NDArrayInt32,
  opts: { returnIndexMap: true },
): Promise<ReindexedMeshResult>;
export async function reindexedByIdsOnPoints(
  m: Mesh,
  ids: NDArrayInt32,
  opts?: { returnIndexMap: true },
): Promise<Mesh | ReindexedMeshResult> {
  const safeIds = ids.dtype === "int32" ? ids : ids.as("int32");
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () =>
        native().dispatch_reindexed_by_ids_on_points_with_maps(
          m._handle,
          safeIds._handle,
        ),
      (raw) => ({
        mesh: new Mesh(raw.mesh),
        faceMap: new IndexMap(raw.faceMap),
        pointMap: new IndexMap(raw.pointMap),
      }),
    );
  }
  return dispatcher().run(
    () =>
      native().dispatch_reindexed_by_ids_on_points(m._handle, safeIds._handle),
    (raw) => new Mesh(raw),
  );
}

// ============================================================================
// concatenateMeshes
// ============================================================================

/** Merge multiple meshes into one, off the main thread. */
export async function concatenateMeshes(meshes: Mesh[]): Promise<Mesh>;
export async function concatenateMeshes(...meshes: Mesh[]): Promise<Mesh>;
export async function concatenateMeshes(...args: any[]): Promise<Mesh> {
  const meshes: Mesh[] = Array.isArray(args[0]) ? args[0] : args;
  const handles = meshes.map((m) => m._handle);
  return dispatcher().run(
    () => native().dispatch_concatenate_meshes(handles),
    (raw) => new Mesh(raw),
  );
}

// ============================================================================
// splitIntoComponents
// ============================================================================

/** Split a mesh by per-face labels, off the main thread. */
export async function splitIntoComponents(
  m: Mesh,
  labels: NDArrayInt32,
): Promise<SplitResult> {
  const safeLabels = labels.dtype === "int32" ? labels : labels.as("int32");
  return dispatcher().run(
    () => native().dispatch_split_into_components(m._handle, safeLabels._handle),
    (raw) => ({
      components: Array.from(raw.components, (h: any) => new Mesh(h)),
      labels: Array.from(raw.labels),
    }),
  );
}
