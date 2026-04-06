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
import { NDArray, type NDArrayInt32, type NDArrayBool } from "../ndarray/NDArray";
import { IndexMap } from "../core/IndexMap";

/** Result of reindexing a mesh with returnIndexMap: true. */
export interface ReindexedMeshResult {
  /** The reindexed mesh. */
  mesh: Mesh;
  /** Face index map (old → new). */
  faceMap: IndexMap;
  /** Point index map (old → new). */
  pointMap: IndexMap;
}

/** Result of splitting a mesh into components. */
export interface SplitResult {
  /** One mesh per component. */
  components: Mesh[];
  /** Per-component label values. */
  labels: NDArrayInt32;
}

// ============================================================================
// reindexed — apply an existing index map
// ============================================================================

/** Apply an index map to an NDArray (gather by keptIds). */
export function reindexed(data: NDArray, map: IndexMap): NDArray;
/** Apply face and point index maps to a Mesh. */
export function reindexed(data: Mesh, faceMap: IndexMap, pointMap: IndexMap): Mesh;
export function reindexed(
  data: NDArray | Mesh,
  mapOrFaceMap: IndexMap,
  pointMap?: IndexMap,
): NDArray | Mesh {
  if (data instanceof Mesh) {
    return new Mesh(
      native().reindexed_mesh(
        data._handle,
        { f: mapOrFaceMap.f._handle, keptIds: mapOrFaceMap.keptIds._handle },
        { f: pointMap!.f._handle, keptIds: pointMap!.keptIds._handle },
      ),
    );
  }
  return data.take(mapOrFaceMap.keptIds);
}

// ============================================================================
// reindexedByMask — filter mesh by face mask
// ============================================================================

/** Filter a mesh by a boolean face mask. */
export function reindexedByMask(m: Mesh, mask: NDArrayBool): Mesh;
/** Filter a mesh by a boolean face mask, returning index maps. */
export function reindexedByMask(
  m: Mesh,
  mask: NDArrayBool,
  opts: { returnIndexMap: true },
): ReindexedMeshResult;
export function reindexedByMask(
  m: Mesh,
  mask: NDArrayBool,
  opts?: { returnIndexMap: true },
): Mesh | ReindexedMeshResult {
  const safeMask = mask.dtype === "bool" ? mask : mask.as("bool");
  if (opts?.returnIndexMap) {
    const raw = native().reindexed_by_mask_with_maps(m._handle, safeMask._handle);
    return {
      mesh: new Mesh(raw.mesh),
      faceMap: new IndexMap(raw.faceMap),
      pointMap: new IndexMap(raw.pointMap),
    };
  }
  return new Mesh(native().reindexed_by_mask(m._handle, safeMask._handle));
}

// ============================================================================
// reindexedByIds — filter mesh by face IDs
// ============================================================================

/** Extract specific faces from a mesh by their IDs. */
export function reindexedByIds(m: Mesh, ids: NDArrayInt32): Mesh;
/** Extract specific faces with index maps. */
export function reindexedByIds(
  m: Mesh,
  ids: NDArrayInt32,
  opts: { returnIndexMap: true },
): ReindexedMeshResult;
export function reindexedByIds(
  m: Mesh,
  ids: NDArrayInt32,
  opts?: { returnIndexMap: true },
): Mesh | ReindexedMeshResult {
  const safeIds = ids.dtype === "int32" ? ids : ids.as("int32");
  if (opts?.returnIndexMap) {
    const raw = native().reindexed_by_ids_with_maps(m._handle, safeIds._handle);
    return {
      mesh: new Mesh(raw.mesh),
      faceMap: new IndexMap(raw.faceMap),
      pointMap: new IndexMap(raw.pointMap),
    };
  }
  return new Mesh(native().reindexed_by_ids(m._handle, safeIds._handle));
}

// ============================================================================
// reindexedByMaskOnPoints — filter mesh by point mask
// ============================================================================

/** Filter a mesh keeping only faces whose ALL vertices pass the point mask. */
export function reindexedByMaskOnPoints(m: Mesh, mask: NDArrayBool): Mesh;
/** Filter by point mask with index maps. */
export function reindexedByMaskOnPoints(
  m: Mesh,
  mask: NDArrayBool,
  opts: { returnIndexMap: true },
): ReindexedMeshResult;
export function reindexedByMaskOnPoints(
  m: Mesh,
  mask: NDArrayBool,
  opts?: { returnIndexMap: true },
): Mesh | ReindexedMeshResult {
  const safeMask = mask.dtype === "bool" ? mask : mask.as("bool");
  if (opts?.returnIndexMap) {
    const raw = native().reindexed_by_mask_on_points_with_maps(
      m._handle,
      safeMask._handle,
    );
    return {
      mesh: new Mesh(raw.mesh),
      faceMap: new IndexMap(raw.faceMap),
      pointMap: new IndexMap(raw.pointMap),
    };
  }
  return new Mesh(
    native().reindexed_by_mask_on_points(m._handle, safeMask._handle),
  );
}

// ============================================================================
// reindexedByIdsOnPoints — filter mesh by point IDs
// ============================================================================

/** Filter a mesh keeping only faces whose ALL vertices are in the point ID list. */
export function reindexedByIdsOnPoints(m: Mesh, ids: NDArrayInt32): Mesh;
/** Filter by point IDs with index maps. */
export function reindexedByIdsOnPoints(
  m: Mesh,
  ids: NDArrayInt32,
  opts: { returnIndexMap: true },
): ReindexedMeshResult;
export function reindexedByIdsOnPoints(
  m: Mesh,
  ids: NDArrayInt32,
  opts?: { returnIndexMap: true },
): Mesh | ReindexedMeshResult {
  const safeIds = ids.dtype === "int32" ? ids : ids.as("int32");
  if (opts?.returnIndexMap) {
    const raw = native().reindexed_by_ids_on_points_with_maps(
      m._handle,
      safeIds._handle,
    );
    return {
      mesh: new Mesh(raw.mesh),
      faceMap: new IndexMap(raw.faceMap),
      pointMap: new IndexMap(raw.pointMap),
    };
  }
  return new Mesh(
    native().reindexed_by_ids_on_points(m._handle, safeIds._handle),
  );
}

// ============================================================================
// concatenateMeshes — merge meshes with index offsetting
// ============================================================================

/** Merge multiple meshes into one, adjusting face indices. Applies transformations if present. */
export function concatenateMeshes(meshes: Mesh[]): Mesh;
export function concatenateMeshes(...meshes: Mesh[]): Mesh;
export function concatenateMeshes(...args: any[]): Mesh {
  const meshes: Mesh[] = Array.isArray(args[0]) ? args[0] : args;
  const handles = meshes.map((m) => m._handle);
  return new Mesh(native().concatenate_meshes(handles));
}

// ============================================================================
// splitIntoComponents — split mesh by per-face labels
// ============================================================================

/** Split a mesh into separate meshes by per-face labels. */
export function splitIntoComponents(m: Mesh, labels: NDArrayInt32): SplitResult {
  const safeLabels = labels.dtype === "int32" ? labels : labels.as("int32");
  const raw = native().split_into_components(m._handle, safeLabels._handle);
  const comps: Mesh[] = [];
  for (let i = 0; i < raw.components.size(); i++)
    comps.push(new Mesh(raw.components.get(i)));
  return {
    components: comps,
    labels: new NDArray(raw.labels, "int32"),
  };
}
