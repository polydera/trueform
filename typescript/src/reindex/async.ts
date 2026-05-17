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
import type { FloatDtype } from "../ndarray/dtype";
import { IndexMap } from "../core/IndexMap";
import { assertSameDtype } from "../internal/dtype";
import type {
  ReindexedMeshResult, SplitResult,
  DomainLabelsInput, SplitDomainsResult,
} from "./sync";

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
    const dt = data.dtype as FloatDtype;
    return dispatcher().run(
      () =>
        native()[`dispatch_reindexed_mesh_${dt}`](
          data._handle,
          { f: mapOrFaceMap.f._handle, keptIds: mapOrFaceMap.keptIds._handle },
          { f: pointMap!.f._handle, keptIds: pointMap!.keptIds._handle },
        ),
      (raw) => new Mesh(raw, dt),
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
  const dt = m.dtype as FloatDtype;
  const safeMask = mask.dtype === "bool" ? mask : mask.as("bool");
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () =>
        native()[`dispatch_reindexed_by_mask_with_maps_${dt}`](
          m._handle, safeMask._handle,
        ),
      (raw) => ({
        mesh: new Mesh(raw.mesh, dt),
        faceMap: new IndexMap(raw.faceMap),
        pointMap: new IndexMap(raw.pointMap),
      }),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_reindexed_by_mask_${dt}`](
      m._handle, safeMask._handle,
    ),
    (raw) => new Mesh(raw, dt),
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
  const dt = m.dtype as FloatDtype;
  const safeIds = ids.dtype === "int32" ? ids : ids.as("int32");
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () =>
        native()[`dispatch_reindexed_by_ids_with_maps_${dt}`](
          m._handle, safeIds._handle,
        ),
      (raw) => ({
        mesh: new Mesh(raw.mesh, dt),
        faceMap: new IndexMap(raw.faceMap),
        pointMap: new IndexMap(raw.pointMap),
      }),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_reindexed_by_ids_${dt}`](
      m._handle, safeIds._handle,
    ),
    (raw) => new Mesh(raw, dt),
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
  const dt = m.dtype as FloatDtype;
  const safeMask = mask.dtype === "bool" ? mask : mask.as("bool");
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () =>
        native()[`dispatch_reindexed_by_mask_on_points_with_maps_${dt}`](
          m._handle, safeMask._handle,
        ),
      (raw) => ({
        mesh: new Mesh(raw.mesh, dt),
        faceMap: new IndexMap(raw.faceMap),
        pointMap: new IndexMap(raw.pointMap),
      }),
    );
  }
  return dispatcher().run(
    () =>
      native()[`dispatch_reindexed_by_mask_on_points_${dt}`](
        m._handle, safeMask._handle,
      ),
    (raw) => new Mesh(raw, dt),
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
  const dt = m.dtype as FloatDtype;
  const safeIds = ids.dtype === "int32" ? ids : ids.as("int32");
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () =>
        native()[`dispatch_reindexed_by_ids_on_points_with_maps_${dt}`](
          m._handle, safeIds._handle,
        ),
      (raw) => ({
        mesh: new Mesh(raw.mesh, dt),
        faceMap: new IndexMap(raw.faceMap),
        pointMap: new IndexMap(raw.pointMap),
      }),
    );
  }
  return dispatcher().run(
    () =>
      native()[`dispatch_reindexed_by_ids_on_points_${dt}`](
        m._handle, safeIds._handle,
      ),
    (raw) => new Mesh(raw, dt),
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
  if (meshes.length === 0) throw new Error("concatenateMeshes: empty array");
  const names = meshes.map((_, i) => `meshes[${i}]`);
  assertSameDtype(meshes, names);
  const dt = meshes[0].dtype as FloatDtype;
  const handles = meshes.map((m) => m._handle);
  return dispatcher().run(
    () => native()[`dispatch_concatenate_meshes_${dt}`](handles),
    (raw) => new Mesh(raw, dt),
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
  const dt = m.dtype as FloatDtype;
  const safeLabels = labels.dtype === "int32" ? labels : labels.as("int32");
  return dispatcher().run(
    () => native()[`dispatch_split_into_components_${dt}`](
      m._handle, safeLabels._handle,
    ),
    (raw) => {
      const comps: Mesh[] = [];
      for (let i = 0; i < raw.components.size(); i++)
        comps.push(new Mesh(raw.components.get(i), dt));
      return { components: comps, labels: new NDArray(raw.labels, "int32") };
    },
  );
}

// ============================================================================
// splitIntoDomains
// ============================================================================

export async function splitIntoDomains(
  m: Mesh, dl: DomainLabelsInput,
): Promise<SplitDomainsResult> {
  const dt = m.dtype as FloatDtype;
  return dispatcher().run(
    () => native()[`dispatch_split_into_domains_${dt}`](
      m._handle, dl.labels._handle, dl.nDomains,
    ),
    (raw) => {
      const comps: Mesh[] = [];
      for (let i = 0; i < raw.components.size(); i++)
        comps.push(new Mesh(raw.components.get(i), dt));
      return { components: comps, labels: new NDArray(raw.labels, "int32") };
    },
  );
}
