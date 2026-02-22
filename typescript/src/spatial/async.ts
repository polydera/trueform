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
import { Primitive, PrimitiveType, Ray } from "../primitive/Primitive";
import { NDArray, NDArrayFloat32, NDArrayBool } from "../ndarray/NDArray";
import type {
  ClosestPointResult, ClosestPointBatchResult,
  ClosestPointPairResult, ClosestPointPairBatchResult,
  NeighborResult, NeighborBatchResult, NeighborPairResult,
  RayCastResult, RayCastPrimBatchResult, RayCastFormBatchResult,
  RayCastOptions,
} from "./sync";

// ============================================================================
// Primitive type → C++ enum mapping
// ============================================================================

const PRIM_TYPE: Record<PrimitiveType, number> = {
  point: 0,
  segment: 1,
  triangle: 2,
  ray: 3,
  line: 4,
  plane: 5,
  aabb: 6,
  polygon: 7,
  vector: -1,
};

function primType(p: Primitive): number {
  const t = PRIM_TYPE[p.type];
  if (t < 0) throw new Error(`"${p.type}" is not a spatial primitive type`);
  return t;
}

// ============================================================================
// Result wrappers
// ============================================================================

function wrapClosestPoint(raw: any): ClosestPointResult {
  return {
    point: new NDArray(raw.point, "float32"),
    distance2: raw.distance2,
  };
}

function wrapClosestPointBatch(raw: any): ClosestPointBatchResult {
  return {
    points: new NDArray(raw.points, "float32"),
    distances: new NDArray(raw.distances, "float32"),
  };
}

function wrapClosestPointPair(raw: any): ClosestPointPairResult {
  return {
    point0: new NDArray(raw.point0, "float32"),
    point1: new NDArray(raw.point1, "float32"),
    distance2: raw.distance2,
  };
}

function wrapClosestPointPairBatch(raw: any): ClosestPointPairBatchResult {
  return {
    points0: new NDArray(raw.points0, "float32"),
    points1: new NDArray(raw.points1, "float32"),
    distances: new NDArray(raw.distances, "float32"),
  };
}

function wrapNeighbor(raw: any): NeighborResult {
  return {
    elementId: raw.elementId,
    point: new NDArray(raw.point, "float32"),
    distance2: raw.distance2,
  };
}

function wrapNeighborBatch(raw: any): NeighborBatchResult {
  return {
    elementIds: new NDArray(raw.elementIds, "int32"),
    points: new NDArray(raw.points, "float32"),
    distances: new NDArray(raw.distances, "float32"),
  };
}

function wrapNeighborPair(raw: any): NeighborPairResult {
  return {
    elementId0: raw.elementId0,
    elementId1: raw.elementId1,
    point0: new NDArray(raw.point0, "float32"),
    point1: new NDArray(raw.point1, "float32"),
    distance2: raw.distance2,
  };
}

// ============================================================================
// distance2
// ============================================================================

/** Squared distance between two primitives, off the main thread. */
export async function distance2(
  a: Primitive, b: Primitive,
): Promise<number | NDArrayFloat32>;
/** Squared distance from a mesh to a primitive, off the main thread. */
export async function distance2(
  mesh: Mesh, prim: Primitive,
): Promise<number | NDArrayFloat32>;
/** Squared distance between two meshes, off the main thread. */
export async function distance2(m0: Mesh, m1: Mesh): Promise<number>;
export async function distance2(
  a: Primitive | Mesh, b: Primitive | Mesh,
): Promise<number | NDArrayFloat32> {
  if (a instanceof Mesh && b instanceof Mesh) {
    return dispatcher().run(
      () => native().dispatch_distance2_ff(a._handle, b._handle),
    );
  }
  if (a instanceof Mesh) {
    const p = b as Primitive;
    return dispatcher().run(
      () => native().dispatch_distance2_fp(
        a._handle, p._handle, primType(p),
      ),
      (raw) => typeof raw === "number" ? raw : new NDArray(raw, "float32"),
    );
  }
  const pa = a as Primitive;
  const pb = b as Primitive;
  return dispatcher().run(
    () => native().dispatch_distance2_pp(
      pa._handle, primType(pa), pb._handle, primType(pb),
    ),
    (raw) => typeof raw === "number" ? raw : new NDArray(raw, "float32"),
  );
}

// ============================================================================
// closestPoint
// ============================================================================

/** Closest point on B to A (prim × prim), off the main thread. */
export async function closestPoint(
  a: Primitive, b: Primitive,
): Promise<ClosestPointResult | ClosestPointBatchResult> {
  const isBatch = a.isBatch || b.isBatch;
  return dispatcher().run(
    () => native().dispatch_closest_metric_point(
      a._handle, primType(a), b._handle, primType(b),
    ),
    (raw) => isBatch ? wrapClosestPointBatch(raw) : wrapClosestPoint(raw),
  );
}

// ============================================================================
// closestPointPair
// ============================================================================

/** Closest pair of points between A and B (prim × prim), off the main thread. */
export async function closestPointPair(
  a: Primitive, b: Primitive,
): Promise<ClosestPointPairResult | ClosestPointPairBatchResult> {
  const isBatch = a.isBatch || b.isBatch;
  return dispatcher().run(
    () => native().dispatch_closest_metric_point_pair(
      a._handle, primType(a), b._handle, primType(b),
    ),
    (raw) => isBatch ? wrapClosestPointPairBatch(raw) : wrapClosestPointPair(raw),
  );
}

// ============================================================================
// neighborSearch
// ============================================================================

/** Find nearest element in a mesh, off the main thread. */
export async function neighborSearch(
  mesh: Mesh, query: Primitive,
): Promise<NeighborResult | NeighborBatchResult>;
/** Find nearest pair of elements between two meshes, off the main thread. */
export async function neighborSearch(
  m0: Mesh, m1: Mesh,
): Promise<NeighborPairResult>;
export async function neighborSearch(
  m0: Mesh, m1OrQuery: Mesh | Primitive,
): Promise<NeighborResult | NeighborBatchResult | NeighborPairResult> {
  if (m1OrQuery instanceof Mesh) {
    return dispatcher().run(
      () => native().dispatch_neighbor_search_ff(m0._handle, m1OrQuery._handle),
      wrapNeighborPair,
    );
  }
  const q = m1OrQuery;
  const isBatch = q.isBatch;
  return dispatcher().run(
    () => native().dispatch_neighbor_search_fp(
      m0._handle, q._handle, primType(q),
    ),
    (raw) => isBatch ? wrapNeighborBatch(raw) : wrapNeighbor(raw),
  );
}

// ============================================================================
// intersects
// ============================================================================

/** Intersection test between two primitives, off the main thread. */
export async function intersects(
  a: Primitive, b: Primitive,
): Promise<boolean | NDArrayBool>;
/** Intersection test between mesh and primitive, off the main thread. */
export async function intersects(
  mesh: Mesh, prim: Primitive,
): Promise<boolean | NDArrayBool>;
/** Intersection test between two meshes, off the main thread. */
export async function intersects(m0: Mesh, m1: Mesh): Promise<boolean>;
export async function intersects(
  a: Primitive | Mesh, b: Primitive | Mesh,
): Promise<boolean | NDArrayBool> {
  if (a instanceof Mesh && b instanceof Mesh) {
    return dispatcher().run(
      () => native().dispatch_intersects_ff(a._handle, b._handle),
    );
  }
  if (a instanceof Mesh) {
    const p = b as Primitive;
    return dispatcher().run(
      () => native().dispatch_intersects_fp(
        a._handle, p._handle, primType(p),
      ),
      (raw) => typeof raw === "boolean" ? raw : new NDArray(raw, "bool"),
    );
  }
  const pa = a as Primitive;
  const pb = b as Primitive;
  return dispatcher().run(
    () => native().dispatch_intersects_pp(
      pa._handle, primType(pa), pb._handle, primType(pb),
    ),
    (raw) => typeof raw === "boolean" ? raw : new NDArray(raw, "bool"),
  );
}

// ============================================================================
// rayCast
// ============================================================================

/** Cast a ray against a primitive, off the main thread. */
export async function rayCast(
  ray: Ray, target: Primitive, opts?: RayCastOptions,
): Promise<RayCastResult | RayCastPrimBatchResult>;
/** Cast a ray against a mesh, off the main thread. */
export async function rayCast(
  ray: Ray, mesh: Mesh, opts?: RayCastOptions,
): Promise<RayCastResult | RayCastFormBatchResult>;
export async function rayCast(
  ray: Ray, target: Primitive | Mesh, opts?: RayCastOptions,
): Promise<RayCastResult | RayCastPrimBatchResult | RayCastFormBatchResult> {
  const minT = opts?.minT ?? 0;
  const maxT = opts?.maxT ?? Infinity;
  if (target instanceof Mesh) {
    return dispatcher().run(
      () => native().dispatch_ray_cast_f(
        ray._handle, target._handle, minT, maxT,
      ),
      (raw) => {
        if (raw.hit !== undefined) return raw as RayCastResult;
        return {
          hits: new NDArray(raw.hits, "bool"),
          ts: new NDArray(raw.ts, "float32"),
          elementIds: new NDArray(raw.elementIds, "int32"),
        };
      },
    );
  }
  return dispatcher().run(
    () => native().dispatch_ray_cast_p(
      ray._handle, target._handle, primType(target), minT, maxT,
    ),
    (raw) => {
      if (raw.hit !== undefined) return raw as RayCastResult;
      return {
        hits: new NDArray(raw.hits, "bool"),
        ts: new NDArray(raw.ts, "float32"),
      };
    },
  );
}
