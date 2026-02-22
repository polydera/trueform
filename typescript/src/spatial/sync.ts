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
import { Primitive, PrimitiveType, Ray } from "../primitive/Primitive";
import { NDArray, NDArrayFloat32, NDArrayBool, NDArrayInt32 } from "../ndarray/NDArray";

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
// Result interfaces
// ============================================================================

/** Result of closest-point query (single). */
export interface ClosestPointResult {
  /** Closest point on B to A. */
  point: NDArrayFloat32;
  /** Squared distance. */
  distance2: number;
}

/** Result of closest-point query (batch). */
export interface ClosestPointBatchResult {
  /** Closest points [N, 3]. */
  points: NDArrayFloat32;
  /** Squared distances [N]. */
  distances: NDArrayFloat32;
}

/** Result of closest-point-pair query (single). */
export interface ClosestPointPairResult {
  /** Closest point on A. */
  point0: NDArrayFloat32;
  /** Closest point on B. */
  point1: NDArrayFloat32;
  /** Squared distance. */
  distance2: number;
}

/** Result of closest-point-pair query (batch). */
export interface ClosestPointPairBatchResult {
  /** Closest points on A [N, 3]. */
  points0: NDArrayFloat32;
  /** Closest points on B [N, 3]. */
  points1: NDArrayFloat32;
  /** Squared distances [N]. */
  distances: NDArrayFloat32;
}

/** Result of neighbor search (form × prim). */
export interface NeighborResult {
  /** Index of the closest element in the form. */
  elementId: number;
  /** Closest point on the form. */
  point: NDArrayFloat32;
  /** Squared distance. */
  distance2: number;
}

/** Result of neighbor search (form × batch prim). */
export interface NeighborBatchResult {
  /** Element indices [N]. */
  elementIds: NDArrayInt32;
  /** Closest points on the form [N, 3]. */
  points: NDArrayFloat32;
  /** Squared distances [N]. */
  distances: NDArrayFloat32;
}

/** Result of neighbor search (form × form). */
export interface NeighborPairResult {
  /** Index of the closest element on form 0. */
  elementId0: number;
  /** Index of the closest element on form 1. */
  elementId1: number;
  /** Closest point on form 0. */
  point0: NDArrayFloat32;
  /** Closest point on form 1. */
  point1: NDArrayFloat32;
  /** Squared distance. */
  distance2: number;
}

/** Result of ray cast (single). */
export interface RayCastResult {
  /** Whether the ray hit the target. */
  hit: boolean;
  /** Parameter t along the ray at the hit point. */
  t: number;
  /** Element index (-1 for primitive targets). */
  elementId: number;
}

/** Result of ray cast against primitives (batch). */
export interface RayCastPrimBatchResult {
  /** Hit flags [N]. */
  hits: NDArrayBool;
  /** Ray parameters [N]. */
  ts: NDArrayFloat32;
}

/** Result of ray cast against a form (batch). */
export interface RayCastFormBatchResult {
  /** Hit flags [N]. */
  hits: NDArrayBool;
  /** Ray parameters [N]. */
  ts: NDArrayFloat32;
  /** Element indices [N]. */
  elementIds: NDArrayInt32;
}

/** Options for ray cast queries. */
export interface RayCastOptions {
  /** Minimum ray parameter (default: 0). */
  minT?: number;
  /** Maximum ray parameter (default: Infinity). */
  maxT?: number;
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
// distance2 — squared distance
// ============================================================================

/** Squared distance between two primitives. */
export function distance2(
  a: Primitive, b: Primitive,
): number | NDArrayFloat32;
/** Squared distance from a mesh to a primitive. */
export function distance2(
  mesh: Mesh, prim: Primitive,
): number | NDArrayFloat32;
/** Squared distance between two meshes. */
export function distance2(m0: Mesh, m1: Mesh): number;
export function distance2(
  a: Primitive | Mesh, b: Primitive | Mesh,
): number | NDArrayFloat32 {
  if (a instanceof Mesh && b instanceof Mesh) {
    return native().distance2_ff(a._handle, b._handle);
  }
  if (a instanceof Mesh) {
    const p = b as Primitive;
    const raw = native().distance2_fp(
      a._handle, p._handle, primType(p),
    );
    return typeof raw === "number" ? raw : new NDArray(raw, "float32");
  }
  const pa = a as Primitive;
  const pb = b as Primitive;
  const raw = native().distance2_pp(
    pa._handle, primType(pa), pb._handle, primType(pb),
  );
  return typeof raw === "number" ? raw : new NDArray(raw, "float32");
}

// ============================================================================
// closestPoint — closest point on B to A
// ============================================================================

/** Closest point on B to A (prim × prim). */
export function closestPoint(
  a: Primitive, b: Primitive,
): ClosestPointResult | ClosestPointBatchResult {
  const raw = native().closest_metric_point(
    a._handle, primType(a), b._handle, primType(b),
  );
  if (a.isBatch || b.isBatch) return wrapClosestPointBatch(raw);
  return wrapClosestPoint(raw);
}

// ============================================================================
// closestPointPair — closest pair of points between A and B
// ============================================================================

/** Closest pair of points between A and B (prim × prim). */
export function closestPointPair(
  a: Primitive, b: Primitive,
): ClosestPointPairResult | ClosestPointPairBatchResult {
  const raw = native().closest_metric_point_pair(
    a._handle, primType(a), b._handle, primType(b),
  );
  if (a.isBatch || b.isBatch) return wrapClosestPointPairBatch(raw);
  return wrapClosestPointPair(raw);
}

// ============================================================================
// neighborSearch — find nearest element in a mesh
// ============================================================================

/** Find the nearest element in a mesh to a primitive. */
export function neighborSearch(
  mesh: Mesh, query: Primitive,
): NeighborResult | NeighborBatchResult;
/** Find the nearest pair of elements between two meshes. */
export function neighborSearch(m0: Mesh, m1: Mesh): NeighborPairResult;
export function neighborSearch(
  m0: Mesh, m1OrQuery: Mesh | Primitive,
): NeighborResult | NeighborBatchResult | NeighborPairResult {
  if (m1OrQuery instanceof Mesh) {
    return wrapNeighborPair(
      native().neighbor_search_ff(m0._handle, m1OrQuery._handle),
    );
  }
  const q = m1OrQuery;
  const raw = native().neighbor_search_fp(
    m0._handle, q._handle, primType(q),
  );
  if (q.isBatch) return wrapNeighborBatch(raw);
  return wrapNeighbor(raw);
}

// ============================================================================
// intersects — intersection test
// ============================================================================

/** Test intersection between two primitives. */
export function intersects(
  a: Primitive, b: Primitive,
): boolean | NDArrayBool;
/** Test intersection between a mesh and a primitive. */
export function intersects(
  mesh: Mesh, prim: Primitive,
): boolean | NDArrayBool;
/** Test intersection between two meshes. */
export function intersects(m0: Mesh, m1: Mesh): boolean;
export function intersects(
  a: Primitive | Mesh, b: Primitive | Mesh,
): boolean | NDArrayBool {
  if (a instanceof Mesh && b instanceof Mesh) {
    return native().intersects_ff(a._handle, b._handle);
  }
  if (a instanceof Mesh) {
    const p = b as Primitive;
    const raw = native().intersects_fp(
      a._handle, p._handle, primType(p),
    );
    return typeof raw === "boolean" ? raw : new NDArray(raw, "bool");
  }
  const pa = a as Primitive;
  const pb = b as Primitive;
  const raw = native().intersects_pp(
    pa._handle, primType(pa), pb._handle, primType(pb),
  );
  return typeof raw === "boolean" ? raw : new NDArray(raw, "bool");
}

// ============================================================================
// rayCast — ray casting
// ============================================================================

/** Cast a ray against a primitive target. */
export function rayCast(
  ray: Ray, target: Primitive, opts?: RayCastOptions,
): RayCastResult | RayCastPrimBatchResult;
/** Cast a ray against a mesh. */
export function rayCast(
  ray: Ray, mesh: Mesh, opts?: RayCastOptions,
): RayCastResult | RayCastFormBatchResult;
export function rayCast(
  ray: Ray, target: Primitive | Mesh, opts?: RayCastOptions,
): RayCastResult | RayCastPrimBatchResult | RayCastFormBatchResult {
  const minT = opts?.minT ?? 0;
  const maxT = opts?.maxT ?? Infinity;
  if (target instanceof Mesh) {
    const raw = native().ray_cast_f(
      ray._handle, target._handle, minT, maxT,
    );
    if (raw.hit !== undefined) return raw as RayCastResult;
    return {
      hits: new NDArray(raw.hits, "bool"),
      ts: new NDArray(raw.ts, "float32"),
      elementIds: new NDArray(raw.elementIds, "int32"),
    };
  }
  const raw = native().ray_cast_p(
    ray._handle, target._handle, primType(target), minT, maxT,
  );
  if (raw.hit !== undefined) return raw as RayCastResult;
  return {
    hits: new NDArray(raw.hits, "bool"),
    ts: new NDArray(raw.ts, "float32"),
  };
}
