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
import { PointCloud } from "../form/PointCloud";
import { Primitive, PrimitiveType, Ray } from "../primitive/Primitive";
import { NDArray, NDArrayFloat32, NDArrayBool, NDArrayInt32 } from "../ndarray/NDArray";
import { full } from "../ndarray/factories";

// ============================================================================
// Form type — Mesh or PointCloud
// ============================================================================

/** A spatial form: either a Mesh or a PointCloud. */
export type Form = Mesh | PointCloud;

function isPC(f: Form): f is PointCloud {
  return f instanceof PointCloud;
}

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

/** Result of k-NN neighbor search (single query). */
export interface NeighborKnnResult {
  /** Element indices [count]. */
  elementIds: NDArrayInt32;
  /** Closest points on the form [count, 3]. */
  points: NDArrayFloat32;
  /** Squared distances [count]. */
  distances: NDArrayFloat32;
}

/** Result of k-NN neighbor search (batch query). */
export interface NeighborKnnBatchResult {
  /** Element indices [N, k] padded with -1. */
  elementIds: NDArrayInt32;
  /** Closest points on the form [N, k, 3]. */
  points: NDArrayFloat32;
  /** Squared distances [N, k]. */
  distances: NDArrayFloat32;
  /** Actual count per query [N]. */
  counts: NDArrayInt32;
}

/** Options for k-NN neighbor search. */
export interface KnnOptions {
  /** Number of nearest neighbors to find. */
  k: number;
  /** Maximum search radius (default: Infinity). */
  radius?: number;
}

/** Options for neighbor search. */
export interface NeighborSearchOptions {
  /** Maximum search radius (default: Infinity). */
  radius?: number;
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
  /** Minimum ray parameter. Scalar applies to all rays; NDArrayFloat32 [N] for per-ray bounds. Default: 0. */
  minT?: number | NDArrayFloat32;
  /** Maximum ray parameter. Scalar applies to all rays; NDArrayFloat32 [N] for per-ray bounds. Default: Infinity. */
  maxT?: number | NDArrayFloat32;
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

function wrapNeighborKnn(raw: any): NeighborKnnResult {
  return {
    elementIds: new NDArray(raw.elementIds, "int32"),
    points: new NDArray(raw.points, "float32"),
    distances: new NDArray(raw.distances, "float32"),
  };
}

function wrapNeighborKnnBatch(raw: any): NeighborKnnBatchResult {
  return {
    elementIds: new NDArray(raw.elementIds, "int32"),
    points: new NDArray(raw.points, "float32"),
    distances: new NDArray(raw.distances, "float32"),
    counts: new NDArray(raw.counts, "int32"),
  };
}

// ============================================================================
// FP/FF dispatch helpers
// ============================================================================

function fpSuffix(form: Form): string {
  return isPC(form) ? "_pc" : "";
}

function ffSuffix(a: Form, b: Form): string {
  if (isPC(a)) return isPC(b) ? "_pc" : "_pm";
  return isPC(b) ? "_mp" : "";
}

// ============================================================================
// distance2 — squared distance
// ============================================================================

/** Squared distance between two primitives. */
export function distance2(
  a: Primitive, b: Primitive,
): number | NDArrayFloat32;
/** Squared distance from a form to a primitive. */
export function distance2(
  form: Form, prim: Primitive,
): number | NDArrayFloat32;
/** Squared distance between two forms. */
export function distance2(a: Form, b: Form): number;
export function distance2(
  a: Primitive | Form, b: Primitive | Form,
): number | NDArrayFloat32 {
  if ((a instanceof Mesh || a instanceof PointCloud) &&
      (b instanceof Mesh || b instanceof PointCloud)) {
    return native()["distance2_ff" + ffSuffix(a, b)](a._handle, b._handle);
  }
  if (a instanceof Mesh || a instanceof PointCloud) {
    const p = b as Primitive;
    const raw = native()["distance2_fp" + fpSuffix(a)](
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
// distance — (signed for plane-point, Euclidean otherwise)
// ============================================================================

/** Distance between two primitives (signed for plane-point). */
export function distance(
  a: Primitive, b: Primitive,
): number | NDArrayFloat32;
/** Distance from a form to a primitive. */
export function distance(
  form: Form, prim: Primitive,
): number | NDArrayFloat32;
/** Distance between two forms. */
export function distance(a: Form, b: Form): number;
export function distance(
  a: Primitive | Form, b: Primitive | Form,
): number | NDArrayFloat32 {
  if ((a instanceof Mesh || a instanceof PointCloud) &&
      (b instanceof Mesh || b instanceof PointCloud)) {
    return native()["distance_ff" + ffSuffix(a, b)](a._handle, b._handle);
  }
  if (a instanceof Mesh || a instanceof PointCloud) {
    const p = b as Primitive;
    const raw = native()["distance_fp" + fpSuffix(a)](
      a._handle, p._handle, primType(p),
    );
    return typeof raw === "number" ? raw : new NDArray(raw, "float32");
  }
  const pa = a as Primitive;
  const pb = b as Primitive;
  const raw = native().distance_pp(
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
// neighborSearch — find nearest element in a form
// ============================================================================

/** Find the nearest element in a form to a primitive. */
export function neighborSearch(
  form: Form, query: Primitive, opts?: NeighborSearchOptions,
): NeighborResult | NeighborBatchResult;
/** Find the nearest pair of elements between two forms. */
export function neighborSearch(
  a: Form, b: Form, opts?: NeighborSearchOptions,
): NeighborPairResult;
/** Find the k nearest elements in a form to a primitive. */
export function neighborSearch(
  form: Form, query: Primitive, opts: KnnOptions,
): NeighborKnnResult | NeighborKnnBatchResult;
export function neighborSearch(
  a: Form, bOrQuery: Form | Primitive,
  opts?: NeighborSearchOptions | KnnOptions,
): NeighborResult | NeighborBatchResult | NeighborPairResult
  | NeighborKnnResult | NeighborKnnBatchResult {
  const radius = opts?.radius ?? Infinity;

  // FF — form × form
  if (bOrQuery instanceof Mesh || bOrQuery instanceof PointCloud) {
    return wrapNeighborPair(
      native()["neighbor_search_ff" + ffSuffix(a, bOrQuery)](
        a._handle, bOrQuery._handle, radius,
      ),
    );
  }

  // FP with k-NN
  const q = bOrQuery;
  if (opts && "k" in opts) {
    const raw = native()["neighbor_search_fp_knn" + fpSuffix(a)](
      a._handle, q._handle, primType(q), opts.k, radius,
    );
    if (q.isBatch) return wrapNeighborKnnBatch(raw);
    return wrapNeighborKnn(raw);
  }

  // FP — basic neighbor search
  const raw = native()["neighbor_search_fp" + fpSuffix(a)](
    a._handle, q._handle, primType(q), radius,
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
/** Test intersection between a form and a primitive. */
export function intersects(
  form: Form, prim: Primitive,
): boolean | NDArrayBool;
/** Test intersection between two forms. */
export function intersects(a: Form, b: Form): boolean;
export function intersects(
  a: Primitive | Form, b: Primitive | Form,
): boolean | NDArrayBool {
  if ((a instanceof Mesh || a instanceof PointCloud) &&
      (b instanceof Mesh || b instanceof PointCloud)) {
    return native()["intersects_ff" + ffSuffix(a, b)](a._handle, b._handle);
  }
  if (a instanceof Mesh || a instanceof PointCloud) {
    const p = b as Primitive;
    const raw = native()["intersects_fp" + fpSuffix(a)](
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
/** Cast a ray against a form. */
export function rayCast(
  ray: Ray, form: Form, opts?: RayCastOptions,
): RayCastResult | RayCastFormBatchResult;
export function rayCast(
  ray: Ray, target: Primitive | Form, opts?: RayCastOptions,
): RayCastResult | RayCastPrimBatchResult | RayCastFormBatchResult {
  const minT = opts?.minT ?? 0;
  const maxT = opts?.maxT ?? Infinity;
  const bothScalar = typeof minT === "number" && typeof maxT === "number";

  if (!bothScalar) {
    const n = ray.count;
    const minH = (minT instanceof NDArray ? minT : full("float32", [n], minT as number))._handle;
    const maxH = (maxT instanceof NDArray ? maxT : full("float32", [n], maxT as number))._handle;

    if (target instanceof Mesh || target instanceof PointCloud) {
      const fn = isPC(target) ? "ray_cast_f_pc_rc" : "ray_cast_f_rc";
      const raw = native()[fn](ray._handle, target._handle, minH, maxH);
      if (raw.hit !== undefined) return raw as RayCastResult;
      return {
        hits: new NDArray(raw.hits, "bool"),
        ts: new NDArray(raw.ts, "float32"),
        elementIds: new NDArray(raw.elementIds, "int32"),
      };
    }
    const raw = native().ray_cast_p_rc(
      ray._handle, target._handle, primType(target), minH, maxH,
    );
    if (raw.hit !== undefined) return raw as RayCastResult;
    return {
      hits: new NDArray(raw.hits, "bool"),
      ts: new NDArray(raw.ts, "float32"),
    };
  }

  if (target instanceof Mesh || target instanceof PointCloud) {
    const fn = isPC(target) ? "ray_cast_f_pc" : "ray_cast_f";
    const raw = native()[fn](
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
    ray._handle, target._handle, primType(target), minT as number, maxT as number,
  );
  if (raw.hit !== undefined) return raw as RayCastResult;
  return {
    hits: new NDArray(raw.hits, "bool"),
    ts: new NDArray(raw.ts, "float32"),
  };
}
