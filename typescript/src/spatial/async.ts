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
import { PointCloud } from "../form/PointCloud";
import { Primitive, PrimitiveType, Ray } from "../primitive/Primitive";
import { NDArray, NDArrayFloat32, NDArrayFloat64, NDArrayBool } from "../ndarray/NDArray";
import { full } from "../ndarray/factories";
import { assertSameDtype } from "../internal/dtype";
import { coerce, own, disposeOwned } from "../internal/owned";
import type {
  Form,
  ClosestPointResult, ClosestPointBatchResult,
  ClosestPointPairResult, ClosestPointPairBatchResult,
  NeighborResult, NeighborBatchResult, NeighborPairResult,
  NeighborKnnResult, NeighborKnnBatchResult,
  KnnOptions, NeighborSearchOptions,
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

function isPC(f: Form): f is PointCloud {
  return f instanceof PointCloud;
}

function fpSuffix(form: Form): string {
  return isPC(form) ? "_pc" : "";
}

function ffSuffix(a: Form, b: Form): string {
  if (isPC(a)) return isPC(b) ? "_pc" : "_pm";
  return isPC(b) ? "_mp" : "";
}

// ============================================================================
// Result wrappers
// ============================================================================

function wrapClosestPoint(raw: any, dtype: string): ClosestPointResult {
  return {
    point: new NDArray(raw.point, dtype) as NDArrayFloat32 | NDArrayFloat64,
    distance2: raw.distance2,
  };
}

function wrapClosestPointBatch(raw: any, dtype: string): ClosestPointBatchResult {
  return {
    points: new NDArray(raw.points, dtype) as NDArrayFloat32 | NDArrayFloat64,
    distances: new NDArray(raw.distances, dtype) as NDArrayFloat32 | NDArrayFloat64,
  };
}

function wrapClosestPointPair(raw: any, dtype: string): ClosestPointPairResult {
  return {
    point0: new NDArray(raw.point0, dtype) as NDArrayFloat32 | NDArrayFloat64,
    point1: new NDArray(raw.point1, dtype) as NDArrayFloat32 | NDArrayFloat64,
    distance2: raw.distance2,
  };
}

function wrapClosestPointPairBatch(raw: any, dtype: string): ClosestPointPairBatchResult {
  return {
    points0: new NDArray(raw.points0, dtype) as NDArrayFloat32 | NDArrayFloat64,
    points1: new NDArray(raw.points1, dtype) as NDArrayFloat32 | NDArrayFloat64,
    distances: new NDArray(raw.distances, dtype) as NDArrayFloat32 | NDArrayFloat64,
  };
}

function wrapNeighbor(raw: any, dtype: string): NeighborResult {
  return {
    elementId: raw.elementId,
    point: new NDArray(raw.point, dtype) as NDArrayFloat32 | NDArrayFloat64,
    distance2: raw.distance2,
  };
}

function wrapNeighborBatch(raw: any, dtype: string): NeighborBatchResult {
  return {
    elementIds: new NDArray(raw.elementIds, "int32"),
    points: new NDArray(raw.points, dtype) as NDArrayFloat32 | NDArrayFloat64,
    distances: new NDArray(raw.distances, dtype) as NDArrayFloat32 | NDArrayFloat64,
  };
}

function wrapNeighborPair(raw: any, dtype: string): NeighborPairResult {
  return {
    elementId0: raw.elementId0,
    elementId1: raw.elementId1,
    point0: new NDArray(raw.point0, dtype) as NDArrayFloat32 | NDArrayFloat64,
    point1: new NDArray(raw.point1, dtype) as NDArrayFloat32 | NDArrayFloat64,
    distance2: raw.distance2,
  };
}

function wrapNeighborKnn(raw: any, dtype: string): NeighborKnnResult {
  return {
    elementIds: new NDArray(raw.elementIds, "int32"),
    points: new NDArray(raw.points, dtype) as NDArrayFloat32 | NDArrayFloat64,
    distances: new NDArray(raw.distances, dtype) as NDArrayFloat32 | NDArrayFloat64,
  };
}

function wrapNeighborKnnBatch(raw: any, dtype: string): NeighborKnnBatchResult {
  return {
    elementIds: new NDArray(raw.elementIds, "int32"),
    points: new NDArray(raw.points, dtype) as NDArrayFloat32 | NDArrayFloat64,
    distances: new NDArray(raw.distances, dtype) as NDArrayFloat32 | NDArrayFloat64,
    counts: new NDArray(raw.counts, "int32"),
  };
}

// ============================================================================
// distance2
// ============================================================================

/** Squared distance between two primitives, off the main thread. */
export async function distance2(
  a: Primitive, b: Primitive,
): Promise<number | NDArrayFloat32 | NDArrayFloat64>;
/** Squared distance from a form to a primitive, off the main thread. */
export async function distance2(
  form: Form, prim: Primitive,
): Promise<number | NDArrayFloat32 | NDArrayFloat64>;
/** Squared distance between two forms, off the main thread. */
export async function distance2(a: Form, b: Form): Promise<number>;
export async function distance2(
  a: Primitive | Form, b: Primitive | Form,
): Promise<number | NDArrayFloat32 | NDArrayFloat64> {
  if ((a instanceof Mesh || a instanceof PointCloud) &&
      (b instanceof Mesh || b instanceof PointCloud)) {
    assertSameDtype([a, b], ["form 0", "form 1"]);
    return dispatcher().run(
      () => native()[`dispatch_distance2_ff${ffSuffix(a, b)}_${a.dtype}`](
        a._handle, b._handle,
      ),
    );
  }
  const owned: NDArray[] = [];
  try {
    if (a instanceof Mesh || a instanceof PointCloud) {
      const p = b as Primitive;
      const dtype = a.dtype;
      const ph = coerce(p, dtype, owned)._handle;
      return await dispatcher().run(
        () => native()[`dispatch_distance2_fp${fpSuffix(a)}_${dtype}`](
          a._handle, ph, primType(p),
        ),
        (raw) => {
          if (typeof raw === "number") return raw;
          return dtype === "float32"
            ? (new NDArray(raw, "float32") as NDArrayFloat32)
            : (new NDArray(raw, "float64") as NDArrayFloat64);
        },
      );
    }
    const pa = a as Primitive;
    const pb = b as Primitive;
    const dt: "float32" | "float64" =
      (pa.dtype === "float64" || pb.dtype === "float64") ? "float64" : "float32";
    const ah = coerce(pa, dt, owned)._handle;
    const bh = coerce(pb, dt, owned)._handle;
    return await dispatcher().run(
      () => native()[`dispatch_distance2_pp_${dt}`](
        ah, primType(pa), bh, primType(pb),
      ),
      (raw) => {
        if (typeof raw === "number") return raw;
        return dt === "float32"
          ? (new NDArray(raw, "float32") as NDArrayFloat32)
          : (new NDArray(raw, "float64") as NDArrayFloat64);
      },
    );
  } finally {
    disposeOwned(owned);
  }
}

// ============================================================================
// distance
// ============================================================================

/** Distance between two primitives (signed for plane-point), off the main thread. */
export async function distance(
  a: Primitive, b: Primitive,
): Promise<number | NDArrayFloat32 | NDArrayFloat64>;
/** Distance from a form to a primitive, off the main thread. */
export async function distance(
  form: Form, prim: Primitive,
): Promise<number | NDArrayFloat32 | NDArrayFloat64>;
/** Distance between two forms, off the main thread. */
export async function distance(a: Form, b: Form): Promise<number>;
export async function distance(
  a: Primitive | Form, b: Primitive | Form,
): Promise<number | NDArrayFloat32 | NDArrayFloat64> {
  if ((a instanceof Mesh || a instanceof PointCloud) &&
      (b instanceof Mesh || b instanceof PointCloud)) {
    assertSameDtype([a, b], ["form 0", "form 1"]);
    return dispatcher().run(
      () => native()[`dispatch_distance_ff${ffSuffix(a, b)}_${a.dtype}`](
        a._handle, b._handle,
      ),
    );
  }
  const owned: NDArray[] = [];
  try {
    if (a instanceof Mesh || a instanceof PointCloud) {
      const p = b as Primitive;
      const dtype = a.dtype;
      const ph = coerce(p, dtype, owned)._handle;
      return await dispatcher().run(
        () => native()[`dispatch_distance_fp${fpSuffix(a)}_${dtype}`](
          a._handle, ph, primType(p),
        ),
        (raw) => {
          if (typeof raw === "number") return raw;
          return dtype === "float32"
            ? (new NDArray(raw, "float32") as NDArrayFloat32)
            : (new NDArray(raw, "float64") as NDArrayFloat64);
        },
      );
    }
    const pa = a as Primitive;
    const pb = b as Primitive;
    const dt: "float32" | "float64" =
      (pa.dtype === "float64" || pb.dtype === "float64") ? "float64" : "float32";
    const ah = coerce(pa, dt, owned)._handle;
    const bh = coerce(pb, dt, owned)._handle;
    return await dispatcher().run(
      () => native()[`dispatch_distance_pp_${dt}`](
        ah, primType(pa), bh, primType(pb),
      ),
      (raw) => {
        if (typeof raw === "number") return raw;
        return dt === "float32"
          ? (new NDArray(raw, "float32") as NDArrayFloat32)
          : (new NDArray(raw, "float64") as NDArrayFloat64);
      },
    );
  } finally {
    disposeOwned(owned);
  }
}

// ============================================================================
// closestPoint
// ============================================================================

/** Closest point on B to A (prim × prim), off the main thread. */
export async function closestPoint(
  a: Primitive, b: Primitive,
): Promise<ClosestPointResult | ClosestPointBatchResult> {
  const isBatch = a.isBatch || b.isBatch;
  const dt: "float32" | "float64" =
    (a.dtype === "float64" || b.dtype === "float64") ? "float64" : "float32";
  const owned: NDArray[] = [];
  try {
    const ah = coerce(a, dt, owned)._handle;
    const bh = coerce(b, dt, owned)._handle;
    return await dispatcher().run(
      () => native()[`dispatch_closest_metric_point_${dt}`](
        ah, primType(a), bh, primType(b),
      ),
      (raw) => isBatch ? wrapClosestPointBatch(raw, dt) : wrapClosestPoint(raw, dt),
    );
  } finally {
    disposeOwned(owned);
  }
}

// ============================================================================
// closestPointPair
// ============================================================================

/** Closest pair of points between A and B (prim × prim), off the main thread. */
export async function closestPointPair(
  a: Primitive, b: Primitive,
): Promise<ClosestPointPairResult | ClosestPointPairBatchResult> {
  const isBatch = a.isBatch || b.isBatch;
  const dt: "float32" | "float64" =
    (a.dtype === "float64" || b.dtype === "float64") ? "float64" : "float32";
  const owned: NDArray[] = [];
  try {
    const ah = coerce(a, dt, owned)._handle;
    const bh = coerce(b, dt, owned)._handle;
    return await dispatcher().run(
      () => native()[`dispatch_closest_metric_point_pair_${dt}`](
        ah, primType(a), bh, primType(b),
      ),
      (raw) => isBatch ? wrapClosestPointPairBatch(raw, dt) : wrapClosestPointPair(raw, dt),
    );
  } finally {
    disposeOwned(owned);
  }
}

// ============================================================================
// neighborSearch
// ============================================================================

/** Find nearest element in a form, off the main thread. */
export async function neighborSearch(
  form: Form, query: Primitive, opts?: NeighborSearchOptions,
): Promise<NeighborResult | NeighborBatchResult>;
/** Find nearest pair of elements between two forms, off the main thread. */
export async function neighborSearch(
  a: Form, b: Form, opts?: NeighborSearchOptions,
): Promise<NeighborPairResult>;
/** Find the k nearest elements in a form, off the main thread. */
export async function neighborSearch(
  form: Form, query: Primitive, opts: KnnOptions,
): Promise<NeighborKnnResult | NeighborKnnBatchResult>;
export async function neighborSearch(
  a: Form, bOrQuery: Form | Primitive,
  opts?: NeighborSearchOptions | KnnOptions,
): Promise<NeighborResult | NeighborBatchResult | NeighborPairResult
  | NeighborKnnResult | NeighborKnnBatchResult> {
  const radius = opts?.radius ?? Infinity;

  // FF — form × form
  if (bOrQuery instanceof Mesh || bOrQuery instanceof PointCloud) {
    assertSameDtype([a, bOrQuery], ["form 0", "form 1"]);
    const dtype = a.dtype;
    return dispatcher().run(
      () => native()[`dispatch_neighbor_search_ff${ffSuffix(a, bOrQuery)}_${dtype}`](
        a._handle, bOrQuery._handle, radius,
      ),
      (raw) => wrapNeighborPair(raw, dtype),
    );
  }

  // FP — primitive query (basic or k-NN)
  const q = bOrQuery;
  const dtype = a.dtype;
  const owned: NDArray[] = [];
  try {
    const qh = coerce(q, dtype, owned)._handle;

    if (opts && "k" in opts) {
      const isBatch = q.isBatch;
      return await dispatcher().run(
        () => native()[`dispatch_neighbor_search_fp_knn${fpSuffix(a)}_${dtype}`](
          a._handle, qh, primType(q), opts.k, radius,
        ),
        (raw) => isBatch ? wrapNeighborKnnBatch(raw, dtype) : wrapNeighborKnn(raw, dtype),
      );
    }

    const isBatch = q.isBatch;
    return await dispatcher().run(
      () => native()[`dispatch_neighbor_search_fp${fpSuffix(a)}_${dtype}`](
        a._handle, qh, primType(q), radius,
      ),
      (raw) => isBatch ? wrapNeighborBatch(raw, dtype) : wrapNeighbor(raw, dtype),
    );
  } finally {
    disposeOwned(owned);
  }
}

// ============================================================================
// intersects
// ============================================================================

/** Intersection test between two primitives, off the main thread. */
export async function intersects(
  a: Primitive, b: Primitive,
): Promise<boolean | NDArrayBool>;
/** Intersection test between form and primitive, off the main thread. */
export async function intersects(
  form: Form, prim: Primitive,
): Promise<boolean | NDArrayBool>;
/** Intersection test between two forms, off the main thread. */
export async function intersects(a: Form, b: Form): Promise<boolean>;
export async function intersects(
  a: Primitive | Form, b: Primitive | Form,
): Promise<boolean | NDArrayBool> {
  if ((a instanceof Mesh || a instanceof PointCloud) &&
      (b instanceof Mesh || b instanceof PointCloud)) {
    assertSameDtype([a, b], ["form 0", "form 1"]);
    return dispatcher().run(
      () => native()[`dispatch_intersects_ff${ffSuffix(a, b)}_${a.dtype}`](
        a._handle, b._handle,
      ),
    );
  }
  const owned: NDArray[] = [];
  try {
    if (a instanceof Mesh || a instanceof PointCloud) {
      const p = b as Primitive;
      const ph = coerce(p, a.dtype, owned)._handle;
      return await dispatcher().run(
        () => native()[`dispatch_intersects_fp${fpSuffix(a)}_${a.dtype}`](
          a._handle, ph, primType(p),
        ),
        (raw) => typeof raw === "boolean" ? raw : new NDArray(raw, "bool"),
      );
    }
    const pa = a as Primitive;
    const pb = b as Primitive;
    const dt: "float32" | "float64" =
      (pa.dtype === "float64" || pb.dtype === "float64") ? "float64" : "float32";
    const ah = coerce(pa, dt, owned)._handle;
    const bh = coerce(pb, dt, owned)._handle;
    return await dispatcher().run(
      () => native()[`dispatch_intersects_pp_${dt}`](
        ah, primType(pa), bh, primType(pb),
      ),
      (raw) => typeof raw === "boolean" ? raw : new NDArray(raw, "bool"),
    );
  } finally {
    disposeOwned(owned);
  }
}

// ============================================================================
// rayCast
// ============================================================================

/** Cast a ray against a primitive, off the main thread. */
export async function rayCast(
  ray: Ray, target: Primitive, opts?: RayCastOptions,
): Promise<RayCastResult | RayCastPrimBatchResult>;
/** Cast a ray against a form, off the main thread. */
export async function rayCast(
  ray: Ray, form: Form, opts?: RayCastOptions,
): Promise<RayCastResult | RayCastFormBatchResult>;
export async function rayCast(
  ray: Ray, target: Primitive | Form, opts?: RayCastOptions,
): Promise<RayCastResult | RayCastPrimBatchResult | RayCastFormBatchResult> {
  const minT = opts?.minT ?? 0;
  const maxT = opts?.maxT ?? Infinity;
  const bothScalar = typeof minT === "number" && typeof maxT === "number";
  const owned: NDArray[] = [];

  try {
    if (!bothScalar) {
      const n = ray.count;

      if (target instanceof Mesh || target instanceof PointCloud) {
        const dtype = target.dtype;
        const rayInput = coerce(ray, dtype, owned);
        const minArr = minT instanceof NDArray
          ? coerce(minT, dtype, owned)
          : own(full(dtype, [n], minT as number), owned);
        const maxArr = maxT instanceof NDArray
          ? coerce(maxT, dtype, owned)
          : own(full(dtype, [n], maxT as number), owned);
        const fn = isPC(target)
          ? `dispatch_ray_cast_f_pc_rc_${dtype}`
          : `dispatch_ray_cast_f_rc_${dtype}`;
        return await dispatcher().run(
          () => native()[fn](
            rayInput._handle, target._handle, minArr._handle, maxArr._handle,
          ),
          (raw) => {
            if (raw.hit !== undefined) return raw as RayCastResult;
            return {
              hits: new NDArray(raw.hits, "bool"),
              ts: new NDArray(raw.ts, dtype) as NDArrayFloat32 | NDArrayFloat64,
              elementIds: new NDArray(raw.elementIds, "int32"),
            };
          },
        );
      }
      const primitive = target as Primitive;
      const dt: "float32" | "float64" =
        (ray.dtype === "float64" || primitive.dtype === "float64")
          ? "float64" : "float32";
      const rayInput = coerce(ray, dt, owned);
      const targetInput = coerce(primitive, dt, owned);
      const minArr = minT instanceof NDArray
        ? coerce(minT, dt, owned)
        : own(full(dt, [n], minT as number), owned);
      const maxArr = maxT instanceof NDArray
        ? coerce(maxT, dt, owned)
        : own(full(dt, [n], maxT as number), owned);
      return await dispatcher().run(
        () => native()[`dispatch_ray_cast_p_rc_${dt}`](
          rayInput._handle, targetInput._handle, primType(primitive),
          minArr._handle, maxArr._handle,
        ),
        (raw) => {
          if (raw.hit !== undefined) return raw as RayCastResult;
          return {
            hits: new NDArray(raw.hits, "bool"),
            ts: new NDArray(raw.ts, dt) as NDArrayFloat32 | NDArrayFloat64,
          } as RayCastPrimBatchResult;
        },
      );
    }

    if (target instanceof Mesh || target instanceof PointCloud) {
      const dtype = target.dtype;
      const rayInput = coerce(ray, dtype, owned);
      const fn = isPC(target)
        ? `dispatch_ray_cast_f_pc_${dtype}`
        : `dispatch_ray_cast_f_${dtype}`;
      return await dispatcher().run(
        () => native()[fn](
          rayInput._handle, target._handle, minT, maxT,
        ),
        (raw) => {
          if (raw.hit !== undefined) return raw as RayCastResult;
          return {
            hits: new NDArray(raw.hits, "bool"),
            ts: new NDArray(raw.ts, dtype) as NDArrayFloat32 | NDArrayFloat64,
            elementIds: new NDArray(raw.elementIds, "int32"),
          };
        },
      );
    }
    const primitive = target as Primitive;
    const dt: "float32" | "float64" =
      (ray.dtype === "float64" || primitive.dtype === "float64")
        ? "float64" : "float32";
    const rayInput = coerce(ray, dt, owned);
    const targetInput = coerce(primitive, dt, owned);
    return await dispatcher().run(
      () => native()[`dispatch_ray_cast_p_${dt}`](
        rayInput._handle, targetInput._handle, primType(primitive),
        minT as number, maxT as number,
      ),
      (raw) => {
        if (raw.hit !== undefined) return raw as RayCastResult;
        return {
          hits: new NDArray(raw.hits, "bool"),
          ts: new NDArray(raw.ts, dt) as NDArrayFloat32 | NDArrayFloat64,
        } as RayCastPrimBatchResult;
      },
    );
  } finally {
    disposeOwned(owned);
  }
}

// ============ Precompute ============

/** Build the spatial AABB tree off the main thread. Cached on the mesh or point cloud. */
export async function buildTree(m: Mesh | PointCloud): Promise<void> {
  if (m instanceof Mesh) {
    const fn = native()[`dispatch_ensure_${m.dtype}`];
    await dispatcher().run(() => fn(m._handle, 0));
  } else {
    const fn = native()[`dispatch_ensure_pc_${m.dtype}`];
    await dispatcher().run(() => fn(m._handle));
  }
}
