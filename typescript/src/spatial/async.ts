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
import { NDArray, NDArrayFloat32, NDArrayBool } from "../ndarray/NDArray";
import { full } from "../ndarray/factories";
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
// distance2
// ============================================================================

/** Squared distance between two primitives, off the main thread. */
export async function distance2(
  a: Primitive, b: Primitive,
): Promise<number | NDArrayFloat32>;
/** Squared distance from a form to a primitive, off the main thread. */
export async function distance2(
  form: Form, prim: Primitive,
): Promise<number | NDArrayFloat32>;
/** Squared distance between two forms, off the main thread. */
export async function distance2(a: Form, b: Form): Promise<number>;
export async function distance2(
  a: Primitive | Form, b: Primitive | Form,
): Promise<number | NDArrayFloat32> {
  if ((a instanceof Mesh || a instanceof PointCloud) &&
      (b instanceof Mesh || b instanceof PointCloud)) {
    return dispatcher().run(
      () => native()["dispatch_distance2_ff" + ffSuffix(a, b)](a._handle, b._handle),
    );
  }
  if (a instanceof Mesh || a instanceof PointCloud) {
    const p = b as Primitive;
    return dispatcher().run(
      () => native()["dispatch_distance2_fp" + fpSuffix(a)](
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
// distance
// ============================================================================

/** Distance between two primitives (signed for plane-point), off the main thread. */
export async function distance(
  a: Primitive, b: Primitive,
): Promise<number | NDArrayFloat32>;
/** Distance from a form to a primitive, off the main thread. */
export async function distance(
  form: Form, prim: Primitive,
): Promise<number | NDArrayFloat32>;
/** Distance between two forms, off the main thread. */
export async function distance(a: Form, b: Form): Promise<number>;
export async function distance(
  a: Primitive | Form, b: Primitive | Form,
): Promise<number | NDArrayFloat32> {
  if ((a instanceof Mesh || a instanceof PointCloud) &&
      (b instanceof Mesh || b instanceof PointCloud)) {
    return dispatcher().run(
      () => native()["dispatch_distance_ff" + ffSuffix(a, b)](a._handle, b._handle),
    );
  }
  if (a instanceof Mesh || a instanceof PointCloud) {
    const p = b as Primitive;
    return dispatcher().run(
      () => native()["dispatch_distance_fp" + fpSuffix(a)](
        a._handle, p._handle, primType(p),
      ),
      (raw) => typeof raw === "number" ? raw : new NDArray(raw, "float32"),
    );
  }
  const pa = a as Primitive;
  const pb = b as Primitive;
  return dispatcher().run(
    () => native().dispatch_distance_pp(
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
    return dispatcher().run(
      () => native()["dispatch_neighbor_search_ff" + ffSuffix(a, bOrQuery)](
        a._handle, bOrQuery._handle, radius,
      ),
      wrapNeighborPair,
    );
  }

  // FP with k-NN
  const q = bOrQuery;
  if (opts && "k" in opts) {
    const isBatch = q.isBatch;
    return dispatcher().run(
      () => native()["dispatch_neighbor_search_fp_knn" + fpSuffix(a)](
        a._handle, q._handle, primType(q), opts.k, radius,
      ),
      (raw) => isBatch ? wrapNeighborKnnBatch(raw) : wrapNeighborKnn(raw),
    );
  }

  // FP — basic neighbor search
  const isBatch = q.isBatch;
  return dispatcher().run(
    () => native()["dispatch_neighbor_search_fp" + fpSuffix(a)](
      a._handle, q._handle, primType(q), radius,
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
    return dispatcher().run(
      () => native()["dispatch_intersects_ff" + ffSuffix(a, b)](a._handle, b._handle),
    );
  }
  if (a instanceof Mesh || a instanceof PointCloud) {
    const p = b as Primitive;
    return dispatcher().run(
      () => native()["dispatch_intersects_fp" + fpSuffix(a)](
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

  if (!bothScalar) {
    const n = ray.count;
    const minH = (minT instanceof NDArray ? minT : full("float32", [n], minT as number))._handle;
    const maxH = (maxT instanceof NDArray ? maxT : full("float32", [n], maxT as number))._handle;

    if (target instanceof Mesh || target instanceof PointCloud) {
      const fn = isPC(target) ? "dispatch_ray_cast_f_pc_rc" : "dispatch_ray_cast_f_rc";
      return dispatcher().run(
        () => native()[fn](ray._handle, target._handle, minH, maxH),
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
      () => native().dispatch_ray_cast_p_rc(
        ray._handle, target._handle, primType(target), minH, maxH,
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

  if (target instanceof Mesh || target instanceof PointCloud) {
    const fn = isPC(target) ? "dispatch_ray_cast_f_pc" : "dispatch_ray_cast_f";
    return dispatcher().run(
      () => native()[fn](
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
      ray._handle, target._handle, primType(target), minT as number, maxT as number,
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

// ============ Precompute ============

/** Build the spatial AABB tree off the main thread. Cached on the mesh or point cloud. */
export async function buildTree(m: Mesh | PointCloud): Promise<void> {
  if (m instanceof Mesh) {
    await dispatcher().run(() => native().dispatch_ensure(m._handle, 0));
  } else {
    await dispatcher().run(() => native().dispatch_ensure_pc(m._handle));
  }
}
