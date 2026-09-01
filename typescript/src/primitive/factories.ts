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
import {
  Primitive, PrimitiveType,
  Point, Vector, Segment, Triangle,
  Ray, Line, Plane, AABB, Polygon,
} from "./Primitive";
import { NDArray, NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import { coerce, disposeOwned } from "../internal/owned";

/** Options selecting the coordinate dtype of a primitive factory. */
export interface PrimitiveDtypeOptions {
  dtype?: "float32" | "float64";
}

type RealArray = Float32Array | Float64Array;
type ArrayLike32_64 = NDArrayFloat32 | NDArrayFloat64;

// ============================================================================
// Helpers
// ============================================================================

function isDtypeOpts(v: unknown): v is PrimitiveDtypeOptions {
  return typeof v === "object" && v !== null && !Array.isArray(v)
    && !(v instanceof NDArray) && !ArrayBuffer.isView(v)
    && "dtype" in (v as object);
}

function inferDtype(
  source: number[] | RealArray | NDArray | Primitive,
  override?: "float32" | "float64",
): "float32" | "float64" {
  if (override) return override;
  if (source instanceof NDArray) {
    return source.dtype === "float64" ? "float64" : "float32";
  }
  if (source instanceof Float64Array) return "float64";
  return "float32";
}

function toTypedArray(
  input: number[] | RealArray | Primitive,
  dtype: "float32" | "float64",
): RealArray {
  if (input instanceof Primitive) {
    const d = input.data as RealArray;
    if (dtype === "float64") {
      return d instanceof Float64Array ? d : Float64Array.from(d);
    }
    return d instanceof Float32Array ? d : Float32Array.from(d);
  }
  if (input instanceof Float64Array) {
    return dtype === "float64" ? input : Float32Array.from(input);
  }
  if (input instanceof Float32Array) {
    return dtype === "float64" ? Float64Array.from(input) : input;
  }
  return dtype === "float64" ? new Float64Array(input) : new Float32Array(input);
}

function makePrimitive(
  data: RealArray, shape: number[], type: PrimitiveType,
  dtype: "float32" | "float64",
): Primitive {
  const ctor = dtype === "float64"
    ? native().NativeFloat64NDArray
    : native().NativeFloat32NDArray;
  return new Primitive(ctor.from_js(data, shape), type, dtype);
}

function stackPrimitive(
  arrays: ArrayLike32_64[], type: PrimitiveType,
  dtype: "float32" | "float64",
): Primitive {
  const axis = arrays[0].ndim > 1 ? 1 : 0;
  const owned: NDArray[] = [];
  try {
    const handles = arrays.map((a) => coerce(a, dtype, owned)._handle);
    const fn = dtype === "float64" ? "stack_float64" : "stack_float32";
    return new Primitive(native()[fn](handles, axis), type, dtype);
  } finally {
    disposeOwned(owned);
  }
}

// ============================================================================
// Point
// ============================================================================

/** Create a single point from coordinates. */
export function point(...coords: number[]): Point;
/** Create a point or batch from flat data + element size. */
export function point(data: number[] | RealArray, dims?: number, opts?: PrimitiveDtypeOptions): Point;
/** Create a point or batch from an NDArray (no copy when dtype matches). */
export function point(arr: ArrayLike32_64, opts?: PrimitiveDtypeOptions): Point;
/** Create a point from coordinates with dtype override. */
export function point(...args: (number | PrimitiveDtypeOptions)[]): Point;
export function point(
  ...args: any[]
): Point {
  // Trailing { dtype } opts
  let dtypeOpt: "float32" | "float64" | undefined;
  if (args.length > 0 && isDtypeOpts(args[args.length - 1])) {
    dtypeOpt = (args.pop() as PrimitiveDtypeOptions).dtype;
  }

  // point(ndarray) — shallow_copy() bumps refcount so both sides own the buffer
  if (args[0] instanceof NDArray && !(args[0] instanceof Primitive)) {
    const arr = args[0] as ArrayLike32_64;
    const dt = inferDtype(arr, dtypeOpt);
    const owned: NDArray[] = [];
    try {
      const src = coerce(arr, dt, owned);
      return new Primitive(src._handle.shallow_copy(), "point", dt) as Point;
    } finally {
      disposeOwned(owned);
    }
  }
  // point(x, y) or point(x, y, z)
  if (typeof args[0] === "number") {
    const dt = dtypeOpt ?? "float32";
    const data = toTypedArray(args as number[], dt);
    return makePrimitive(data, [data.length], "point", dt) as Point;
  }
  // point(data) or point(data, dims)
  const src = args[0] as number[] | RealArray;
  const dt = inferDtype(src, dtypeOpt);
  const data = toTypedArray(src, dt);
  const dims = args[1] as number | undefined;
  if (dims !== undefined) {
    const n = data.length / dims;
    return makePrimitive(data, [n, dims], "point", dt) as Point;
  }
  return makePrimitive(data, [data.length], "point", dt) as Point;
}

// ============================================================================
// Vector
// ============================================================================

/** Create a single vector from components. */
export function vector(...coords: number[]): Vector;
/** Create a vector or batch from flat data + element size. */
export function vector(data: number[] | RealArray, dims?: number, opts?: PrimitiveDtypeOptions): Vector;
/** Create a vector or batch from an NDArray. */
export function vector(arr: ArrayLike32_64, opts?: PrimitiveDtypeOptions): Vector;
/** Create a vector from components with dtype override. */
export function vector(...args: (number | PrimitiveDtypeOptions)[]): Vector;
export function vector(
  ...args: any[]
): Vector {
  let dtypeOpt: "float32" | "float64" | undefined;
  if (args.length > 0 && isDtypeOpts(args[args.length - 1])) {
    dtypeOpt = (args.pop() as PrimitiveDtypeOptions).dtype;
  }

  if (args[0] instanceof NDArray && !(args[0] instanceof Primitive)) {
    const arr = args[0] as ArrayLike32_64;
    const dt = inferDtype(arr, dtypeOpt);
    const owned: NDArray[] = [];
    try {
      const src = coerce(arr, dt, owned);
      return new Primitive(src._handle.shallow_copy(), "vector", dt) as Vector;
    } finally {
      disposeOwned(owned);
    }
  }
  if (typeof args[0] === "number") {
    const dt = dtypeOpt ?? "float32";
    const data = toTypedArray(args as number[], dt);
    return makePrimitive(data, [data.length], "vector", dt) as Vector;
  }
  const src = args[0] as number[] | RealArray;
  const dt = inferDtype(src, dtypeOpt);
  const data = toTypedArray(src, dt);
  const dims = args[1] as number | undefined;
  if (dims !== undefined) {
    const n = data.length / dims;
    return makePrimitive(data, [n, dims], "vector", dt) as Vector;
  }
  return makePrimitive(data, [data.length], "vector", dt) as Vector;
}

// ============================================================================
// Segment
// ============================================================================

/** Create a segment from two point Primitives. */
export function segment(start: Point, end: Point, opts?: PrimitiveDtypeOptions): Segment;
/** Create a segment from two NDArrays (start points, end points). */
export function segment(start: ArrayLike32_64, end: ArrayLike32_64, opts?: PrimitiveDtypeOptions): Segment;
/** Create a segment from a flat array [start..., end...]. */
export function segment(data: number[] | RealArray, opts?: PrimitiveDtypeOptions): Segment;
/** Create a segment or batch from flat data + spatial dimension. */
export function segment(data: number[] | RealArray, dims: number, opts?: PrimitiveDtypeOptions): Segment;
export function segment(
  startOrData: Point | ArrayLike32_64 | number[] | RealArray,
  endOrDims?: Point | ArrayLike32_64 | number | PrimitiveDtypeOptions,
  maybeOpts?: PrimitiveDtypeOptions,
): Segment {
  let dtypeOpt: "float32" | "float64" | undefined;
  if (isDtypeOpts(maybeOpts)) dtypeOpt = maybeOpts.dtype;
  else if (isDtypeOpts(endOrDims)) {
    dtypeOpt = (endOrDims as PrimitiveDtypeOptions).dtype;
    endOrDims = undefined;
  }

  if (startOrData instanceof NDArray && !(startOrData instanceof Primitive)) {
    const a = startOrData as ArrayLike32_64;
    const b = endOrDims as ArrayLike32_64;
    const dt = inferDtype(a, dtypeOpt) === "float64" || b.dtype === "float64"
      ? (dtypeOpt ?? "float64") : (dtypeOpt ?? "float32");
    return stackPrimitive([a, b], "segment", dt) as Segment;
  }
  if (startOrData instanceof Primitive) {
    const a = startOrData as Point;
    const b = endOrDims as Point;
    const dt = dtypeOpt
      ?? (a.dtype === "float64" || b.dtype === "float64" ? "float64" : "float32");
    const s = toTypedArray(a, dt);
    const e = toTypedArray(b, dt);
    const dims = s.length;
    const buf = dt === "float64" ? new Float64Array(dims * 2) : new Float32Array(dims * 2);
    buf.set(s, 0);
    buf.set(e, dims);
    return makePrimitive(buf, [2, dims], "segment", dt) as Segment;
  }
  const src = startOrData as number[] | RealArray;
  const dt = inferDtype(src, dtypeOpt);
  const data = toTypedArray(src, dt);
  if (typeof endOrDims === "number") {
    const dims = endOrDims;
    const n = data.length / (2 * dims);
    return makePrimitive(data, [n, 2, dims], "segment", dt) as Segment;
  }
  const dims = data.length / 2;
  return makePrimitive(data, [2, dims], "segment", dt) as Segment;
}

// ============================================================================
// Triangle
// ============================================================================

/** Create a triangle from three point Primitives. */
export function triangle(a: Point, b: Point, c: Point, opts?: PrimitiveDtypeOptions): Triangle;
/** Create a triangle from three NDArrays (vertex sets). */
export function triangle(a: ArrayLike32_64, b: ArrayLike32_64, c: ArrayLike32_64, opts?: PrimitiveDtypeOptions): Triangle;
/** Create a triangle from a flat array [a..., b..., c...]. */
export function triangle(data: number[] | RealArray, opts?: PrimitiveDtypeOptions): Triangle;
/** Create a triangle or batch from flat data + spatial dimension. */
export function triangle(data: number[] | RealArray, dims: number, opts?: PrimitiveDtypeOptions): Triangle;
export function triangle(
  aOrData: Point | ArrayLike32_64 | number[] | RealArray,
  bOrDims?: Point | ArrayLike32_64 | number | PrimitiveDtypeOptions,
  c?: Point | ArrayLike32_64 | PrimitiveDtypeOptions,
  maybeOpts?: PrimitiveDtypeOptions,
): Triangle {
  let dtypeOpt: "float32" | "float64" | undefined;
  if (isDtypeOpts(maybeOpts)) dtypeOpt = maybeOpts.dtype;
  else if (isDtypeOpts(c)) { dtypeOpt = (c as PrimitiveDtypeOptions).dtype; c = undefined; }
  else if (isDtypeOpts(bOrDims)) { dtypeOpt = (bOrDims as PrimitiveDtypeOptions).dtype; bOrDims = undefined; }

  if (aOrData instanceof NDArray && !(aOrData instanceof Primitive)) {
    const a = aOrData as ArrayLike32_64;
    const b = bOrDims as ArrayLike32_64;
    const cc = c as ArrayLike32_64;
    const dt = dtypeOpt
      ?? (a.dtype === "float64" || b.dtype === "float64" || cc.dtype === "float64"
        ? "float64" : "float32");
    return stackPrimitive([a, b, cc], "triangle", dt) as Triangle;
  }
  if (aOrData instanceof Primitive) {
    const a = aOrData as Point;
    const b = bOrDims as Point;
    const cc = c as Point;
    const dt = dtypeOpt
      ?? (a.dtype === "float64" || b.dtype === "float64" || cc.dtype === "float64"
        ? "float64" : "float32");
    const da = toTypedArray(a, dt);
    const db = toTypedArray(b, dt);
    const dc = toTypedArray(cc, dt);
    const dims = da.length;
    const buf = dt === "float64" ? new Float64Array(dims * 3) : new Float32Array(dims * 3);
    buf.set(da, 0);
    buf.set(db, dims);
    buf.set(dc, dims * 2);
    return makePrimitive(buf, [3, dims], "triangle", dt) as Triangle;
  }
  const src = aOrData as number[] | RealArray;
  const dt = inferDtype(src, dtypeOpt);
  const data = toTypedArray(src, dt);
  if (typeof bOrDims === "number") {
    const dims = bOrDims;
    const n = data.length / (3 * dims);
    return makePrimitive(data, [n, 3, dims], "triangle", dt) as Triangle;
  }
  const dims = data.length / 3;
  return makePrimitive(data, [3, dims], "triangle", dt) as Triangle;
}

// ============================================================================
// Ray
// ============================================================================

/** Create a ray from origin Point and direction Vector. */
export function ray(origin: Point, direction: Vector, opts?: PrimitiveDtypeOptions): Ray;
/** Create a ray from two NDArrays (origins, directions). */
export function ray(origins: ArrayLike32_64, directions: ArrayLike32_64, opts?: PrimitiveDtypeOptions): Ray;
/** Create a ray from a flat array [origin..., direction...]. */
export function ray(data: number[] | RealArray, opts?: PrimitiveDtypeOptions): Ray;
/** Create a ray or batch from flat data + spatial dimension. */
export function ray(data: number[] | RealArray, dims: number, opts?: PrimitiveDtypeOptions): Ray;
export function ray(
  originOrData: Point | ArrayLike32_64 | number[] | RealArray,
  dirOrDims?: Vector | ArrayLike32_64 | number | PrimitiveDtypeOptions,
  maybeOpts?: PrimitiveDtypeOptions,
): Ray {
  let dtypeOpt: "float32" | "float64" | undefined;
  if (isDtypeOpts(maybeOpts)) dtypeOpt = maybeOpts.dtype;
  else if (isDtypeOpts(dirOrDims)) {
    dtypeOpt = (dirOrDims as PrimitiveDtypeOptions).dtype;
    dirOrDims = undefined;
  }

  if (originOrData instanceof NDArray && !(originOrData instanceof Primitive)) {
    const o = originOrData as ArrayLike32_64;
    const d = dirOrDims as ArrayLike32_64;
    const dt = dtypeOpt
      ?? (o.dtype === "float64" || d.dtype === "float64" ? "float64" : "float32");
    return stackPrimitive([o, d], "ray", dt) as Ray;
  }
  if (originOrData instanceof Primitive) {
    const o = originOrData as Point;
    const d = dirOrDims as Vector;
    const dt = dtypeOpt
      ?? (o.dtype === "float64" || d.dtype === "float64" ? "float64" : "float32");
    const od = toTypedArray(o, dt);
    const dd = toTypedArray(d, dt);
    const dims = od.length;
    const buf = dt === "float64" ? new Float64Array(dims * 2) : new Float32Array(dims * 2);
    buf.set(od, 0);
    buf.set(dd, dims);
    return makePrimitive(buf, [2, dims], "ray", dt) as Ray;
  }
  const src = originOrData as number[] | RealArray;
  const dt = inferDtype(src, dtypeOpt);
  const data = toTypedArray(src, dt);
  if (typeof dirOrDims === "number") {
    const dims = dirOrDims;
    const n = data.length / (2 * dims);
    return makePrimitive(data, [n, 2, dims], "ray", dt) as Ray;
  }
  const dims = data.length / 2;
  return makePrimitive(data, [2, dims], "ray", dt) as Ray;
}

// ============================================================================
// Line
// ============================================================================

/** Create a line from origin Point and direction Vector. */
export function line(origin: Point, direction: Vector, opts?: PrimitiveDtypeOptions): Line;
/** Create a line from two NDArrays (origins, directions). */
export function line(origins: ArrayLike32_64, directions: ArrayLike32_64, opts?: PrimitiveDtypeOptions): Line;
/** Create a line from a flat array [origin..., direction...]. */
export function line(data: number[] | RealArray, opts?: PrimitiveDtypeOptions): Line;
/** Create a line or batch from flat data + spatial dimension. */
export function line(data: number[] | RealArray, dims: number, opts?: PrimitiveDtypeOptions): Line;
export function line(
  originOrData: Point | ArrayLike32_64 | number[] | RealArray,
  dirOrDims?: Vector | ArrayLike32_64 | number | PrimitiveDtypeOptions,
  maybeOpts?: PrimitiveDtypeOptions,
): Line {
  let dtypeOpt: "float32" | "float64" | undefined;
  if (isDtypeOpts(maybeOpts)) dtypeOpt = maybeOpts.dtype;
  else if (isDtypeOpts(dirOrDims)) {
    dtypeOpt = (dirOrDims as PrimitiveDtypeOptions).dtype;
    dirOrDims = undefined;
  }

  if (originOrData instanceof NDArray && !(originOrData instanceof Primitive)) {
    const o = originOrData as ArrayLike32_64;
    const d = dirOrDims as ArrayLike32_64;
    const dt = dtypeOpt
      ?? (o.dtype === "float64" || d.dtype === "float64" ? "float64" : "float32");
    return stackPrimitive([o, d], "line", dt) as Line;
  }
  if (originOrData instanceof Primitive) {
    const o = originOrData as Point;
    const d = dirOrDims as Vector;
    const dt = dtypeOpt
      ?? (o.dtype === "float64" || d.dtype === "float64" ? "float64" : "float32");
    const od = toTypedArray(o, dt);
    const dd = toTypedArray(d, dt);
    const dims = od.length;
    const buf = dt === "float64" ? new Float64Array(dims * 2) : new Float32Array(dims * 2);
    buf.set(od, 0);
    buf.set(dd, dims);
    return makePrimitive(buf, [2, dims], "line", dt) as Line;
  }
  const src = originOrData as number[] | RealArray;
  const dt = inferDtype(src, dtypeOpt);
  const data = toTypedArray(src, dt);
  if (typeof dirOrDims === "number") {
    const dims = dirOrDims;
    const n = data.length / (2 * dims);
    return makePrimitive(data, [n, 2, dims], "line", dt) as Line;
  }
  const dims = data.length / 2;
  return makePrimitive(data, [2, dims], "line", dt) as Line;
}

// ============================================================================
// Plane
// ============================================================================

/** Create a plane from normal Vector and offset. */
export function plane(normal: Vector, offset: number, opts?: PrimitiveDtypeOptions): Plane;
/** Create a plane from normal NDArray and offset. */
export function plane(normal: ArrayLike32_64, offset: number, opts?: PrimitiveDtypeOptions): Plane;
/** Create a plane or batch from packed coefficients shaped [4] or [N, 4]. */
export function plane(coefficients: ArrayLike32_64, opts?: PrimitiveDtypeOptions): Plane;
/** Create a plane from a flat [a, b, c, d] array. */
export function plane(data: number[] | RealArray, opts?: PrimitiveDtypeOptions): Plane;
export function plane(
  normalOrData: Vector | ArrayLike32_64 | number[] | RealArray,
  offsetOrOpts?: number | PrimitiveDtypeOptions,
  maybeOpts?: PrimitiveDtypeOptions,
): Plane {
  let dtypeOpt: "float32" | "float64" | undefined;
  if (isDtypeOpts(maybeOpts)) dtypeOpt = maybeOpts.dtype;
  else if (isDtypeOpts(offsetOrOpts)) {
    dtypeOpt = (offsetOrOpts as PrimitiveDtypeOptions).dtype;
    offsetOrOpts = undefined;
  }

  if (normalOrData instanceof NDArray) {
    const arr = normalOrData as ArrayLike32_64;
    const dt = inferDtype(arr, dtypeOpt);
    if (typeof offsetOrOpts !== "number") {
      if (!((arr.ndim === 1 && arr.shape[0] === 4)
        || (arr.ndim === 2 && arr.shape[1] === 4))) {
        throw new Error(
          `plane coefficients must have shape [4] or [N,4] (got [${arr.shape.join(",")}])`,
        );
      }
      const owned: NDArray[] = [];
      try {
        const source = coerce(arr, dt, owned);
        return new Primitive(source._handle.shallow_copy(), "plane", dt) as Plane;
      } finally {
        disposeOwned(owned);
      }
    }
    const n = arr.data as RealArray;
    const offset = offsetOrOpts as number;
    const buf = dt === "float64"
      ? new Float64Array([n[0], n[1], n[2], offset])
      : new Float32Array([n[0], n[1], n[2], offset]);
    return makePrimitive(buf, [4], "plane", dt) as Plane;
  }
  const src = normalOrData as number[] | RealArray;
  const dt = inferDtype(src, dtypeOpt);
  const data = toTypedArray(src, dt);
  if (data.length === 4) {
    return makePrimitive(data, [4], "plane", dt) as Plane;
  }
  const count = data.length / 4;
  return makePrimitive(data, [count, 4], "plane", dt) as Plane;
}

// ============================================================================
// AABB
// ============================================================================

/** Create an AABB from min and max Points. */
export function aabb(min: Point, max: Point, opts?: PrimitiveDtypeOptions): AABB;
/** Create an AABB from two NDArrays (mins, maxs). */
export function aabb(mins: ArrayLike32_64, maxs: ArrayLike32_64, opts?: PrimitiveDtypeOptions): AABB;
/** Create an AABB from a flat array [min..., max...]. */
export function aabb(data: number[] | RealArray, opts?: PrimitiveDtypeOptions): AABB;
/** Create an AABB or batch from flat data + spatial dimension. */
export function aabb(data: number[] | RealArray, dims: number, opts?: PrimitiveDtypeOptions): AABB;
export function aabb(
  minOrData: Point | ArrayLike32_64 | number[] | RealArray,
  maxOrDims?: Point | ArrayLike32_64 | number | PrimitiveDtypeOptions,
  maybeOpts?: PrimitiveDtypeOptions,
): AABB {
  let dtypeOpt: "float32" | "float64" | undefined;
  if (isDtypeOpts(maybeOpts)) dtypeOpt = maybeOpts.dtype;
  else if (isDtypeOpts(maxOrDims)) {
    dtypeOpt = (maxOrDims as PrimitiveDtypeOptions).dtype;
    maxOrDims = undefined;
  }

  if (minOrData instanceof NDArray && !(minOrData instanceof Primitive)) {
    const lo = minOrData as ArrayLike32_64;
    const hi = maxOrDims as ArrayLike32_64;
    const dt = dtypeOpt
      ?? (lo.dtype === "float64" || hi.dtype === "float64" ? "float64" : "float32");
    return stackPrimitive([lo, hi], "aabb", dt) as AABB;
  }
  if (minOrData instanceof Primitive) {
    const lo = minOrData as Point;
    const hi = maxOrDims as Point;
    const dt = dtypeOpt
      ?? (lo.dtype === "float64" || hi.dtype === "float64" ? "float64" : "float32");
    const ld = toTypedArray(lo, dt);
    const hd = toTypedArray(hi, dt);
    const dims = ld.length;
    const buf = dt === "float64" ? new Float64Array(dims * 2) : new Float32Array(dims * 2);
    buf.set(ld, 0);
    buf.set(hd, dims);
    return makePrimitive(buf, [2, dims], "aabb", dt) as AABB;
  }
  const src = minOrData as number[] | RealArray;
  const dt = inferDtype(src, dtypeOpt);
  const data = toTypedArray(src, dt);
  if (typeof maxOrDims === "number") {
    const dims = maxOrDims;
    const n = data.length / (2 * dims);
    return makePrimitive(data, [n, 2, dims], "aabb", dt) as AABB;
  }
  const dims = data.length / 2;
  return makePrimitive(data, [2, dims], "aabb", dt) as AABB;
}

/** Oriented bounding box result (dtype follows the input). */
export interface OBB {
  /** Corner origin point [3]. */
  origin: NDArray;
  /** Orthonormal axes, each row is a unit vector [3, 3]. */
  axes: NDArray;
  /** Full extents along each axis [3]. */
  extent: NDArray;
}

/** Compute the oriented bounding box of points or a mesh. */
export function obbFrom(input: NDArray): OBB;
export function obbFrom(input: { points: NDArray }): OBB;
export function obbFrom(input: NDArray | { points: NDArray }): OBB {
  const pts = input instanceof NDArray ? input : input.points;
  const dt = pts.dtype === "float64" ? "float64" : "float32";
  const owned: NDArray[] = [];
  try {
    const raw = native()[`obb_from_${dt}`](coerce(pts, dt, owned)._handle);
    return {
      origin: new NDArray(raw.origin, dt),
      axes: new NDArray(raw.axes, dt),
      extent: new NDArray(raw.extent, dt),
    };
  } finally {
    disposeOwned(owned);
  }
}

/** Compute the axis-aligned bounding box of points, a mesh, or a point cloud. */
export function aabbFrom(input: ArrayLike32_64): AABB;
export function aabbFrom(input: { points: ArrayLike32_64 }): AABB;
export function aabbFrom(input: ArrayLike32_64 | { points: ArrayLike32_64 }): AABB {
  const pts = input instanceof NDArray ? input : input.points;
  return aabb(pts.min(0) as ArrayLike32_64, pts.max(0) as ArrayLike32_64);
}

// ============================================================================
// Polygon
// ============================================================================

/** Create a polygon from an array of Point Primitives. */
export function polygon(vertices: Point[], opts?: PrimitiveDtypeOptions): Polygon;
/** Create a polygon or batch from an NDArray (no copy when dtype matches). */
export function polygon(arr: ArrayLike32_64, opts?: PrimitiveDtypeOptions): Polygon;
/** Create a polygon from a flat array + dimension. */
export function polygon(data: number[] | RealArray, dims: number, opts?: PrimitiveDtypeOptions): Polygon;
export function polygon(
  verticesOrData: Point[] | ArrayLike32_64 | number[] | RealArray,
  dimsOrOpts?: number | PrimitiveDtypeOptions,
  maybeOpts?: PrimitiveDtypeOptions,
): Polygon {
  let dtypeOpt: "float32" | "float64" | undefined;
  if (isDtypeOpts(maybeOpts)) dtypeOpt = maybeOpts.dtype;
  else if (isDtypeOpts(dimsOrOpts)) {
    dtypeOpt = (dimsOrOpts as PrimitiveDtypeOptions).dtype;
    dimsOrOpts = undefined;
  }

  if (verticesOrData instanceof NDArray && !(verticesOrData instanceof Primitive)) {
    const arr = verticesOrData as ArrayLike32_64;
    const dt = inferDtype(arr, dtypeOpt);
    const owned: NDArray[] = [];
    try {
      const src = coerce(arr, dt, owned);
      return new Primitive(src._handle.shallow_copy(), "polygon", dt) as Polygon;
    } finally {
      disposeOwned(owned);
    }
  }
  if (Array.isArray(verticesOrData) && verticesOrData.length > 0
      && verticesOrData[0] instanceof Primitive) {
    const verts = verticesOrData as Point[];
    const dt = dtypeOpt
      ?? (verts.some((v) => v.dtype === "float64") ? "float64" : "float32");
    const d = verts[0].data.length;
    const buf = dt === "float64"
      ? new Float64Array(verts.length * d)
      : new Float32Array(verts.length * d);
    for (let i = 0; i < verts.length; i++) {
      const td = toTypedArray(verts[i], dt);
      buf.set(td, i * d);
    }
    return makePrimitive(buf, [verts.length, d], "polygon", dt) as Polygon;
  }
  const src = verticesOrData as number[] | RealArray;
  const dt = inferDtype(src, dtypeOpt);
  const data = toTypedArray(src, dt);
  const d = dimsOrOpts as number;
  const numVerts = data.length / d;
  return makePrimitive(data, [numVerts, d], "polygon", dt) as Polygon;
}
