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
import { NDArray, NDArrayFloat32 } from "../ndarray/NDArray";

// ============================================================================
// Helpers
// ============================================================================

function makePrimitive(
  data: Float32Array, shape: number[], type: PrimitiveType,
): Primitive {
  return new Primitive(native().NativeFloat32NDArray.from_js(data, shape), type);
}

function toF32(input: number[] | Float32Array | Primitive): Float32Array {
  if (input instanceof Primitive) return input.data;
  if (input instanceof Float32Array) return input;
  return new Float32Array(input);
}

function stackPrimitive(
  arrays: NDArrayFloat32[], type: PrimitiveType,
): Primitive {
  const axis = arrays[0].ndim > 1 ? 1 : 0;
  const handles = arrays.map((a) => a._handle);
  return new Primitive(native().stack_float32(handles, axis), type);
}

// ============================================================================
// Point
// ============================================================================

/** Create a single point from coordinates. */
export function point(...coords: number[]): Point;
/** Create a point or batch from flat data + element size. */
export function point(data: number[] | Float32Array, dims?: number): Point;
/** Create a point or batch from an NDArray (no copy). */
export function point(arr: NDArrayFloat32): Point;
export function point(
  ...args: any[]
): Point {
  // point(ndarray) — shared_view() bumps refcount so both sides own the buffer
  if (args[0] instanceof NDArray && !(args[0] instanceof Primitive)) {
    return new Primitive(args[0]._handle.shared_view(), "point") as Point;
  }
  // point(x, y) or point(x, y, z)
  if (typeof args[0] === "number") {
    const f32 = new Float32Array(args);
    return makePrimitive(f32, [f32.length], "point") as Point;
  }
  // point(data) or point(data, dims)
  const f32 = toF32(args[0]);
  const dims = args[1] as number | undefined;
  if (dims !== undefined) {
    const n = f32.length / dims;
    return makePrimitive(f32, [n, dims], "point") as Point;
  }
  return makePrimitive(f32, [f32.length], "point") as Point;
}

// ============================================================================
// Vector
// ============================================================================

/** Create a single vector from components. */
export function vector(...coords: number[]): Vector;
/** Create a vector or batch from flat data + element size. */
export function vector(data: number[] | Float32Array, dims?: number): Vector;
/** Create a vector or batch from an NDArray (no copy). */
export function vector(arr: NDArrayFloat32): Vector;
export function vector(
  ...args: any[]
): Vector {
  if (args[0] instanceof NDArray && !(args[0] instanceof Primitive)) {
    return new Primitive(args[0]._handle.shared_view(), "vector") as Vector;
  }
  if (typeof args[0] === "number") {
    const f32 = new Float32Array(args);
    return makePrimitive(f32, [f32.length], "vector") as Vector;
  }
  const f32 = toF32(args[0]);
  const dims = args[1] as number | undefined;
  if (dims !== undefined) {
    const n = f32.length / dims;
    return makePrimitive(f32, [n, dims], "vector") as Vector;
  }
  return makePrimitive(f32, [f32.length], "vector") as Vector;
}

// ============================================================================
// Segment
// ============================================================================

/** Create a segment from two point Primitives. */
export function segment(start: Point, end: Point): Segment;
/** Create a segment from two NDArrays (start points, end points). */
export function segment(start: NDArrayFloat32, end: NDArrayFloat32): Segment;
/** Create a segment from a flat array [start..., end...]. */
export function segment(data: number[] | Float32Array): Segment;
/** Create a segment or batch from flat data + spatial dimension. */
export function segment(data: number[] | Float32Array, dims: number): Segment;
export function segment(
  startOrData: Point | NDArrayFloat32 | number[] | Float32Array,
  endOrDims?: Point | NDArrayFloat32 | number,
): Segment {
  if (startOrData instanceof NDArray && !(startOrData instanceof Primitive)) {
    return stackPrimitive([startOrData, endOrDims as NDArrayFloat32], "segment") as Segment;
  }
  if (startOrData instanceof Primitive) {
    const s = startOrData.data;
    const e = (endOrDims as Point).data;
    const dims = s.length;
    const f32 = new Float32Array(dims * 2);
    f32.set(s, 0);
    f32.set(e, dims);
    return makePrimitive(f32, [2, dims], "segment") as Segment;
  }
  const f32 = toF32(startOrData);
  if (typeof endOrDims === "number") {
    const dims = endOrDims;
    const n = f32.length / (2 * dims);
    return makePrimitive(f32, [n, 2, dims], "segment") as Segment;
  }
  const dims = f32.length / 2;
  return makePrimitive(f32, [2, dims], "segment") as Segment;
}

// ============================================================================
// Triangle
// ============================================================================

/** Create a triangle from three point Primitives. */
export function triangle(a: Point, b: Point, c: Point): Triangle;
/** Create a triangle from three NDArrays (vertex sets). */
export function triangle(a: NDArrayFloat32, b: NDArrayFloat32, c: NDArrayFloat32): Triangle;
/** Create a triangle from a flat array [a..., b..., c...]. */
export function triangle(data: number[] | Float32Array): Triangle;
/** Create a triangle or batch from flat data + spatial dimension. */
export function triangle(data: number[] | Float32Array, dims: number): Triangle;
export function triangle(
  aOrData: Point | NDArrayFloat32 | number[] | Float32Array,
  bOrDims?: Point | NDArrayFloat32 | number,
  c?: Point | NDArrayFloat32,
): Triangle {
  if (aOrData instanceof NDArray && !(aOrData instanceof Primitive)) {
    return stackPrimitive([aOrData, bOrDims as NDArrayFloat32, c as NDArrayFloat32], "triangle") as Triangle;
  }
  if (aOrData instanceof Primitive) {
    const da = aOrData.data;
    const db = (bOrDims as Point).data;
    const dc = (c as Point).data;
    const dims = da.length;
    const f32 = new Float32Array(dims * 3);
    f32.set(da, 0);
    f32.set(db, dims);
    f32.set(dc, dims * 2);
    return makePrimitive(f32, [3, dims], "triangle") as Triangle;
  }
  const f32 = toF32(aOrData);
  if (typeof bOrDims === "number") {
    const dims = bOrDims;
    const n = f32.length / (3 * dims);
    return makePrimitive(f32, [n, 3, dims], "triangle") as Triangle;
  }
  const dims = f32.length / 3;
  return makePrimitive(f32, [3, dims], "triangle") as Triangle;
}

// ============================================================================
// Ray
// ============================================================================

/** Create a ray from origin Point and direction Vector. */
export function ray(origin: Point, direction: Vector): Ray;
/** Create a ray from two NDArrays (origins, directions). */
export function ray(origins: NDArrayFloat32, directions: NDArrayFloat32): Ray;
/** Create a ray from a flat array [origin..., direction...]. */
export function ray(data: number[] | Float32Array): Ray;
/** Create a ray or batch from flat data + spatial dimension. */
export function ray(data: number[] | Float32Array, dims: number): Ray;
export function ray(
  originOrData: Point | NDArrayFloat32 | number[] | Float32Array,
  dirOrDims?: Vector | NDArrayFloat32 | number,
): Ray {
  if (originOrData instanceof NDArray && !(originOrData instanceof Primitive)) {
    return stackPrimitive([originOrData, dirOrDims as NDArrayFloat32], "ray") as Ray;
  }
  if (originOrData instanceof Primitive) {
    const o = originOrData.data;
    const d = (dirOrDims as Vector).data;
    const dims = o.length;
    const f32 = new Float32Array(dims * 2);
    f32.set(o, 0);
    f32.set(d, dims);
    return makePrimitive(f32, [2, dims], "ray") as Ray;
  }
  const f32 = toF32(originOrData);
  if (typeof dirOrDims === "number") {
    const dims = dirOrDims;
    const n = f32.length / (2 * dims);
    return makePrimitive(f32, [n, 2, dims], "ray") as Ray;
  }
  const dims = f32.length / 2;
  return makePrimitive(f32, [2, dims], "ray") as Ray;
}

// ============================================================================
// Line
// ============================================================================

/** Create a line from origin Point and direction Vector. */
export function line(origin: Point, direction: Vector): Line;
/** Create a line from two NDArrays (origins, directions). */
export function line(origins: NDArrayFloat32, directions: NDArrayFloat32): Line;
/** Create a line from a flat array [origin..., direction...]. */
export function line(data: number[] | Float32Array): Line;
/** Create a line or batch from flat data + spatial dimension. */
export function line(data: number[] | Float32Array, dims: number): Line;
export function line(
  originOrData: Point | NDArrayFloat32 | number[] | Float32Array,
  dirOrDims?: Vector | NDArrayFloat32 | number,
): Line {
  if (originOrData instanceof NDArray && !(originOrData instanceof Primitive)) {
    return stackPrimitive([originOrData, dirOrDims as NDArrayFloat32], "line") as Line;
  }
  if (originOrData instanceof Primitive) {
    const o = originOrData.data;
    const d = (dirOrDims as Vector).data;
    const dims = o.length;
    const f32 = new Float32Array(dims * 2);
    f32.set(o, 0);
    f32.set(d, dims);
    return makePrimitive(f32, [2, dims], "line") as Line;
  }
  const f32 = toF32(originOrData);
  if (typeof dirOrDims === "number") {
    const dims = dirOrDims;
    const n = f32.length / (2 * dims);
    return makePrimitive(f32, [n, 2, dims], "line") as Line;
  }
  const dims = f32.length / 2;
  return makePrimitive(f32, [2, dims], "line") as Line;
}

// ============================================================================
// Plane
// ============================================================================

/** Create a plane from normal Vector and offset. */
export function plane(normal: Vector, offset: number): Plane;
/** Create a plane from normal NDArray and offset. */
export function plane(normal: NDArrayFloat32, offset: number): Plane;
/** Create a plane from a flat [a, b, c, d] array. */
export function plane(data: number[] | Float32Array): Plane;
export function plane(
  normalOrData: Vector | NDArrayFloat32 | number[] | Float32Array,
  offset?: number,
): Plane {
  if (normalOrData instanceof NDArray) {
    const n = normalOrData.data as Float32Array;
    const f32 = new Float32Array([n[0], n[1], n[2], offset!]);
    return makePrimitive(f32, [4], "plane") as Plane;
  }
  const f32 = toF32(normalOrData);
  if (f32.length === 4) {
    return makePrimitive(f32, [4], "plane") as Plane;
  }
  const count = f32.length / 4;
  return makePrimitive(f32, [count, 4], "plane") as Plane;
}

// ============================================================================
// AABB
// ============================================================================

/** Create an AABB from min and max Points. */
export function aabb(min: Point, max: Point): AABB;
/** Create an AABB from two NDArrays (mins, maxs). */
export function aabb(mins: NDArrayFloat32, maxs: NDArrayFloat32): AABB;
/** Create an AABB from a flat array [min..., max...]. */
export function aabb(data: number[] | Float32Array): AABB;
/** Create an AABB or batch from flat data + spatial dimension. */
export function aabb(data: number[] | Float32Array, dims: number): AABB;
export function aabb(
  minOrData: Point | NDArrayFloat32 | number[] | Float32Array,
  maxOrDims?: Point | NDArrayFloat32 | number,
): AABB {
  if (minOrData instanceof NDArray && !(minOrData instanceof Primitive)) {
    return stackPrimitive([minOrData, maxOrDims as NDArrayFloat32], "aabb") as AABB;
  }
  if (minOrData instanceof Primitive) {
    const lo = minOrData.data;
    const hi = (maxOrDims as Point).data;
    const dims = lo.length;
    const f32 = new Float32Array(dims * 2);
    f32.set(lo, 0);
    f32.set(hi, dims);
    return makePrimitive(f32, [2, dims], "aabb") as AABB;
  }
  const f32 = toF32(minOrData);
  if (typeof maxOrDims === "number") {
    const dims = maxOrDims;
    const n = f32.length / (2 * dims);
    return makePrimitive(f32, [n, 2, dims], "aabb") as AABB;
  }
  const dims = f32.length / 2;
  return makePrimitive(f32, [2, dims], "aabb") as AABB;
}

/** Compute the axis-aligned bounding box of points, a mesh, or a point cloud. */
export function aabbFrom(input: NDArrayFloat32): AABB;
export function aabbFrom(input: { points: NDArrayFloat32 }): AABB;
export function aabbFrom(input: NDArrayFloat32 | { points: NDArrayFloat32 }): AABB {
  const pts = input instanceof NDArray ? input : input.points;
  return aabb(pts.min(0) as NDArrayFloat32, pts.max(0) as NDArrayFloat32);
}

// ============================================================================
// Polygon
// ============================================================================

/** Create a polygon from an array of Point Primitives. */
export function polygon(vertices: Point[]): Polygon;
/** Create a polygon or batch from an NDArray (no copy). */
export function polygon(arr: NDArrayFloat32): Polygon;
/** Create a polygon from a flat array + dimension. */
export function polygon(data: number[] | Float32Array, dims: number): Polygon;
export function polygon(
  verticesOrData: Point[] | NDArrayFloat32 | number[] | Float32Array,
  dims?: number,
): Polygon {
  if (verticesOrData instanceof NDArray && !(verticesOrData instanceof Primitive)) {
    return new Primitive(verticesOrData._handle.shared_view(), "polygon") as Polygon;
  }
  if (Array.isArray(verticesOrData) && verticesOrData.length > 0
      && verticesOrData[0] instanceof Primitive) {
    const verts = verticesOrData as Point[];
    const d = verts[0].data.length;
    const f32 = new Float32Array(verts.length * d);
    for (let i = 0; i < verts.length; i++)
      f32.set(verts[i].data, i * d);
    return makePrimitive(f32, [verts.length, d], "polygon") as Polygon;
  }
  const f32 = toF32(verticesOrData as number[] | Float32Array);
  const d = dims!;
  const numVerts = f32.length / d;
  return makePrimitive(f32, [numVerts, d], "polygon") as Polygon;
}
