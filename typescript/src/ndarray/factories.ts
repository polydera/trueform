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
import { NDArray, NDArrayInt8, NDArrayInt32, NDArrayFloat32, NDArrayBool } from "./NDArray";

function nd(dtype: string): string {
  return dtype === "bool" ? "int8" : dtype;
}

/** Create a WASM-resident NDArray from a typed array (or plain number[]) + shape. */
export function ndarray(data: number[], shape: number[]): NDArrayFloat32;
export function ndarray(data: Float32Array, shape: number[]): NDArrayFloat32;
export function ndarray(data: Int32Array, shape: number[]): NDArrayInt32;
export function ndarray(data: Int8Array, shape: number[]): NDArrayInt8;
export function ndarray(
  data: number[] | Float32Array | Int32Array | Int8Array, shape: number[],
): NDArray {
  const m = native();
  if (data instanceof Int32Array)
    return new NDArray(m.NativeInt32NDArray.from_js(data, shape), "int32");
  if (data instanceof Int8Array)
    return new NDArray(m.NativeInt8NDArray.from_js(data, shape), "int8");
  const f32 = data instanceof Float32Array ? data : new Float32Array(data);
  return new NDArray(m.NativeFloat32NDArray.from_js(f32, shape), "float32");
}

/** Generate a random WASM-resident NDArray. */
export function random(
  dtype: "float32", shape: number[], min?: number, max?: number,
): NDArrayFloat32;
export function random(
  dtype: "int32", shape: number[], min?: number, max?: number,
): NDArrayInt32;
export function random(
  dtype: string, shape: number[], min?: number, max?: number,
): NDArray {
  const lo = min ?? 0;
  const hi = max ?? 1;
  if (dtype === "int32")
    return new NDArray(native().random_int32(shape, lo, hi), "int32");
  return new NDArray(native().random_float32(shape, lo, hi), "float32");
}

/** Stack arrays along a new axis (NumPy-style). All inputs must have identical shape. */
export function stack(arrays: NDArrayFloat32[], axis?: number): NDArrayFloat32;
export function stack(arrays: NDArrayInt32[], axis?: number): NDArrayInt32;
export function stack(arrays: NDArrayInt8[], axis?: number): NDArrayInt8;
export function stack(arrays: NDArrayBool[], axis?: number): NDArrayBool;
export function stack(arrays: NDArray[], axis: number = 0): NDArray {
  const dtype = arrays[0].dtype;
  const handles = arrays.map((a) => a._handle);
  const raw = native()[`stack_${nd(dtype)}`](handles, axis);
  return new NDArray(raw, dtype);
}

/** Concatenate arrays along an existing axis (NumPy-style). */
export function concatenate(arrays: NDArrayFloat32[], axis?: number): NDArrayFloat32;
export function concatenate(arrays: NDArrayInt32[], axis?: number): NDArrayInt32;
export function concatenate(arrays: NDArrayInt8[], axis?: number): NDArrayInt8;
export function concatenate(arrays: NDArrayBool[], axis?: number): NDArrayBool;
export function concatenate(arrays: NDArray[], axis: number = 0): NDArray {
  const dtype = arrays[0].dtype;
  const handles = arrays.map((a) => a._handle);
  const raw = native()[`concatenate_${nd(dtype)}`](handles, axis);
  return new NDArray(raw, dtype);
}

/** Repeat array along axes (NumPy-style np.tile). */
export function tile(arr: NDArrayFloat32, reps: number | number[]): NDArrayFloat32;
export function tile(arr: NDArrayInt32, reps: number | number[]): NDArrayInt32;
export function tile(arr: NDArrayInt8, reps: number | number[]): NDArrayInt8;
export function tile(arr: NDArrayBool, reps: number | number[]): NDArrayBool;
export function tile(arr: NDArray, reps: number | number[]): NDArray {
  const r = typeof reps === "number" ? [reps] : reps;
  const raw = native()[`tile_${nd(arr.dtype)}`](arr._handle, r);
  return new NDArray(raw, arr.dtype);
}

/** Element-wise conditional: cond ? x : y. */
export function where(cond: NDArrayBool, x: NDArray, y: NDArray): NDArray {
  const raw = native()[`where_${nd(x.dtype)}`](cond._handle, x._handle, y._handle);
  return new NDArray(raw, x.dtype);
}

/** Create a WASM-resident NDArray filled with zeros. */
export function zeros(dtype: "float32", shape: number[]): NDArrayFloat32;
export function zeros(dtype: "int32", shape: number[]): NDArrayInt32;
export function zeros(dtype: "int8", shape: number[]): NDArrayInt8;
export function zeros(dtype: "bool", shape: number[]): NDArrayBool;
export function zeros(dtype: string, shape: number[]): NDArray {
  const raw = native()[`zeros_${nd(dtype)}`](shape);
  return new NDArray(raw, dtype);
}

/** Create a WASM-resident NDArray filled with ones. */
export function ones(dtype: "float32", shape: number[]): NDArrayFloat32;
export function ones(dtype: "int32", shape: number[]): NDArrayInt32;
export function ones(dtype: "int8", shape: number[]): NDArrayInt8;
export function ones(dtype: "bool", shape: number[]): NDArrayBool;
export function ones(dtype: string, shape: number[]): NDArray {
  const raw = native()[`ones_${nd(dtype)}`](shape);
  return new NDArray(raw, dtype);
}

/** Create a WASM-resident NDArray filled with a constant value. */
export function full(dtype: "float32", shape: number[], value: number): NDArrayFloat32;
export function full(dtype: "int32", shape: number[], value: number): NDArrayInt32;
export function full(dtype: "int8", shape: number[], value: number): NDArrayInt8;
export function full(dtype: string, shape: number[], value: number): NDArray {
  const raw = native()[`full_${nd(dtype)}`](shape, value);
  return new NDArray(raw, dtype);
}

/** Create an NxN identity matrix (ones on diagonal, zeros elsewhere). */
export function eye(dtype: "float32", n: number): NDArrayFloat32;
export function eye(dtype: "int32", n: number): NDArrayInt32;
export function eye(dtype: string, n: number): NDArray {
  const raw = native()[`zeros_${nd(dtype)}`]([n, n]);
  const arr = new NDArray(raw, dtype);
  const d = arr.data;
  for (let i = 0; i < n; i++) d[i * n + i] = 1;
  return arr;
}

/** Create a WASM-resident NDArray with evenly spaced values [start, stop). */
export function arange(dtype: "float32", start: number, stop: number, step?: number): NDArrayFloat32;
export function arange(dtype: "int32", start: number, stop: number, step?: number): NDArrayInt32;
export function arange(dtype: string, start: number, stop: number, step: number = 1): NDArray {
  const raw = native()[`arange_${nd(dtype)}`](start, stop, step);
  return new NDArray(raw, dtype);
}

/** Create a WASM-resident float32 NDArray with n evenly spaced values [start, stop]. */
export function linspace(start: number, stop: number, n: number): NDArrayFloat32 {
  const raw = native().linspace_float32(start, stop, n);
  return new NDArray(raw, "float32");
}

/** Select slices along axis by indices (np.take). */
export function take(arr: NDArrayFloat32, indices: NDArrayInt32, axis?: number): NDArrayFloat32;
export function take(arr: NDArrayInt32, indices: NDArrayInt32, axis?: number): NDArrayInt32;
export function take(arr: NDArrayInt8, indices: NDArrayInt32, axis?: number): NDArrayInt8;
export function take(arr: NDArrayBool, indices: NDArrayInt32, axis?: number): NDArrayBool;
export function take(arr: NDArray, indices: NDArray, axis: number = 0): NDArray {
  return arr.take(indices as NDArray<Int32Array>, axis);
}

/** Index along axis using per-element indices (np.take_along_axis). */
export function takeAlongAxis(arr: NDArrayFloat32, indices: NDArrayInt32, axis: number): NDArrayFloat32;
export function takeAlongAxis(arr: NDArrayInt32, indices: NDArrayInt32, axis: number): NDArrayInt32;
export function takeAlongAxis(arr: NDArrayInt8, indices: NDArrayInt32, axis: number): NDArrayInt8;
export function takeAlongAxis(arr: NDArrayBool, indices: NDArrayInt32, axis: number): NDArrayBool;
export function takeAlongAxis(arr: NDArray, indices: NDArray, axis: number): NDArray {
  return arr.takeAlongAxis(indices as NDArray<Int32Array>, axis);
}

/** Return sorted copy (lexicographic row sort for nD). */
export function sort(arr: NDArrayFloat32): NDArrayFloat32;
export function sort(arr: NDArrayInt32): NDArrayInt32;
export function sort(arr: NDArrayInt8): NDArrayInt8;
export function sort(arr: NDArray): NDArray {
  return new NDArray(native()[`sort_${nd(arr.dtype)}`](arr._handle), arr.dtype);
}

/** Sort in-place (lexicographic row sort for nD). */
export function sort_(arr: NDArrayFloat32): NDArrayFloat32;
export function sort_(arr: NDArrayInt32): NDArrayInt32;
export function sort_(arr: NDArrayInt8): NDArrayInt8;
export function sort_(arr: NDArray): NDArray {
  native()[`sort_inplace_${nd(arr.dtype)}`](arr._handle);
  return arr;
}

/** Return int32 row permutation (lexicographic for nD). */
export function argsort(arr: NDArray): NDArrayInt32 {
  return new NDArray<Int32Array>(native()[`argsort_${nd(arr.dtype)}`](arr._handle), "int32");
}

/** Remove duplicate rows from sorted array. */
export function unique(arr: NDArrayFloat32): NDArrayFloat32;
export function unique(arr: NDArrayInt32): NDArrayInt32;
export function unique(arr: NDArrayInt8): NDArrayInt8;
export function unique(arr: NDArray): NDArray {
  return new NDArray(native()[`unique_${nd(arr.dtype)}`](arr._handle), arr.dtype);
}

/** Sorted union of two sorted arrays. */
export function setUnion(a: NDArrayFloat32, b: NDArrayFloat32): NDArrayFloat32;
export function setUnion(a: NDArrayInt32, b: NDArrayInt32): NDArrayInt32;
export function setUnion(a: NDArrayInt8, b: NDArrayInt8): NDArrayInt8;
export function setUnion(a: NDArray, b: NDArray): NDArray {
  return new NDArray(native()[`set_union_${nd(a.dtype)}`](a._handle, b._handle), a.dtype);
}

/** Sorted intersection of two sorted arrays. */
export function setIntersection(a: NDArrayFloat32, b: NDArrayFloat32): NDArrayFloat32;
export function setIntersection(a: NDArrayInt32, b: NDArrayInt32): NDArrayInt32;
export function setIntersection(a: NDArrayInt8, b: NDArrayInt8): NDArrayInt8;
export function setIntersection(a: NDArray, b: NDArray): NDArray {
  return new NDArray(native()[`set_intersection_${nd(a.dtype)}`](a._handle, b._handle), a.dtype);
}

/** Sorted difference: elements in a not in b (both sorted). */
export function setDifference(a: NDArrayFloat32, b: NDArrayFloat32): NDArrayFloat32;
export function setDifference(a: NDArrayInt32, b: NDArrayInt32): NDArrayInt32;
export function setDifference(a: NDArrayInt8, b: NDArrayInt8): NDArrayInt8;
export function setDifference(a: NDArray, b: NDArray): NDArray {
  return new NDArray(native()[`set_difference_${nd(a.dtype)}`](a._handle, b._handle), a.dtype);
}
