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
import { NDArray, NDArrayInt8, NDArrayInt32, NDArrayFloat32, NDArrayFloat64, NDArrayBool } from "./NDArray";
import type { Dtype, DtypeToArray, FullDtype, RandomDtype, RangeDtype } from "./dtype";

function nd(dtype: string): string {
  return dtype === "bool" ? "int8" : dtype;
}

/** Create a WASM-resident NDArray from a typed array (or plain number[]) + shape. */
export function ndarray(data: number[], shape?: number[]): NDArrayFloat64;
export function ndarray(data: Float32Array, shape?: number[]): NDArrayFloat32;
export function ndarray(data: Float64Array, shape?: number[]): NDArrayFloat64;
export function ndarray(data: Int32Array, shape?: number[]): NDArrayInt32;
export function ndarray(data: Int8Array, shape?: number[]): NDArrayInt8;
export function ndarray(
  data: number[] | Float32Array | Float64Array | Int32Array | Int8Array,
  shape?: number[],
): NDArray {
  const m = native();
  const s = shape ?? [data.length];
  if (data instanceof Int32Array)
    return new NDArray(m.NativeInt32NDArray.from_js(data, s), "int32");
  if (data instanceof Int8Array)
    return new NDArray(m.NativeInt8NDArray.from_js(data, s), "int8");
  if (data instanceof Float32Array)
    return new NDArray(m.NativeFloat32NDArray.from_js(data, s), "float32");
  if (data instanceof Float64Array)
    return new NDArray(m.NativeFloat64NDArray.from_js(data, s), "float64");
  // raw number[] defaults to float64 (JS number is double-precision internally)
  const f64 = new Float64Array(data);
  return new NDArray(m.NativeFloat64NDArray.from_js(f64, s), "float64");
}

/** Generate a random WASM-resident NDArray. */
export function random<D extends RandomDtype>(
  dtype: D, shape: number[], min?: number, max?: number,
): DtypeToArray<D>;
export function random(
  dtype: RandomDtype, shape: number[], min?: number, max?: number,
): NDArray {
  const lo = min ?? 0;
  const hi = max ?? 1;
  const raw = native()[`random_${nd(dtype)}`](shape, lo, hi);
  return new NDArray(raw, dtype);
}

/** Stack arrays along a new axis (NumPy-style). All inputs must have identical shape. */
export function stack<T extends NDArray>(arrays: T[], axis?: number): T;
export function stack(arrays: NDArray[], axis: number = 0): NDArray {
  const dtype = arrays[0].dtype;
  const handles = arrays.map((a) => a._handle);
  const raw = native()[`stack_${nd(dtype)}`](handles, axis);
  return new NDArray(raw, dtype);
}

/** Concatenate arrays along an existing axis (NumPy-style). */
export function concatenate<T extends NDArray>(arrays: T[], axis?: number): T;
export function concatenate(arrays: NDArray[], axis: number = 0): NDArray {
  const dtype = arrays[0].dtype;
  const handles = arrays.map((a) => a._handle);
  const raw = native()[`concatenate_${nd(dtype)}`](handles, axis);
  return new NDArray(raw, dtype);
}

/** Repeat array along axes (NumPy-style np.tile). */
export function tile<T extends NDArray>(arr: T, reps: number | number[]): T;
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
export function zeros<D extends Dtype>(dtype: D, shape: number[]): DtypeToArray<D>;
export function zeros(dtype: Dtype, shape: number[]): NDArray {
  const raw = native()[`zeros_${nd(dtype)}`](shape);
  return new NDArray(raw, dtype);
}

/** Create a WASM-resident NDArray filled with ones. */
export function ones<D extends Dtype>(dtype: D, shape: number[]): DtypeToArray<D>;
export function ones(dtype: Dtype, shape: number[]): NDArray {
  const raw = native()[`ones_${nd(dtype)}`](shape);
  return new NDArray(raw, dtype);
}

/** Create a WASM-resident NDArray filled with a constant value. */
export function full<D extends FullDtype>(dtype: D, shape: number[], value: number): DtypeToArray<D>;
export function full(dtype: FullDtype, shape: number[], value: number): NDArray {
  const raw = native()[`full_${nd(dtype)}`](shape, value);
  return new NDArray(raw, dtype);
}

/** Create an NxN identity matrix (ones on diagonal, zeros elsewhere). */
export function eye<D extends RangeDtype>(dtype: D, n: number): DtypeToArray<D>;
export function eye(dtype: RangeDtype, n: number): NDArray {
  const raw = native()[`zeros_${nd(dtype)}`]([n, n]);
  const arr = new NDArray(raw, dtype);
  const d = arr.data;
  for (let i = 0; i < n; i++) d[i * n + i] = 1;
  return arr;
}

/** Create a WASM-resident NDArray with evenly spaced values [start, stop). */
export function arange<D extends RangeDtype>(dtype: D, stop: number): DtypeToArray<D>;
export function arange<D extends RangeDtype>(dtype: D, start: number, stop: number, step?: number): DtypeToArray<D>;
export function arange(dtype: RangeDtype, startOrStop: number, stop?: number, step: number = 1): NDArray {
  const actualStart = stop === undefined ? 0 : startOrStop;
  const actualStop = stop === undefined ? startOrStop : stop;
  const raw = native()[`arange_${nd(dtype)}`](actualStart, actualStop, step);
  return new NDArray(raw, dtype);
}

/** Create a WASM-resident float64 NDArray with n evenly spaced values [start, stop]. */
export function linspace(start: number, stop: number, n: number): NDArrayFloat64 {
  const raw = native().linspace_float64(start, stop, n);
  return new NDArray(raw, "float64");
}

/** Select slices along axis by indices (np.take). */
export function take<T extends NDArray>(arr: T, indices: NDArrayInt32, axis?: number): T;
export function take(arr: NDArray, indices: NDArray, axis: number = 0): NDArray {
  return arr.take(indices as NDArray<Int32Array>, axis);
}

/** Index along axis using per-element indices (np.take_along_axis). */
export function takeAlongAxis<T extends NDArray>(arr: T, indices: NDArrayInt32, axis: number): T;
export function takeAlongAxis(arr: NDArray, indices: NDArray, axis: number): NDArray {
  return arr.takeAlongAxis(indices as NDArray<Int32Array>, axis);
}

/** Return sorted copy (lexicographic row sort for nD). */
export function sort<T extends NDArrayFloat32 | NDArrayFloat64 | NDArrayInt32 | NDArrayInt8>(arr: T): T;
export function sort(arr: NDArray): NDArray {
  return new NDArray(native()[`sort_${nd(arr.dtype)}`](arr._handle), arr.dtype);
}

/** Sort in-place (lexicographic row sort for nD). */
export function sort_<T extends NDArrayFloat32 | NDArrayFloat64 | NDArrayInt32 | NDArrayInt8>(arr: T): T;
export function sort_(arr: NDArray): NDArray {
  native()[`sort_inplace_${nd(arr.dtype)}`](arr._handle);
  return arr;
}

/** Return int32 row permutation (lexicographic for nD). */
export function argsort(arr: NDArray): NDArrayInt32 {
  return new NDArray<Int32Array>(native()[`argsort_${nd(arr.dtype)}`](arr._handle), "int32");
}

/** Remove duplicate rows from sorted array. */
export function unique<T extends NDArrayFloat32 | NDArrayFloat64 | NDArrayInt32 | NDArrayInt8>(arr: T): T;
export function unique(arr: NDArray): NDArray {
  return new NDArray(native()[`unique_${nd(arr.dtype)}`](arr._handle), arr.dtype);
}

/** Sorted union of two sorted arrays. */
export function setUnion<T extends NDArrayFloat32 | NDArrayFloat64 | NDArrayInt32 | NDArrayInt8>(a: T, b: T): T;
export function setUnion(a: NDArray, b: NDArray): NDArray {
  return new NDArray(native()[`set_union_${nd(a.dtype)}`](a._handle, b._handle), a.dtype);
}

/** Sorted intersection of two sorted arrays. */
export function setIntersection<T extends NDArrayFloat32 | NDArrayFloat64 | NDArrayInt32 | NDArrayInt8>(a: T, b: T): T;
export function setIntersection(a: NDArray, b: NDArray): NDArray {
  return new NDArray(native()[`set_intersection_${nd(a.dtype)}`](a._handle, b._handle), a.dtype);
}

/** Sorted difference: elements in a not in b (both sorted). */
export function setDifference<T extends NDArrayFloat32 | NDArrayFloat64 | NDArrayInt32 | NDArrayInt8>(a: T, b: T): T;
export function setDifference(a: NDArray, b: NDArray): NDArray {
  return new NDArray(native()[`set_difference_${nd(a.dtype)}`](a._handle, b._handle), a.dtype);
}
