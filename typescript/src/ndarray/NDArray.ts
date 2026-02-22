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

import { registry } from "../internal/registry";
import { native } from "../native";

export interface NativeNDArray<T = any> {
  data(): T;
  size(): number;
  ndim(): number;
  shape(): number[];
  shape_at(i: number): number;
  set_shape(shape: number[]): void;
  row(i: number): NativeNDArray<T>;
  slice(start: number, end: number): NativeNDArray<T>;
  destroy(): void;
  is_valid(): boolean;
  shared_view(): NativeNDArray<T>;
  delete(): void;
}

/** Maps TS dtype to the native (embind) suffix. */
function nativeDtype(dtype: string): string {
  return dtype === "bool" ? "int8" : dtype;
}

/**
 * A WASM-resident typed array with shape metadata.
 *
 * Data lives in the WASM heap. Views (row, slice) share the same
 * underlying buffer via reference counting.
 *
 * Iteration behavior depends on dimensionality:
 * - 1D: yields individual values (numbers)
 * - 2D: yields subarray views (rows)
 * - 3D+: yields NDArray views (sub-arrays via C++ row())
 */
export class NDArray<T = any> {
  /** @internal */
  readonly _handle: NativeNDArray<T>;
  /** Runtime type discriminator: "int8", "int32", "float32", or "bool". */
  readonly dtype: string;

  /** @internal */
  constructor(handle: NativeNDArray<T>, dtype: string) {
    this._handle = handle;
    this.dtype = dtype;
    registry.register(this, { handle });
  }

  get data(): T {
    return this._handle.data();
  }

  get length(): number {
    return this._handle.size();
  }

  get shape(): number[] {
    return this._handle.shape();
  }

  set shape(newShape: number[]) {
    this._handle.set_shape(newShape);
  }

  get ndim(): number {
    return this._handle.ndim();
  }

  row(i: number): NDArray<T> {
    return new NDArray<T>(this._handle.row(i), this.dtype);
  }

  slice(start: number, end?: number): NDArray<T> {
    return new NDArray<T>(
      this._handle.slice(start, end ?? this.shape[0]),
      this.dtype,
    );
  }

  *[Symbol.iterator](): Iterator<number | NDArray<T>> {
    const ndim = this.ndim;
    if (ndim === 1) {
      const d: any = this.data;
      for (let i = 0; i < d.length; i++) {
        yield d[i];
      }
    } else if (ndim === 2) {
      const cols = this.shape[1];
      const d: any = this.data;
      for (let i = 0; i < d.length; i += cols) {
        yield d.subarray(i, i + cols);
      }
    } else {
      for (let i = 0; i < this.shape[0]; i++) {
        yield this.row(i);
      }
    }
  }

  // ============ Type casting ============

  as(dtype: "float32"): NDArrayFloat32;
  as(dtype: "int32"): NDArrayInt32;
  as(dtype: "int8"): NDArrayInt8;
  as(dtype: "bool"): NDArrayBool;
  as(dtype: string): NDArray {
    const srcNative = nativeDtype(this.dtype);
    const dstNative = nativeDtype(dtype);
    if (srcNative === dstNative) {
      // Same underlying storage — just relabel (e.g. int8 ↔ bool)
      return new NDArray(this._handle, dtype);
    }
    const raw = native()[`cast_${srcNative}_to_${dstNative}`](this._handle);
    return new NDArray(raw, dtype);
  }

  // ============ Reductions ============

  sum(axis?: number): number | NDArray {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const outDtype = this.dtype === "float32" ? "float32" : "int32";
    const raw = native()[`sum_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, outDtype);
  }

  min(axis?: number): number | NDArray<T> {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`min_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, this.dtype);
  }

  max(axis?: number): number | NDArray<T> {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`max_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, this.dtype);
  }

  mean(axis?: number): number | NDArray {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`mean_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "float32");
  }

  // ============ Element-wise (copy) ============

  add(other: NDArray | number): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray<T>(
        native()[`add_scalar_${nd}`](this._handle, other),
        this.dtype,
      );
    }
    return new NDArray<T>(
      native()[`add_${nd}`](this._handle, other._handle),
      this.dtype,
    );
  }

  sub(other: NDArray | number): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray<T>(
        native()[`sub_scalar_${nd}`](this._handle, other),
        this.dtype,
      );
    }
    return new NDArray<T>(
      native()[`sub_${nd}`](this._handle, other._handle),
      this.dtype,
    );
  }

  mul(other: NDArray | number): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray<T>(
        native()[`mul_scalar_${nd}`](this._handle, other),
        this.dtype,
      );
    }
    return new NDArray<T>(
      native()[`mul_${nd}`](this._handle, other._handle),
      this.dtype,
    );
  }

  div(other: NDArray | number): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray<T>(
        native()[`div_scalar_${nd}`](this._handle, other),
        this.dtype,
      );
    }
    return new NDArray<T>(
      native()[`div_${nd}`](this._handle, other._handle),
      this.dtype,
    );
  }

  matMul(other: NDArray): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`mat_mul_${nd}`](this._handle, other._handle),
      this.dtype,
    );
  }

  clip(lo: number, hi: number): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(native()[`clip_${nd}`](this._handle, lo, hi), this.dtype);
  }

  // ============ Element-wise (in-place) ============

  div_(other: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      native()[`div_scalar_inplace_${nd}`](this._handle, other);
    } else {
      native()[`div_inplace_${nd}`](this._handle, other._handle);
    }
    return this;
  }

  clip_(lo: number, hi: number): this {
    const nd = nativeDtype(this.dtype);
    native()[`clip_inplace_${nd}`](this._handle, lo, hi);
    return this;
  }

  add_(other: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      native()[`add_scalar_inplace_${nd}`](this._handle, other);
    } else {
      native()[`add_inplace_${nd}`](this._handle, other._handle);
    }
    return this;
  }

  sub_(other: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      native()[`sub_scalar_inplace_${nd}`](this._handle, other);
    } else {
      native()[`sub_inplace_${nd}`](this._handle, other._handle);
    }
    return this;
  }

  mul_(other: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      native()[`mul_scalar_inplace_${nd}`](this._handle, other);
    } else {
      native()[`mul_inplace_${nd}`](this._handle, other._handle);
    }
    return this;
  }

  // ============ Relational (return NDArrayBool) ============

  eq(other: NDArray | number): NDArrayBool {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray(
        native()[`eq_scalar_${nd}`](this._handle, other), "bool",
      );
    }
    return new NDArray(
      native()[`eq_${nd}`](this._handle, other._handle), "bool",
    );
  }

  neq(other: NDArray | number): NDArrayBool {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray(
        native()[`neq_scalar_${nd}`](this._handle, other), "bool",
      );
    }
    return new NDArray(
      native()[`neq_${nd}`](this._handle, other._handle), "bool",
    );
  }

  lt(other: NDArray | number): NDArrayBool {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray(
        native()[`lt_scalar_${nd}`](this._handle, other), "bool",
      );
    }
    return new NDArray(
      native()[`lt_${nd}`](this._handle, other._handle), "bool",
    );
  }

  gt(other: NDArray | number): NDArrayBool {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray(
        native()[`gt_scalar_${nd}`](this._handle, other), "bool",
      );
    }
    return new NDArray(
      native()[`gt_${nd}`](this._handle, other._handle), "bool",
    );
  }

  lte(other: NDArray | number): NDArrayBool {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray(
        native()[`lte_scalar_${nd}`](this._handle, other), "bool",
      );
    }
    return new NDArray(
      native()[`lte_${nd}`](this._handle, other._handle), "bool",
    );
  }

  gte(other: NDArray | number): NDArrayBool {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray(
        native()[`gte_scalar_${nd}`](this._handle, other), "bool",
      );
    }
    return new NDArray(
      native()[`gte_${nd}`](this._handle, other._handle), "bool",
    );
  }

  // ============ Logical (bool only) ============

  not(): NDArrayBool {
    return new NDArray(native().not_int8(this._handle), "bool");
  }

  and(other: NDArray): NDArrayBool {
    return new NDArray(
      native().and_int8(this._handle, other._handle), "bool",
    );
  }

  or(other: NDArray): NDArrayBool {
    return new NDArray(
      native().or_int8(this._handle, other._handle), "bool",
    );
  }

  // ============ Reductions (argmin/argmax/any/all) ============

  argmin(axis?: number): number | NDArrayInt32 {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`argmin_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "int32");
  }

  argmax(axis?: number): number | NDArrayInt32 {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`argmax_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "int32");
  }

  any(axis?: number): number | NDArrayBool {
    const a = axis ?? -1;
    const raw = native().any_int8(this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "bool");
  }

  all(axis?: number): number | NDArrayBool {
    const a = axis ?? -1;
    const raw = native().all_int8(this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "bool");
  }

  // ============ Indexing ============

  take(indices: NDArray<Int32Array>, axis: number = 0): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`take_${nd}`](this._handle, indices._handle, axis), this.dtype,
    );
  }

  takeAlongAxis(indices: NDArray<Int32Array>, axis: number): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`take_along_axis_${nd}`](this._handle, indices._handle, axis), this.dtype,
    );
  }

  sort(): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(native()[`sort_${nd}`](this._handle), this.dtype);
  }

  sort_(): this {
    const nd = nativeDtype(this.dtype);
    native()[`sort_inplace_${nd}`](this._handle);
    return this;
  }

  argsort(): NDArrayInt32 {
    const nd = nativeDtype(this.dtype);
    return new NDArray<Int32Array>(native()[`argsort_${nd}`](this._handle), "int32");
  }

  booleanIndex(mask: NDArrayBool): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`boolean_index_${nd}`](this._handle, mask._handle), this.dtype,
    );
  }

  // ============ Assign ============

  assign(value: number): this;
  assign(value: NDArray): this;
  assign(selector: NDArray, value: number): this;
  assign(selector: NDArray, value: NDArray): this;
  assign(selectorOrValue: NDArray | number, value?: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (value === undefined) {
      // 1-arg: assign(scalar) or assign(array)
      if (typeof selectorOrValue === "number") {
        native()[`assign_scalar_${nd}`](this._handle, selectorOrValue);
      } else {
        native()[`assign_array_${nd}`](this._handle, selectorOrValue._handle);
      }
    } else {
      // 2-arg: assign(selector, value)
      const sel = selectorOrValue as NDArray;
      if (sel.dtype === "int32") {
        if (typeof value === "number") {
          native()[`assign_indexed_scalar_${nd}`](this._handle, sel._handle, value);
        } else {
          native()[`assign_indexed_array_${nd}`](this._handle, sel._handle, (value as NDArray)._handle);
        }
      } else {
        if (typeof value === "number") {
          native()[`assign_masked_scalar_${nd}`](this._handle, sel._handle, value);
        } else {
          native()[`assign_masked_array_${nd}`](this._handle, sel._handle, (value as NDArray)._handle);
        }
      }
    }
    return this;
  }

  // ============ Transpose ============

  get T(): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`transpose_${nd}`](this._handle, undefined), this.dtype,
    );
  }

  transpose(axes?: number[]): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`transpose_${nd}`](this._handle, axes), this.dtype,
    );
  }

  // ============ Shape ops (zero-copy) ============

  flatten(): NDArray<T> {
    const view = this._handle.shared_view();
    view.set_shape([this.length]);
    return new NDArray<T>(view, this.dtype);
  }

  squeeze(axis?: number): NDArray<T> {
    const s = this.shape;
    const newShape: number[] = [];
    for (let i = 0; i < s.length; i++) {
      if (s[i] === 1 && (axis === undefined || axis === i)) continue;
      newShape.push(s[i]);
    }
    if (newShape.length === 0) newShape.push(1);
    const view = this._handle.shared_view();
    view.set_shape(newShape);
    return new NDArray<T>(view, this.dtype);
  }

  unsqueeze(axis: number): NDArray<T> {
    const s = this.shape.slice();
    s.splice(axis, 0, 1);
    const view = this._handle.shared_view();
    view.set_shape(s);
    return new NDArray<T>(view, this.dtype);
  }

  reshape(shape: number[]): NDArray<T> {
    const view = this._handle.shared_view();
    view.set_shape(shape);
    return new NDArray<T>(view, this.dtype);
  }

  // ============ Clone ============

  clone(): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(native()[`clone_${nd}`](this._handle), this.dtype);
  }

  // ============ Lifecycle ============

  delete(): void {
    this._handle.destroy();
  }

  [Symbol.dispose](): void {
    this._handle.destroy();
  }
}

/** WASM-resident int8 NDArray. */
export type NDArrayInt8 = NDArray<Int8Array>;
/** WASM-resident int32 NDArray. */
export type NDArrayInt32 = NDArray<Int32Array>;
/** WASM-resident float32 NDArray. */
export type NDArrayFloat32 = NDArray<Float32Array>;
/** WASM-resident boolean NDArray (int8 storage, 0/1 values). */
export type NDArrayBool = NDArray<Int8Array>;
