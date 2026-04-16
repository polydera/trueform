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
  shallow_copy(): NativeNDArray<T>;
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

  /** Typed array view into the WASM heap (Float32Array, Int32Array, or Int8Array). */
  get data(): T {
    return this._handle.data();
  }

  /** Total number of elements. */
  get length(): number {
    return this._handle.size();
  }

  /** Shape tuple, e.g. `[N, 3]`. Settable to reshape in-place. */
  get shape(): number[] {
    return this._handle.shape();
  }

  /** Reshape in-place (must preserve total element count). */
  set shape(newShape: number[]) {
    this._handle.set_shape(newShape);
  }

  /** Number of dimensions. */
  get ndim(): number {
    return this._handle.ndim();
  }

  /** Extract row `i` as a shared view (zero-copy). */
  row(i: number): NDArray<T> {
    return new NDArray<T>(this._handle.row(i), this.dtype);
  }

  /** Get element at index `i`. Returns a number for 1D, or a row view for nD. */
  get(i: number): number | NDArray<T> {
    if (this.ndim === 1) {
      return (this.data as any)[i];
    }
    return this.row(i);
  }

  /** Slice along axis 0: `[start, end)`. Returns a shared view (zero-copy). */
  slice(start: number, end?: number): NDArray<T> {
    return new NDArray<T>(
      this._handle.slice(start, end ?? this.shape[0]),
      this.dtype,
    );
  }

  /** Iterate over elements (1D → numbers, nD → row views). */
  *[Symbol.iterator](): Iterator<number | NDArray<T>> {
    const ndim = this.ndim;
    if (ndim === 1) {
      const d: any = this.data;
      for (let i = 0; i < d.length; i++) {
        yield d[i];
      }
    } else {
      for (let i = 0; i < this.shape[0]; i++) {
        yield this.row(i);
      }
    }
  }

  // ============ Type casting ============

  /** Cast to a different dtype. Same-storage casts (int8 ↔ bool) are zero-copy. */
  as(dtype: "float32"): NDArrayFloat32;
  as(dtype: "int32"): NDArrayInt32;
  as(dtype: "int8"): NDArrayInt8;
  as(dtype: "bool"): NDArrayBool;
  as(dtype: string): NDArray {
    const srcNative = nativeDtype(this.dtype);
    const dstNative = nativeDtype(dtype);
    if (srcNative === dstNative) {
      // Same underlying storage — just relabel (e.g. int8 ↔ bool)
      // shallow_copy() bumps refcount so both sides own the buffer
      return new NDArray(this._handle.shallow_copy(), dtype);
    }
    const raw = native()[`cast_${srcNative}_to_${dstNative}`](this._handle);
    return new NDArray(raw, dtype);
  }

  // ============ Reductions ============

  /** Sum of elements. Without axis: scalar. With axis: reduced NDArray. */
  sum(axis?: number): number | NDArray {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const outDtype = this.dtype === "float32" ? "float32" : "int32";
    const raw = native()[`sum_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, outDtype);
  }

  /** Minimum value. Without axis: scalar. With axis: reduced NDArray. */
  min(axis?: number): number | NDArray<T> {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`min_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, this.dtype);
  }

  /** Maximum value. Without axis: scalar. With axis: reduced NDArray. */
  max(axis?: number): number | NDArray<T> {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`max_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, this.dtype);
  }

  /** Arithmetic mean. Without axis: scalar. With axis: reduced NDArray (float32). */
  mean(axis?: number): number | NDArray {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`mean_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "float32");
  }

  /** L2 norm. Without axis: scalar. With axis: per-slice norms (float32). */
  norm(axis?: number): number | NDArray {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`norm_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "float32");
  }

  // ============ Element-wise (copy) ============

  /** Element-wise addition. Broadcasts. */
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

  /** Element-wise subtraction. Broadcasts. */
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

  /** Element-wise multiplication. Broadcasts. */
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

  /** Element-wise division. Broadcasts. */
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

  /** Element-wise remainder (truncated division). Broadcasts. */
  mod(other: NDArray | number): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      return new NDArray<T>(
        native()[`mod_scalar_${nd}`](this._handle, other),
        this.dtype,
      );
    }
    return new NDArray<T>(
      native()[`mod_${nd}`](this._handle, other._handle),
      this.dtype,
    );
  }

  /** Matrix multiplication. Supports batch broadcasting on leading dims. */
  matMul(other: NDArray): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`mat_mul_${nd}`](this._handle, other._handle),
      this.dtype,
    );
  }

  /** Clamp values to `[lo, hi]`. */
  clip(lo: number, hi: number): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(native()[`clip_${nd}`](this._handle, lo, hi), this.dtype);
  }

  // ============ Element-wise (in-place) ============

  /** In-place division. Broadcasts. */
  div_(other: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      native()[`div_scalar_inplace_${nd}`](this._handle, other);
    } else {
      native()[`div_inplace_${nd}`](this._handle, other._handle);
    }
    return this;
  }

  /** In-place clamp to `[lo, hi]`. */
  clip_(lo: number, hi: number): this {
    const nd = nativeDtype(this.dtype);
    native()[`clip_inplace_${nd}`](this._handle, lo, hi);
    return this;
  }

  /** In-place addition. Broadcasts. */
  add_(other: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      native()[`add_scalar_inplace_${nd}`](this._handle, other);
    } else {
      native()[`add_inplace_${nd}`](this._handle, other._handle);
    }
    return this;
  }

  /** In-place subtraction. Broadcasts. */
  sub_(other: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      native()[`sub_scalar_inplace_${nd}`](this._handle, other);
    } else {
      native()[`sub_inplace_${nd}`](this._handle, other._handle);
    }
    return this;
  }

  /** In-place multiplication. Broadcasts. */
  mul_(other: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      native()[`mul_scalar_inplace_${nd}`](this._handle, other);
    } else {
      native()[`mul_inplace_${nd}`](this._handle, other._handle);
    }
    return this;
  }

  /** In-place remainder. Broadcasts. */
  mod_(other: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (typeof other === "number") {
      native()[`mod_scalar_inplace_${nd}`](this._handle, other);
    } else {
      native()[`mod_inplace_${nd}`](this._handle, other._handle);
    }
    return this;
  }

  // ============ Relational (return NDArrayBool) ============

  /** Element-wise equal. Broadcasts. */
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

  /** Element-wise not-equal. Broadcasts. */
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

  /** Element-wise less-than. Broadcasts. */
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

  /** Element-wise greater-than. Broadcasts. */
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

  /** Element-wise less-than-or-equal. Broadcasts. */
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

  /** Element-wise greater-than-or-equal. Broadcasts. */
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

  /** Element-wise NaN detection. Returns true where value is NaN. */
  isNaN(): NDArrayBool {
    return this.neq(this);
  }

  // ============ Logical (bool only) ============

  /** Logical NOT (bool arrays only). */
  not(): NDArrayBool {
    return new NDArray(native().not_int8(this._handle), "bool");
  }

  /** Logical AND (bool arrays only). */
  and(other: NDArray): NDArrayBool {
    return new NDArray(
      native().and_int8(this._handle, other._handle), "bool",
    );
  }

  /** Logical OR (bool arrays only). */
  or(other: NDArray): NDArrayBool {
    return new NDArray(
      native().or_int8(this._handle, other._handle), "bool",
    );
  }

  // ============ Reductions (argmin/argmax/any/all) ============

  /** Index of minimum value. Without axis: flat index. With axis: per-slice indices. */
  argmin(axis?: number): number | NDArrayInt32 {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`argmin_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "int32");
  }

  /** Index of maximum value. Without axis: flat index. With axis: per-slice indices. */
  argmax(axis?: number): number | NDArrayInt32 {
    const a = axis ?? -1;
    const nd = nativeDtype(this.dtype);
    const raw = native()[`argmax_${nd}`](this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "int32");
  }

  /** True if any element is nonzero (bool arrays). Without axis: scalar. With axis: per-slice. */
  any(axis?: number): number | NDArrayBool {
    const a = axis ?? -1;
    const raw = native().any_int8(this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "bool");
  }

  /** True if all elements are nonzero (bool arrays). Without axis: scalar. With axis: per-slice. */
  all(axis?: number): number | NDArrayBool {
    const a = axis ?? -1;
    const raw = native().all_int8(this._handle, a);
    return typeof raw === "number" ? raw : new NDArray(raw, "bool");
  }

  // ============ Indexing ============

  /**
   * Select elements by index.
   *
   * Single-axis: `take(indices, axis?)` — gather along one axis (default 0).
   *
   * Multi-axis (Cartesian): `take(spec0, spec1, ...)` where each spec is:
   * - `null` — keep whole axis
   * - `number` — single index (squeezes that dimension)
   * - `number[]` or `NDArrayInt32` — fancy index
   *
   * Unspecified trailing axes default to `null`.
   *
   * @example
   * ```ts
   * pts.take(null, 0)       // column 0: [N]
   * pts.take(null, [0, 2])  // columns 0,2: [N, 2]
   * pts.take([0, 2])        // rows 0,2: [2, 3]
   * pts.take(5)             // row 5 squeezed: [3]
   * ```
   */
  take(indices: NDArray<Int32Array>, axis?: number): NDArray<T>;
  take(...specs: (null | number | number[] | NDArray<Int32Array>)[]): NDArray<T>;
  take(...args: any[]): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    // Old path: take(NDArray, axis?)
    if (args[0] instanceof NDArray) {
      const axis = args[1] ?? 0;
      return new NDArray<T>(
        native()[`take_${nd}`](this._handle, args[0]._handle, axis), this.dtype,
      );
    }
    // New path: multi-axis specs
    const jsSpecs = args.map((s: any) => {
      if (s === null || s === undefined) return null;
      if (typeof s === "number") return s;
      if (Array.isArray(s))
        return native().NativeInt32NDArray.from_js(new Int32Array(s), [s.length]);
      return s._handle; // NDArray<Int32Array>
    });
    return new NDArray<T>(
      native()[`multi_take_${nd}`](this._handle, jsSpecs), this.dtype,
    );
  }

  /** Gather per-element along axis using index array (np.take_along_axis). */
  takeAlongAxis(indices: NDArray<Int32Array>, axis: number): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`take_along_axis_${nd}`](this._handle, indices._handle, axis), this.dtype,
    );
  }

  /** Return sorted copy (lexicographic row sort for nD). */
  sort(): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(native()[`sort_${nd}`](this._handle), this.dtype);
  }

  /** Sort in-place (lexicographic row sort for nD). */
  sort_(): this {
    const nd = nativeDtype(this.dtype);
    native()[`sort_inplace_${nd}`](this._handle);
    return this;
  }

  /** Return int32 indices that would sort this array (lexicographic for nD). */
  argsort(): NDArrayInt32 {
    const nd = nativeDtype(this.dtype);
    return new NDArray<Int32Array>(native()[`argsort_${nd}`](this._handle), "int32");
  }

  /** Filter elements (1D) or rows (nD) by boolean mask. */
  booleanIndex(mask: NDArrayBool): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`boolean_index_${nd}`](this._handle, mask._handle), this.dtype,
    );
  }

  // ============ Assign ============

  /**
   * In-place assignment with broadcasting.
   *
   * - `assign(value)` — fill with scalar or broadcast array.
   * - `assign(boolMask, value)` — masked assignment (set where mask is true).
   * - `assign(int32Indices, value)` — indexed assignment (set at given row indices).
   */
  assign(value: number): this;
  assign(value: NDArray): this;
  assign(selector: NDArray | number[], value: number): this;
  assign(selector: NDArray | number[], value: NDArray): this;
  assign(selectorOrValue: NDArray | number | number[], value?: NDArray | number): this {
    const nd = nativeDtype(this.dtype);
    if (value === undefined) {
      // 1-arg: assign(scalar) or assign(array)
      if (typeof selectorOrValue === "number") {
        native()[`assign_scalar_${nd}`](this._handle, selectorOrValue);
      } else {
        native()[`assign_array_${nd}`](this._handle, (selectorOrValue as NDArray)._handle);
      }
    } else {
      // 2-arg: assign(selector, value) — auto-convert number[] to int32 NDArray
      let sel: NDArray;
      if (Array.isArray(selectorOrValue)) {
        const arr = new Int32Array(selectorOrValue);
        sel = new NDArray(native().NativeInt32NDArray.from_js(arr, [arr.length]), "int32");
      } else {
        sel = selectorOrValue as NDArray;
      }
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

  /** Transpose (reverse axes). Shorthand for `transpose()`. */
  get T(): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`transpose_${nd}`](this._handle, undefined), this.dtype,
    );
  }

  /** Transpose with optional axis permutation. Without axes: reverses all dimensions. */
  transpose(axes?: number[]): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(
      native()[`transpose_${nd}`](this._handle, axes), this.dtype,
    );
  }

  // ============ Shape ops (zero-copy) ============

  /** Flatten to 1D. Zero-copy (shared view). */
  flatten(): NDArray<T> {
    const view = this._handle.shallow_copy();
    view.set_shape([this.length]);
    return new NDArray<T>(view, this.dtype);
  }

  /** Remove size-1 dimensions. If axis given, only squeeze that axis. Zero-copy. */
  squeeze(axis?: number): NDArray<T> {
    const s = this.shape;
    const newShape: number[] = [];
    for (let i = 0; i < s.length; i++) {
      if (s[i] === 1 && (axis === undefined || axis === i)) continue;
      newShape.push(s[i]);
    }
    if (newShape.length === 0) newShape.push(1);
    const view = this._handle.shallow_copy();
    view.set_shape(newShape);
    return new NDArray<T>(view, this.dtype);
  }

  /** Insert a size-1 dimension at the given axis. Zero-copy. */
  unsqueeze(axis: number): NDArray<T> {
    const s = this.shape.slice();
    s.splice(axis, 0, 1);
    const view = this._handle.shallow_copy();
    view.set_shape(s);
    return new NDArray<T>(view, this.dtype);
  }

  /** Reshape to new dimensions (must preserve total element count). Zero-copy. */
  reshape(shape: number[]): NDArray<T> {
    const view = this._handle.shallow_copy();
    view.set_shape(shape);
    return new NDArray<T>(view, this.dtype);
  }

  // ============ Clone ============

  /** Deep copy of this array. */
  clone(): NDArray<T> {
    const nd = nativeDtype(this.dtype);
    return new NDArray<T>(native()[`clone_${nd}`](this._handle), this.dtype);
  }

  // ============ Lifecycle ============

  /** Free WASM memory. Called automatically by GC via FinalizationRegistry. */
  delete(): void {
    this._handle.destroy();
  }

  /** Disposable protocol — allows `using arr = ...`. */
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
