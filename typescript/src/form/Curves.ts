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
import { NDArray, NativeNDArray, NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer, NativeOffsetBlockedIntBuffer } from "../ndarray/OffsetBlockedBuffer";

type NativePointNDArray = NativeNDArray<Float32Array> | NativeNDArray<Float64Array>;

interface NativeCurves {
  paths(): NativeOffsetBlockedIntBuffer;
  points(): NativePointNDArray;
  size(): number;
  destroy(): void;
  is_valid(): boolean;
  delete(): void;
}

function wrapPointArray(
  handle: NativePointNDArray,
  dtype: "float32" | "float64",
): NDArrayFloat32 | NDArrayFloat64 {
  return dtype === "float64"
    ? new NDArray<Float64Array>(handle as NativeNDArray<Float64Array>, "float64")
    : new NDArray<Float32Array>(handle as NativeNDArray<Float32Array>, "float32");
}

/**
 * WASM-resident curves: a collection of polyline paths with shared points.
 *
 * Each path is a sequence of vertex indices into the shared points buffer.
 * Used for intersection curves, isocontour curves, etc.
 */
export class Curves {
  /** @internal */
  readonly _handle: NativeCurves;
  /** Runtime type discriminator for curve point coordinates: "float32" or "float64". */
  readonly dtype: "float32" | "float64";

  /** @internal */
  constructor(handle: NativeCurves, dtype: "float32" | "float64") {
    this._handle = handle;
    this.dtype = dtype;
    registry.register(this, { handle });
  }

  /** Path index sequences as OffsetBlockedBuffer. */
  get paths(): OffsetBlockedBuffer {
    return new OffsetBlockedBuffer(this._handle.paths());
  }

  /** Curve point coordinates as NDArray [V, 3] with matching dtype. */
  get points(): NDArrayFloat32 | NDArrayFloat64 {
    return wrapPointArray(this._handle.points(), this.dtype);
  }

  /** Number of paths. */
  get length(): number {
    return this._handle.size();
  }

  delete(): void {
    this._handle.destroy();
  }

  [Symbol.dispose](): void {
    this._handle.destroy();
  }
}
