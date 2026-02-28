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
import { NDArray, NativeNDArray, NDArrayInt32 } from "./NDArray";

interface NativeOffsetBlockedIntBuffer {
  offsets(): NativeNDArray<Int32Array>;
  data(): NativeNDArray<Int32Array>;
  size(): number;
  get(i: number): NativeNDArray<Int32Array>;
  destroy(): void;
  is_valid(): boolean;
  delete(): void;
}

/**
 * A WASM-resident offset-blocked buffer of int32 values.
 *
 * Wraps two flat arrays (offsets + data) where block i spans
 * data[offsets[i] .. offsets[i+1]]. Access and iteration use
 * JS-side subarray views — no C++ calls per element.
 *
 * Used for topology structures: faceMembership, faceLink, vertexLink.
 */
export class OffsetBlockedBuffer {
  /** @internal */
  readonly _handle: NativeOffsetBlockedIntBuffer;

  /** @internal */
  constructor(handle: NativeOffsetBlockedIntBuffer) {
    this._handle = handle;
    registry.register(this, { handle });
  }

  /** Number of blocks. */
  get length(): number {
    return this._handle.size();
  }

  /** Offsets array [N+1] as NDArrayInt32. */
  get offsets(): NDArrayInt32 {
    return new NDArray(this._handle.offsets(), "int32");
  }

  /** Flat data array as NDArrayInt32. */
  get data(): NDArrayInt32 {
    return new NDArray(this._handle.data(), "int32");
  }

  /** Returns block i as NDArrayInt32. */
  get(i: number): NDArrayInt32 {
    return new NDArray(this._handle.get(i), "int32");
  }

  /** Iterate over blocks as NDArrayInt32. */
  *[Symbol.iterator](): Iterator<NDArrayInt32> {
    const n = this._handle.size();
    for (let i = 0; i < n; i++) {
      yield new NDArray(this._handle.get(i), "int32");
    }
  }

  delete(): void {
    this._handle.destroy();
  }

  [Symbol.dispose](): void {
    this._handle.destroy();
  }
}
