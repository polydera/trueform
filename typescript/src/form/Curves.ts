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
import { NDArray, NDArrayFloat32 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";

interface NativeOffsetBlockedIntBuffer {
  offsets(): any;
  data(): any;
  size(): number;
  get(i: number): any;
  destroy(): void;
  is_valid(): boolean;
  delete(): void;
}

interface NativeCurves {
  paths(): NativeOffsetBlockedIntBuffer;
  points(): any;
  size(): number;
  destroy(): void;
  is_valid(): boolean;
  delete(): void;
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

  /** @internal */
  constructor(handle: NativeCurves) {
    this._handle = handle;
    registry.register(this, { handle });
  }

  /** Path index sequences as OffsetBlockedBuffer. */
  get paths(): OffsetBlockedBuffer {
    return new OffsetBlockedBuffer(this._handle.paths());
  }

  /** Curve point coordinates as NDArrayFloat32 [V, 3]. */
  get points(): NDArrayFloat32 {
    return new NDArray(this._handle.points(), "float32");
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
