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

import { NDArray, NativeNDArray } from "../ndarray/NDArray";

export type PrimitiveType =
  | "point" | "vector" | "segment" | "triangle"
  | "ray" | "line" | "plane" | "aabb" | "polygon";

/**
 * A WASM-resident geometric primitive backed by an NDArray<Float32Array>.
 *
 * The `type` field discriminates between primitive kinds.
 * Dimensionality (2D/3D) is inferred from shape.
 * Shape encodes single vs batch:
 * - Single point: shape [3]
 * - Batch of N points: shape [N, 3]
 */
export class Primitive extends NDArray<Float32Array> {
  readonly type: PrimitiveType;

  /** @internal */
  constructor(handle: NativeNDArray<Float32Array>, type: PrimitiveType) {
    super(handle, "float32");
    this.type = type;
  }

  /** Whether this primitive represents a batch (leading batch dimension). */
  get isBatch(): boolean {
    return this.ndim > singleNdim(this.type);
  }

  /** Number of elements in a batch, or 1 for a single primitive. */
  get count(): number {
    return this.isBatch ? this.shape[0] : 1;
  }

  /** For batch primitives, returns the i-th element as a single Primitive. */
  at(i: number): Primitive {
    if (!this.isBatch) {
      if (i !== 0) throw new RangeError("Single primitive, index must be 0");
      return this;
    }
    return new Primitive(this._handle.row(i), this.type);
  }
}

function singleNdim(type: PrimitiveType): number {
  switch (type) {
    case "point":
    case "vector":
    case "plane":
      return 1;
    case "segment":
    case "triangle":
    case "ray":
    case "line":
    case "aabb":
    case "polygon":
      return 2;
  }
}

export type Point    = Primitive & { readonly type: "point" };
export type Vector   = Primitive & { readonly type: "vector" };
export type Segment  = Primitive & { readonly type: "segment" };
export type Triangle = Primitive & { readonly type: "triangle" };
export type Ray      = Primitive & { readonly type: "ray" };
export type Line     = Primitive & { readonly type: "line" };
export type Plane    = Primitive & { readonly type: "plane" };
export type AABB     = Primitive & { readonly type: "aabb" };
export type Polygon  = Primitive & { readonly type: "polygon" };
