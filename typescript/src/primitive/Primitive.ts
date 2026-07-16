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

/** Discriminator for geometric primitive kinds. */
export type PrimitiveType =
  | "point" | "vector" | "segment" | "triangle"
  | "ray" | "line" | "plane" | "aabb" | "polygon";

/**
 * A WASM-resident geometric primitive backed by an NDArray.
 *
 * The `type` field discriminates between primitive kinds.
 * The `dtype` field selects coordinate precision ("float32" or "float64").
 * Dimensionality (2D/3D) is inferred from shape.
 * Shape encodes single vs batch:
 * - Single point: shape [3]
 * - Batch of N points: shape [N, 3]
 */
export class Primitive extends NDArray {
  readonly type: PrimitiveType;

  /** @internal */
  constructor(
    handle: NativeNDArray<any>,
    type: PrimitiveType,
    dtype: "float32" | "float64",
  ) {
    super(handle, dtype);
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
    return new Primitive(
      this._handle.row(i),
      this.type,
      this.dtype as "float32" | "float64",
    );
  }

  /**
   * Shallow copy: a new Primitive over the same WASM buffer, with the
   * same shape, offset, and primitive type. Values are shared;
   * metadata is independent afterwards.
   */
  shallowCopy(): Primitive {
    return new Primitive(
      this._handle.shallow_copy(),
      this.type,
      this.dtype as "float32" | "float64",
    );
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

/** A point in 2D or 3D space. Shape: `[D]` or batch `[N, D]`. */
export type Point    = Primitive & { readonly type: "point" };
/** A direction vector. Shape: `[D]` or batch `[N, D]`. */
export type Vector   = Primitive & { readonly type: "vector" };
/** A line segment defined by two endpoints. Shape: `[2, D]` or batch `[N, 2, D]`. */
export type Segment  = Primitive & { readonly type: "segment" };
/** A triangle defined by three vertices. Shape: `[3, D]` or batch `[N, 3, D]`. */
export type Triangle = Primitive & { readonly type: "triangle" };
/** A ray defined by origin and direction. Shape: `[2, D]` or batch `[N, 2, D]`. */
export type Ray      = Primitive & { readonly type: "ray" };
/** A line extending infinitely in both directions. Shape: `[2, D]` or batch `[N, 2, D]`. */
export type Line     = Primitive & { readonly type: "line" };
/** An infinite plane (ax+by+cz+d=0). Shape: `[4]` or batch `[N, 4]`. */
export type Plane    = Primitive & { readonly type: "plane" };
/** An axis-aligned bounding box. Shape: `[2, D]` or batch `[N, 2, D]`. */
export type AABB     = Primitive & { readonly type: "aabb" };
/** A polygon defined by ordered vertices. Shape: `[V, D]`. */
export type Polygon  = Primitive & { readonly type: "polygon" };
