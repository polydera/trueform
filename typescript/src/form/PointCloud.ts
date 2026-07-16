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
import { registry } from "../internal/registry";
import { NDArray, NativeNDArray, NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer, NativeOffsetBlockedIntBuffer } from "../ndarray/OffsetBlockedBuffer";
import { ndarray } from "../ndarray/factories";
import { Mesh } from "./Mesh";

type NativePointNDArray = NativeNDArray<Float32Array> | NativeNDArray<Float64Array>;

interface NativePointCloud {
  points(): NativePointNDArray;
  number_of_points(): number;
  set_points(points: NativePointNDArray): void;
  vertex_link(): NativeOffsetBlockedIntBuffer;
  has_vertex_link(): boolean;
  set_vertex_link(vl: NativeOffsetBlockedIntBuffer): void;
  normals(): NativePointNDArray;
  has_normals(): boolean;
  set_normals(n: NativePointNDArray): void;
  shallow_copy(): NativePointCloud;
  has_transformation(): boolean;
  transformation(): NativePointNDArray;
  set_transformation(t: NativePointNDArray): void;
  clear_transformation(): void;
  build_tree(): void;
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
 * A WASM-resident point cloud with lazy spatial tree.
 *
 * Holds point coordinates [V*3] in the WASM heap. The spatial AABB tree is
 * built lazily on first spatial query. Vertex link and normals are settable
 * (no auto-computation since there are no faces).
 *
 * Use `PointCloud.fromMesh(m)` to create from a Mesh, copying its
 * vertex link and point normals.
 */
export class PointCloud {
  /** @internal */
  readonly _handle: NativePointCloud;
  /** Runtime type discriminator for vertex coordinates: "float32" or "float64". */
  readonly dtype: "float32" | "float64";

  /** @internal */
  constructor(handle: NativePointCloud, dtype: "float32" | "float64") {
    this._handle = handle;
    this.dtype = dtype;
    registry.register(this, { handle });
  }

  /** Vertex coordinates as NDArray [V, 3] with matching dtype. */
  get points(): NDArrayFloat32 | NDArrayFloat64 {
    return wrapPointArray(this._handle.points(), this.dtype);
  }

  /** Replace vertex coordinates. Invalidates the spatial tree. */
  set points(p: NDArrayFloat32 | NDArrayFloat64 | Float32Array | Float64Array) {
    if (p instanceof NDArray) {
      this._handle.set_points(p._handle);
    } else {
      const arr: NDArrayFloat32 | NDArrayFloat64 = p instanceof Float64Array
        ? ndarray(p, [p.length / 3, 3])
        : ndarray(p, [p.length / 3, 3]);
      this._handle.set_points(arr._handle);
    }
  }

  /** Number of points. */
  get numberOfPoints(): number {
    return this._handle.number_of_points();
  }

  /** Per-vertex adjacent vertices, or null if not set. */
  get vertexLink(): OffsetBlockedBuffer | null {
    if (!this._handle.has_vertex_link()) return null;
    return new OffsetBlockedBuffer(this._handle.vertex_link());
  }

  /** Set vertex link (precomputed adjacency structure). */
  set vertexLink(vl: OffsetBlockedBuffer) {
    this._handle.set_vertex_link(vl._handle);
  }

  /** Point normals as NDArray [V, 3] with matching dtype, or null if not set. */
  get normals(): NDArrayFloat32 | NDArrayFloat64 | null {
    if (!this._handle.has_normals()) return null;
    return wrapPointArray(this._handle.normals(), this.dtype);
  }

  /** Set point normals. */
  set normals(n: NDArrayFloat32 | NDArrayFloat64) {
    this._handle.set_normals(n._handle);
  }

  /** 4x4 transformation matrix as NDArray [4,4] with matching dtype, or null if none. */
  get transformation(): NDArrayFloat32 | NDArrayFloat64 | null {
    if (!this._handle.has_transformation()) return null;
    return wrapPointArray(this._handle.transformation(), this.dtype);
  }

  /** Set a 4x4 transformation matrix, or null to clear. A matrix of the
   * other float dtype is converted to the point cloud's dtype. */
  set transformation(t: NDArrayFloat32 | NDArrayFloat64 | Float32Array | Float64Array | null) {
    if (t === null) {
      this._handle.clear_transformation();
      return;
    }
    const arr =
      t instanceof NDArray ? t
      : t instanceof Float64Array ? ndarray(t, [4, 4])
      : ndarray(t, [4, 4]);
    const safe = arr.dtype === this.dtype ? arr : arr.as(this.dtype);
    this._handle.set_transformation(safe._handle);
  }

  /**
   * Forks the point cloud into a new handle that shares the underlying
   * buffers but has its own cache slots; the transformation is cleared
   * so the copy can take a different pose.
   */
  shallowCopy(): PointCloud {
    return new PointCloud(this._handle.shallow_copy(), this.dtype);
  }

  /** Pre-build the spatial AABB tree. No-op if already built and up-to-date. */
  buildTree(): void {
    this._handle.build_tree();
  }

  /**
   * Create a PointCloud from a Mesh.
   * Copies the mesh's points, vertex link, and point normals.
   * The resulting PointCloud inherits the mesh's dtype.
   */
  static fromMesh(m: Mesh): PointCloud {
    const n = native();
    const Native = m.dtype === "float64" ? n.NativeFloat64PointCloud : n.NativeFloat32PointCloud;
    const pc = new PointCloud(
      Native.create(m._handle.points()),
      m.dtype,
    );
    pc._handle.set_vertex_link(m._handle.vertex_link());
    pc._handle.set_normals(m._handle.point_normals());
    return pc;
  }

  delete(): void {
    this._handle.destroy();
  }

  [Symbol.dispose](): void {
    this._handle.destroy();
  }
}
