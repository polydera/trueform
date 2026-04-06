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
import { NDArray, NativeNDArray, NDArrayFloat32 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer, NativeOffsetBlockedIntBuffer } from "../ndarray/OffsetBlockedBuffer";
import { ndarray } from "../ndarray/factories";
import { Mesh } from "./Mesh";

interface NativePointCloud {
  points(): NativeNDArray<Float32Array>;
  number_of_points(): number;
  set_points(points: NativeNDArray<Float32Array>): void;
  vertex_link(): NativeOffsetBlockedIntBuffer;
  has_vertex_link(): boolean;
  set_vertex_link(vl: NativeOffsetBlockedIntBuffer): void;
  normals(): NativeNDArray<Float32Array>;
  has_normals(): boolean;
  set_normals(n: NativeNDArray<Float32Array>): void;
  shared_view(): NativePointCloud;
  has_transformation(): boolean;
  transformation(): NativeNDArray<Float32Array>;
  set_transformation(t: NativeNDArray<Float32Array>): void;
  clear_transformation(): void;
  ensure_tree(): void;
  destroy(): void;
  is_valid(): boolean;
  delete(): void;
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

  /** @internal */
  constructor(handle: NativePointCloud) {
    this._handle = handle;
    registry.register(this, { handle });
  }

  /** Vertex coordinates as NDArrayFloat32 [V, 3]. */
  get points(): NDArrayFloat32 {
    return new NDArray(this._handle.points(), "float32");
  }

  /** Replace vertex coordinates. Invalidates the spatial tree. */
  set points(p: NDArrayFloat32 | Float32Array) {
    if (p instanceof NDArray) {
      this._handle.set_points(p._handle);
    } else {
      this._handle.set_points(ndarray(p, [p.length / 3, 3])._handle);
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

  /** Point normals as NDArrayFloat32 [V, 3], or null if not set. */
  get normals(): NDArrayFloat32 | null {
    if (!this._handle.has_normals()) return null;
    return new NDArray(this._handle.normals(), "float32");
  }

  /** Set point normals. */
  set normals(n: NDArrayFloat32) {
    this._handle.set_normals(n._handle);
  }

  /** 4x4 transformation matrix as NDArrayFloat32 [4,4], or null if none. */
  get transformation(): NDArrayFloat32 | null {
    if (!this._handle.has_transformation()) return null;
    return new NDArray(this._handle.transformation(), "float32");
  }

  /** Set a 4x4 transformation matrix, or null to clear. */
  set transformation(t: NDArrayFloat32 | Float32Array | null) {
    if (t === null) {
      this._handle.clear_transformation();
    } else if (t instanceof NDArray) {
      this._handle.set_transformation(t._handle);
    } else {
      this._handle.set_transformation(ndarray(t, [4, 4])._handle);
    }
  }

  /**
   * Creates a shared view — cheap copy sharing all data and caches.
   * Transformation is not shared (the view has no transformation).
   */
  sharedView(): PointCloud {
    return new PointCloud(this._handle.shared_view());
  }

  /** Pre-build the spatial AABB tree. No-op if already built and up-to-date. */
  buildTree(): void {
    this._handle.ensure_tree();
  }

  /**
   * Create a PointCloud from a Mesh.
   * Copies the mesh's points, vertex link, and point normals.
   */
  static fromMesh(m: Mesh): PointCloud {
    const pc = new PointCloud(
      native().NativePointCloud.create(m._handle.points()),
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
