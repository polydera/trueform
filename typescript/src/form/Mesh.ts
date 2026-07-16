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
import { NDArray, NativeNDArray, NDArrayInt32, NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer, NativeOffsetBlockedIntBuffer } from "../ndarray/OffsetBlockedBuffer";
import { ndarray } from "../ndarray/factories";

type NativePointNDArray = NativeNDArray<Float32Array> | NativeNDArray<Float64Array>;

function wrapPointArray(
  handle: NativePointNDArray,
  dtype: "float32" | "float64",
): NDArrayFloat32 | NDArrayFloat64 {
  return dtype === "float64"
    ? new NDArray<Float64Array>(handle as NativeNDArray<Float64Array>, "float64")
    : new NDArray<Float32Array>(handle as NativeNDArray<Float32Array>, "float32");
}

interface NativeMesh {
  faces(): NativeNDArray<Int32Array>;
  points(): NativePointNDArray;
  number_of_faces(): number;
  number_of_points(): number;
  set_faces(faces: NativeNDArray<Int32Array>): void;
  set_points(points: NativePointNDArray): void;
  shallow_copy(): NativeMesh;
  build_tree(): void;
  has_transformation(): boolean;
  transformation(): NativePointNDArray;
  set_transformation(t: NativePointNDArray): void;
  clear_transformation(): void;
  face_membership(): NativeOffsetBlockedIntBuffer;
  manifold_edge_link(): NativeNDArray<Int32Array>;
  face_link(): NativeOffsetBlockedIntBuffer;
  vertex_link(): NativeOffsetBlockedIntBuffer;
  set_face_membership(fm: NativeOffsetBlockedIntBuffer): void;
  set_vertex_link(vl: NativeOffsetBlockedIntBuffer): void;
  set_face_link(fl: NativeOffsetBlockedIntBuffer): void;
  set_manifold_edge_link(mel: NativeNDArray<Int32Array>): void;
  normals(): NativePointNDArray;
  point_normals(): NativePointNDArray;
  set_normals(n: NativePointNDArray): void;
  set_point_normals(pn: NativePointNDArray): void;
  destroy(): void;
  is_valid(): boolean;
  delete(): void;
}

/**
 * A WASM-resident triangle mesh with lazy topology structures.
 *
 * Holds flat face indices [F*3] and point coordinates [V*3] in the WASM heap.
 * Topology structures (faceMembership, faceLink, vertexLink, manifoldEdgeLink)
 * are built lazily on first access and cached. Mutating faces/points via
 * Setting faces/points invalidates stale caches.
 *
 * shallowCopy() creates a forked copy that shares the underlying buffers
 * but has its own cache slots and (cleared) transformation.
 * Each accessor returns an independent handle safe against manual .delete().
 */
export class Mesh {
  /** @internal */
  readonly _handle: NativeMesh;
  /** Runtime type discriminator for vertex coordinates: "float32" or "float64". */
  readonly dtype: "float32" | "float64";

  /** @internal */
  constructor(handle: NativeMesh, dtype: "float32" | "float64") {
    this._handle = handle;
    this.dtype = dtype;
    registry.register(this, { handle });
  }

  /** Face indices as NDArrayInt32 [F, 3]. */
  get faces(): NDArrayInt32 {
    return new NDArray(this._handle.faces(), "int32");
  }

  /** Replace face indices. Accepts NDArray or Int32Array. Invalidates topology caches. */
  set faces(f: NDArrayInt32 | Int32Array) {
    if (f instanceof NDArray) {
      this._handle.set_faces(f._handle);
    } else {
      this._handle.set_faces(ndarray(f, [f.length / 3, 3])._handle);
    }
  }

  /** Vertex coordinates as NDArray [V, 3] with matching dtype. */
  get points(): NDArrayFloat32 | NDArrayFloat64 {
    return wrapPointArray(this._handle.points(), this.dtype);
  }

  /** Replace vertex coordinates. Accepts NDArray or typed array matching the mesh dtype. */
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

  /** Number of triangular faces. */
  get numberOfFaces(): number {
    return this._handle.number_of_faces();
  }

  /** Number of vertices. */
  get numberOfPoints(): number {
    return this._handle.number_of_points();
  }

  /** 4x4 transformation matrix as NDArray [4,4] with matching dtype, or null if none. */
  get transformation(): NDArrayFloat32 | NDArrayFloat64 | null {
    if (!this._handle.has_transformation()) return null;
    return wrapPointArray(this._handle.transformation(), this.dtype);
  }

  /** Set a 4x4 transformation matrix, or null to clear. A matrix of the
   * other float dtype is converted to the mesh's dtype. */
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
   * Forks the mesh into a new handle that shares the underlying buffers
   * but has its own cache slots; the transformation is cleared so the
   * copy can take a different pose.
   */
  shallowCopy(): Mesh {
    return new Mesh(this._handle.shallow_copy(), this.dtype);
  }

  /**
   * Per-vertex face membership: vertex i belongs to faces in block i.
   * Ranges are sorted in descending order.
   */
  get faceMembership(): OffsetBlockedBuffer {
    return new OffsetBlockedBuffer(this._handle.face_membership());
  }

  /** Set precomputed face membership (bypasses lazy computation). */
  set faceMembership(fm: OffsetBlockedBuffer) {
    this._handle.set_face_membership(fm._handle);
  }

  /**
   * Per-face neighbor faces via shared manifold edges, as NDArrayInt32 [F, 3].
   *
   * Sentinel values:
   * - `>= 0`: neighbor face index (simple manifold edge)
   * - `-1`: boundary edge (edge belongs to only one face)
   * - `-2`: non-manifold edge (3+ faces share this edge)
   * - `-3`: non-manifold representative edge
   */
  get manifoldEdgeLink(): NDArrayInt32 {
    return new NDArray(this._handle.manifold_edge_link(), "int32");
  }

  /** Set precomputed manifold edge link (bypasses lazy computation). */
  set manifoldEdgeLink(mel: NDArrayInt32) {
    this._handle.set_manifold_edge_link(mel._handle);
  }

  /** Per-face adjacent faces (faces sharing at least one vertex). */
  get faceLink(): OffsetBlockedBuffer {
    return new OffsetBlockedBuffer(this._handle.face_link());
  }

  /** Set precomputed face link (bypasses lazy computation). */
  set faceLink(fl: OffsetBlockedBuffer) {
    this._handle.set_face_link(fl._handle);
  }

  /** Per-vertex adjacent vertices (connected by an edge). */
  get vertexLink(): OffsetBlockedBuffer {
    return new OffsetBlockedBuffer(this._handle.vertex_link());
  }

  /** Set precomputed vertex link (bypasses lazy computation). */
  set vertexLink(vl: OffsetBlockedBuffer) {
    this._handle.set_vertex_link(vl._handle);
  }

  /** Face normals as NDArray [F, 3] with matching dtype. Lazily computed and cached. */
  get normals(): NDArrayFloat32 | NDArrayFloat64 {
    return wrapPointArray(this._handle.normals(), this.dtype);
  }

  /** Set precomputed face normals (bypasses lazy computation). */
  set normals(n: NDArrayFloat32 | NDArrayFloat64) {
    this._handle.set_normals(n._handle);
  }

  /** Vertex normals as NDArray [V, 3] with matching dtype. Lazily computed and cached. */
  get pointNormals(): NDArrayFloat32 | NDArrayFloat64 {
    return wrapPointArray(this._handle.point_normals(), this.dtype);
  }

  /** Set precomputed vertex normals (bypasses lazy computation). */
  set pointNormals(pn: NDArrayFloat32 | NDArrayFloat64) {
    this._handle.set_point_normals(pn._handle);
  }

  /** Pre-build the spatial AABB tree. No-op if already built and up-to-date. */
  buildTree(): void {
    this._handle.build_tree();
  }

  delete(): void {
    this._handle.destroy();
  }

  [Symbol.dispose](): void {
    this._handle.destroy();
  }
}
