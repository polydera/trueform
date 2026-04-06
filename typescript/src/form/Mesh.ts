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
import { NDArray, NativeNDArray, NDArrayInt32, NDArrayFloat32 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer, NativeOffsetBlockedIntBuffer } from "../ndarray/OffsetBlockedBuffer";
import { ndarray } from "../ndarray/factories";

interface NativeMesh {
  faces(): NativeNDArray<Int32Array>;
  points(): NativeNDArray<Float32Array>;
  number_of_faces(): number;
  number_of_points(): number;
  set_faces(faces: NativeNDArray<Int32Array>): void;
  set_points(points: NativeNDArray<Float32Array>): void;
  shared_view(): NativeMesh;
  has_transformation(): boolean;
  transformation(): NativeNDArray<Float32Array>;
  set_transformation(t: NativeNDArray<Float32Array>): void;
  clear_transformation(): void;
  face_membership(): NativeOffsetBlockedIntBuffer;
  manifold_edge_link(): NativeNDArray<Int32Array>;
  face_link(): NativeOffsetBlockedIntBuffer;
  vertex_link(): NativeOffsetBlockedIntBuffer;
  set_face_membership(fm: NativeOffsetBlockedIntBuffer): void;
  set_vertex_link(vl: NativeOffsetBlockedIntBuffer): void;
  set_face_link(fl: NativeOffsetBlockedIntBuffer): void;
  set_manifold_edge_link(mel: NativeNDArray<Int32Array>): void;
  normals(): NativeNDArray<Float32Array>;
  point_normals(): NativeNDArray<Float32Array>;
  set_normals(n: NativeNDArray<Float32Array>): void;
  set_point_normals(pn: NativeNDArray<Float32Array>): void;
  ensure_tree(): void;
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
 * sharedView() creates a cheap copy that shares all data and caches.
 * Each accessor returns an independent handle safe against manual .delete().
 */
export class Mesh {
  /** @internal */
  readonly _handle: NativeMesh;

  /** @internal */
  constructor(handle: NativeMesh) {
    this._handle = handle;
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

  /** Vertex coordinates as NDArrayFloat32 [V, 3]. */
  get points(): NDArrayFloat32 {
    return new NDArray(this._handle.points(), "float32");
  }

  /** Replace vertex coordinates. Accepts NDArray or Float32Array. */
  set points(p: NDArrayFloat32 | Float32Array) {
    if (p instanceof NDArray) {
      this._handle.set_points(p._handle);
    } else {
      this._handle.set_points(ndarray(p, [p.length / 3, 3])._handle);
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
  sharedView(): Mesh {
    return new Mesh(this._handle.shared_view());
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

  /** Face normals as NDArrayFloat32 [F, 3]. Lazily computed and cached. */
  get normals(): NDArrayFloat32 {
    return new NDArray(this._handle.normals(), "float32");
  }

  /** Set precomputed face normals (bypasses lazy computation). */
  set normals(n: NDArrayFloat32) {
    this._handle.set_normals(n._handle);
  }

  /** Vertex normals as NDArrayFloat32 [V, 3]. Lazily computed and cached. */
  get pointNormals(): NDArrayFloat32 {
    return new NDArray(this._handle.point_normals(), "float32");
  }

  /** Set precomputed vertex normals (bypasses lazy computation). */
  set pointNormals(pn: NDArrayFloat32) {
    this._handle.set_point_normals(pn._handle);
  }

  /** Pre-build the spatial AABB tree. No-op if already built and up-to-date. */
  buildTree(): void {
    this._handle.ensure_tree();
  }

  delete(): void {
    this._handle.destroy();
  }

  [Symbol.dispose](): void {
    this._handle.destroy();
  }
}
