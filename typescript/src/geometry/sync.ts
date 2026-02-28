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
import { NDArray, type NDArrayFloat32 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { Mesh } from "../form/Mesh";
import { PointCloud } from "../form/PointCloud";
import { Primitive, Polygon } from "../primitive";

export type { MeshLike } from "../form/MeshLike";
import type { MeshLike } from "../form/MeshLike";

/** Triangulate polygons into a triangle mesh. */
export function triangulate(input: Mesh | MeshLike | Polygon): Mesh {
  if (input instanceof Mesh) {
    return new Mesh(native().triangulate_mesh(input._handle));
  }
  if (input instanceof Primitive) {
    return new Mesh(native().triangulate_polygon(input._handle));
  }
  // MeshLike
  if (input.faces instanceof OffsetBlockedBuffer) {
    return new Mesh(native().triangulate_dynamic(input.faces._handle, input.points._handle));
  }
  return new Mesh(native().triangulate_fixed(input.faces._handle, input.points._handle));
}

/** Create a UV sphere mesh centered at origin. */
export function sphereMesh(radius: number, stacks: number, segments: number): Mesh {
  return new Mesh(native().make_sphere_mesh(radius, stacks, segments));
}

/** Create a cylinder mesh centered at origin along the z-axis. */
export function cylinderMesh(radius: number, height: number, segments: number): Mesh {
  return new Mesh(native().make_cylinder_mesh(radius, height, segments));
}

/** Create an axis-aligned box mesh centered at origin. Optionally subdivided. */
export function boxMesh(
  width: number, height: number, depth: number,
  widthTicks?: number, heightTicks?: number, depthTicks?: number,
): Mesh {
  if (widthTicks !== undefined && heightTicks !== undefined && depthTicks !== undefined) {
    return new Mesh(native().make_box_mesh_subdivided(width, height, depth, widthTicks, heightTicks, depthTicks));
  }
  return new Mesh(native().make_box_mesh(width, height, depth));
}

/** Create a flat rectangular plane mesh in the XY plane, centered at origin. */
export function planeMesh(
  width: number, height: number,
  widthTicks?: number, heightTicks?: number,
): Mesh {
  const wt = widthTicks ?? 1;
  const ht = heightTicks ?? 1;
  return new Mesh(native().make_plane_mesh(width, height, wt, ht));
}

// ============ Measurements ============

/** Total surface area of a mesh. Respects transformation. */
export function area(m: Mesh): number {
  return native().mesh_area(m._handle);
}

/** Signed volume of a closed 3D mesh. Positive for outward normals. Respects transformation. */
export function signedVolume(m: Mesh): number {
  return native().mesh_signed_volume(m._handle);
}

/** Absolute volume of a closed 3D mesh. Respects transformation. */
export function volume(m: Mesh): number {
  return Math.abs(native().mesh_signed_volume(m._handle));
}

/** Mean edge length across all faces. Respects transformation. */
export function meanEdgeLength(m: Mesh): number {
  return native().mesh_mean_edge_length(m._handle);
}

/** Minimum edge length across all faces. Respects transformation. */
export function minEdgeLength(m: Mesh): number {
  return native().mesh_min_edge_length(m._handle);
}

/** Maximum edge length across all faces. Respects transformation. */
export function maxEdgeLength(m: Mesh): number {
  return native().mesh_max_edge_length(m._handle);
}

// ============ Orientation ============

/** Return a new mesh with consistently oriented, outward-pointing normals. */
export function positivelyOriented(m: Mesh, isConsistent?: boolean): Mesh {
  return new Mesh(native().positively_oriented(m._handle, isConsistent ?? false));
}

// ============ Curvature ============

/** Principal curvature result: k0 (max) and k1 (min) per vertex. */
export interface PrincipalCurvatures {
  k0: NDArrayFloat32;
  k1: NDArrayFloat32;
}

/** Principal curvature result with directions. */
export interface PrincipalDirections {
  k0: NDArrayFloat32;
  k1: NDArrayFloat32;
  d0: NDArrayFloat32;
  d1: NDArrayFloat32;
}

/** Compute principal curvatures (k0, k1) at each vertex. */
export function principalCurvatures(m: Mesh, k: number = 2): PrincipalCurvatures {
  const raw = native().principal_curvatures(m._handle, k);
  return {
    k0: new NDArray(raw.k0, "float32"),
    k1: new NDArray(raw.k1, "float32"),
  };
}

/** Compute principal curvatures and directions at each vertex. */
export function principalDirections(m: Mesh, k: number = 2): PrincipalDirections {
  const raw = native().principal_directions(m._handle, k);
  return {
    k0: new NDArray(raw.k0, "float32"),
    k1: new NDArray(raw.k1, "float32"),
    d0: new NDArray(raw.d0, "float32"),
    d1: new NDArray(raw.d1, "float32"),
  };
}

/** Compute shape index at each vertex. Values in [-1, 1]. */
export function shapeIndex(m: Mesh, k: number = 2): NDArrayFloat32 {
  return new NDArray(native().shape_index(m._handle, k), "float32");
}

// ============ Smoothing ============

/** Laplacian smoothing. Returns a new mesh with smoothed vertex positions. */
export function laplacianSmoothed(m: Mesh, iterations: number, lambda: number = 0.5): Mesh {
  return new Mesh(native().laplacian_smoothed(m._handle, iterations, lambda));
}

/** Taubin smoothing (volume-preserving). Returns a new mesh with smoothed vertex positions. */
export function taubinSmoothed(m: Mesh, iterations: number, lambda: number = 0.5, kpb: number = 0.1): Mesh {
  return new Mesh(native().taubin_smoothed(m._handle, iterations, lambda, kpb));
}

// ============ Registration / Alignment ============

/** Options for ICP alignment. */
export interface IcpOptions {
  maxIterations?: number;
  nSamples?: number;
  k?: number;
  sigma?: number;
  outlierProportion?: number;
  minRelativeImprovement?: number;
  emaAlpha?: number;
}

/** Options for OBB alignment. */
export interface ObbOptions {
  sampleSize?: number;
}

/** Options for Chamfer error. */
export interface ChamferOptions {
  outlierProportion?: number;
}

/** Kabsch/SVD rigid alignment. Requires 1:1 correspondence (|source| == |target|). Returns 4x4 delta matrix. */
export function fitRigidAlignment(source: PointCloud, target: PointCloud): NDArrayFloat32 {
  return new NDArray(native().fit_rigid_alignment(source._handle, target._handle), "float32");
}

/** Iterative Closest Point alignment. Returns 4x4 delta matrix. */
export function fitIcpAlignment(source: PointCloud, target: PointCloud, opts?: IcpOptions): NDArrayFloat32 {
  return new NDArray(native().fit_icp_alignment(
    source._handle, target._handle,
    opts?.maxIterations ?? 100,
    opts?.nSamples ?? 1000,
    opts?.k ?? 1,
    opts?.sigma ?? -1,
    opts?.outlierProportion ?? 0,
    opts?.minRelativeImprovement ?? 1e-6,
    opts?.emaAlpha ?? 0.3,
  ), "float32");
}

/** Coarse alignment via oriented bounding boxes. Returns 4x4 delta matrix. */
export function fitObbAlignment(source: PointCloud, target: PointCloud, opts?: ObbOptions): NDArrayFloat32 {
  return new NDArray(native().fit_obb_alignment(
    source._handle, target._handle,
    opts?.sampleSize ?? 100,
  ), "float32");
}

/** One-way Chamfer distance (mean nearest-neighbor distance from source to target). */
export function chamferError(source: PointCloud, target: PointCloud, opts?: ChamferOptions): number {
  return native().chamfer_error(
    source._handle, target._handle,
    opts?.outlierProportion ?? 0,
  );
}
