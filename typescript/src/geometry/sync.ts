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
import { NDArray, type NDArrayFloat32, type NDArrayFloat64, type NDArrayInt32 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { Curves } from "../form/Curves";
import { Mesh } from "../form/Mesh";
import { PointCloud } from "../form/PointCloud";
import { Primitive, Polygon } from "../primitive";
import { assertSameDtype } from "../internal/dtype";

export type { MeshLike } from "../form/MeshLike";
import type { MeshLike } from "../form/MeshLike";

/** Triangulate polygons into a triangle mesh. */
export function triangulate(input: Mesh | MeshLike | Polygon): Mesh {
  if (input instanceof Mesh) {
    const dt = input.dtype;
    return new Mesh(native()[`triangulate_mesh_${dt}`](input._handle), dt);
  }
  if (input instanceof Primitive) {
    const dt = input.dtype === "float64" ? "float64" : "float32";
    return new Mesh(native()[`triangulate_polygon_${dt}`](input._handle), dt);
  }
  // MeshLike
  const dt = input.points.dtype === "float64" ? "float64" : "float32";
  if (input.faces instanceof OffsetBlockedBuffer) {
    return new Mesh(native()[`triangulate_dynamic_${dt}`](input.faces._handle, input.points._handle), dt);
  }
  return new Mesh(native()[`triangulate_fixed_${dt}`](input.faces._handle, input.points._handle), dt);
}

/** Options selecting the coordinate dtype of returned mesh primitives. */
export interface DtypeOptions {
  dtype?: "float32" | "float64";
}

/** Create a UV sphere mesh centered at origin. */
export function sphereMesh(radius: number, stacks: number, segments: number, opts?: DtypeOptions): Mesh {
  const dt = opts?.dtype ?? "float32";
  return new Mesh(native()[`make_sphere_mesh_${dt}`](radius, stacks, segments), dt);
}

/** Create a cylinder mesh centered at origin along the z-axis. */
export function cylinderMesh(radius: number, height: number, segments: number, opts?: DtypeOptions): Mesh {
  const dt = opts?.dtype ?? "float32";
  return new Mesh(native()[`make_cylinder_mesh_${dt}`](radius, height, segments), dt);
}

/** Create an axis-aligned box mesh centered at origin. Optionally subdivided. */
export function boxMesh(
  width: number, height: number, depth: number,
  widthTicks?: number, heightTicks?: number, depthTicks?: number,
  opts?: DtypeOptions,
): Mesh {
  const dt = opts?.dtype ?? "float32";
  if (widthTicks !== undefined && heightTicks !== undefined && depthTicks !== undefined) {
    return new Mesh(native()[`make_box_mesh_subdivided_${dt}`](width, height, depth, widthTicks, heightTicks, depthTicks), dt);
  }
  return new Mesh(native()[`make_box_mesh_${dt}`](width, height, depth), dt);
}

/** Create a tube mesh from curves using parallel transport frames. Inherits dtype from the input curves. */
export function tubeMesh(curves: Curves, radius: number, radialSegments: number = 8): Mesh {
  const dt = curves.dtype;
  return new Mesh(native()[`make_tube_mesh_${dt}`](curves._handle, radius, radialSegments), dt);
}

/** Create a flat rectangular plane mesh in the XY plane, centered at origin. */
export function planeMesh(
  width: number, height: number,
  widthTicks?: number, heightTicks?: number,
  opts?: DtypeOptions,
): Mesh {
  const wt = widthTicks ?? 1;
  const ht = heightTicks ?? 1;
  const dt = opts?.dtype ?? "float32";
  return new Mesh(native()[`make_plane_mesh_${dt}`](width, height, wt, ht), dt);
}

// ============ Edge Analysis ============

/** Sharp edges where the dihedral angle exceeds the threshold (in degrees). Returns [N, 2] Int32 array of vertex index pairs. */
export function sharpEdges(m: Mesh, angleDeg: number): NDArrayInt32 {
  const dt = m.dtype;
  return new NDArray(native()[`sharp_edges_${dt}`](m._handle, angleDeg), "int32");
}

// ============ Measurements ============

/** Total surface area of a mesh. Respects transformation. */
export function area(m: Mesh): number {
  return native()[`mesh_area_${m.dtype}`](m._handle);
}

/** Signed volume of a closed 3D mesh. Positive for outward normals. Respects transformation. */
export function signedVolume(m: Mesh): number {
  return native()[`mesh_signed_volume_${m.dtype}`](m._handle);
}

/** Absolute volume of a closed 3D mesh. Respects transformation. */
export function volume(m: Mesh): number {
  return Math.abs(native()[`mesh_signed_volume_${m.dtype}`](m._handle));
}

/** Mean edge length across all faces. Respects transformation. */
export function meanEdgeLength(m: Mesh): number {
  return native()[`mesh_mean_edge_length_${m.dtype}`](m._handle);
}

/** Minimum edge length across all faces. Respects transformation. */
export function minEdgeLength(m: Mesh): number {
  return native()[`mesh_min_edge_length_${m.dtype}`](m._handle);
}

/** Maximum edge length across all faces. Respects transformation. */
export function maxEdgeLength(m: Mesh): number {
  return native()[`mesh_max_edge_length_${m.dtype}`](m._handle);
}

// ============ Orientation ============

/** Return a new mesh with reversed face winding (flipped normals). */
export function reverseWinding(m: Mesh): Mesh {
  const dt = m.dtype;
  return new Mesh(native()[`reverse_winding_${dt}`](m._handle), dt);
}

/** Return a new mesh with consistently oriented, outward-pointing normals. */
export function positivelyOriented(m: Mesh, isConsistent?: boolean): Mesh {
  const dt = m.dtype;
  return new Mesh(native()[`positively_oriented_${dt}`](m._handle, isConsistent ?? false), dt);
}

// ============ Curvature ============

/** Principal curvature result: k0 (max) and k1 (min) per vertex. */
export interface PrincipalCurvatures {
  k0: NDArrayFloat32 | NDArrayFloat64;
  k1: NDArrayFloat32 | NDArrayFloat64;
}

/** Principal curvature result with directions. */
export interface PrincipalDirections {
  k0: NDArrayFloat32 | NDArrayFloat64;
  k1: NDArrayFloat32 | NDArrayFloat64;
  d0: NDArrayFloat32 | NDArrayFloat64;
  d1: NDArrayFloat32 | NDArrayFloat64;
}

/** Compute principal curvatures (k0, k1) at each vertex. */
export function principalCurvatures(m: Mesh, k: number = 2): PrincipalCurvatures {
  const dt = m.dtype;
  const raw = native()[`principal_curvatures_${dt}`](m._handle, k);
  return {
    k0: new NDArray(raw.k0, dt) as NDArrayFloat32 | NDArrayFloat64,
    k1: new NDArray(raw.k1, dt) as NDArrayFloat32 | NDArrayFloat64,
  };
}

/** Compute principal curvatures and directions at each vertex. */
export function principalDirections(m: Mesh, k: number = 2): PrincipalDirections {
  const dt = m.dtype;
  const raw = native()[`principal_directions_${dt}`](m._handle, k);
  return {
    k0: new NDArray(raw.k0, dt) as NDArrayFloat32 | NDArrayFloat64,
    k1: new NDArray(raw.k1, dt) as NDArrayFloat32 | NDArrayFloat64,
    d0: new NDArray(raw.d0, dt) as NDArrayFloat32 | NDArrayFloat64,
    d1: new NDArray(raw.d1, dt) as NDArrayFloat32 | NDArrayFloat64,
  };
}

/** Compute shape index at each vertex. Values in [-1, 1]. */
export function shapeIndex(m: Mesh, k: number = 2): NDArrayFloat32 | NDArrayFloat64 {
  const dt = m.dtype;
  return new NDArray(native()[`shape_index_${dt}`](m._handle, k), dt) as NDArrayFloat32 | NDArrayFloat64;
}

// ============ Smoothing ============

/** Laplacian smoothing. Returns a new mesh with smoothed vertex positions. */
export function laplacianSmoothed(m: Mesh, iterations: number, lambda: number = 0.5): Mesh {
  const dt = m.dtype;
  return new Mesh(native()[`laplacian_smoothed_${dt}`](m._handle, iterations, lambda), dt);
}

/** Taubin smoothing (volume-preserving). Returns a new mesh with smoothed vertex positions. */
export function taubinSmoothed(m: Mesh, iterations: number, lambda: number = 0.5, kpb: number = 0.1): Mesh {
  const dt = m.dtype;
  return new Mesh(native()[`taubin_smoothed_${dt}`](m._handle, iterations, lambda, kpb), dt);
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
export function fitRigidAlignment(source: PointCloud, target: PointCloud): NDArrayFloat32 | NDArrayFloat64 {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return new NDArray(native()[`fit_rigid_alignment_${dt}`](source._handle, target._handle), dt) as NDArrayFloat32 | NDArrayFloat64;
}

/** Iterative Closest Point alignment. Returns 4x4 delta matrix. */
export function fitIcpAlignment(source: PointCloud, target: PointCloud, opts?: IcpOptions): NDArrayFloat32 | NDArrayFloat64 {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return new NDArray(native()[`fit_icp_alignment_${dt}`](
    source._handle, target._handle,
    opts?.maxIterations ?? 100,
    opts?.nSamples ?? 1000,
    opts?.k ?? 1,
    opts?.sigma ?? -1,
    opts?.outlierProportion ?? 0,
    opts?.minRelativeImprovement ?? 1e-6,
    opts?.emaAlpha ?? 0.3,
  ), dt) as NDArrayFloat32 | NDArrayFloat64;
}

/** Coarse alignment via oriented bounding boxes. Returns 4x4 delta matrix. */
export function fitObbAlignment(source: PointCloud, target: PointCloud, opts?: ObbOptions): NDArrayFloat32 | NDArrayFloat64 {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return new NDArray(native()[`fit_obb_alignment_${dt}`](
    source._handle, target._handle,
    opts?.sampleSize ?? 100,
  ), dt) as NDArrayFloat32 | NDArrayFloat64;
}

/** One-way Chamfer distance (mean nearest-neighbor distance from source to target). */
export function chamferError(source: PointCloud, target: PointCloud, opts?: ChamferOptions): number {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return native()[`chamfer_error_${dt}`](
    source._handle, target._handle,
    opts?.outlierProportion ?? 0,
  );
}
