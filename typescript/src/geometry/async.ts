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

import { native, dispatcher } from "../native";
import { NDArray, NDArrayInt32, NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { Curves } from "../form/Curves";
import { Mesh } from "../form/Mesh";
import { PointCloud } from "../form/PointCloud";
import { Primitive, Polygon } from "../primitive";
import { assertSameDtype } from "../internal/dtype";

import type { PrincipalCurvatures, PrincipalDirections, IcpOptions, ObbOptions, ChamferOptions, DtypeOptions } from "./sync";

import type { MeshLike } from "../form/MeshLike";

/** Triangulate polygons into a triangle mesh (async). */
export async function triangulate(input: Mesh | MeshLike | Polygon): Promise<Mesh> {
  if (input instanceof Mesh) {
    const dt = input.dtype;
    return dispatcher().run(
      () => native()[`dispatch_triangulate_mesh_${dt}`](input._handle),
      (raw) => new Mesh(raw, dt),
    );
  }
  if (input instanceof Primitive) {
    const dt = input.dtype === "float64" ? "float64" : "float32";
    return dispatcher().run(
      () => native()[`dispatch_triangulate_polygon_${dt}`](input._handle),
      (raw) => new Mesh(raw, dt),
    );
  }
  // MeshLike
  const dt = input.points.dtype === "float64" ? "float64" : "float32";
  if (input.faces instanceof OffsetBlockedBuffer) {
    return dispatcher().run(
      () => native()[`dispatch_triangulate_dynamic_${dt}`](input.faces._handle, input.points._handle),
      (raw) => new Mesh(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_triangulate_fixed_${dt}`](
      (input.faces as NDArrayInt32)._handle, input.points._handle,
    ),
    (raw) => new Mesh(raw, dt),
  );
}

/** Create a UV sphere mesh centered at origin (async). */
export async function sphereMesh(radius: number, stacks: number, segments: number, opts?: DtypeOptions): Promise<Mesh> {
  const dt = opts?.dtype ?? "float32";
  return dispatcher().run(
    () => native()[`dispatch_make_sphere_mesh_${dt}`](radius, stacks, segments),
    (raw) => new Mesh(raw, dt),
  );
}

/** Create a cylinder mesh centered at origin along the z-axis (async). */
export async function cylinderMesh(radius: number, height: number, segments: number, opts?: DtypeOptions): Promise<Mesh> {
  const dt = opts?.dtype ?? "float32";
  return dispatcher().run(
    () => native()[`dispatch_make_cylinder_mesh_${dt}`](radius, height, segments),
    (raw) => new Mesh(raw, dt),
  );
}

/** Create an axis-aligned box mesh centered at origin (async). Optionally subdivided. */
export async function boxMesh(
  width: number, height: number, depth: number,
  widthTicks?: number, heightTicks?: number, depthTicks?: number,
  opts?: DtypeOptions,
): Promise<Mesh> {
  const dt = opts?.dtype ?? "float32";
  if (widthTicks !== undefined && heightTicks !== undefined && depthTicks !== undefined) {
    return dispatcher().run(
      () => native()[`dispatch_make_box_mesh_subdivided_${dt}`](width, height, depth, widthTicks, heightTicks, depthTicks),
      (raw) => new Mesh(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_make_box_mesh_${dt}`](width, height, depth),
    (raw) => new Mesh(raw, dt),
  );
}

/** Create a tube mesh from curves using parallel transport frames (async). Inherits dtype from the input curves. */
export async function tubeMesh(curves: Curves, radius: number, radialSegments: number = 8): Promise<Mesh> {
  const dt = curves.dtype;
  return dispatcher().run(
    () => native()[`dispatch_make_tube_mesh_${dt}`](curves._handle, radius, radialSegments),
    (raw) => new Mesh(raw, dt),
  );
}

/** Create a flat rectangular plane mesh in the XY plane, centered at origin (async). */
export async function planeMesh(
  width: number, height: number,
  widthTicks?: number, heightTicks?: number,
  opts?: DtypeOptions,
): Promise<Mesh> {
  const wt = widthTicks ?? 1;
  const ht = heightTicks ?? 1;
  const dt = opts?.dtype ?? "float32";
  return dispatcher().run(
    () => native()[`dispatch_make_plane_mesh_${dt}`](width, height, wt, ht),
    (raw) => new Mesh(raw, dt),
  );
}

// ============ Edge Analysis ============

/** Sharp edges where the dihedral angle exceeds the threshold (in degrees, async). Returns [N, 2] Int32 array of vertex index pairs. */
export async function sharpEdges(m: Mesh, angleDeg: number): Promise<NDArrayInt32> {
  const dt = m.dtype;
  return dispatcher().run(
    () => native()[`dispatch_sharp_edges_${dt}`](m._handle, angleDeg),
    (raw) => new NDArray(raw, "int32"),
  );
}

// ============ Measurements ============

/** Total surface area of a mesh (async). */
export async function area(m: Mesh): Promise<number> {
  return dispatcher().run(() => native()[`dispatch_mesh_area_${m.dtype}`](m._handle));
}

/** Signed volume of a closed 3D mesh (async). */
export async function signedVolume(m: Mesh): Promise<number> {
  return dispatcher().run(() => native()[`dispatch_mesh_signed_volume_${m.dtype}`](m._handle));
}

/** Absolute volume of a closed 3D mesh (async). */
export async function volume(m: Mesh): Promise<number> {
  return dispatcher().run(
    () => native()[`dispatch_mesh_signed_volume_${m.dtype}`](m._handle),
    (v: number) => Math.abs(v),
  );
}

/** Mean edge length across all faces (async). */
export async function meanEdgeLength(m: Mesh): Promise<number> {
  return dispatcher().run(() => native()[`dispatch_mesh_mean_edge_length_${m.dtype}`](m._handle));
}

/** Minimum edge length across all faces (async). */
export async function minEdgeLength(m: Mesh): Promise<number> {
  return dispatcher().run(() => native()[`dispatch_mesh_min_edge_length_${m.dtype}`](m._handle));
}

/** Maximum edge length across all faces (async). */
export async function maxEdgeLength(m: Mesh): Promise<number> {
  return dispatcher().run(() => native()[`dispatch_mesh_max_edge_length_${m.dtype}`](m._handle));
}

// ============ Orientation ============

/** Return a new mesh with reversed face winding (flipped normals, async). */
export async function reverseWinding(m: Mesh): Promise<Mesh> {
  const dt = m.dtype;
  return dispatcher().run(
    () => native()[`dispatch_reverse_winding_${dt}`](m._handle),
    (raw) => new Mesh(raw, dt),
  );
}

/** Return a new mesh with consistently oriented, outward-pointing normals (async). */
export async function positivelyOriented(m: Mesh, isConsistent?: boolean): Promise<Mesh> {
  const ic = isConsistent ?? false;
  const dt = m.dtype;
  return dispatcher().run(
    () => native()[`dispatch_positively_oriented_${dt}`](m._handle, ic),
    (raw) => new Mesh(raw, dt),
  );
}

// ============ Curvature ============

/** Compute principal curvatures (k0, k1) at each vertex (async). */
export async function principalCurvatures(m: Mesh, k: number = 2): Promise<PrincipalCurvatures> {
  const dt = m.dtype;
  return dispatcher().run(
    () => native()[`dispatch_principal_curvatures_${dt}`](m._handle, k),
    (raw) => ({
      k0: new NDArray(raw.k0, dt) as NDArrayFloat32 | NDArrayFloat64,
      k1: new NDArray(raw.k1, dt) as NDArrayFloat32 | NDArrayFloat64,
    }),
  );
}

/** Compute principal curvatures and directions at each vertex (async). */
export async function principalDirections(m: Mesh, k: number = 2): Promise<PrincipalDirections> {
  const dt = m.dtype;
  return dispatcher().run(
    () => native()[`dispatch_principal_directions_${dt}`](m._handle, k),
    (raw) => ({
      k0: new NDArray(raw.k0, dt) as NDArrayFloat32 | NDArrayFloat64,
      k1: new NDArray(raw.k1, dt) as NDArrayFloat32 | NDArrayFloat64,
      d0: new NDArray(raw.d0, dt) as NDArrayFloat32 | NDArrayFloat64,
      d1: new NDArray(raw.d1, dt) as NDArrayFloat32 | NDArrayFloat64,
    }),
  );
}

/** Compute shape index at each vertex (async). Values in [-1, 1]. */
export async function shapeIndex(m: Mesh, k: number = 2): Promise<NDArrayFloat32 | NDArrayFloat64> {
  const dt = m.dtype;
  return dispatcher().run(
    () => native()[`dispatch_shape_index_${dt}`](m._handle, k),
    (raw) => new NDArray(raw, dt) as NDArrayFloat32 | NDArrayFloat64,
  );
}

// ============ Smoothing ============

/** Laplacian smoothing (async). Returns a new mesh with smoothed vertex positions. */
export async function laplacianSmoothed(m: Mesh, iterations: number, lambda: number = 0.5): Promise<Mesh> {
  const dt = m.dtype;
  return dispatcher().run(
    () => native()[`dispatch_laplacian_smoothed_${dt}`](m._handle, iterations, lambda),
    (raw) => new Mesh(raw, dt),
  );
}

/** Taubin smoothing (volume-preserving, async). Returns a new mesh with smoothed vertex positions. */
export async function taubinSmoothed(m: Mesh, iterations: number, lambda: number = 0.5, kpb: number = 0.1): Promise<Mesh> {
  const dt = m.dtype;
  return dispatcher().run(
    () => native()[`dispatch_taubin_smoothed_${dt}`](m._handle, iterations, lambda, kpb),
    (raw) => new Mesh(raw, dt),
  );
}

// ============ Registration / Alignment (async) ============

/** Kabsch/SVD rigid alignment (async). Returns 4x4 delta matrix. */
export async function fitRigidAlignment(source: PointCloud, target: PointCloud): Promise<NDArrayFloat32 | NDArrayFloat64> {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return dispatcher().run(
    () => native()[`dispatch_fit_rigid_alignment_${dt}`](source._handle, target._handle),
    (raw) => new NDArray(raw, dt) as NDArrayFloat32 | NDArrayFloat64,
  );
}

/** Iterative Closest Point alignment (async). Returns 4x4 delta matrix. */
export async function fitIcpAlignment(source: PointCloud, target: PointCloud, opts?: IcpOptions): Promise<NDArrayFloat32 | NDArrayFloat64> {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return dispatcher().run(
    () => native()[`dispatch_fit_icp_alignment_${dt}`](
      source._handle, target._handle,
      opts?.maxIterations ?? 100,
      opts?.nSamples ?? 1000,
      opts?.k ?? 1,
      opts?.sigma ?? -1,
      opts?.outlierProportion ?? 0,
      opts?.minRelativeImprovement ?? 1e-6,
      opts?.emaAlpha ?? 0.3,
    ),
    (raw) => new NDArray(raw, dt) as NDArrayFloat32 | NDArrayFloat64,
  );
}

/** Coarse alignment via oriented bounding boxes (async). Returns 4x4 delta matrix. */
export async function fitObbAlignment(source: PointCloud, target: PointCloud, opts?: ObbOptions): Promise<NDArrayFloat32 | NDArrayFloat64> {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return dispatcher().run(
    () => native()[`dispatch_fit_obb_alignment_${dt}`](
      source._handle, target._handle,
      opts?.sampleSize ?? 100,
    ),
    (raw) => new NDArray(raw, dt) as NDArrayFloat32 | NDArrayFloat64,
  );
}

/** One-way Chamfer distance (async). */
export async function chamferError(source: PointCloud, target: PointCloud, opts?: ChamferOptions): Promise<number> {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return dispatcher().run(
    () => native()[`dispatch_chamfer_error_${dt}`](
      source._handle, target._handle,
      opts?.outlierProportion ?? 0,
    ),
  );
}

// ============ Precompute ============

/** Compute face normals off the main thread. Result is cached on the mesh. */
export async function computeNormals(m: Mesh): Promise<NDArrayFloat32 | NDArrayFloat64> {
  const fn = native()[`dispatch_ensure_${m.dtype}`];
  await dispatcher().run(() => fn(m._handle, 1));
  return m.normals;
}

/** Compute vertex normals off the main thread. Result is cached on the mesh. */
export async function computePointNormals(m: Mesh): Promise<NDArrayFloat32 | NDArrayFloat64> {
  const fn = native()[`dispatch_ensure_${m.dtype}`];
  await dispatcher().run(() => fn(m._handle, 2));
  return m.pointNormals;
}
