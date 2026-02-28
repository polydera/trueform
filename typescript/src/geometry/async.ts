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
import { NDArray, NDArrayInt32, NDArrayFloat32 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { Mesh } from "../form/Mesh";
import { PointCloud } from "../form/PointCloud";
import { Primitive, Polygon } from "../primitive";

import type { PrincipalCurvatures, PrincipalDirections, IcpOptions, ObbOptions, ChamferOptions } from "./sync";

import type { MeshLike } from "../form/MeshLike";

/** Triangulate polygons into a triangle mesh (async). */
export async function triangulate(input: Mesh | MeshLike | Polygon): Promise<Mesh> {
  if (input instanceof Mesh) {
    return dispatcher().run(
      () => native().dispatch_triangulate_mesh(input._handle),
      (raw) => new Mesh(raw),
    );
  }
  if (input instanceof Primitive) {
    return dispatcher().run(
      () => native().dispatch_triangulate_polygon(input._handle),
      (raw) => new Mesh(raw),
    );
  }
  // MeshLike
  if (input.faces instanceof OffsetBlockedBuffer) {
    return dispatcher().run(
      () => native().dispatch_triangulate_dynamic(input.faces._handle, input.points._handle),
      (raw) => new Mesh(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_triangulate_fixed(
      (input.faces as NDArrayInt32)._handle, input.points._handle,
    ),
    (raw) => new Mesh(raw),
  );
}

/** Create a UV sphere mesh centered at origin (async). */
export async function sphereMesh(radius: number, stacks: number, segments: number): Promise<Mesh> {
  return dispatcher().run(
    () => native().dispatch_make_sphere_mesh(radius, stacks, segments),
    (raw) => new Mesh(raw),
  );
}

/** Create a cylinder mesh centered at origin along the z-axis (async). */
export async function cylinderMesh(radius: number, height: number, segments: number): Promise<Mesh> {
  return dispatcher().run(
    () => native().dispatch_make_cylinder_mesh(radius, height, segments),
    (raw) => new Mesh(raw),
  );
}

/** Create an axis-aligned box mesh centered at origin (async). Optionally subdivided. */
export async function boxMesh(
  width: number, height: number, depth: number,
  widthTicks?: number, heightTicks?: number, depthTicks?: number,
): Promise<Mesh> {
  if (widthTicks !== undefined && heightTicks !== undefined && depthTicks !== undefined) {
    return dispatcher().run(
      () => native().dispatch_make_box_mesh_subdivided(width, height, depth, widthTicks, heightTicks, depthTicks),
      (raw) => new Mesh(raw),
    );
  }
  return dispatcher().run(
    () => native().dispatch_make_box_mesh(width, height, depth),
    (raw) => new Mesh(raw),
  );
}

/** Create a flat rectangular plane mesh in the XY plane, centered at origin (async). */
export async function planeMesh(
  width: number, height: number,
  widthTicks?: number, heightTicks?: number,
): Promise<Mesh> {
  const wt = widthTicks ?? 1;
  const ht = heightTicks ?? 1;
  return dispatcher().run(
    () => native().dispatch_make_plane_mesh(width, height, wt, ht),
    (raw) => new Mesh(raw),
  );
}

// ============ Measurements ============

/** Total surface area of a mesh (async). */
export async function area(m: Mesh): Promise<number> {
  return dispatcher().run(() => native().dispatch_mesh_area(m._handle));
}

/** Signed volume of a closed 3D mesh (async). */
export async function signedVolume(m: Mesh): Promise<number> {
  return dispatcher().run(() => native().dispatch_mesh_signed_volume(m._handle));
}

/** Absolute volume of a closed 3D mesh (async). */
export async function volume(m: Mesh): Promise<number> {
  return dispatcher().run(
    () => native().dispatch_mesh_signed_volume(m._handle),
    (v: number) => Math.abs(v),
  );
}

/** Mean edge length across all faces (async). */
export async function meanEdgeLength(m: Mesh): Promise<number> {
  return dispatcher().run(() => native().dispatch_mesh_mean_edge_length(m._handle));
}

/** Minimum edge length across all faces (async). */
export async function minEdgeLength(m: Mesh): Promise<number> {
  return dispatcher().run(() => native().dispatch_mesh_min_edge_length(m._handle));
}

/** Maximum edge length across all faces (async). */
export async function maxEdgeLength(m: Mesh): Promise<number> {
  return dispatcher().run(() => native().dispatch_mesh_max_edge_length(m._handle));
}

// ============ Orientation ============

/** Return a new mesh with consistently oriented, outward-pointing normals (async). */
export async function positivelyOriented(m: Mesh, isConsistent?: boolean): Promise<Mesh> {
  const ic = isConsistent ?? false;
  return dispatcher().run(
    () => native().dispatch_positively_oriented(m._handle, ic),
    (raw) => new Mesh(raw),
  );
}

// ============ Curvature ============

/** Compute principal curvatures (k0, k1) at each vertex (async). */
export async function principalCurvatures(m: Mesh, k: number = 2): Promise<PrincipalCurvatures> {
  return dispatcher().run(
    () => native().dispatch_principal_curvatures(m._handle, k),
    (raw) => ({
      k0: new NDArray(raw.k0, "float32"),
      k1: new NDArray(raw.k1, "float32"),
    }),
  );
}

/** Compute principal curvatures and directions at each vertex (async). */
export async function principalDirections(m: Mesh, k: number = 2): Promise<PrincipalDirections> {
  return dispatcher().run(
    () => native().dispatch_principal_directions(m._handle, k),
    (raw) => ({
      k0: new NDArray(raw.k0, "float32"),
      k1: new NDArray(raw.k1, "float32"),
      d0: new NDArray(raw.d0, "float32"),
      d1: new NDArray(raw.d1, "float32"),
    }),
  );
}

/** Compute shape index at each vertex (async). Values in [-1, 1]. */
export async function shapeIndex(m: Mesh, k: number = 2): Promise<NDArrayFloat32> {
  return dispatcher().run(
    () => native().dispatch_shape_index(m._handle, k),
    (raw) => new NDArray(raw, "float32"),
  );
}

// ============ Smoothing ============

/** Laplacian smoothing (async). Returns a new mesh with smoothed vertex positions. */
export async function laplacianSmoothed(m: Mesh, iterations: number, lambda: number = 0.5): Promise<Mesh> {
  return dispatcher().run(
    () => native().dispatch_laplacian_smoothed(m._handle, iterations, lambda),
    (raw) => new Mesh(raw),
  );
}

/** Taubin smoothing (volume-preserving, async). Returns a new mesh with smoothed vertex positions. */
export async function taubinSmoothed(m: Mesh, iterations: number, lambda: number = 0.5, kpb: number = 0.1): Promise<Mesh> {
  return dispatcher().run(
    () => native().dispatch_taubin_smoothed(m._handle, iterations, lambda, kpb),
    (raw) => new Mesh(raw),
  );
}

// ============ Registration / Alignment (async) ============

/** Kabsch/SVD rigid alignment (async). Returns 4x4 delta matrix. */
export async function fitRigidAlignment(source: PointCloud, target: PointCloud): Promise<NDArrayFloat32> {
  return dispatcher().run(
    () => native().dispatch_fit_rigid_alignment(source._handle, target._handle),
    (raw) => new NDArray(raw, "float32"),
  );
}

/** Iterative Closest Point alignment (async). Returns 4x4 delta matrix. */
export async function fitIcpAlignment(source: PointCloud, target: PointCloud, opts?: IcpOptions): Promise<NDArrayFloat32> {
  return dispatcher().run(
    () => native().dispatch_fit_icp_alignment(
      source._handle, target._handle,
      opts?.maxIterations ?? 100,
      opts?.nSamples ?? 1000,
      opts?.k ?? 1,
      opts?.sigma ?? -1,
      opts?.outlierProportion ?? 0,
      opts?.minRelativeImprovement ?? 1e-6,
      opts?.emaAlpha ?? 0.3,
    ),
    (raw) => new NDArray(raw, "float32"),
  );
}

/** Coarse alignment via oriented bounding boxes (async). Returns 4x4 delta matrix. */
export async function fitObbAlignment(source: PointCloud, target: PointCloud, opts?: ObbOptions): Promise<NDArrayFloat32> {
  return dispatcher().run(
    () => native().dispatch_fit_obb_alignment(
      source._handle, target._handle,
      opts?.sampleSize ?? 100,
    ),
    (raw) => new NDArray(raw, "float32"),
  );
}

/** One-way Chamfer distance (async). */
export async function chamferError(source: PointCloud, target: PointCloud, opts?: ChamferOptions): Promise<number> {
  return dispatcher().run(
    () => native().dispatch_chamfer_error(
      source._handle, target._handle,
      opts?.outlierProportion ?? 0,
    ),
  );
}
